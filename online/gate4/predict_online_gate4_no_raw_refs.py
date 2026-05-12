from __future__ import annotations
from online.trading import config
import os

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from catboost import CatBoostClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("IMB_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DB_DSN = config.DB_DSN

SOURCE_TABLE = "online_gate4_features"
TARGET_TABLE = "online_gate4_predictions_no_raw_refs"

MODEL_DIR = ROOT / "pipeline/test/gate4/gate4_y_side_clean_multiclass_no_raw_refs"
MODEL_PATH = MODEL_DIR / "gate4_y_side_clean_multiclass.cbm"
FEATURES_CSV = MODEL_DIR / "features.csv"
MEDIANS_JSON = MODEL_DIR / "feature_medians.json"

BATCH_LIMIT = 100000


# ============================================================
# HELPERS
# ============================================================

def connect_db():
    return psycopg2.connect(DB_DSN)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError("not found: {}".format(path))


def load_feature_list(path: Path) -> List[str]:
    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise RuntimeError("{}: missing column 'feature', columns={}".format(path, list(df.columns)))

    features = df["feature"].dropna().astype(str).tolist()
    features = list(dict.fromkeys(features))

    if not features:
        raise RuntimeError("{}: empty feature list".format(path))

    return features


def load_feature_medians(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "medians" in raw and isinstance(raw["medians"], dict):
        raw = raw["medians"]

    if not isinstance(raw, dict):
        raise RuntimeError("{}: expected dict with feature medians".format(path))

    medians = {}

    for k, v in raw.items():
        try:
            fv = float(v)
        except Exception:
            fv = 0.0

        if not np.isfinite(fv):
            fv = 0.0

        medians[str(k)] = fv

    return medians


def table_exists(conn, table_name: str) -> bool:
    sql = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = %s
    )
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table_name,))
        return bool(cur.fetchone()[0])


def get_table_columns(conn, table_name: str) -> List[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = %s
    ORDER BY ordinal_position
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table_name,))
        return [str(row[0]) for row in cur.fetchall()]


def find_ts_col(columns: List[str]) -> str:
    for c in ["signal_ts", "ts", "entry_ts", "timestamp", "open_time", "time", "datetime", "dt"]:
        if c in columns:
            return c
    raise RuntimeError("timestamp column not found; source columns={}".format(columns[:50]))


def find_signal_key_col(columns: List[str]) -> Optional[str]:
    for c in ["signal_key", "id"]:
        if c in columns:
            return c
    return None


def detect_model_mode(model: CatBoostClassifier, proba: np.ndarray) -> str:
    if proba.ndim != 2:
        raise RuntimeError("bad predict_proba shape: {}".format(proba.shape))

    if proba.shape[1] == 2:
        return "binary"

    if proba.shape[1] == 3:
        return "multiclass_3"

    raise RuntimeError("unsupported class count in predict_proba: {}".format(proba.shape[1]))


def normalize_ts_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def make_signal_key(symbol: str, ts_value: pd.Timestamp) -> str:
    ts_utc = pd.Timestamp(ts_value)

    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    else:
        ts_utc = ts_utc.tz_convert("UTC")

    return "{}|{}|GATE4_NO_RAW_REFS".format(
        str(symbol).upper(),
        ts_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    )


# ============================================================
# LOAD ONLINE FEATURES
# ============================================================

def load_source_rows(conn, model_features: List[str]) -> pd.DataFrame:
    if not table_exists(conn, SOURCE_TABLE):
        raise RuntimeError("source table not found: {}".format(SOURCE_TABLE))

    if not table_exists(conn, TARGET_TABLE):
        raise RuntimeError("target table not found: {}".format(TARGET_TABLE))

    source_cols = get_table_columns(conn, SOURCE_TABLE)
    target_cols = get_table_columns(conn, TARGET_TABLE)

    if "symbol" not in source_cols:
        raise RuntimeError("{}: missing column symbol".format(SOURCE_TABLE))

    ts_col = find_ts_col(source_cols)
    signal_key_col = find_signal_key_col(source_cols)

    missing_model_features = [c for c in model_features if c not in source_cols]
    if missing_model_features:
        raise RuntimeError(
            "{}: missing model features: {} total={}".format(
                SOURCE_TABLE,
                missing_model_features[:50],
                len(missing_model_features),
            )
        )

    select_cols = []

    if signal_key_col is not None:
        select_cols.append(signal_key_col)

    select_cols.extend(["symbol", ts_col])

    if "upstream_split" in source_cols:
        select_cols.append("upstream_split")

    if "upstream_is_oos" in source_cols:
        select_cols.append("upstream_is_oos")

    select_cols.extend(model_features)
    select_cols = list(dict.fromkeys(select_cols))

    quoted_select_cols = ", ".join(['src."{}"'.format(c) for c in select_cols])

    if "signal_key" in target_cols:
        if signal_key_col is not None:
            signal_key_expr = 'src."{}"'.format(signal_key_col)
        else:
            signal_key_expr = (
                "src.symbol || '|' || "
                "to_char(src.\"{}\" AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS+00:00') || "
                "'|GATE4_NO_RAW_REFS'"
            ).format(ts_col)

        sql = """
        SELECT
            {signal_key_expr} AS __signal_key,
            {quoted_select_cols}
        FROM {source_table} src
        LEFT JOIN {target_table} tgt
          ON tgt.signal_key = {signal_key_expr}
        WHERE tgt.signal_key IS NULL
        ORDER BY src."{ts_col}", src.symbol
        LIMIT {batch_limit}
        """.format(
            signal_key_expr=signal_key_expr,
            quoted_select_cols=quoted_select_cols,
            source_table=SOURCE_TABLE,
            target_table=TARGET_TABLE,
            ts_col=ts_col,
            batch_limit=int(BATCH_LIMIT),
        )
    else:
        sql = """
        SELECT
            {quoted_select_cols}
        FROM {source_table} src
        ORDER BY src."{ts_col}", src.symbol
        LIMIT {batch_limit}
        """.format(
            quoted_select_cols=quoted_select_cols,
            source_table=SOURCE_TABLE,
            ts_col=ts_col,
            batch_limit=int(BATCH_LIMIT),
        )

    df = pd.read_sql_query(sql, conn)

    if ts_col in df.columns and ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})

    if signal_key_col is not None and signal_key_col in df.columns and signal_key_col != "signal_key":
        df = df.rename(columns={signal_key_col: "signal_key"})

    df["ts"] = normalize_ts_series(df["ts"])
    df["symbol"] = df["symbol"].astype(str).str.upper()

    bad_ts = int(df["ts"].isna().sum())
    if bad_ts:
        raise RuntimeError("bad ts rows: {}".format(bad_ts))

    if "signal_key" not in df.columns:
        if "__signal_key" in df.columns:
            df["signal_key"] = df["__signal_key"].astype(str)
        else:
            df["signal_key"] = [
                make_signal_key(symbol=row_symbol, ts_value=row_ts)
                for row_symbol, row_ts in zip(df["symbol"], df["ts"])
            ]

    if "upstream_split" not in df.columns:
        df["upstream_split"] = "online"

    if "upstream_is_oos" not in df.columns:
        df["upstream_is_oos"] = np.nan

    return df


# ============================================================
# PREPARE X EXACTLY FOR MODEL
# ============================================================

def prepare_x_from_online(
    df: pd.DataFrame,
    model_features: List[str],
    medians: Dict[str, float],
) -> pd.DataFrame:
    missing = [c for c in model_features if c not in df.columns]
    if missing:
        raise RuntimeError("dataset missing model features: {} total={}".format(missing[:50], len(missing)))

    x = df[model_features].copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    for c in model_features:
        fill_value = float(medians.get(c, 0.0))
        if not np.isfinite(fill_value):
            fill_value = 0.0
        x[c] = x[c].fillna(fill_value)

    x = x.fillna(0.0)
    return x


# ============================================================
# PREDICT LIKE OFFLINE all_predictions_raw.py
# ============================================================

def build_prediction_frame(
    source_df: pd.DataFrame,
    model: CatBoostClassifier,
    model_features: List[str],
    medians: Dict[str, float],
) -> pd.DataFrame:
    x = prepare_x_from_online(
        df=source_df,
        model_features=model_features,
        medians=medians,
    )

    proba = model.predict_proba(x)
    mode = detect_model_mode(model, proba)

    if mode == "binary":
        proba_short = proba[:, 0]
        proba_long = proba[:, 1]
        pred_label = pd.Series(model.predict(x)).astype(str)

        out = pd.DataFrame({
            "signal_key": source_df["signal_key"].astype(str),
            "ts": source_df["ts"],
            "signal_ts": source_df["ts"],
            "symbol": source_df["symbol"].astype(str),
            "upstream_split": source_df["upstream_split"],
            "upstream_is_oos": source_df["upstream_is_oos"],
            "pred_label": pred_label,
            "proba_0": proba_short,
            "proba_1": proba_long,
            "proba_long": proba_long,
            "proba_short": proba_short,
            "model_mode": mode,
        })

    else:
        proba_short = proba[:, 0]
        proba_ambig = proba[:, 1]
        proba_long = proba[:, 2]
        pred_label = np.argmax(proba, axis=1).astype(int)

        out = pd.DataFrame({
            "signal_key": source_df["signal_key"].astype(str),
            "ts": source_df["ts"],
            "signal_ts": source_df["ts"],
            "symbol": source_df["symbol"].astype(str),
            "upstream_split": source_df["upstream_split"],
            "upstream_is_oos": source_df["upstream_is_oos"],
            "pred_label": pred_label,
            "proba_0": proba_short,
            "proba_1": proba_ambig,
            "proba_2": proba_long,
            "proba_short": proba_short,
            "proba_ambig": proba_ambig,
            "proba_long": proba_long,
            "model_mode": mode,
        })

    dup_count = int(out.duplicated(["symbol", "ts"]).sum())
    if dup_count:
        raise RuntimeError("duplicated symbol+ts rows in output: {}".format(dup_count))

    out["model_path"] = str(MODEL_PATH)
    out["features_path"] = str(FEATURES_CSV)
    out["medians_path"] = str(MEDIANS_JSON)

    return out


# ============================================================
# WRITE TO DB
# ============================================================

def clean_db_value(v):
    if pd.isna(v):
        return None

    if isinstance(v, pd.Timestamp):
        if v.tzinfo is None:
            return v.to_pydatetime()
        return v.to_pydatetime()

    if isinstance(v, np.generic):
        return v.item()

    return v


def write_predictions(conn, pred_df: pd.DataFrame) -> int:
    if len(pred_df) == 0:
        return 0

    target_cols = get_table_columns(conn, TARGET_TABLE)

    insert_cols = [
        c for c in [
            "signal_key",
            "ts",
            "signal_ts",
            "symbol",
            "upstream_split",
            "upstream_is_oos",
            "pred_label",
            "proba_0",
            "proba_1",
            "proba_2",
            "proba_short",
            "proba_ambig",
            "proba_long",
            "gate4_pred_side",
            "gate4_pred_side_int",
            "gate4_confidence",
            "gate4_pred_side_gap",
            "gate4_pred_side_ratio",
            "model_mode",
            "model_path",
            "features_path",
            "medians_path",
        ]
        if c in target_cols and c in pred_df.columns
    ]

    has_updated_at = "updated_at" in target_cols

    if "signal_key" not in insert_cols:
        raise RuntimeError("{}: target table must have signal_key for safe upsert".format(TARGET_TABLE))

    quoted_cols = ", ".join(['"{}"'.format(c) for c in insert_cols])
    values_template = "(" + ", ".join(["%s"] * len(insert_cols)) + ")"

    update_cols = [c for c in insert_cols if c != "signal_key"]
    update_assignments = ['"{0}" = EXCLUDED."{0}"'.format(c) for c in update_cols]

    if has_updated_at:
        quoted_cols = quoted_cols + ', "updated_at"'
        values_template = values_template[:-1] + ", now())"
        update_assignments.append('"updated_at" = now()')

    sql = """
    INSERT INTO {target_table} ({quoted_cols})
    VALUES %s
    ON CONFLICT (signal_key) DO UPDATE SET
        {update_assignments}
    """.format(
        target_table=TARGET_TABLE,
        quoted_cols=quoted_cols,
        update_assignments=",\n        ".join(update_assignments),
    )

    rows = []
    for row in pred_df[insert_cols].itertuples(index=False, name=None):
        rows.append(tuple(clean_db_value(v) for v in row))

    print("DB_BATCH_UPSERT_GATE4_PREDICTIONS_ROWS:", len(rows), flush=True)

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            rows,
            template=values_template,
            page_size=5000,
        )

    conn.commit()

    print("DB_BATCH_UPSERT_GATE4_PREDICTIONS_DONE:", len(rows), flush=True)

    return len(rows)
# ============================================================
# MAIN
# ============================================================

def main() -> None:
    require_file(MODEL_PATH)
    require_file(FEATURES_CSV)
    require_file(MEDIANS_JSON)

    feature_cols = load_feature_list(FEATURES_CSV)
    medians = load_feature_medians(MEDIANS_JSON)

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    model_feats = list(dict.fromkeys(model.feature_names_ or []))
    if not model_feats:
        raise RuntimeError("model.feature_names_ is empty")

    missing_in_features_csv = sorted(set(model_feats) - set(feature_cols))
    if missing_in_features_csv:
        raise RuntimeError(
            "model features missing in features.csv: {} total={}".format(
                missing_in_features_csv[:50],
                len(missing_in_features_csv),
            )
        )

    missing_medians = [c for c in model_feats if c not in medians]
    if missing_medians:
        raise RuntimeError(
            "missing medians for model features: {} total={}".format(
                missing_medians[:50],
                len(missing_medians),
            )
        )

    with connect_db() as conn:
        print("ROOT:", ROOT)
        print("SOURCE_TABLE:", SOURCE_TABLE)
        print("TARGET_TABLE:", TARGET_TABLE)
        print("MODEL_PATH:", MODEL_PATH)
        print("FEATURES_CSV:", FEATURES_CSV)
        print("MEDIANS_JSON:", MEDIANS_JSON)
        print("MODEL_FEATURE_COUNT:", len(model_feats))
        print("BATCH_LIMIT:", BATCH_LIMIT)
        print()

        source_df = load_source_rows(conn, model_feats)

        print("ROWS TO PREDICT:", len(source_df))

        if len(source_df) == 0:
            print("NO NEW ROWS")
            return

        pred_df = build_prediction_frame(
            source_df=source_df,
            model=model,
            model_features=model_feats,
            medians=medians,
        )

        pred_df["proba_long"] = pd.to_numeric(pred_df["proba_long"], errors="coerce")
        pred_df["proba_short"] = pd.to_numeric(pred_df["proba_short"], errors="coerce")

        pred_df["gate4_confidence"] = np.maximum(
            pred_df["proba_long"],
            pred_df["proba_short"],
        )
        pred_df["gate4_pred_side"] = np.where(
            pred_df["proba_long"] >= pred_df["proba_short"],
            "LONG",
            "SHORT",
        )

        pred_df["gate4_pred_side_int"] = np.where(
            pred_df["gate4_pred_side"] == "LONG",
            1,
            0,
        )
        pred_df["gate4_pred_side_gap"] = (
                pred_df["proba_long"] - pred_df["proba_short"]
        ).abs()

        denom = np.minimum(
            pred_df["proba_long"].to_numpy(dtype=float),
            pred_df["proba_short"].to_numpy(dtype=float),
        )

        numer = np.maximum(
            pred_df["proba_long"].to_numpy(dtype=float),
            pred_df["proba_short"].to_numpy(dtype=float),
        )

        pred_df["gate4_pred_side_ratio"] = np.where(
            denom > 0,
            numer / denom,
            np.nan,
        )

        null_gate4_confidence = int(pred_df["gate4_confidence"].isna().sum())
        if null_gate4_confidence:
            raise RuntimeError(
                "gate4_confidence has NULL rows before DB write: {}".format(
                    null_gate4_confidence
                )
            )

        written = write_predictions(conn, pred_df)

        print("WRITTEN:", written)
        print("MODEL MODE:", pred_df["model_mode"].iloc[0] if len(pred_df) else None)
        print("TS_MIN:", pred_df["ts"].min())
        print("TS_MAX:", pred_df["ts"].max())
        print("SYMBOLS:", pred_df["symbol"].nunique())
        print()
        print("COLUMNS:")
        print(list(pred_df.columns))
        print()
        print("GATE4_CONFIDENCE:")
        print(pred_df["gate4_confidence"].describe().to_string())
        print()
        print("PROBA_LONG:")
        print(pred_df["proba_long"].describe().to_string())
        print()
        print("PROBA_SHORT:")
        print(pred_df["proba_short"].describe().to_string())

        if "proba_ambig" in pred_df.columns:
            print()
            print("PROBA_AMBIG:")
            print(pred_df["proba_ambig"].describe().to_string())

if __name__ == "__main__":
    main()
