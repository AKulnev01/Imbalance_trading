from online.trading import config
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from online.oos_context import append_oos_sql_filters, get_online_oos_context
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from catboost import CatBoostClassifier, Pool


warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)


ROOT = Path(os.environ.get("IMB_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DB_DSN = config.DB_DSN

ONLINE_GATE2_FEATURES_TABLE = "public.online_gate2_features"
ONLINE_GATE2_PREDICTIONS_TABLE = "public.online_gate2_predictions"

GATE2_MODELS_ROOT = ROOT / "production" / "models" / "gate2_mod_5features" / "cls"

REPORT_DIR = ROOT / "online" / "_reports_gate2"
REPORT_CSV = REPORT_DIR / "online_gate2_predictions_report.csv"
REPORT_JSON = REPORT_DIR / "online_gate2_predictions_report.json"

MODEL_NAMES = [
    "up_reach_high",
    "dn_reach_high",
]

DEFAULT_THRESHOLDS = {
    "up_reach_high": 0.354285714286,
    "dn_reach_high": 0.336122448980,
}

SERVICE_COLS = {
    "entry_ts",
    "signal_ts",
    "entry_bar_open_ts",
    "entry_ts_exec",
    "entry_px_exec",
    "online_source",
    "online_feature_builder",
    "online_created_at",
    "online_updated_at",
}

TARGET_OR_FUTURE_PREFIXES = (
    "y_",
    "target_",
    "label_",
    "mfe_",
    "mae_",
    "first_",
    "future_",
    "fwd_",
)

TARGET_OR_FUTURE_SUBSTRINGS = (
    "_target",
    "_label",
    "_mfe",
    "_mae",
    "_first_hit",
    "_first_tp",
    "_first_sl",
    "_future",
    "_fwd",
)


def connect_db():
    return psycopg2.connect(DB_DSN)


def utc_now_floor_second() -> pd.Timestamp:
    return pd.Timestamp.utcnow().floor("s")


def split_table_name(table_name: str) -> Tuple[str, str]:
    if "." not in table_name:
        return "public", table_name
    schema, name = table_name.split(".", 1)
    return schema, name


def table_exists(table_name: str) -> bool:
    schema, name = split_table_name(table_name)
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (schema, name))
            return bool(cur.fetchone()[0])


def get_table_columns(table_name: str) -> List[str]:
    schema, name = split_table_name(table_name)
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """
    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=(schema, name))
    return [str(x) for x in df["column_name"].tolist()]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_features_csv(path: Path) -> List[str]:
    if not path.exists():
        raise RuntimeError(f"features.csv not found: {path}")

    df = pd.read_csv(path)

    for col in ["feature", "feature_name", "column_name", "name"]:
        if col in df.columns:
            features = [str(x) for x in df[col].dropna().tolist()]
            if features:
                return features

    if df.shape[1] == 1:
        features = [str(x) for x in df.iloc[:, 0].dropna().tolist()]
        if features:
            return features

    raise RuntimeError(f"cannot parse features.csv columns={list(df.columns)} path={path}")


def load_threshold(model_dir: Path, model_name: str) -> float:
    report_path = model_dir / "report.json"

    if report_path.exists():
        try:
            obj = json.loads(report_path.read_text(encoding="utf-8"))

            candidates = [
                "threshold",
                "best_threshold",
                "selected_threshold",
                "thr",
                "proba_threshold",
            ]

            for key in candidates:
                if key in obj and obj[key] is not None:
                    return float(obj[key])

            for block_key in ["selected", "best", "prod", "thresholds"]:
                block = obj.get(block_key)
                if isinstance(block, dict):
                    for key in candidates:
                        if key in block and block[key] is not None:
                            return float(block[key])
        except Exception:
            pass

    return float(DEFAULT_THRESHOLDS[model_name])


def is_target_or_future_feature(col: str) -> bool:
    low = col.lower()

    for prefix in TARGET_OR_FUTURE_PREFIXES:
        if low.startswith(prefix):
            return True

    for part in TARGET_OR_FUTURE_SUBSTRINGS:
        if part in low:
            return True

    return False


def load_models() -> Dict[str, Dict[str, object]]:
    out = {}

    for model_name in MODEL_NAMES:
        model_dir = GATE2_MODELS_ROOT / model_name
        model_path = model_dir / f"{model_name}.cbm"
        features_path = model_dir / "features.csv"

        if not model_path.exists():
            raise RuntimeError(f"model file not found: {model_path}")

        features = read_features_csv(features_path)

        bad_future = [c for c in features if is_target_or_future_feature(c)]
        if bad_future:
            raise RuntimeError(f"{model_name}: target/future features in model features: {bad_future}")

        model = CatBoostClassifier()
        model.load_model(str(model_path))

        threshold = load_threshold(model_dir, model_name)

        out[model_name] = {
            "name": model_name,
            "dir": model_dir,
            "path": model_path,
            "features": features,
            "threshold": float(threshold),
            "model": model,
        }

    return out


def ensure_predictions_table() -> None:
    query = f"""
        CREATE TABLE IF NOT EXISTS {ONLINE_GATE2_PREDICTIONS_TABLE} (
            symbol TEXT NOT NULL,
            entry_ts TIMESTAMPTZ NOT NULL,

            up_reach_high_proba DOUBLE PRECISION,
            up_reach_high_threshold DOUBLE PRECISION,
            up_reach_high_pass BOOLEAN,

            dn_reach_high_proba DOUBLE PRECISION,
            dn_reach_high_threshold DOUBLE PRECISION,
            dn_reach_high_pass BOOLEAN,

            gate2_any_pass BOOLEAN,
            gate2_both_pass BOOLEAN,
            gate2_side TEXT,

            online_model_root TEXT,
            online_predict_builder TEXT,
            online_created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            online_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            PRIMARY KEY (symbol, entry_ts)
        )
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()


def clear_predictions_table() -> None:
    if not table_exists(ONLINE_GATE2_PREDICTIONS_TABLE):
        return

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {ONLINE_GATE2_PREDICTIONS_TABLE}")
        conn.commit()


def get_symbols_from_features() -> List[str]:
    query = f"""
        SELECT DISTINCT symbol
        FROM {ONLINE_GATE2_FEATURES_TABLE}
        ORDER BY symbol
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn)

    return [str(x).upper() for x in df["symbol"].tolist()]


def load_feature_rows_for_symbol(symbol: str, rebuild: bool, limit_latest: Optional[int]) -> pd.DataFrame:
    if rebuild or not table_exists(ONLINE_GATE2_PREDICTIONS_TABLE):
        where_missing = ""
    else:
        where_missing = f"""
            AND NOT EXISTS (
                SELECT 1
                FROM {ONLINE_GATE2_PREDICTIONS_TABLE} p
                WHERE p.symbol = f.symbol
                  AND p.entry_ts = f.entry_ts
            )
        """

    limit_clause = ""
    if limit_latest is not None and int(limit_latest) > 0:
        limit_clause = f"LIMIT {int(limit_latest)}"

    query = f"""
        SELECT f.*
        FROM {ONLINE_GATE2_FEATURES_TABLE} f
        WHERE f.symbol = %s
        {where_missing}
        ORDER BY f.entry_ts DESC
        {limit_clause}
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=(symbol,))

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    return df



def load_feature_rows_batch(symbols: List[str], rebuild: bool, limit_latest: Optional[int]) -> pd.DataFrame:
    symbols_clean = sorted(set(str(x).upper().strip() for x in symbols if str(x).strip()))
    if not symbols_clean:
        return pd.DataFrame()

    predictions_exists = table_exists(ONLINE_GATE2_PREDICTIONS_TABLE)

    if rebuild or not predictions_exists:
        missing_sql = ""
    else:
        missing_sql = """
            AND NOT EXISTS (
                SELECT 1
                FROM {pred_table} p
                WHERE p.symbol = f.symbol
                  AND p.entry_ts = f.entry_ts
            )
        """.format(pred_table=ONLINE_GATE2_PREDICTIONS_TABLE)

    where_parts = ["UPPER(f.symbol) = ANY(%s)"]
    params: List[object] = [symbols_clean]

    append_oos_sql_filters(
        where_parts=where_parts,
        params=params,
        table_alias="f",
        ts_column="entry_ts",
        symbol_column="symbol",
    )

    where_sql = " AND ".join(where_parts)

    if limit_latest is not None and int(limit_latest) > 0:
        query = """
            WITH src AS (
                SELECT f.*
                FROM {features_table} f
                WHERE {where_sql}
                {missing_sql}
            ),
            ranked AS (
                SELECT
                    src.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY UPPER(src.symbol)
                        ORDER BY src.entry_ts DESC
                    ) AS rn
                FROM src
            )
            SELECT *
            FROM ranked
            WHERE rn <= %s
            ORDER BY symbol ASC, entry_ts ASC
        """.format(
            features_table=ONLINE_GATE2_FEATURES_TABLE,
            where_sql=where_sql,
            missing_sql=missing_sql,
        )
        params.append(int(limit_latest))
    else:
        query = """
            SELECT f.*
            FROM {features_table} f
            WHERE {where_sql}
            {missing_sql}
            ORDER BY f.symbol ASC, f.entry_ts ASC
        """.format(
            features_table=ONLINE_GATE2_FEATURES_TABLE,
            where_sql=where_sql,
            missing_sql=missing_sql,
        )

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    if "rn" in df.columns:
        df = df.drop(columns=["rn"])

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["symbol", "entry_ts"]).sort_values(["symbol", "entry_ts"]).reset_index(drop=True)

    return df
def validate_model_features(df: pd.DataFrame, models: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []

    for model_name, cfg in models.items():
        features = cfg["features"]

        for feature in features:
            exists = feature in df.columns
            forbidden_name = is_target_or_future_feature(feature)

            null_rows = None
            dtype = None

            if exists:
                null_rows = int(df[feature].isna().sum())
                dtype = str(df[feature].dtype)

            bad = False

            if not exists:
                bad = True

            if forbidden_name:
                bad = True

            if exists and feature != "symbol":
                converted = pd.to_numeric(df[feature], errors="coerce")
                coerced_bad = int(converted.isna().sum() - df[feature].isna().sum())
                inf_rows = int(np.isinf(converted.replace([np.inf, -np.inf], np.nan)).sum())
                if coerced_bad > 0:
                    bad = True
            else:
                coerced_bad = 0
                inf_rows = 0

            if bad:
                rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "exists": bool(exists),
                        "forbidden_name": bool(forbidden_name),
                        "null_rows": null_rows,
                        "coerced_bad_rows": int(coerced_bad),
                        "inf_rows": int(inf_rows),
                        "dtype": dtype,
                    }
                )

    return pd.DataFrame(rows)


def prepare_pool(df: pd.DataFrame, features: List[str]) -> Pool:
    x = df[features].copy()

    cat_features = []

    for col in features:
        if col == "symbol":
            x[col] = x[col].astype(str).fillna("__MISSING__")
            cat_features.append(col)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[col] = x[col].replace([np.inf, -np.inf], np.nan)

    return Pool(x, cat_features=cat_features)


def predict_for_df(df: pd.DataFrame, models: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    out = df[["symbol", "entry_ts"]].copy()

    for model_name, cfg in models.items():
        model = cfg["model"]
        features = cfg["features"]
        threshold = float(cfg["threshold"])

        pool = prepare_pool(df, features)
        proba = model.predict_proba(pool)[:, 1]

        out[f"{model_name}_proba"] = proba.astype(float)
        out[f"{model_name}_threshold"] = threshold
        out[f"{model_name}_pass"] = out[f"{model_name}_proba"] >= threshold

    out["gate2_any_pass"] = out["up_reach_high_pass"] | out["dn_reach_high_pass"]
    out["gate2_both_pass"] = out["up_reach_high_pass"] & out["dn_reach_high_pass"]

    conditions = [
        out["up_reach_high_pass"] & ~out["dn_reach_high_pass"],
        out["dn_reach_high_pass"] & ~out["up_reach_high_pass"],
        out["up_reach_high_pass"] & out["dn_reach_high_pass"] & (out["up_reach_high_proba"] >= out["dn_reach_high_proba"]),
        out["up_reach_high_pass"] & out["dn_reach_high_pass"] & (out["dn_reach_high_proba"] > out["up_reach_high_proba"]),
    ]

    choices = [
        "LONG",
        "SHORT",
        "LONG",
        "SHORT",
    ]

    out["gate2_side"] = np.select(conditions, choices, default="NONE")

    out["online_model_root"] = str(GATE2_MODELS_ROOT)
    out["online_predict_builder"] = "online.gate2.predict_online_gate2"
    out["online_created_at"] = utc_now_floor_second()
    out["online_updated_at"] = out["online_created_at"]

    return out


def insert_predictions(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    cols = [
        "symbol",
        "entry_ts",
        "up_reach_high_proba",
        "up_reach_high_threshold",
        "up_reach_high_pass",
        "dn_reach_high_proba",
        "dn_reach_high_threshold",
        "dn_reach_high_pass",
        "gate2_any_pass",
        "gate2_both_pass",
        "gate2_side",
        "online_model_root",
        "online_predict_builder",
        "online_created_at",
        "online_updated_at",
    ]

    rows = []
    for row in df[cols].itertuples(index=False, name=None):
        rows.append(tuple(row))

    insert_cols = ", ".join([quote_ident(c) for c in cols])

    update_cols = [
        "up_reach_high_proba",
        "up_reach_high_threshold",
        "up_reach_high_pass",
        "dn_reach_high_proba",
        "dn_reach_high_threshold",
        "dn_reach_high_pass",
        "gate2_any_pass",
        "gate2_both_pass",
        "gate2_side",
        "online_model_root",
        "online_predict_builder",
        "online_updated_at",
    ]

    set_clause = ", ".join(
        [f"{quote_ident(c)} = EXCLUDED.{quote_ident(c)}" for c in update_cols]
    )

    query = f"""
        INSERT INTO {ONLINE_GATE2_PREDICTIONS_TABLE} ({insert_cols})
        VALUES %s
        ON CONFLICT (symbol, entry_ts)
        DO UPDATE SET {set_clause}
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows, page_size=1000)
        conn.commit()

    return int(len(rows))


def parse_args() -> Tuple[Optional[str], bool, Optional[int], bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit-latest", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    symbol = None
    if args.symbol:
        symbol = str(args.symbol).upper().strip()

    return symbol, bool(args.rebuild), args.limit_latest, bool(args.dry_run)



def main() -> None:
    symbol_arg, rebuild, limit_latest, dry_run = parse_args()

    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("ONLINE_GATE2_FEATURES_TABLE:", ONLINE_GATE2_FEATURES_TABLE)
    print("ONLINE_GATE2_PREDICTIONS_TABLE:", ONLINE_GATE2_PREDICTIONS_TABLE)
    print("GATE2_MODELS_ROOT:", GATE2_MODELS_ROOT)
    print("REBUILD:", rebuild)
    print("LIMIT_LATEST:", limit_latest)
    print("DRY_RUN:", dry_run)
    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    models = load_models()

    print("MODELS:")
    for name, cfg in models.items():
        print(
            f"  {name}: features={len(cfg['features'])} "
            f"threshold={cfg['threshold']} "
            f"path={cfg['path']}"
        )
    print()

    ensure_predictions_table()
    oos_ctx = get_online_oos_context()

    if rebuild and not dry_run:
        if oos_ctx.enabled:
            print("REBUILD_WITH_OOS: full predictions table truncate disabled; OOS rows will be overwritten by upsert")
        else:
            clear_predictions_table()

    if symbol_arg:
        symbols = [symbol_arg]
    elif oos_ctx.enabled:
        symbols = list(oos_ctx.symbols)
    else:
        symbols = get_symbols_from_features()

    print("OOS_MODE:", oos_ctx.enabled)
    print("OOS_SYMBOLS:", ",".join(oos_ctx.symbols))
    print("OOS_START:", oos_ctx.start_text)
    print("OOS_END:", oos_ctx.end_text)
    print("SYMBOLS:", len(symbols))
    print("DB_BATCH_LOAD: gate2 feature rows for all symbols")
    print()

    features_all = load_feature_rows_batch(
        symbols=symbols,
        rebuild=rebuild,
        limit_latest=limit_latest,
    )

    print("FEATURE_ROWS_BATCH:", len(features_all))
    print("FEATURE_SYMBOLS_BATCH:", int(features_all["symbol"].nunique()) if not features_all.empty else 0)
    print()

    reports = []
    total_built = 0
    total_inserted = 0
    pred_all = pd.DataFrame()

    if not features_all.empty:
        bad_features = validate_model_features(features_all, models)
        if not bad_features.empty:
            bad_path = REPORT_DIR / "online_gate2_bad_model_features.csv"
            bad_features.to_csv(bad_path, index=False)
            raise RuntimeError("bad Gate2 model features; wrote: {}".format(bad_path))

        pred_all = predict_for_df(features_all, models)
        total_built = int(len(pred_all))

        if dry_run:
            total_inserted = 0
        else:
            total_inserted = insert_predictions(pred_all)

    feature_counts = {}
    if not features_all.empty:
        feature_counts = {
            str(k).upper(): int(v)
            for k, v in features_all.groupby("symbol", dropna=False).size().to_dict().items()
        }

    pred_counts = {}
    up_pass_counts = {}
    dn_pass_counts = {}
    any_pass_counts = {}

    if not pred_all.empty:
        pred_counts = {
            str(k).upper(): int(v)
            for k, v in pred_all.groupby("symbol", dropna=False).size().to_dict().items()
        }

        up_pass_counts = {
            str(k).upper(): int(v)
            for k, v in pred_all.groupby("symbol")["up_reach_high_pass"].sum().to_dict().items()
        }

        dn_pass_counts = {
            str(k).upper(): int(v)
            for k, v in pred_all.groupby("symbol")["dn_reach_high_pass"].sum().to_dict().items()
        }

        any_pass_counts = {
            str(k).upper(): int(v)
            for k, v in pred_all.groupby("symbol")["gate2_any_pass"].sum().to_dict().items()
        }

    for i, symbol in enumerate(symbols, start=1):
        symbol = str(symbol).upper()
        print(f"[{i}/{len(symbols)}] {symbol}")

        feature_rows = int(feature_counts.get(symbol, 0))
        prediction_rows = int(pred_counts.get(symbol, 0))
        up_pass = int(up_pass_counts.get(symbol, 0))
        dn_pass = int(dn_pass_counts.get(symbol, 0))
        any_pass = int(any_pass_counts.get(symbol, 0))

        if feature_rows <= 0:
            rep = {
                "symbol": symbol,
                "status": "no_missing",
                "feature_rows": 0,
                "prediction_rows": 0,
                "inserted_rows": 0,
                "up_pass": 0,
                "dn_pass": 0,
                "any_pass": 0,
                "err": "",
            }
            print("    status=no_missing | features=0 | predictions=0 | inserted=0")
        else:
            rep = {
                "symbol": symbol,
                "status": "ok",
                "feature_rows": feature_rows,
                "prediction_rows": prediction_rows,
                "inserted_rows": 0 if dry_run else prediction_rows,
                "up_pass": up_pass,
                "dn_pass": dn_pass,
                "any_pass": any_pass,
                "err": "",
            }
            print(
                f"    status=ok | features={feature_rows} | predictions={prediction_rows} "
                f"| inserted={rep['inserted_rows']} | up_pass={up_pass} | dn_pass={dn_pass} | any_pass={any_pass}"
            )

        reports.append(rep)

    rep_df = pd.DataFrame(reports)
    rep_df.to_csv(REPORT_CSV, index=False)

    summary = {
        "created_at_utc": str(utc_now_floor_second()),
        "root": str(ROOT),
        "online_gate2_features_table": ONLINE_GATE2_FEATURES_TABLE,
        "online_gate2_predictions_table": ONLINE_GATE2_PREDICTIONS_TABLE,
        "gate2_models_root": str(GATE2_MODELS_ROOT),
        "symbols_count": int(len(symbols)),
        "rebuild": bool(rebuild),
        "limit_latest": limit_latest,
        "dry_run": bool(dry_run),
        "total_built": int(total_built),
        "total_inserted": int(total_inserted),
        "status_counts": rep_df["status"].value_counts(dropna=False).to_dict() if len(rep_df) else {},
        "report_csv": str(REPORT_CSV),
    }

    REPORT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 120)
    print("DONE")
    print("STATUS COUNTS:", summary["status_counts"])
    print("TOTAL BUILT:", total_built)
    print("TOTAL INSERTED:", total_inserted)
    print("WROTE:", REPORT_CSV)
    print("WROTE:", REPORT_JSON)


if __name__ == "__main__":
    main()
