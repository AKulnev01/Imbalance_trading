from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

try:
    import psycopg2
    import psycopg2.extras
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: psycopg2. Install it in project venv: pip install psycopg2-binary"
    ) from exc


# ============================================================
# PATHS / DB CONFIG
# ============================================================

ROOT = Path(os.environ.get("IMB_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DB_DSN = os.environ.get(
    "IMB_DB_DSN",
    "dbname=imb_traid host=localhost port=5432",
)

GATE5_2_SOURCE_TABLE = os.environ.get(
    "GATE5_2_SOURCE_TABLE",
    "online_gate5_2_ranker",
)

GATE5_3_OUTPUT_TABLE = os.environ.get(
    "GATE5_3_OUTPUT_TABLE",
    "online_gate5_3_decisions",
)

GATE5_3_MODEL_ROOT = (
    ROOT
    / "pipeline"
    / "test"
    / "gate5"
    / "gate5_3_cls_models_no_raw_refs_thr010"
)

BATCH_LIMIT = int(os.environ.get("GATE5_3_BATCH_LIMIT", "100000"))

LOOKBACK_HOURS_TXT = os.environ.get("GATE5_3_LOOKBACK_HOURS", "").strip()
LOOKBACK_HOURS: Optional[int] = int(LOOKBACK_HOURS_TXT) if LOOKBACK_HOURS_TXT else None


# ============================================================
# ONLINE CONFIG
# ============================================================

PROD_PAIR_NAME = "tp225_sl075__vs__tp100_sl075"

SAFE_GRID_NAME = "tp225_sl075"
AGG_GRID_NAME = "tp100_sl075"

PAIR_MODEL_NAME = f"{SAFE_GRID_NAME}__vs__{AGG_GRID_NAME}"

MODEL_PATH = GATE5_3_MODEL_ROOT / PAIR_MODEL_NAME / f"{PAIR_MODEL_NAME}.cbm"
FEATURES_PATH = GATE5_3_MODEL_ROOT / PAIR_MODEL_NAME / "features.csv"

GRID_LIST = [
    SAFE_GRID_NAME,
    AGG_GRID_NAME,
]

REQUIRE_FULL_GRID_COVERAGE = True
DELETE_STALE_OUTPUT_FOR_PAIR = False
RAW_REF_COLS = {
    "ref_close",
    "ref_btc_close",
    "ref_eth_close",
    "ref_close_feat",
    "ref_btc_close_feat",
    "ref_eth_close_feat",
}

FORBIDDEN_FEATURE_TOKENS = [
    "target",
    "score",
    "label",
    "future",
    "lookahead",
    "realized",
    "oracle",
    "winner",
    "pnl",
    "profit",
    "loss",
    "ret_",
    "_ret",
    "mfe",
    "mae",
    "first_tp",
    "first_sl",
    "tp_hit",
    "sl_hit",
    "tp_before_sl",
    "sl_before_tp",
    "ambiguous",
    "no_hit",
    "exit_",
    "_exit",
    "hold_minutes",
    "duration",
    "minute",
    "time_to",
    "bars_to",
    "exec",
    "grid_idx",
    "signal_best",
    "signal_worst",
    "g5_",
    "safe_g5_",
    "agg_g5_",
]

CAT_COLS = [
    "side",
]


# ============================================================
# SQL HELPERS
# ============================================================

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return '"' + name + '"'


def quote_relation(name: str) -> str:
    parts = name.split(".")
    if len(parts) not in (1, 2):
        raise ValueError(f"Unsafe SQL relation name: {name}")
    return ".".join(quote_ident(p) for p in parts)


def connect_db():
    return psycopg2.connect(DB_DSN)


def fetch_table_columns(conn, table_name: str) -> List[str]:
    if "." in table_name:
        schema_name, rel_name = table_name.split(".", 1)
    else:
        schema_name, rel_name = "public", table_name

    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """

    with conn.cursor() as cur:
        cur.execute(sql, (schema_name, rel_name))
        rows = cur.fetchall()

    return [str(r[0]) for r in rows]


def require_cols(cols: List[str], required: List[str], table_name: str) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(f"{table_name}: missing columns: {missing}")


def ensure_output_table(conn) -> None:
    table_sql = quote_relation(GATE5_3_OUTPUT_TABLE)

    sql = f"""
        CREATE TABLE IF NOT EXISTS {table_sql} (
            id BIGSERIAL PRIMARY KEY,

            signal_key TEXT NOT NULL,
            signal_id BIGINT NULL,
            symbol TEXT NOT NULL,
            signal_ts TIMESTAMPTZ NOT NULL,
            side TEXT NOT NULL,

            prod_pair_name TEXT NOT NULL,
            pair_model_name TEXT NOT NULL,

            safe_grid_name TEXT NOT NULL,
            agg_grid_name TEXT NOT NULL,
            chosen_grid_name TEXT NOT NULL,
            chosen_tp_atr DOUBLE PRECISION NOT NULL,
            chosen_sl_atr DOUBLE PRECISION NOT NULL,
            chosen_rr DOUBLE PRECISION NOT NULL,

            pred_label INTEGER NOT NULL,
            pred_proba DOUBLE PRECISION NOT NULL,

            safe_grid_proba DOUBLE PRECISION NOT NULL,
            agg_grid_proba DOUBLE PRECISION NOT NULL,
            proba_diff DOUBLE PRECISION NOT NULL,
            proba_ratio DOUBLE PRECISION NULL,

            safe_grid_tp_atr DOUBLE PRECISION NOT NULL,
            safe_grid_sl_atr DOUBLE PRECISION NOT NULL,
            safe_grid_rr DOUBLE PRECISION NOT NULL,

            agg_grid_tp_atr DOUBLE PRECISION NOT NULL,
            agg_grid_sl_atr DOUBLE PRECISION NOT NULL,
            agg_grid_rr DOUBLE PRECISION NOT NULL,

            rr_diff DOUBLE PRECISION NOT NULL,
            tp_diff DOUBLE PRECISION NOT NULL,
            sl_diff DOUBLE PRECISION NOT NULL,
            rr_ratio DOUBLE PRECISION NULL,

            gate5_3_model_path TEXT NOT NULL,
            gate5_3_model_feature_count INTEGER NOT NULL,
            missing_feature_count INTEGER NOT NULL DEFAULT 0,
            missing_features JSONB NOT NULL DEFAULT '[]'::jsonb,

            source_gate5_2_updated_at TIMESTAMPTZ NULL,

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            UNIQUE (signal_key, pair_model_name)
        )
    """

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()


def delete_stale_output(conn, current_signal_keys: List[str]) -> int:
    if not DELETE_STALE_OUTPUT_FOR_PAIR:
        return 0

    table_sql = quote_relation(GATE5_3_OUTPUT_TABLE)

    if not current_signal_keys:
        sql = f"""
            DELETE FROM {table_sql}
            WHERE prod_pair_name = %s
              AND pair_model_name = %s
        """
        params: Tuple[object, ...] = (PROD_PAIR_NAME, PAIR_MODEL_NAME)
    else:
        sql = f"""
            DELETE FROM {table_sql}
            WHERE prod_pair_name = %s
              AND pair_model_name = %s
              AND signal_key <> ALL(%s)
        """
        params = (PROD_PAIR_NAME, PAIR_MODEL_NAME, current_signal_keys)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        deleted = int(cur.rowcount)

    conn.commit()
    return deleted


# ============================================================
# DATA HELPERS
# ============================================================

def safe_float(value) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def load_model_and_features() -> Tuple[CatBoostClassifier, List[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Gate5_3 model not found: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Gate5_3 features file not found: {FEATURES_PATH}")

    features_df = pd.read_csv(FEATURES_PATH)
    if "feature" not in features_df.columns:
        raise RuntimeError(f"{FEATURES_PATH}: missing column 'feature'")

    feature_names = features_df["feature"].dropna().astype(str).tolist()
    if not feature_names:
        raise RuntimeError(f"{FEATURES_PATH}: empty feature list")

    bad_features = []
    for f in feature_names:
        f_low = f.lower()

        if f in RAW_REF_COLS:
            bad_features.append(f)
            continue

        if any(token in f_low for token in FORBIDDEN_FEATURE_TOKENS):
            bad_features.append(f)

    if bad_features:
        raise RuntimeError(
            "Gate5_3 feature file contains forbidden/leak-like features: "
            + ", ".join(sorted(bad_features))
        )

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    model_features = list(model.feature_names_ or [])
    if not model_features:
        raise RuntimeError(
            "Gate5_3 model.feature_names_ is empty. "
            "Нельзя безопасно запускать online prediction."
        )

    if model_features != feature_names:
        raise RuntimeError(
            "Gate5_3 model feature_names_ != features.csv. "
            "Нельзя безопасно запускать online prediction."
        )

    print("LOADED GATE5_3 MODEL:", MODEL_PATH)
    print("FEATURES_PATH:", FEATURES_PATH)
    print("FEATURE_COUNT:", len(feature_names))

    return model, feature_names


def fetch_gate5_2_batch(conn) -> pd.DataFrame:
    source_sql = quote_relation(GATE5_2_SOURCE_TABLE)
    output_sql = quote_relation(GATE5_3_OUTPUT_TABLE)

    source_cols = fetch_table_columns(conn, GATE5_2_SOURCE_TABLE)
    output_cols = fetch_table_columns(conn, GATE5_3_OUTPUT_TABLE)

    if not source_cols:
        raise RuntimeError(f"Source table not found or has no columns: {GATE5_2_SOURCE_TABLE}")

    if not output_cols:
        raise RuntimeError(f"Output table not found or has no columns: {GATE5_3_OUTPUT_TABLE}")

    required_source_cols = [
        "signal_key",
        "signal_id",
        "symbol",
        "signal_ts",
        "side",
        "prod_pair_name",
        "grid_name",
        "grid_idx",
        "tp_atr",
        "sl_atr",
        "rr",
        "grid_proba",
        "grid_tp_atr",
        "grid_sl_atr",
        "grid_rr",
        "sig_top1_proba",
        "sig_top2_proba",
        "sig_top1_minus_top2_proba",
        "sig_mean_proba",
        "sig_std_proba",
        "sig_max_rr",
        "sig_max_tp_atr",
        "sig_min_sl_atr",
        "grid_proba_to_top1_ratio",
        "grid_tp_minus_sig_max_tp",
        "grid_sl_minus_sig_min_sl",
        "grid_rr_minus_sig_max_rr",
        "grid_proba_vs_sig_mean",
        "updated_at",
    ]

    required_output_cols = [
        "signal_key",
        "pair_model_name",
        "source_gate5_2_updated_at",
    ]

    require_cols(source_cols, required_source_cols, GATE5_2_SOURCE_TABLE)
    require_cols(output_cols, required_output_cols, GATE5_3_OUTPUT_TABLE)

    where_parts = [
        "s.prod_pair_name = %s",
        "s.grid_name = ANY(%s)",
    ]

    params: List[object] = [
        PROD_PAIR_NAME,
        GRID_LIST,
    ]

    if LOOKBACK_HOURS is not None:
        where_parts.append("s.signal_ts >= NOW() - (%s::text)::interval")
        params.append(f"{LOOKBACK_HOURS} hours")

    where_sql = " AND ".join(where_parts)

    sql = f"""
        WITH dirty_keys AS (
            SELECT DISTINCT
                s.signal_key,
                MAX(s.signal_ts) AS signal_ts
            FROM {source_sql} s
            LEFT JOIN {output_sql} o
              ON o.signal_key = s.signal_key
             AND o.pair_model_name = %s
            WHERE {where_sql}
              AND (
                    o.signal_key IS NULL
                 OR o.source_gate5_2_updated_at IS NULL
                 OR s.updated_at > o.source_gate5_2_updated_at
              )
            GROUP BY s.signal_key
            ORDER BY MAX(s.signal_ts) DESC
            LIMIT %s
        )
        SELECT
            s.*
        FROM {source_sql} s
        INNER JOIN dirty_keys dk
            ON dk.signal_key = s.signal_key
        WHERE s.prod_pair_name = %s
          AND s.grid_name = ANY(%s)
        ORDER BY s.signal_ts DESC, s.symbol ASC, s.side ASC, s.grid_name ASC
    """

    final_params: List[object] = [PAIR_MODEL_NAME]
    final_params.extend(params)
    final_params.append(BATCH_LIMIT)
    final_params.extend([PROD_PAIR_NAME, GRID_LIST])

    df = pd.read_sql_query(sql, conn, params=final_params)
    return df

def normalize_gate5_2_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["signal_key"] = out["signal_key"].astype(str)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["signal_ts"] = pd.to_datetime(out["signal_ts"], errors="coerce", utc=True)
    out["side"] = out["side"].astype(str).str.upper()
    out["prod_pair_name"] = out["prod_pair_name"].astype(str)
    out["grid_name"] = out["grid_name"].astype(str)
    out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce", utc=True)

    numeric_cols = [
        "signal_id",
        "grid_idx",
        "tp_atr",
        "sl_atr",
        "rr",
        "grid_proba",
        "grid_tp_atr",
        "grid_sl_atr",
        "grid_rr",
        "sig_top1_proba",
        "sig_top2_proba",
        "sig_top1_minus_top2_proba",
        "sig_mean_proba",
        "sig_std_proba",
        "sig_max_rr",
        "sig_max_tp_atr",
        "sig_min_sl_atr",
        "grid_proba_to_top1_ratio",
        "grid_tp_minus_sig_max_tp",
        "grid_sl_minus_sig_min_sl",
        "grid_rr_minus_sig_max_rr",
        "grid_proba_vs_sig_mean",
    ]

    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(
        subset=[
            "signal_key",
            "symbol",
            "signal_ts",
            "side",
            "prod_pair_name",
            "grid_name",
            "grid_idx",
            "grid_proba",
            "grid_tp_atr",
            "grid_sl_atr",
            "grid_rr",
            "sig_top1_proba",
            "sig_top2_proba",
            "sig_top1_minus_top2_proba",
            "sig_mean_proba",
            "sig_std_proba",
            "sig_max_rr",
            "sig_max_tp_atr",
            "sig_min_sl_atr",
            "grid_proba_to_top1_ratio",
            "grid_tp_minus_sig_max_tp",
            "grid_sl_minus_sig_min_sl",
            "grid_rr_minus_sig_max_rr",
            "grid_proba_vs_sig_mean",
        ]
    ).copy()

    out = out[out["prod_pair_name"] == PROD_PAIR_NAME].copy()
    out = out[out["grid_name"].isin(GRID_LIST)].copy()
    out = out[out["side"].isin(["LONG", "SHORT"])].copy()

    bad_proba = (
        (out["grid_proba"] < 0.0)
        | (out["grid_proba"] > 1.0)
        | (~np.isfinite(out["grid_proba"]))
    )
    if bad_proba.any():
        raise RuntimeError(f"grid_proba has invalid rows: {int(bad_proba.sum())}")

    out = (
        out.sort_values(["signal_key", "grid_name", "updated_at"])
        .drop_duplicates(["signal_key", "grid_name"], keep="last")
        .reset_index(drop=True)
    )

    return out


def build_pair_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    counts = df.groupby("signal_key")["grid_name"].nunique()
    full_keys = counts[counts == len(GRID_LIST)].index.tolist()
    incomplete_keys = counts[counts != len(GRID_LIST)].index.tolist()

    report = {
        "raw_rows": int(len(df)),
        "unique_signals_raw": int(df["signal_key"].nunique()),
        "full_coverage_signals": int(len(full_keys)),
        "incomplete_coverage_signals": int(len(incomplete_keys)),
        "incomplete_signal_keys_sample": incomplete_keys[:30],
    }

    if REQUIRE_FULL_GRID_COVERAGE:
        work = df[df["signal_key"].isin(full_keys)].copy()
    else:
        work = df.copy()

    if work.empty:
        raise RuntimeError("No Gate5_2 rows after full grid coverage filter")

    base_cols = [
        "signal_key",
        "signal_id",
        "symbol",
        "signal_ts",
        "side",
        "sig_top1_proba",
        "sig_top2_proba",
        "sig_top1_minus_top2_proba",
        "sig_mean_proba",
        "sig_std_proba",
        "sig_max_rr",
        "sig_max_tp_atr",
        "sig_min_sl_atr",
    ]

    grid_cols = [
        "grid_proba",
        "grid_tp_atr",
        "grid_sl_atr",
        "grid_rr",
        "grid_idx",
        "grid_proba_to_top1_ratio",
        "grid_tp_minus_sig_max_tp",
        "grid_sl_minus_sig_min_sl",
        "grid_rr_minus_sig_max_rr",
        "grid_proba_vs_sig_mean",
    ]

    safe = work[work["grid_name"] == SAFE_GRID_NAME].copy()
    agg = work[work["grid_name"] == AGG_GRID_NAME].copy()

    safe = safe[base_cols + grid_cols + ["updated_at"]].copy()
    agg = agg[base_cols + grid_cols + ["updated_at"]].copy()

    safe = safe.rename(columns={c: f"safe_{c}" for c in grid_cols})
    agg = agg.rename(columns={c: f"agg_{c}" for c in grid_cols})

    safe = safe.rename(columns={"updated_at": "safe_source_gate5_2_updated_at"})
    agg = agg.rename(columns={"updated_at": "agg_source_gate5_2_updated_at"})

    pair_df = safe.merge(
        agg,
        on=base_cols,
        how="inner",
    )

    if pair_df.empty:
        raise RuntimeError("Pair merge produced zero rows")

    pair_df["safe_grid_name"] = SAFE_GRID_NAME
    pair_df["agg_grid_name"] = AGG_GRID_NAME

    pair_df["proba_diff"] = pair_df["agg_grid_proba"] - pair_df["safe_grid_proba"]
    pair_df["rr_diff"] = pair_df["agg_grid_rr"] - pair_df["safe_grid_rr"]
    pair_df["tp_diff"] = pair_df["agg_grid_tp_atr"] - pair_df["safe_grid_tp_atr"]
    pair_df["sl_diff"] = pair_df["agg_grid_sl_atr"] - pair_df["safe_grid_sl_atr"]

    pair_df["proba_ratio"] = np.where(
        pd.to_numeric(pair_df["safe_grid_proba"], errors="coerce") != 0.0,
        pd.to_numeric(pair_df["agg_grid_proba"], errors="coerce")
        / pd.to_numeric(pair_df["safe_grid_proba"], errors="coerce"),
        np.nan,
    )

    pair_df["rr_ratio"] = np.where(
        pd.to_numeric(pair_df["safe_grid_rr"], errors="coerce") != 0.0,
        pd.to_numeric(pair_df["agg_grid_rr"], errors="coerce")
        / pd.to_numeric(pair_df["safe_grid_rr"], errors="coerce"),
        np.nan,
    )

    pair_df["source_gate5_2_updated_at"] = pair_df[
        ["safe_source_gate5_2_updated_at", "agg_source_gate5_2_updated_at"]
    ].max(axis=1)

    pair_df = pair_df.drop(
        columns=[
            "safe_source_gate5_2_updated_at",
            "agg_source_gate5_2_updated_at",
        ],
        errors="ignore",
    )

    pair_df = pair_df.sort_values(["signal_ts", "symbol", "side"]).reset_index(drop=True)

    return pair_df, report


def make_model_matrix(
    df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    x = pd.DataFrame(index=df.index)
    missing_features: List[str] = []

    for f in feature_names:
        if f in df.columns:
            x[f] = df[f]
        else:
            x[f] = 0.0
            missing_features.append(f)

    for c in x.columns:
        if c in CAT_COLS:
            x[c] = x[c].astype(str).fillna("UNKNOWN")
        elif pd.api.types.is_bool_dtype(x[c]):
            x[c] = x[c].astype(int)
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)

    for c in x.columns:
        if c not in CAT_COLS:
            x[c] = x[c].fillna(0.0)

    return x, missing_features


def dataframe_to_upsert_rows(
    df: pd.DataFrame,
    proba: np.ndarray,
    pred_label: np.ndarray,
    feature_names: List[str],
    missing_features: List[str],
) -> List[Tuple[object, ...]]:
    rows: List[Tuple[object, ...]] = []

    for pos, (_, r) in enumerate(df.iterrows()):
        signal_id_value = None
        if pd.notna(r.get("signal_id")):
            signal_id_value = int(r["signal_id"])

        label = int(pred_label[pos])
        chosen_grid = AGG_GRID_NAME if label == 1 else SAFE_GRID_NAME

        if label == 1:
            chosen_tp_atr = float(r["agg_grid_tp_atr"])
            chosen_sl_atr = float(r["agg_grid_sl_atr"])
            chosen_rr = float(r["agg_grid_rr"])
        else:
            chosen_tp_atr = float(r["safe_grid_tp_atr"])
            chosen_sl_atr = float(r["safe_grid_sl_atr"])
            chosen_rr = float(r["safe_grid_rr"])

        rows.append(
            (
                str(r["signal_key"]),
                signal_id_value,
                str(r["symbol"]),
                pd.Timestamp(r["signal_ts"]).to_pydatetime(),
                str(r["side"]),

                PROD_PAIR_NAME,
                PAIR_MODEL_NAME,

                SAFE_GRID_NAME,
                AGG_GRID_NAME,
                chosen_grid,
                chosen_tp_atr,
                chosen_sl_atr,
                chosen_rr,

                label,
                float(proba[pos]),

                float(r["safe_grid_proba"]),
                float(r["agg_grid_proba"]),
                float(r["proba_diff"]),
                safe_float(r.get("proba_ratio")),

                float(r["safe_grid_tp_atr"]),
                float(r["safe_grid_sl_atr"]),
                float(r["safe_grid_rr"]),

                float(r["agg_grid_tp_atr"]),
                float(r["agg_grid_sl_atr"]),
                float(r["agg_grid_rr"]),

                float(r["rr_diff"]),
                float(r["tp_diff"]),
                float(r["sl_diff"]),
                safe_float(r.get("rr_ratio")),

                str(MODEL_PATH),
                int(len(feature_names)),
                int(len(missing_features)),
                json.dumps(missing_features, ensure_ascii=False),

                None if pd.isna(r.get("source_gate5_2_updated_at")) else pd.Timestamp(r["source_gate5_2_updated_at"]).to_pydatetime(),
            )
        )

    return rows


def upsert_gate5_3_decisions(conn, rows: List[Tuple[object, ...]]) -> None:
    if not rows:
        return

    table_sql = quote_relation(GATE5_3_OUTPUT_TABLE)

    sql = f"""
        INSERT INTO {table_sql} (
            signal_key,
            signal_id,
            symbol,
            signal_ts,
            side,

            prod_pair_name,
            pair_model_name,

            safe_grid_name,
            agg_grid_name,
            chosen_grid_name,
            chosen_tp_atr,
            chosen_sl_atr,
            chosen_rr,

            pred_label,
            pred_proba,

            safe_grid_proba,
            agg_grid_proba,
            proba_diff,
            proba_ratio,

            safe_grid_tp_atr,
            safe_grid_sl_atr,
            safe_grid_rr,

            agg_grid_tp_atr,
            agg_grid_sl_atr,
            agg_grid_rr,

            rr_diff,
            tp_diff,
            sl_diff,
            rr_ratio,

            gate5_3_model_path,
            gate5_3_model_feature_count,
            missing_feature_count,
            missing_features,

            source_gate5_2_updated_at
        )
        VALUES %s
        ON CONFLICT (signal_key, pair_model_name)
        DO UPDATE SET
            signal_id = EXCLUDED.signal_id,
            symbol = EXCLUDED.symbol,
            signal_ts = EXCLUDED.signal_ts,
            side = EXCLUDED.side,

            prod_pair_name = EXCLUDED.prod_pair_name,
            safe_grid_name = EXCLUDED.safe_grid_name,
            agg_grid_name = EXCLUDED.agg_grid_name,
            chosen_grid_name = EXCLUDED.chosen_grid_name,
            chosen_tp_atr = EXCLUDED.chosen_tp_atr,
            chosen_sl_atr = EXCLUDED.chosen_sl_atr,
            chosen_rr = EXCLUDED.chosen_rr,

            pred_label = EXCLUDED.pred_label,
            pred_proba = EXCLUDED.pred_proba,

            safe_grid_proba = EXCLUDED.safe_grid_proba,
            agg_grid_proba = EXCLUDED.agg_grid_proba,
            proba_diff = EXCLUDED.proba_diff,
            proba_ratio = EXCLUDED.proba_ratio,

            safe_grid_tp_atr = EXCLUDED.safe_grid_tp_atr,
            safe_grid_sl_atr = EXCLUDED.safe_grid_sl_atr,
            safe_grid_rr = EXCLUDED.safe_grid_rr,

            agg_grid_tp_atr = EXCLUDED.agg_grid_tp_atr,
            agg_grid_sl_atr = EXCLUDED.agg_grid_sl_atr,
            agg_grid_rr = EXCLUDED.agg_grid_rr,

            rr_diff = EXCLUDED.rr_diff,
            tp_diff = EXCLUDED.tp_diff,
            sl_diff = EXCLUDED.sl_diff,
            rr_ratio = EXCLUDED.rr_ratio,

            gate5_3_model_path = EXCLUDED.gate5_3_model_path,
            gate5_3_model_feature_count = EXCLUDED.gate5_3_model_feature_count,
            missing_feature_count = EXCLUDED.missing_feature_count,
            missing_features = EXCLUDED.missing_features,

            source_gate5_2_updated_at = EXCLUDED.source_gate5_2_updated_at,
            updated_at = NOW()
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)

    conn.commit()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("GATE5_2_SOURCE_TABLE:", GATE5_2_SOURCE_TABLE)
    print("GATE5_3_OUTPUT_TABLE:", GATE5_3_OUTPUT_TABLE)
    print("PROD_PAIR_NAME:", PROD_PAIR_NAME)
    print("PAIR_MODEL_NAME:", PAIR_MODEL_NAME)
    print("SAFE_GRID_NAME:", SAFE_GRID_NAME)
    print("AGG_GRID_NAME:", AGG_GRID_NAME)
    print("BATCH_LIMIT:", BATCH_LIMIT)
    print("LOOKBACK_HOURS:", LOOKBACK_HOURS)
    print("REQUIRE_FULL_GRID_COVERAGE:", REQUIRE_FULL_GRID_COVERAGE)
    print()

    model, feature_names = load_model_and_features()

    with connect_db() as conn:
        ensure_output_table(conn)

        source_cols = fetch_table_columns(conn, GATE5_2_SOURCE_TABLE)
        print("GATE5_2 SOURCE COLUMN COUNT:", len(source_cols))

        raw = fetch_gate5_2_batch(conn)
        print("RAW GATE5_2 ROWS:", len(raw))

        if raw.empty:
            print("NO GATE5_2 ROWS TO PROCESS")
            return

        norm = normalize_gate5_2_frame(raw)
        print("NORMALIZED GATE5_2 ROWS:", len(norm))
        print("NORMALIZED UNIQUE SIGNALS:", norm["signal_key"].nunique())

        pair_df, coverage_report = build_pair_frame(norm)

        print()
        print("COVERAGE REPORT:")
        print(json.dumps(coverage_report, ensure_ascii=False, indent=2))
        print()

        print("PAIR FRAME ROWS:", len(pair_df))
        print("PAIR FRAME UNIQUE SIGNALS:", pair_df["signal_key"].nunique())
        print("TS MIN:", pair_df["signal_ts"].min())
        print("TS MAX:", pair_df["signal_ts"].max())
        print("SYMBOLS:", pair_df["symbol"].nunique())
        print("SIDE DISTRIBUTION:")
        print(pair_df["side"].value_counts(dropna=False).to_string())
        print()

        x, missing_features = make_model_matrix(pair_df, feature_names)

        if missing_features:
            print("WARNING: MISSING MODEL FEATURES:", missing_features)

        proba = model.predict_proba(x)[:, 1]
        pred_label = (proba >= 0.5).astype(int)

        pair_df["pred_proba"] = proba
        pair_df["pred_label"] = pred_label
        pair_df["chosen_grid_name"] = np.where(
            pair_df["pred_label"].eq(1),
            AGG_GRID_NAME,
            SAFE_GRID_NAME,
        )

        print("PREDICTION SUMMARY:")
        print("  pred_proba_min:", float(np.min(proba)))
        print("  pred_proba_mean:", float(np.mean(proba)))
        print("  pred_proba_max:", float(np.max(proba)))
        print("  choose_safe:", int((pair_df["chosen_grid_name"] == SAFE_GRID_NAME).sum()))
        print("  choose_agg:", int((pair_df["chosen_grid_name"] == AGG_GRID_NAME).sum()))
        print()

        current_keys = sorted(pair_df["signal_key"].dropna().astype(str).unique().tolist())
        deleted_stale = delete_stale_output(conn, current_keys)

        rows = dataframe_to_upsert_rows(
            df=pair_df,
            proba=proba,
            pred_label=pred_label,
            feature_names=feature_names,
            missing_features=missing_features,
        )

        upsert_gate5_3_decisions(conn, rows)

        print("DELETED STALE GATE5_3 ROWS:", deleted_stale)
        print("UPSERTED GATE5_3 DECISION ROWS:", len(rows))
        print("DONE")


if __name__ == "__main__":
    main()
