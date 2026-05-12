from __future__ import annotations

from online.trading import config
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# PATHS
# ============================================================

ROOT = Path(os.environ.get("IMB_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

GATE5_1_MODEL_DIR = (
    ROOT
    / "pipeline"
    / "test"
    / "gate5"
    / "gate5_1_oof_no_raw_refs_thr010"
)

# ============================================================
# DATABASE CONFIG
# ============================================================

DB_DSN = os.environ.get(
    "IMB_DB_DSN",
    config.DB_DSN,
)

GATE4_SOURCE_TABLE = os.environ.get(
    "GATE4_SOURCE_TABLE",
    "online_gate4_predictions_no_raw_refs",
)

GATE4_FEATURE_TABLE = os.environ.get(
    "GATE4_FEATURE_TABLE",
    "online_gate4_features",
)

GATE5_1_OUTPUT_TABLE = os.environ.get(
    "GATE5_1_OUTPUT_TABLE",
    "online_gate5_1_scores",
)

BATCH_LIMIT = int(os.environ.get("GATE5_1_BATCH_LIMIT", "100000"))

# Если нужно прогонять только самые свежие сигналы, поставь, например:
# GATE5_1_LOOKBACK_HOURS=336
LOOKBACK_HOURS_TXT = os.environ.get("GATE5_1_LOOKBACK_HOURS", "").strip()
LOOKBACK_HOURS: Optional[int] = int(LOOKBACK_HOURS_TXT) if LOOKBACK_HOURS_TXT else None


# ============================================================
# ONLINE CONFIG
# ============================================================

PROD_PAIR_NAME = "tp225_sl075__vs__tp100_sl075"

GRID_CONFIGS: List[Dict[str, object]] = [
    {
        "grid_name": "tp225_sl075",
        "tp_atr": 2.25,
        "sl_atr": 0.75,
        "model_path": GATE5_1_MODEL_DIR / "model_tp225_sl075.cbm",
    },
    {
        "grid_name": "tp100_sl075",
        "tp_atr": 1.00,
        "sl_atr": 0.75,
        "model_path": GATE5_1_MODEL_DIR / "model_tp100_sl075.cbm",
    },
]

GATE4_CONFIDENCE_THRESHOLD = 0.55
SIDE_RATIO_MIN = 1.25
KEEP_UNRESOLVED_SIDE = False

SIGNAL_ID_CANDIDATES = [
    "signal_id",
    "id",
]

TS_COL_CANDIDATES = [
    "entry_ts",
    "signal_ts",
    "ts",
    "bar_ts",
    "candle_ts",
]

SYMBOL_COL = "symbol"

CORE_REQUIRED_COLS = [
    "symbol",
    "close",
    "atr14",
    "proba_long",
    "proba_short",
]

FORBIDDEN_FEATURE_TOKENS = [
    "future",
    "fwd",
    "lookahead",
    "outcome",
    "realized",
    "label",
    "target",
    "mfe",
    "mae",
    "first_tp",
    "first_sl",
    "tp_before_sl",
    "sl_before_tp",
    "ambiguous",
    "no_hit",
    "g5_target",
    "g5_mfe",
    "g5_mae",
    "g5_ttl",
    "g5_first",
    "g5_tp_hit",
    "g5_sl_hit",
    "g5_tp_before",
    "g5_sl_before",
    "g5_ambiguous",
    "g5_no_hit",
]

RAW_REF_COLS = {
    "ref_close",
    "ref_btc_close",
    "ref_eth_close",
    "ref_close_feat",
    "ref_btc_close_feat",
    "ref_eth_close_feat",
}


# ============================================================
# DB HELPERS
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


def ensure_output_table(conn) -> None:
    table_sql = quote_relation(GATE5_1_OUTPUT_TABLE)

    sql = f"""
        CREATE TABLE IF NOT EXISTS {table_sql} (
            id BIGSERIAL PRIMARY KEY,

            signal_key TEXT NOT NULL,
            signal_id BIGINT NULL,
            symbol TEXT NOT NULL,
            signal_ts TIMESTAMPTZ NOT NULL,
            side TEXT NOT NULL,

            prod_pair_name TEXT NOT NULL,
            grid_name TEXT NOT NULL,
            tp_atr DOUBLE PRECISION NOT NULL,
            sl_atr DOUBLE PRECISION NOT NULL,
            rr DOUBLE PRECISION NOT NULL,

            gate4_confidence DOUBLE PRECISION NULL,
            pred_side_confidence DOUBLE PRECISION NULL,
            pred_side_ratio DOUBLE PRECISION NULL,

            gate5_1_proba DOUBLE PRECISION NOT NULL,
            gate5_1_model_path TEXT NOT NULL,
            gate5_1_model_feature_count INTEGER NOT NULL,

            missing_feature_count INTEGER NOT NULL DEFAULT 0,
            missing_features JSONB NOT NULL DEFAULT '[]'::jsonb,

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            UNIQUE (signal_key, grid_name)
        )
    """

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()


def find_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    existing = set(cols)
    for c in candidates:
        if c in existing:
            return c
    return None

def fetch_gate4_batch(conn) -> pd.DataFrame:
    pred_sql = quote_relation(GATE4_SOURCE_TABLE)
    feat_sql = quote_relation(GATE4_FEATURE_TABLE)
    out_sql = quote_relation(GATE5_1_OUTPUT_TABLE)

    pred_cols = fetch_table_columns(conn, GATE4_SOURCE_TABLE)
    feat_cols = fetch_table_columns(conn, GATE4_FEATURE_TABLE)

    if not pred_cols:
        raise RuntimeError(f"Source table not found or has no columns: {GATE4_SOURCE_TABLE}")

    if not feat_cols:
        raise RuntimeError(f"Feature table not found or has no columns: {GATE4_FEATURE_TABLE}")

    pred_required = [
        "symbol",
        "signal_ts",
        "proba_long",
        "proba_short",
    ]

    feat_required = [
        "symbol",
        "entry_ts",
        "close",
        "atr14",
    ]

    missing_pred = [c for c in pred_required if c not in pred_cols]
    if missing_pred:
        raise RuntimeError(
            f"{GATE4_SOURCE_TABLE}: missing required columns for Gate5_1 online: {missing_pred}"
        )

    missing_feat = [c for c in feat_required if c not in feat_cols]
    if missing_feat:
        raise RuntimeError(
            f"{GATE4_FEATURE_TABLE}: missing required columns for Gate5_1 online: {missing_feat}"
        )

    where_parts = ["1 = 1"]
    params: List[object] = []

    if LOOKBACK_HOURS is not None:
        where_parts.append("p.signal_ts >= NOW() - (%s::text)::interval")
        params.append(f"{LOOKBACK_HOURS} hours")

    where_sql = " AND ".join(where_parts)

    grid_names = [str(x["grid_name"]) for x in GRID_CONFIGS]

    sql = f"""
        WITH gate4_base AS (
            SELECT
                f.*,

                p.signal_ts AS ts,

                p.proba_long AS proba_long,
                p.proba_short AS proba_short,

                p.pred_label AS gate4_pred_label,
                p.proba_0 AS gate4_proba_0,
                p.proba_1 AS gate4_proba_1,
                p.proba_2 AS gate4_proba_2,
                p.proba_ambig AS gate4_proba_ambig,
                p.model_mode AS gate4_model_mode,
                p.model_path AS gate4_model_path,
                p.features_path AS gate4_features_path,
                p.medians_path AS gate4_medians_path,
                p.created_at AS gate4_inserted_at,
                p.updated_at AS gate4_updated_at,

                GREATEST(p.proba_long, p.proba_short) AS gate4_confidence_sql,

                CASE
                    WHEN p.proba_long >= p.proba_short * %s
                     AND NOT (p.proba_short >= p.proba_long * %s)
                        THEN 'LONG'
                    WHEN p.proba_short >= p.proba_long * %s
                     AND NOT (p.proba_long >= p.proba_short * %s)
                        THEN 'SHORT'
                    ELSE 'UNRESOLVED'
                END AS pred_side_sql,

                CASE
                    WHEN LEAST(p.proba_long, p.proba_short) > 0
                        THEN GREATEST(p.proba_long, p.proba_short) / LEAST(p.proba_long, p.proba_short)
                    ELSE NULL
                END AS pred_side_ratio_sql

            FROM {pred_sql} p
            INNER JOIN {feat_sql} f
                ON p.symbol = f.symbol
               AND p.signal_ts = f.entry_ts

            WHERE {where_sql}
        ),
        gate4_resolved AS (
            SELECT
                *,
                (
                    UPPER(symbol)
                    || '|'
                    || to_char(ts AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"+00:00"')
                    || '|'
                    || pred_side_sql
                ) AS g5_signal_key
            FROM gate4_base
            WHERE gate4_confidence_sql >= %s
              AND pred_side_sql IN ('LONG', 'SHORT')
              AND pred_side_ratio_sql >= %s
        ),
        existing AS (
            SELECT
                signal_key,
                COUNT(DISTINCT grid_name) AS existing_grid_count
            FROM {out_sql}
            WHERE prod_pair_name = %s
              AND grid_name = ANY(%s)
            GROUP BY signal_key
        )
        SELECT
            r.*
        FROM gate4_resolved r
        LEFT JOIN existing e
            ON e.signal_key = r.g5_signal_key
        WHERE COALESCE(e.existing_grid_count, 0) < %s
        ORDER BY r.ts DESC, r.symbol ASC
        LIMIT %s
    """

    params = [
        SIDE_RATIO_MIN,
        SIDE_RATIO_MIN,
        SIDE_RATIO_MIN,
        SIDE_RATIO_MIN,
        *params,
        GATE4_CONFIDENCE_THRESHOLD,
        SIDE_RATIO_MIN,
        PROD_PAIR_NAME,
        grid_names,
        len(grid_names),
        BATCH_LIMIT,
    ]

    df = pd.read_sql_query(sql, conn, params=params)

    if len(df) == 0:
        return df

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    missing_core = [c for c in CORE_REQUIRED_COLS if c not in df.columns]
    if missing_core:
        raise RuntimeError(
            f"Gate5_1 merged frame: missing required columns after JOIN: {missing_core}"
        )

    return df

# ============================================================
# FEATURE HELPERS
# ============================================================

def normalize_gate4_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)

    for c in ["close", "atr14", "proba_long", "proba_short"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["symbol", "ts", "close", "atr14", "proba_long", "proba_short"]).copy()

    out["gate4_confidence"] = np.maximum(out["proba_long"], out["proba_short"])

    # ВАЖНО:
    # Это полностью повторяет offline Gate5 builder:
    # Gate5 сам решает сторону из proba_long/proba_short,
    # а не доверяет заранее записанной стороне из Gate4.
    long_ok = out["proba_long"] >= out["proba_short"] * SIDE_RATIO_MIN
    short_ok = out["proba_short"] >= out["proba_long"] * SIDE_RATIO_MIN

    resolved_side = np.where(
        long_ok & ~short_ok,
        "LONG",
        np.where(short_ok & ~long_ok, "SHORT", "UNRESOLVED"),
    )

    out["pred_side"] = resolved_side

    out["pred_side_int"] = np.where(
        out["pred_side"] == "LONG",
        1,
        np.where(out["pred_side"] == "SHORT", 0, np.nan),
    )

    out["pred_side_confidence"] = np.maximum(out["proba_long"], out["proba_short"])
    out["pred_side_gap"] = (out["proba_long"] - out["proba_short"]).abs()
    out["pred_side_ratio"] = np.where(
        np.minimum(out["proba_long"], out["proba_short"]) > 0,
        np.maximum(out["proba_long"], out["proba_short"]) / np.minimum(out["proba_long"], out["proba_short"]),
        np.nan,
    )

    out = out[out["gate4_confidence"] >= GATE4_CONFIDENCE_THRESHOLD].copy()

    if not KEEP_UNRESOLVED_SIDE:
        out = out[out["pred_side"].isin(["LONG", "SHORT"])].copy()
        out = out[out["pred_side_ratio"] >= SIDE_RATIO_MIN].copy()

    out = out.dropna(subset=["pred_side_int", "pred_side_confidence", "pred_side_ratio"]).copy()

    return out.reset_index(drop=True)


def build_signal_key(row: pd.Series, signal_id_col: Optional[str]) -> str:
    if signal_id_col is not None and pd.notna(row.get(signal_id_col)):
        return "id:" + str(int(row[signal_id_col]))

    ts = pd.Timestamp(row["ts"]).isoformat()
    symbol = str(row["symbol"])
    side = str(row["pred_side"])
    return f"{symbol}|{ts}|{side}"


def load_gate5_1_models() -> Dict[str, Dict[str, object]]:
    loaded: Dict[str, Dict[str, object]] = {}

    for cfg in GRID_CONFIGS:
        grid_name = str(cfg["grid_name"])
        model_path = Path(cfg["model_path"])

        if not model_path.exists():
            raise FileNotFoundError(f"Gate5_1 model not found for {grid_name}: {model_path}")

        model = CatBoostClassifier()
        model.load_model(str(model_path))

        feature_names = list(model.feature_names_ or [])
        if not feature_names:
            raise RuntimeError(
                f"{grid_name}: model has empty feature_names_. "
                "Нельзя безопасно собрать online-признаки без порядка features."
            )

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
                f"{grid_name}: forbidden/leak-like features inside CatBoost model: {bad_features}"
            )

        loaded[grid_name] = {
            "config": cfg,
            "model": model,
            "feature_names": feature_names,
            "model_path": str(model_path),
        }

        print(f"LOADED MODEL {grid_name}: {model_path}")
        print(f"  feature_count: {len(feature_names)}")

    return loaded


def make_model_matrix(
    df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    data: Dict[str, pd.Series] = {}
    missing_features: List[str] = []

    for f in feature_names:
        if f in df.columns:
            data[f] = df[f]
        else:
            data[f] = pd.Series(0.0, index=df.index)
            missing_features.append(f)

    x = pd.DataFrame(data, index=df.index)

    bool_cols = [c for c in x.columns if pd.api.types.is_bool_dtype(x[c])]
    if bool_cols:
        x[bool_cols] = x[bool_cols].astype(int)

    for c in x.columns:
        if c not in bool_cols:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return x, missing_features

def build_prediction_rows(
    df: pd.DataFrame,
    models: Dict[str, Dict[str, object]],
    source_cols: List[str],
) -> List[Tuple[object, ...]]:
    signal_id_col = find_first_existing(source_cols, SIGNAL_ID_CANDIDATES)

    rows: List[Tuple[object, ...]] = []

    signal_keys = [
        build_signal_key(row, signal_id_col)
        for _, row in df.iterrows()
    ]

    signal_ids: List[Optional[int]] = []
    if signal_id_col is not None:
        for _, row in df.iterrows():
            if pd.notna(row.get(signal_id_col)):
                signal_ids.append(int(row[signal_id_col]))
            else:
                signal_ids.append(None)
    else:
        signal_ids = [None] * len(df)

    symbols = df["symbol"].astype(str).tolist()
    signal_ts_values = [pd.Timestamp(x).to_pydatetime() for x in df["ts"].tolist()]
    sides = df["pred_side"].astype(str).tolist()

    gate4_confidences = [safe_float(x) for x in df["gate4_confidence"].tolist()]
    pred_side_confidences = [safe_float(x) for x in df["pred_side_confidence"].tolist()]
    pred_side_ratios = [safe_float(x) for x in df["pred_side_ratio"].tolist()]

    for grid_name, payload in models.items():
        cfg = payload["config"]
        model = payload["model"]
        feature_names = payload["feature_names"]
        model_path = str(payload["model_path"])

        x, missing_features = make_model_matrix(df, feature_names)
        proba = model.predict_proba(x)[:, 1]

        tp_atr = float(cfg["tp_atr"])
        sl_atr = float(cfg["sl_atr"])
        rr = tp_atr / sl_atr if sl_atr > 0 else np.nan

        missing_feature_count = int(len(missing_features))
        missing_features_json = json.dumps(missing_features, ensure_ascii=False)
        model_feature_count = int(len(feature_names))

        for pos in range(len(df)):
            rows.append(
                (
                    signal_keys[pos],
                    signal_ids[pos],
                    symbols[pos],
                    signal_ts_values[pos],
                    sides[pos],

                    PROD_PAIR_NAME,
                    grid_name,
                    tp_atr,
                    sl_atr,
                    rr,

                    gate4_confidences[pos],
                    pred_side_confidences[pos],
                    pred_side_ratios[pos],

                    float(proba[pos]),
                    model_path,
                    model_feature_count,
                    missing_feature_count,
                    missing_features_json,
                )
            )

    return rows

def safe_float(value) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def upsert_gate5_1_scores(conn, rows: List[Tuple[object, ...]]) -> None:
    if not rows:
        return

    table_sql = quote_relation(GATE5_1_OUTPUT_TABLE)

    sql = f"""
        INSERT INTO {table_sql} (
            signal_key,
            signal_id,
            symbol,
            signal_ts,
            side,

            prod_pair_name,
            grid_name,
            tp_atr,
            sl_atr,
            rr,

            gate4_confidence,
            pred_side_confidence,
            pred_side_ratio,

            gate5_1_proba,
            gate5_1_model_path,
            gate5_1_model_feature_count,
            missing_feature_count,
            missing_features
        )
        VALUES %s
        ON CONFLICT (signal_key, grid_name)
        DO UPDATE SET
            signal_id = EXCLUDED.signal_id,
            symbol = EXCLUDED.symbol,
            signal_ts = EXCLUDED.signal_ts,
            side = EXCLUDED.side,

            prod_pair_name = EXCLUDED.prod_pair_name,
            tp_atr = EXCLUDED.tp_atr,
            sl_atr = EXCLUDED.sl_atr,
            rr = EXCLUDED.rr,

            gate4_confidence = EXCLUDED.gate4_confidence,
            pred_side_confidence = EXCLUDED.pred_side_confidence,
            pred_side_ratio = EXCLUDED.pred_side_ratio,

            gate5_1_proba = EXCLUDED.gate5_1_proba,
            gate5_1_model_path = EXCLUDED.gate5_1_model_path,
            gate5_1_model_feature_count = EXCLUDED.gate5_1_model_feature_count,
            missing_feature_count = EXCLUDED.missing_feature_count,
            missing_features = EXCLUDED.missing_features,
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
    print("GATE4_SOURCE_TABLE:", GATE4_SOURCE_TABLE)
    print("GATE4_FEATURE_TABLE:", GATE4_FEATURE_TABLE)
    print("GATE5_1_OUTPUT_TABLE:", GATE5_1_OUTPUT_TABLE)
    print("PROD_PAIR_NAME:", PROD_PAIR_NAME)
    print("GRID_LIST:", [str(x["grid_name"]) for x in GRID_CONFIGS])
    print("GATE4_CONFIDENCE_THRESHOLD:", GATE4_CONFIDENCE_THRESHOLD)
    print("SIDE_RATIO_MIN:", SIDE_RATIO_MIN)
    print("BATCH_LIMIT:", BATCH_LIMIT)
    print("LOOKBACK_HOURS:", LOOKBACK_HOURS)
    print()

    models = load_gate5_1_models()

    with connect_db() as conn:
        ensure_output_table(conn)

        pred_cols = fetch_table_columns(conn, GATE4_SOURCE_TABLE)
        feat_cols = fetch_table_columns(conn, GATE4_FEATURE_TABLE)

        print("GATE4 PRED COLUMN COUNT:", len(pred_cols))
        print("GATE4 FEATURE COLUMN COUNT:", len(feat_cols))

        df_raw = fetch_gate4_batch(conn)
        print("RAW MERGED ROWS:", len(df_raw))
        if len(df_raw):
            print("RAW TS MIN:", df_raw["ts"].min())
            print("RAW TS MAX:", df_raw["ts"].max())
            print("RAW SYMBOLS:", df_raw["symbol"].nunique())


        source_cols = list(df_raw.columns)

        if len(df_raw) == 0:
            print("NO ROWS TO PROCESS")
            return

        df = normalize_gate4_frame(df_raw)

        print("ROWS AFTER GATE4 FILTER + SIDE RESOLUTION:", len(df))
        if len(df):
            print("TS MIN:", df["ts"].min())
            print("TS MAX:", df["ts"].max())
            print("SYMBOLS:", df["symbol"].nunique())
            print("SIDE DISTRIBUTION:")
            print(df["pred_side"].value_counts(dropna=False).to_string())
            print()

        if len(df) == 0:
            print("NO ROWS AFTER FILTERS")
            return

        rows = build_prediction_rows(
            df=df,
            models=models,
            source_cols=source_cols,
        )

        upsert_gate5_1_scores(conn, rows)

        print("UPSERTED GATE5_1 SCORE ROWS:", len(rows))
        print("SIGNALS PROCESSED:", len(df))
        print("GRIDS PER SIGNAL:", len(GRID_CONFIGS))
        print("DONE")


if __name__ == "__main__":
    main()
