from __future__ import annotations

from online.trading import config
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from online.oos_context import append_oos_sql_filters, get_online_oos_context

import numpy as np
import pandas as pd

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

DB_DSN = config.DB_DSN

GATE5_1_SOURCE_TABLE = os.environ.get(
    "GATE5_1_SOURCE_TABLE",
    "online_gate5_1_scores",
)

GATE5_2_OUTPUT_TABLE = os.environ.get(
    "GATE5_2_OUTPUT_TABLE",
    "online_gate5_2_ranker",
)

BATCH_LIMIT = int(os.environ.get("GATE5_2_BATCH_LIMIT", "100000"))

LOOKBACK_HOURS_TXT = os.environ.get("GATE5_2_LOOKBACK_HOURS", "").strip()
LOOKBACK_HOURS: Optional[int] = int(LOOKBACK_HOURS_TXT) if LOOKBACK_HOURS_TXT else None


# ============================================================
# ONLINE CONFIG
# ============================================================

PROD_PAIR_NAME = "tp225_sl075__vs__tp100_sl075"

GRID_LIST = [
    "tp225_sl075",
    "tp100_sl075",
]

GRID_IDX_MAP = {
    "tp225_sl075": 0,
    "tp100_sl075": 1,
}

REQUIRE_FULL_GRID_COVERAGE = True
DELETE_STALE_OUTPUT_FOR_PAIR = False


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
    table_sql = quote_relation(GATE5_2_OUTPUT_TABLE)

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
            grid_idx INTEGER NOT NULL,
            tp_atr DOUBLE PRECISION NOT NULL,
            sl_atr DOUBLE PRECISION NOT NULL,
            rr DOUBLE PRECISION NOT NULL,

            grid_proba DOUBLE PRECISION NOT NULL,
            grid_tp_atr DOUBLE PRECISION NOT NULL,
            grid_sl_atr DOUBLE PRECISION NOT NULL,
            grid_rr DOUBLE PRECISION NOT NULL,

            sig_top1_proba DOUBLE PRECISION NOT NULL,
            sig_top2_proba DOUBLE PRECISION NOT NULL,
            sig_top1_minus_top2_proba DOUBLE PRECISION NOT NULL,
            sig_mean_proba DOUBLE PRECISION NOT NULL,
            sig_std_proba DOUBLE PRECISION NOT NULL,
            sig_max_rr DOUBLE PRECISION NOT NULL,
            sig_max_tp_atr DOUBLE PRECISION NOT NULL,
            sig_min_sl_atr DOUBLE PRECISION NOT NULL,

            grid_proba_to_top1_ratio DOUBLE PRECISION NOT NULL,
            grid_tp_minus_sig_max_tp DOUBLE PRECISION NOT NULL,
            grid_sl_minus_sig_min_sl DOUBLE PRECISION NOT NULL,
            grid_rr_minus_sig_max_rr DOUBLE PRECISION NOT NULL,
            grid_proba_vs_sig_mean DOUBLE PRECISION NOT NULL,

            gate4_confidence DOUBLE PRECISION NULL,
            pred_side_confidence DOUBLE PRECISION NULL,
            pred_side_ratio DOUBLE PRECISION NULL,

            source_gate5_1_updated_at TIMESTAMPTZ NULL,

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            UNIQUE (signal_key, grid_name)
        )
    """

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()

    existing_cols = set(fetch_table_columns(conn, GATE5_2_OUTPUT_TABLE))
    required_existing_cols = {
        "tp_atr",
        "sl_atr",
        "rr",
    }
    missing_existing_cols = sorted(required_existing_cols - existing_cols)

    if missing_existing_cols:
        raise RuntimeError(
            "{} exists but missing required columns: {}. "
            "Table owner must add these columns or table must be recreated by owner.".format(
                GATE5_2_OUTPUT_TABLE,
                missing_existing_cols,
            )
        )


def delete_stale_output(conn, current_signal_keys: List[str]) -> int:
    if not DELETE_STALE_OUTPUT_FOR_PAIR:
        return 0

    table_sql = quote_relation(GATE5_2_OUTPUT_TABLE)

    if not current_signal_keys:
        sql = f"""
            DELETE FROM {table_sql}
            WHERE prod_pair_name = %s
        """
        params: Tuple[object, ...] = (PROD_PAIR_NAME,)
    else:
        sql = f"""
            DELETE FROM {table_sql}
            WHERE prod_pair_name = %s
              AND signal_key <> ALL(%s)
        """
        params = (PROD_PAIR_NAME, current_signal_keys)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        deleted = int(cur.rowcount)

    conn.commit()
    return deleted


# ============================================================
# DATA HELPERS
# ============================================================

def parse_tp_sl(grid_name: str) -> Tuple[float, float]:
    left, right = grid_name.split("_")
    tp_atr = float(left.replace("tp", "")) / 100.0
    sl_atr = float(right.replace("sl", "")) / 100.0
    return tp_atr, sl_atr


def safe_float(value) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def fetch_gate5_1_batch(conn) -> pd.DataFrame:
    source_sql = quote_relation(GATE5_1_SOURCE_TABLE)
    output_sql = quote_relation(GATE5_2_OUTPUT_TABLE)

    source_cols = fetch_table_columns(conn, GATE5_1_SOURCE_TABLE)
    output_cols = fetch_table_columns(conn, GATE5_2_OUTPUT_TABLE)

    if not source_cols:
        raise RuntimeError(f"Source table not found or has no columns: {GATE5_1_SOURCE_TABLE}")

    if not output_cols:
        raise RuntimeError(f"Output table not found or has no columns: {GATE5_2_OUTPUT_TABLE}")

    required_source_cols = [
        "signal_key",
        "signal_id",
        "symbol",
        "signal_ts",
        "side",
        "prod_pair_name",
        "grid_name",
        "tp_atr",
        "sl_atr",
        "rr",
        "gate5_1_proba",
        "gate4_confidence",
        "pred_side_confidence",
        "pred_side_ratio",
        "updated_at",
    ]

    required_output_cols = [
        "signal_key",
        "grid_name",
        "source_gate5_1_updated_at",
    ]

    require_cols(source_cols, required_source_cols, GATE5_1_SOURCE_TABLE)
    require_cols(output_cols, required_output_cols, GATE5_2_OUTPUT_TABLE)

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

    append_oos_sql_filters(
        where_parts=where_parts,
        params=params,
        table_alias="s",
        ts_column="signal_ts",
        symbol_column="symbol",
    )

    where_sql = " AND ".join(where_parts)

    sql = f"""
        WITH dirty_keys AS (
            SELECT DISTINCT
                s.signal_key,
                MAX(s.signal_ts) AS signal_ts
            FROM {source_sql} s
            LEFT JOIN {output_sql} o
              ON o.signal_key = s.signal_key
             AND o.grid_name = s.grid_name
            WHERE {where_sql}
              AND (
                    o.signal_key IS NULL
                 OR o.source_gate5_1_updated_at IS NULL
                 OR s.updated_at > o.source_gate5_1_updated_at
              )
            GROUP BY s.signal_key
            ORDER BY MAX(s.signal_ts) DESC
            LIMIT %s
        )
        SELECT
            s.signal_key,
            s.signal_id,
            s.symbol,
            s.signal_ts,
            s.side,
            s.prod_pair_name,
            s.grid_name,
            s.tp_atr,
            s.sl_atr,
            s.rr,
            s.gate5_1_proba,
            s.gate4_confidence,
            s.pred_side_confidence,
            s.pred_side_ratio,
            s.updated_at
        FROM {source_sql} s
        INNER JOIN dirty_keys dk
            ON dk.signal_key = s.signal_key
        WHERE s.prod_pair_name = %s
          AND s.grid_name = ANY(%s)
        ORDER BY s.signal_ts DESC, s.symbol ASC, s.side ASC, s.grid_name ASC
    """

    params.append(BATCH_LIMIT)
    params.extend([PROD_PAIR_NAME, GRID_LIST])

    df = pd.read_sql_query(sql, conn, params=params)
    return df

def normalize_gate5_1_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["signal_key"] = out["signal_key"].astype(str)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["signal_ts"] = pd.to_datetime(out["signal_ts"], errors="coerce", utc=True)
    out["side"] = out["side"].astype(str).str.upper()
    out["prod_pair_name"] = out["prod_pair_name"].astype(str)
    out["grid_name"] = out["grid_name"].astype(str)

    numeric_cols = [
        "signal_id",
        "tp_atr",
        "sl_atr",
        "rr",
        "gate5_1_proba",
        "gate4_confidence",
        "pred_side_confidence",
        "pred_side_ratio",
    ]

    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce", utc=True)

    out = out.dropna(
        subset=[
            "signal_key",
            "symbol",
            "signal_ts",
            "side",
            "grid_name",
            "tp_atr",
            "sl_atr",
            "rr",
            "gate5_1_proba",
        ]
    ).copy()

    out = out[out["prod_pair_name"] == PROD_PAIR_NAME].copy()
    out = out[out["grid_name"].isin(GRID_LIST)].copy()
    out = out[out["side"].isin(["LONG", "SHORT"])].copy()

    bad_proba = (
        (out["gate5_1_proba"] < 0.0)
        | (out["gate5_1_proba"] > 1.0)
        | (~np.isfinite(out["gate5_1_proba"]))
    )
    if bad_proba.any():
        bad_n = int(bad_proba.sum())
        raise RuntimeError(f"gate5_1_proba has invalid rows: {bad_n}")

    out = (
        out.sort_values(["signal_key", "grid_name", "updated_at"])
        .drop_duplicates(["signal_key", "grid_name"], keep="last")
        .reset_index(drop=True)
    )

    return out


def build_signal_level_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
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
        raise RuntimeError("No Gate5_1 rows after full grid coverage filter")

    signal_meta_cols = [
        "signal_key",
        "signal_id",
        "symbol",
        "signal_ts",
        "side",
    ]

    meta_df = (
        work.sort_values(["signal_key", "updated_at"])
        .groupby("signal_key", as_index=False)
        .last()[signal_meta_cols]
        .copy()
    )

    pivot = work.pivot_table(
        index=["signal_key"],
        columns="grid_name",
        values=["gate5_1_proba", "tp_atr", "sl_atr", "rr"],
        aggfunc="last",
        dropna=False,
    )

    pivot.columns = [f"{metric}__{grid}" for metric, grid in pivot.columns]
    pivot = pivot.reset_index()

    pivot = pivot.merge(
        meta_df,
        on="signal_key",
        how="left",
    )

    for grid in GRID_LIST:
        required = [
            f"gate5_1_proba__{grid}",
            f"tp_atr__{grid}",
            f"sl_atr__{grid}",
            f"rr__{grid}",
        ]
        missing = [c for c in required if c not in pivot.columns]
        if missing:
            raise RuntimeError(f"pivot missing columns for {grid}: {missing}")

    proba_cols = [f"gate5_1_proba__{g}" for g in GRID_LIST]
    rr_cols = [f"rr__{g}" for g in GRID_LIST]
    tp_cols = [f"tp_atr__{g}" for g in GRID_LIST]
    sl_cols = [f"sl_atr__{g}" for g in GRID_LIST]

    if pivot[proba_cols + rr_cols + tp_cols + sl_cols].isna().any().any():
        bad = pivot[proba_cols + rr_cols + tp_cols + sl_cols].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise RuntimeError(f"NaN after pivot: {bad}")

    proba_mat = pivot[proba_cols].to_numpy(dtype=float)
    rr_mat = pivot[rr_cols].to_numpy(dtype=float)
    tp_mat = pivot[tp_cols].to_numpy(dtype=float)
    sl_mat = pivot[sl_cols].to_numpy(dtype=float)

    if not np.isfinite(proba_mat).all():
        raise RuntimeError("Non-finite proba in signal-level matrix")

    n = len(pivot)
    proba_order = np.argsort(-proba_mat, axis=1)

    top1_idx = proba_order[:, 0]
    top2_idx = proba_order[:, 1]

    pivot["sig_top1_proba"] = proba_mat[np.arange(n), top1_idx]
    pivot["sig_top2_proba"] = proba_mat[np.arange(n), top2_idx]
    pivot["sig_top1_minus_top2_proba"] = pivot["sig_top1_proba"] - pivot["sig_top2_proba"]
    pivot["sig_mean_proba"] = np.mean(proba_mat, axis=1)
    pivot["sig_std_proba"] = np.std(proba_mat, axis=1)
    pivot["sig_max_rr"] = np.max(rr_mat, axis=1)
    pivot["sig_max_tp_atr"] = np.max(tp_mat, axis=1)
    pivot["sig_min_sl_atr"] = np.min(sl_mat, axis=1)

    source_meta = (
        work.sort_values(["signal_key", "updated_at"])
        .groupby("signal_key", as_index=False)
        .last()[
            [
                "signal_key",
                "gate4_confidence",
                "pred_side_confidence",
                "pred_side_ratio",
                "updated_at",
            ]
        ]
        .rename(columns={"updated_at": "source_gate5_1_updated_at"})
    )

    pivot = pivot.merge(source_meta, on="signal_key", how="left")

    return pivot.reset_index(drop=True), report

def build_ranker_rows(signal_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

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
        "gate4_confidence",
        "pred_side_confidence",
        "pred_side_ratio",
        "source_gate5_1_updated_at",
    ]

    for grid in GRID_LIST:
        block = signal_df[base_cols].copy()

        block["prod_pair_name"] = PROD_PAIR_NAME
        block["grid_name"] = grid
        block["grid_idx"] = int(GRID_IDX_MAP[grid])

        block["grid_proba"] = pd.to_numeric(signal_df[f"gate5_1_proba__{grid}"], errors="coerce")
        block["grid_tp_atr"] = pd.to_numeric(signal_df[f"tp_atr__{grid}"], errors="coerce")
        block["grid_sl_atr"] = pd.to_numeric(signal_df[f"sl_atr__{grid}"], errors="coerce")
        block["grid_rr"] = pd.to_numeric(signal_df[f"rr__{grid}"], errors="coerce")
        block["tp_atr"] = block["grid_tp_atr"]
        block["sl_atr"] = block["grid_sl_atr"]
        block["rr"] = block["grid_rr"]

        block["grid_proba_to_top1_ratio"] = np.where(
            pd.to_numeric(block["sig_top1_proba"], errors="coerce") > 0.0,
            pd.to_numeric(block["grid_proba"], errors="coerce") / pd.to_numeric(block["sig_top1_proba"], errors="coerce"),
            0.0,
        )

        block["grid_tp_minus_sig_max_tp"] = block["grid_tp_atr"] - block["sig_max_tp_atr"]
        block["grid_sl_minus_sig_min_sl"] = block["grid_sl_atr"] - block["sig_min_sl_atr"]
        block["grid_rr_minus_sig_max_rr"] = block["grid_rr"] - block["sig_max_rr"]
        block["grid_proba_vs_sig_mean"] = block["grid_proba"] - block["sig_mean_proba"]

        rows.append(block)

    out = pd.concat(rows, ignore_index=True)

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
        "gate4_confidence",
        "pred_side_confidence",
        "pred_side_ratio",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    required_non_null = [
        "signal_key",
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
    ]

    null_counts = out[required_non_null].isna().sum()
    bad = null_counts[null_counts > 0].to_dict()
    if bad:
        raise RuntimeError(f"Gate5_2 ranker rows have NULL in required columns: {bad}")

    out = out.sort_values(["signal_ts", "symbol", "side", "grid_idx"]).reset_index(drop=True)
    return out


def dataframe_to_upsert_rows(df: pd.DataFrame) -> List[Tuple[object, ...]]:
    rows: List[Tuple[object, ...]] = []

    for _, r in df.iterrows():
        signal_id_value = None
        if pd.notna(r.get("signal_id")):
            signal_id_value = int(r["signal_id"])

        rows.append(
            (
                str(r["signal_key"]),
                signal_id_value,
                str(r["symbol"]),
                pd.Timestamp(r["signal_ts"]).to_pydatetime(),
                str(r["side"]),

                str(r["prod_pair_name"]),
                str(r["grid_name"]),
                int(r["grid_idx"]),

                float(r["tp_atr"]),
                float(r["sl_atr"]),
                float(r["rr"]),

                float(r["grid_proba"]),
                float(r["grid_tp_atr"]),
                float(r["grid_sl_atr"]),
                float(r["grid_rr"]),

                float(r["sig_top1_proba"]),
                float(r["sig_top2_proba"]),
                float(r["sig_top1_minus_top2_proba"]),
                float(r["sig_mean_proba"]),
                float(r["sig_std_proba"]),
                float(r["sig_max_rr"]),
                float(r["sig_max_tp_atr"]),
                float(r["sig_min_sl_atr"]),

                float(r["grid_proba_to_top1_ratio"]),
                float(r["grid_tp_minus_sig_max_tp"]),
                float(r["grid_sl_minus_sig_min_sl"]),
                float(r["grid_rr_minus_sig_max_rr"]),
                float(r["grid_proba_vs_sig_mean"]),

                safe_float(r.get("gate4_confidence")),
                safe_float(r.get("pred_side_confidence")),
                safe_float(r.get("pred_side_ratio")),

                None if pd.isna(r.get("source_gate5_1_updated_at")) else pd.Timestamp(r["source_gate5_1_updated_at"]).to_pydatetime(),
            )
        )

    return rows


def upsert_gate5_2_ranker(conn, rows: List[Tuple[object, ...]]) -> None:
    if not rows:
        return

    table_sql = quote_relation(GATE5_2_OUTPUT_TABLE)

    sql = f"""
        INSERT INTO {table_sql} (
            signal_key,
            signal_id,
            symbol,
            signal_ts,
            side,

            prod_pair_name,
            grid_name,
            grid_idx,
            tp_atr,
            sl_atr,
            rr,

            grid_proba,
            grid_tp_atr,
            grid_sl_atr,
            grid_rr,

            sig_top1_proba,
            sig_top2_proba,
            sig_top1_minus_top2_proba,
            sig_mean_proba,
            sig_std_proba,
            sig_max_rr,
            sig_max_tp_atr,
            sig_min_sl_atr,

            grid_proba_to_top1_ratio,
            grid_tp_minus_sig_max_tp,
            grid_sl_minus_sig_min_sl,
            grid_rr_minus_sig_max_rr,
            grid_proba_vs_sig_mean,

            gate4_confidence,
            pred_side_confidence,
            pred_side_ratio,

            source_gate5_1_updated_at
        )
        VALUES %s
        ON CONFLICT (signal_key, grid_name)
        DO UPDATE SET
            signal_id = EXCLUDED.signal_id,
            symbol = EXCLUDED.symbol,
            signal_ts = EXCLUDED.signal_ts,
            side = EXCLUDED.side,

            prod_pair_name = EXCLUDED.prod_pair_name,
            grid_idx = EXCLUDED.grid_idx,
            tp_atr = EXCLUDED.tp_atr,
            sl_atr = EXCLUDED.sl_atr,
            rr = EXCLUDED.rr,

            grid_proba = EXCLUDED.grid_proba,
            grid_tp_atr = EXCLUDED.grid_tp_atr,
            grid_sl_atr = EXCLUDED.grid_sl_atr,
            grid_rr = EXCLUDED.grid_rr,

            sig_top1_proba = EXCLUDED.sig_top1_proba,
            sig_top2_proba = EXCLUDED.sig_top2_proba,
            sig_top1_minus_top2_proba = EXCLUDED.sig_top1_minus_top2_proba,
            sig_mean_proba = EXCLUDED.sig_mean_proba,
            sig_std_proba = EXCLUDED.sig_std_proba,
            sig_max_rr = EXCLUDED.sig_max_rr,
            sig_max_tp_atr = EXCLUDED.sig_max_tp_atr,
            sig_min_sl_atr = EXCLUDED.sig_min_sl_atr,

            grid_proba_to_top1_ratio = EXCLUDED.grid_proba_to_top1_ratio,
            grid_tp_minus_sig_max_tp = EXCLUDED.grid_tp_minus_sig_max_tp,
            grid_sl_minus_sig_min_sl = EXCLUDED.grid_sl_minus_sig_min_sl,
            grid_rr_minus_sig_max_rr = EXCLUDED.grid_rr_minus_sig_max_rr,
            grid_proba_vs_sig_mean = EXCLUDED.grid_proba_vs_sig_mean,

            gate4_confidence = EXCLUDED.gate4_confidence,
            pred_side_confidence = EXCLUDED.pred_side_confidence,
            pred_side_ratio = EXCLUDED.pred_side_ratio,

            source_gate5_1_updated_at = EXCLUDED.source_gate5_1_updated_at,
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
    print("GATE5_1_SOURCE_TABLE:", GATE5_1_SOURCE_TABLE)
    print("GATE5_2_OUTPUT_TABLE:", GATE5_2_OUTPUT_TABLE)
    print("PROD_PAIR_NAME:", PROD_PAIR_NAME)
    print("GRID_LIST:", GRID_LIST)
    print("BATCH_LIMIT:", BATCH_LIMIT)
    oos_ctx = get_online_oos_context()
    print("LOOKBACK_HOURS:", LOOKBACK_HOURS)
    print("OOS_MODE:", oos_ctx.enabled)
    print("OOS_SYMBOLS:", ",".join(oos_ctx.symbols))
    print("OOS_START:", oos_ctx.start_text)
    print("OOS_END:", oos_ctx.end_text)
    print("REQUIRE_FULL_GRID_COVERAGE:", REQUIRE_FULL_GRID_COVERAGE)
    print()

    with connect_db() as conn:
        ensure_output_table(conn)

        source_cols = fetch_table_columns(conn, GATE5_1_SOURCE_TABLE)
        print("GATE5_1 SOURCE COLUMN COUNT:", len(source_cols))

        raw = fetch_gate5_1_batch(conn)
        print("RAW GATE5_1 ROWS:", len(raw))

        if raw.empty:
            print("NO GATE5_1 ROWS TO PROCESS")
            return

        norm = normalize_gate5_1_frame(raw)
        print("NORMALIZED GATE5_1 ROWS:", len(norm))
        print("NORMALIZED UNIQUE SIGNALS:", norm["signal_key"].nunique())

        signal_df, coverage_report = build_signal_level_frame(norm)
        print()
        print("COVERAGE REPORT:")
        print(json.dumps(coverage_report, ensure_ascii=False, indent=2))

        ranker_df = build_ranker_rows(signal_df)
        print()
        print("GATE5_2 RANKER ROWS:", len(ranker_df))
        print("GATE5_2 UNIQUE SIGNALS:", ranker_df["signal_key"].nunique())
        print("ROWS PER SIGNAL MEAN:", float(ranker_df.groupby("signal_key").size().mean()))
        print("TS MIN:", ranker_df["signal_ts"].min())
        print("TS MAX:", ranker_df["signal_ts"].max())
        print("SYMBOLS:", ranker_df["symbol"].nunique())
        print("SIDE DISTRIBUTION:")
        print(ranker_df.drop_duplicates("signal_key")["side"].value_counts(dropna=False).to_string())
        print("GRID DISTRIBUTION:")
        print(ranker_df["grid_name"].value_counts(dropna=False).to_string())
        print()

        current_keys = sorted(ranker_df["signal_key"].dropna().astype(str).unique().tolist())
        deleted_stale = delete_stale_output(conn, current_keys)

        rows = dataframe_to_upsert_rows(ranker_df)
        upsert_gate5_2_ranker(conn, rows)

        print("DELETED STALE GATE5_2 ROWS:", deleted_stale)
        print("UPSERTED GATE5_2 RANKER ROWS:", len(rows))
        print("DONE")


if __name__ == "__main__":
    main()
