
from __future__ import annotations

from online.trading import config
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import traceback
import warnings

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)

ROOT = Path(os.environ.get("IMB_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DB_DSN = config.DB_DSN

ONLINE_GATE1_FEATURES_TABLE = "public.online_gate1_features"
ONLINE_GATE1_PREDICTIONS_TABLE = "public.online_gate1_predictions"
ONLINE_GATE2_PREDICTIONS_TABLE = "public.online_gate2_predictions"
ONLINE_GATE3_FEATURES_TABLE = "public.online_gate3_features"
ONLINE_GATE3_PREDICTIONS_TABLE = "public.online_gate3_predictions"

ONLINE_GATE4_FEATURES_TABLE = "public.online_gate4_features"
ONLINE_GATE4_PROCESSED_TABLE = "public.online_gate4_processed"

POLICY_CSV = ROOT / "production" / "models" / "ks" / "gate3_symbol_policy.csv.updated"

REPORT_DIR = ROOT / "online" / "_reports_gate4"
REPORT_CSV = REPORT_DIR / "online_gate4_features_report.csv"
REPORT_JSON = REPORT_DIR / "online_gate4_features_report.json"

SOURCE_NAME = "online_gate4_features_from_online_gate1_gate2_gate3_v1"
FEATURE_BUILDER = "online/gate4/build_online_gate4_features.py"

GATE1_PROBA_MIN = 0.50
G2_CLS_BASE_MIN = 0.50
G2_CLS_EXTREME_MIN = 0.90
G3_SCORE_EXTREME_MIN = 0.90
REQUIRE_GATE1_PASS = True

EXCLUDED_SYMBOLS = {
    "AGTUSDT",
    "GORKUSDT",
    "DMCUSDT",
    "MILKUSDT",
    "EPTUSDT",
    "A2ZUSDT",
    "OBTUSDT",
    "AINUSDT",
}

BASE_CONTEXT_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr14",
    "atr4h",
    "atr_ratio6",
    "atr_to_price",
    "atr_rank_48",
    "volat_ret12",
    "vol_ratio6",
    "vol_regime",
    "regime_index",
    "market_heat",
    "amihud20",
    "ret_l1",
    "ret_l2",
    "rng_pct",
    "prev_day_close",
    "prev_day_range",
    "prev_day_ret",
    "ref_close",
    "ref_btc_close",
    "ref_eth_close",
    "ret_vs_btc",
    "ret_vs_eth",
    "ret_vs_ref",
    "ret_vs_btc_z",
    "ret_vs_eth_z",
    "ret_vs_ref_z",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_monday",
    "hod_sin",
    "hod_cos",
    "sess_asia",
    "sess_eu",
    "sess_us",
    "body",
    "body_pct_rng",
    "upper_wick",
    "lower_wick",
    "wick_asymmetry",
    "body_to_prev",
    "doji_score",
    "hammer_like",
    "pinbar_like",
    "engulf_bull",
    "engulf_bear",
    "fvg_bull",
    "fvg_bear",
    "fvg_size",
    "bar_sequence_len",
    "candle_entropy",
    "sma20",
    "sma50",
    "sma100",
    "ema12",
    "ema26",
    "slope6",
    "slope12",
    "momentum6",
    "momentum12",
    "trend_strength",
    "cross_fast_slow",
    "rsi14",
    "rsi_z",
    "cci20",
    "mfi14",
    "adx14",
    "plus_di",
    "minus_di",
    "macd",
    "macd_sig",
    "macd_hist",
    "bb_width",
    "bbp",
    "range_z",
    "dist_to_high",
    "dist_to_low",
    "hl_spread_ratio",
    "hl_spread_med48",
    "gap_to_prev_close",
    "local_high_break",
    "local_low_break",
    "price_distance_ma20",
    "price_vs_vwap",
    "price_vol_corr12",
    "momentum_vol_corr",
]

FORBIDDEN_EXACT = {
    "side",
    "side_num",
    "entry_px",
    "exit_ts",
    "exit_px",
    "exit_reason",
    "pnl_net",
    "y",
    "y_fast",
    "tp_px",
    "sl_px",
    "ret",
    "dir_prev",
    "ks_tp_scale",
    "ks_sl_scale",
    "ks_ttl_hours",
    "ks_tp_abs",
    "ks_sl_abs",
    "ks_ret_adj",
    "ks_tp_abs_best",
    "ks_sl_abs_best",
    "ks_ttl_hours_best",
    "mfe_up_atr_16h",
    "mfe_dn_atr_16h",
    "first_up_hit_bar",
    "first_dn_hit_bar",
    "y_dir_mfe",
    "y_dir_first",
    "y_dir",
    "y_dir_int",
    "y_side_clean",
    "y_side_clean_int",
    "edge_atr_clean",
    "abs_edge_atr_clean",
}

FORBIDDEN_SUBSTRINGS = [
    "future",
    "target",
    "label",
    "pnl",
    "exit_",
    "tp_hit",
    "sl_hit",
    "first_tp",
    "first_sl",
    "tp_before_sl",
    "sl_before_tp",
    "mfe_",
    "mae_",
    "y_side",
    "y_dir",
    "edge_delta",
    "abs_edge_delta",
]


def connect_db():
    return psycopg2.connect(DB_DSN)


def split_table_name(table_name: str) -> Tuple[str, str]:
    parts = table_name.split(".")
    if len(parts) != 2:
        raise RuntimeError("table name must be schema.table")
    return parts[0], parts[1]


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def table_qname(table_name: str) -> str:
    schema, table = split_table_name(table_name)
    return schema + "." + qident(table)


def to_db_utc_datetime(value: Any) -> Optional[datetime]:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def table_exists(table_name: str) -> bool:
    schema, table = split_table_name(table_name)
    sql = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (schema, table))
            return bool(cur.fetchone()[0])


def get_table_columns(table_name: str) -> List[str]:
    schema, table = split_table_name(table_name)
    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """
    with connect_db() as conn:
        df = pd.read_sql_query(sql, conn, params=(schema, table))
    return [str(x) for x in df["column_name"].tolist()]


def pg_type_from_series(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s.dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return "timestamptz"
    if pd.api.types.is_integer_dtype(s.dtype):
        return "double precision"
    if pd.api.types.is_float_dtype(s.dtype):
        return "double precision"
    return "text"


def ensure_gate4_table(df: pd.DataFrame) -> None:
    ddl = f"""
        CREATE TABLE IF NOT EXISTS {ONLINE_GATE4_FEATURES_TABLE} (
            symbol text NOT NULL,
            entry_ts timestamptz NOT NULL,
            online_source text NOT NULL DEFAULT '{SOURCE_NAME}',
            online_feature_builder text NOT NULL DEFAULT '{FEATURE_BUILDER}',
            online_inserted_at timestamptz NOT NULL DEFAULT now(),
            online_updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, entry_ts)
        );

        CREATE INDEX IF NOT EXISTS idx_online_gate4_features_symbol_ts_desc
        ON public.online_gate4_features (symbol, entry_ts DESC);

        CREATE INDEX IF NOT EXISTS idx_online_gate4_features_entry_ts
        ON public.online_gate4_features (entry_ts);
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

    existing = set(get_table_columns(ONLINE_GATE4_FEATURES_TABLE))

    with connect_db() as conn:
        with conn.cursor() as cur:
            for c in df.columns:
                if c in existing:
                    continue
                if c in {"symbol", "entry_ts"}:
                    continue
                pg_type = pg_type_from_series(df[c])
                cur.execute(
                    f"ALTER TABLE {ONLINE_GATE4_FEATURES_TABLE} ADD COLUMN {qident(c)} {pg_type};"
                )
        conn.commit()


def py_value(v: Any) -> Any:
    if isinstance(v, pd.Timestamp):
        return to_db_utc_datetime(v)
    if isinstance(v, datetime):
        return v

    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, float):
        return float(v)

    return v


def upsert_gate4_features(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    df = df.copy()
    df["online_source"] = SOURCE_NAME
    df["online_feature_builder"] = FEATURE_BUILDER

    ensure_gate4_table(df)

    table_cols = get_table_columns(ONLINE_GATE4_FEATURES_TABLE)
    cols = [c for c in df.columns if c in table_cols]

    records = []
    for row in df[cols].itertuples(index=False, name=None):
        records.append(tuple(py_value(v) for v in row))

    quoted_cols = ", ".join(qident(c) for c in cols)

    update_cols = [
        c for c in cols
        if c not in {"symbol", "entry_ts", "online_inserted_at", "online_updated_at"}
    ]

    update_sql = ", ".join(
        f"{qident(c)} = EXCLUDED.{qident(c)}"
        for c in update_cols
    )

    sql = f"""
        INSERT INTO {ONLINE_GATE4_FEATURES_TABLE} ({quoted_cols})
        VALUES %s
        ON CONFLICT (symbol, entry_ts)
        DO UPDATE SET
            {update_sql},
            online_updated_at = now()
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, records, page_size=5000)
        conn.commit()

    return len(records)


def delete_existing_gate4_for_symbol(symbol: str) -> int:
    deleted_total = 0

    with connect_db() as conn:
        with conn.cursor() as cur:
            if table_exists(ONLINE_GATE4_FEATURES_TABLE):
                cur.execute(
                    f"""
                    DELETE FROM {ONLINE_GATE4_FEATURES_TABLE}
                    WHERE symbol = %s
                    """,
                    (symbol,),
                )
                deleted_total += int(cur.rowcount)

            if table_exists(ONLINE_GATE4_PROCESSED_TABLE):
                cur.execute(
                    f"""
                    DELETE FROM {ONLINE_GATE4_PROCESSED_TABLE}
                    WHERE symbol = %s
                    """,
                    (symbol,),
                )
                deleted_total += int(cur.rowcount)

        conn.commit()

    return deleted_total


def ensure_gate4_processed_table() -> None:
    ddl = f"""
        CREATE TABLE IF NOT EXISTS {ONLINE_GATE4_PROCESSED_TABLE} (
            symbol text NOT NULL,
            entry_ts timestamptz NOT NULL,

            status text NOT NULL,
            target_rows integer NOT NULL DEFAULT 0,
            candidate_rows integer NOT NULL DEFAULT 0,
            inserted_rows integer NOT NULL DEFAULT 0,

            pass_long integer NOT NULL DEFAULT 0,
            pass_short integer NOT NULL DEFAULT 0,
            pass_any integer NOT NULL DEFAULT 0,

            err text NOT NULL DEFAULT '',

            online_source text NOT NULL DEFAULT '{SOURCE_NAME}',
            online_feature_builder text NOT NULL DEFAULT '{FEATURE_BUILDER}',
            online_inserted_at timestamptz NOT NULL DEFAULT now(),
            online_updated_at timestamptz NOT NULL DEFAULT now(),

            PRIMARY KEY (symbol, entry_ts)
        );

        CREATE INDEX IF NOT EXISTS idx_online_gate4_processed_symbol_ts_desc
        ON public.online_gate4_processed (symbol, entry_ts DESC);

        CREATE INDEX IF NOT EXISTS idx_online_gate4_processed_entry_ts
        ON public.online_gate4_processed (entry_ts);

        CREATE INDEX IF NOT EXISTS idx_online_gate4_processed_status
        ON public.online_gate4_processed (status);
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def mark_gate4_processed(
    target_rows: pd.DataFrame,
    rep: Dict[str, Any],
    inserted_rows: int,
) -> int:
    if target_rows.empty:
        return 0

    ensure_gate4_processed_table()

    status = str(rep.get("status", "unknown") or "unknown")
    err = str(rep.get("err", "") or "")

    candidate_rows = int(rep.get("candidate_rows", 0) or 0)
    pass_long = int(rep.get("pass_long", 0) or 0)
    pass_short = int(rep.get("pass_short", 0) or 0)
    pass_any = int(rep.get("pass_any", 0) or 0)

    rows = []
    clean_targets = target_rows.copy()
    clean_targets["symbol"] = clean_targets["symbol"].astype(str).str.upper()
    clean_targets["entry_ts"] = pd.to_datetime(clean_targets["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    clean_targets = clean_targets.dropna(subset=["symbol", "entry_ts"])
    clean_targets = clean_targets.drop_duplicates(["symbol", "entry_ts"], keep="last")

    for row in clean_targets.itertuples(index=False):
        rows.append(
            (
                str(getattr(row, "symbol")).upper(),
                to_db_utc_datetime(getattr(row, "entry_ts")),
                status,
                1,
                candidate_rows,
                int(inserted_rows),
                pass_long,
                pass_short,
                pass_any,
                err,
                SOURCE_NAME,
                FEATURE_BUILDER,
            )
        )

    if not rows:
        return 0

    sql = f"""
        INSERT INTO {ONLINE_GATE4_PROCESSED_TABLE} (
            symbol,
            entry_ts,
            status,
            target_rows,
            candidate_rows,
            inserted_rows,
            pass_long,
            pass_short,
            pass_any,
            err,
            online_source,
            online_feature_builder
        )
        VALUES %s
        ON CONFLICT (symbol, entry_ts)
        DO UPDATE SET
            status = EXCLUDED.status,
            target_rows = EXCLUDED.target_rows,
            candidate_rows = EXCLUDED.candidate_rows,
            inserted_rows = EXCLUDED.inserted_rows,
            pass_long = EXCLUDED.pass_long,
            pass_short = EXCLUDED.pass_short,
            pass_any = EXCLUDED.pass_any,
            err = EXCLUDED.err,
            online_source = EXCLUDED.online_source,
            online_feature_builder = EXCLUDED.online_feature_builder,
            online_updated_at = now()
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=5000)
        conn.commit()

    return int(len(rows))


def build_gate4_processed_rows(
    target_rows: pd.DataFrame,
    rep: Dict[str, Any],
    inserted_rows: int,
) -> pd.DataFrame:
    if target_rows.empty:
        return pd.DataFrame()

    status = str(rep.get("status", "unknown") or "unknown")
    err = str(rep.get("err", "") or "")

    out = target_rows[["symbol", "entry_ts"]].copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    out = out.dropna(subset=["symbol", "entry_ts"])
    out = out.drop_duplicates(["symbol", "entry_ts"], keep="last").reset_index(drop=True)

    if out.empty:
        return pd.DataFrame()

    out["status"] = status
    out["target_rows"] = 1
    out["candidate_rows"] = int(rep.get("candidate_rows", 0) or 0)
    out["inserted_rows"] = int(inserted_rows)
    out["pass_long"] = int(rep.get("pass_long", 0) or 0)
    out["pass_short"] = int(rep.get("pass_short", 0) or 0)
    out["pass_any"] = int(rep.get("pass_any", 0) or 0)
    out["err"] = err
    out["online_source"] = SOURCE_NAME
    out["online_feature_builder"] = FEATURE_BUILDER

    return out


def upsert_gate4_processed_batch(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    ensure_gate4_processed_table()

    cols = [
        "symbol",
        "entry_ts",
        "status",
        "target_rows",
        "candidate_rows",
        "inserted_rows",
        "pass_long",
        "pass_short",
        "pass_any",
        "err",
        "online_source",
        "online_feature_builder",
    ]

    work = df[cols].copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["entry_ts"] = pd.to_datetime(work["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    work = work.dropna(subset=["symbol", "entry_ts"])
    work = work.drop_duplicates(["symbol", "entry_ts"], keep="last").reset_index(drop=True)

    if work.empty:
        return 0

    rows = []
    for row in work.itertuples(index=False, name=None):
        rows.append(tuple(py_value(v) for v in row))

    sql = f"""
        INSERT INTO {ONLINE_GATE4_PROCESSED_TABLE} (
            symbol,
            entry_ts,
            status,
            target_rows,
            candidate_rows,
            inserted_rows,
            pass_long,
            pass_short,
            pass_any,
            err,
            online_source,
            online_feature_builder
        )
        VALUES %s
        ON CONFLICT (symbol, entry_ts)
        DO UPDATE SET
            status = EXCLUDED.status,
            target_rows = EXCLUDED.target_rows,
            candidate_rows = EXCLUDED.candidate_rows,
            inserted_rows = EXCLUDED.inserted_rows,
            pass_long = EXCLUDED.pass_long,
            pass_short = EXCLUDED.pass_short,
            pass_any = EXCLUDED.pass_any,
            err = EXCLUDED.err,
            online_source = EXCLUDED.online_source,
            online_feature_builder = EXCLUDED.online_feature_builder,
            online_updated_at = now()
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=5000)
        conn.commit()

    return int(len(rows))


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).gt(0.5).astype(int)


def select_side_pattern_cols(cols: List[str], side: str) -> List[str]:
    side = str(side).strip().lower()
    out = []

    for c in cols:
        c_low = str(c).lower()

        is_long = ("_up" in c_low) or ("buy" in c_low) or ("_lo_" in c_low)
        is_short = ("_dn" in c_low) or ("sell" in c_low) or ("_hi_" in c_low)

        if side == "long" and is_long and not is_short:
            out.append(c)
        if side == "short" and is_short and not is_long:
            out.append(c)

    return sorted(set(out))


def add_active_set_features(df: pd.DataFrame, active_cols: List[str], prefix: str) -> pd.DataFrame:
    block = pd.DataFrame(index=df.index)

    if prefix == "g3_long":
        primary_col = "active_pa_atr_squeeze_break_up"
        secondary_col = "active_pa_bos_up_24"
    elif prefix == "g3_short":
        primary_col = "active_pa_atr_squeeze_break_dn"
        secondary_col = "active_pa_bos_dn_24"
    else:
        primary_col = ""
        secondary_col = ""

    if not active_cols:
        block[f"{prefix}_any_active"] = 0
        block[f"{prefix}_active_count"] = 0
        block[f"{prefix}_active_primary"] = 0
        block[f"{prefix}_active_secondary"] = 0
        block[f"{prefix}_active_overlap_primary_secondary"] = 0
        block[f"{prefix}_max_active_age"] = 0
        return pd.concat([df, block], axis=1).copy()

    act = df[active_cols].copy()
    for c in active_cols:
        act[c] = safe_bool_series(act[c])

    act_sum = act.sum(axis=1)

    block[f"{prefix}_any_active"] = (act_sum > 0).astype(int)
    block[f"{prefix}_active_count"] = act_sum.astype(int)

    if primary_col and primary_col in act.columns:
        block[f"{prefix}_active_primary"] = act[primary_col].astype(int)
    else:
        block[f"{prefix}_active_primary"] = 0

    if secondary_col and secondary_col in act.columns:
        block[f"{prefix}_active_secondary"] = act[secondary_col].astype(int)
    else:
        block[f"{prefix}_active_secondary"] = 0

    block[f"{prefix}_active_overlap_primary_secondary"] = (
        (block[f"{prefix}_active_primary"] == 1) &
        (block[f"{prefix}_active_secondary"] == 1)
    ).astype(int)

    max_age = np.zeros(len(df), dtype=int)

    for c in active_cols:
        x = act[c].to_numpy(dtype=int)
        age = np.zeros(len(x), dtype=int)
        run = 0

        for i in range(len(x)):
            if x[i] == 1:
                run += 1
            else:
                run = 0
            age[i] = run

        max_age = np.maximum(max_age, age)

    block[f"{prefix}_max_active_age"] = max_age

    return pd.concat([df, block], axis=1).copy()


def is_forbidden_col(col: str) -> bool:
    c = str(col)
    c_low = c.lower()

    if c in FORBIDDEN_EXACT:
        return True

    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in c_low:
            return True

    return False


def build_base_feature_cols(df: pd.DataFrame) -> List[str]:
    out = []

    for c in BASE_CONTEXT_COLS:
        if c not in df.columns:
            continue
        if is_forbidden_col(c):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        out.append(c)

    extra_feat_cols = []
    for c in df.columns:
        if is_forbidden_col(c):
            continue
        if not str(c).endswith("_feat"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        extra_feat_cols.append(c)

    out.extend(sorted(set(extra_feat_cols)))
    return sorted(set(out))


def get_symbols_from_gate1_features() -> List[str]:
    sql = f"""
        SELECT DISTINCT symbol
        FROM {ONLINE_GATE1_FEATURES_TABLE}
        ORDER BY symbol
    """
    with connect_db() as conn:
        df = pd.read_sql_query(sql, conn)

    if df.empty:
        return []

    symbols = [str(x).upper() for x in df["symbol"].tolist()]
    symbols = [s for s in symbols if s not in EXCLUDED_SYMBOLS]
    return sorted(set(symbols))


def get_target_rows(symbol: str, rebuild: bool, limit_latest: Optional[int]) -> pd.DataFrame:
    if rebuild or not table_exists(ONLINE_GATE4_PROCESSED_TABLE):
        sql = f"""
            SELECT symbol, entry_ts
            FROM {ONLINE_GATE1_FEATURES_TABLE}
            WHERE symbol = %s
            ORDER BY entry_ts ASC
        """
        params = [symbol]
    else:
        sql = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE1_FEATURES_TABLE} f
            LEFT JOIN {ONLINE_GATE4_PROCESSED_TABLE} p
              ON p.symbol = f.symbol
             AND p.entry_ts = f.entry_ts
            WHERE f.symbol = %s
              AND p.entry_ts IS NULL
            ORDER BY f.entry_ts ASC
        """
        params = [symbol]

    with connect_db() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    if limit_latest is not None and int(limit_latest) > 0 and len(df) > int(limit_latest):
        df = df.tail(int(limit_latest)).reset_index(drop=True)

    return df


def normalize_symbol_entry_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    out = (
        out.dropna(subset=["symbol", "entry_ts"])
        .sort_values(["symbol", "entry_ts"])
        .drop_duplicates(["symbol", "entry_ts"], keep="last")
        .reset_index(drop=True)
    )
    return out


def split_by_symbol(df: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    out = {str(s).upper(): pd.DataFrame() for s in symbols}

    if df.empty:
        return out

    for symbol, part in df.groupby("symbol", sort=False):
        out[str(symbol).upper()] = part.reset_index(drop=True)

    return out


def get_target_rows_batch(symbols: List[str], rebuild: bool, limit_latest: Optional[int]) -> Dict[str, pd.DataFrame]:
    symbols = sorted(set(str(s).upper() for s in symbols))
    empty = {s: pd.DataFrame(columns=["symbol", "entry_ts"]) for s in symbols}

    if not symbols:
        return empty

    if rebuild or not table_exists(ONLINE_GATE4_PROCESSED_TABLE):
        sql = f"""
            SELECT symbol, entry_ts
            FROM {ONLINE_GATE1_FEATURES_TABLE}
            WHERE symbol = ANY(%s)
            ORDER BY symbol ASC, entry_ts ASC
        """
    else:
        sql = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE1_FEATURES_TABLE} f
            LEFT JOIN {ONLINE_GATE4_PROCESSED_TABLE} p
              ON p.symbol = f.symbol
             AND p.entry_ts = f.entry_ts
            WHERE f.symbol = ANY(%s)
              AND p.entry_ts IS NULL
            ORDER BY f.symbol ASC, f.entry_ts ASC
        """

    with connect_db() as conn:
        df = pd.read_sql_query(sql, conn, params=[symbols])

    if df.empty:
        return empty

    df = normalize_symbol_entry_df(df)

    if limit_latest is not None and int(limit_latest) > 0:
        df = (
            df.sort_values(["symbol", "entry_ts"])
            .groupby("symbol", group_keys=False)
            .tail(int(limit_latest))
            .reset_index(drop=True)
        )

    return split_by_symbol(df, symbols)


def load_table_context_batch(
    table_name: str,
    symbols: List[str],
    target_rows_all: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    symbols = sorted(set(str(s).upper() for s in symbols))
    empty = {s: pd.DataFrame() for s in symbols}

    if not symbols or target_rows_all is None or target_rows_all.empty:
        return empty

    keys = target_rows_all[["symbol", "entry_ts"]].copy()
    keys = normalize_symbol_entry_df(keys)

    if keys.empty:
        return empty

    key_symbols = [str(x).upper() for x in keys["symbol"].tolist()]
    key_entry_ts = [to_db_utc_datetime(x) for x in keys["entry_ts"].tolist()]

    sql = f"""
        WITH target_keys AS (
            SELECT
                *
            FROM UNNEST(%s::text[], %s::timestamptz[]) AS t(symbol, entry_ts)
        )
        SELECT src.*
        FROM {table_name} src
        INNER JOIN target_keys t
            ON t.symbol = src.symbol
           AND t.entry_ts = src.entry_ts
        ORDER BY src.symbol ASC, src.entry_ts ASC
    """

    with connect_db() as conn:
        df = pd.read_sql_query(sql, conn, params=[key_symbols, key_entry_ts])

    if df.empty:
        return empty

    df = normalize_symbol_entry_df(df)
    return split_by_symbol(df, symbols)


def get_df_for_symbol(cache: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
    return cache.get(str(symbol).upper(), pd.DataFrame())



def load_table_context(table_name: str, symbol: str, max_ts: pd.Timestamp) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM {table_name}
        WHERE symbol = %s
          AND entry_ts <= %s
        ORDER BY entry_ts ASC
    """
    with connect_db() as conn:
        df = pd.read_sql_query(sql, conn, params=[symbol, to_db_utc_datetime(max_ts)])

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = (
        df.dropna(subset=["entry_ts"])
        .sort_values("entry_ts")
        .drop_duplicates(["symbol", "entry_ts"], keep="last")
        .reset_index(drop=True)
    )
    return df


def load_policy() -> pd.DataFrame:
    if not POLICY_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(POLICY_CSV)
    if "symbol" not in df.columns:
        return pd.DataFrame()

    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def policy_row_for_symbol(policy: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    if policy.empty:
        return {}
    part = policy[policy["symbol"] == symbol]
    if part.empty:
        return {}
    return part.tail(1).iloc[0].to_dict()


def numeric_from_obj(obj: Any, default: float) -> float:
    val = pd.to_numeric(obj, errors="coerce")
    if pd.isna(val) or not np.isfinite(float(val)):
        return float(default)
    return float(val)


def merge_optional(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keep_cols: List[str],
    rename_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    if right.empty:
        return left

    cols = ["symbol", "entry_ts"]
    for c in keep_cols:
        if c in right.columns and c not in cols:
            cols.append(c)

    if len(cols) <= 2:
        return left

    part = right[cols].copy()
    if rename_map:
        part = part.rename(columns=rename_map)

    return left.merge(part, on=["symbol", "entry_ts"], how="left")


def normalize_bool_col(df: pd.DataFrame, col: str, default: int) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=int)
    return safe_bool_series(df[col])


def normalize_num_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def build_gate4_for_symbol(
    symbol: str,
    target_rows: pd.DataFrame,
    policy_row: Dict[str, Any],
    g1f_context: Optional[pd.DataFrame] = None,
    g1p_context: Optional[pd.DataFrame] = None,
    g2p_context: Optional[pd.DataFrame] = None,
    g3f_context: Optional[pd.DataFrame] = None,
    g3p_context: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report = {
        "symbol": symbol,
        "status": "init",
        "target_rows": int(len(target_rows)),
        "context_rows": 0,
        "base_cols": 0,
        "candidate_rows": 0,
        "inserted": 0,
        "pass_long": 0,
        "pass_short": 0,
        "pass_any": 0,
        "err": "",
    }

    if target_rows.empty:
        report["status"] = "no_missing"
        return pd.DataFrame(), report

    max_ts = pd.Timestamp(target_rows["entry_ts"].max())
    target_key_count = int(
        target_rows[["symbol", "entry_ts"]]
        .drop_duplicates(["symbol", "entry_ts"])
        .shape[0]
    )

    if g1f_context is None:
        g1f = load_table_context(ONLINE_GATE1_FEATURES_TABLE, symbol, max_ts)
    else:
        g1f = g1f_context.copy()

    if g1p_context is None:
        g1p = load_table_context(ONLINE_GATE1_PREDICTIONS_TABLE, symbol, max_ts)
    else:
        g1p = g1p_context.copy()

    if g2p_context is None:
        g2p = load_table_context(ONLINE_GATE2_PREDICTIONS_TABLE, symbol, max_ts)
    else:
        g2p = g2p_context.copy()

    if g3f_context is None:
        g3f = load_table_context(ONLINE_GATE3_FEATURES_TABLE, symbol, max_ts)
    else:
        g3f = g3f_context.copy()

    if g3p_context is None:
        g3p = load_table_context(ONLINE_GATE3_PREDICTIONS_TABLE, symbol, max_ts)
    else:
        g3p = g3p_context.copy()

    upstream_counts = {
        "g1f_rows": int(len(g1f)),
        "g1p_rows": int(len(g1p)),
        "g2p_rows": int(len(g2p)),
        "g3f_rows": int(len(g3f)),
        "g3p_rows": int(len(g3p)),
    }

    report.update(upstream_counts)

    missing_upstream = [
        name
        for name, count in upstream_counts.items()
        if count < target_key_count
    ]

    if missing_upstream:
        report["status"] = "waiting_upstream"
        report["err"] = (
            "missing upstream rows for target_key_count="
            + str(target_key_count)
            + "; "
            + ", ".join(name + "=" + str(upstream_counts[name]) for name in missing_upstream)
        )
        return pd.DataFrame(), report

    report["context_rows"] = int(len(g1f))

    base_cols = build_base_feature_cols(g1f)
    report["base_cols"] = int(len(base_cols))

    if not base_cols:
        report["status"] = "empty_base_cols"
        return pd.DataFrame(), report

    work = g1f[["symbol", "entry_ts"] + base_cols].copy()

    work = merge_optional(
        work,
        g1p,
        keep_cols=["gate1_proba", "gate1_pass"],
    )

    work = merge_optional(
        work,
        g2p,
        keep_cols=[
            "up_reach_high_proba",
            "dn_reach_high_proba",
            "gate2_best_proba",
            "gate2_any_pass",
            "gate2_side",
            "gate2_best_side",
            "gate2_margin_abs",
            "gate2_margin_ratio",
        ],
        rename_map={
            "up_reach_high_proba": "g2_cls_up_reach_high_proba",
            "dn_reach_high_proba": "g2_cls_dn_reach_high_proba",
        },
    )

    if not g3f.empty:
        g3_feature_keep = []
        for c in g3f.columns:
            if c in {"symbol", "entry_ts"}:
                continue
            if str(c).startswith("active_pa_"):
                g3_feature_keep.append(c)
            elif c in {
                "gate3_active_count",
                "gate3_active_primary",
                "gate3_active_secondary",
                "gate3_active_overlap_primary_secondary",
                "gate3_max_active_age",
                "gate3_side_bias",
                "gate3_score_long",
                "gate3_score_short",
                "gate3_rank_long",
                "gate3_rank_short",
                "pa_quality",
                "pa_quality_sq",
                "active_any",
                "active_density",
                "gate3_any_active",
            }:
                g3_feature_keep.append(c)

        work = merge_optional(work, g3f, keep_cols=sorted(set(g3_feature_keep)))

    work = merge_optional(
        work,
        g3p,
        keep_cols=[
            "g3_long_score_proba",
            "g3_short_score_proba",
            "g3_long_score_pass",
            "g3_short_score_pass",
            "gate3_proba_long",
            "gate3_proba_short",
            "gate3_pass_long",
            "gate3_pass_short",
            "gate3_any_pass",
            "gate3_best_side",
            "gate3_best_proba",
            "gate3_margin_long",
            "gate3_margin_short",
            "gate3_threshold_long",
            "gate3_threshold_short",
            "g3_score_spread",
            "g3_score_abs_spread",
            "g3_score_max",
            "gate3_precision_meta_long",
            "gate3_wilson_meta_long",
            "gate3_delta_wilson_meta_long",
            "gate3_pvalue_meta_long",
            "gate3_kept_n_meta_long",
            "gate3_valid_pos_rate_meta_long",
            "gate3_thr_kept_lift_meta_long",
            "gate3_precision_meta_short",
            "gate3_wilson_meta_short",
            "gate3_delta_wilson_meta_short",
            "gate3_pvalue_meta_short",
            "gate3_kept_n_meta_short",
            "gate3_valid_pos_rate_meta_short",
            "gate3_thr_kept_lift_meta_short",
            "gate3_precision_meta",
            "gate3_wilson_meta",
            "gate3_delta_wilson_meta",
            "gate3_pvalue_meta",
            "gate3_kept_n_meta",
            "gate3_valid_pos_rate_meta",
            "gate3_thr_kept_lift_meta",
            "has_gate3_long_bundle",
            "has_gate3_short_bundle",
            "has_any_gate3_bundle",
            "has_full_gate3_bundle",
            "long_feature_count",
            "short_feature_count",
        ],
    )

    target_ts = set(pd.Timestamp(x) for x in target_rows["entry_ts"].tolist())
    work = work[work["entry_ts"].isin(target_ts)].copy()

    if work.empty:
        report["status"] = "target_rows_not_found_after_merge"
        return pd.DataFrame(), report

    for c in work.columns:
        if c in {"symbol", "entry_ts", "gate2_side", "gate2_best_side", "gate3_best_side"}:
            continue
        if pd.api.types.is_numeric_dtype(work[c]) or pd.api.types.is_bool_dtype(work[c]):
            continue
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work["gate1_pass"] = (pd.to_numeric(work["gate1_proba"], errors="coerce") >= GATE1_PROBA_MIN).astype(int)

    work["g2_cls_up_reach_high_proba"] = normalize_num_col(work, "g2_cls_up_reach_high_proba", np.nan)
    work["g2_cls_dn_reach_high_proba"] = normalize_num_col(work, "g2_cls_dn_reach_high_proba", np.nan)

    work["g2_cls_spread"] = work["g2_cls_up_reach_high_proba"] - work["g2_cls_dn_reach_high_proba"]
    work["g2_cls_abs_spread"] = work["g2_cls_spread"].abs()
    work["g2_cls_max"] = np.maximum(
        pd.to_numeric(work["g2_cls_up_reach_high_proba"], errors="coerce"),
        pd.to_numeric(work["g2_cls_dn_reach_high_proba"], errors="coerce"),
    )
    work["g2_up_dominant"] = work["g2_cls_spread"].gt(0).astype(int)
    work["g2_dn_dominant"] = work["g2_cls_spread"].lt(0).astype(int)
    work["gate2_proba"] = work["g2_cls_max"]

    long_pattern = str(policy_row.get("gate3_pattern_long", "") or "")
    short_pattern = str(policy_row.get("gate3_pattern_short", "") or "")

    all_active_cols = sorted([c for c in work.columns if str(c).startswith("active_pa_")])
    long_active_cols = select_side_pattern_cols(all_active_cols, "long")
    short_active_cols = select_side_pattern_cols(all_active_cols, "short")

    work = add_active_set_features(work, long_active_cols, prefix="g3_long")
    work = add_active_set_features(work, short_active_cols, prefix="g3_short")

    if long_pattern and long_pattern in work.columns:
        g3_long_active = safe_bool_series(work[long_pattern])
    else:
        g3_long_active = pd.Series(0, index=work.index, dtype=int)

    if short_pattern and short_pattern in work.columns:
        g3_short_active = safe_bool_series(work[short_pattern])
    else:
        g3_short_active = pd.Series(0, index=work.index, dtype=int)

    work["g3_long_active"] = g3_long_active.astype(int)
    work["g3_short_active"] = g3_short_active.astype(int)
    work["g3_any_active"] = ((work["g3_long_active"] == 1) | (work["g3_short_active"] == 1)).astype(int)
    work["g3_both_active"] = ((work["g3_long_active"] == 1) & (work["g3_short_active"] == 1)).astype(int)

    work["gate3_active_count"] = (
        pd.to_numeric(work["g3_long_active_count"], errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(work["g3_short_active_count"], errors="coerce").fillna(0).astype(int)
    )

    work["gate3_active_primary"] = (
        (pd.to_numeric(work["g3_long_active_primary"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(work["g3_short_active_primary"], errors="coerce").fillna(0) > 0)
    ).astype(int)

    work["gate3_active_secondary"] = (
        (pd.to_numeric(work["g3_long_active_secondary"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(work["g3_short_active_secondary"], errors="coerce").fillna(0) > 0)
    ).astype(int)

    work["gate3_active_overlap_primary_secondary"] = (
        (pd.to_numeric(work["g3_long_active_overlap_primary_secondary"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(work["g3_short_active_overlap_primary_secondary"], errors="coerce").fillna(0) > 0)
    ).astype(int)

    work["gate3_max_active_age"] = np.maximum(
        pd.to_numeric(work["g3_long_max_active_age"], errors="coerce").fillna(0),
        pd.to_numeric(work["g3_short_max_active_age"], errors="coerce").fillna(0),
    )

    for c in [
        "has_gate3_long_bundle",
        "has_gate3_short_bundle",
        "has_any_gate3_bundle",
        "has_full_gate3_bundle",
        "g3_long_score_pass",
        "g3_short_score_pass",
        "gate3_pass_long",
        "gate3_pass_short",
    ]:
        work[c] = normalize_bool_col(work, c, 0)

    for c in [
        "g3_long_score_proba",
        "g3_short_score_proba",
        "gate3_proba_long",
        "gate3_proba_short",
        "gate3_threshold_long",
        "gate3_threshold_short",
        "gate3_margin_long",
        "gate3_margin_short",
        "g3_score_spread",
        "g3_score_abs_spread",
        "g3_score_max",
    ]:
        if c not in work.columns:
            work[c] = np.nan

    if "gate3_score_long" not in work.columns:
        work["gate3_score_long"] = np.nan
    if "gate3_score_short" not in work.columns:
        work["gate3_score_short"] = np.nan
    if "gate3_rank_long" not in work.columns:
        work["gate3_rank_long"] = np.nan
    if "gate3_rank_short" not in work.columns:
        work["gate3_rank_short"] = np.nan
    if "gate3_side_bias" not in work.columns:
        work["gate3_side_bias"] = np.nan

    work["g2_side_sign"] = np.where(
        work["g2_cls_spread"] > 0,
        1,
        np.where(work["g2_cls_spread"] < 0, -1, 0),
    )

    work["g3_side_sign"] = np.where(
        pd.to_numeric(work["g3_score_spread"], errors="coerce") > 0,
        1,
        np.where(pd.to_numeric(work["g3_score_spread"], errors="coerce") < 0, -1, 0),
    )

    work["g2_g3_side_agree"] = (
        (work["has_any_gate3_bundle"] == 1)
        & (work["g2_side_sign"] != 0)
        & (work["g3_side_sign"] != 0)
        & (work["g2_side_sign"] == work["g3_side_sign"])
    ).astype(int)

    work["g2_g3_side_conflict"] = (
        (work["has_any_gate3_bundle"] == 1)
        & (work["g2_side_sign"] != 0)
        & (work["g3_side_sign"] != 0)
        & (work["g2_side_sign"] != work["g3_side_sign"])
    ).astype(int)

    work["g1_g2_strength"] = pd.to_numeric(work["gate1_proba"], errors="coerce") * pd.to_numeric(work["g2_cls_max"], errors="coerce")
    work["g1_g3_strength"] = np.where(
        work["has_any_gate3_bundle"] == 1,
        pd.to_numeric(work["gate1_proba"], errors="coerce") * pd.to_numeric(work["g3_score_max"], errors="coerce"),
        np.nan,
    )

    work["g2g3_joint_long"] = np.where(
        work["has_gate3_long_bundle"] == 1,
        pd.to_numeric(work["g2_cls_up_reach_high_proba"], errors="coerce")
        * pd.to_numeric(work["g3_long_score_proba"], errors="coerce").fillna(0.0),
        np.nan,
    )

    work["g2g3_joint_short"] = np.where(
        work["has_gate3_short_bundle"] == 1,
        pd.to_numeric(work["g2_cls_dn_reach_high_proba"], errors="coerce")
        * pd.to_numeric(work["g3_short_score_proba"], errors="coerce").fillna(0.0),
        np.nan,
    )

    work["g2g3_joint_long_minus_short"] = work["g2g3_joint_long"] - work["g2g3_joint_short"]
    work["g2g3_joint_abs_spread"] = work["g2g3_joint_long_minus_short"].abs()

    gate1_ok = work["gate1_pass"].eq(1) if REQUIRE_GATE1_PASS else pd.Series(True, index=work.index)

    long_g3_score = pd.to_numeric(work["g3_long_score_proba"], errors="coerce").fillna(0.0)
    short_g3_score = pd.to_numeric(work["g3_short_score_proba"], errors="coerce").fillna(0.0)

    long_g3_extreme = (
        (work["g3_long_active"] == 1)
        & (long_g3_score >= G3_SCORE_EXTREME_MIN)
    )

    short_g3_extreme = (
        (work["g3_short_active"] == 1)
        & (short_g3_score >= G3_SCORE_EXTREME_MIN)
    )

    work["base_long_candidate"] = (
        gate1_ok
        & (
            (pd.to_numeric(work["g2_cls_up_reach_high_proba"], errors="coerce") >= G2_CLS_BASE_MIN)
            | (
                (work["has_gate3_long_bundle"] == 1)
                & (
                    (work["g3_long_active"] == 1)
                    | (work["g3_long_score_pass"] == 1)
                )
            )
        )
    ).astype(int)

    work["base_short_candidate"] = (
        gate1_ok
        & (
            (pd.to_numeric(work["g2_cls_dn_reach_high_proba"], errors="coerce") >= G2_CLS_BASE_MIN)
            | (
                (work["has_gate3_short_bundle"] == 1)
                & (
                    (work["g3_short_active"] == 1)
                    | (work["g3_short_score_pass"] == 1)
                )
            )
        )
    ).astype(int)

    work["extreme_long_candidate"] = (
        gate1_ok
        & (
            (pd.to_numeric(work["g2_cls_up_reach_high_proba"], errors="coerce") >= G2_CLS_EXTREME_MIN)
            | (
                (work["has_gate3_long_bundle"] == 1)
                & long_g3_extreme
            )
        )
    ).astype(int)

    work["extreme_short_candidate"] = (
        gate1_ok
        & (
            (pd.to_numeric(work["g2_cls_dn_reach_high_proba"], errors="coerce") >= G2_CLS_EXTREME_MIN)
            | (
                (work["has_gate3_short_bundle"] == 1)
                & short_g3_extreme
            )
        )
    ).astype(int)

    work["pass_long"] = ((work["base_long_candidate"] == 1) | (work["extreme_long_candidate"] == 1)).astype(int)
    work["pass_short"] = ((work["base_short_candidate"] == 1) | (work["extreme_short_candidate"] == 1)).astype(int)
    work["pass_any"] = ((work["pass_long"] == 1) | (work["pass_short"] == 1)).astype(int)
    work["pass_both"] = ((work["pass_long"] == 1) & (work["pass_short"] == 1)).astype(int)
    work["pass_long_only"] = ((work["pass_long"] == 1) & (work["pass_short"] == 0)).astype(int)
    work["pass_short_only"] = ((work["pass_short"] == 1) & (work["pass_long"] == 0)).astype(int)

    keep_cols = [
        "symbol",
        "entry_ts",
        *base_cols,
        "gate1_proba",
        "gate1_pass",
        "g2_cls_up_reach_high_proba",
        "g2_cls_dn_reach_high_proba",
        "g2_cls_spread",
        "g2_cls_abs_spread",
        "g2_cls_max",
        "g2_up_dominant",
        "g2_dn_dominant",
        "gate2_proba",
        "g3_long_active",
        "g3_short_active",
        "g3_any_active",
        "g3_both_active",
        "g3_long_any_active",
        "g3_long_active_count",
        "g3_long_active_primary",
        "g3_long_active_secondary",
        "g3_long_active_overlap_primary_secondary",
        "g3_long_max_active_age",
        "g3_short_any_active",
        "g3_short_active_count",
        "g3_short_active_primary",
        "g3_short_active_secondary",
        "g3_short_active_overlap_primary_secondary",
        "g3_short_max_active_age",
        "g3_long_score_proba",
        "g3_short_score_proba",
        "g3_long_score_pass",
        "g3_short_score_pass",
        "g3_score_spread",
        "g3_score_abs_spread",
        "g3_score_max",
        "gate3_pass_long",
        "gate3_pass_short",
        "gate3_proba_long",
        "gate3_proba_short",
        "gate3_margin_long",
        "gate3_margin_short",
        "gate3_threshold_long",
        "gate3_threshold_short",
        "gate3_precision_meta_long",
        "gate3_wilson_meta_long",
        "gate3_delta_wilson_meta_long",
        "gate3_pvalue_meta_long",
        "gate3_kept_n_meta_long",
        "gate3_valid_pos_rate_meta_long",
        "gate3_thr_kept_lift_meta_long",
        "gate3_precision_meta_short",
        "gate3_wilson_meta_short",
        "gate3_delta_wilson_meta_short",
        "gate3_pvalue_meta_short",
        "gate3_kept_n_meta_short",
        "gate3_valid_pos_rate_meta_short",
        "gate3_thr_kept_lift_meta_short",
        "gate3_precision_meta",
        "gate3_wilson_meta",
        "gate3_delta_wilson_meta",
        "gate3_pvalue_meta",
        "gate3_kept_n_meta",
        "gate3_valid_pos_rate_meta",
        "gate3_thr_kept_lift_meta",
        "gate3_active_count",
        "gate3_active_primary",
        "gate3_active_secondary",
        "gate3_active_overlap_primary_secondary",
        "gate3_max_active_age",
        "gate3_side_bias",
        "gate3_score_long",
        "gate3_score_short",
        "gate3_rank_long",
        "gate3_rank_short",
        "g2_g3_side_agree",
        "g2_g3_side_conflict",
        "g1_g2_strength",
        "g1_g3_strength",
        "g2g3_joint_long",
        "g2g3_joint_short",
        "g2g3_joint_long_minus_short",
        "g2g3_joint_abs_spread",
        "has_gate3_long_bundle",
        "has_gate3_short_bundle",
        "has_any_gate3_bundle",
        "has_full_gate3_bundle",
        "base_long_candidate",
        "base_short_candidate",
        "extreme_long_candidate",
        "extreme_short_candidate",
        "pass_long",
        "pass_short",
        "pass_any",
        "pass_both",
        "pass_long_only",
        "pass_short_only",
    ]

    keep_cols = [c for c in keep_cols if c in work.columns and not is_forbidden_col(c)]
    work = work.loc[:, keep_cols].copy()
    work = work.loc[:, ~work.columns.duplicated(keep="first")].copy()

    candidates = work[work["pass_any"] == 1].copy()
    candidates = candidates.sort_values(["entry_ts", "symbol"]).reset_index(drop=True)

    report["candidate_rows"] = int(len(candidates))
    report["pass_long"] = int(candidates["pass_long"].sum()) if len(candidates) else 0
    report["pass_short"] = int(candidates["pass_short"].sum()) if len(candidates) else 0
    report["pass_any"] = int(candidates["pass_any"].sum()) if len(candidates) else 0
    report["status"] = "ok" if len(candidates) else "no_candidates"

    return candidates, report


def parse_args() -> Tuple[Optional[str], bool, Optional[int]]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default="")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit-latest", type=int, default=0)
    args = ap.parse_args()

    symbol = str(args.symbol).strip().upper() if args.symbol else None
    rebuild = bool(args.rebuild)
    limit_latest = int(args.limit_latest) if int(args.limit_latest or 0) > 0 else None

    return symbol, rebuild, limit_latest


def main() -> None:
    symbol_arg, rebuild, limit_latest = parse_args()

    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("ONLINE_GATE4_FEATURES_TABLE:", ONLINE_GATE4_FEATURES_TABLE)
    print("ONLINE_GATE4_PROCESSED_TABLE:", ONLINE_GATE4_PROCESSED_TABLE)
    print("REBUILD:", rebuild)
    print("LIMIT_LATEST:", limit_latest)
    print()

    required_tables = [
        ONLINE_GATE1_FEATURES_TABLE,
        ONLINE_GATE1_PREDICTIONS_TABLE,
        ONLINE_GATE2_PREDICTIONS_TABLE,
        ONLINE_GATE3_FEATURES_TABLE,
        ONLINE_GATE3_PREDICTIONS_TABLE,
    ]

    for table_name in required_tables:
        ok = table_exists(table_name)
        print("TABLE_EXISTS:", table_name, ok)
        if not ok:
            raise RuntimeError(f"missing table: {table_name}")

    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_gate4_processed_table()

    policy = load_policy()

    if symbol_arg:
        symbols = [symbol_arg]
    else:
        symbols = get_symbols_from_gate1_features()

    print("SYMBOLS:", len(symbols))
    print("DB_BATCH_LOAD: target gate4 rows + gate1/gate2/gate3 contexts")
    print()

    target_rows_by_symbol = get_target_rows_batch(
        symbols=symbols,
        rebuild=rebuild,
        limit_latest=limit_latest,
    )

    all_target_rows = [
        df for df in target_rows_by_symbol.values()
        if df is not None and not df.empty
    ]

    if all_target_rows:
        target_all = pd.concat(all_target_rows, ignore_index=True)
        max_target_ts = pd.Timestamp(target_all["entry_ts"].max())
        target_rows_total = int(len(target_all))
        target_symbols_total = int(target_all["symbol"].nunique())
    else:
        target_all = pd.DataFrame(columns=["symbol", "entry_ts"])
        max_target_ts = None
        target_rows_total = 0
        target_symbols_total = 0

    print("TARGET_GATE4_ROWS_TOTAL:", target_rows_total)
    print("TARGET_GATE4_SYMBOLS:", target_symbols_total)

    if target_rows_total > 0:
        g1f_by_symbol = load_table_context_batch(ONLINE_GATE1_FEATURES_TABLE, symbols, target_all)
        g1p_by_symbol = load_table_context_batch(ONLINE_GATE1_PREDICTIONS_TABLE, symbols, target_all)
        g2p_by_symbol = load_table_context_batch(ONLINE_GATE2_PREDICTIONS_TABLE, symbols, target_all)
        g3f_by_symbol = load_table_context_batch(ONLINE_GATE3_FEATURES_TABLE, symbols, target_all)
        g3p_by_symbol = load_table_context_batch(ONLINE_GATE3_PREDICTIONS_TABLE, symbols, target_all)

        print("GATE1_FEATURE_CONTEXT_ROWS_BATCH:", int(sum(len(df) for df in g1f_by_symbol.values())))
        print("GATE1_PRED_CONTEXT_ROWS_BATCH:", int(sum(len(df) for df in g1p_by_symbol.values())))
        print("GATE2_PRED_CONTEXT_ROWS_BATCH:", int(sum(len(df) for df in g2p_by_symbol.values())))
        print("GATE3_FEATURE_CONTEXT_ROWS_BATCH:", int(sum(len(df) for df in g3f_by_symbol.values())))
        print("GATE3_PRED_CONTEXT_ROWS_BATCH:", int(sum(len(df) for df in g3p_by_symbol.values())))
    else:
        g1f_by_symbol = {s: pd.DataFrame() for s in symbols}
        g1p_by_symbol = {s: pd.DataFrame() for s in symbols}
        g2p_by_symbol = {s: pd.DataFrame() for s in symbols}
        g3f_by_symbol = {s: pd.DataFrame() for s in symbols}
        g3p_by_symbol = {s: pd.DataFrame() for s in symbols}

        print("GATE1_FEATURE_CONTEXT_ROWS_BATCH:", 0)
        print("GATE1_PRED_CONTEXT_ROWS_BATCH:", 0)
        print("GATE2_PRED_CONTEXT_ROWS_BATCH:", 0)
        print("GATE3_FEATURE_CONTEXT_ROWS_BATCH:", 0)
        print("GATE3_PRED_CONTEXT_ROWS_BATCH:", 0)

    print()

    reports = []
    pending_gate4_feature_frames = []
    pending_gate4_processed_frames = []
    total_candidates = 0
    total_inserted = 0
    total_processed = 0

    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{len(symbols)}] {symbol}")

        try:
            if rebuild:
                deleted = delete_existing_gate4_for_symbol(symbol)
            else:
                deleted = 0

            target_rows = get_df_for_symbol(target_rows_by_symbol, symbol)

            policy_row = policy_row_for_symbol(policy, symbol)

            out_df, rep = build_gate4_for_symbol(
                symbol=symbol,
                target_rows=target_rows,
                policy_row=policy_row,
                g1f_context=get_df_for_symbol(g1f_by_symbol, symbol),
                g1p_context=get_df_for_symbol(g1p_by_symbol, symbol),
                g2p_context=get_df_for_symbol(g2p_by_symbol, symbol),
                g3f_context=get_df_for_symbol(g3f_by_symbol, symbol),
                g3p_context=get_df_for_symbol(g3p_by_symbol, symbol),
            )

            if not out_df.empty:
                pending_gate4_feature_frames.append(out_df)

            inserted = int(len(out_df)) if not out_df.empty else 0

            processed_rows = 0
            if rep.get("status") in {"ok", "no_candidates"}:
                processed_df = build_gate4_processed_rows(
                    target_rows=target_rows,
                    rep=rep,
                    inserted_rows=inserted,
                )

                if not processed_df.empty:
                    pending_gate4_processed_frames.append(processed_df)
                    processed_rows = int(len(processed_df))

            rep["deleted_before_rebuild"] = int(deleted)
            rep["inserted"] = int(inserted)
            rep["processed_rows"] = int(processed_rows)

            reports.append(rep)

            total_candidates += int(rep.get("candidate_rows", 0))
            total_inserted += int(inserted)

            print(
                f"    status={rep['status']} | "
                f"target={rep.get('target_rows', 0)} | "
                f"context={rep.get('context_rows', 0)} | "
                f"base_cols={rep.get('base_cols', 0)} | "
                f"candidates={rep.get('candidate_rows', 0)} | "
                f"inserted={inserted} | "
                f"processed={processed_rows}"
            )

            if rep.get("err"):
                print(f"    err={rep['err']}")

        except Exception as e:
            rep = {
                "symbol": symbol,
                "status": "error",
                "target_rows": 0,
                "context_rows": 0,
                "base_cols": 0,
                "candidate_rows": 0,
                "inserted": 0,
                "pass_long": 0,
                "pass_short": 0,
                "pass_any": 0,
                "err": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
            reports.append(rep)
            print(f"    ERROR: {rep['err']}")

    if pending_gate4_feature_frames:
        batch_features_df = pd.concat(pending_gate4_feature_frames, ignore_index=True)
        batch_features_df = (
            batch_features_df
            .sort_values(["symbol", "entry_ts"])
            .drop_duplicates(["symbol", "entry_ts"], keep="last")
            .reset_index(drop=True)
        )

        print()
        print("DB_BATCH_UPSERT_GATE4_FEATURES_ROWS:", len(batch_features_df))

        total_inserted = int(upsert_gate4_features(batch_features_df))

        print("DB_BATCH_UPSERT_GATE4_FEATURES_DONE:", total_inserted)
    else:
        total_inserted = 0

    if pending_gate4_processed_frames:
        batch_processed_df = pd.concat(pending_gate4_processed_frames, ignore_index=True)
        batch_processed_df = (
            batch_processed_df
            .sort_values(["symbol", "entry_ts"])
            .drop_duplicates(["symbol", "entry_ts"], keep="last")
            .reset_index(drop=True)
        )

        print("DB_BATCH_UPSERT_GATE4_PROCESSED_ROWS:", len(batch_processed_df))

        total_processed = int(upsert_gate4_processed_batch(batch_processed_df))

        print("DB_BATCH_UPSERT_GATE4_PROCESSED_DONE:", total_processed)
    else:
        total_processed = 0

    rep_df = pd.DataFrame(reports)
    rep_df.to_csv(REPORT_CSV, index=False)

    status_counts = rep_df["status"].value_counts(dropna=False).sort_index().to_dict() if len(rep_df) else {}

    summary = {
        "created_at_utc": str(pd.Timestamp.now(tz="UTC")),
        "root": str(ROOT),
        "db_dsn": DB_DSN,
        "online_gate4_features_table": ONLINE_GATE4_FEATURES_TABLE,
        "online_gate4_processed_table": ONLINE_GATE4_PROCESSED_TABLE,
        "source": SOURCE_NAME,
        "feature_builder": FEATURE_BUILDER,
        "symbols_count": int(len(symbols)),
        "rebuild": bool(rebuild),
        "limit_latest": limit_latest,
        "max_target_ts": str(max_target_ts) if max_target_ts is not None else "",
        "status_counts": status_counts,
        "total_candidates": int(total_candidates),
        "total_inserted": int(total_inserted),
        "total_processed": int(total_processed),
        "gate1_proba_min": float(GATE1_PROBA_MIN),
        "g2_cls_base_min": float(G2_CLS_BASE_MIN),
        "g2_cls_extreme_min": float(G2_CLS_EXTREME_MIN),
        "g3_score_extreme_min": float(G3_SCORE_EXTREME_MIN),
        "require_gate1_pass": bool(REQUIRE_GATE1_PASS),
        "report_csv": str(REPORT_CSV),
    }

    REPORT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("DONE")
    print("STATUS COUNTS:", status_counts)
    print("TOTAL CANDIDATES:", total_candidates)
    print("TOTAL INSERTED:", total_inserted)
    print("TOTAL PROCESSED:", total_processed)
    print("WROTE:", REPORT_CSV)
    print("WROTE:", REPORT_JSON)


if __name__ == "__main__":
    main()
