
from __future__ import annotations
from online.trading import config
import os

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)
import psycopg2
from psycopg2.extras import execute_values


ROOT = Path(os.environ.get("IMB_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DB_DSN = config.DB_DSN

H4_CONTEXT_DIR = ROOT / "data" / "h4_3"

ONLINE_GATE1_FEATURES_TABLE = "public.online_gate1_features"
ONLINE_GATE1_PREDICTIONS_TABLE = "public.online_gate1_predictions"
ONLINE_GATE2_FEATURES_TABLE = "public.online_gate2_features"

REPORT_DIR = ROOT / "online" / "_reports_gate2"
REPORT_CSV = REPORT_DIR / "online_gate2_features_report.csv"
REPORT_JSON = REPORT_DIR / "online_gate2_features_report.json"

ENTRY_DELAY_SECONDS = 90
H4_STEP = pd.Timedelta(hours=4)
N_CONTEXT_BARS = 500

SOURCE_NAME = "online_gate2_5features_v1"

DROP_COLS_FROM_SOURCE = {
    "side",
    "side_num",
    "side_num_feat",

    "symbol_id",
    "symbol_id_feat",

    "tp_px",
    "sl_px",
    "exit_ts",
    "exit_px",
    "exit_reason",
    "pnl_net",

    "y",
    "y_fast",

    "upstream_split",
    "upstream_valid_start_ts",

    "entry_px",
    "ref_close",
    "ref_close_feat",
    "ref_btc_close",
    "ref_btc_close_feat",
    "ref_eth_close",
    "ref_eth_close_feat",

    "online_feature_builder",
    "online_source",
    "online_inserted_at",
    "online_updated_at",
}

DROP_PREFIXES_FROM_SOURCE = (
    "ks_",
    "sym_",
)

TIME_COL_CANDIDATES = ["entry_ts", "ts", "open_time", "time", "datetime", "timestamp"]
H4_TIME_COL_CANDIDATES = ["ts", "entry_ts", "open_time", "time", "datetime", "timestamp"]

META_COLS_COMMON = [
    "symbol",
    "entry_ts",
    "signal_ts",
    "entry_bar_open_ts",
    "entry_ts_exec",
    "entry_px_exec",
]

GATE1_PRED_COLS = [
    "gate1_proba",
    "gate1_threshold",
    "gate1_pass",
    "gate1_pass_model_threshold",
    "gate1_pass_050",
]


def connect_db():
    return psycopg2.connect(DB_DSN)


def split_table_name(full_name: str) -> Tuple[str, str]:
    parts = full_name.split(".")
    if len(parts) != 2:
        raise RuntimeError("table name must be schema.table")
    return parts[0], parts[1]


def utc_now_floor_second() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("s").tz_convert(None)


def to_utc_naive(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).tz_convert(None)


def to_utc_naive_series(values: Any) -> pd.Series:
    s = pd.to_datetime(pd.Series(values), errors="coerce", utc=True)
    return s.dt.tz_convert(None)


def to_db_utc_datetime(ts: Any) -> Optional[datetime]:
    if ts is None or pd.isna(ts):
        return None

    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    return t.to_pydatetime()


def find_first_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def normalize_symbol_from_path(path: Path) -> str:
    name = path.stem.upper()
    for suffix in ["_H4", "_4H"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def pg_type_for_series(s: pd.Series, col: str) -> str:
    if col in {"symbol", "online_source", "online_feature_builder"}:
        return "text"

    if col in {"entry_ts", "signal_ts", "entry_bar_open_ts", "entry_ts_exec", "online_inserted_at", "online_updated_at"}:
        return "timestamptz"

    if pd.api.types.is_bool_dtype(s):
        return "boolean"

    if pd.api.types.is_integer_dtype(s):
        return "double precision"

    if pd.api.types.is_float_dtype(s):
        return "double precision"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "timestamptz"

    return "text"


def get_table_columns(table_name: str) -> List[str]:
    schema, table = split_table_name(table_name)

    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (schema, table))
            rows = cur.fetchall()

    return [r[0] for r in rows]


def table_exists(table_name: str) -> bool:
    schema, table = split_table_name(table_name)

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
            cur.execute(query, (schema, table))
            return bool(cur.fetchone()[0])


def create_or_update_table_from_df(table_name: str, df: pd.DataFrame) -> None:
    schema, table = split_table_name(table_name)

    if df.empty:
        raise RuntimeError("cannot create/update table from empty dataframe")

    columns = list(df.columns)

    col_defs = []
    for c in columns:
        pg_type = pg_type_for_series(df[c], c)
        not_null = ""
        if c in {"symbol", "entry_ts"}:
            not_null = " NOT NULL"
        col_defs.append(f"{sql_ident(c)} {pg_type}{not_null}")

    ddl = f"""
        CREATE TABLE IF NOT EXISTS {schema}.{table} (
            {", ".join(col_defs)},
            PRIMARY KEY (symbol, entry_ts)
        )
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)

            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = %s
                """,
                (schema, table),
            )
            existing = {r[0] for r in cur.fetchall()}

            for c in columns:
                if c in existing:
                    continue
                pg_type = pg_type_for_series(df[c], c)
                cur.execute(
                    f"ALTER TABLE {schema}.{table} ADD COLUMN {sql_ident(c)} {pg_type}"
                )

            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_symbol_ts_desc
                ON {schema}.{table} (symbol, entry_ts DESC)
                """
            )

            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_entry_ts
                ON {schema}.{table} (entry_ts)
                """
            )

        conn.commit()


def get_symbols_from_online_gate1() -> List[str]:
    query = f"""
        SELECT DISTINCT symbol
        FROM {ONLINE_GATE1_FEATURES_TABLE}
        ORDER BY symbol
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn)

    return [str(x).upper() for x in df["symbol"].tolist()]


def load_missing_entry_ts_batch(symbols: List[str], rebuild: bool) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "entry_ts"])

    if rebuild or not table_exists(ONLINE_GATE2_FEATURES_TABLE):
        query = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE1_FEATURES_TABLE} f
            WHERE UPPER(f.symbol) = ANY(%s)
            ORDER BY f.symbol, f.entry_ts
        """
        params = (symbols,)
    else:
        query = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE1_FEATURES_TABLE} f
            LEFT JOIN {ONLINE_GATE2_FEATURES_TABLE} g2
              ON g2.symbol = f.symbol
             AND g2.entry_ts = f.entry_ts
            WHERE UPPER(f.symbol) = ANY(%s)
              AND g2.entry_ts IS NULL
            ORDER BY f.symbol, f.entry_ts
        """
        params = (symbols,)

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["symbol", "entry_ts"]).sort_values(["symbol", "entry_ts"]).reset_index(drop=True)
    return df


def load_online_gate1_features_batch(symbols: List[str], min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    query = f"""
        SELECT *
        FROM {ONLINE_GATE1_FEATURES_TABLE}
        WHERE UPPER(symbol) = ANY(%s)
          AND entry_ts >= %s
          AND entry_ts <= %s
        ORDER BY symbol, entry_ts
    """

    params = (
        symbols,
        to_db_utc_datetime(min_ts),
        to_db_utc_datetime(max_ts),
    )

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["symbol", "entry_ts"]).sort_values(["symbol", "entry_ts"]).reset_index(drop=True)
    return df


def load_online_gate1_predictions_batch(symbols: List[str], min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    query = f"""
        SELECT *
        FROM {ONLINE_GATE1_PREDICTIONS_TABLE}
        WHERE UPPER(symbol) = ANY(%s)
          AND entry_ts >= %s
          AND entry_ts <= %s
        ORDER BY symbol, entry_ts
    """

    params = (
        symbols,
        to_db_utc_datetime(min_ts),
        to_db_utc_datetime(max_ts),
    )

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["symbol", "entry_ts"]).sort_values(["symbol", "entry_ts"]).reset_index(drop=True)
    return df


def load_h4_db_batch(symbols: List[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])

    query = """
        SELECT symbol, entry_ts AS ts, open, high, low, close, volume
        FROM public.candles_h4
        WHERE UPPER(symbol) = ANY(%s)
        ORDER BY symbol, entry_ts
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=(symbols,))

    if df.empty:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["symbol", "ts"]).sort_values(["symbol", "ts"]).drop_duplicates(["symbol", "ts"], keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    return df[["symbol", "ts", "open", "high", "low", "close", "volume"]].copy()


def split_by_symbol(df: pd.DataFrame, time_col: str) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    tmp = df.copy()
    tmp["symbol"] = tmp["symbol"].astype(str).str.upper()
    tmp[time_col] = pd.to_datetime(tmp[time_col], utc=True, errors="coerce").dt.tz_convert(None)
    tmp = tmp.dropna(subset=["symbol", time_col]).sort_values(["symbol", time_col]).reset_index(drop=True)

    for symbol, part in tmp.groupby("symbol", sort=True):
        out[str(symbol).upper()] = part.reset_index(drop=True)

    return out


def get_missing_entry_ts_for_symbol(symbol: str, rebuild: bool) -> pd.DataFrame:
    if rebuild or not table_exists(ONLINE_GATE2_FEATURES_TABLE):
        query = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE1_FEATURES_TABLE} f
            WHERE f.symbol = %s
            ORDER BY f.entry_ts
        """
        params = (symbol,)
    else:
        query = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE1_FEATURES_TABLE} f
            LEFT JOIN {ONLINE_GATE2_FEATURES_TABLE} g2
              ON g2.symbol = f.symbol
             AND g2.entry_ts = f.entry_ts
            WHERE f.symbol = %s
              AND g2.entry_ts IS NULL
            ORDER BY f.entry_ts
        """
        params = (symbol,)

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)
    return df


def load_online_gate1_features(symbol: str, min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM {ONLINE_GATE1_FEATURES_TABLE}
        WHERE symbol = %s
          AND entry_ts >= %s
          AND entry_ts <= %s
        ORDER BY entry_ts
    """

    params = (
        symbol,
        to_db_utc_datetime(min_ts),
        to_db_utc_datetime(max_ts),
    )

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)
    return df


def load_online_gate1_predictions(symbol: str, min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM {ONLINE_GATE1_PREDICTIONS_TABLE}
        WHERE symbol = %s
          AND entry_ts >= %s
          AND entry_ts <= %s
        ORDER BY entry_ts
    """

    params = (
        symbol,
        to_db_utc_datetime(min_ts),
        to_db_utc_datetime(max_ts),
    )

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)
    return df


def load_h4_parquet(symbol: str) -> pd.DataFrame:
    path = H4_CONTEXT_DIR / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = pd.read_parquet(path)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "index" in df.columns and "ts" not in df.columns:
            df = df.rename(columns={"index": "ts"})

    tcol = find_first_col(list(df.columns), H4_TIME_COL_CANDIDATES)
    if tcol is None:
        raise RuntimeError(f"{symbol}: cannot find h4 time column in {path}")

    df["ts"] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["ts"]).copy()

    need = ["open", "high", "low", "close"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"{symbol}: parquet h4 missing cols={miss}")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    out = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    out["symbol"] = symbol
    out = out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return out


def load_h4_db(symbol: str) -> pd.DataFrame:
    query = """
        SELECT symbol, entry_ts AS ts, open, high, low, close, volume
        FROM public.candles_h4
        WHERE symbol = %s
        ORDER BY entry_ts
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=(symbol,))

    if df.empty:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return df[["symbol", "ts", "open", "high", "low", "close", "volume"]].copy()


def load_h4_stitched(symbol: str, h4_db_by_symbol: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    p = load_h4_parquet(symbol)

    if h4_db_by_symbol is None:
        d = load_h4_db(symbol)
    else:
        d = h4_db_by_symbol.get(str(symbol).upper())
        if d is None:
            d = pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])

    parts = []
    if not p.empty:
        parts.append(p)
    if not d.empty:
        parts.append(d)

    if not parts:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])

    out = pd.concat(parts, ignore_index=True)
    out["symbol"] = symbol
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce").dt.tz_convert(None)
    out = out.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    return out[["symbol", "ts", "open", "high", "low", "close", "volume"]].copy()


def true_range(df: pd.DataFrame) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev).abs(), (l - prev).abs()))
    return pd.Series(tr, index=df.index, dtype=float)


def atr14(df: pd.DataFrame) -> pd.Series:
    return true_range(df).rolling(14, min_periods=14).mean()


def add_gate2_5features(h4: pd.DataFrame) -> pd.DataFrame:
    out = h4.copy().sort_values("ts").reset_index(drop=True)

    o = out["open"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)

    tr = true_range(out)

    out["atr14"] = atr14(out)
    out["atr4h"] = tr.rolling(4, min_periods=4).mean()

    out["ret_l1"] = c.pct_change(1)
    out["ret_l2"] = c.pct_change(2)

    body = (c - o).abs()
    rng = (h - l).replace(0.0, np.nan)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)

    out["hammer_like"] = (
        (lower_wick >= 2.0 * body) &
        (upper_wick <= 0.35 * rng) &
        ((body / rng) <= 0.45)
    ).astype(float)

    atr14_to_price = out["atr14"] / c.replace(0.0, np.nan)
    q33 = atr14_to_price.rolling(96, min_periods=32).quantile(0.33)
    q66 = atr14_to_price.rolling(96, min_periods=32).quantile(0.66)

    out["vol_regime"] = np.where(
        atr14_to_price <= q33,
        0.0,
        np.where(atr14_to_price <= q66, 1.0, 2.0),
    )
    out.loc[q33.isna() | q66.isna(), "vol_regime"] = np.nan

    return out


def select_safe_source_cols(df: pd.DataFrame) -> List[str]:
    cols = []

    for c in df.columns:
        if c in DROP_COLS_FROM_SOURCE:
            continue

        if any(str(c).startswith(p) for p in DROP_PREFIXES_FROM_SOURCE):
            continue

        if str(c).endswith("_feat"):
            continue

        if str(c) in ("symbol_id_feat", "side_num_feat"):
            continue

        cols.append(c)

    if "entry_ts" not in cols:
        cols.append("entry_ts")

    return cols


def normalize_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["gate1_pass", "gate1_pass_model_threshold", "gate1_pass_050"]:
        if c in out.columns:
            if out[c].dtype == bool:
                continue
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(float).gt(0.5)

    return out


def build_symbol_gate2_features(
    symbol: str,
    missing_df: pd.DataFrame,
    gate1_features_by_symbol: Dict[str, pd.DataFrame],
    gate1_predictions_by_symbol: Dict[str, pd.DataFrame],
    h4_db_by_symbol: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report = {
        "symbol": symbol,
        "status": "init",
        "missing_rows": int(len(missing_df)),
        "gate1_feature_rows": 0,
        "gate1_prediction_rows": 0,
        "h4_stitched_rows": 0,
        "built_rows": 0,
        "inserted_rows": 0,
        "err": "",
    }

    if missing_df.empty:
        report["status"] = "no_missing"
        return pd.DataFrame(), report

    min_ts = pd.Timestamp(missing_df["entry_ts"].min())
    max_ts = pd.Timestamp(missing_df["entry_ts"].max())

    g1 = gate1_features_by_symbol.get(symbol)
    if g1 is None:
        g1 = pd.DataFrame()

    if not g1.empty:
        g1 = g1[
            (pd.to_datetime(g1["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None) >= min_ts)
            & (pd.to_datetime(g1["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None) <= max_ts)
        ].copy()

    report["gate1_feature_rows"] = int(len(g1))

    if g1.empty:
        report["status"] = "missing_gate1_features"
        return pd.DataFrame(), report

    pred = gate1_predictions_by_symbol.get(symbol)
    if pred is None:
        pred = pd.DataFrame()

    if not pred.empty:
        pred = pred[
            (pd.to_datetime(pred["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None) >= min_ts)
            & (pd.to_datetime(pred["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None) <= max_ts)
        ].copy()

    report["gate1_prediction_rows"] = int(len(pred))

    if pred.empty:
        report["status"] = "missing_gate1_predictions"
        return pd.DataFrame(), report

    pred_keep = ["symbol", "entry_ts"] + [c for c in GATE1_PRED_COLS if c in pred.columns]
    pred = pred[pred_keep].copy()

    base = g1.merge(pred, on=["symbol", "entry_ts"], how="left", suffixes=("", "_pred"))
    base = normalize_bool_columns(base)

    missing_pred = int(base["gate1_proba"].isna().sum()) if "gate1_proba" in base.columns else len(base)
    if missing_pred > 0:
        report["status"] = "gate1_predictions_not_full"
        report["err"] = f"missing gate1 predictions rows={missing_pred}"
        return pd.DataFrame(), report

    h4 = load_h4_stitched(symbol, h4_db_by_symbol=h4_db_by_symbol)
    report["h4_stitched_rows"] = int(len(h4))

    if h4.empty:
        report["status"] = "missing_h4_stitched"
        return pd.DataFrame(), report

    h4f = add_gate2_5features(h4)
    g2_add = h4f[["ts", "atr4h", "ret_l1", "ret_l2", "hammer_like", "vol_regime"]].copy()
    g2_add = g2_add.rename(columns={"ts": "entry_ts"})
    g2_add["entry_ts"] = pd.to_datetime(g2_add["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    g2_add["entry_ts"] = g2_add["entry_ts"] + H4_STEP

    safe_cols = select_safe_source_cols(base)
    work = base[safe_cols].copy()

    work = work.sort_values("entry_ts").drop_duplicates(["symbol", "entry_ts"], keep="last").reset_index(drop=True)
    work = work.merge(g2_add, on="entry_ts", how="left", suffixes=("", "_g2calc"))

    for c in ["atr4h", "ret_l1", "ret_l2", "hammer_like", "vol_regime"]:
        alt = f"{c}_g2calc"
        if alt in work.columns:
            work[c] = work[alt]
            work = work.drop(columns=[alt])

    work["signal_ts"] = work["entry_ts"]
    work["entry_bar_open_ts"] = work["entry_ts"]
    work["entry_ts_exec"] = work["entry_ts"] + pd.Timedelta(seconds=ENTRY_DELAY_SECONDS)
    work["entry_px_exec"] = np.nan

    work["online_source"] = SOURCE_NAME
    work["online_feature_builder"] = "online/gate2/build_online_gate2_features.py"
    now = utc_now_floor_second()
    work["online_inserted_at"] = now
    work["online_updated_at"] = now

    leading_cols = [c for c in META_COLS_COMMON if c in work.columns]
    gate1_cols = [c for c in GATE1_PRED_COLS if c in work.columns]
    service_cols = ["online_source", "online_feature_builder", "online_inserted_at", "online_updated_at"]

    other_cols = [
        c for c in work.columns
        if c not in leading_cols
        and c not in gate1_cols
        and c not in service_cols
    ]

    work = work[leading_cols + gate1_cols + other_cols + service_cols].copy()

    core = ["atr4h", "ret_l1", "ret_l2", "hammer_like", "vol_regime"]
    null_core = {}
    for c in core:
        if c in work.columns:
            null_core[c] = int(work[c].isna().sum())
        else:
            null_core[c] = int(len(work))

    report["core_nulls"] = null_core

    bad_core = {k: v for k, v in null_core.items() if v > 0}
    if bad_core:
        report["status"] = "core_feature_nulls"
        report["err"] = json.dumps(bad_core, ensure_ascii=False)
        return pd.DataFrame(), report

    report["built_rows"] = int(len(work))
    report["status"] = "ok"
    return work, report


def py_value(v: Any) -> Any:
    if pd.isna(v):
        return None

    if isinstance(v, pd.Timestamp):
        return to_db_utc_datetime(v)

    if isinstance(v, datetime):
        return v

    if isinstance(v, np.bool_):
        return bool(v)

    if isinstance(v, bool):
        return bool(v)

    if isinstance(v, np.integer):
        return float(v)

    if isinstance(v, int):
        return float(v)

    if isinstance(v, np.floating):
        return float(v)

    if isinstance(v, float):
        return float(v)

    return v


def insert_dataframe(table_name: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    create_or_update_table_from_df(table_name, df)

    table_cols = get_table_columns(table_name)
    cols = [c for c in df.columns if c in table_cols]

    records = []
    for row in df[cols].itertuples(index=False, name=None):
        records.append(tuple(py_value(v) for v in row))

    schema, table = split_table_name(table_name)

    col_sql = ", ".join(sql_ident(c) for c in cols)
    values_sql = "%s"

    update_cols = [
        c for c in cols
        if c not in {"symbol", "entry_ts", "online_inserted_at", "online_updated_at"}
    ]

    update_sql = ", ".join(
        f"{sql_ident(c)} = EXCLUDED.{sql_ident(c)}"
        for c in update_cols
    )

    sql = f"""
        INSERT INTO {schema}.{table} ({col_sql})
        VALUES {values_sql}
        ON CONFLICT (symbol, entry_ts)
        DO UPDATE SET
            {update_sql},
            online_updated_at = now()
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, records, page_size=1000)
        conn.commit()

    return len(records)


def parse_args() -> Tuple[Optional[str], bool, Optional[int]]:
    import argparse

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
    print("ONLINE_GATE1_FEATURES_TABLE:", ONLINE_GATE1_FEATURES_TABLE)
    print("ONLINE_GATE1_PREDICTIONS_TABLE:", ONLINE_GATE1_PREDICTIONS_TABLE)
    print("ONLINE_GATE2_FEATURES_TABLE:", ONLINE_GATE2_FEATURES_TABLE)
    print("H4_CONTEXT_DIR:", H4_CONTEXT_DIR)
    print("REBUILD:", rebuild)
    print("LIMIT_LATEST:", limit_latest)
    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if symbol_arg:
        symbols = [symbol_arg]
    else:
        symbols = get_symbols_from_online_gate1()

    print("SYMBOLS:", len(symbols))
    print("DB_BATCH_LOAD: missing gate2 targets + gate1 features + gate1 predictions + candles_h4")
    print()

    missing_all = load_missing_entry_ts_batch(symbols=symbols, rebuild=rebuild)

    if limit_latest is not None and not missing_all.empty:
        missing_all = (
            missing_all.groupby("symbol", group_keys=False)
            .tail(limit_latest)
            .sort_values(["symbol", "entry_ts"])
            .reset_index(drop=True)
        )

    missing_by_symbol = split_by_symbol(missing_all, "entry_ts")

    if missing_all.empty:
        gate1_features_all = pd.DataFrame()
        gate1_predictions_all = pd.DataFrame()
    else:
        min_ts = pd.Timestamp(missing_all["entry_ts"].min())
        max_ts = pd.Timestamp(missing_all["entry_ts"].max())
        active_symbols = sorted(missing_all["symbol"].astype(str).str.upper().unique().tolist())

        gate1_features_all = load_online_gate1_features_batch(active_symbols, min_ts, max_ts)
        gate1_predictions_all = load_online_gate1_predictions_batch(active_symbols, min_ts, max_ts)

    h4_db_all = load_h4_db_batch(symbols)

    gate1_features_by_symbol = split_by_symbol(gate1_features_all, "entry_ts")
    gate1_predictions_by_symbol = split_by_symbol(gate1_predictions_all, "entry_ts")
    h4_db_by_symbol = split_by_symbol(h4_db_all, "ts")

    print("MISSING_TARGET_ROWS_TOTAL:", len(missing_all))
    print("MISSING_TARGET_SYMBOLS:", len(missing_by_symbol))
    print("GATE1_FEATURE_ROWS_BATCH:", len(gate1_features_all))
    print("GATE1_PREDICTION_ROWS_BATCH:", len(gate1_predictions_all))
    print("H4_DB_ROWS_BATCH:", len(h4_db_all))
    print()

    reports = []
    total_inserted = 0
    total_built = 0

    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{len(symbols)}] {symbol}")

        try:
            missing = missing_by_symbol.get(symbol)
            if missing is None:
                missing = pd.DataFrame(columns=["symbol", "entry_ts"])

            df, rep = build_symbol_gate2_features(
                symbol=symbol,
                missing_df=missing,
                gate1_features_by_symbol=gate1_features_by_symbol,
                gate1_predictions_by_symbol=gate1_predictions_by_symbol,
                h4_db_by_symbol=h4_db_by_symbol,
            )

            inserted = insert_dataframe(ONLINE_GATE2_FEATURES_TABLE, df) if not df.empty else 0
            rep["inserted_rows"] = int(inserted)

            total_inserted += int(inserted)
            total_built += int(rep.get("built_rows", 0))

            reports.append(rep)

            print(
                f"    status={rep['status']} | "
                f"missing={rep.get('missing_rows', 0)} | "
                f"built={rep.get('built_rows', 0)} | "
                f"inserted={inserted}"
            )

            if rep.get("err"):
                print(f"    err={rep['err']}")

        except Exception as e:
            rep = {
                "symbol": symbol,
                "status": "error",
                "err": f"{type(e).__name__}: {e}",
                "missing_rows": 0,
                "built_rows": 0,
                "inserted_rows": 0,
            }
            reports.append(rep)
            print(f"    ERROR: {rep['err']}")

    rep_df = pd.DataFrame(reports)
    rep_df.to_csv(REPORT_CSV, index=False)

    summary = {
        "created_at_utc": str(utc_now_floor_second()),
        "root": str(ROOT),
        "online_gate1_features_table": ONLINE_GATE1_FEATURES_TABLE,
        "online_gate1_predictions_table": ONLINE_GATE1_PREDICTIONS_TABLE,
        "online_gate2_features_table": ONLINE_GATE2_FEATURES_TABLE,
        "h4_context_dir": str(H4_CONTEXT_DIR),
        "symbols_count": int(len(symbols)),
        "rebuild": bool(rebuild),
        "limit_latest": limit_latest,
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
