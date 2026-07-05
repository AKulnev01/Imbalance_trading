from __future__ import annotations

from online.trading import config
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from online.oos_context import append_oos_sql_filters, get_online_oos_context

from production.features.build_features_full import build_features_single_symbol


ROOT = Path(__file__).resolve().parents[2]

DB_DSN = os.environ.get(
    "IMB_DB_DSN",
    config.DB_DSN,
)

H4_CONTEXT_DIR = ROOT / "data" / "h4_3"
GATE1_TEMPLATE_DIR = ROOT / "production" / "dataset" / "gate1"

OUT_REPORT_DIR = ROOT / "online" / "_reports_gate1"
OUT_REPORT_JSON = OUT_REPORT_DIR / "online_gate1_features_report.json"
OUT_REPORT_CSV = OUT_REPORT_DIR / "online_gate1_features_report.csv"

ONLINE_TABLE = "online_gate1_features"

BTC_SYMBOL = "BTCUSDT"
ETH_SYMBOL = "ETHUSDT"

FEATURE_BUILDER_NAME = "production.features.build_features_full.build_features_single_symbol"
SOURCE_NAME = "db_candles_h4_only"

SILENCE_FEATURE_BUILDER_STDOUT = True
BRIDGE_BARS_BEFORE_FIRST_DB = 3
H4_STEP = pd.Timedelta(hours=4)

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

TS_CANDIDATES = ["entry_ts", "ts", "timestamp", "open_time", "time", "datetime", "dt"]


def connect_db():
    return psycopg2.connect(DB_DSN)


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def table_qname(table_name: str) -> str:
    return "public." + qident(table_name)


def to_utc_naive(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


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


def find_ts_col(df: pd.DataFrame) -> str:
    for c in TS_CANDIDATES:
        if c in df.columns:
            return c
    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"
    raise RuntimeError(f"timestamp column not found; cols={list(df.columns)[:40]}")


def normalize_h4_parquet_symbol(path: Path) -> str:
    name = path.stem.upper()
    for suffix in ["_H4", "_4H"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def read_h4_parquet(symbol: str) -> pd.DataFrame:
    direct = H4_CONTEXT_DIR / f"{symbol}.parquet"
    alt1 = H4_CONTEXT_DIR / f"{symbol}_h4.parquet"
    alt2 = H4_CONTEXT_DIR / f"{symbol}_4h.parquet"

    path = None
    for p in [direct, alt1, alt2]:
        if p.exists():
            path = p
            break

    if path is None:
        raise FileNotFoundError(f"{symbol}: h4 parquet not found in {H4_CONTEXT_DIR}")

    df = pd.read_parquet(path)
    if df.empty:
        raise RuntimeError(f"{symbol}: h4 parquet is empty: {path}")

    ts_col = find_ts_col(df)

    if ts_col == "__index__":
        df = df.reset_index().rename(columns={"index": "entry_ts"})
    elif ts_col != "entry_ts":
        df = df.rename(columns={ts_col: "entry_ts"})

    need = ["entry_ts", "open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"{symbol}: h4 parquet missing cols={miss}; path={path}")

    out = df[need].copy()
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    out = out.dropna(subset=["entry_ts", "open", "high", "low", "close"])
    out = out.sort_values("entry_ts").drop_duplicates("entry_ts", keep="last").reset_index(drop=True)

    return out




def read_h4_db_all(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    symbols = sorted(set(str(x).upper() for x in symbols if str(x).strip()))

    empty_cols = ["entry_ts", "open", "high", "low", "close", "volume"]
    if not symbols:
        return {}

    where_parts = ["UPPER(c.symbol) = ANY(%s)"]
    params: List[object] = [symbols]

    append_oos_sql_filters(
        where_parts=where_parts,
        params=params,
        table_alias="c",
        ts_column="entry_ts",
        symbol_column="symbol",
    )

    where_sql = " AND ".join(where_parts)

    sql = f"""
        SELECT
            c.symbol,
            c.entry_ts,
            c.open,
            c.high,
            c.low,
            c.close,
            c.volume
        FROM public.candles_h4 c
        WHERE {where_sql}
        ORDER BY c.symbol ASC, c.entry_ts ASC;
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC';")
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    if not rows:
        return {symbol: pd.DataFrame(columns=empty_cols) for symbol in symbols}

    df = pd.DataFrame(
        rows,
        columns=["symbol", "entry_ts", "open", "high", "low", "close", "volume"],
    )

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["symbol", "entry_ts", "open", "high", "low", "close"])
    df = df.sort_values(["symbol", "entry_ts"])
    df = df.drop_duplicates(["symbol", "entry_ts"], keep="last").reset_index(drop=True)

    result: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        part = df[df["symbol"] == symbol][empty_cols].copy()
        part = part.sort_values("entry_ts").drop_duplicates("entry_ts", keep="last").reset_index(drop=True)
        result[symbol] = part

    return result
def read_h4_db(symbol: str) -> pd.DataFrame:
    return read_h4_db_all([symbol]).get(
        str(symbol).upper(),
        pd.DataFrame(columns=["entry_ts", "open", "high", "low", "close", "volume"]),
    )


def load_full_h4_context(
    symbol: str,
    db_h4_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    symbol = str(symbol).upper()

    if db_h4_by_symbol is None:
        out = read_h4_db(symbol)
    else:
        out = db_h4_by_symbol.get(
            symbol,
            pd.DataFrame(columns=["entry_ts", "open", "high", "low", "close", "volume"]),
        ).copy()

    if out.empty:
        return out

    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    out = out.dropna(subset=["entry_ts", "open", "high", "low", "close"])
    out = out.sort_values("entry_ts").drop_duplicates("entry_ts", keep="last").reset_index(drop=True)

    out["symbol"] = symbol
    out["side"] = "BOTH"
    out["side_num"] = 0
    out["y"] = 0
    out["ret"] = out["close"] / out["open"] - 1.0

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=14).mean()

    return out

def add_refs_to_bars(bars: pd.DataFrame, btc4h: pd.DataFrame, eth4h: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()
    x["entry_ts"] = pd.to_datetime(x["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)

    btc = btc4h[["entry_ts", "close"]].copy()
    eth = eth4h[["entry_ts", "close"]].copy()

    btc["entry_ts"] = pd.to_datetime(btc["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    eth["entry_ts"] = pd.to_datetime(eth["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)

    btc = btc.sort_values("entry_ts").drop_duplicates("entry_ts", keep="last")
    eth = eth.sort_values("entry_ts").drop_duplicates("entry_ts", keep="last")

    x = pd.merge_asof(
        x.sort_values("entry_ts"),
        btc.rename(columns={"close": "ref_btc_close"}),
        on="entry_ts",
        direction="backward",
    )

    x = pd.merge_asof(
        x.sort_values("entry_ts"),
        eth.rename(columns={"close": "ref_eth_close"}),
        on="entry_ts",
        direction="backward",
    )

    x["ref_btc_close"] = x["ref_btc_close"].ffill()
    x["ref_eth_close"] = x["ref_eth_close"].ffill()

    x["ref_close"] = (
        x["ref_btc_close"].fillna(x["ref_eth_close"]) +
        x["ref_eth_close"].fillna(x["ref_btc_close"])
    ) / 2.0

    return x


def get_template_file() -> Path:
    files = sorted(
        p for p in GATE1_TEMPLATE_DIR.glob("*.parquet")
        if not p.name.startswith("_")
    )
    if not files:
        raise RuntimeError(f"no gate1 template parquet files found in {GATE1_TEMPLATE_DIR}")
    return files[0]


def load_template_schema() -> pd.DataFrame:
    path = get_template_file()
    df = pd.read_parquet(path)
    if "entry_ts" not in df.columns:
        raise RuntimeError(f"template missing entry_ts: {path}")
    if "symbol" not in df.columns:
        raise RuntimeError(f"template missing symbol: {path}")
    return df


def pg_type_from_series(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return "timestamptz"
    if pd.api.types.is_bool_dtype(s.dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(s.dtype):
        return "double precision"
    if pd.api.types.is_float_dtype(s.dtype):
        return "double precision"
    return "text"


def create_or_update_online_gate1_features_table(template_df: pd.DataFrame) -> None:
    cols = list(template_df.columns)

    col_defs = []
    for c in cols:
        pg_type = pg_type_from_series(template_df[c])
        not_null = " NOT NULL" if c in {"symbol", "entry_ts"} else ""
        col_defs.append(f"{qident(c)} {pg_type}{not_null}")

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_qname(ONLINE_TABLE)} (
        {", ".join(col_defs)},
        online_feature_builder text NOT NULL DEFAULT '{FEATURE_BUILDER_NAME}',
        online_source text NOT NULL DEFAULT '{SOURCE_NAME}',
        online_inserted_at timestamptz NOT NULL DEFAULT now(),
        online_updated_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY ({qident("symbol")}, {qident("entry_ts")})
    );

    CREATE INDEX IF NOT EXISTS idx_{ONLINE_TABLE}_entry_ts
    ON {table_qname(ONLINE_TABLE)} ({qident("entry_ts")});

    CREATE INDEX IF NOT EXISTS idx_{ONLINE_TABLE}_symbol_entry_ts_desc
    ON {table_qname(ONLINE_TABLE)} ({qident("symbol")}, {qident("entry_ts")} DESC);
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)

            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                """,
                (ONLINE_TABLE,),
            )
            existing = {r[0] for r in cur.fetchall()}

            for c in cols:
                if c not in existing:
                    pg_type = pg_type_from_series(template_df[c])
                    cur.execute(
                        f"ALTER TABLE {table_qname(ONLINE_TABLE)} ADD COLUMN {qident(c)} {pg_type};"
                    )

        conn.commit()




def get_existing_feature_ts_all(symbols: List[str]) -> Dict[str, set]:
    symbols = sorted(set(str(x).upper() for x in symbols if str(x).strip()))

    result: Dict[str, set] = {symbol: set() for symbol in symbols}
    if not symbols:
        return result

    where_parts = ["UPPER(f.symbol) = ANY(%s)"]
    params: List[object] = [symbols]

    append_oos_sql_filters(
        where_parts=where_parts,
        params=params,
        table_alias="f",
        ts_column="entry_ts",
        symbol_column="symbol",
    )

    where_sql = " AND ".join(where_parts)

    sql = f"""
        SELECT
            f.symbol,
            f.entry_ts
        FROM {table_qname(ONLINE_TABLE)} f
        WHERE {where_sql};
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC';")
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    for symbol, entry_ts in rows:
        symbol_u = str(symbol).upper()
        ts = pd.to_datetime(entry_ts, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        result.setdefault(symbol_u, set()).add(ts.tz_convert(None))

    return result
def get_existing_feature_ts(symbol: str) -> set:
    symbol_u = str(symbol).upper()
    return get_existing_feature_ts_all([symbol_u]).get(symbol_u, set())

def align_to_template(df_new: pd.DataFrame, template_df: pd.DataFrame) -> pd.DataFrame:
    tpl_cols = list(template_df.columns)
    out = df_new.reindex(columns=tpl_cols).copy()

    for c in tpl_cols:
        tpl_dtype = template_df[c].dtype

        if c == "entry_ts":
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce").dt.tz_convert(None)
            continue

        if c == "symbol":
            out[c] = out[c].astype(str)
            continue

        if pd.api.types.is_datetime64_any_dtype(tpl_dtype):
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce").dt.tz_convert(None)
        elif pd.api.types.is_numeric_dtype(tpl_dtype):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = out[c].where(out[c].notna(), None)

    return out


def value_to_db(v: Any, col: str) -> Any:
    if col == "entry_ts":
        return to_db_utc_datetime(v)

    if isinstance(v, pd.Timestamp):
        return to_db_utc_datetime(v)

    if pd.isna(v):
        return None

    if isinstance(v, (np.integer, np.floating)):
        return float(v)

    if isinstance(v, np.bool_):
        return bool(v)

    return v


def insert_features(df: pd.DataFrame, template_cols: list[str]) -> int:
    if df.empty:
        return 0

    insert_cols = template_cols + [
        "online_feature_builder",
        "online_source",
    ]

    records = []
    for row in df.itertuples(index=False):
        values = []
        row_dict = row._asdict()
        for c in template_cols:
            values.append(value_to_db(row_dict.get(c), c))
        values.append(FEATURE_BUILDER_NAME)
        values.append(SOURCE_NAME)
        records.append(tuple(values))

    quoted_cols = ", ".join(qident(c) for c in insert_cols)

    update_cols = [
        c for c in insert_cols
        if c not in {"symbol", "entry_ts"}
    ]

    update_sql = ", ".join(
        f"{qident(c)} = EXCLUDED.{qident(c)}"
        for c in update_cols
    )

    update_sql += ", online_updated_at = now()"

    sql = f"""
        INSERT INTO {table_qname(ONLINE_TABLE)} ({quoted_cols})
        VALUES %s
        ON CONFLICT ({qident("symbol")}, {qident("entry_ts")})
        DO UPDATE SET
            {update_sql}
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, records, page_size=5000)
        conn.commit()

    return len(records)


def get_symbols_from_h4_context_dir() -> List[str]:
    sql = """
        SELECT DISTINCT symbol
        FROM public.candles_h4
        ORDER BY symbol ASC;
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    symbols = []
    for row in rows:
        symbol = str(row[0]).upper()
        if symbol in EXCLUDED_SYMBOLS:
            continue
        symbols.append(symbol)

    symbols = sorted(set(symbols))
    if not symbols:
        raise RuntimeError("zero symbols found in public.candles_h4")

    return symbols



def get_db_h4_ts_for_symbol(
    symbol: str,
    db_h4_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[pd.Timestamp]:
    symbol = str(symbol).upper()

    if db_h4_by_symbol is None:
        db_df = read_h4_db(symbol)
    else:
        db_df = db_h4_by_symbol.get(
            symbol,
            pd.DataFrame(columns=["entry_ts", "open", "high", "low", "close", "volume"]),
        )

    if db_df.empty:
        return []

    ts = pd.to_datetime(db_df["entry_ts"], utc=True, errors="coerce").dropna()
    return sorted(set(x.tz_convert(None) for x in ts))


def build_features_for_symbol(
    symbol: str,
    btc4h: pd.DataFrame,
    eth4h: pd.DataFrame,
    template_df: pd.DataFrame,
    db_h4_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    existing_feature_ts_by_symbol: Optional[Dict[str, set]] = None,
) -> Dict[str, Any]:
    symbol = str(symbol).upper()

    report = {
        "symbol": symbol,
        "status": "ok",
        "error": "",
        "db_h4_rows": 0,
        "h4_context_rows": 0,
        "bridge_target_rows": 0,
        "existing_feature_rows": 0,
        "missing_target_rows": 0,
        "built_rows_before_align": 0,
        "built_rows_after_filter": 0,
        "inserted_rows": 0,
        "min_missing_ts": "",
        "max_missing_ts": "",
    }

    bars = load_full_h4_context(symbol, db_h4_by_symbol=db_h4_by_symbol)
    if bars.empty:
        report["status"] = "no_h4_context_rows"
        return report

    bars["entry_ts"] = pd.to_datetime(bars["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    bars = bars.dropna(subset=["entry_ts"]).sort_values("entry_ts").drop_duplicates("entry_ts", keep="last").reset_index(drop=True)
    report["h4_context_rows"] = int(len(bars))

    db_ts = get_db_h4_ts_for_symbol(symbol, db_h4_by_symbol=db_h4_by_symbol)
    db_ts = sorted(set(pd.Timestamp(x) for x in db_ts if pd.notna(x)))
    report["db_h4_rows"] = int(len(db_ts))

    full_ts_raw = sorted(set(pd.Timestamp(x) for x in bars["entry_ts"].tolist() if pd.notna(x)))

    target_ts = set(db_ts)

    report["bridge_target_rows"] = 0

    if not target_ts:
        report["status"] = "no_target_h4_rows"
        return report

    if existing_feature_ts_by_symbol is None:
        existing_ts = get_existing_feature_ts(symbol)
    else:
        existing_ts = existing_feature_ts_by_symbol.get(symbol, set())

    report["existing_feature_rows"] = int(len(existing_ts))

    missing_ts = sorted(target_ts - existing_ts)
    report["missing_target_rows"] = int(len(missing_ts))

    if not missing_ts:
        report["status"] = "already_done"
        return report

    report["min_missing_ts"] = str(min(missing_ts))
    report["max_missing_ts"] = str(max(missing_ts))

    bars = add_refs_to_bars(bars, btc4h=btc4h, eth4h=eth4h)

    if SILENCE_FEATURE_BUILDER_STDOUT:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            feat = build_features_single_symbol(bars)
    else:
        feat = build_features_single_symbol(bars)

    if feat is None or feat.empty:
        report["status"] = "feature_builder_empty"
        return report

    feat["entry_ts"] = pd.to_datetime(feat["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    feat = feat.dropna(subset=["entry_ts"])
    feat = feat.sort_values("entry_ts").drop_duplicates(["symbol", "entry_ts"], keep="last").reset_index(drop=True)

    report["built_rows_before_align"] = int(len(feat))

    missing_ts_set = set(missing_ts)
    feat_new = feat[feat["entry_ts"].isin(missing_ts_set)].copy()

    report["built_rows_after_filter"] = int(len(feat_new))

    if feat_new.empty:
        report["status"] = "missing_rows_not_ready_after_feature_builder"
        return report

    aligned = align_to_template(feat_new, template_df)
    inserted = insert_features(aligned, list(template_df.columns))

    report["inserted_rows"] = int(inserted)

    return report


def main() -> None:
    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("H4_CONTEXT_DIR:", H4_CONTEXT_DIR)
    print("GATE1_TEMPLATE_DIR:", GATE1_TEMPLATE_DIR)
    print("ONLINE_TABLE:", f"public.{ONLINE_TABLE}")
    print()

    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    template_df = load_template_schema()
    template_file = get_template_file()

    print("TEMPLATE_FILE:", template_file)
    print("TEMPLATE_COLUMNS:", len(template_df.columns))
    print("FIRST 20 TEMPLATE COLUMNS:", list(template_df.columns[:20]))
    print()

    create_or_update_online_gate1_features_table(template_df)
    oos_ctx = get_online_oos_context()

    if oos_ctx.enabled:
        symbols = list(oos_ctx.symbols)
    else:
        symbols = get_symbols_from_h4_context_dir()

    print("OOS_MODE:", oos_ctx.enabled)
    print("OOS_SYMBOLS:", ",".join(oos_ctx.symbols))
    print("OOS_START:", oos_ctx.start_text)
    print("OOS_END:", oos_ctx.end_text)
    print("SYMBOLS:", len(symbols))
    print("DB_BATCH_LOAD: candles_h4 + existing online_gate1_features")
    print()

    db_h4_by_symbol = read_h4_db_all(symbols)
    existing_feature_ts_by_symbol = get_existing_feature_ts_all(symbols)

    btc4h = load_full_h4_context(BTC_SYMBOL, db_h4_by_symbol=db_h4_by_symbol)
    eth4h = load_full_h4_context(ETH_SYMBOL, db_h4_by_symbol=db_h4_by_symbol)

    reports = []

    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{len(symbols)}] {symbol}")

        try:
            rep = build_features_for_symbol(
                symbol=symbol,
                btc4h=btc4h,
                eth4h=eth4h,
                template_df=template_df,
                db_h4_by_symbol=db_h4_by_symbol,
                existing_feature_ts_by_symbol=existing_feature_ts_by_symbol,
            )
        except Exception as e:
            rep = {
                "symbol": symbol,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "db_h4_rows": 0,
                "h4_context_rows": 0,
                "bridge_target_rows": 0,
                "existing_feature_rows": 0,
                "missing_target_rows": 0,
                "built_rows_before_align": 0,
                "built_rows_after_filter": 0,
                "inserted_rows": 0,
                "min_missing_ts": "",
                "max_missing_ts": "",
            }

        reports.append(rep)

        print(
            f"    status={rep['status']} | "
            f"db_h4={rep.get('db_h4_rows', 0)} | "
            f"missing={rep.get('missing_target_rows', 0)} | "
            f"built={rep.get('built_rows_after_filter', 0)} | "
            f"inserted={rep.get('inserted_rows', 0)}"
        )

        if rep.get("error"):
            print(f"    ERROR: {rep['error']}")

    rep_df = pd.DataFrame(reports)
    rep_df.to_csv(OUT_REPORT_CSV, index=False)

    status_counts = rep_df["status"].value_counts().sort_index().to_dict()

    summary = {
        "created_at_utc": str(pd.Timestamp.now(tz="UTC")),
        "root": str(ROOT),
        "db_dsn": DB_DSN,
        "h4_context_dir": str(H4_CONTEXT_DIR),
        "gate1_template_dir": str(GATE1_TEMPLATE_DIR),
        "template_file": str(template_file),
        "online_table": f"public.{ONLINE_TABLE}",
        "feature_builder": FEATURE_BUILDER_NAME,
        "source": SOURCE_NAME,
        "symbols_count": int(len(symbols)),
        "template_columns_count": int(len(template_df.columns)),
        "template_columns": list(template_df.columns),
        "status_counts": status_counts,
        "total_inserted_rows": int(rep_df["inserted_rows"].fillna(0).sum()) if "inserted_rows" in rep_df.columns else 0,
        "report_csv": str(OUT_REPORT_CSV),
    }

    OUT_REPORT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("DONE")
    print("STATUS COUNTS:", status_counts)
    print("TOTAL INSERTED:", summary["total_inserted_rows"])
    print("WROTE:", OUT_REPORT_CSV)
    print("WROTE:", OUT_REPORT_JSON)

if __name__ == "__main__":
    main()
