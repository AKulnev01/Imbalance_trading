from __future__ import annotations

from online.trading import config
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import time
from typing import Any

import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values


ROOT = Path(__file__).resolve().parents[1]

DB_DSN = config.DB_DSN

M1_BOOTSTRAP_DIR = ROOT / "data" / "m1_4"

BOOTSTRAP_STATE_PATH = ROOT / "online" / "_state_m1_bootstrap_from_parquet.json"
SYNC_REPORT_PATH = ROOT / "online" / "_sync_candles_m1_report.json"

BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_KLINE_ENDPOINT = "/v5/market/kline"

INTERVAL = "1"
LIMIT = 1000
REQUEST_SLEEP_SECONDS = 0.12
RETRY_SLEEP_SECONDS = 2.0
MAX_RETRIES = 5

SOURCE_NAME = "bybit"

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

CATEGORY_PRIORITY = ["linear", "spot"]


@dataclass(frozen=True)
class SymbolBootstrap:
    symbol: str
    last_parquet_ts: pd.Timestamp
    next_fetch_from_ts: pd.Timestamp
    parquet_path: str


def utc_now_floor_minute() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC").floor("min")
    return now.tz_convert(None)


def to_utc_naive(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def ts_to_ms(ts: pd.Timestamp) -> int:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def ms_to_ts(ms: Any) -> pd.Timestamp:
    return pd.to_datetime(int(ms), unit="ms", utc=True).tz_convert(None)

def to_db_utc_datetime(ts: pd.Timestamp) -> datetime:
    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.to_pydatetime()

def find_ts_col(df: pd.DataFrame) -> str:
    for col in ["ts", "timestamp", "open_time", "time", "datetime", "dt", "entry_ts"]:
        if col in df.columns:
            return col

    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"

    raise RuntimeError(f"timestamp column not found; cols={list(df.columns)[:40]}")


def normalize_symbol_from_path(path: Path) -> str:
    name = path.stem.upper()
    if name.endswith("_M1"):
        name = name[:-3]
    return name


def connect_db():
    return psycopg2.connect(DB_DSN)


def create_candles_m1_table() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS public.candles_m1 (
        symbol text NOT NULL,
        market_category text NOT NULL,
        entry_ts timestamptz NOT NULL,
        open double precision NOT NULL,
        high double precision NOT NULL,
        low double precision NOT NULL,
        close double precision NOT NULL,
        volume double precision NOT NULL,
        turnover double precision,
        source text NOT NULL DEFAULT 'bybit',
        inserted_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (symbol, market_category, entry_ts)
    );

    CREATE INDEX IF NOT EXISTS idx_candles_m1_symbol_category_ts_desc
    ON public.candles_m1 (symbol, market_category, entry_ts DESC);

    CREATE INDEX IF NOT EXISTS idx_candles_m1_entry_ts
    ON public.candles_m1 (entry_ts);
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def get_existing_symbols_in_db() -> set[str]:
    query = """
        SELECT DISTINCT symbol
        FROM public.candles_m1
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return {str(r[0]).upper() for r in rows}


def get_last_db_ts_by_symbol(symbol: str) -> pd.Timestamp | None:
    query = """
        SELECT max(entry_ts)
        FROM public.candles_m1
        WHERE symbol = %s
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            value = cur.fetchone()[0]

    if value is None:
        return None

    return to_utc_naive(value)


def get_last_db_ts_by_symbol_category(symbol: str, category: str) -> pd.Timestamp | None:
    query = """
        SELECT max(entry_ts)
        FROM public.candles_m1
        WHERE symbol = %s
          AND market_category = %s
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol, category))
            value = cur.fetchone()[0]

    if value is None:
        return None

    return to_utc_naive(value)


def scan_parquet_bootstrap_points() -> dict[str, SymbolBootstrap]:
    if not M1_BOOTSTRAP_DIR.exists():
        raise RuntimeError(f"M1_BOOTSTRAP_DIR not found: {M1_BOOTSTRAP_DIR}")

    files = sorted(M1_BOOTSTRAP_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet files found in {M1_BOOTSTRAP_DIR}")

    result: dict[str, SymbolBootstrap] = {}

    for path in files:
        symbol = normalize_symbol_from_path(path)

        if symbol in EXCLUDED_SYMBOLS:
            continue

        df = pd.read_parquet(path)
        if df.empty:
            continue

        ts_col = find_ts_col(df)

        if ts_col == "__index__":
            ts_series = pd.Series(df.index)
        else:
            ts_series = df[ts_col]

        ts = pd.to_datetime(ts_series, errors="coerce", utc=True).dropna()
        if ts.empty:
            raise RuntimeError(f"{symbol}: no valid timestamps in {path}")

        last_ts = ts.max().tz_convert(None)
        next_fetch_from_ts = last_ts + pd.Timedelta(minutes=1)

        result[symbol] = SymbolBootstrap(
            symbol=symbol,
            last_parquet_ts=last_ts,
            next_fetch_from_ts=next_fetch_from_ts,
            parquet_path=str(path),
        )

    if not result:
        raise RuntimeError("bootstrap scan produced zero symbols")

    return result


def save_bootstrap_state(bootstrap: dict[str, SymbolBootstrap]) -> None:
    payload = {
        "created_at_utc": str(utc_now_floor_minute()),
        "m1_bootstrap_dir": str(M1_BOOTSTRAP_DIR),
        "excluded_symbols": sorted(EXCLUDED_SYMBOLS),
        "symbols": {
            symbol: {
                "last_parquet_ts": str(item.last_parquet_ts),
                "next_fetch_from_ts": str(item.next_fetch_from_ts),
                "parquet_path": item.parquet_path,
            }
            for symbol, item in sorted(bootstrap.items())
        },
    }

    BOOTSTRAP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BOOTSTRAP_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_bootstrap_state() -> dict[str, SymbolBootstrap]:
    if not BOOTSTRAP_STATE_PATH.exists():
        bootstrap = scan_parquet_bootstrap_points()
        save_bootstrap_state(bootstrap)
        return bootstrap

    with open(BOOTSTRAP_STATE_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result: dict[str, SymbolBootstrap] = {}

    for symbol, item in payload["symbols"].items():
        result[str(symbol).upper()] = SymbolBootstrap(
            symbol=str(symbol).upper(),
            last_parquet_ts=to_utc_naive(item["last_parquet_ts"]),
            next_fetch_from_ts=to_utc_naive(item["next_fetch_from_ts"]),
            parquet_path=str(item["parquet_path"]),
        )

    return result


def request_bybit_kline(
    symbol: str,
    category: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[list[str]]:
    params = {
        "category": category,
        "symbol": symbol,
        "interval": INTERVAL,
        "start": ts_to_ms(start_ts),
        "end": ts_to_ms(end_ts),
        "limit": LIMIT,
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                BYBIT_BASE_URL + BYBIT_KLINE_ENDPOINT,
                params=params,
                timeout=20,
            )

            response.raise_for_status()
            payload = response.json()

            ret_code = int(payload.get("retCode", -1))
            if ret_code != 0:
                ret_msg = str(payload.get("retMsg", ""))
                raise RuntimeError(
                    f"Bybit retCode={ret_code}, retMsg={ret_msg}, "
                    f"symbol={symbol}, category={category}, "
                    f"start={start_ts}, end={end_ts}"
                )

            result = payload.get("result") or {}
            rows = result.get("list") or []

            return rows

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
            else:
                break

    raise RuntimeError(
        f"Bybit request failed after {MAX_RETRIES} retries: "
        f"symbol={symbol}, category={category}, "
        f"start={start_ts}, end={end_ts}, error={last_error}"
    )


def bybit_rows_to_df(symbol: str, category: str, rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "market_category",
                "entry_ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
                "source",
            ]
        )

    parsed = []

    for row in rows:
        if len(row) < 6:
            continue

        parsed.append(
            {
                "symbol": symbol,
                "market_category": category,
                "entry_ts": ms_to_ts(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "turnover": float(row[6]) if len(row) > 6 and row[6] not in [None, ""] else None,
                "source": SOURCE_NAME,
            }
        )

    df = pd.DataFrame(parsed)

    if df.empty:
        return df

    df = (
        df.dropna(subset=["entry_ts", "open", "high", "low", "close", "volume"])
        .sort_values("entry_ts")
        .drop_duplicates(["symbol", "market_category", "entry_ts"], keep="last")
        .reset_index(drop=True)
    )

    return df


def insert_candles_m1(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    records = [
        (
            row.symbol,
            row.market_category,
            to_db_utc_datetime(row.entry_ts),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
            None if pd.isna(row.turnover) else float(row.turnover),
            row.source,
        )
        for row in df.itertuples(index=False)
    ]

    sql = """
        INSERT INTO public.candles_m1 (
            symbol,
            market_category,
            entry_ts,
            open,
            high,
            low,
            close,
            volume,
            turnover,
            source
        )
        VALUES %s
        ON CONFLICT (symbol, market_category, entry_ts)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            turnover = EXCLUDED.turnover,
            source = EXCLUDED.source,
            updated_at = now()
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, records, page_size=5000)
        conn.commit()

    return len(records)


def choose_symbol_category(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> tuple[str | None, pd.DataFrame]:
    errors = []

    for category in CATEGORY_PRIORITY:
        try:
            rows = request_bybit_kline(symbol, category, start_ts, end_ts)
            df = bybit_rows_to_df(symbol, category, rows)

            if not df.empty:
                return category, df

            errors.append(f"{category}: empty")

        except Exception as e:
            errors.append(f"{category}: {e}")

        time.sleep(REQUEST_SLEEP_SECONDS)

    print(f"WARNING: {symbol}: no data from categories; {' | '.join(errors)}")
    return None, pd.DataFrame()


def fetch_symbol_range(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict[str, Any]:
    current_start = start_ts
    total_inserted = 0
    used_categories: set[str] = set()
    batches = 0
    empty_batches = 0
    last_inserted_ts = None
    safety_counter = 0

    while current_start <= end_ts:
        safety_counter += 1
        if safety_counter > 100000:
            raise RuntimeError(f"{symbol}: safety counter exceeded")

        batch_end = min(
            current_start + pd.Timedelta(minutes=LIMIT - 1),
            end_ts,
        )

        category, df = choose_symbol_category(symbol, current_start, batch_end)

        if category is None or df.empty:
            empty_batches += 1
            current_start = batch_end + pd.Timedelta(minutes=1)
            continue

        df = df[(df["entry_ts"] >= current_start) & (df["entry_ts"] <= batch_end)].copy()

        if df.empty:
            empty_batches += 1
            current_start = batch_end + pd.Timedelta(minutes=1)
            continue

        inserted = insert_candles_m1(df)
        total_inserted += inserted
        batches += 1
        used_categories.add(category)

        last_ts = pd.to_datetime(df["entry_ts"], errors="coerce").max()
        last_inserted_ts = last_ts

        current_start = batch_end + pd.Timedelta(minutes=1)

        time.sleep(REQUEST_SLEEP_SECONDS)

    return {
        "symbol": symbol,
        "start_ts": str(start_ts),
        "end_ts": str(end_ts),
        "total_inserted": int(total_inserted),
        "batches": int(batches),
        "empty_batches": int(empty_batches),
        "used_categories": sorted(used_categories),
        "last_inserted_ts": None if last_inserted_ts is None else str(last_inserted_ts),
    }

def resolve_start_ts_for_symbol(symbol: str, bootstrap: dict[str, SymbolBootstrap]) -> pd.Timestamp | None:
    db_last = get_last_db_ts_by_symbol(symbol)

    if db_last is not None:
        return db_last + pd.Timedelta(minutes=1)

    item = bootstrap.get(symbol)
    if item is None:
        print(f"WARNING: {symbol}: no db rows and no bootstrap state")
        return None

    return item.next_fetch_from_ts

def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)

def main() -> None:
    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("M1_BOOTSTRAP_DIR:", M1_BOOTSTRAP_DIR)
    print("BOOTSTRAP_STATE_PATH:", BOOTSTRAP_STATE_PATH)
    print("SYNC_REPORT_PATH:", SYNC_REPORT_PATH)
    print()

    create_candles_m1_table()

    bootstrap = load_bootstrap_state()

    symbols = sorted(s for s in bootstrap.keys() if s not in EXCLUDED_SYMBOLS)

    end_ts = utc_now_floor_minute() - pd.Timedelta(minutes=1)

    print("SYMBOLS:", len(symbols))
    print("FETCH END TS:", end_ts)
    print()

    reports = []

    for idx, symbol in enumerate(symbols, start=1):
        start_ts = resolve_start_ts_for_symbol(symbol, bootstrap)

        if start_ts is None:
            reports.append(
                {
                    "symbol": symbol,
                    "status": "skipped_no_start_ts",
                }
            )
            continue

        if start_ts > end_ts:
            print(f"[{idx}/{len(symbols)}] {symbol}: already up to date | start={start_ts} > end={end_ts}")
            reports.append(
                {
                    "symbol": symbol,
                    "status": "already_up_to_date",
                    "start_ts": str(start_ts),
                    "end_ts": str(end_ts),
                    "total_inserted": 0,
                }
            )
            continue

        print(f"[{idx}/{len(symbols)}] {symbol}: fetch {start_ts} -> {end_ts}")

        try:
            rep = fetch_symbol_range(symbol, start_ts, end_ts)
            rep["status"] = "ok"
            reports.append(rep)

            print(
                f"    inserted={rep['total_inserted']} | "
                f"batches={rep['batches']} | "
                f"categories={rep['used_categories']} | "
                f"last={rep['last_inserted_ts']}"
            )

        except Exception as e:
            reports.append(
                {
                    "symbol": symbol,
                    "status": "error",
                    "start_ts": str(start_ts),
                    "end_ts": str(end_ts),
                    "error": str(e),
                }
            )
            print(f"    ERROR: {e}")

    summary = {
        "created_at_utc": str(utc_now_floor_minute()),
        "root": str(ROOT),
        "db_dsn": DB_DSN,
        "m1_bootstrap_dir": str(M1_BOOTSTRAP_DIR),
        "bootstrap_state_path": str(BOOTSTRAP_STATE_PATH),
        "sync_report_path": str(SYNC_REPORT_PATH),
        "excluded_symbols": sorted(EXCLUDED_SYMBOLS),
        "symbols_count": len(symbols),
        "fetch_end_ts": str(end_ts),
        "total_inserted": int(sum(int(r.get("total_inserted", 0)) for r in reports)),
        "status_counts": dict(pd.Series([r.get("status") for r in reports]).value_counts().sort_index()),
        "reports": reports,
    }

    with open(SYNC_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    print()
    print("=" * 120)
    print("DONE")
    print("TOTAL INSERTED:", summary["total_inserted"])
    print("STATUS COUNTS:", summary["status_counts"])
    print("WROTE:", SYNC_REPORT_PATH)


if __name__ == "__main__":
    main()