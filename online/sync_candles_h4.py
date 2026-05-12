from __future__ import annotations

from online.trading import config
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import local
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import os
import time

import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values


ROOT = Path(__file__).resolve().parents[1]

DB_DSN = os.environ.get(
    "IMB_DB_DSN",
    config.DB_DSN,
)

H4_BOOTSTRAP_DIR = ROOT / "data" / "h4_3"

BOOTSTRAP_STATE_PATH = ROOT / "online" / "_state_h4_bootstrap_from_parquet.json"
SYNC_REPORT_PATH = ROOT / "online" / "_sync_candles_h4_report.json"

BYBIT_BASE_URL = os.environ.get("BYBIT_BASE_URL", "https://api.bybit.com")
BYBIT_KLINE_ENDPOINT = "/v5/market/kline"

INTERVAL = "240"
LIMIT = int(os.environ.get("H4_SYNC_LIMIT", "1000"))

MAX_WORKERS = int(os.environ.get("H4_SYNC_MAX_WORKERS", "24"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("H4_SYNC_REQUEST_TIMEOUT_SECONDS", "12"))
RETRY_SLEEP_SECONDS = float(os.environ.get("H4_SYNC_RETRY_SLEEP_SECONDS", "0.7"))
MAX_RETRIES = int(os.environ.get("H4_SYNC_MAX_RETRIES", "4"))

SOURCE_NAME = "bybit"
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

CATEGORY_PRIORITY = ["linear", "spot"]

_thread_local = local()


@dataclass(frozen=True)
class SymbolBootstrap:
    symbol: str
    last_parquet_ts: pd.Timestamp
    next_fetch_from_ts: pd.Timestamp
    parquet_path: str


@dataclass(frozen=True)
class SymbolTask:
    symbol: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp


def utc_now_floor_minute() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC").floor("min")
    return now.tz_convert(None)


def last_closed_h4_open_ts() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    current_h4_open = now.floor("4h")
    last_closed_open = current_h4_open - H4_STEP
    return last_closed_open.tz_convert(None)



def delete_unclosed_h4_rows(max_closed_open_ts: pd.Timestamp) -> int:
    sql = """
        DELETE FROM public.candles_h4
        WHERE entry_ts > %s
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (to_db_utc_datetime(max_closed_open_ts),))
            deleted = int(cur.rowcount)
        conn.commit()

    return deleted


def filter_closed_h4_records(records: List[Tuple[Any, ...]], max_closed_open_ts: pd.Timestamp) -> List[Tuple[Any, ...]]:
    if not records:
        return []

    out: List[Tuple[Any, ...]] = []
    cutoff = pd.Timestamp(max_closed_open_ts)

    for rec in records:
        entry_ts = pd.Timestamp(rec[10])
        if entry_ts <= cutoff:
            out.append(rec)

    return out


def to_utc_naive(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def ts_to_ms(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
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

    raise RuntimeError("timestamp column not found; cols={}".format(list(df.columns)[:40]))


def normalize_symbol_from_path(path: Path) -> str:
    name = path.stem.upper()

    for suffix in ["_H4", "_4H"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name


def connect_db():
    return psycopg2.connect(DB_DSN)


def get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=MAX_WORKERS * 2,
            pool_maxsize=MAX_WORKERS * 2,
            max_retries=0,
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _thread_local.session = sess
    return sess


def create_candles_h4_table() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS public.candles_h4 (
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

    CREATE INDEX IF NOT EXISTS idx_candles_h4_symbol_category_ts_desc
    ON public.candles_h4 (symbol, market_category, entry_ts DESC);

    CREATE INDEX IF NOT EXISTS idx_candles_h4_entry_ts
    ON public.candles_h4 (entry_ts);
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def get_last_db_ts_all_symbols() -> Dict[str, pd.Timestamp]:
    query = """
        SELECT symbol, max(entry_ts) AS max_entry_ts
        FROM public.candles_h4
        GROUP BY symbol
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return {}

    out: Dict[str, pd.Timestamp] = {}
    for row in df.itertuples(index=False):
        symbol = str(row.symbol).upper()
        ts = to_utc_naive(row.max_entry_ts)
        if not pd.isna(ts):
            out[symbol] = ts

    return out


def scan_parquet_bootstrap_points() -> Dict[str, SymbolBootstrap]:
    if not H4_BOOTSTRAP_DIR.exists():
        raise RuntimeError("H4_BOOTSTRAP_DIR not found: {}".format(H4_BOOTSTRAP_DIR))

    files = sorted(H4_BOOTSTRAP_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError("no parquet files found in {}".format(H4_BOOTSTRAP_DIR))

    result: Dict[str, SymbolBootstrap] = {}

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
            raise RuntimeError("{}: no valid timestamps in {}".format(symbol, path))

        last_ts = ts.max().tz_convert(None)
        next_fetch_from_ts = last_ts + H4_STEP

        result[symbol] = SymbolBootstrap(
            symbol=symbol,
            last_parquet_ts=last_ts,
            next_fetch_from_ts=next_fetch_from_ts,
            parquet_path=str(path),
        )

    if not result:
        raise RuntimeError("bootstrap scan produced zero symbols")

    return result


def save_bootstrap_state(bootstrap: Dict[str, SymbolBootstrap]) -> None:
    payload = {
        "created_at_utc": str(utc_now_floor_minute()),
        "h4_bootstrap_dir": str(H4_BOOTSTRAP_DIR),
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


def load_bootstrap_state() -> Dict[str, SymbolBootstrap]:
    if not BOOTSTRAP_STATE_PATH.exists():
        bootstrap = scan_parquet_bootstrap_points()
        save_bootstrap_state(bootstrap)
        return bootstrap

    with open(BOOTSTRAP_STATE_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result: Dict[str, SymbolBootstrap] = {}

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
) -> List[List[str]]:
    params = {
        "category": category,
        "symbol": symbol,
        "interval": INTERVAL,
        "start": ts_to_ms(start_ts),
        "end": ts_to_ms(end_ts),
        "limit": LIMIT,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = get_session().get(
                BYBIT_BASE_URL + BYBIT_KLINE_ENDPOINT,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            payload = response.json()

            ret_code = int(payload.get("retCode", -1))
            if ret_code != 0:
                ret_msg = str(payload.get("retMsg", ""))
                raise RuntimeError(
                    "Bybit retCode={}, retMsg={}, symbol={}, category={}, interval={}, start={}, end={}".format(
                        ret_code,
                        ret_msg,
                        symbol,
                        category,
                        INTERVAL,
                        start_ts,
                        end_ts,
                    )
                )

            result = payload.get("result") or {}
            rows = result.get("list") or []
            return rows

        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)

    raise RuntimeError(
        "Bybit request failed after {} retries: symbol={}, category={}, interval={}, start={}, end={}, error={}".format(
            MAX_RETRIES,
            symbol,
            category,
            INTERVAL,
            start_ts,
            end_ts,
            last_error,
        )
    )


def bybit_rows_to_records(symbol: str, category: str, rows: List[List[str]]) -> List[Tuple[Any, ...]]:
    records: List[Tuple[Any, ...]] = []

    for row in rows:
        if len(row) < 6:
            continue

        entry_ts = ms_to_ts(row[0])

        records.append(
            (
                symbol,
                category,
                to_db_utc_datetime(entry_ts),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]) if len(row) > 6 and row[6] not in [None, ""] else None,
                SOURCE_NAME,
                entry_ts,
            )
        )

    return records


def insert_candles_h4_records(records: List[Tuple[Any, ...]]) -> int:
    if not records:
        return 0

    max_closed_open_ts = last_closed_h4_open_ts()
    records = filter_closed_h4_records(records, max_closed_open_ts)

    if not records:
        return 0

    clean_records = [
        r[:10]
        for r in records
    ]

    sql = """
        INSERT INTO public.candles_h4 (
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
            execute_values(cur, sql, clean_records, page_size=10000)
        conn.commit()

    return len(clean_records)


def choose_symbol_category_records(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Tuple[Optional[str], List[Tuple[Any, ...]], List[str]]:
    errors: List[str] = []

    for category in CATEGORY_PRIORITY:
        try:
            rows = request_bybit_kline(symbol, category, start_ts, end_ts)
            records = bybit_rows_to_records(symbol, category, rows)

            max_closed_open_ts = last_closed_h4_open_ts()

            filtered_records = []
            for rec in records:
                entry_ts = pd.Timestamp(rec[10])
                if entry_ts >= start_ts and entry_ts <= end_ts and entry_ts <= max_closed_open_ts:
                    filtered_records.append(rec)

            if filtered_records:
                return category, filtered_records, errors

            errors.append("{}: empty".format(category))

        except Exception as exc:
            errors.append("{}: {}".format(category, exc))

    return None, [], errors


def fetch_symbol_range(task: SymbolTask) -> Dict[str, Any]:
    symbol = task.symbol
    current_start = task.start_ts
    end_ts = task.end_ts

    total_records: List[Tuple[Any, ...]] = []
    used_categories: Set[str] = set()
    batches = 0
    empty_batches = 0
    errors: List[str] = []
    last_inserted_ts: Optional[pd.Timestamp] = None
    safety_counter = 0

    while current_start <= end_ts:
        safety_counter += 1
        if safety_counter > 100000:
            raise RuntimeError("{}: safety counter exceeded".format(symbol))

        batch_end = min(
            current_start + H4_STEP * (LIMIT - 1),
            end_ts,
        )

        category, records, category_errors = choose_symbol_category_records(
            symbol=symbol,
            start_ts=current_start,
            end_ts=batch_end,
        )

        errors.extend(category_errors)

        if category is None or not records:
            empty_batches += 1
            current_start = batch_end + H4_STEP
            continue

        total_records.extend(records)
        batches += 1
        used_categories.add(category)

        batch_last_ts = max(pd.Timestamp(r[10]) for r in records)
        if last_inserted_ts is None or batch_last_ts > last_inserted_ts:
            last_inserted_ts = batch_last_ts

        current_start = batch_end + H4_STEP

    unique_records_map: Dict[Tuple[str, str, datetime], Tuple[Any, ...]] = {}
    for rec in total_records:
        key = (str(rec[0]), str(rec[1]), rec[2])
        unique_records_map[key] = rec

    unique_records = list(unique_records_map.values())

    return {
        "symbol": symbol,
        "start_ts": str(task.start_ts),
        "end_ts": str(task.end_ts),
        "status": "ok",
        "records": unique_records,
        "total_inserted": int(len(unique_records)),
        "batches": int(batches),
        "empty_batches": int(empty_batches),
        "used_categories": sorted(used_categories),
        "last_inserted_ts": None if last_inserted_ts is None else str(last_inserted_ts),
        "errors": errors[:20],
    }


def resolve_start_ts_for_symbol(
    symbol: str,
    bootstrap: Dict[str, SymbolBootstrap],
    db_last_by_symbol: Dict[str, pd.Timestamp],
) -> Optional[pd.Timestamp]:
    db_last = db_last_by_symbol.get(symbol)

    if db_last is not None:
        return db_last + H4_STEP

    item = bootstrap.get(symbol)
    if item is None:
        return None

    return item.next_fetch_from_ts


def build_tasks(
    symbols: List[str],
    bootstrap: Dict[str, SymbolBootstrap],
    end_ts: pd.Timestamp,
) -> Tuple[List[SymbolTask], List[Dict[str, Any]]]:
    db_last_by_symbol = get_last_db_ts_all_symbols()

    tasks: List[SymbolTask] = []
    reports: List[Dict[str, Any]] = []

    for symbol in symbols:
        start_ts = resolve_start_ts_for_symbol(
            symbol=symbol,
            bootstrap=bootstrap,
            db_last_by_symbol=db_last_by_symbol,
        )

        if start_ts is None:
            reports.append(
                {
                    "symbol": symbol,
                    "status": "skipped_no_start_ts",
                }
            )
            continue

        if start_ts > end_ts:
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

        tasks.append(SymbolTask(symbol=symbol, start_ts=start_ts, end_ts=end_ts))

    return tasks, reports


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def main() -> None:
    started_at = time.time()

    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("H4_BOOTSTRAP_DIR:", H4_BOOTSTRAP_DIR)
    print("BOOTSTRAP_STATE_PATH:", BOOTSTRAP_STATE_PATH)
    print("SYNC_REPORT_PATH:", SYNC_REPORT_PATH)
    print("MAX_WORKERS:", MAX_WORKERS)
    print("REQUEST_TIMEOUT_SECONDS:", REQUEST_TIMEOUT_SECONDS)
    print()

    create_candles_h4_table()

    end_ts = last_closed_h4_open_ts()
    deleted_unclosed_rows = delete_unclosed_h4_rows(end_ts)

    bootstrap = load_bootstrap_state()
    symbols = sorted(s for s in bootstrap.keys() if s not in EXCLUDED_SYMBOLS)

    tasks, reports = build_tasks(
        symbols=symbols,
        bootstrap=bootstrap,
        end_ts=end_ts,
    )

    print("SYMBOLS:", len(symbols))
    print("FETCH END TS:", end_ts)
    print("DELETED UNCLOSED H4 ROWS:", deleted_unclosed_rows)
    print("TASKS TO FETCH:", len(tasks))
    print("ALREADY/SKIPPED:", len(reports))
    print()

    all_records: List[Tuple[Any, ...]] = []

    if tasks:
        max_workers = max(1, min(MAX_WORKERS, len(tasks)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(fetch_symbol_range, task): task
                for task in tasks
            }

            done_count = 0

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                done_count += 1

                try:
                    rep = future.result()
                    records = rep.pop("records", [])
                    all_records.extend(records)
                    reports.append(rep)

                    print(
                        "[{}/{}] {}: inserted={} | batches={} | categories={} | last={}".format(
                            done_count,
                            len(tasks),
                            task.symbol,
                            rep.get("total_inserted", 0),
                            rep.get("batches", 0),
                            rep.get("used_categories", []),
                            rep.get("last_inserted_ts"),
                        )
                    )

                except Exception as exc:
                    reports.append(
                        {
                            "symbol": task.symbol,
                            "status": "error",
                            "start_ts": str(task.start_ts),
                            "end_ts": str(task.end_ts),
                            "error": str(exc),
                        }
                    )
                    print("[{}/{}] {}: ERROR: {}".format(done_count, len(tasks), task.symbol, exc))

    inserted_rows = insert_candles_h4_records(all_records)

    elapsed = time.time() - started_at

    summary = {
        "created_at_utc": str(utc_now_floor_minute()),
        "root": str(ROOT),
        "db_dsn": DB_DSN,
        "h4_bootstrap_dir": str(H4_BOOTSTRAP_DIR),
        "bootstrap_state_path": str(BOOTSTRAP_STATE_PATH),
        "sync_report_path": str(SYNC_REPORT_PATH),
        "excluded_symbols": sorted(EXCLUDED_SYMBOLS),
        "symbols_count": len(symbols),
        "fetch_end_ts": str(end_ts),
        "deleted_unclosed_h4_rows": int(deleted_unclosed_rows),
        "tasks_to_fetch": len(tasks),
        "max_workers": MAX_WORKERS,
        "records_fetched": int(len(all_records)),
        "total_inserted": int(inserted_rows),
        "elapsed_sec": round(elapsed, 3),
        "status_counts": dict(pd.Series([r.get("status") for r in reports]).value_counts().sort_index()) if reports else {},
        "reports": reports,
    }

    with open(SYNC_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    print()
    print("=" * 120)
    print("DONE")
    print("TASKS TO FETCH:", summary["tasks_to_fetch"])
    print("DELETED UNCLOSED H4 ROWS:", summary["deleted_unclosed_h4_rows"])
    print("RECORDS FETCHED:", summary["records_fetched"])
    print("TOTAL INSERTED:", summary["total_inserted"])
    print("ELAPSED_SEC:", summary["elapsed_sec"])
    print("STATUS COUNTS:", summary["status_counts"])
    print("WROTE:", SYNC_REPORT_PATH)


if __name__ == "__main__":
    main()
