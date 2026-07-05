from __future__ import annotations

import argparse
import json
import time
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import local
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[3]

M1_DIR = Path("data/m1_4")
H4_DIR = Path("data/h4_3")
REPORT_PATH = ROOT / "online" / "new" / "actions" / "_download_symbol_m1_h4_parquet_report.json"

BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_KLINE_ENDPOINT = "/v5/market/kline"

INTERVAL_M1 = "1"
LIMIT = 1000

REQUEST_TIMEOUT_SECONDS = 20.0
REQUEST_SLEEP_SECONDS = 0.12
RETRY_SLEEP_SECONDS = 2.0
MAX_RETRIES = 5

CATEGORY_PRIORITY = ["linear", "spot"]

RESAMPLE_RULE = "4h"
RESAMPLE_LABEL = "right"
RESAMPLE_CLOSED = "right"

M1_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
H4_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

_thread_local = local()


@dataclass(frozen=True)
class FetchResult:
    symbol: str
    category: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    rows: int
    batches: int
    empty_batches: int


def fail(message: str) -> None:
    raise SystemExit(message)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_floor_minute() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("min").tz_convert(None)


def last_closed_minute_ts() -> pd.Timestamp:
    return utc_now_floor_minute() - pd.Timedelta(minutes=1)


def to_utc_naive(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def ts_to_ms(ts: pd.Timestamp) -> int:
    value = pd.Timestamp(ts)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    return int(value.timestamp() * 1000)


def ms_to_ts(ms: Any) -> pd.Timestamp:
    return pd.to_datetime(int(ms), unit="ms", utc=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)



def backup_existing_artifact(path: Path, artifact_group: str, symbol: str, report=None) -> str:
    """
    Backup only real data/model artifacts before overwrite.
    This function is intentionally used by new online/new/actions wrappers
    when they are about to overwrite existing production/data artifacts.
    It does NOT create backups of the wrapper script itself.
    """
    p = Path(path)

    if not p.exists():
        return ""

    symbol_safe = str(symbol).upper().replace("/", "_").replace("\\", "_")
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")

    backup_dir = (
        ROOT
        / "online"
        / "new"
        / "actions"
        / "_artifact_backups"
        / str(artifact_group)
        / symbol_safe
        / run_id
    )
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_path = backup_dir / p.name

    suffix_i = 1
    while backup_path.exists():
        backup_path = backup_dir / "{}.dup{}{}".format(p.stem, suffix_i, p.suffix)
        suffix_i += 1

    shutil.copy2(str(p), str(backup_path))

    if report is not None:
        try:
            backup_paths = report.setdefault("backup_paths", [])
            backup_paths.append(str(backup_path))
        except Exception:
            pass

    print("    BACKUP_ARTIFACT:", p, "->", backup_path, flush=True)
    return str(backup_path)

def get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=8,
            pool_maxsize=8,
            max_retries=0,
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _thread_local.session = sess
    return sess


def find_ts_col(df: pd.DataFrame) -> str:
    for col in ["ts", "entry_ts", "timestamp", "open_time", "time", "datetime", "dt"]:
        if col in df.columns:
            return col

    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"

    raise RuntimeError("timestamp column not found; cols={}".format(list(df.columns)[:40]))


def normalize_symbol(raw: str) -> str:
    symbol = str(raw).strip().upper()
    if not symbol:
        fail("empty symbol")
    return symbol



def format_cutoff_folder_name(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)

    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)

    return "to_{}".format(t.strftime("%Y%m%d"))


def choose_training_snapshot_output_path(
    base_path: Path,
    symbol: str,
    parquet_cutoff_ts: pd.Timestamp,
    snapshot_if_base_exists: bool,
    overwrite_snapshot: bool,
    report: dict,
) -> Path:
    """
    Training-cutoff candle output policy.

    If base parquet does not exist:
        write normal base path:
            data/m1_4/<SYMBOL>.parquet

    If base parquet exists and snapshot_if_base_exists=True:
        write snapshot path next to base directory:
            data/m1_4/to_YYYYMMDD/<SYMBOL>.parquet

    Existing base parquet is never overwritten in snapshot mode.
    Existing snapshot parquet is overwritten only with --overwrite-snapshot.
    """
    base_path = Path(base_path)

    report["base_parquet_path"] = str(base_path)
    report["base_parquet_exists_before"] = bool(base_path.exists())

    if not bool(snapshot_if_base_exists):
        report["output_path_mode"] = "base_direct"
        return base_path

    if not base_path.exists():
        report["output_path_mode"] = "base_new_symbol"
        return base_path

    folder_name = format_cutoff_folder_name(parquet_cutoff_ts)
    snapshot_dir = base_path.parent / folder_name
    snapshot_path = snapshot_dir / base_path.name

    report["output_path_mode"] = "snapshot_existing_symbol"
    report["snapshot_dir"] = str(snapshot_dir)
    report["snapshot_parquet_path"] = str(snapshot_path)

    if snapshot_path.exists() and not bool(overwrite_snapshot):
        raise RuntimeError(
            "SNAPSHOT_ALREADY_EXISTS: {}. Pass --overwrite-snapshot to replace it.".format(snapshot_path)
        )

    if snapshot_path.exists() and bool(overwrite_snapshot):
        backup_existing_artifact(snapshot_path, "training_cutoff_snapshot_parquet", symbol, report)

    return snapshot_path

def parse_symbols(raw_values: List[str]) -> List[str]:
    out: List[str] = []

    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            symbol = normalize_symbol(part)
            if symbol:
                out.append(symbol)

    seen = set()
    unique: List[str] = []

    for symbol in out:
        if symbol in seen:
            continue
        unique.append(symbol)
        seen.add(symbol)

    if not unique:
        fail("no symbols provided")

    return unique


def read_existing_ts_series(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]")

    ts_col = find_ts_col(df)

    if ts_col == "__index__":
        ts_raw = pd.Series(df.index)
    else:
        ts_raw = df[ts_col]

    ts = pd.to_datetime(ts_raw, utc=True, errors="coerce").dropna()
    return ts


def resolve_auto_start_ts() -> pd.Timestamp:
    if not M1_DIR.exists():
        fail("M1_DIR not found: {}".format(M1_DIR))

    values: List[pd.Timestamp] = []

    for path in sorted(M1_DIR.glob("*.parquet")):
        if path.name.startswith("_"):
            continue

        try:
            ts = read_existing_ts_series(path)
            if not ts.empty:
                values.append(ts.min().tz_convert(None))
        except Exception as exc:
            print("WARN: cannot read min ts from {}: {}".format(path, exc), flush=True)

    if not values:
        fail("cannot resolve auto start from {}".format(M1_DIR))

    return min(values)


def parse_start_ts(value: str) -> pd.Timestamp:
    text = str(value).strip()
    if text.lower() == "auto":
        return resolve_auto_start_ts()

    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        fail("bad --start value: {}".format(value))
    return ts.tz_convert(None)


def parse_end_ts(value: str) -> pd.Timestamp:
    text = str(value).strip()
    if text.lower() == "now":
        return last_closed_minute_ts()

    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        fail("bad --end value: {}".format(value))
    return ts.tz_convert(None)


def request_bybit_kline(
    symbol: str,
    category: str,
    interval: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> List[List[str]]:
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
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
                raise RuntimeError(
                    "Bybit retCode={}, retMsg={}, symbol={}, category={}, interval={}, start={}, end={}".format(
                        ret_code,
                        payload.get("retMsg", ""),
                        symbol,
                        category,
                        interval,
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
            interval,
            start_ts,
            end_ts,
            last_error,
        )
    )


def bybit_rows_to_m1_df(rows: List[List[str]]) -> pd.DataFrame:
    parsed: List[Dict[str, Any]] = []

    for row in rows:
        if len(row) < 6:
            continue

        parsed.append(
            {
                "ts": ms_to_ts(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )

    if not parsed:
        return pd.DataFrame(columns=M1_COLUMNS)

    df = pd.DataFrame(parsed)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df[M1_COLUMNS]
        .dropna(subset=M1_COLUMNS)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )

    return df


def choose_category(symbol: str, end_ts: pd.Timestamp) -> str:
    probe_start = end_ts - pd.Timedelta(minutes=LIMIT - 1)
    errors: List[str] = []

    for category in CATEGORY_PRIORITY:
        try:
            rows = request_bybit_kline(
                symbol=symbol,
                category=category,
                interval=INTERVAL_M1,
                start_ts=probe_start,
                end_ts=end_ts,
            )
            df = bybit_rows_to_m1_df(rows)
            if not df.empty:
                return category
            errors.append("{}: empty".format(category))
        except Exception as exc:
            errors.append("{}: {}".format(category, exc))

        time.sleep(REQUEST_SLEEP_SECONDS)

    raise RuntimeError("{}: no m1 data from categories: {}".format(symbol, " | ".join(errors)))


def fetch_m1_backward(
    symbol: str,
    category: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Tuple[pd.DataFrame, FetchResult]:
    frames: List[pd.DataFrame] = []
    batches = 0
    empty_batches = 0

    current_end = end_ts

    while current_end >= start_ts:
        batch_start = max(start_ts, current_end - pd.Timedelta(minutes=LIMIT - 1))

        rows = request_bybit_kline(
            symbol=symbol,
            category=category,
            interval=INTERVAL_M1,
            start_ts=batch_start,
            end_ts=current_end,
        )
        df = bybit_rows_to_m1_df(rows)

        if df.empty:
            empty_batches += 1
            break

        df = df[(df["ts"] >= pd.Timestamp(batch_start).tz_localize("UTC")) & (df["ts"] <= pd.Timestamp(current_end).tz_localize("UTC"))].copy()

        if not df.empty:
            frames.append(df)
            batches += 1

        first_ts = pd.to_datetime(df["ts"], utc=True, errors="coerce").min()
        if pd.isna(first_ts):
            empty_batches += 1
            break

        first_naive = pd.Timestamp(first_ts).tz_convert(None)
        if first_naive <= start_ts:
            break

        current_end = first_naive - pd.Timedelta(minutes=1)
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not frames:
        raise RuntimeError("{}: no m1 data fetched".format(symbol))

    out = pd.concat(frames, ignore_index=True)
    out = (
        out[M1_COLUMNS]
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )

    result = FetchResult(
        symbol=symbol,
        category=category,
        start_ts=pd.to_datetime(out["ts"], utc=True, errors="coerce").min().tz_convert(None),
        end_ts=pd.to_datetime(out["ts"], utc=True, errors="coerce").max().tz_convert(None),
        rows=int(len(out)),
        batches=int(batches),
        empty_batches=int(empty_batches),
    )

    return out, result


def read_existing_m1(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=M1_COLUMNS)

    df = pd.read_parquet(path)
    ts_col = find_ts_col(df)

    if ts_col == "__index__":
        df = df.reset_index().rename(columns={"index": "ts"})
    elif ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df[M1_COLUMNS]
        .dropna(subset=M1_COLUMNS)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )


def merge_m1(existing: pd.DataFrame, new: pd.DataFrame, force_rebuild: bool) -> pd.DataFrame:
    if force_rebuild or existing.empty:
        merged = new.copy()
    else:
        merged = pd.concat([existing, new], ignore_index=True)

    merged["ts"] = pd.to_datetime(merged["ts"], utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return (
        merged[M1_COLUMNS]
        .dropna(subset=M1_COLUMNS)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )


def m1_to_h4_like_existing(m1_df: pd.DataFrame) -> pd.DataFrame:
    x = m1_df.copy()
    x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
    x = x.dropna(subset=["ts"]).sort_values("ts")
    x = x.set_index("ts")

    open_s = x["open"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).first()
    high_s = x["high"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).max()
    low_s = x["low"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).min()
    close_s = x["close"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).last()
    volume_s = x["volume"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).sum()

    out = pd.DataFrame(
        {
            "ts": open_s.index,
            "open": open_s.values,
            "high": high_s.values,
            "low": low_s.values,
            "close": close_s.values,
            "volume": volume_s.values,
        }
    )

    out = out.dropna(subset=H4_COLUMNS)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce").dt.tz_convert(None)

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = (
        out[H4_COLUMNS]
        .dropna(subset=H4_COLUMNS)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )

    return out


def validate_grid(df: pd.DataFrame, ts_col: str, expected_delta: pd.Timedelta) -> Dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "min_ts": None,
            "max_ts": None,
            "gap_count": 0,
            "gap_examples": [],
            "dt_top": {},
        }

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna().sort_values()
    dt = ts.diff()
    gaps = dt[(dt != expected_delta) & (~dt.isna())]
    dt_top = dt.dropna().value_counts().head(10)

    return {
        "rows": int(len(ts)),
        "min_ts": str(ts.min()),
        "max_ts": str(ts.max()),
        "gap_count": int(len(gaps)),
        "gap_examples": [str(v) for v in gaps.head(10).tolist()],
        "dt_top": {str(k): int(v) for k, v in dt_top.items()},
    }


def write_m1(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[M1_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = (
        out.dropna(subset=M1_COLUMNS)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )

    out.to_parquet(path, index=False)


def write_h4(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[H4_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce").dt.tz_convert(None)

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = (
        out.dropna(subset=H4_COLUMNS)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )

    out.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Symbol M1 H4 Parquet. "
            "Downloads Bybit M1 candles and builds H4 parquet files for model-training symbol onboarding. "
            "--start auto means: use the earliest existing training M1 parquet timestamp as requested start; "
            "if the symbol was listed later, the resulting file starts from the first available Bybit candle."
        )
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="now")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--parquet-cutoff", default="")
    parser.add_argument("--snapshot-if-base-exists", action="store_true")
    parser.add_argument("--overwrite-snapshot", action="store_true")
    parser.add_argument("--m1-dir", default="data/m1_4")
    parser.add_argument("--h4-dir", default="data/h4_3")
    args = parser.parse_args()

    requested_m1_dir = Path(str(args.m1_dir))
    requested_h4_dir = Path(str(args.h4_dir))

    for _name in [
        "M1_DIR",
        "M1_ROOT",
        "M1_DATA_DIR",
        "DATA_M1_DIR",
        "OUT_M1_DIR",
        "M1_PARQUET_DIR",
    ]:
        if _name in globals():
            globals()[_name] = requested_m1_dir

    for _name in [
        "H4_DIR",
        "H4_ROOT",
        "H4_DATA_DIR",
        "DATA_H4_DIR",
        "OUT_H4_DIR",
        "H4_PARQUET_DIR",
    ]:
        if _name in globals():
            globals()[_name] = requested_h4_dir


    parquet_cutoff_ts = None
    if str(args.parquet_cutoff).strip():
        parquet_cutoff_ts = pd.to_datetime(str(args.parquet_cutoff).strip(), utc=True, errors="coerce")
        if pd.isna(parquet_cutoff_ts):
            raise RuntimeError("bad --parquet-cutoff value: {}".format(args.parquet_cutoff))
        parquet_cutoff_ts = pd.Timestamp(parquet_cutoff_ts)

    ensure_dir(M1_DIR)
    ensure_dir(H4_DIR)
    ensure_dir(REPORT_PATH.parent)

    symbols = parse_symbols(args.symbols)
    start_ts = parse_start_ts(args.start)
    end_ts = parse_end_ts(args.end)
    if parquet_cutoff_ts is not None:
        end_ts = parquet_cutoff_ts

    if start_ts > end_ts:
        fail("start_ts > end_ts: {} > {}".format(start_ts, end_ts))

    print("Download Symbol M1 H4 Parquet")
    print("ROOT:", ROOT)
    print("M1_DIR:", M1_DIR)
    print("H4_DIR:", H4_DIR)
    print("REPORT_PATH:", REPORT_PATH)
    print("SYMBOLS:", symbols)
    print("REQUESTED_START_TS:", start_ts)
    print("END_TS:", end_ts)
    print("FORCE_REBUILD:", bool(args.force_rebuild))
    print("AUTO_START_POLICY: old/listed-before-training-start -> requested training start; listed-later -> first available Bybit candle")
    print("M1_FORMAT:", M1_COLUMNS, "ts=datetime64[ns, UTC]")
    print("H4_FORMAT:", H4_COLUMNS, "ts=datetime64[ns], resample label=right closed=right")
    print("=" * 120)

    reports: List[Dict[str, Any]] = []

    for idx, symbol in enumerate(symbols, start=1):
        started_at = time.time()

        m1_path = M1_DIR / "{}.parquet".format(symbol)
        h4_path = H4_DIR / "{}.parquet".format(symbol)

        report: Dict[str, Any] = {
            "symbol": symbol,
            "status": "OK",
            "error": "",
            "m1_path": str(m1_path),
            "h4_path": str(h4_path),
            "start_ts": str(start_ts),
            "end_ts": str(end_ts),
            "force_rebuild": bool(args.force_rebuild),
        }

        print("[{}/{}] {}: download m1 {} -> {}".format(idx, len(symbols), symbol, start_ts, end_ts), flush=True)

        try:
            category = choose_category(symbol=symbol, end_ts=end_ts)

            m1_new, fetch_result = fetch_m1_backward(
                symbol=symbol,
                category=category,
                start_ts=start_ts,
                end_ts=end_ts,
            )

            m1_existing = read_existing_m1(m1_path)
            m1_out = merge_m1(
                existing=m1_existing,
                new=m1_new,
                force_rebuild=bool(args.force_rebuild),
            )

            if parquet_cutoff_ts is not None:
                cutoff_aware = pd.Timestamp(parquet_cutoff_ts)
                if cutoff_aware.tzinfo is None:
                    cutoff_aware = cutoff_aware.tz_localize("UTC")
                else:
                    cutoff_aware = cutoff_aware.tz_convert("UTC")

                m1_out["ts"] = pd.to_datetime(m1_out["ts"], utc=True, errors="coerce")
                rows_before_cutoff_trim = int(len(m1_out))
                m1_out = m1_out[m1_out["ts"] <= cutoff_aware].copy().reset_index(drop=True)
                report["parquet_cutoff_ts"] = str(cutoff_aware)
                report["rows_before_cutoff_trim"] = rows_before_cutoff_trim
                report["rows_after_cutoff_trim"] = int(len(m1_out))

                if m1_out.empty:
                    raise RuntimeError("{}: no m1 rows after applying parquet cutoff {}".format(symbol, cutoff_aware))

            h4_out = m1_to_h4_like_existing(m1_out)

            m1_write_path = choose_training_snapshot_output_path(
                base_path=m1_path,
                symbol=symbol,
                parquet_cutoff_ts=parquet_cutoff_ts if parquet_cutoff_ts is not None else end_ts,
                snapshot_if_base_exists=bool(args.snapshot_if_base_exists),
                overwrite_snapshot=bool(args.overwrite_snapshot),
                report=report,
            )
            h4_write_path = choose_training_snapshot_output_path(
                base_path=h4_path,
                symbol=symbol,
                parquet_cutoff_ts=parquet_cutoff_ts if parquet_cutoff_ts is not None else end_ts,
                snapshot_if_base_exists=bool(args.snapshot_if_base_exists),
                overwrite_snapshot=bool(args.overwrite_snapshot),
                report=report,
            )

            if m1_write_path == m1_path:
                backup_existing_artifact(m1_path, "m1_parquet", symbol, report)
            if h4_write_path == h4_path:
                backup_existing_artifact(h4_path, "h4_parquet", symbol, report)

            write_m1(m1_write_path, m1_out)
            write_h4(h4_write_path, h4_out)

            report["m1_output_path"] = str(m1_write_path)
            report["h4_output_path"] = str(h4_write_path)

            m1_grid = validate_grid(m1_out, "ts", pd.Timedelta(minutes=1))
            h4_grid = validate_grid(h4_out, "ts", pd.Timedelta(hours=4))

            requested_start_ts = pd.Timestamp(start_ts)
            effective_m1_start_ts = pd.to_datetime(m1_out["ts"], utc=True, errors="coerce").min()
            if pd.isna(effective_m1_start_ts):
                effective_m1_start_naive = None
            else:
                effective_m1_start_naive = pd.Timestamp(effective_m1_start_ts).tz_convert(None)

            if effective_m1_start_naive is None:
                start_resolution = "unknown"
            elif effective_m1_start_naive <= requested_start_ts + pd.Timedelta(minutes=1):
                start_resolution = "training_start"
            else:
                start_resolution = "symbol_first_available_candle"

            report.update(
                {
                    "category": category,
                    "requested_start_ts": str(requested_start_ts),
                    "effective_m1_start_ts": None if effective_m1_start_naive is None else str(effective_m1_start_naive),
                    "start_resolution": start_resolution,
                    "fetched_m1_rows": int(fetch_result.rows),
                    "fetch_batches": int(fetch_result.batches),
                    "fetch_empty_batches": int(fetch_result.empty_batches),
                    "m1_rows_total": int(len(m1_out)),
                    "h4_rows_total": int(len(h4_out)),
                    "m1_min_ts": m1_grid["min_ts"],
                    "m1_max_ts": m1_grid["max_ts"],
                    "m1_gap_count": m1_grid["gap_count"],
                    "m1_gap_examples": m1_grid["gap_examples"],
                    "h4_min_ts": h4_grid["min_ts"],
                    "h4_max_ts": h4_grid["max_ts"],
                    "h4_gap_count": h4_grid["gap_count"],
                    "h4_gap_examples": h4_grid["gap_examples"],
                    "elapsed_sec": round(time.time() - started_at, 3),
                }
            )

            print(
                "    OK | category={} | start_resolution={} | effective_start={} | fetched_m1={} | m1_total={} | h4_total={} | m1_gaps={} | h4_gaps={} | elapsed_sec={}".format(
                    report["category"],
                    report["start_resolution"],
                    report["effective_m1_start_ts"],
                    report["fetched_m1_rows"],
                    report["m1_rows_total"],
                    report["h4_rows_total"],
                    report["m1_gap_count"],
                    report["h4_gap_count"],
                    report["elapsed_sec"],
                ),
                flush=True,
            )
            print("    WROTE_M1:", m1_write_path, flush=True)
            print("    WROTE_H4:", h4_write_path, flush=True)

        except Exception as exc:
            report["status"] = "ERR"
            report["error"] = "{}: {}".format(type(exc).__name__, exc)
            report["elapsed_sec"] = round(time.time() - started_at, 3)
            print("    ERR:", report["error"], flush=True)

        reports.append(report)

    summary = {
        "created_at_utc": str(utc_now_floor_minute()),
        "name": "Download Symbol M1 H4 Parquet",
        "root": str(ROOT),
        "m1_dir": str(M1_DIR),
        "h4_dir": str(H4_DIR),
        "symbols": symbols,
        "requested_start_ts": str(start_ts),
        "end_ts": str(end_ts),
        "auto_start_policy": "old/listed-before-training-start -> requested training start; listed-later -> first available Bybit candle",
        "force_rebuild": bool(args.force_rebuild),
        "parquet_cutoff": "" if parquet_cutoff_ts is None else str(parquet_cutoff_ts),
        "snapshot_if_base_exists": bool(args.snapshot_if_base_exists),
        "overwrite_snapshot": bool(args.overwrite_snapshot),
        "m1_format": {
            "columns": M1_COLUMNS,
            "ts_dtype": "datetime64[ns, UTC]",
        },
        "h4_format": {
            "columns": H4_COLUMNS,
            "ts_dtype": "datetime64[ns]",
            "resample_rule": RESAMPLE_RULE,
            "label": RESAMPLE_LABEL,
            "closed": RESAMPLE_CLOSED,
        },
        "status_counts": dict(pd.Series([r.get("status") for r in reports]).value_counts().sort_index()),
        "reports": reports,
    }

    REPORT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("=" * 120)
    print("DONE")
    print("STATUS_COUNTS:", summary["status_counts"])
    print("WROTE_REPORT:", REPORT_PATH)

    if any(r.get("status") != "OK" for r in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
