from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_VALID_DAYS = 60
DEFAULT_MIN_TRAIN_DAYS = 120


def parse_ts(value: Optional[str], field_name: str) -> Optional[pd.Timestamp]:
    text = str(value or "").strip()
    if not text:
        return None

    if text.lower() in {"now", "utcnow"}:
        return now_utc_minute()

    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        raise RuntimeError("bad {}: {}".format(field_name, value))

    return pd.Timestamp(ts)


def now_utc_minute() -> pd.Timestamp:
    dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return pd.Timestamp(dt)


def fmt_naive(ts: pd.Timestamp) -> str:
    value = pd.Timestamp(ts)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    return value.strftime("%Y-%m-%d %H:%M:%S")


def fmt_iso(ts: pd.Timestamp) -> str:
    value = pd.Timestamp(ts)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    return value.isoformat()


def days_between(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float:
    return float((pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() / 86400.0)


def build_plan(
    symbol: str,
    download_start: pd.Timestamp,
    download_end: pd.Timestamp,
    valid_days: int,
    min_train_days: int,
) -> Dict[str, Any]:
    symbol_norm = str(symbol or "").strip().upper().replace("_", "")
    if not symbol_norm:
        raise RuntimeError("symbol is empty")

    if valid_days <= 0:
        raise RuntimeError("valid_days must be positive")

    if min_train_days <= 0:
        raise RuntimeError("min_train_days must be positive")

    valid_end = pd.Timestamp(download_end)
    valid_start = valid_end - pd.Timedelta(days=int(valid_days))

    train_start = pd.Timestamp(download_start)
    train_end = valid_start

    errors: List[str] = []

    if train_start >= train_end:
        errors.append(
            "train_start must be earlier than train_end: train_start={} train_end={}".format(
                fmt_naive(train_start),
                fmt_naive(train_end),
            )
        )

    train_days = days_between(train_start, train_end) if train_start < train_end else 0.0
    valid_days_actual = days_between(valid_start, valid_end)

    if train_days < float(min_train_days):
        errors.append(
            "not enough train days: train_days={:.2f} min_train_days={}".format(
                train_days,
                min_train_days,
            )
        )

    allowed = len(errors) == 0

    windows = {
        "download_start": fmt_naive(train_start),
        "download_end": fmt_naive(valid_end),
        "train_start": fmt_naive(train_start),
        "train_end": fmt_naive(train_end),
        "valid_start": fmt_naive(valid_start),
        "valid_end": fmt_naive(valid_end),
        "oos_start": fmt_naive(valid_start),
        "oos_end": fmt_naive(valid_end),
    }

    windows_iso = {
        "download_start": fmt_iso(train_start),
        "download_end": fmt_iso(valid_end),
        "train_start": fmt_iso(train_start),
        "train_end": fmt_iso(train_end),
        "valid_start": fmt_iso(valid_start),
        "valid_end": fmt_iso(valid_end),
        "oos_start": fmt_iso(valid_start),
        "oos_end": fmt_iso(valid_end),
    }

    return {
        "symbol": symbol_norm,
        "allowed": allowed,
        "errors": errors,
        "rule": {
            "valid_window": "last_N_days_from_download_end",
            "valid_days": int(valid_days),
            "train_rule": "all available futures candles before valid_start",
            "db_load_rule": "load only valid/OOS candles into DB",
            "parquet_rule": "download_start..download_end goes to parquet",
            "time_basis": "UTC",
        },
        "windows": windows,
        "windows_iso": windows_iso,
        "durations": {
            "total_days": days_between(train_start, valid_end),
            "train_days": train_days,
            "valid_days": valid_days_actual,
            "min_train_days": int(min_train_days),
        },
        "offline_pipeline_args": [
            "--start",
            windows["download_start"],
            "--end",
            windows["download_end"],
            "--train-end",
            windows["train_end"],
            "--valid-start",
            windows["valid_start"],
            "--valid-end",
            windows["valid_end"],
            "--oos-start",
            windows["oos_start"],
            "--oos-end",
            windows["oos_end"],
        ],
        "db_loader_args": [
            "--oos-start",
            windows["oos_start"],
            "--oos-end",
            windows["oos_end"],
        ],
        "online_oos_runner_args": [
            "--start",
            windows["oos_start"],
            "--end",
            windows["oos_end"],
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--symbol", required=True)

    parser.add_argument(
        "--download-start",
        required=True,
        help="UTC start of available futures history. For new symbols this should be listing_first_kline_utc.",
    )

    parser.add_argument(
        "--download-end",
        default="now",
        help="UTC end of downloaded futures history. Default: now.",
    )

    parser.add_argument(
        "--valid-days",
        type=int,
        default=DEFAULT_VALID_DAYS,
        help="Trailing OOS/validation window in days. Default: 60.",
    )

    parser.add_argument(
        "--valid-months",
        type=int,
        default=None,
        help="Deprecated compatibility argument. Ignored; use --valid-days.",
    )

    parser.add_argument(
        "--min-train-days",
        type=int,
        default=DEFAULT_MIN_TRAIN_DAYS,
    )

    parser.add_argument("--json-out", default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    download_start = parse_ts(args.download_start, "download_start")
    download_end = parse_ts(args.download_end, "download_end")

    if download_start is None:
        raise RuntimeError("download_start is empty")

    if download_end is None:
        download_end = now_utc_minute()

    plan = build_plan(
        symbol=args.symbol,
        download_start=download_start,
        download_end=download_end,
        valid_days=int(args.valid_days),
        min_train_days=int(args.min_train_days),
    )

    text = json.dumps(plan, ensure_ascii=False, indent=2)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    print(text)

    if not plan["allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
