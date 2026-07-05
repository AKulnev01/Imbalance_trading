from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

DEFAULT_M1_DIR = Path("data/m1_4")
DEFAULT_H4_DIR = Path("data/h4_3")

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
DEFAULT_CATEGORY = "linear"
DEFAULT_INTERVAL = "1"

DEFAULT_MIN_HISTORY_DAYS = 150
DEFAULT_SEARCH_START_UTC = "2019-01-01T00:00:00+00:00"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def parse_utc_dt(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    text = text.replace("/", "")
    text = text.replace("-", "")
    text = text.replace("_", "")

    if not text:
        raise ValueError("empty symbol")

    return text


def symbol_m1_path(symbol: str, m1_dir: Path) -> Path:
    return Path(m1_dir) / "{}.parquet".format(normalize_symbol(symbol))


def symbol_h4_path(symbol: str, h4_dir: Path) -> Path:
    return Path(h4_dir) / "{}.parquet".format(normalize_symbol(symbol))


def read_gate5_3_symbol_status(symbol: str) -> Dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)

    try:
        import psycopg2
        from online.trading import config

        dsn = getattr(config, "DB_DSN", None)

        if not dsn:
            return {
                "checked": False,
                "rows": 0,
                "error": "DB_DSN is empty",
            }

        conn = psycopg2.connect(dsn)

        try:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT COUNT(*), MIN(signal_ts), MAX(signal_ts)
                        FROM online_gate5_3_decisions
                        WHERE symbol = %s
                        """,
                        (normalized_symbol,),
                    )
                    row = cur.fetchone()

                    return {
                        "checked": True,
                        "table": "online_gate5_3_decisions",
                        "rows": int(row[0] or 0),
                        "min_signal_ts": None if row[1] is None else str(row[1]),
                        "max_signal_ts": None if row[2] is None else str(row[2]),
                    }

                except Exception as exc:
                    conn.rollback()

                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM online_gate5_3_decisions
                        WHERE symbol = %s
                        """,
                        (normalized_symbol,),
                    )
                    row = cur.fetchone()

                    return {
                        "checked": True,
                        "table": "online_gate5_3_decisions",
                        "rows": int(row[0] or 0),
                        "warning": "signal_ts bounds query failed: {}".format(repr(exc)),
                    }

        finally:
            conn.close()

    except Exception as exc:
        return {
            "checked": False,
            "rows": 0,
            "error": repr(exc),
        }


def symbol_exists_in_system(symbol: str, m1_dir: Path) -> bool:
    gate5_3_status = read_gate5_3_symbol_status(symbol)
    return int(gate5_3_status.get("rows") or 0) > 0


def read_parquet_ts_bounds(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "rows": 0,
            "min_ts": None,
            "max_ts": None,
        }

    df = pd.read_parquet(path)
    rows = int(len(df))

    ts_col = None
    for candidate in ["ts", "timestamp", "open_time", "datetime", "entry_ts"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        return {
            "exists": True,
            "rows": rows,
            "min_ts": None,
            "max_ts": None,
            "warning": "timestamp column not found",
        }

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()

    if len(ts) == 0:
        return {
            "exists": True,
            "rows": rows,
            "min_ts": None,
            "max_ts": None,
            "warning": "timestamp column exists but all values are null",
        }

    return {
        "exists": True,
        "rows": rows,
        "ts_col": ts_col,
        "min_ts": ts.min().isoformat(),
        "max_ts": ts.max().isoformat(),
    }


def http_get_json(url: str, params: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    query = urllib.parse.urlencode(params)
    full_url = "{}?{}".format(url, query)

    req = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "ImbalanceSearcher-symbol-onboarding/1.0",
        },
        method="GET",
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError("Bybit response is not dict")

    return obj


def bybit_kline_exists(
    symbol: str,
    start_ms: int,
    end_ms: int,
    category: str,
    interval: str,
    timeout_sec: int,
) -> bool:
    if end_ms <= start_ms:
        return False

    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": int(start_ms),
        "end": int(end_ms),
        "limit": 1,
    }

    obj = http_get_json(BYBIT_KLINE_URL, params=params, timeout_sec=timeout_sec)

    ret_code = obj.get("retCode")
    if ret_code != 0:
        return False

    result = obj.get("result", {})
    if not isinstance(result, dict):
        return False

    rows = result.get("list", [])
    return isinstance(rows, list) and len(rows) > 0


def bybit_fetch_one_kline(
    symbol: str,
    start_ms: int,
    end_ms: int,
    category: str,
    interval: str,
    timeout_sec: int,
) -> Optional[List[Any]]:
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": int(start_ms),
        "end": int(end_ms),
        "limit": 1,
    }

    obj = http_get_json(BYBIT_KLINE_URL, params=params, timeout_sec=timeout_sec)

    ret_code = obj.get("retCode")
    if ret_code != 0:
        return None

    result = obj.get("result", {})
    if not isinstance(result, dict):
        return None

    rows = result.get("list", [])
    if not isinstance(rows, list) or not rows:
        return None

    row = rows[0]
    if not isinstance(row, list):
        return None

    return row


def check_bybit_availability(
    symbol: str,
    category: str,
    interval: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    now_ms = dt_to_ms(utc_now())
    start_ms = now_ms - 7 * 24 * 60 * 60 * 1000

    try:
        row = bybit_fetch_one_kline(
            symbol=symbol,
            start_ms=start_ms,
            end_ms=now_ms,
            category=category,
            interval=interval,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": repr(exc),
        }

    if row is None:
        return {
            "available": False,
            "error": "no recent kline returned",
        }

    return {
        "available": True,
        "sample_kline": row,
    }


def find_first_available_kline_ms(
    symbol: str,
    category: str,
    interval: str,
    search_start_ms: int,
    search_end_ms: int,
    timeout_sec: int,
    sleep_sec: float,
) -> Optional[int]:
    if search_end_ms <= search_start_ms:
        return None

    try:
        exists_any = bybit_kline_exists(
            symbol=symbol,
            start_ms=search_start_ms,
            end_ms=search_end_ms,
            category=category,
            interval=interval,
            timeout_sec=timeout_sec,
        )
    except Exception:
        return None

    if not exists_any:
        return None

    low = int(search_start_ms)
    high = int(search_end_ms)

    # Binary search by "does any candle exist in [search_start, mid]".
    # It gives an approximate first availability boundary.
    while high - low > 60 * 1000:
        mid = int((low + high) // 2)

        try:
            has_left = bybit_kline_exists(
                symbol=symbol,
                start_ms=search_start_ms,
                end_ms=mid,
                category=category,
                interval=interval,
                timeout_sec=timeout_sec,
            )
        except Exception:
            has_left = False

        if has_left:
            high = mid
        else:
            low = mid + 60 * 1000

        if sleep_sec > 0:
            time.sleep(float(sleep_sec))

    # Final local probe around high.
    probe_start = max(search_start_ms, high - 24 * 60 * 60 * 1000)
    probe_end = min(search_end_ms, high + 24 * 60 * 60 * 1000)

    row = bybit_fetch_one_kline(
        symbol=symbol,
        start_ms=probe_start,
        end_ms=probe_end,
        category=category,
        interval=interval,
        timeout_sec=timeout_sec,
    )

    if row is None:
        return int(high)

    try:
        return int(float(row[0]))
    except Exception:
        return int(high)


def build_run_tag(prefix: str, now_dt: Optional[datetime] = None) -> str:
    if now_dt is None:
        now_dt = utc_now()
    return "{}_{}".format(prefix, now_dt.strftime("%Y%m%d_%H%M%S"))


def build_candidate_paths(run_tag: str) -> Dict[str, str]:
    return {
        "m1_dir": "data/m1_4/{}".format(run_tag),
        "h4_dir": "data/h4_3/{}".format(run_tag),

        "gate1_dataset_dir": "production/dataset/gate1_candidates/{}".format(run_tag),
        "gate1_models_root": "production/models/final_gate1_candidates/{}".format(run_tag),

        "gate2_dataset_root": "production/dataset/gate2_candidates/{}".format(run_tag),
        "gate2_models_root": "production/models/gate2_mod_5features_candidates/{}".format(run_tag),

        "gate3_dataset_dir": "production/dataset/pa_gate3_v3_long_short_candidates/{}".format(run_tag),
        "gate3_ks_dir": "production/models/ks_candidates/{}".format(run_tag),
        "gate3_models_root": "production/models/final_gate3_score_long_short_candidates/{}".format(run_tag),

        "gate4_dataset_root": "production/dataset/gate4_candidates/{}".format(run_tag),
        "gate4_models_root": "production/models/gate4_candidates/{}".format(run_tag),

        "gate5_dataset_root": "production/dataset/gate5_candidates/{}".format(run_tag),
        "gate5_1_models_root": "production/models/gate5_1_candidates/{}".format(run_tag),
        "gate5_3_models_root": "production/models/gate5_3_candidates/{}".format(run_tag),
    }


def build_new_symbol_prod_paths() -> Dict[str, str]:
    return {
        "m1_dir": "data/m1_4",
        "h4_dir": "data/h4_3",
        "gate1_dataset_dir": "production/dataset/gate1",
        "gate1_models_root": "production/models/final_gate1",
        "gate3_dataset_dir": "production/dataset/pa_gate3_v3_long_short_by_symbol",
        "gate3_ks_dir": "production/models/ks",
        "gate3_models_root": "production/models/final_gate3_score_long_short",
    }


def decide_symbol_request(
    symbol_raw: str,
    mode: str,
    m1_dir: Path,
    h4_dir: Path,
    min_history_days: int,
    category: str,
    interval: str,
    search_start_utc: str,
    timeout_sec: int,
    sleep_sec: float,
    run_tag: str,
) -> Dict[str, Any]:
    symbol = normalize_symbol(symbol_raw)
    now_dt = utc_now()
    now_ms = dt_to_ms(now_dt)

    m1_path = symbol_m1_path(symbol=symbol, m1_dir=m1_dir)
    h4_path = symbol_h4_path(symbol=symbol, h4_dir=h4_dir)

    gate5_3_status = read_gate5_3_symbol_status(symbol)
    existing = int(gate5_3_status.get("rows") or 0) > 0
    partial_onboarding = bool((m1_path.exists() or h4_path.exists()) and not existing)

    base_result: Dict[str, Any] = {
        "symbol_raw": symbol_raw,
        "symbol": symbol,
        "mode": mode,
        "checked_at_utc": now_dt.isoformat(),
        "m1_dir": str(m1_dir),
        "h4_dir": str(h4_dir),
        "m1_path": str(m1_path),
        "h4_path": str(h4_path),
        "exists_in_system": bool(existing),
        "gate5_3_status": gate5_3_status,
        "partial_onboarding": bool(partial_onboarding),
        "min_history_days": int(min_history_days),
    }

    if mode == "add":
        if existing:
            base_result.update({
                "decision": "SYMBOL_ALREADY_EXISTS",
                "allowed": False,
                "next_action": "PROPOSE_BACKTEST",
                "message": "Символ уже есть в системе. Обучение не запускаем; предложить пользователю backtest по символу.",
                "existing_m1_parquet": read_parquet_ts_bounds(m1_path),
                "existing_h4_parquet": read_parquet_ts_bounds(h4_path),
            })
            return base_result

        availability = check_bybit_availability(
            symbol=symbol,
            category=category,
            interval=interval,
            timeout_sec=timeout_sec,
        )

        base_result["bybit_availability"] = availability

        if not availability.get("available", False):
            base_result.update({
                "decision": "SYMBOL_NOT_AVAILABLE_ON_BYBIT",
                "allowed": False,
                "next_action": "REJECT",
                "message": "Символ не найден или недоступен в Bybit API. Введите другой символ.",
            })
            return base_result

        search_start_ms = dt_to_ms(parse_utc_dt(search_start_utc))

        first_ms = find_first_available_kline_ms(
            symbol=symbol,
            category=category,
            interval=interval,
            search_start_ms=search_start_ms,
            search_end_ms=now_ms,
            timeout_sec=timeout_sec,
            sleep_sec=sleep_sec,
        )

        if first_ms is None:
            base_result.update({
                "decision": "CANNOT_DETERMINE_LISTING_START",
                "allowed": False,
                "next_action": "REJECT",
                "message": "Не удалось определить дату начала истории по символу через Bybit API.",
            })
            return base_result

        history_days = float((now_ms - first_ms) / (24 * 60 * 60 * 1000))

        base_result.update({
            "listing_first_kline_ms": int(first_ms),
            "listing_first_kline_utc": ms_to_iso(int(first_ms)),
            "available_history_days": history_days,
        })

        if history_days < float(min_history_days):
            base_result.update({
                "decision": "TOO_LITTLE_HISTORY",
                "allowed": False,
                "next_action": "REJECT",
                "message": "Слишком мало данных по символу, введите другой.",
            })
            return base_result

        base_result.update({
            "decision": "START_OFFLINE_ONBOARDING",
            "allowed": True,
            "next_action": "RUN_OFFLINE_PIPELINE_FOR_NEW_SYMBOL",
            "message": "Символ доступен на Bybit, истории достаточно. Можно запускать offline pipeline добавления символа.",
            "target_paths": build_new_symbol_prod_paths(),
            "notes": [
                "Gate1 model is per-symbol and must be trained for the new symbol.",
                "Gate2 common prod model is not retrained during simple new-symbol onboarding unless admin explicitly starts candidate retrain.",
                "Online DB should receive only validation/OOS period after offline training split is known.",
            ],
        })
        return base_result

    if mode == "backtest":
        if not existing:
            base_result.update({
                "decision": "SYMBOL_NOT_IN_SYSTEM",
                "allowed": False,
                "next_action": "OFFER_ADD_SYMBOL",
                "message": "Символа нет в системе. Сначала нужно добавить символ через offline onboarding.",
            })
            return base_result

        base_result.update({
            "decision": "PROPOSE_BACKTEST",
            "allowed": True,
            "next_action": "RUN_BACKTEST_BY_SYMBOL_AND_THRESHOLDS",
            "message": "Символ есть в системе. Можно запускать backtest по заданным датам и порогам моделей.",
            "existing_m1_parquet": read_parquet_ts_bounds(m1_path),
            "existing_h4_parquet": read_parquet_ts_bounds(h4_path),
        })
        return base_result

    if mode == "retrain":
        if not existing:
            base_result.update({
                "decision": "SYMBOL_NOT_IN_SYSTEM",
                "allowed": False,
                "next_action": "OFFER_ADD_SYMBOL",
                "message": "Для retrain символ уже должен быть в системе. Для нового символа используйте mode=add.",
            })
            return base_result

        if not run_tag:
            run_tag = build_run_tag("retrain")

        base_result.update({
            "decision": "START_CANDIDATE_RETRAIN",
            "allowed": True,
            "next_action": "RUN_OFFLINE_PIPELINE_TO_CANDIDATE_PATHS",
            "message": "Символ уже есть. Можно запускать расширенное переобучение в candidate-пути без перетирания prod.",
            "run_tag": run_tag,
            "candidate_paths": build_candidate_paths(run_tag),
            "existing_m1_parquet": read_parquet_ts_bounds(m1_path),
            "existing_h4_parquet": read_parquet_ts_bounds(h4_path),
            "notes": [
                "Prod models must not be overwritten.",
                "Candidate models must be validated by backtest before promotion.",
                "Promotion means changing online builder/predictor model paths only after approval.",
            ],
        })
        return base_result

    raise ValueError("unsupported mode: {}".format(mode))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symbol onboarding decision module for UI/orchestrator control."
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--mode", choices=["add", "backtest", "retrain"], default="add")

    parser.add_argument("--m1-dir", default=str(DEFAULT_M1_DIR))
    parser.add_argument("--h4-dir", default=str(DEFAULT_H4_DIR))

    parser.add_argument("--min-history-days", type=int, default=DEFAULT_MIN_HISTORY_DAYS)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--search-start-utc", default=DEFAULT_SEARCH_START_UTC)
    parser.add_argument("--timeout-sec", type=int, default=15)
    parser.add_argument("--sleep-sec", type=float, default=0.03)

    parser.add_argument("--run-tag", default="")
    parser.add_argument("--json-out", default="")

    args = parser.parse_args()

    result = decide_symbol_request(
        symbol_raw=args.symbol,
        mode=args.mode,
        m1_dir=Path(str(args.m1_dir)),
        h4_dir=Path(str(args.h4_dir)),
        min_history_days=int(args.min_history_days),
        category=str(args.category),
        interval=str(args.interval),
        search_start_utc=str(args.search_start_utc),
        timeout_sec=int(args.timeout_sec),
        sleep_sec=float(args.sleep_sec),
        run_tag=str(args.run_tag),
    )

    text = json.dumps(result, ensure_ascii=True, indent=2)
    print(text)

    if str(args.json_out).strip():
        out_path = Path(str(args.json_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
