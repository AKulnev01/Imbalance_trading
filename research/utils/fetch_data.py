# utils/fetch_data.py
import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import USE_MAINNET_MARKET_DATA, BYBIT_CATEGORY

logger = logging.getLogger(__name__)

# Публичные базовые урлы
BASE_PUBLIC_MAIN = "https://api.bybit.com"
BASE_PUBLIC_TEST = "https://api-testnet.bybit.com"

# Bybit v5 kline intervals:
# 1m=1, 3m=3, 5m=5, 15m=15, 30m=30, 1h=60, 2h=120, 4h=240, 6h=360, 12h=720, 1d=D, 1w=W, 1M=M
INTERVAL_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W", "1M": "M",
}

def _interval_ms(bybit_interval: str) -> int:
    """Вернёт длительность интервала в миллисекундах (для пагинации)."""
    if bybit_interval.isdigit():
        return int(bybit_interval) * 60_000
    if bybit_interval == "D":
        return 24 * 60 * 60_000
    if bybit_interval == "W":
        return 7 * 24 * 60 * 60_000
    if bybit_interval == "M":
        # для месячного — возьмём 30d как шаг для пагинации
        return 30 * 24 * 60 * 60_000
    raise ValueError(f"Unknown interval: {bybit_interval}")

def _empty_df() -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume", "turnover"]
    return pd.DataFrame(columns=cols).astype({
        "open": float, "high": float, "low": float, "close": float,
        "volume": float, "turnover": float,
    })


def get_bybit_klines_range(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    category: str = None,
    limit: int = 1000,
    max_retries: int = 3,
    timeout: int = 10,
) -> pd.DataFrame:
    """
    Жёсткая выборка свечей по временному диапазону [start_dt, end_dt].
    Фолбэк: если для linear пусто — пробуем spot.
    """
    cat = (category or BYBIT_CATEGORY).lower()
    if cat not in ("linear", "spot"):
        raise ValueError("category must be 'linear' or 'spot'")

    allowed = {"1","3","5","15","30","60","120","240","360","720","D","W","M"}
    bybit_interval = INTERVAL_MAP.get(interval, interval if interval in allowed else None)
    if not bybit_interval:
        raise ValueError(f"Unsupported interval: {interval}")

    base_url = (BASE_PUBLIC_MAIN if USE_MAINNET_MARKET_DATA else BASE_PUBLIC_TEST) + "/v5/market/kline"

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp()   * 1000)
    step_ms  = _interval_ms(bybit_interval)

    def _load_for_category(category_name: str) -> pd.DataFrame:
        all_rows = []
        cursor = start_ms
        while cursor <= end_ms:
            params = {
                "category": category_name,
                "symbol": symbol,
                "interval": bybit_interval,
                "start": cursor,
                "end": end_ms,
                "limit": int(limit),
            }
            attempt = 0
            while True:
                try:
                    resp = requests.get(base_url, params=params, timeout=timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("retCode") != 0:
                        raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
                    lst = data.get("result", {}).get("list", [])
                    if not lst:
                        cursor = end_ms + 1
                        break

                    all_rows.extend(lst)

                    if len(lst) < limit:
                        cursor = end_ms + 1
                        break

                    last_ts = int(lst[-1][0])
                    cursor = last_ts + step_ms
                    break

                except (requests.RequestException, RuntimeError) as e:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(f"get_bybit_klines_range failed {symbol} {interval} cat={category_name}: {e}")
                        cursor = end_ms + 1
                        break
                    time.sleep(1.5 * attempt)

        if not all_rows:
            return _empty_df()

        df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume","turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float,"turnover":float})
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        return df

    df = _load_for_category(cat)
    if df.empty and cat == "linear":
        logger.info(f"[get_bybit_klines_range] empty for linear, trying spot fallback: {symbol} {interval}")
        df = _load_for_category("spot")

    if df.empty:
        logger.warning(f"No kline RANGE for {symbol} {interval} {start_dt}->{end_dt} (mainnet={USE_MAINNET_MARKET_DATA}, cat={cat})")
        return _empty_df()
    return df


def get_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "4h",
    lookback_days: int = 30,
    category: str = None,
    limit: int = 1000,
    max_retries: int = 3,
    timeout: int = 10,
) -> pd.DataFrame:
    """
    Классический lookback по последним N дням (никогда не N=0).
    Фолбэк: linear → spot, если пусто.
    """
    cat = (category or BYBIT_CATEGORY).lower()
    if cat not in ("linear", "spot"):
        raise ValueError("category must be 'linear' or 'spot'")

    allowed = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}
    bybit_interval = INTERVAL_MAP.get(interval, interval if interval in allowed else None)
    if not bybit_interval:
        raise ValueError(f"Unsupported interval: {interval}")

    base_url = (BASE_PUBLIC_MAIN if USE_MAINNET_MARKET_DATA else BASE_PUBLIC_TEST) + "/v5/market/kline"

    # ---- НИКОГДА не 0 дней ----
    try:
        _lb = int(lookback_days or 0)
    except Exception:
        _lb = 0
    if _lb <= 0:
        _lb = int(os.getenv("INTRABAR_LOOKBACK_DAYS_FALLBACK", "60"))
    lookback_days = max(1, _lb)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    step_ms = _interval_ms(bybit_interval)

    def _load_for_category(category_name: str) -> pd.DataFrame:
        all_rows = []
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "category": category_name,
                "symbol": symbol,
                "interval": bybit_interval,
                "start": cursor,
                "end": end_ms,
                "limit": int(limit),
            }
            attempt = 0
            while True:
                try:
                    resp = requests.get(base_url, params=params, timeout=timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("retCode") != 0:
                        raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
                    lst = data.get("result", {}).get("list", [])
                    if not lst:
                        cursor = end_ms
                        break
                    all_rows.extend(lst)
                    if len(lst) < limit:
                        cursor = end_ms
                        break
                    last_ts = int(lst[-1][0])  # startTime ms
                    cursor = last_ts + step_ms
                    break
                except (requests.RequestException, RuntimeError) as e:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(f"get_bybit_klines failed for {symbol} ({interval}) cat={category_name}: {e}")
                        cursor = end_ms
                        break
                    sleep_s = 1.5 * attempt
                    logger.warning(f"Retry {attempt}/{max_retries} get_bybit_klines {symbol} cat={category_name}: {e} → sleep {sleep_s:.1f}s")
                    time.sleep(sleep_s)

        if not all_rows:
            return _empty_df()

        df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume","turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float,"turnover":float})
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        return df

    # ---- основной вызов + фолбэк на spot ----
    df = _load_for_category(cat)
    if df.empty and cat == "linear":
        logger.info(f"[get_bybit_klines] empty for linear, trying spot fallback: {symbol} {interval}")
        df = _load_for_category("spot")

    if df.empty:
        logger.warning(f"No kline data for {symbol} {interval} "
                       f"(lookback={lookback_days}d, mainnet={USE_MAINNET_MARKET_DATA}, cat={cat})")
        return _empty_df()

    return df