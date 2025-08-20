# utils/fetch_data.py
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
    Грузит свечи Bybit v5 /v5/market/kline c пагинацией и безопасной обработкой ошибок.
    Возвращает DataFrame с индексом DatetimeIndex (UTC) и колонками: open, high, low, close, volume, turnover.

    • Источник данных выбирается флагом USE_MAINNET_MARKET_DATA:
      - True  → api.bybit.com (реальные котировки)
      - False → api-testnet.bybit.com (песочница)
    • category по умолчанию берётся из BYBIT_CATEGORY (linear/spot).
    """
    cat = (category or BYBIT_CATEGORY).lower()
    if cat not in ("linear", "spot"):
        raise ValueError("category must be 'linear' or 'spot'")

    allowed = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}
    bybit_interval = INTERVAL_MAP.get(interval, interval if interval in allowed else None)
    if not bybit_interval:
        raise ValueError(f"Unsupported interval: {interval}")

    base_url = (BASE_PUBLIC_MAIN if USE_MAINNET_MARKET_DATA else BASE_PUBLIC_TEST) + "/v5/market/kline"

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    step_ms = _interval_ms(bybit_interval)

    all_rows = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "category": cat,
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
                    # данных больше нет — выходим из пагинации
                    cursor = end_ms
                    break

                # накапливаем
                all_rows.extend(lst)

                # если пришло меньше, чем limit — дальше нечего грузить
                if len(lst) < limit:
                    cursor = end_ms
                    break

                # сдвигаем курсор по последней свече + шаг
                last_ts = int(lst[-1][0])  # start time в ms
                cursor = last_ts + step_ms
                break

            except (requests.RequestException, RuntimeError) as e:
                attempt += 1
                if attempt > max_retries:
                    logger.error(f"get_bybit_klines failed for {symbol} ({interval}): {e}")
                    # возвращаем то, что успели собрать
                    cursor = end_ms
                    break
                sleep_s = 1.5 * attempt
                logger.warning(f"Retry {attempt}/{max_retries} get_bybit_klines {symbol}: {e} → sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)

    if not all_rows:
        logger.warning(f"No kline data for {symbol} {interval} (lookback={lookback_days}d, mainnet={USE_MAINNET_MARKET_DATA})")
        return _empty_df()

    # Формат строки v5: [start, open, high, low, close, volume, turnover]
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    # Приведение типов и индекс
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float,
        "volume": float, "turnover": float,
    })
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)

    return df