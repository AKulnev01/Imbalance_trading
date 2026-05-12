#!/usr/bin/env python3
import argparse
import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ==== настройки по умолчанию ====
BYBIT_PUBLIC_MAIN = "https://api.bybit.com"
DEFAULT_CATEGORY = "linear"   # что сканируем: linear|spot|inverse|option
DEFAULT_INTERVAL = "1"        # 1m
WINDOW_DAYS_STEP = 180        # шаг окна назад при поиске
TIMEOUT = 15
UA = {"User-Agent": "listing-report/1.0"}

def _ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

def get_klines(symbol: str, category: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    url = BYBIT_PUBLIC_MAIN + "/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "end": end_ms,
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=TIMEOUT, headers=UA)
    r.raise_for_status()
    j = r.json()
    if int(j.get("retCode", -1)) != 0:
        return []
    return (j.get("result") or {}).get("list", []) or []

def find_earliest_kline_ts(symbol: str, category: str = DEFAULT_CATEGORY, interval: str = DEFAULT_INTERVAL) -> int:
    """
    Возвращает ms-время САМOЙ РАННЕЙ доступной свечи (если нет — -1).
    Алгоритм:
      1) Идём назад батчами по WINDOW_DAYS_STEP, пока найдём окно с данными.
      2) Внутри этого окна бинарным поиском уточняем «самую раннюю».
    """
    now = datetime.now(timezone.utc)
    end = now
    step = timedelta(days=WINDOW_DAYS_STEP)
    # 1) идём назад, пока в окне [end-step, end] нет данных
    found_window_start = None
    found_window_end = None

    # ограничимся 2018 годом как нижней границей
    lower_bound = datetime(2018, 1, 1, tzinfo=timezone.utc)

    while end > lower_bound:
        start = max(lower_bound, end - step)
        kl = get_klines(symbol, category, interval, _ms(start), _ms(end))
        if kl:
            found_window_start, found_window_end = start, end
            break
        end = start  # сдвигаем окно назад

    if not found_window_start:
        return -1  # совсем нет данных

    # 2) бинарным поиском внутри окна находим точный earliest
    lo = found_window_start
    hi = found_window_end

    # чтобы избежать бесконечного цикла на минутах
    while (hi - lo) > timedelta(minutes=1):
        mid = lo + (hi - lo) / 2
        kl = get_klines(symbol, category, interval, _ms(lo), _ms(mid))
        if kl:
            # данные есть в левой половине => earliest в [lo, mid]
            hi = mid
        else:
            # данных нет слева => earliest в (mid, hi]
            lo = mid

        # не спамим API
        time.sleep(0.05)

    # последний проход: возьмём небольшую «подлупу» на 30 минут
    final_lo = hi - timedelta(minutes=30)
    final_kl = get_klines(symbol, category, interval, _ms(final_lo), _ms(hi))
    if not final_kl:
        # fallback: попробуем точно на границе lo..hi
        final_kl = get_klines(symbol, category, interval, _ms(lo), _ms(hi))
        if not final_kl:
            return _ms(hi)

    # kline формат: [start, open, high, low, close, volume, turnover]
    earliest = min(int(row[0]) for row in final_kl)
    return earliest

def main():
    parser = argparse.ArgumentParser(description="Bybit listing/first-candle report")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="linear|spot|inverse|option (default: linear)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="kline interval, default 1 (1m)")
    parser.add_argument("--out", default="", help="path to save .xlsx (optional)")
    parser.add_argument("--universe", default="", help="comma-separated symbols; если пусто — возьмём из config.TRADE_UNIVERSE")
    args = parser.parse_args()

    # берём юниверс из config, если не передали
    symbols = []
    if args.universe:
        symbols = [s.strip().upper() for s in args.universe.split(",") if s.strip()]
    else:
        try:
            from config import TRADE_UNIVERSE
            symbols = list(dict.fromkeys(TRADE_UNIVERSE))
        except Exception:
            print("⚠️ Не удалось импортировать TRADE_UNIVERSE из config.py — укажи --universe")
            return

    rows = []
    for i, sym in enumerate(symbols, 1):
        try:
            ts = find_earliest_kline_ts(sym, category=args.category, interval=args.interval)
            if ts < 0:
                rows.append({"symbol": sym, "category": args.category, "first_candle_utc": None, "days_history": None, "note": "no_klines"})
                print(f"[{i}/{len(symbols)}] {sym}: no_klines")
                continue
            dt = _from_ms(ts)
            days = (datetime.now(timezone.utc) - dt).days
            rows.append({"symbol": sym, "category": args.category, "first_candle_utc": dt, "days_history": days, "interval_used": args.interval})
            print(f"[{i}/{len(symbols)}] {sym}: {dt.isoformat()}  (~{days} days)")
        except Exception as e:
            rows.append({"symbol": sym, "category": args.category, "first_candle_utc": None, "days_history": None, "note": f"error: {e}"})
            print(f"[{i}/{len(symbols)}] {sym}: ERROR {e}")

        # щадим rate-limit
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    if "first_candle_utc" in df.columns:
        df["first_candle_utc"] = pd.to_datetime(df["first_candle_utc"], utc=True, errors="coerce").dt.tz_localize(None)

    if not args.out:
        # дефолтный путь
        ts_now = datetime.utcnow().strftime("%Y%m%d_%H%M")
        args.out = f"{ts_now}_bybit_listing_{args.category}_{args.interval}.xlsx"

    df.to_excel(args.out, index=False)
    print(f"\n✅ Saved: {args.out}  (rows={len(df)})")

if __name__ == "__main__":
    main()