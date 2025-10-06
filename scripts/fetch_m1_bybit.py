# scripts/fetch_m1_bybit.py
import os
import time
import json
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Публичный REST v5. Без ключей.
# Пишет минутки в data/m1/<SYMBOL>_m1.parquet со схемой: ts,open,high,low,close,volume

INTERVAL = "1"   # 1m

def _ms(ts: datetime) -> int:
    return int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _make_session(retries: int, backoff: float, connect_timeout: float, read_timeout: float) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        raise_on_redirect=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    # сохраним таймауты на сессию (tuple)
    sess.request_timeout = (connect_timeout, read_timeout)
    return sess

def _fetch_window(session: requests.Session, base_url: str, symbol: str, category: str, start_ms: int, end_ms: int, limit: int) -> pd.DataFrame:
    url = base_url.rstrip("/") + "/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": INTERVAL,
        "start": start_ms,
        "end": end_ms,
        "limit": limit
    }
    r = session.get(url, params=params, timeout=session.request_timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        # 10001 — Not supported symbols, 30034 — symbol not found, и т.п.
        raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
    rows = (data.get("result", {}) or {}).get("list") or []
    out = []
    for it in rows:
        ts = int(it[0])
        o, h, l, c = float(it[1]), float(it[2]), float(it[3]), float(it[4])
        v = float(it[5])
        out.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

def _merge_save(parquet_dir: str, symbol: str, df_new: pd.DataFrame) -> None:
    _ensure_dir(parquet_dir)
    path = os.path.join(parquet_dir, f"{symbol}_m1.parquet")
    if os.path.exists(path):
        try:
            df_old = pd.read_parquet(path)
        except Exception:
            df_old = pd.DataFrame(columns=["ts","open","high","low","close","volume"])
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    else:
        df = df_new.reset_index(drop=True)
    df.to_parquet(path, index=False)

def fetch_symbol(symbol: str, category: str, days: int, parquet_dir: str,
                 session: requests.Session,
                 limit: int, window_minutes: int, sleep_sec: float,
                 base_urls: list[str]) -> dict:
    t_end = _now_utc()
    t_start = t_end - timedelta(days=int(days))
    end_ms = _ms(t_end)
    start_ms = _ms(t_start)

    got = 0
    cursor = start_ms
    batches = 0

    # окно в минутах (не обязано равняться limit, гибко дробим)
    step_ms = max(1, int(window_minutes)) * 60_000

    # будем циклично пробовать несколько базовых доменов
    base_idx = 0

    while cursor < end_ms:
        window_end = min(end_ms, cursor + step_ms)
        base = base_urls[base_idx % len(base_urls)]
        try:
            df = _fetch_window(session, base, symbol, category, cursor, window_end, limit=limit)
        except requests.exceptions.ReadTimeout:
            # мягко пропустим окно и попробуем другой базовый домен
            base_idx += 1
            time.sleep(sleep_sec)
            continue
        except requests.exceptions.RequestException as e:
            # сетевые/HTTP-шумы — перескочим на другой домен или окно
            base_idx += 1
            time.sleep(sleep_sec)
            continue
        except RuntimeError as e:
            # символ неподдержан — сразу выходим
            return {"symbol": symbol, "ok": False, "error": str(e), "rows": 0}

        if df.empty:
            # Данных нет — сдвигаем курсор на окно
            cursor = window_end + 1
            batches += 1
            time.sleep(sleep_sec)
            continue

        # Сохраняем по мере накопления (чтобы не потерять прогресс)
        _merge_save(parquet_dir, symbol, df)

        got += len(df)
        last_ts = int(df["ts"].iloc[-1])
        cursor = last_ts + 60_000  # следующая минута
        batches += 1
        time.sleep(sleep_sec)

    return {"symbol": symbol, "ok": True, "error": "", "rows": int(got), "batches": batches}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="", help="Список символов через запятую (BTCUSDT,ETHUSDT). Если не указан — берём из config.TRADE_UNIVERSE.")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--category", type=str, default=None, help="'spot' | 'linear'. Если не указано — из config.BYBIT_CATEGORY.")
    ap.add_argument("--out", type=str, default="./data/m1")

    # сетевые параметры
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=0.5)
    ap.add_argument("--connect-timeout", type=float, default=6.0)
    ap.add_argument("--read-timeout", type=float, default=15.0)
    ap.add_argument("--sleep", type=float, default=0.35)

    # окно и лимит
    ap.add_argument("--limit", type=int, default=900)           # <= 1000 по API
    ap.add_argument("--window-min", type=int, default=900)      # 900 мин ~ 15ч (чуть меньше лимита)
    args = ap.parse_args()

    try:
        from config import TRADE_UNIVERSE, filter_universe, BYBIT_CATEGORY
    except Exception:
        TRADE_UNIVERSE, BYBIT_CATEGORY, filter_universe = [], "spot", lambda x: x

    symbols = [s.strip().upper() for s in (args.symbols.split(",") if args.symbols else []) if s.strip()]
    if not symbols:
        symbols = filter_universe(TRADE_UNIVERSE or [])
    category = (args.category or BYBIT_CATEGORY or "spot").lower()

    if not symbols:
        print("WARN: пустой список символов. Укажи --symbols или заполни TRADE_UNIVERSE в config.py")
        return

    _ensure_dir(args.out)

    # два базовых домена: основной и алиас (часто помогает при timeouts)
    base_urls = ["https://api.bybit.com", "https://api.bytick.com"]

    session = _make_session(
        retries=args.retries,
        backoff=args.backoff,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout
    )

    results = []
    for sym in symbols:
        info = fetch_symbol(
            symbol=sym,
            category=category,
            days=args.days,
            parquet_dir=args.out,
            session=session,
            limit=min(1000, max(100, int(args.limit))),
            window_minutes=max(60, int(args.window_min)),
            sleep_sec=max(0.05, float(args.sleep)),
            base_urls=base_urls
        )
        results.append(info)
        status = "OK" if info["ok"] else "ERR"
        print(f"[{status}] {sym} rows={info.get('rows',0)} batches={info.get('batches',0)} {info.get('error','')}")

    print(json.dumps({"summary": results}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()