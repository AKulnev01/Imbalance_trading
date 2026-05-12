# scripts/fetch_m1_bybit.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
PROGRESS_CHUNK = 60_000  # порог логирования прогресса по символу


# ---------------- util: time & io ----------------

def _ms(ts: datetime) -> int:
    return int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)

def _now_utc() -> datetime:
    # округлим вниз до целой минуты, чтобы не просить «будущее»
    now = datetime.now(timezone.utc)
    return now.replace(second=0, microsecond=0)

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


# ---------------- fetch ----------------

def _fetch_window(session: requests.Session, base_url: str, symbol: str, category: str,
                  start_ms: int, end_ms: int, limit: int) -> pd.DataFrame:
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
        # формат: [startTime, open, high, low, close, volume, turnover]
        ts = int(it[0])
        o, h, l, c = float(it[1]), float(it[2]), float(it[3]), float(it[4])
        v = float(it[5])
        out.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


# ---------------- append-safe save ----------------

def _read_existing_bounds(path: str) -> tuple[int, int, int]:
    """
    Возвращает (min_ts, max_ts, rows) из уже сохранённого файла.
    Если файла нет/пуст — (-1, -1, 0).
    """
    if not os.path.exists(path):
        return -1, -1, 0
    try:
        df_old = pd.read_parquet(path, columns=["ts"])
        if df_old is None or df_old.empty or "ts" not in df_old.columns:
            return -1, -1, 0
        ts = pd.to_numeric(df_old["ts"], errors="coerce").dropna()
        if len(ts) == 0:
            return -1, -1, 0
        return int(ts.min()), int(ts.max()), int(len(ts))
    except Exception:
        return -1, -1, 0

def _merge_save(parquet_dir: str, symbol: str, df_new: pd.DataFrame) -> int:
    """
    Добавляет только новые строки. Возвращает кол-во добавленных строк.
    """
    _ensure_dir(parquet_dir)
    path = os.path.join(parquet_dir, f"{symbol}_m1.parquet")

    min_old, max_old, _ = _read_existing_bounds(path)
    if max_old >= 0:
        df_new = df_new[(df_new["ts"] < min_old) | (df_new["ts"] > max_old)]

    if df_new.empty:
        return 0

    if os.path.exists(path):
        try:
            df_old = pd.read_parquet(path)
        except Exception:
            # битый файл — начнём «с чистого листа»
            df_old = pd.DataFrame(columns=["ts","open","high","low","close","volume"])
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new.copy()

    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    # атомарная запись
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return len(df_new)


# ---------------- progress helpers ----------------

def _should_log_pct(prev_pct: float, cur_pct: float, step_pct: float) -> bool:
    if step_pct <= 0:
        return False
    # Логируем при каждом превышении очередного порога кратного step_pct
    next_gate = (int(prev_pct // step_pct) + 1) * step_pct
    return cur_pct >= next_gate


# ---------------- main per-symbol fetchers ----------------

def _backfill_early(symbol: str, category: str, parquet_dir: str,
                    session: requests.Session, limit: int, window_minutes: int, sleep_sec: float,
                    base_urls: list[str], backfill_days: int, progress_every_pct: float) -> dict:
    """
    Докачивает "прошлое" до (min_ts - backfill_days).
    Если файла нет или backfill_days <= 0 — ничего не делает.
    """
    out_path = os.path.join(parquet_dir, f"{symbol}_m1.parquet")
    min_old, max_old, existing_rows = _read_existing_bounds(out_path)
    if min_old < 0 or backfill_days <= 0:
        return {"symbol": symbol, "ok": True, "error": "", "rows": 0, "batches": 0, "note": "no_backfill"}

    target_start_ms = max(0, min_old - int(backfill_days) * 24 * 60 * 60 * 1000)
    total_ms = max(0, (min_old - target_start_ms))
    if total_ms == 0:
        return {"symbol": symbol, "ok": True, "error": "", "rows": 0, "batches": 0, "note": "already_older"}

    print(f"[INIT-BACK] {symbol}: backfill to >= {pd.to_datetime(target_start_ms, unit='ms', utc=True)} "
          f"(current earliest {pd.to_datetime(min_old, unit='ms', utc=True)}) span≈{total_ms//60000:,} min",
          flush=True)

    got = 0
    batches = 0
    milestone = 0

    step_ms = max(1, int(window_minutes)) * 60_000
    base_idx = 0
    cursor_end = min_old - 60_000  # идём назад

    pct_prev = 0.0

    while cursor_end > target_start_ms:
        window_start = max(target_start_ms, cursor_end - step_ms + 60_000)
        base = base_urls[base_idx % len(base_urls)]
        try:
            df = _fetch_window(session, base, symbol, category, start_ms=window_start, end_ms=cursor_end, limit=limit)
        except requests.exceptions.ReadTimeout:
            base_idx += 1
            time.sleep(sleep_sec)
            continue
        except requests.exceptions.RequestException:
            base_idx += 1
            time.sleep(sleep_sec)
            continue
        except RuntimeError as e:
            return {"symbol": symbol, "ok": False, "error": str(e), "rows": 0, "batches": batches}

        if df.empty:
            cursor_end = window_start - 60_000
            batches += 1
            # процент прогресса даже если пусто
            processed_ms = (min_old - cursor_end)
            pct = min(100.0, (processed_ms / total_ms) * 100.0)
            if _should_log_pct(pct_prev, pct, progress_every_pct):
                print(f"[PCT-BACK] {symbol}: {pct:.1f}% done (earliest≈{pd.to_datetime(cursor_end, unit='ms', utc=True)})",
                      flush=True)
                pct_prev = pct
            time.sleep(sleep_sec)
            continue

        added = _merge_save(parquet_dir, symbol, df)
        got += int(added)

        if got // PROGRESS_CHUNK > milestone:
            milestone = got // PROGRESS_CHUNK
            first_ts = int(df["ts"].iloc[0])
            first_iso = pd.to_datetime(first_ts, unit="ms", utc=True)
            print(f"[PROGRESS-BACK] {symbol}: +{milestone*PROGRESS_CHUNK:,} early "
                  f"(rows ~{existing_rows:,} → ~{existing_rows + milestone*PROGRESS_CHUNK:,}) "
                  f"down to {first_iso}", flush=True)

        # двигаем курсор назад
        first_ts = int(df["ts"].iloc[0])
        cursor_end = first_ts - 60_000
        batches += 1

        # процент прогресса
        processed_ms = (min_old - cursor_end)
        pct = min(100.0, (processed_ms / total_ms) * 100.0)
        if _should_log_pct(pct_prev, pct, progress_every_pct):
            print(f"[PCT-BACK] {symbol}: {pct:.1f}% done (earliest≈{pd.to_datetime(first_ts, unit='ms', utc=True)})",
                  flush=True)
            pct_prev = pct

        time.sleep(sleep_sec)

    if got % PROGRESS_CHUNK != 0 and got > 0:
        first_iso = pd.to_datetime(max(target_start_ms, 0), unit="ms", utc=True)
        print(f"[PROGRESS-BACK] {symbol}: +{got:,} early (new earliest ≥ {first_iso})", flush=True)

    return {"symbol": symbol, "ok": True, "error": "", "rows": int(got), "batches": batches, "note": "backfill"}


def _append_forward(symbol: str, category: str, days: int, parquet_dir: str,
                    session: requests.Session, limit: int, window_minutes: int, sleep_sec: float,
                    base_urls: list[str], progress_every_pct: float) -> dict:
    """
    Старое поведение: если файла нет — качаем последние days.
    Если файл есть — докачка от (max_ts+60s) до now.
    """
    out_path = os.path.join(parquet_dir, f"{symbol}_m1.parquet")
    now_utc = _now_utc()
    end_ms = _ms(now_utc)

    min_old_ts, max_old_ts, existing_rows = _read_existing_bounds(out_path)
    if max_old_ts >= 0:
        start_ms = max_old_ts + 60_000
        note = "resume"
        start_iso = pd.to_datetime(max_old_ts, unit="ms", utc=True)
    else:
        t_start = now_utc - timedelta(days=int(days))
        start_ms = _ms(t_start)
        note = "bootstrap"
        start_iso = t_start

    total_ms = max(0, end_ms - start_ms)
    if total_ms == 0:
        return {"symbol": symbol, "ok": True, "error": "", "rows": 0, "batches": 0, "note": "up_to_date"}

    print(f"[INIT-FWD] {symbol}: {note} from {start_iso} → {now_utc} span≈{total_ms//60000:,} min", flush=True)

    got = 0
    batches = 0
    milestone = 0

    step_ms = max(1, int(window_minutes)) * 60_000
    base_idx = 0
    cursor = start_ms

    pct_prev = 0.0

    while cursor < end_ms:
        window_end = min(end_ms, cursor + step_ms)
        base = base_urls[base_idx % len(base_urls)]
        try:
            df = _fetch_window(session, base, symbol, category, cursor, window_end, limit=limit)
        except requests.exceptions.ReadTimeout:
            base_idx += 1
            time.sleep(sleep_sec)
            continue
        except requests.exceptions.RequestException:
            base_idx += 1
            time.sleep(sleep_sec)
            continue
        except RuntimeError as e:
            return {"symbol": symbol, "ok": False, "error": str(e), "rows": 0, "batches": batches}

        if df.empty:
            cursor = window_end + 1
            batches += 1
            # процент прогресса (даже если пусто)
            processed_ms = (cursor - start_ms)
            pct = min(100.0, (processed_ms / total_ms) * 100.0)
            if _should_log_pct(pct_prev, pct, progress_every_pct):
                print(f"[PCT-FWD] {symbol}: {pct:.1f}% done (latest≈{pd.to_datetime(cursor-60_000, unit='ms', utc=True)})",
                      flush=True)
                pct_prev = pct
            time.sleep(sleep_sec)
            continue

        added = _merge_save(parquet_dir, symbol, df)
        got += int(added)

        if got // PROGRESS_CHUNK > milestone:
            milestone = got // PROGRESS_CHUNK
            last_ts = int(df["ts"].iloc[-1])
            last_iso = pd.to_datetime(last_ts, unit="ms", utc=True)
            print(f"[PROGRESS-FWD] {symbol}: +{milestone*PROGRESS_CHUNK:,} new "
                  f"(rows {existing_rows:,} → {existing_rows + milestone*PROGRESS_CHUNK:,}) "
                  f"up to {last_iso}", flush=True)

        last_ts = int(df["ts"].iloc[-1])
        cursor = last_ts + 60_000
        batches += 1

        # процент прогресса
        processed_ms = (cursor - start_ms)
        pct = min(100.0, (processed_ms / total_ms) * 100.0)
        if _should_log_pct(pct_prev, pct, progress_every_pct):
            print(f"[PCT-FWD] {symbol}: {pct:.1f}% done (latest≈{pd.to_datetime(last_ts, unit='ms', utc=True)})",
                  flush=True)
            pct_prev = pct

        time.sleep(sleep_sec)

    if got % PROGRESS_CHUNK != 0 and got > 0:
        last_iso = pd.to_datetime(cursor - 60_000, unit="ms", utc=True)
        print(f"[PROGRESS-FWD] {symbol}: +{got:,} new (rows {existing_rows:,} → {existing_rows + got:,}) "
              f"up to {last_iso}", flush=True)

    note = ("append" if max_old_ts >= 0 else "bootstrap")
    return {"symbol": symbol, "ok": True, "error": "", "rows": int(got), "batches": batches, "note": note}


def fetch_symbol(symbol: str, category: str, days: int, backfill_days: int, parquet_dir: str,
                 session: requests.Session,
                 limit: int, window_minutes: int, sleep_sec: float,
                 base_urls: list[str], progress_every_pct: float) -> dict:
    """
    Полный цикл: (опционально) бэкфилл «в прошлое» + догонка «вперёд».
    """
    back = _backfill_early(
        symbol=symbol, category=category, parquet_dir=parquet_dir,
        session=session, limit=limit, window_minutes=window_minutes, sleep_sec=sleep_sec,
        base_urls=base_urls, backfill_days=backfill_days, progress_every_pct=progress_every_pct
    )
    if not back["ok"]:
        return back

    fwd = _append_forward(
        symbol=symbol, category=category, days=days, parquet_dir=parquet_dir,
        session=session, limit=limit, window_minutes=window_minutes, sleep_sec=sleep_sec,
        base_urls=base_urls, progress_every_pct=progress_every_pct
    )
    return fwd


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="", help="Список символов через запятую (BTCUSDT,ETHUSDT). Если не указан — берём из config.TRADE_UNIVERSE.")
    ap.add_argument("--days", type=int, default=180, help="Если файл отсутствует: качаем последние N дней (bootstrap). Если файл есть: игнорируется (берём от последней минуты).")
    ap.add_argument("--backfill-days", type=int, default=0, help="Если файл уже есть: докачать 'в прошлое' на N дней до текущего min(ts).")
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

    # прогресс-логи
    ap.add_argument("--progress-every-pct", type=float, default=5.0,
                    help="Шаг процента для логов прогресса по символу (например, 5 = каждые 5%). 0 = отключить процентные логи.")

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
            backfill_days=args.backfill_days,
            parquet_dir=args.out,
            session=session,
            limit=min(1000, max(100, int(args.limit))),
            window_minutes=max(60, int(args.window_min)),
            sleep_sec=max(0.05, float(args.sleep)),
            base_urls=base_urls,
            progress_every_pct=max(0.0, float(args.progress_every_pct))
        )
        results.append(info)
        status = "OK" if info["ok"] else "ERR"
        note = f" {info.get('note','')}".strip()
        print(f"[{status}] {sym} rows_added={info.get('rows',0)} batches={info.get('batches',0)}"
              f"{(' ' + info.get('error','')) if not info['ok'] else ''}{(' ' + note) if note else ''}",
              flush=True)

    print(json.dumps({"summary": results}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()