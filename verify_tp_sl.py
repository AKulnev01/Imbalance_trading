#!/usr/bin/env python3
# verify_tp_sl.py
# Проверяет по публичным свечам Bybit, что для каждой сделки в отчёте
# первым сработало именно то, что записано в колонке exit_reason (tp/sl/timeout),
# с честным разрешением спорных 4h-баров на низком таймфрейме (1m/5m).

# python verify_tp_sl.py "/Users/tema/Documents/отчеты/signals_eval_eval.xlsx" \
#   --interval 240 \
#   --ttl-days 5 \
#   --category linear \
#   --tp-priority sl \
#   --intrabar "1,5" \
#   --intrabar-lookback-days 14

import argparse
import os
import sys
import math
import time
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional

import requests
import pandas as pd

BYBIT_MAIN = "https://api.bybit.com"
BYBIT_TEST = "https://api-demo.bybit.com"

# Bybit v5 kline intervals: "1","3","5","15","30","60","120","240","360","720","D","W","M"
VALID_INTERVALS = {"1","3","5","15","30","60","120","240","360","720","D","W","M"}

def parse_args():
    p = argparse.ArgumentParser(description="Verify TP/SL hits against Bybit candles for your eval report.")
    p.add_argument("excel_path", help="Path to signals_eval.xlsx (лист results).")
    p.add_argument("--sheet", default=None, help="Sheet name (default: 'results' if exists, иначе первый).")
    p.add_argument("--interval", default="240", help="Bybit kline interval (default 240=4h).")
    p.add_argument("--ttl-days", type=int, default=5, help="TTL в днях для окна проверки после t_start (default 5).")
    p.add_argument("--category", default="linear", choices=["spot","linear","inverse","option"],
                   help="Bybit category (default: linear).")
    p.add_argument("--testnet", action="store_true", help="Use Bybit testnet endpoints (public).")
    p.add_argument("--max-rows", type=int, default=0, help="Проверить только первые N строк (0 = все).")
    p.add_argument("--tp-priority", choices=["tp","sl"], default="sl",
                   help="Если TP и SL в одном (LTF) баре: что считать «раньше» (default sl — консервативно).")
    # Intrabar
    p.add_argument("--intrabar", default="1,5",
                   help="Список LTF минутных интервалов, по порядку попыток (например '1,5').")
    p.add_argument("--intrabar-lookback-days", type=int, default=14,
                   help="Резервный lookback (в днях) для подкачки LTF, если окно короткое (default 14).")
    p.add_argument("--output", default=None, help="Путь к CSV отчёту (default: рядом с excel, *_tp_sl_verification.csv).")
    return p.parse_args()

def to_utc_ts_ms(dt_val):
    if pd.isna(dt_val):
        return None
    if isinstance(dt_val, pd.Timestamp):
        ts = dt_val
    elif isinstance(dt_val, datetime):
        ts = pd.Timestamp(dt_val, tz="UTC")
    else:
        try:
            ts = pd.to_datetime(dt_val, utc=True)
        except Exception:
            return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)

def from_ms(ms):
    if ms is None or (isinstance(ms, float) and math.isnan(ms)):
        return pd.NaT
    return pd.to_datetime(int(ms), unit="ms", utc=True)

def _base_url(testnet: bool) -> str:
    return BYBIT_TEST if testnet else BYBIT_MAIN

def fetch_klines(symbol: str,
                 start_ms: int,
                 end_ms: int,
                 interval: str = "240",
                 category: str = "linear",
                 testnet: bool = False,
                 limit: int = 1000) -> List[dict]:
    """
    Возвращает список баров (dict): start(open time ms), open, high, low, close, volume, confirm
    """
    base = _base_url(testnet)
    url = base + "/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": str(interval),
        "start": str(start_ms),
        "end": str(end_ms),
        "limit": str(limit)
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if int(data.get("retCode", -1)) != 0:
            raise RuntimeError(f"retCode={data.get('retCode')} retMsg={data.get('retMsg')}")
        rows = (data.get("result") or {}).get("list") or []
        bars = []
        for it in rows:
            if isinstance(it, dict):
                start = int(it.get("start"))
                o = float(it.get("open", 0))
                h = float(it.get("high", 0))
                l = float(it.get("low", 0))
                c = float(it.get("close", 0))
                v = float(it.get("volume", 0) or 0)
                conf = bool(it.get("confirm", True))
            else:
                start = int(it[0])
                o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4])
                v = float(it[5]) if len(it) > 5 else 0.0
                conf = True
            bars.append({"start": start, "open": o, "high": h, "low": l, "close": c, "volume": v, "confirm": conf})
        bars.sort(key=lambda x: x["start"])
        return bars
    except Exception as e:
        print(f"[fetch_klines] {symbol} {interval}: {e}", file=sys.stderr)
        return []

def _hits_in_bar(side: str, hi: float, lo: float, tp: float, sl: float) -> Tuple[bool, bool]:
    side = (side or "").upper()
    if side == "BUY":
        return (hi >= tp, lo <= sl)
    elif side == "SELL":
        return (lo <= tp, hi >= sl)
    return (False, False)

def resolve_intrabar(symbol: str,
                     side: str,
                     entry_time_ms: int,
                     bar_close_ms: int,
                     tp: float,
                     sl: float,
                     category: str,
                     testnet: bool,
                     intrabar_list: List[int],
                     priority: str,
                     lookback_days_fallback: int) -> Tuple[str, Optional[int]]:
    """
    Честное разрешение порядка TP/SL внутри спорного 4h-интервала:
      1) Грузим LTF бары в диапазоне (entry_time_ms; bar_close_ms]
      2) Идём по времени: если в LTF-баре ударили и TP, и SL — возвращаем приоритет (по флагу).
      3) Если только TP — 'tp', только SL — 'sl'.
      4) Если LTF недоступен/пусто — fallback: считаем 'sl' (консервативно) либо 'tp' (если priority='tp').
    """
    start_ms = entry_time_ms
    end_ms   = bar_close_ms

    # подстраховка: иногда окно меньше 1 бара → возьмём запас lookback дней
    if end_ms is None or start_ms is None or end_ms <= start_ms:
        end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

    for iv in intrabar_list:
        bars = fetch_klines(symbol, start_ms, end_ms, interval=str(iv), category=category, testnet=testnet)
        if not bars:
            continue

        for b in bars:
            hi = float(b["high"]); lo = float(b["low"])
            tp_hit, sl_hit = _hits_in_bar(side, hi, lo, tp, sl)

            if tp_hit and sl_hit:
                # даже внутри LTF-баре не знаем точный порядок → используем приоритет
                return (priority, int(b["start"]))
            if tp_hit:
                return ("tp", int(b["start"]))
            if sl_hit:
                return ("sl", int(b["start"]))

        # если ни на одном LTF-баре не зацепили — пробуем другой интервал из списка
    # окончательный fallback
    return (priority, end_ms)

def first_hit_with_intrabar(side: str,
                            tp: float,
                            sl: float,
                            bars_4h: List[dict],
                            symbol: str,
                            entry_ms: int,
                            category: str,
                            testnet: bool,
                            intrabar_list: List[int],
                            priority: str,
                            lookback_days_fallback: int) -> Tuple[str, Optional[int]]:
    """
    Проход по 4h барам. В спорном баре (и TP, и SL) — уходим на LTF и выясняем порядок.
    Возвращает ('tp'|'sl'|'none', ms_time)
    """
    last_ms = entry_ms
    for b in bars_4h:
        hi = float(b["high"]); lo = float(b["low"])
        tp_hit, sl_hit = _hits_in_bar(side, hi, lo, tp, sl)

        if tp_hit and sl_hit:
            # спорный 4h -> уходим в LTF между last_ms и b["start"] или концом бара (Bybit kline start = open time)
            # На v5 старт бара = open time. Для окна берём (last_ms; next_open_ms]
            # Здесь у нас есть текущий 4h бар b со start=open_ms. Нам нужен конец этого 4h бара:
            # вычислим как start + 4h (но безопаснее взять следующий старт, если есть). У нас его нет — считаем +4h.
            bar_open_ms = int(b["start"])
            bar_close_ms = bar_open_ms + 4*60*60*1000  # 4h
            hit, hit_ms = resolve_intrabar(
                symbol=symbol, side=side,
                entry_time_ms=last_ms,
                bar_close_ms=bar_close_ms,
                tp=tp, sl=sl, category=category, testnet=testnet,
                intrabar_list=intrabar_list, priority=priority,
                lookback_days_fallback=lookback_days_fallback
            )
            return (hit, hit_ms)

        if tp_hit:
            return ("tp", int(b["start"]))
        if sl_hit:
            return ("sl", int(b["start"]))
        last_ms = int(b["start"])
    return ("none", None)

def main():
    args = parse_args()

    if args.interval not in VALID_INTERVALS:
        print(f"⚠️ interval {args.interval} неизвестен Bybit. Используй одно из: {sorted(VALID_INTERVALS)}")
        sys.exit(2)

    # читаем Excel
    xls = pd.ExcelFile(args.excel_path)
    sheet = args.sheet
    if sheet is None:
        sheet = "results" if "results" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(args.excel_path, sheet_name=sheet)

    # нормализуем даты/поля
    for col in ("t_start", "imb_time", "close_time", "exit_time", "as_of"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # обязательные поля
    need_cols = ["symbol","type","t_start","stop_eval","tp_eval","exit_reason","close_time"]
    for c in need_cols:
        if c not in df.columns:
            print(f"⚠️ В листе нет обязательной колонки: {c}")
            sys.exit(3)

    # список LTF интервалов (в минутах)
    intrabar_list = []
    for tok in (args.intrabar or "1,5").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            intrabar_list.append(int(tok))
        except Exception:
            pass
    if not intrabar_list:
        intrabar_list = [1, 5]

    # ограничение по строкам
    work = df.copy()
    if args.max_rows and args.max_rows > 0:
        work = work.head(args.max_rows).copy()

    now_utc = pd.Timestamp.now(tz="UTC")
    results = []
    checked = 0
    matched = 0

    for i, row in work.iterrows():
        sym   = str(row.get("symbol"))
        side  = str(row.get("type","")).upper()
        t0    = row.get("t_start")   # момент входа
        sl    = row.get("stop_eval")
        tp    = row.get("tp_eval")
        exit_reason_reported = str(row.get("exit_reason") or "").lower()
        close_time_reported  = row.get("close_time")

        if pd.isna(t0):
            results.append({
                "row": int(i),
                "symbol": sym, "side": side,
                "t_start": pd.NaT,
                "tp_eval": tp, "stop_eval": sl,
                "first_hit": "none",
                "first_hit_time": pd.NaT,
                "reported_exit_reason": exit_reason_reported,
                "reported_close_time": close_time_reported,
                "match": (exit_reason_reported in ("timeout_no_fill","") or pd.isna(close_time_reported)),
                "note": "no_fill_in_report"
            })
            continue

        # уровни должны быть числами
        try:
            tp = float(tp); sl = float(sl)
        except Exception:
            results.append({
                "row": int(i),
                "symbol": sym, "side": side,
                "t_start": t0,
                "tp_eval": tp, "stop_eval": sl,
                "first_hit": "none",
                "first_hit_time": pd.NaT,
                "reported_exit_reason": exit_reason_reported,
                "reported_close_time": close_time_reported,
                "match": False,
                "note": "tp/sl empty"
            })
            continue

        entry_ms = to_utc_ts_ms(t0)
        end_wall = min(t0 + timedelta(days=int(args.ttl_days)), now_utc)
        end_ms = to_utc_ts_ms(end_wall)

        # 4h бары в окне (t0; t0+TTL]
        bars = fetch_klines(sym, entry_ms, end_ms, interval=args.interval,
                            category=args.category, testnet=args.testnet)

        # основной хит с LTF-разрешением спорных баров
        hit, hit_ms = first_hit_with_intrabar(
            side=side, tp=tp, sl=sl, bars_4h=bars, symbol=sym,
            entry_ms=entry_ms, category=args.category, testnet=args.testnet,
            intrabar_list=intrabar_list, priority=args.tp_priority,
            lookback_days_fallback=int(args.intrabar_lookback_days)
        )
        hit_time = from_ms(hit_ms)

        # нормализуем, как в отчёте
        reported = exit_reason_reported
        if reported in ("timeout_last_close","timeout_no_fill","unknown") or pd.isna(close_time_reported):
            reported_norm = "none"
        elif reported in ("tp","take","take_profit"):
            reported_norm = "tp"
        elif reported in ("sl","stop","stop_loss"):
            reported_norm = "sl"
        else:
            reported_norm = reported

        is_match = (reported_norm == hit)
        if is_match:
            matched += 1
        checked += 1

        results.append({
            "row": int(i),
            "symbol": sym,
            "side": side,
            "t_start": t0,
            "tp_eval": tp,
            "stop_eval": sl,
            "first_hit": hit,
            "first_hit_time": hit_time,
            "reported_exit_reason": reported,
            "reported_close_time": close_time_reported,
            "match": bool(is_match),
            "note": "" if is_match else f"mismatch: report={reported_norm}, bybit={hit}"
        })

        time.sleep(0.03)  # не душим API

    out_df = pd.DataFrame(results)
    if args.output:
        out_path = args.output
    else:
        base, _ = os.path.splitext(os.path.abspath(args.excel_path))
        out_path = base + "_tp_sl_verification.csv"

    out_df.to_csv(out_path, index=False)
    acc = (matched / checked * 100.0) if checked else 0.0
    print(f"✅ проверено строк: {checked}, совпадений: {matched} ({acc:.2f}%).")
    print(f"💾 отчёт: {out_path}")

if __name__ == "__main__":
    try:
        import pandas as pd  # noqa
    except Exception:
        print("Установи зависимости: pip install pandas requests openpyxl", file=sys.stderr)
        sys.exit(1)
    main()