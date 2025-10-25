# main.py
# ===================== ДОСТУПНЫЕ КОМАНДЫ =====================
# Консоль и Telegram:
#
# /help — список команд
#
# /bulk_closed <days> [interval] [max_fill_days]
#   → отчёт по перекрытым FVG (filled==True), max_days_to_fill — ограничение на "через сколько дней закрыт"
#
# /bulk_open <days> [interval]
#   → отчёт по свежим FVG (не важно touched/filled)
#
# /bulk_all <days> [interval]
#   → отчёт по ВСЕМ FVG (filled и не filled), считает винрейт
#
# /missed_signals <hours> [interval]
#   → свежие FVG за последние N часов (любые сигналы)
#
# /missed_trades <hours> [interval]
#   → УПУЩЕННЫЕ сделки: были имб, но entry не задет
#
# /active_waiting [lookback_days] [interval]
#   → актуальные имб, которые ждут касания entry
#
# Тест API (Bybit):
# /probe_limit <symbol> <side> <price> [usd] [tp] [sl] [tif]
# /probe_market <symbol> <side> [usd] [tp] [sl]
# /probe_prices <symbol> — сверка lastPrice mainnet vs testnet
#
# Примеры:
#   python main.py bulk_closed 90 4h 7
#   python main.py bulk_all 360 4h
#   python main.py missed_trades 48 4h
#   python main.py active_waiting 7 4h
# ==============================================================

import os
import sys
import time
import logging
import threading
import datetime
import datetime as _dt
from typing import List, Optional

import pandas as pd
from telegram import Bot, Update
from telegram.ext import (
    Updater, CommandHandler, MessageHandler,
    Filters, CallbackContext
)

from config import (
    TELEGRAM_TOKEN, CHAT_ID,
    MAX_FILL_DAYS,                 # для closed-отчёта
    TRADE_UNIVERSE,                # фиксированный список монет (как в бою)
    DEFAULT_MIN_STRENGTH,          # мин. сила FVG
    RISK_REWARD_RATIO,             # для вычисления TP
    ENTRY_MODE,
    MOMENTUM_TP_PCT,
    MOMENTUM_SL_PCT,
)

# УБРАНО: сетевой импорт get_bybit_klines
# from utils.fetch_data import get_bybit_klines
from utils.detect_fvg import detect_fvg_imbalances
from utils.evaluate_imbalances import evaluate_imbalances
from utils.strategy import scan_universe, get_klines_4h, filter_universe_to_local
from utils.symbols import fetch_top_symbols

# === Bybit тестовые вызовы ===
from utils.bybit_trade import (
    create_order, usd_to_qty, open_position_market,
)
from utils.bybit_trade import get_dual_prices  # для /probe_prices

# ===================== Логирование =====================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

# === ДЕФОЛТНЫЕ ТФ ===
# теперь bulk == live по умолчанию
DEFAULT_LIVE_INTERVAL   = os.getenv("DEFAULT_LIVE_INTERVAL", "4h")
DEFAULT_BULK_INTERVAL   = os.getenv("DEFAULT_BULK_INTERVAL", DEFAULT_LIVE_INTERVAL)
DEFAULT_MISSED_INTERVAL = os.getenv("DEFAULT_MISSED_INTERVAL", DEFAULT_LIVE_INTERVAL)

# === Параметры детектора (как в бою) ===
VOL_MULT = float(os.getenv("FVG_VOL_MULT", "1.5"))
TOL_PCT  = float(os.getenv("FVG_TOLERANCE_PCT", "0"))

# TG-бот
bot = Bot(token=TELEGRAM_TOKEN)

# ---------- утилита для очистки df ----------
def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in ("open","high","low","close","volume","turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    df = df[~df.index.isna()]
    return df

def _ensure_tznaive_inplace(df: pd.DataFrame, cols: List[str]):
    """Excel не поддерживает tz-aware; делаем tz-naive перед сохранением."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_localize(None)

def _get_ohlc_since(symbol: str, interval: str, start_ts: pd.Timestamp, lookback_days_fallback: int = 7) -> pd.DataFrame:
    """
    Возвращает OHLCV с индексом по времени (UTC) НЕ РАНЬШЕ start_ts.
    С учётом локального режима: берём 4h через get_klines_4h() и фильтруем по времени.
    """
    lb_days = max(1, lookback_days_fallback)
    try:
        df = get_klines_4h(symbol=symbol, lookback_days=lb_days, interval=interval or "4h")
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = _sanitize_ohlcv(df)
    start_ts = pd.to_datetime(start_ts, utc=True, errors="coerce")
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    return df[df.index >= start_ts]

# ===================== BULK-ОТЧЁТЫ =====================

def _universe() -> List[str]:
    """Всегда как в бою. Если пуст — fallback к TOP100."""
    u = list(dict.fromkeys(TRADE_UNIVERSE))
    return u if u else fetch_top_symbols()[:100]

def bulk_closed_report(days: int = 90, interval: Optional[str] = None, max_days_to_fill: Optional[int] = None) -> str:
    """
    ИСТОРИКА: только перекрытые FVG за период (filled==True) с ограничением по дням перекрытия.
    Возвращает путь к Excel.
    """
    import datetime as _dt
    interval = interval or DEFAULT_BULK_INTERVAL
    if max_days_to_fill is None:
        max_days_to_fill = MAX_FILL_DAYS

    symbols = _universe()
    symbols = filter_universe_to_local(symbols)
    print(f"🔄 CLOSED scan: {len(symbols)} sym, {days}d, TF={interval}, fill ≤ {max_days_to_fill}d")

    rows = []
    for sym in symbols:
        try:
            df = get_klines_4h(symbol=sym, lookback_days=days, interval=interval)
            df = _sanitize_ohlcv(df)
            if df is None or df.empty:
                print(f"   → {sym}: пусто, пропуск.")
                continue

            imbs = detect_fvg_imbalances(
                df,
                volume_multiplier=VOL_MULT,
                tolerance_pct=TOL_PCT,
                min_strength_pct=DEFAULT_MIN_STRENGTH,
                max_days_to_fill=max_days_to_fill,
            )

            added = 0
            for imb in imbs:
                if not bool(imb.get("filled")):
                    continue
                dfill = imb.get("days_to_fill")
                if dfill is None:
                    continue
                if int(dfill) > int(max_days_to_fill):
                    continue

                side = str(imb["type"]).upper()
                entry = float(imb["low2"] if side == "BUY" else imb["high2"])

                if ENTRY_MODE == "MOMENTUM":
                    # TP/SL из .env
                    if side == "BUY":
                        stop = float(entry) * (1.0 - float(MOMENTUM_SL_PCT))
                        tp = float(entry) * (1.0 + float(MOMENTUM_TP_PCT))
                    else:  # SELL
                        stop = float(entry) * (1.0 + float(MOMENTUM_SL_PCT))
                        tp = float(entry) * (1.0 - float(MOMENTUM_TP_PCT))
                else:
                    # Старый подход через RR для ретест/брейкаут отчётов
                    if side == "BUY":
                        stop = float(imb["low2"]) * 0.998
                        tp = float(entry) + (float(entry) - float(stop)) * float(RISK_REWARD_RATIO)
                    else:
                        stop = float(imb["high2"]) * 1.002
                        tp = float(entry) - (float(stop) - float(entry)) * float(RISK_REWARD_RATIO)

                t0 = pd.to_datetime(imb["time"], utc=True)
                fill_time = t0 + pd.Timedelta(days=int(dfill))

                rows.append({
                    "symbol":       sym,
                    "imb_time":     t0,
                    "type":         side,
                    "entry":        entry,
                    "stop":         float(stop),
                    "tp":           float(tp),
                    "strength":     float(imb.get("strength", 0.0)),
                    "filled":       True,
                    "days_to_fill": int(dfill),
                    "fill_time":    fill_time,
                })
                added += 1
            print(f"   → {sym}: +{added}")

        except Exception as e:
            print(f"⚠️ Ошибка по {sym}: {e}")

    df_out = pd.DataFrame(rows)
    _ensure_tznaive_inplace(df_out, ["imb_time", "fill_time"])

    reports_dir = os.path.expanduser("~/Documents/отчеты")
    os.makedirs(reports_dir, exist_ok=True)
    date_str = _dt.datetime.now().strftime("%Y%m%d")
    path = os.path.join(reports_dir, f"signals_closed_{len(symbols)}sym_{days}d_{interval}_fill≤{max_days_to_fill}_{date_str}.xlsx")
    df_out.to_excel(path, index=False)
    print(f"✅ Closed report saved: {path}")
    return path


def bulk_open_report(days: int = 30, interval: Optional[str] = None) -> str:
    """
    ОТКРЫТЫЕ (свежие) FVG за период — без требования filled/max_days_to_fill.
    Возвращает путь к Excel.
    """
    import datetime as _dt
    interval = interval or DEFAULT_BULK_INTERVAL

    symbols = _universe()
    symbols = filter_universe_to_local(symbols)
    print(f"🔄 OPEN scan: {len(symbols)} sym, {days}d, TF={interval}")

    rows = []
    for sym in symbols:
        try:
            df = get_klines_4h(symbol=sym, lookback_days=days, interval=interval)
            df = _sanitize_ohlcv(df)
            if df is None or df.empty:
                print(f"   → {sym}: пустые данные, пропуск.")
                continue

            imbs = detect_fvg_imbalances(
                df,
                volume_multiplier=VOL_MULT,
                tolerance_pct=TOL_PCT,
                min_strength_pct=DEFAULT_MIN_STRENGTH,
            )

            added = 0
            for imb in imbs:
                side = str(imb["type"]).upper()
                if side not in ("BUY","SELL"):
                    continue

                entry = float(imb["low2"] if side == "BUY" else imb["high2"])
                if ENTRY_MODE == "MOMENTUM":
                    # TP/SL из .env
                    if side == "BUY":
                        stop = float(entry) * (1.0 - float(MOMENTUM_SL_PCT))
                        tp = float(entry) * (1.0 + float(MOMENTUM_TP_PCT))
                    else:  # SELL
                        stop = float(entry) * (1.0 + float(MOMENTUM_SL_PCT))
                        tp = float(entry) * (1.0 - float(MOMENTUM_TP_PCT))
                else:
                    # Старый подход через RR для ретест/брейкаут отчётов
                    if side == "BUY":
                        stop = float(imb["low2"]) * 0.998
                        tp = float(entry) + (float(entry) - float(stop)) * float(RISK_REWARD_RATIO)
                    else:
                        stop = float(imb["high2"]) * 1.002
                        tp = float(entry) - (float(stop) - float(entry)) * float(RISK_REWARD_RATIO)

                t0 = pd.to_datetime(imb["time"], utc=True)

                rows.append({
                    "symbol":       sym,
                    "imb_time":     t0,
                    "type":         side,
                    "entry":        entry,
                    "stop":         float(stop),
                    "tp":           float(tp),
                    "strength":     float(imb.get("strength", 0.0)),
                    "touched":      bool(imb.get("touched", False)),
                    "filled":       bool(imb.get("filled", False)),
                    "days_to_fill": imb.get("days_to_fill"),
                })
                added += 1
            print(f"   → {sym}: +{added}")

        except Exception as e:
            print(f"⚠️ Ошибка по {sym}: {e}")

    df_out = pd.DataFrame(rows)
    _ensure_tznaive_inplace(df_out, ["imb_time"])

    reports_dir = os.path.expanduser("~/Documents/отчеты")
    os.makedirs(reports_dir, exist_ok=True)
    date_str = _dt.datetime.now().strftime("%Y%m%d")
    path = os.path.join(reports_dir, f"signals_open_{len(symbols)}sym_{days}d_{interval}_{date_str}.xlsx")
    df_out.to_excel(path, index=False)
    print(f"✅ Open report saved: {path}")
    return path


def bulk_all_report(days: int = 360, interval: Optional[str] = None) -> str:
    """
    Отчёт по ВСЕМ имбалансам за период (filled и не filled) с базовым винрейт по filled.
    Возвращает путь к Excel.
    """
    import datetime as _dt
    interval = interval or DEFAULT_BULK_INTERVAL

    symbols = _universe()
    symbols = filter_universe_to_local(symbols)
    print(f"🔄 ALL scan: {len(symbols)} sym, {days}d, TF={interval}")

    rows = []
    for sym in symbols:
        try:
            df = get_klines_4h(symbol=sym, lookback_days=days, interval=interval)
            df = _sanitize_ohlcv(df)
            if df is None or df.empty:
                print(f"   → {sym}: пусто, пропуск.")
                continue

            imbs = detect_fvg_imbalances(
                df,
                volume_multiplier=VOL_MULT,
                tolerance_pct=TOL_PCT,
                min_strength_pct=DEFAULT_MIN_STRENGTH,
            )
            if not imbs:
                continue

            evaluated = evaluate_imbalances(df, imbs, max_days=MAX_FILL_DAYS)

            for imb in evaluated.to_dict("records"):
                side = str(imb["type"]).upper()
                if side not in ("BUY", "SELL"):
                    continue

                entry = float(imb["low2"] if side == "BUY" else imb["high2"])
                if ENTRY_MODE == "MOMENTUM":
                    # TP/SL из .env
                    if side == "BUY":
                        stop = float(entry) * (1.0 - float(MOMENTUM_SL_PCT))
                        tp = float(entry) * (1.0 + float(MOMENTUM_TP_PCT))
                    else:  # SELL
                        stop = float(entry) * (1.0 + float(MOMENTUM_SL_PCT))
                        tp = float(entry) * (1.0 - float(MOMENTUM_TP_PCT))
                else:
                    # Старый подход через RR для ретест/брейкаут отчётов
                    if side == "BUY":
                        stop = float(imb["low2"]) * 0.998
                        tp = float(entry) + (float(entry) - float(stop)) * float(RISK_REWARD_RATIO)
                    else:
                        stop = float(imb["high2"]) * 1.002
                        tp = float(entry) - (float(stop) - float(entry)) * float(RISK_REWARD_RATIO)

                t0 = pd.to_datetime(imb["time"], utc=True)

                rows.append({
                    "symbol":       sym,
                    "imb_time":     t0,
                    "type":         side,
                    "entry":        entry,
                    "stop":         float(stop),
                    "tp":           float(tp),
                    "strength":     float(imb.get("strength", 0.0)),
                    "filled":       bool(imb.get("filled", False)),
                    "days_to_fill": imb.get("days_to_fill"),
                })
            print(f"   → {sym}: +{len(evaluated)}")

        except Exception as e:
            print(f"⚠️ Ошибка по {sym}: {e}")

    if not rows:
        print("❌ Нет данных.")
        return ""

    df_out = pd.DataFrame(rows)
    _ensure_tznaive_inplace(df_out, ["imb_time"])

    total = len(df_out)
    filled_count = int(df_out["filled"].sum()) if "filled" in df_out.columns else 0
    accuracy = (filled_count / total * 100.0) if total else 0.0
    print(f"\n📊 Winrate (filled/total) = {accuracy:.2f}%  ({filled_count}/{total})")

    reports_dir = os.path.expanduser("~/Documents/отчеты")
    os.makedirs(reports_dir, exist_ok=True)
    date_str = _dt.datetime.now().strftime("%Y%m%d")
    path = os.path.join(reports_dir, f"signals_all_{len(symbols)}sym_{days}d_{interval}_{date_str}.xlsx")
    df_out.to_excel(path, index=False)
    print(f"✅ All report saved: {path}")
    return path


# ===================== Missed / Active =====================
def check_signal_touches(df, entry, stop, tp, side):
    """
    Проверяет, был ли вход (entry), тейк (tp) или стоп (stop) затронут в котировках
    df: DataFrame с колонками ['open','high','low','close']
    side: 'BUY' или 'SELL'
    """
    entry_hit, tp_hit, stop_hit = False, False, False

    for _, row in df.iterrows():
        if side == 'BUY':
            if not entry_hit and row['low'] <= entry <= row['high']:
                entry_hit = True
            if entry_hit and row['high'] >= tp:
                tp_hit = True
                break
            if entry_hit and row['low'] <= stop:
                stop_hit = True
                break

        elif side == 'SELL':
            if not entry_hit and row['low'] <= entry <= row['high']:
                entry_hit = True
            if entry_hit and row['low'] <= tp:
                tp_hit = True
                break
            if entry_hit and row['high'] >= stop:
                stop_hit = True
                break

    return entry_hit, tp_hit, stop_hit

def missed_signals(hours: int, interval: Optional[str] = None) -> str:
    import math
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    since_time = now - timedelta(hours=int(hours))
    symbols = _universe()
    symbols = filter_universe_to_local(symbols)
    lookback_days = max(1, math.ceil(int(hours) / 24))
    interval = interval or DEFAULT_MISSED_INTERVAL

    print(f"🔎 Missed-scan: {len(symbols)} sym, window={hours}h, since={since_time} UTC, TF={interval}")
    df = scan_universe(universe=symbols, lookback_days=lookback_days, mode="all", interval=interval)
    if df is None:
        df = pd.DataFrame()

    tcol = "imb_time" if "imb_time" in df.columns else ("fill_time" if "fill_time" in df.columns else None)
    if tcol and not df.empty:
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None)
        since_time_naive = pd.to_datetime(since_time).tz_localize(None)
        df = df[df[tcol] >= since_time_naive]

    out_dir = os.path.expanduser("~/Documents/отчеты")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"missed_signals_{hours}h_{interval}_{now.strftime('%Y%m%d_%H%M')}.xlsx")

    safe = df.copy()
    _ensure_tznaive_inplace(safe, ["time","imb_time","fill_time","close_time","filled_at"])

    with pd.ExcelWriter(out_path) as xw:
        safe.to_excel(xw, index=False, sheet_name="data")
        pd.DataFrame({
            "param": ["hours","interval","symbols","generated_utc"],
            "value": [hours, interval, len(symbols), now.strftime("%Y-%m-%d %H:%M:%S")]
        }).to_excel(xw, index=False, sheet_name="meta")

    print(f"💾 Saved: {out_path} (rows={len(safe)})")
    return out_path


def missed_trades(hours: int, interval: Optional[str] = None) -> str:
    """
    Упущенные сделки: имб был, но ФАКТИЧЕСКИ entry НЕ задет (по свечам).
    """
    import math
    from datetime import datetime, timedelta
    from config import ENABLE_BUY, ENABLE_SELL

    now = datetime.utcnow()
    since_time = now - timedelta(hours=int(hours))
    interval = interval or DEFAULT_LIVE_INTERVAL

    symbols = _universe()
    symbols = filter_universe_to_local(symbols)
    lookback_days = max(1, math.ceil(int(hours) / 24) + 2)

    print(f"🔎 Missed-trades: {len(symbols)} sym, window={hours}h, since={since_time} UTC, TF={interval}")
    df_sig = scan_universe(universe=symbols, lookback_days=lookback_days, mode="live", interval=interval)
    if df_sig is None:
        df_sig = pd.DataFrame()

    tcol = "imb_time" if "imb_time" in df_sig.columns else ("fill_time" if "fill_time" in df_sig.columns else None)
    if tcol and not df_sig.empty:
        df_sig[tcol] = pd.to_datetime(df_sig[tcol], utc=True, errors="coerce")
        df_sig = df_sig[df_sig[tcol] >= pd.to_datetime(since_time, utc=True)]

    if "type" in df_sig.columns and not df_sig.empty:
        df_sig["type"] = df_sig["type"].str.upper()
        if not ENABLE_BUY:
            df_sig = df_sig[df_sig["type"] != "BUY"]
        if not ENABLE_SELL:
            df_sig = df_sig[df_sig["type"] != "SELL"]

    rows = []
    for _, r in df_sig.iterrows():
        sym   = str(r["symbol"])
        side  = str(r["type"]).upper()
        entry = r.get("entry")
        stop  = r.get("stop") or r.get("stop_eval")
        tp    = r.get("tp")   or r.get("tp_eval")
        t0    = pd.to_datetime(r.get("imb_time") or r.get("time"), utc=True, errors="coerce")

        if side not in ("BUY","SELL") or pd.isna(entry) or pd.isna(t0):
            continue

        entry = float(entry); stop = float(stop or 0.0); tp = float(tp or 0.0)

        df_ohlc = _get_ohlc_since(sym, interval, t0, lookback_days_fallback=lookback_days)
        if df_ohlc.empty:
            rows.append({"symbol":sym,"type":side,"imb_time":t0,"entry":entry,"stop":stop,"tp":tp,
                         "entry_hit":False,"tp_hit":False,"stop_hit":False,"note":"no_ohlc"})
            continue

        entry_hit, tp_hit, stop_hit = check_signal_touches(df_ohlc, entry, stop, tp, side)
        if not entry_hit:
            rows.append({"symbol":sym,"type":side,"imb_time":t0,"entry":entry,"stop":stop,"tp":tp,
                         "entry_hit":entry_hit,"tp_hit":tp_hit,"stop_hit":stop_hit})

    df_out = pd.DataFrame(rows).sort_values("imb_time") if rows else pd.DataFrame(
        columns=["symbol","type","imb_time","entry","stop","tp","entry_hit","tp_hit","stop_hit","note"]
    )
    _ensure_tznaive_inplace(df_out, ["imb_time"])

    out_dir = os.path.expanduser("~/Documents/отчеты")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"missed_trades_{hours}h_{interval}_{now.strftime('%Y%m%d_%H%M')}.xlsx")

    with pd.ExcelWriter(out_path) as xw:
        df_out.to_excel(xw, index=False, sheet_name="data")
        pd.DataFrame({
            "param": ["hours","interval","symbols","generated_utc","rows"],
            "value": [hours, interval, len(symbols), now.strftime("%Y-%m-%d %H:%M:%S"), len(df_out)]
        }).to_excel(xw, index=False, sheet_name="meta")

    print(f"💾 Saved: {out_path} (rows={len(df_out)})")
    return out_path


def active_waiting(lookback_days: int = 7, interval: Optional[str] = None) -> str:
    from datetime import datetime, timezone

    interval = interval or DEFAULT_LIVE_INTERVAL
    symbols  = _universe()
    symbols  = filter_universe_to_local(symbols)

    print(f"🔍 Active-waiting (real): {len(symbols)} sym, lookback={lookback_days}d, TF={interval}")
    df_sig = scan_universe(universe=symbols, lookback_days=lookback_days, mode="all", interval=interval)
    if df_sig is None:
        df_sig = pd.DataFrame()
    print(f"   → scan_universe rows: {len(df_sig)}")

    rows = []
    now_utc = datetime.now(timezone.utc)
    for _, r in df_sig.iterrows():
        sym   = str(r.get("symbol"))
        side  = str(r.get("type","")).upper()
        entry = r.get("entry")
        stop  = r.get("stop") or r.get("stop_eval")
        tp    = r.get("tp")   or r.get("tp_eval")
        t0    = pd.to_datetime(r.get("imb_time") or r.get("time"), utc=True, errors="coerce")

        if side not in ("BUY","SELL") or pd.isna(entry) or pd.isna(t0):
            continue

        entry = float(entry); stop = float(stop or 0.0); tp = float(tp or 0.0)
        df_ohlc = _get_ohlc_since(sym, interval, t0, lookback_days_fallback=lookback_days)
        # исключаем сам бар имба
        if not df_ohlc.empty:
            df_ohlc = df_ohlc[df_ohlc.index > t0]

        if df_ohlc.empty:
            entry_hit = tp_hit = stop_hit = False
        else:
            entry_hit, tp_hit, stop_hit = check_signal_touches(df_ohlc, entry, stop, tp, side)

        touched_simple = False
        first_touch_at = None
        time_to_entry_hours = None
        if not df_ohlc.empty:
            for ts, row in df_ohlc.iterrows():
                lo = float(row["low"]); hi = float(row["high"])
                if lo <= entry <= hi:
                    touched_simple = True
                    first_touch_at = pd.to_datetime(ts, utc=True)
                    time_to_entry_hours = round((first_touch_at - t0).total_seconds() / 3600.0, 2)
                    break

        days_waiting = (now_utc - t0).total_seconds() / 86400.0
        rows.append({
            "symbol": sym, "type": side, "imb_time": t0,
            "entry": entry, "stop": stop, "tp": tp,
            "strength": float(r.get("strength", 0.0)),
            "days_waiting": round(days_waiting, 2),
            "time_to_entry_hours": time_to_entry_hours,
            "entry_hit": bool(entry_hit), "tp_hit": bool(tp_hit), "stop_hit": bool(stop_hit),
            "touched_simple": bool(touched_simple), "first_touch_at": first_touch_at,
            "waiting": (not entry_hit),
        })

    df_show = pd.DataFrame(rows).sort_values(
        ["waiting","days_waiting","imb_time"],
        ascending=[False, False, True]
    ) if rows else pd.DataFrame(columns=[
        "symbol","type","imb_time","entry","stop","tp","strength",
        "days_waiting","time_to_entry_hours","entry_hit","tp_hit","stop_hit",
        "touched_simple","first_touch_at","waiting"
    ])

    _ensure_tznaive_inplace(df_show, ["imb_time", "first_touch_at"])

    out_dir = os.path.expanduser("~/Documents/отчеты")
    os.makedirs(out_dir, exist_ok=True)
    ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(out_dir, f"active_waiting_{lookback_days}d_{interval}_{ts}.xlsx")

    with pd.ExcelWriter(out_path) as xw:
        df_show.to_excel(xw, index=False, sheet_name="data")
        pd.DataFrame({
            "param": ["lookback_days","interval","symbols","generated_utc","rows","scan_rows"],
            "value": [lookback_days, interval, len(symbols), ts, len(df_show), len(df_sig)]
        }).to_excel(xw, index=False, sheet_name="meta")

    print(f"💾 Saved: {out_path} (rows={len(df_show)}, scan_rows={len(df_sig)})")
    return out_path



# ===================== Telegram Handlers =====================

HELP_TEXT = (
    "🤖 Доступные команды:\n"
    "/help — список команд\n"
    "/bulk_closed <days> [interval] [max_fill_days] — отчёт по перекрытым FVG\n"
    "/bulk_open <days> [interval] — отчёт по свежим FVG (без filled)\n"
    "/bulk_all <days> [interval] — ВСЕ имб, базовый винрейт\n"
    "/missed_signals <hours> [interval] — свежие FVG за окно\n"
    "/missed_trades <hours> [interval] — упущенные сделки (entry не задет)\n"
    "/active_waiting [lookback_days] [interval] — актуальные имб, ждём касания\n"
    "\n📡 Тест API:\n"
    "/probe_limit <symbol> <side> <price> [usd] [tp] [sl] [tif]\n"
    "/probe_market <symbol> <side> [usd] [tp] [sl]\n"
    "/probe_prices <symbol>\n"
    "\nПримеры:\n"
    "/bulk_closed 90 4h 7\n"
    "/missed_trades 48 4h\n"
    "/active_waiting 7 4h\n"
    "/probe_limit BTCUSDT BUY 60000 50 60600 58800 GTC\n"
    "/probe_market BTCUSDT BUY 50 61200 58800\n"
)

def _parse_args(text: str) -> List[str]:
    parts = (text or "").strip().split()
    return parts[1:] if len(parts) > 1 else []

def tg_send_file(update: Update, context: CallbackContext, path: str, caption: str):
    if not path:
        update.message.reply_text("❌ Файл не сформирован.")
        return
    try:
        with open(path, "rb") as f:
            context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=f,
                filename=os.path.basename(path),
                caption=caption
            )
    except Exception as e:
        update.message.reply_text(f"❌ Не удалось отправить файл: {e}")

def cmd_help(update: Update, context: CallbackContext):
    update.message.reply_text(HELP_TEXT)

def cmd_bulk_closed(update: Update, context: CallbackContext):
    args = _parse_args(update.message.text)
    try:
        days = int(args[0]) if len(args) >= 1 else 90
        interval = args[1] if len(args) >= 2 else None
        mfd = int(args[2]) if len(args) >= 3 else None
        update.message.reply_text(f"🔄 Формирую closed-отчёт: {days}d, TF={interval or DEFAULT_BULK_INTERVAL}, mfd={mfd or MAX_FILL_DAYS}…")
        path = bulk_closed_report(days=days, interval=interval, max_days_to_fill=mfd)
        tg_send_file(update, context, path, "📎 Closed report")
    except Exception as e:
        update.message.reply_text(f"❌ Ошибка: {e}")

def cmd_bulk_open(update: Update, context: CallbackContext):
    args = _parse_args(update.message.text)
    try:
        days = int(args[0]) if len(args) >= 1 else 30
        interval = args[1] if len(args) >= 2 else None
        update.message.reply_text(f"🔄 Формирую open-отчёт: {days}d, TF={interval or DEFAULT_BULK_INTERVAL}…")
        path = bulk_open_report(days=days, interval=interval)
        tg_send_file(update, context, path, "📎 Open report")
    except Exception as e:
        update.message.reply_text(f"❌ Ошибка: {e}")

def cmd_bulk_all(update: Update, context: CallbackContext):
    args = _parse_args(update.message.text)
    try:
        days = int(args[0]) if len(args) >= 1 else 360
        interval = args[1] if len(args) >= 2 else None
        update.message.reply_text(f"🔄 Формирую all-отчёт: {days}d, TF={interval or DEFAULT_BULK_INTERVAL}…")
        path = bulk_all_report(days=days, interval=interval)
        tg_send_file(update, context, path, "📎 All report")
    except Exception as e:
        update.message.reply_text(f"❌ Ошибка: {e}")

def cmd_missed_signals(update: Update, context: CallbackContext):
    args = _parse_args(update.message.text)
    try:
        hours = int(args[0]) if len(args) >= 1 else 24
        interval = args[1] if len(args) >= 2 else None
        update.message.reply_text(f"🔎 Ищу свежие FVG за {hours}ч (TF={interval or DEFAULT_MISSED_INTERVAL})…")
        path = missed_signals(hours=hours, interval=interval)
        tg_send_file(update, context, path, "📎 Missed signals")
    except Exception as e:
        update.message.reply_text(f"❌ Ошибка: {e}")

def cmd_missed_trades(update: Update, context: CallbackContext):
    args = _parse_args(update.message.text)
    try:
        hours = int(args[0]) if len(args) >= 1 else 24
        interval = args[1] if len(args) >= 2 else None
        update.message.reply_text(f"🔎 Ищу УПУЩЕННЫЕ сделки за {hours}ч (TF={interval or DEFAULT_LIVE_INTERVAL})…")
        path = missed_trades(hours=hours, interval=interval)
        tg_send_file(update, context, path, "📎 Missed trades")
    except Exception as e:
        update.message.reply_text(f"❌ Ошибка: {e}")

def cmd_active_waiting(update: Update, context: CallbackContext):
    args = _parse_args(update.message.text)
    try:
        lookback = int(args[0]) if len(args) >= 1 else 7
        interval = args[1] if len(args) >= 2 else None
        update.message.reply_text(f"⏳ Свежие имб, которые ждут касания (lookback={lookback}d, TF={interval or DEFAULT_LIVE_INTERVAL})…")
        path = active_waiting(lookback_days=lookback, interval=interval)
        tg_send_file(update, context, path, "📎 Active waiting")
    except Exception as e:
        update.message.reply_text(f"❌ Ошибка: {e}")

# ===================== PROBE COMMANDS (Bybit) =====================

def _to_float_safe(x, default=None):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

def cmd_probe_prices(update: Update, context: CallbackContext):
    args = _parse_args(update.message.text)
    if len(args) < 1:
        update.message.reply_text("Usage: /probe_prices <symbol>")
        return
    symbol = args[0].upper()
    try:
        px = get_dual_prices(symbol)
        update.message.reply_text(
            f"💹 {symbol} lastPrice:\n• mainnet: {px['mainnet']}\n• testnet: {px['testnet']}"
        )
    except Exception as e:
        update.message.reply_text(f"❌ prices error: {e}")

def cmd_probe_limit(update: Update, context: CallbackContext):
    """
    /probe_limit <symbol> <side> <price> [usd] [tp] [sl] [tif]
    side: BUY/SELL
    tif: GTC | PostOnly (по умолчанию GTC)
    """
    args = _parse_args(update.message.text)
    if len(args) < 3:
        update.message.reply_text("Usage: /probe_limit <symbol> <side BUY|SELL> <price> [usd] [tp] [sl] [tif]")
        return
    symbol = args[0].upper()
    side   = args[1].upper()
    price  = _to_float_safe(args[2])
    if side not in ("BUY","SELL") or price is None:
        update.message.reply_text("❌ side должен быть BUY/SELL, price — число.")
        return

    usd = _to_float_safe(args[3], 25.0) if len(args) >= 4 else 25.0
    tp  = _to_float_safe(args[4]) if len(args) >= 5 else None
    sl  = _to_float_safe(args[5]) if len(args) >= 6 else None
    tif = str(args[6]).upper() if len(args) >= 7 else "GTC"

    try:
        qty = usd_to_qty(symbol, usd, price=price)
        kwargs = {}
        if tif == "POSTONLY" or tif == "POST_ONLY":
            kwargs["timeInForce"] = "PostOnly"
        else:
            kwargs["timeInForce"] = "GTC"

        res = create_order(
            symbol=symbol,
            side=side,
            order_type="Limit",
            qty=str(qty),
            price=str(price),
            take_profit=str(tp) if tp is not None else None,
            stop_loss=str(sl) if sl is not None else None,
            reduce_only=False,
            order_link_id=f"probe_{symbol}_{int(time.time())}",
            **kwargs
        )
        oid = (res.get("result") or {}).get("orderId", "—")
        update.message.reply_text(
            f"✅ PROBE LIMIT ok:\n"
            f"• {symbol} {side} qty={qty} @ {price}\n"
            f"• TP={tp or '—'} SL={sl or '—'} TIF={kwargs.get('timeInForce')}\n"
            f"• orderId={oid}"
        )
    except Exception as e:
        update.message.reply_text(f"❌ PROBE LIMIT fail: {e}")

def cmd_probe_market(update: Update, context: CallbackContext):
    """
    /probe_market <symbol> <side> [usd] [tp] [sl]
    Откроет MARKET-позицию на объём usd (по умолчанию 25 USDT) с TP/SL (опционально).
    """
    args = _parse_args(update.message.text)
    if len(args) < 2:
        update.message.reply_text("Usage: /probe_market <symbol> <side BUY|SELL> [usd] [tp] [sl]")
        return
    symbol = args[0].upper()
    side   = args[1].upper()
    if side not in ("BUY","SELL"):
        update.message.reply_text("❌ side должен быть BUY/SELL.")
        return
    usd = _to_float_safe(args[2], 25.0) if len(args) >= 3 else 25.0
    tp  = _to_float_safe(args[3]) if len(args) >= 4 else None
    sl  = _to_float_safe(args[4]) if len(args) >= 5 else None

    try:
        res = open_position_market(
            symbol=symbol,
            side=side,
            usd_value=usd,
            tp_price=tp,
            sl_price=sl,
        )
        oid = (res.get("result") or {}).get("orderId", "—")
        update.message.reply_text(
            f"✅ PROBE MARKET ok:\n"
            f"• {symbol} {side} usd={usd}\n"
            f"• TP={tp or '—'} SL={sl or '—'}\n"
            f"• orderId={oid}"
        )
    except Exception as e:
        update.message.reply_text(f"❌ PROBE MARKET fail: {e}")

def run_telegram_bot():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Команды
    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("bulk_closed", cmd_bulk_closed))
    dp.add_handler(CommandHandler("bulk_open", cmd_bulk_open))
    dp.add_handler(CommandHandler("bulk_all", cmd_bulk_all))
    dp.add_handler(CommandHandler("missed_signals", cmd_missed_signals))
    dp.add_handler(CommandHandler("missed_trades", cmd_missed_trades))
    dp.add_handler(CommandHandler("active_waiting", cmd_active_waiting))

    # Тест API
    dp.add_handler(CommandHandler("probe_limit", cmd_probe_limit))
    dp.add_handler(CommandHandler("probe_market", cmd_probe_market))
    dp.add_handler(CommandHandler("probe_prices", cmd_probe_prices))

    # fallback: покажем help
    dp.add_handler(MessageHandler(Filters.command, cmd_help))

    updater.start_polling()
    log.info("Telegram command bot started.")
    updater.idle()


# ===================== Console Entry =====================

if __name__ == "__main__":
    # bulk (совместимость) == bulk_closed
    if len(sys.argv) >= 2 and sys.argv[1] == "bulk":
        days = int(sys.argv[2]) if len(sys.argv) >= 3 else 90
        tf   = sys.argv[3] if len(sys.argv) >= 4 else None
        path = bulk_closed_report(days=days, interval=tf, max_days_to_fill=None)
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "bulk_closed":
        days = int(sys.argv[2]) if len(sys.argv) >= 3 else 90
        tf   = sys.argv[3] if len(sys.argv) >= 4 else None
        mfd  = int(sys.argv[4]) if len(sys.argv) >= 5 else None
        bulk_closed_report(days=days, interval=tf, max_days_to_fill=mfd)
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "bulk_open":
        days = int(sys.argv[2]) if len(sys.argv) >= 3 else 30
        tf   = sys.argv[3] if len(sys.argv) >= 4 else None
        bulk_open_report(days=days, interval=tf)
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "bulk_all":
        days = int(sys.argv[2]) if len(sys.argv) >= 3 else 360
        tf   = sys.argv[3] if len(sys.argv) >= 4 else None
        bulk_all_report(days=days, interval=tf)
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "missed_signals":
        hrs = int(sys.argv[2]) if len(sys.argv) >= 3 else 24
        tf  = sys.argv[3] if len(sys.argv) >= 4 else None
        missed_signals(hours=hrs, interval=tf)
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "missed_trades":
        hrs = int(sys.argv[2]) if len(sys.argv) >= 3 else 24
        tf  = sys.argv[3] if len(sys.argv) >= 4 else None
        missed_trades(hours=hrs, interval=tf)
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "active_waiting":
        lookback = int(sys.argv[2]) if len(sys.argv) >= 3 else 7
        tf       = sys.argv[3] if len(sys.argv) >= 4 else None
        active_waiting(lookback_days=lookback, interval=tf)
        sys.exit(0)

    # Если аргументов нет — просто запускаем TG-бота команд
    run_telegram_bot()