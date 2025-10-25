# evaluate_momentum.py — быстрый оффлайн-«зеркал» лайва:
# — вход по CLOSE 4h, TP/SL от entry либо по Фибо (если включено)
# — комиссии/слиппедж из ENV/CLI
# — поддержка внешнего price_cache (для ускорения множества прогонов)

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import math
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from evaluate_common import (
    get_cfg, INITIAL_CAPITAL, POSITION_FRACTION, DEFAULT_TTL_DAYS,
    to_utc_safe, fetch_ltf_window, load_price_cache, load_signals,
    enforce_one_at_a_time_per_symbol, simulate_capital_notional,
    safe_group_exit_reason, finalize_write, calc_sl_tp,
)

# ======= НАСТРОЙКА ИНДИКАТОРОВ =======
# switches
USE_FIB_4H = bool(get_cfg("USE_FIB_4H", cast=bool, default=False))
USE_DIV_4H = bool(get_cfg("USE_DIV_4H", cast=bool, default=False))
USE_OBOS_4H = bool(get_cfg("USE_OBOS_4H", cast=bool, default=False))

# fib
FIB_4H_LOOKBACK_BARS = int(get_cfg("FIB_4H_LOOKBACK_BARS", cast=int, default=120))
FIB_4H_PIVOT_LEN     = int(get_cfg("FIB_4H_PIVOT_LEN", cast=int, default=3))
FIB_SET              = [float(x) for x in str(get_cfg("FIB_SET", cast=str, default="0.236,0.382,0.5,0.618,0.786,1.0,1.272,1.618")).split(",")]
FIB_TP_INDEX         = int(get_cfg("FIB_TP_INDEX", cast=int, default=3))  # 0.618 по умолчанию при стандартной сетке
FIB_TOUCH_MODE       = str(get_cfg("FIB_TOUCH_MODE", cast=str, default="wick")).strip().lower()  # wick|close  # TODO: реализовать
FIB_SL_MODE          = str(get_cfg("FIB_SL_MODE", cast=str, default="current")).strip().lower()  # current|beyond_prev_fib

# divergence
DIV4H_TYPE           = str(get_cfg("DIV4H_TYPE", cast=str, default="off")).strip().lower()  # off|rsi|macd
RSI_PERIOD           = int(get_cfg("RSI_PERIOD", cast=int, default=14))
MACD_FAST            = int(get_cfg("MACD_FAST", cast=int, default=12))
MACD_SLOW            = int(get_cfg("MACD_SLOW", cast=int, default=26))
MACD_SIGNAL          = int(get_cfg("MACD_SIGNAL", cast=int, default=9))
DIV4H_PIVOT_LEN      = int(get_cfg("DIV4H_PIVOT_LEN", cast=int, default=3))
DIV4H_LOOKBACK_BARS  = int(get_cfg("DIV4H_LOOKBACK_BARS", cast=int, default=120))
DIV4H_CONFIRM_BARS   = int(get_cfg("DIV4H_CONFIRM_BARS", cast=int, default=2))
DIV4H_POLICY         = str(get_cfg("DIV4H_POLICY", cast=str, default="tighten_tp")).strip().lower()  # skip_entry|tighten_tp|trail_to_prev_fib  # TODO: trail_to_prev_fib
DIV_TIGHTEN_STEP     = int(get_cfg("DIV_TIGHTEN_STEP", cast=int, default=1))

# ob/os
OBOS_TYPE            = str(get_cfg("OBOS_TYPE", cast=str, default="off")).strip().lower()  # off|rsi|stoch|wpr|cci
OB_RSI_OB            = float(get_cfg("OB_RSI_OB", cast=float, default=70.0))
OB_RSI_OS            = float(get_cfg("OB_RSI_OS", cast=float, default=30.0))
STO_K                = int(get_cfg("STO_K", cast=int, default=14))
STO_D                = int(get_cfg("STO_D", cast=int, default=3))
STO_SMA              = int(get_cfg("STO_SMA", cast=int, default=3))
OB_STOCH_OB          = float(get_cfg("OB_STOCH_OB", cast=float, default=80.0))
OB_STOCH_OS          = float(get_cfg("OB_STOCH_OS", cast=float, default=20.0))
OB_WPR_OB            = float(get_cfg("OB_WPR_OB", cast=float, default=-20.0))
OB_WPR_OS            = float(get_cfg("OB_WPR_OS", cast=float, default=-80.0))
OB_CCI_OB            = float(get_cfg("OB_CCI_OB", cast=float, default=100.0))
OB_CCI_OS            = float(get_cfg("OB_CCI_OS", cast=float, default=-100.0))
OBOS_POLICY          = str(get_cfg("OBOS_POLICY", cast=str, default="tp_bias")).strip().lower()  # filter_entry|tp_bias|sl_bias  # TODO: sl_bias

# =========================
# ПАРАМЕТРЫ ЧЕРЕЗ ENV
# =========================

# RR (в долях, например 0.03 = 3%)
TAKE_PCT = float(get_cfg("MOMENTUM_TP_PCT", cast=float))
STOP_PCT = float(get_cfg("MOMENTUM_SL_PCT", cast=float))

# Принудительно «entry»-схема (как в лайве)
TP_SL_MODE = "entry"

# Комиссия по сделке (taker), доля
FEE_TAKER = float(get_cfg("FEE_TAKER", cast=float, default=0.0))

# Слиппедж (общий и/или поканальный)
SLIPPAGE_PCT_DEFAULT = float(get_cfg("SLIPPAGE_PCT", cast=float, default=0.003))  # 0.3% по умолчанию
ENTRY_SLIPPAGE_PCT = float(get_cfg("ENTRY_SLIPPAGE_PCT", cast=float, default=SLIPPAGE_PCT_DEFAULT))
EXIT_SLIPPAGE_PCT  = float(get_cfg("EXIT_SLIPPAGE_PCT",  cast=float, default=SLIPPAGE_PCT_DEFAULT))
STOP_SLIPPAGE_PCT  = float(get_cfg("STOP_SLIPPAGE_PCT",  cast=float, default=EXIT_SLIPPAGE_PCT))

# Минутная дорисовка: поддерживаем ОБА набора ключей для совместимости
DISABLE_MINUTE_FALLBACK = bool(
    get_cfg("DISABLE_MINUTE_FALLBACK", cast=bool,
            default=get_cfg("MOMENTUM_DISABLE_MINUTE_FALLBACK", cast=bool, default=False))
)
MINUTE_EXIT_FOR_SINGLE = bool(
    get_cfg("MINUTE_EXIT_FOR_SINGLE", cast=bool,
            default=get_cfg("MOMENTUM_MINUTE_EXIT_FOR_SINGLE_HIT", cast=bool, default=True))
)

# -------- хелперы индикаторов --------
def _rsi(x: pd.Series, n: int) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0.0).rolling(n, min_periods=n).mean()
    dn = (-d.clip(upper=0.0)).rolling(n, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _macd(x: pd.Series, f=12, s=26, sig=9):
    ema_f = x.ewm(span=f, adjust=False).mean()
    ema_s = x.ewm(span=s, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def _stoch(df: pd.DataFrame, k=14, d=3, sma=3):
    ll = df["low"].rolling(k, min_periods=k).min()
    hh = df["high"].rolling(k, min_periods=k).max()
    k_raw = (df["close"] - ll) / (hh - ll).replace(0, np.nan) * 100.0
    k_sma = k_raw.rolling(sma, min_periods=sma).mean()
    d_sma = k_sma.rolling(d, min_periods=d).mean()
    return k_sma, d_sma

def _wpr(df: pd.DataFrame, n=14):
    hh = df["high"].rolling(n, min_periods=n).max()
    ll = df["low"].rolling(n, min_periods=n).min()
    return (df["close"] - hh) / (hh - ll).replace(0, np.nan) * 100.0  # ~[-100..0], OB>-20, OS<-80

def _cci(df: pd.DataFrame, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(n, min_periods=n).mean()
    md = (tp - ma).abs().rolling(n, min_periods=n).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def _find_swing_range(df4h: pd.DataFrame, t0: pd.Timestamp, lookback: int, piv_len: int):
    bars = df4h[df4h.index < t0].tail(lookback).copy()
    if bars.empty: return (np.nan, np.nan, pd.NaT, pd.NaT)
    # простые пивоты по окрестности
    hi = bars["high"].rolling(piv_len*2+1, center=True).apply(lambda a: 1.0 if a[piv_len]==max(a) else 0.0, raw=True)
    lo = bars["low"].rolling(piv_len*2+1, center=True).apply(lambda a: 1.0 if a[piv_len]==min(a) else 0.0, raw=True)
    pv_hi = bars[hi==1.0]
    pv_lo = bars[lo==1.0]
    if pv_hi.empty or pv_lo.empty:  # fallback
        L = float(bars["low"].min()); H = float(bars["high"].max())
        return (L, H, bars["low"].idxmin(), bars["high"].idxmax())
    last_hi = pv_hi.iloc[-1]; last_lo = pv_lo.iloc[-1]
    # диапазон по последней паре, ближней к t0
    if last_hi.name > last_lo.name:
        L, Lt = float(pv_lo["low"].iloc[-1]), pv_lo.index[-1]
        H, Ht = float(pv_hi["high"].iloc[-1]), pv_hi.index[-1]
    else:
        L, Lt = float(pv_lo["low"].iloc[-1]), pv_lo.index[-1]
        H, Ht = float(pv_hi["high"].iloc[-1]), pv_hi.index[-1]
    return (L, H, Lt, Ht)

def _build_fib_levels(L: float, H: float, fib_set: list, side: str):
    if not np.isfinite(L) or not np.isfinite(H) or L<=0 or H<=0 or H<=L:
        return []
    if side=="BUY":
        return [L + (H-L)*r for r in fib_set]
    else:
        # для шорта уровни от H вниз симметрично
        return [H - (H-L)*r for r in fib_set]

def _choose_fib_tp(levels: list, idx: int, entry_px: float, side: str, tighten_steps=0):
    if not levels: return np.nan, -1
    i = max(0, min(len(levels)-1, idx))
    i = max(0, min(len(levels)-1, i - tighten_steps))  # сдвиг "ужесточения"
    return float(levels[i]), i

def _detect_divergence(df4h: pd.DataFrame, t0: pd.Timestamp, side: str, osc: pd.Series,
                       lookback: int, piv_len: int, confirm_bars: int):
    bars = df4h[df4h.index < t0].tail(lookback).copy()
    if bars.empty or osc is None or osc.empty: return False, None, None
    osc = osc.reindex(bars.index)
    hi = bars["high"].rolling(piv_len*2+1, center=True).apply(lambda a: 1.0 if a[piv_len]==max(a) else 0.0, raw=True)
    lo = bars["low"].rolling(piv_len*2+1, center=True).apply(lambda a: 1.0 if a[piv_len]==min(a) else 0.0, raw=True)
    pivH = bars[hi==1.0].tail(2)
    pivL = bars[lo==1.0].tail(2)
    if side=="BUY" and len(pivL)==2:
        # бычья дивергенция: цена делает lower low, осциллятор higher low
        p1, p2 = pivL.index[-2], pivL.index[-1]
        cond_price = bars.loc[p2,"low"] < bars.loc[p1,"low"]
        cond_osc   = osc.loc[p2] > osc.loc[p1]
        confirmed  = (bars.index.get_loc(p2) <= len(bars)-1) and (len(bars)-1 - bars.index.get_loc(p2) >= confirm_bars-1)
        return bool(cond_price and cond_osc and confirmed), "bull", p2
    if side=="SELL" and len(pivH)==2:
        # медвежья: цена higher high, осциллятор lower high
        p1, p2 = pivH.index[-2], pivH.index[-1]
        cond_price = bars.loc[p2,"high"] > bars.loc[p1,"high"]
        cond_osc   = osc.loc[p2] < osc.loc[p1]
        confirmed  = (bars.index.get_loc(p2) <= len(bars)-1) and (len(bars)-1 - bars.index.get_loc(p2) >= confirm_bars-1)
        return bool(cond_price and cond_osc and confirmed), "bear", p2
    return False, None, None

def _calc_obos(df4h: pd.DataFrame, at: pd.Timestamp):
    df = df4h[df4h.index <= at].copy()
    if df.empty: return ("none", np.nan)
    if OBOS_TYPE=="rsi":
        r = _rsi(df["close"], RSI_PERIOD).iloc[-1]
        if r>=OB_RSI_OB: return ("ob", float(r))
        if r<=OB_RSI_OS: return ("os", float(r))
        return ("none", float(r))
    if OBOS_TYPE=="stoch":
        k, d = _stoch(df, STO_K, STO_D, STO_SMA)
        v = float(k.iloc[-1])
        if v>=OB_STOCH_OB: return ("ob", v)
        if v<=OB_STOCH_OS: return ("os", v)
        return ("none", v)
    if OBOS_TYPE=="wpr":
        v = float(_wpr(df, STO_K).iloc[-1])  # используем STO_K как окно
        if v>=OB_WPR_OB: return ("ob", v)
        if v<=OB_WPR_OS: return ("os", v)
        return ("none", v)
    if OBOS_TYPE=="cci":
        v = float(_cci(df, 20).iloc[-1])
        if v>=OB_CCI_OB: return ("ob", v)
        if v<=OB_CCI_OS: return ("os", v)
        return ("none", v)
    return ("none", np.nan)

def _normalize_side(v: str) -> str:
    s = str(v).strip().upper()
    if s in ("BUY", "LONG"):  return "BUY"
    if s in ("SELL", "SHORT"):return "SELL"
    return s

def _apply_slippage(price: float, pct: float, action: str) -> float:
    """Худший fill: BUY — дороже, SELL — дешевле."""
    sp = float(pct or 0.0)
    if not isinstance(price, (int, float)) or math.isnan(price):
        return float('nan')
    return float(price) * (1.0 + sp) if action == "BUY" else float(price) * (1.0 - sp)

def _pnl_pct_with_fees(entry_px_adj: float, exit_px_adj: float, side: str, fee_taker_pct: float) -> float:
    if any(not isinstance(x, (int, float)) or math.isnan(x) for x in (entry_px_adj, exit_px_adj)):
        return float('nan')
    move = ((exit_px_adj - entry_px_adj) / entry_px_adj * 100.0) if side == "BUY" \
           else ((entry_px_adj - exit_px_adj) / entry_px_adj * 100.0)
    fees = float(fee_taker_pct or 0.0) * 2.0 * 100.0
    return float(move) - float(fees)

def _entry_from_4h_close(df_4h: pd.DataFrame, t0):
    """Вход в момент t0 по цене CLOSE 4h-свечи (бар со стартом t0-4h)."""
    t0 = to_utc_safe(t0)
    if pd.isna(t0) or df_4h is None or df_4h.empty:
        return (pd.NaT, float("nan"))
    pos = df_4h.index.searchsorted(t0)
    ix = pos - 1
    if ix < 0:
        return (pd.NaT, float("nan"))
    try:
        entry_px_ref = float(df_4h.iloc[ix]["close"])
    except Exception:
        entry_px_ref = float("nan")
    return (t0, entry_px_ref)

def _resolve_exit_minute(symbol: str, side: str, entry_at, stop_eval: float, tp_eval: float, t_end):
    """Минутная дорисовка порядка TP/SL. Если оба — считаем, что SL первым."""
    t0 = to_utc_safe(entry_at)
    t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end):
        return (None, pd.NaT, float('nan'), "bad_ts")

    ltf = fetch_ltf_window(symbol, t0, t_end, candidates=["1m"])
    if ltf is None or ltf.empty:
        return (None, pd.NaT, float('nan'), "uncertain_no_ltf")

    for ts, c in ltf.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if side == "BUY":
            hit_tp = hi >= float(tp_eval)
            hit_sl = lo <= float(stop_eval)
        else:
            hit_tp = lo <= float(tp_eval)
            hit_sl = hi >= float(stop_eval)

        if hit_tp and hit_sl:
            return (False, ts, float(stop_eval), "sl")
        if hit_tp:
            return (True, ts, float(tp_eval), "tp")
        if hit_sl:
            return (False, ts, float(stop_eval), "sl")

    last_ts = ltf.index[-1]
    last_close = float(ltf.iloc[-1]["close"])
    return (False, last_ts, last_close, "timeout_last_close")

def _resolve_exit_hybrid(symbol, side, entry_at, stop_eval, tp_eval, t_end, df_4h):
    """
    Быстрый 4h-проход; если бар содержит оба уровня — спускаемся на 1m.
    Для одиночного хита — опционально уточняем минутой.
    """
    t0 = to_utc_safe(entry_at)
    t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end) or df_4h is None or df_4h.empty:
        return (None, pd.NaT, float("nan"), "bad_ts", "4h")

    bars = df_4h[(df_4h.index >= t0) & (df_4h.index < t_end)]
    for ts, c in bars.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        bar_start, bar_end = ts, ts + pd.Timedelta(hours=4)
        if side == "BUY":
            hit_tp = hi >= float(tp_eval)
            hit_sl = lo <= float(stop_eval)
        else:
            hit_tp = lo <= float(tp_eval)
            hit_sl = hi >= float(stop_eval)

        if hit_tp and hit_sl:
            if DISABLE_MINUTE_FALLBACK:
                return (None, bar_end, float('nan'), "uncertain_both_hit_4h", "4h")
            win_m, close_t, trig_px, reason = _resolve_exit_minute(
                symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end)
            )
            return (win_m, close_t, trig_px, reason, "1m")

        if hit_tp:
            if MINUTE_EXIT_FOR_SINGLE and not DISABLE_MINUTE_FALLBACK:
                win_m, close_t, trig_px, reason = _resolve_exit_minute(
                    symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end)
                )
                if reason in ("tp", "sl", "timeout_last_close"):
                    return (win_m, close_t, trig_px, reason, "1m")
            return (True, bar_end, float(tp_eval), "tp", "4h")

        if hit_sl:
            if MINUTE_EXIT_FOR_SINGLE and not DISABLE_MINUTE_FALLBACK:
                win_m, close_t, trig_px, reason = _resolve_exit_minute(
                    symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end)
                )
                if reason in ("tp", "sl", "timeout_last_close"):
                    return (win_m, close_t, trig_px, reason, "1m")
            return (False, bar_end, float(stop_eval), "sl", "4h")

    if not bars.empty:
        last_ts = bars.index[-1] + pd.Timedelta(hours=4)
        last_close = float(bars.iloc[-1]["close"])
    else:
        last_ts = t0
        last_close = float("nan")
    return (False, last_ts, last_close, "timeout_last_close", "4h")

# ---------------- основная функция ----------------
def evaluate_momentum(
    signals_path: str,
    result_path: str,
    lookback_days: int = 360,
    interval: str = "4h",
    max_days: int = None,
    only_filled: bool = False,
    dedup: bool = False,
    initial_capital: float = None,
    capital_aware: bool = True,
    *,
    price_cache: dict = None,   # ← внешнее кэш-хранилище свечей (ускоряет циклы)
):
    # Модельная дата = mtime входного файла (как «срез истории»)
    try:
        as_of = datetime.fromtimestamp(os.path.getmtime(signals_path), tz=timezone.utc)
    except Exception:
        as_of = datetime.now(tz=timezone.utc)

    # Сигналы
    df_sig = load_signals(signals_path, only_filled=only_filled, dedup=dedup, require_entry=False)
    if df_sig.empty:
        print("⚠️ Сигналов нет после фильтров.")
        finalize_write(result_path, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return

    symbols = df_sig["symbol"].dropna().unique().tolist()

    # Кэш свечей (если не передали — загрузим)
    if price_cache is None:
        price_cache = load_price_cache(symbols, interval=interval, lookback_days=lookback_days)

    results = []
    for _, row in df_sig.iterrows():
        symbol = row["symbol"]
        side = _normalize_side(row.get("type"))
        t0 = to_utc_safe(row.get("imb_time"))
        df4h = price_cache.get(symbol)

        if df4h is None or df4h.empty or side not in ("BUY", "SELL") or pd.isna(t0):
            continue

        # Вход по CLOSE 4h
        entry_at, entry_px_ref = _entry_from_4h_close(df4h, t0)
        if pd.isna(entry_at) or not isinstance(entry_px_ref, (int, float)) or math.isnan(entry_px_ref):
            continue

        entry_action = "BUY" if side == "BUY" else "SELL"
        entry_px_adj = _apply_slippage(entry_px_ref, ENTRY_SLIPPAGE_PCT, entry_action)

        # --- 4h indicators / targets ---
        fib_tp_4h = pd.NA; fib_idx = pd.NA
        fib_L = pd.NA; fib_H = pd.NA
        div_flag = False; div_side = pd.NA; div_conf_at = pd.NaT
        obos_flag = "none"; obos_value = pd.NA

        # базовый RR (если не будем использовать фибо)
        rr = float(TAKE_PCT) / max(float(STOP_PCT), 1e-9)

        # FIB (опционально)
        if USE_FIB_4H:
            L, H, Lt, Ht = _find_swing_range(df4h, entry_at, FIB_4H_LOOKBACK_BARS, FIB_4H_PIVOT_LEN)
            fib_L, fib_H = (float(L) if np.isfinite(L) else pd.NA), (float(H) if np.isfinite(H) else pd.NA)
            fib_levels = _build_fib_levels(L, H, FIB_SET, side)
        else:
            fib_levels = []

        # OSC для дивергенций
        osc_series = None
        if USE_DIV_4H and DIV4H_TYPE in ("rsi","macd"):
            if DIV4H_TYPE=="rsi":
                osc_series = _rsi(df4h["close"], RSI_PERIOD)
            else:
                macd, signal, hist = _macd(df4h["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
                osc_series = macd  # дивергенции по линии MACD

        if USE_DIV_4H and osc_series is not None:
            div_flag, div_side, div_conf_at = _detect_divergence(
                df4h, entry_at, side, osc_series, DIV4H_LOOKBACK_BARS, DIV4H_PIVOT_LEN, DIV4H_CONFIRM_BARS
            )

        # OB/OS (на момент входа)
        if USE_OBOS_4H and OBOS_TYPE != "off":
            obos_flag, obos_value = _calc_obos(df4h, entry_at)

        # выбрать TP/SL (учитывая Фибо/дивер/OBOS)
        if USE_FIB_4H and fib_levels:
            tighten = 0
            if div_flag and DIV4H_POLICY == "tighten_tp":
                tighten = int(DIV_TIGHTEN_STEP)
            tp_eval, fib_idx = _choose_fib_tp(fib_levels, FIB_TP_INDEX, float(entry_px_adj), side, tighten_steps=tighten)

            # bias по OB/OS
            if USE_OBOS_4H and OBOS_POLICY in ("tp_bias","sl_bias") and obos_flag in ("ob","os"):
                if OBOS_POLICY == "tp_bias":
                    # для лонга +OB → ближе; для шорта +OS → ближе
                    closer = 1
                    if (side=="BUY" and obos_flag=="ob") or (side=="SELL" and obos_flag=="os"):
                        tp_eval, fib_idx = _choose_fib_tp(fib_levels, max(0, int(fib_idx)-closer), float(entry_px_adj), side, tighten_steps=0)
                else:
                    # TODO: sl_bias — смещение SL по сетке (или расширение)
                    pass

            # SL
            if FIB_SL_MODE == "beyond_prev_fib" and isinstance(fib_idx, int) and fib_idx>0:
                prev_level = float(fib_levels[fib_idx-1])
                # для BUY и SELL prev_level — «соседний» уровень сетки
                sl_eval = prev_level
            else:
                sl_eval, _ = calc_sl_tp(float(entry_px_adj), side, float(STOP_PCT), rr)

            fib_tp_4h = float(tp_eval) if np.isfinite(tp_eval) else pd.NA

        else:
            sl_eval, tp_eval = calc_sl_tp(float(entry_px_adj), side, float(STOP_PCT), rr)
            fib_tp_4h = pd.NA
            fib_idx = pd.NA

        # политика filter_entry по OB/OS
        if USE_OBOS_4H and OBOS_POLICY=="filter_entry":
            if (side=="BUY" and obos_flag=="ob") or (side=="SELL" and obos_flag=="os"):
                out = row.to_dict();
                out.update({
                    "symbol": symbol, "type": side, "strength": row.get("strength", np.nan),
                    "imb_time": t0, "t_start": entry_at,
                    "entry": float(entry_px_adj),
                    "stop": float(sl_eval) if pd.notna(sl_eval) else np.nan,
                    "tp": np.nan,
                    "win": np.nan, "pnl_pct": np.nan, "pnl_usd": np.nan, "move_pct": np.nan,
                    "close_time": pd.NaT, "close_price": np.nan,
                    "exit_reason": "filtered_obos", "is_open_mark": False,
                    "variant": "MOMENTUM", "as_of": as_of, "skipped": True,
                    "lt_resolution": "none",
                    "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),
                    # мета
                    "fib_tp_4h": float(fib_tp_4h) if pd.notna(fib_tp_4h) else np.nan,
                    "fib_set": (",".join(map(str, FIB_SET)) if USE_FIB_4H else None),
                    "fib_anchor_L": float(fib_L) if pd.notna(fib_L) else np.nan,
                    "fib_anchor_H": float(fib_H) if pd.notna(fib_H) else np.nan,
                    "div4h_flag": bool(div_flag),
                    "div4h_type": (DIV4H_TYPE if USE_DIV_4H else None),
                    "div4h_side": (div_side if isinstance(div_side, str) else None),
                    "div4h_confirm_at": div_conf_at,  # это datetime — норм
                    "obos_flag": (obos_flag if USE_OBOS_4H else None),
                    "obos_type": (OBOS_TYPE if USE_OBOS_4H else None),
                    "obos_value": float(obos_value) if pd.notna(obos_value) else np.nan,
                }); results.append(out); continue

        # политика skip_entry по дивергенции
        if USE_DIV_4H and div_flag and DIV4H_POLICY=="skip_entry":
            out = row.to_dict();
            out.update({
                "symbol": symbol, "type": side, "strength": row.get("strength", np.nan),
                "imb_time": t0, "t_start": entry_at,
                "entry": float(entry_px_adj),
                "stop": float(sl_eval) if pd.notna(sl_eval) else np.nan,
                "tp": np.nan,
                "win": np.nan, "pnl_pct": np.nan, "pnl_usd": np.nan, "move_pct": np.nan,
                "close_time": pd.NaT, "close_price": np.nan,
                "exit_reason": "filtered_obos", "is_open_mark": False,
                "variant": "MOMENTUM", "as_of": as_of, "skipped": True,
                "lt_resolution": "none",
                "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),
                # мета
                "fib_tp_4h": float(fib_tp_4h) if pd.notna(fib_tp_4h) else np.nan,
                "fib_set": (",".join(map(str, FIB_SET)) if USE_FIB_4H else None),
                "fib_anchor_L": float(fib_L) if pd.notna(fib_L) else np.nan,
                "fib_anchor_H": float(fib_H) if pd.notna(fib_H) else np.nan,
                "div4h_flag": bool(div_flag),
                "div4h_type": (DIV4H_TYPE if USE_DIV_4H else None),
                "div4h_side": (div_side if isinstance(div_side, str) else None),
                "div4h_confirm_at": div_conf_at,  # это datetime — норм
                "obos_flag": (obos_flag if USE_OBOS_4H else None),
                "obos_type": (OBOS_TYPE if USE_OBOS_4H else None),
                "obos_value": float(obos_value) if pd.notna(obos_value) else np.nan,
            }); results.append(out); continue

        # Окно жизни
        ttl_days = int(DEFAULT_TTL_DAYS if max_days is None else max_days)
        window_end = min(t0 + pd.Timedelta(days=ttl_days), as_of)

        # Выход
        win, close_time, trigger_price, exit_reason, lt_res = _resolve_exit_hybrid(
            symbol=symbol,
            side=side,
            entry_at=entry_at,
            stop_eval=float(sl_eval),
            tp_eval=float(tp_eval),
            t_end=window_end,
            df_4h=df4h,
        )

        if win is None:
            # помечаем пропуск (например, нет 1m-данных)
            results.append({
                "symbol": symbol, "type": side, "strength": row.get("strength", pd.NA),
                "imb_time": t0, "t_start": entry_at,
                "entry": float(entry_px_adj), "stop": float(sl_eval), "tp": float(tp_eval),
                "win": pd.NA, "pnl_pct": pd.NA, "pnl_usd": pd.NA,
                "close_time": close_time, "close_price": pd.NA, "exit_reason": str(exit_reason),
                "exit_hours": pd.NA, "usd_alloc": pd.NA,
                "variant": "MOMENTUM", "as_of": as_of, "skipped": True,
                "lt_resolution": lt_res,
                # мета по индикаторам
                "fib_tp_4h": fib_tp_4h,
                "fib_set": (",".join(map(str, FIB_SET)) if USE_FIB_4H else pd.NA),
                "fib_anchor_L": fib_L, "fib_anchor_H": fib_H,
                "div4h_flag": bool(div_flag), "div4h_type": DIV4H_TYPE if USE_DIV_4H else pd.NA,
                "div4h_side": div_side, "div4h_confirm_at": div_conf_at,
                "obos_flag": obos_flag if USE_OBOS_4H else pd.NA,
                "obos_type": OBOS_TYPE if USE_OBOS_4H else pd.NA,
                "obos_value": float(obos_value) if (USE_OBOS_4H and pd.notna(obos_value)) else pd.NA,
            })
            continue

        # Выходная цена + худший слиппедж
        exit_action = "SELL" if side == "BUY" else "BUY"
        exit_slip = STOP_SLIPPAGE_PCT if exit_reason == "sl" else EXIT_SLIPPAGE_PCT
        exit_px_base = float(trigger_price) if isinstance(trigger_price, (int, float)) else float('nan')
        exit_px_adj = _apply_slippage(exit_px_base, exit_slip, exit_action)
        pnl_pct_net = _pnl_pct_with_fees(entry_px_adj, exit_px_adj, side, FEE_TAKER)

        results.append({
            "symbol": symbol, "type": side, "strength": row.get("strength", pd.NA),
            "imb_time": t0, "t_start": entry_at,
            "entry": float(entry_px_adj), "stop": float(sl_eval), "tp": float(tp_eval),
            "win": True if exit_reason == "tp" else False, "pnl_pct": float(pnl_pct_net), "pnl_usd": pd.NA,
            "close_time": close_time, "close_price": float(exit_px_adj), "exit_reason": str(exit_reason),
            "exit_hours": (
                (pd.to_datetime(close_time, utc=True) - pd.to_datetime(entry_at, utc=True)).total_seconds() / 3600.0
                if pd.notna(close_time) else pd.NA
            ),
            "usd_alloc": pd.NA,
            "variant": "MOMENTUM", "as_of": as_of, "skipped": False,
            "lt_resolution": lt_res,
            # мета по индикаторам
            "fib_tp_4h": fib_tp_4h,
            "fib_set": (",".join(map(str, FIB_SET)) if USE_FIB_4H else pd.NA),
            "fib_anchor_L": fib_L, "fib_anchor_H": fib_H,
            "div4h_flag": bool(div_flag), "div4h_type": DIV4H_TYPE if USE_DIV_4H else pd.NA,
            "div4h_side": div_side, "div4h_confirm_at": div_conf_at,
            "obos_flag": obos_flag if USE_OBOS_4H else pd.NA,
            "obos_type": OBOS_TYPE if USE_OBOS_4H else pd.NA,
            "obos_value": float(obos_value) if (USE_OBOS_4H and pd.notna(obos_value)) else pd.NA,
        })

    df = pd.DataFrame(results)

    NUMERIC_COLS = [
        "entry", "stop", "tp", "close_price",
        "pnl_pct", "pnl_usd", "usd_alloc", "exit_hours",
        "fib_tp_4h", "fib_anchor_L", "fib_anchor_H", "obos_value", "strength"
    ]
    DATETIME_COLS = ["imb_time", "t_start", "close_time", "as_of", "div4h_confirm_at"]

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in DATETIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    # строковые мета — строго строки/None (чтобы не уехали в datetime)
    for c in ["fib_set", "div4h_type", "div4h_side", "obos_flag", "obos_type", "lt_resolution", "exit_reason",
              "variant", "type", "symbol"]:
        if c in df.columns:
            df[c] = df[c].astype("object")

    # уберём TZ для Excel
    for c in ["imb_time", "t_start", "close_time", "as_of", "div4h_confirm_at"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.tz_convert(None)

    if df.empty:
        print("⚠️ Ничего не оценили.")
        finalize_write(result_path, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return

    # Запрет перекрытий по символу (как в лайве)
    df = enforce_one_at_a_time_per_symbol(df)

    # === Капитал-сим ===
    sim = df.copy()
    exec_mask = (sim.get("skipped") == False)
    sim.loc[exec_mask, "exit_reason"] = "price"  # для симулятора

    init_cap = float(initial_capital) if initial_capital is not None else float(INITIAL_CAPITAL or 0.0)
    if init_cap > 0 and capital_aware:
        sim_out, eq_sheet = simulate_capital_notional(
            sim, init_cap, POSITION_FRACTION, stop_pct=STOP_PCT, take_pct=TAKE_PCT
        )
        # переносим рассчитанные поля
        for col in ("usd_alloc", "pnl_usd"):
            if col in sim_out.columns:
                df[col] = sim_out[col]
    else:
        eq_sheet = pd.DataFrame()

    # Сводки
    df_exec = df[df.get("skipped") == False].copy()
    by_variant = (
        df_exec.groupby("variant")
        .agg(
            trades=("win", "size"),
            wins=("win", "sum"),
            winrate_pct=("win", lambda s: round(100.0 * float(s.sum()) / max(int(s.size), 1), 2)),
            pnl_pct=("pnl_pct", "sum"),
            pnl_usd=("pnl_usd", "sum"),
        )
        .reset_index()
    ) if not df_exec.empty else pd.DataFrame()

    by_exit_reason = safe_group_exit_reason(df)

    # Excel любит наивное время — убираем таймзону
    for c in ["imb_time", "t_start", "close_time", "as_of"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.tz_convert(None)

    finalize_write(result_path, df, eq_sheet, by_variant, by_exit_reason)
    print(f"✅ MOMENTUM eval saved → {result_path}")


if __name__ == "__main__":
    import argparse
    def _b(v): return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

    p = argparse.ArgumentParser(
        description="Evaluate MOMENTUM (4h close entry, entry TP/SL or Fibonacci; fees/slippage via ENV; minute tie-resolution)."
    )
    p.add_argument("signals")
    p.add_argument("--out", default=None)
    p.add_argument("--interval", default="4h")
    p.add_argument("--lookback-days", type=int, default=360)
    p.add_argument("--ttl-days", type=int, default=None)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--capital-aware", type=_b, default=True)
    p.add_argument("--only-filled", action="store_true")
    p.add_argument("--dedup", action="store_true")

    # CLI-оверрайды fee/slippage
    p.add_argument("--fee-taker-pct", type=float, default=None)
    p.add_argument("--entry-slippage-pct", type=float, default=None)
    p.add_argument("--exit-slippage-pct", type=float, default=None)
    p.add_argument("--stop-slippage-pct", type=float, default=None)

    args = p.parse_args()

    # Подхватываем CLI в ENV, чтобы get_cfg увидел
    if args.fee_taker_pct        is not None: os.environ["FEE_TAKER"]          = str(args.fee_taker_pct)
    if args.entry_slippage_pct   is not None: os.environ["ENTRY_SLIPPAGE_PCT"] = str(args.entry_slippage_pct)
    if args.exit_slippage_pct    is not None: os.environ["EXIT_SLIPPAGE_PCT"]  = str(args.exit_slippage_pct)
    if args.stop_slippage_pct    is not None: os.environ["STOP_SLIPPAGE_PCT"]  = str(args.stop_slippage_pct)

    sig_path = os.path.expanduser(args.signals)
    res_path = os.path.expanduser(args.out) if args.out else os.path.splitext(sig_path)[0] + "_momentum_eval.xlsx"

    evaluate_momentum(
        signals_path=sig_path,
        result_path=res_path,
        lookback_days=int(args.lookback_days),
        interval=str(args.interval),
        max_days=(int(args.ttl_days) if args.ttl_days is not None else None),
        only_filled=bool(args.only_filled),
        dedup=bool(args.dedup),
        initial_capital=(float(args.initial_capital) if args.initial_capital is not None else None),
        capital_aware=bool(args.capital_aware),
    )