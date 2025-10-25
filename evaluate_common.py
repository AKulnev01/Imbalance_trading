import os
import glob
import math
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as CFG

# локальный режим + 4h из минуток (агрегация делается в utils.strategy)
from utils.strategy import get_klines_4h, USE_LOCAL_MINUTES

# где лежат минутки
LTF_ROOT = os.getenv("LTF_ROOT", "./data/m1")

# =============================================================================
# CONFIG HELPERS
# =============================================================================
def get_cfg(name, *, required=True, cast=None, default=None):
    val = getattr(CFG, name, None)
    if val is None:
        val = os.getenv(name, None)
    if val is None:
        if required and default is None:
            raise RuntimeError(f"Missing required setting '{name}' in config.py or .env.")
        val = default
    if cast is not None and val is not None:
        try:
            if cast is bool:
                s = str(val).strip().lower()
                return s in ("1", "true", "yes", "y", "on")
            if cast is list:
                return [x.strip() for x in str(val).split(",") if x.strip()]
            return cast(val)
        except Exception:
            raise RuntimeError(f"Bad value for '{name}': {val!r} (expected {cast.__name__}).")
    return val

# =============================================================================
# CORE PARAMS (shared)
# =============================================================================
INITIAL_CAPITAL   = float(get_cfg("INITIAL_CAPITAL",   cast=float, default=0.0))
POSITION_FRACTION = float(get_cfg("POSITION_FRACTION", cast=float, default=0.0))
FEE_TAKER         = float(get_cfg("FEE_TAKER",         cast=float, default=0.0))
SLIPPAGE_PCT      = float(get_cfg("SLIPPAGE_PCT",      cast=float, default=0.0))
DEFAULT_TTL_DAYS  = int(get_cfg("DEFAULT_TTL_DAYS",    cast=int,   default=30))

INTRABAR_INTERVALS              = get_cfg("INTRABAR_INTERVALS",              cast=list, default=["1m"])
INTRABAR_LOOKBACK_DAYS_FALLBACK = int(get_cfg("INTRABAR_LOOKBACK_DAYS_FALLBACK", cast=int, default=14))
INTRABAR_MAX_LOOKBACK_DAYS      = int(get_cfg("INTRABAR_MAX_LOOKBACK_DAYS",      cast=int, default=720))
MAX_CONCURRENT_POSITIONS        = int(get_cfg("MAX_CONCURRENT_POSITIONS",    cast=int,   default=3))
MOMENTUM_MIN_LTF_BARS           = int(get_cfg("MOMENTUM_MIN_LTF_BARS",       cast=int,   default=1))

# =============================================================================
# TIME & NUMERIC UTILS
# =============================================================================
def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        except Exception:
            pass
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_convert('UTC')
    return df

def to_num(x):
    return pd.to_numeric(str(x).replace(',', '.').strip(), errors='coerce')

def to_utc_safe(ts):
    if pd.isna(ts):
        return pd.NaT
    t = pd.to_datetime(ts, errors='coerce')
    if t is pd.NaT:
        return pd.NaT
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize('UTC')
    return t.tz_convert('UTC')

# =============================================================================
# OHLCV LOADERS (local minutes + resample)
# =============================================================================
def _find_local_ltf_file(root: str, symbol: str, tf: str = "m1") -> Optional[str]:
    """
    Ищет минутные файлы в папке root.
    Поддерживает: SYMBOL.parquet, SYMBOL_m1.parquet, SYMBOL-1m.parquet, SYMBOL.1m.parquet, папка/SYMBOL/1m.parquet.
    """
    sym = str(symbol).upper().replace("/", "")
    patterns = [
        f"{sym}.parquet", f"{sym}.pq", f"{sym}.csv",
        f"{sym}_{tf}.parquet", f"{sym}_{tf}.pq", f"{sym}_{tf}.csv",
        f"{sym}-1m.parquet", f"{sym}-1m.pq", f"{sym}-1m.csv",
        f"{sym}.{tf}.parquet", f"{sym}.{tf}.pq", f"{sym}.{tf}.csv",
        os.path.join(sym, f"{tf}.parquet"),
        os.path.join(sym, "1m.parquet"),
    ]
    for pat in patterns:
        for p in glob.glob(os.path.join(root, pat)):
            if os.path.isfile(p):
                return p
    return None

def _normalize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает индекс DatetimeIndex(UTC) из одной из колонок времени.
    Поддерживает эпоху (сек/мс).
    """
    if df is None or df.empty:
        return df

    candidates = ["time", "timestamp", "open_time", "datetime", "date", "ts"]
    time_col = next((c for c in candidates if c in df.columns), None)

    if time_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            out = df.copy()
            out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
            return out.sort_index()
        raise ValueError(f"Не нашёл колонку времени. Доступные: {list(df.columns)}")

    t = df[time_col]
    if np.issubdtype(t.dtype, np.number):
        mx = float(t.max()) if len(t) else 0.0
        dt = pd.to_datetime(t, unit=("ms" if mx > 1e12 else "s"), utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(t, utc=True, errors="coerce")

    out = df.copy()
    out = out.assign(__t=dt).dropna(subset=["__t"]).set_index("__t").sort_index()
    out.index.name = "time"
    return out

def _load_local_minutes(symbol: str) -> pd.DataFrame:
    sym = str(symbol).upper().replace("/", "").strip()
    p = Path(LTF_ROOT) / f"{sym}_m1.parquet"
    alts = [
        Path(LTF_ROOT) / f"{sym}.parquet",
        Path(LTF_ROOT) / f"{sym}-1m.parquet",
        Path(LTF_ROOT) / f"{sym}.1m.parquet",
        Path(LTF_ROOT) / sym / "1m.parquet",
    ]
    cand = [p] + alts
    path = next((pp for pp in cand if pp.exists()), None)
    if path is None:
        found = _find_local_ltf_file(LTF_ROOT, sym, "m1")
        if found is None:
            raise FileNotFoundError(f"no local minutes for {symbol} under {LTF_ROOT}")
        path = Path(found)

    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path.suffix.lower() in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported minutes file type: {path}")

    df = _normalize_time_index(df)
    for c in ("open","high","low","close","volume","turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])
    return df

def _freq_from_iv(iv: str) -> Optional[str]:
    iv = str(iv).strip().lower()
    if iv in ("1m","1min","1minute"): return "1T"
    if iv in ("3m","3min"): return "3T"
    if iv in ("5m","5min"): return "5T"
    if iv in ("15m","15min"): return "15T"
    if iv in ("30m","30min"): return "30T"
    if iv in ("45m","45min"): return "45T"
    if iv in ("1h","60m"): return "1H"
    if iv in ("2h","120m"): return "2H"
    if iv in ("4h","240m"): return "4H"
    if iv in ("6h","360m"): return "6H"
    if iv in ("12h","720m"): return "12H"
    if iv in ("1d","d","24h"): return "1D"
    return None

def _resample_from_1m(df1m: pd.DataFrame, iv: str) -> pd.DataFrame:
    """Ресемпл 1m в заданный ТФ."""
    freq = _freq_from_iv(iv)
    if not freq:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
    if freq == "1T":
        return df1m.copy()
    o = df1m["open"].resample(freq, label="left", closed="left").first()
    h = df1m["high"].resample(freq).max()
    l = df1m["low"].resample(freq).min()
    c = df1m["close"].resample(freq).last()
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
    if "volume" in df1m.columns:
        df["volume"] = df1m["volume"].resample(freq).sum()
    return df.dropna()

def fetch_ltf_window(symbol, t_start, t_end, candidates=None):
    """Возвращает окно LTF (из локальных минуток). Если кандидаты содержат 1m — вернём 1m."""
    t_start = to_utc_safe(t_start); t_end = to_utc_safe(t_end)
    if candidates is None:
        candidates = INTRABAR_INTERVALS or ["1m"]

    # небольшой паддинг по краям
    s, e = t_start - pd.Timedelta(minutes=1), t_end + pd.Timedelta(minutes=1)

    if USE_LOCAL_MINUTES:
        try:
            df1m = _load_local_minutes(symbol)
        except Exception:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))

        cut = df1m[(df1m.index >= s) & (df1m.index <= e)].copy()
        if cut.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))

        # если просили 1m — вернём 1m
        for iv in [c.strip() for c in candidates if c and c.strip()]:
            if iv.lower() in ("1m","1min","1minute"):
                return cut[(cut.index >= t_start) & (cut.index <= t_end)].copy()

        # иначе — ресемпл
        for iv in [c.strip() for c in candidates if c and c.strip()]:
            df = _resample_from_1m(cut, iv)
            win = df[(df.index >= t_start) & (df.index <= t_end)].copy()
            if not win.empty:
                return win

        return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
    else:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))

# =============================================================================
# TP/SL & EXIT (minute-level)
# =============================================================================
def calc_sl_tp(entry: float, side: str, risk_pct_price: float, rr: float) -> Tuple[float, float]:
    """
    Возвращает (SL, TP) в ЦЕНАХ, а не процентах, исходя из entry, процентного риска и RR.
    risk_pct_price — доля, например 0.015 (=1.5%), rr — отношение TP/SL по расстоянию.
    """
    entry = float(entry)
    k = float(risk_pct_price)
    rr = float(rr)
    if side == "SELL":  # short: стоп выше entry, тейк ниже
        sl = entry * (1.0 + k)
        tp = entry * (1.0 - k * rr)
    else:               # long: стоп ниже entry, тейк выше
        sl = entry * (1.0 - k)
        tp = entry * (1.0 + k * rr)
    return float(sl), float(tp)

def exit_on_ltf(symbol: str,
                side: str,
                entry_at: pd.Timestamp,
                stop_eval: float,
                tp_eval: float,
                t_end: pd.Timestamp) -> Tuple[bool, pd.Timestamp, float, str]:
    """
    Пошагово по 1m-барам после entry; при «оба в минуте» — консервативно SL первым.
    Всегда форсим 1m, чтобы определить порядок срабатывания.
    """
    t0 = to_utc_safe(entry_at); t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end):
        return (False, t_end, float(stop_eval), "sl")
    ltf = fetch_ltf_window(symbol, t0, t_end, candidates=["1m"])
    if ltf.empty:
        return (False, t_end, float(stop_eval), "sl")
    for ts, c in ltf.iterrows():
        hi, lo = float(c['high']), float(c['low'])
        if side == 'BUY':
            hit_tp = (hi >= float(tp_eval));  hit_sl = (lo <= float(stop_eval))
        else:
            hit_tp = (lo <= float(tp_eval));  hit_sl = (hi >= float(stop_eval))
        if hit_tp and hit_sl:
            return (False, ts, float(stop_eval), "sl")
        if hit_tp:
            return (True, ts, float(tp_eval), "tp")
        if hit_sl:
            return (False, ts, float(stop_eval), "sl")
    last_ts = ltf.index[-1]
    last_close = float(ltf.iloc[-1]["close"])
    return (False, last_ts, last_close, "timeout_last_close")

# =============================================================================
# INDICATORS FOR FEATURES (RSI/MACD/MA) + FIBONACCI + DIVERGENCE
# =============================================================================
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0).rolling(n, min_periods=n).mean()
    dn = (-d.clip(upper=0.0)).rolling(n, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def ma_features(df4h: pd.DataFrame, at_ts: pd.Timestamp, fast=50, slow=200, ema=False, slope_lookback=1):
    at_ts = to_utc_safe(at_ts)
    d = df4h[df4h.index <= at_ts].copy()
    if d.empty:
        return {"ma_50": np.nan, "ma_200": np.nan, "ma_slow_slope": np.nan,
                "trend_bull": False, "trend_bear": False}
    px = d["close"]
    if ema:
        ma_fast = px.ewm(span=fast, adjust=False).mean()
        ma_slow = px.ewm(span=slow, adjust=False).mean()
    else:
        ma_fast = px.rolling(fast, min_periods=fast).mean()
        ma_slow = px.rolling(slow, min_periods=slow).mean()
    mf = float(ma_fast.iloc[-1]) if not pd.isna(ma_fast.iloc[-1]) else np.nan
    ms = float(ma_slow.iloc[-1]) if not pd.isna(ma_slow.iloc[-1]) else np.nan
    if len(ma_slow) > slope_lookback:
        slope = float(ma_slow.iloc[-1] - ma_slow.iloc[-1 - slope_lookback])
    else:
        slope = np.nan
    close = float(px.iloc[-1])
    bull = (close > ms) and (mf > ms) and (slope > 0)
    bear = (close < ms) and (mf < ms) and (slope < 0)
    return {"ma_50": mf, "ma_200": ms, "ma_slow_slope": slope,
            "trend_bull": bool(bull), "trend_bear": bool(bear)}

def _find_swing_range(df4h: pd.DataFrame, t0: pd.Timestamp, lookback: int, piv_len: int):
    bars = df4h[df4h.index < to_utc_safe(t0)].tail(lookback).copy()
    if bars.empty:
        return (np.nan, np.nan, pd.NaT, pd.NaT)
    win = piv_len * 2 + 1
    hi = bars["high"].rolling(win, center=True).apply(lambda a: 1.0 if a[piv_len] == np.max(a) else 0.0, raw=True)
    lo = bars["low"].rolling(win, center=True).apply(lambda a: 1.0 if a[piv_len] == np.min(a) else 0.0, raw=True)
    pv_hi = bars[hi == 1.0]
    pv_lo = bars[lo == 1.0]
    if pv_hi.empty or pv_lo.empty:
        L = float(bars["low"].min()); H = float(bars["high"].max())
        return (L, H, bars["low"].idxmin(), bars["high"].idxmax())
    last_hi = pv_hi.iloc[-1]; last_lo = pv_lo.iloc[-1]
    if last_hi.name > last_lo.name:
        L, Lt = float(pv_lo["low"].iloc[-1]), pv_lo.index[-1]
        H, Ht = float(pv_hi["high"].iloc[-1]), pv_hi.index[-1]
    else:
        L, Lt = float(pv_lo["low"].iloc[-1]), pv_lo.index[-1]
        H, Ht = float(pv_hi["high"].iloc[-1]), pv_hi.index[-1]
    return (L, H, Lt, Ht)

def _build_fib_levels(L: float, H: float, fib_set: List[float], side: str):
    if not np.isfinite(L) or not np.isfinite(H) or L<=0 or H<=0 or H<=L:
        return []
    if side.upper() == "BUY":
        return [L + (H-L)*r for r in fib_set]
    else:
        return [H - (H-L)*r for r in fib_set]

def _choose_fib_tp(levels: List[float], idx: int, tighten_steps: int = 0):
    if not levels:
        return np.nan, -1
    i = max(0, min(len(levels)-1, int(idx)))
    i = max(0, min(len(levels)-1, i - int(tighten_steps)))
    return float(levels[i]), i

def _pivots(series: pd.Series, piv_len: int, want: str):
    win = piv_len * 2 + 1
    if win < 3 or len(series) < win:
        return series.iloc[0:0]
    if want == "high":
        mask = series.rolling(win, center=True).apply(lambda a: 1.0 if a[piv_len] == np.max(a) else 0.0, raw=True)
    else:
        mask = series.rolling(win, center=True).apply(lambda a: 1.0 if a[piv_len] == np.min(a) else 0.0, raw=True)
    return series[mask == 1.0]

def detect_divergence(df4h: pd.DataFrame, at_ts: pd.Timestamp, side: str,
                      osc: pd.Series, piv_len=4, lookback=180,
                      confirm_bars=1, price_eps=0.002, osc_eps=2.0):
    """
    BUY(bull): price LL + osc HL; SELL(bear): price HH + osc LH.
    confirm_bars — сколько баров прошло после второго пивота (подтверждение).
    Доп.пороги: price_eps (относит.дрейф цены), osc_eps (дельта осциллятора).
    """
    side = str(side).upper()
    at_ts = to_utc_safe(at_ts)
    bars = df4h[df4h.index < at_ts].tail(lookback).copy()
    if bars.empty or osc is None or osc.empty:
        return False, None, None

    osc = osc.reindex(bars.index)
    pivH = _pivots(bars["high"], piv_len, "high").tail(2)
    pivL = _pivots(bars["low"],  piv_len, "low").tail(2)
    if side == "BUY" and len(pivL) == 2:
        p1, p2 = pivL.index[-2], pivL.index[-1]
        price_ok = float(bars.loc[p2, "low"]) < float(bars.loc[p1, "low"]) * (1 - price_eps)
        osc_ok = float(osc.loc[p2]) > float(osc.loc[p1]) + osc_eps
        idx2 = bars.index.get_indexer_for([p2])[0]
        confirmed = (len(bars) - 1 - idx2) >= (confirm_bars - 1)
        if price_ok and osc_ok and confirmed:
            return True, "bull", p2
    if side == "SELL" and len(pivH) == 2:
        p1, p2 = pivH.index[-2], pivH.index[-1]
        price_ok = float(bars.loc[p2, "high"]) > float(bars.loc[p1, "high"]) * (1 + price_eps)
        osc_ok = float(osc.loc[p2]) < float(osc.loc[p1]) - osc_eps
        idx2 = bars.index.get_indexer_for([p2])[0]
        confirmed = (len(bars) - 1 - idx2) >= (confirm_bars - 1)
        if price_ok and osc_ok and confirmed:
            return True, "bear", p2
    return False, None, None

def compute_entry_features(
    df4h: pd.DataFrame,
    entry_at: pd.Timestamp,
    side: str,
    *,
    # MA
    ma_fast: int = 50,
    ma_slow: int = 200,
    ma_use_ema: bool = False,
    ma_slope_lookback: int = 1,
    # FIB
    use_fib: bool = True,
    fib_lookback_bars: int = 120,
    fib_pivot_len: int = 3,
    fib_set: str = "0.236,0.382,0.5,0.618,0.786,1.0,1.272,1.618",
    fib_tp_index: int = 3,
    div_type: str = "off",      # "off"|"rsi"|"macd"
    rsi_period: int = 21,
    macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
    div_piv_len: int = 4, div_lookback: int = 180, div_confirm: int = 1,
    div_price_eps: float = 0.002, div_osc_eps: float = 2.0
) -> Dict[str, object]:
    """
    Единая точка формирования фичей (MA50/200, fib якоря и fib TP, дивергенция).
    Ничего не фильтрует — только вычисляет.
    """
    out = {
        "ma_50": np.nan, "ma_200": np.nan, "ma_slow_slope": np.nan,
        "trend_bull": False, "trend_bear": False,
        "fib_anchor_L": pd.NA, "fib_anchor_H": pd.NA, "fib_tp_4h": pd.NA,
        "div4h_flag": False, "div4h_type": pd.NA, "div4h_side": pd.NA, "div4h_confirm_at": pd.NaT,
    }
    if df4h is None or df4h.empty or pd.isna(entry_at):
        return out

    # MA
    out.update(ma_features(df4h, entry_at, fast=ma_fast, slow=ma_slow,
                           ema=ma_use_ema, slope_lookback=ma_slope_lookback))

    # FIB
    if use_fib:
        fib_list = [float(x) for x in str(fib_set).split(",") if str(x).strip()]
        L, H, _, _ = _find_swing_range(df4h, entry_at, fib_lookback_bars, fib_pivot_len)
        levels = _build_fib_levels(L, H, fib_list, side)
        tp_val, _ = _choose_fib_tp(levels, fib_tp_index, tighten_steps=0)
        out["fib_anchor_L"] = float(L) if np.isfinite(L) else pd.NA
        out["fib_anchor_H"] = float(H) if np.isfinite(H) else pd.NA
        out["fib_tp_4h"]    = float(tp_val) if np.isfinite(tp_val) else pd.NA

    # Divergence
    div_type = str(div_type).lower().strip()
    if div_type in ("rsi","macd"):
        if div_type == "rsi":
            osc = rsi(df4h["close"], rsi_period)
            osc_eps = float(div_osc_eps)
        else:
            ml, _, _ = macd(df4h["close"], macd_fast, macd_slow, macd_signal)
            osc = ml
            osc_eps = 0.0  # для macd линии обычно достаточно знака/наклона
        flag, kind, atp = detect_divergence(
            df4h, entry_at, side, osc,
            piv_len=div_piv_len, lookback=div_lookback, confirm_bars=div_confirm,
            price_eps=div_price_eps, osc_eps=osc_eps
        )
        out["div4h_flag"] = bool(flag)
        out["div4h_type"] = div_type
        out["div4h_side"] = kind
        out["div4h_confirm_at"] = atp

    return out

# =============================================================================
# CAPITAL SIM & POST
# =============================================================================
def enforce_one_at_a_time_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сохраняет исходный порядок строк.
    Внутри символа идём по текущему порядку и запрещаем перекрытия:
    если новая сделка стартует до закрытия предыдущей — помечаем её skipped.
    """
    if df is None or df.empty or "symbol" not in df.columns:
        return df

    out = df.copy()
    out["__pos"] = np.arange(len(out))  # запомним исходные позиции

    def _scan_group(g: pd.DataFrame) -> pd.DataFrame:
        last_close = pd.NaT
        rows = []
        for idx, r in g.iterrows():
            t_start = pd.to_datetime(r.get("t_start"), utc=True, errors="coerce")
            close_t = pd.to_datetime(r.get("close_time"), utc=True, errors="coerce")
            rec = r.copy()

            overlap = pd.notna(last_close) and pd.notna(t_start) and (t_start < last_close)
            if overlap:
                rec["skipped"] = True
                # не трогаем существующий exit_reason, но если его нет — ставим понятный
                if pd.isna(rec.get("exit_reason")) or str(rec.get("exit_reason")) == "None":
                    rec["exit_reason"] = "overlap_skip"
            else:
                # принять как активную, продвинуть "окно"
                if pd.notna(close_t):
                    last_close = close_t
            rows.append(rec)
        return pd.DataFrame(rows)

    grouped = []
    # важно: groupby без сортировки
    for sym, g in out.groupby("symbol", sort=False, group_keys=False):
        grouped.append(_scan_group(g))
    out2 = pd.concat(grouped, axis=0)

    # вернуть исходную стабильную последовательность
    out2 = out2.sort_values("__pos", kind="mergesort").drop(columns="__pos").reset_index(drop=True)
    return out2

def simulate_capital_passthrough(
    trades: pd.DataFrame,
    initial_equity: float,
    position_fraction: float,
    *,
    max_concurrent: int = None,
):
    """
    Pass-through симуляция:
      - НЕ меняет exit_reason/win
      - использует уже посчитанный pnl_pct
      - учитывает занятость капитала между t_start..close_time
      - аллокация = position_fraction * equity на момент ОТКРЫТИЯ
      - если нет свободного кэша — сделка помечается alloc_usd=0 (опционально: колонка no_capital=True)
      - equity изменяется ТОЛЬКО в момент закрытия сделки
    Возвращает: (df_out, eq_sheet)
      df_out: добавлены usd_alloc, pnl_usd, equity_after, pnl_usd_comp
      eq_sheet: кривая капитала по времени закрытий
    """
    import numpy as np
    import pandas as pd

    if trades is None or trades.empty:
        return trades.copy(), pd.DataFrame()

    df = trades.copy()

    # требуем ключевые поля
    for c in ("t_start", "close_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct"), errors="coerce")
    df["skipped"] = df.get("skipped")
    if "skipped" not in df.columns:
        df["skipped"] = False

    # глобальная хронология: закрытия должны идти после открытий с тем же временем
    df = df.sort_values(["t_start", "imb_time", "symbol"], kind="mergesort").reset_index(drop=True)

    # события: закрытия обрабатываем ПЕРЕД открытиями на одинаковом timestamp — освобождаем кэш
    open_ev  = pd.DataFrame({"ts": df["t_start"],    "kind": "open",  "idx": df.index})
    close_ev = pd.DataFrame({"ts": df["close_time"], "kind": "close", "idx": df.index})
    ev = pd.concat([open_ev, close_ev], ignore_index=True)
    ev = ev.dropna(subset=["ts"]).copy()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, errors="coerce")
    ev = ev[~ev["ts"].isna()]
    # порядок: сначала close, потом open; затем по imb_time, затем по symbol
    pr = pd.Series({"close": 0, "open": 1})
    ev["prio"] = ev["kind"].map(pr)
    # подмешаем детерминирующие ключи для стабильности
    ev["imb"] = df.loc[ev["idx"], "imb_time"].values
    ev["sym"] = df.loc[ev["idx"], "symbol"].values
    ev = ev.sort_values(["ts", "prio", "imb", "sym"], kind="mergesort").reset_index(drop=True)

    equity = float(initial_equity)
    free_cash = float(initial_equity)
    active = {}  # idx -> alloc
    eq_rows = []

    # подготовим колонки результата
    df["usd_alloc"]    = np.nan
    df["pnl_usd"]      = np.nan
    df["equity_after"] = np.nan
    df["pnl_usd_comp"] = np.nan
    df["no_capital"]   = False  # информативная метка, если аллокация не выдалась

    cum_pnl = 0.0

    for _, e in ev.iterrows():
        i   = int(e["idx"])
        kind= e["kind"]
        ts  = e["ts"]

        if kind == "close":
            # если позиция действительно открыта и аллокация > 0 — фиксируем PnL
            alloc = active.pop(i, 0.0)
            if alloc > 0:
                pct = float(df.at[i, "pnl_pct"]) if pd.notna(df.at[i, "pnl_pct"]) else 0.0
                pnl_usd = alloc * (pct / 100.0)
                cum_pnl += pnl_usd
                equity   += pnl_usd
                free_cash += alloc + pnl_usd
                df.at[i, "pnl_usd"] = pnl_usd
                df.at[i, "equity_after"] = equity
                df.at[i, "pnl_usd_comp"] = cum_pnl
            # пишем точку на эквити-кривой
            eq_rows.append({"time": ts.tz_convert(None), "equity": equity})

        else:  # open
            # пропускаем: пропущенные сделки, NaN pnl_pct, отсутствующие t_start/close_time
            if bool(df.at[i, "skipped"]) or not pd.notna(df.at[i, "pnl_pct"]) \
               or not pd.notna(df.at[i, "t_start"]) or not pd.notna(df.at[i, "close_time"]):
                df.at[i, "usd_alloc"] = 0.0
                continue

            # ограничение на число одновременных позиций (если задано)
            if max_concurrent is not None and len(active) >= int(max_concurrent):
                df.at[i, "usd_alloc"] = 0.0
                df.at[i, "no_capital"] = True
                continue

            # целевая аллокация
            target_alloc = float(position_fraction) * equity

            # если свободного кэша не хватает — сделка не берётся
            if free_cash + 1e-9 < target_alloc:
                df.at[i, "usd_alloc"] = 0.0
                df.at[i, "no_capital"] = True
                continue

            # блокируем кэш
            free_cash -= target_alloc
            active[i] = target_alloc
            df.at[i, "usd_alloc"] = target_alloc

    # на случай, если ни одного закрытия — всё равно вернём хотя бы стартовую точку
    if not eq_rows:
        eq_rows.append({"time": pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(None), "equity": equity})
    eq_sheet = pd.DataFrame(eq_rows).sort_values("time").reset_index(drop=True)

    return df, eq_sheet

import pandas as pd
def simulate_capital_notional(df: pd.DataFrame,
                              initial_capital: float,
                              position_fraction: float,
                              *,
                              stop_pct: float,
                              take_pct: float):
    """
    Простой капитал-сим: аллокация = текущий equity * position_fraction,
    PnL в $ = alloc * pnl_pct/100. Предполагаем, что сделки не перекрываются
    (перекрытия уже сняты enforce_one_at_a_time_per_symbol).
    Возвращает (df_sim, eq_sheet).
    """
    if df is None or df.empty:
        return df.copy(), pd.DataFrame()

    sim = df.copy()

    # гарантируем хронологию, чтобы equity шёл последовательно
    for c in ("t_start", "imb_time", "close_time"):
        if c in sim.columns:
            sim[c] = pd.to_datetime(sim[c], utc=True, errors="coerce")
    sort_keys = [c for c in ("t_start", "imb_time", "close_time") if c in sim.columns]
    if sort_keys:
        sim = sim.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    equity = float(initial_capital or 0.0)
    allocs = []
    pnls = []
    equities = []
    cum_pnls = []

    cum_pnl = 0.0
    exec_mask = sim.get("skipped") == False if "skipped" in sim.columns else pd.Series(True, index=sim.index)

    for i, row in sim.iterrows():
        if bool(exec_mask.iloc[i]):
            alloc = equity * float(position_fraction or 0.0)
            pnl_pct = float(row.get("pnl_pct", 0.0) or 0.0)
            pnl_usd = alloc * (pnl_pct / 100.0)
            equity = equity + pnl_usd
        else:
            alloc = np.nan
            pnl_usd = np.nan

        cum_pnl = (cum_pnl + (0.0 if np.isnan(pnl_usd) else float(pnl_usd)))
        allocs.append(alloc)
        pnls.append(pnl_usd)
        equities.append(equity)
        cum_pnls.append(cum_pnl)

    sim["usd_alloc"] = allocs
    sim["pnl_usd"] = pnls
    sim["equity_after"] = equities
    sim["pnl_usd_comp"] = cum_pnls

    # equity-curve (по времени закрытия, если есть; иначе по t_start)
    tcol = "close_time" if "close_time" in sim.columns else ("t_start" if "t_start" in sim.columns else None)
    if tcol is not None:
        eq = sim[[tcol, "equity_after"]].rename(columns={tcol: "time"})
        eq = eq.dropna(subset=["time"]).copy()
        eq["time"] = pd.to_datetime(eq["time"], utc=True, errors="coerce")
        eq = eq.dropna(subset=["time"]).sort_values("time")
    else:
        # на всякий случай — без времени просто индекс
        eq = pd.DataFrame({"time": pd.RangeIndex(len(sim)), "equity_after": sim["equity_after"]})

    return sim, eq

def safe_group_exit_reason(df_res: pd.DataFrame) -> pd.DataFrame:
    df = df_res.copy()
    if 'skipped' in df.columns:
        df = df[df['skipped'] == False].copy()
    for col, default in [
        ('exit_reason','unknown'), ('win',False), ('pnl_pct',0.0), ('pnl_usd',0.0), ('exit_days', pd.NA)
    ]:
        if col not in df.columns:
            df[col] = default
    df['win'] = df['win'].astype('bool')
    df['pnl_pct'] = pd.to_numeric(df['pnl_pct'], errors='coerce').fillna(0.0)
    df['pnl_usd'] = pd.to_numeric(df['pnl_usd'], errors='coerce').fillna(0.0)
    df['exit_days'] = pd.to_numeric(df['exit_days'], errors='coerce')

    if df.empty:
        return pd.DataFrame(columns=['exit_reason','trades','wins','winrate_pct','pnl_pct','pnl_usd','avg_exit_days','med_exit_days'])
    g = (df.groupby('exit_reason', dropna=False)
           .agg(trades=('win','size'), wins=('win','sum'),
                pnl_pct=('pnl_pct','sum'), pnl_usd=('pnl_usd','sum'),
                avg_exit_days=('exit_days','mean'), med_exit_days=('exit_days','median'))
           .reset_index())
    g['winrate_pct'] = g['wins'].div(g['trades']).fillna(0).astype(float).mul(100).round(2)
    return g.sort_values(['pnl_usd','winrate_pct'], ascending=[False, False])

# =============================================================================
# IO
# =============================================================================
def load_signals(signals_path: str, *, only_filled=False, dedup=False, require_entry: bool = True) -> pd.DataFrame:
    # читаем (если файл c листом 'data' — подхватим его)
    try:
        head = pd.read_excel(signals_path, nrows=0, sheet_name="data")
        parse_dates = [c for c in ["imb_time", "entry_at"] if c in head.columns]
        df = pd.read_excel(signals_path, parse_dates=parse_dates, sheet_name="data")
    except Exception:
        head = pd.read_excel(signals_path, nrows=0)
        parse_dates = [c for c in ["imb_time", "entry_at"] if c in head.columns]
        df = pd.read_excel(signals_path, parse_dates=parse_dates)

    # символы
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df = df[df["symbol"].notna()]

    # время
    if "imb_time" in df.columns:
        df["imb_time"] = pd.to_datetime(df["imb_time"], utc=True, errors="coerce")
        df = df[df["imb_time"].notna()]

    # сторона: side/type → type
    cols = {c.lower(): c for c in df.columns}
    if "type" not in df.columns and "side" in cols:
        df = df.rename(columns={cols["side"]: "type"})
    elif "type" in cols and cols["type"] != "type":
        df = df.rename(columns={cols["type"]: "type"})

    if "type" in df.columns:
        df["type"] = (
            df["type"]
            .astype(str).str.upper().str.strip()
            .map({"BUY": "BUY", "LONG": "BUY", "SELL": "SELL", "SHORT": "SELL"})
        )
        df = df[df["type"].isin(["BUY", "SELL"])]
    else:
        return pd.DataFrame()

    # entry (если требуется)
    if require_entry:
        if "entry" in df.columns:
            df["entry"] = pd.to_numeric(df["entry"], errors="coerce")
            df = df[df["entry"].notna()]
        else:
            return pd.DataFrame()

    # фильтр по filled (если есть)
    if "filled" in df.columns and only_filled:
        df["filled"] = df["filled"].map(lambda x: str(x).strip().lower() in ("1","true","yes","y","да","истина"))
        df = df[df["filled"] == True]

    # дедуп по (symbol, imb_time)
    if dedup and {"symbol","imb_time"}.issubset(df.columns):
        df = (
            df.sort_values(["symbol", "imb_time"])
              .drop_duplicates(subset=["symbol", "imb_time"], keep="first")
              .reset_index(drop=True)
        )

    return df

def load_price_cache(symbols: List[str], interval: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
    """
    Грузим исторические бары для расчётов (локально).
    Сейчас поддержан сценарий '4h' → get_klines_4h из минуток.
    """
    cache = {}
    for sym in symbols:
        try:
            df_hist = get_klines_4h(symbol=sym, lookback_days=lookback_days, interval=interval)
        except Exception:
            df_hist = pd.DataFrame()
        cache[sym] = ensure_dt_index(df_hist)
    return cache

def simulate_capital_passthrough(df: pd.DataFrame,
                                 initial_equity: float,
                                 position_fraction: float = 1.0):
    """
    Последовательная симуляция капитала БЕЗ изменения исходов сделок:
      usd_alloc = equity_before * position_fraction
      pnl_usd   = usd_alloc * (pnl_pct / 100)
      equity_after = equity_before + pnl_usд
    Требование: df уже отсортирован по времени и НЕ содержит перекрытий.
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        out["usd_alloc"] = pd.NA
        out["pnl_usd"] = pd.NA
        out["equity_after"] = initial_equity
        out["pnl_usd_comp"] = pd.NA
        return out, pd.DataFrame()

    x = df.copy().reset_index(drop=True)
    x["pnl_pct"] = pd.to_numeric(x.get("pnl_pct"), errors="coerce")

    equity = float(initial_equity)
    usd_alloc = []
    pnl_usd = []
    equity_after = []

    for _, r in x.iterrows():
        if bool(r.get("skipped", False)) or not np.isfinite(r.get("pnl_pct", np.nan)):
            usd_alloc.append(pd.NA)
            pnl_usd.append(pd.NA)
            equity_after.append(equity)
            continue

        alloc = equity * float(position_fraction)
        pl = alloc * (float(r["pnl_pct"]) / 100.0)
        equity = equity + pl

        usd_alloc.append(alloc)
        pnl_usd.append(pl)
        equity_after.append(equity)

    x["usd_alloc"] = usd_alloc
    x["pnl_usd"] = pnl_usd
    x["equity_after"] = equity_after
    # совместимость: дублированная колонка, если где-то суммируешь
    x["pnl_usd_comp"] = x["pnl_usd"]

    eq_sheet = x[["t_start", "equity_after"]].rename(columns={"t_start": "time"})
    return x, eq_sheet

def finalize_write(result_path: str,
                   df_out: pd.DataFrame,
                   eq_sheet: pd.DataFrame,
                   by_variant: pd.DataFrame,
                   by_exit_reason: pd.DataFrame,
                   extra_sheets: dict = None):

    import os
    import pandas as pd
    from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

    def _strip_tz_inplace(df: pd.DataFrame):
        if df is None or df.empty:
            return
        for col in df.columns:
            s = df[col]
            if is_datetime64tz_dtype(s):
                df[col] = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)
            elif is_datetime64_any_dtype(s):
                try:
                    df[col] = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
                except Exception:
                    df[col] = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)

    def _clean_objects_inplace(df: pd.DataFrame):
        if df is None or df.empty:
            return
        for c in df.columns:
            if df[c].dtype == "object":
                try:
                    df[c] = df[c].where(pd.notna(df[c]), None)
                except Exception:
                    pass

    df_out = df_out if isinstance(df_out, pd.DataFrame) else pd.DataFrame()
    eq_sheet = eq_sheet if isinstance(eq_sheet, pd.DataFrame) else pd.DataFrame()
    by_variant = by_variant if isinstance(by_variant, pd.DataFrame) else pd.DataFrame()
    by_exit_reason = by_exit_reason if isinstance(by_exit_reason, pd.DataFrame) else pd.DataFrame()
    extra_sheets = extra_sheets or {}

    _strip_tz_inplace(df_out)
    _strip_tz_inplace(eq_sheet)
    _strip_tz_inplace(by_variant)
    _strip_tz_inplace(by_exit_reason)
    for _, v in extra_sheets.items():
        _strip_tz_inplace(v if isinstance(v, pd.DataFrame) else pd.DataFrame())

    _clean_objects_inplace(df_out)
    _clean_objects_inplace(eq_sheet)
    _clean_objects_inplace(by_variant)
    _clean_objects_inplace(by_exit_reason)
    for _, v in extra_sheets.items():
        _clean_objects_inplace(v if isinstance(v, pd.DataFrame) else pd.DataFrame())

    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)

    with pd.ExcelWriter(result_path, engine="xlsxwriter") as wr:
        (df_out if not df_out.empty else pd.DataFrame()).to_excel(wr, sheet_name="trades", index=False)
        if not by_exit_reason.empty:
            by_exit_reason.to_excel(wr, sheet_name="by_exit_reason", index=False)
        if not by_variant.empty:
            by_variant.to_excel(wr, sheet_name="by_variant", index=False)
        if not eq_sheet.empty:
            eq_sheet.to_excel(wr, sheet_name="equity", index=False)
        for name, df in extra_sheets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(wr, sheet_name=str(name)[:31], index=False)