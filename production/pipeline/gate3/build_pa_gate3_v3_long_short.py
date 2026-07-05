from __future__ import annotations

import os
import glob
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# FS
# ============================================================

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_symbol_parquets(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "*.parquet")))


def _symbol_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _read_parquet(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def _ensure_dt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_label_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        lc = c.lower()
        if lc.startswith("y_") or "label" in lc:
            out.append(c)
    return out


# ============================================================
# NUMERIC HELPERS
# ============================================================

def _rolling_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 3:
        return float("nan")
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    yy = y.astype(float) - float(np.mean(y))
    denom = float(np.sum(x * x))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * yy) / denom)


def _ema(x: np.ndarray, span: int) -> float:
    if len(x) < 2:
        return float("nan")
    alpha = 2.0 / (span + 1.0)
    v = float(x[0])
    for i in range(1, len(x)):
        v = alpha * float(x[i]) + (1.0 - alpha) * v
    return v


def _calc_rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return float("nan")
    diff = np.diff(close.astype(float))
    up = np.clip(diff, 0, None)
    dn = -np.clip(diff, None, 0)
    au = np.mean(up[-period:])
    ad = np.mean(dn[-period:])
    if ad == 0:
        return 100.0
    rs = au / ad
    return float(100.0 - (100.0 / (1.0 + rs)))


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return tr.astype(float)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    tr = _true_range(high, low, close)
    if len(tr) < period:
        return float("nan")
    return float(np.mean(tr[-period:]))


def _zscore_last(x: np.ndarray) -> float:
    if len(x) < 5:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if sd == 0:
        return 0.0
    return float((x[-1] - mu) / sd)


def _ret(c: np.ndarray, n: int) -> float:
    if len(c) < n + 1:
        return float("nan")
    if c[-(n + 1)] == 0:
        return float("nan")
    return float(c[-1] / c[-(n + 1)] - 1.0)


# ============================================================
# CANDLES / PA PRIMITIVES
# ============================================================

def _candle_features(o: float, h: float, l: float, c: float) -> Dict[str, float]:
    rng = float(h - l)
    body = float(abs(c - o))
    up_wick = float(h - max(o, c))
    dn_wick = float(min(o, c) - l)
    body_pct = body / rng if rng > 0 else 0.0
    up_wick_pct = up_wick / rng if rng > 0 else 0.0
    dn_wick_pct = dn_wick / rng if rng > 0 else 0.0
    is_bull = 1.0 if c > o else 0.0
    is_bear = 1.0 if c < o else 0.0
    doji = 1.0 if rng > 0 and body_pct <= 0.1 else 0.0
    pin_up = 1.0 if rng > 0 and up_wick_pct >= 0.55 and body_pct <= 0.35 else 0.0
    pin_dn = 1.0 if rng > 0 and dn_wick_pct >= 0.55 and body_pct <= 0.35 else 0.0
    return {
        "pa_rng": rng,
        "pa_body": body,
        "pa_up_wick": up_wick,
        "pa_dn_wick": dn_wick,
        "pa_body_pct": float(body_pct),
        "pa_up_wick_pct": float(up_wick_pct),
        "pa_dn_wick_pct": float(dn_wick_pct),
        "pa_is_bull": is_bull,
        "pa_is_bear": is_bear,
        "pa_is_doji": doji,
        "pa_pin_up": pin_up,
        "pa_pin_dn": pin_dn,
    }


def _engulfing(prev_o: float, prev_c: float, o: float, c: float) -> Tuple[float, float]:
    prev_hi = max(prev_o, prev_c)
    prev_lo = min(prev_o, prev_c)
    cur_hi = max(o, c)
    cur_lo = min(o, c)
    bull = 1.0 if (c > o) and (prev_c < prev_o) and (cur_hi >= prev_hi) and (cur_lo <= prev_lo) else 0.0
    bear = 1.0 if (c < o) and (prev_c > prev_o) and (cur_hi >= prev_hi) and (cur_lo <= prev_lo) else 0.0
    return bull, bear


def _inside_outside(prev_h: float, prev_l: float, h: float, l: float) -> Tuple[float, float]:
    inside = 1.0 if (h <= prev_h) and (l >= prev_l) else 0.0
    outside = 1.0 if (h >= prev_h) and (l <= prev_l) else 0.0
    return inside, outside


def _swing_counts(high: np.ndarray, low: np.ndarray, lookback: int = 12) -> Tuple[float, float]:
    if len(high) < lookback + 2:
        return float("nan"), float("nan")
    hh = high[-(lookback + 2):]
    ll = low[-(lookback + 2):]
    swing_hi = 0
    swing_lo = 0
    for i in range(1, len(hh) - 1):
        if hh[i] > hh[i - 1] and hh[i] > hh[i + 1]:
            swing_hi += 1
        if ll[i] < ll[i - 1] and ll[i] < ll[i + 1]:
            swing_lo += 1
    return float(swing_hi), float(swing_lo)


def _roll_prev_max(x: np.ndarray, w: int) -> float:
    if len(x) < w + 1:
        return float("nan")
    return float(np.max(x[-(w + 1):-1]))


def _roll_prev_min(x: np.ndarray, w: int) -> float:
    if len(x) < w + 1:
        return float("nan")
    return float(np.min(x[-(w + 1):-1]))


def _median_safe(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.median(x.astype(float)))


# ============================================================
# WINDOW CHECK
# ============================================================

@dataclass
class WindowCheck:
    ok: bool
    missing_ratio: float
    missing_bars: int


def _get_window_strict(h4: pd.DataFrame, entry_ts: pd.Timestamp, context_bars: int) -> Tuple[Optional[pd.DataFrame], WindowCheck]:
    if entry_ts not in h4.index:
        return None, WindowCheck(False, 1.0, context_bars)

    pos = h4.index.get_loc(entry_ts)
    if isinstance(pos, slice) or isinstance(pos, np.ndarray):
        try:
            pos = int(np.asarray(pos).ravel()[0])
        except Exception:
            return None, WindowCheck(False, 1.0, context_bars)

    start = pos - (context_bars - 1)
    if start < 0:
        return None, WindowCheck(False, 1.0, context_bars)

    win = h4.iloc[start:pos + 1]
    if len(win) != context_bars:
        miss = context_bars - len(win)
        return None, WindowCheck(False, miss / context_bars, miss)

    diffs = win.index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    bad = int(np.sum(diffs != 14400.0))
    if bad > 0:
        return None, WindowCheck(False, bad / max(1, len(diffs)), bad)

    return win, WindowCheck(True, 0.0, 0)


def _load_h4_symbol(h4_root: str, symbol: str) -> pd.DataFrame:
    p = os.path.join(h4_root, f"{symbol}.parquet")
    df = _read_parquet(p)

    ts_col = _find_first_col(df, ["ts", "open_time", "time", "timestamp", "datetime"])
    if ts_col is None:
        raise RuntimeError(f"h4 missing time col for {symbol}: {p}")

    df[ts_col] = _ensure_dt_utc(df[ts_col])
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.rename(columns={ts_col: "ts"})

    need = ["open", "high", "low", "close"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"h4 missing {c} for {symbol}: {p}")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    df = df.drop_duplicates(subset=["ts"], keep="last")
    df = df.set_index("ts", drop=True)

    return df


# ============================================================
# OLD MARKET CONTEXT (must stay compatible)
# ============================================================

def _market_context_from_window(win: pd.DataFrame) -> Dict[str, float]:
    o = win["open"].to_numpy(dtype=float)
    h = win["high"].to_numpy(dtype=float)
    l = win["low"].to_numpy(dtype=float)
    c = win["close"].to_numpy(dtype=float)
    v = win["volume"].to_numpy(dtype=float) if "volume" in win.columns else None

    atr14 = _atr(h, l, c, 14)
    atr48 = _atr(h, l, c, 48)
    atrp14 = float(atr14 / c[-1]) if (not math.isnan(atr14)) and c[-1] != 0 else float("nan")
    atrp48 = float(atr48 / c[-1]) if (not math.isnan(atr48)) and c[-1] != 0 else float("nan")

    ema_fast = _ema(c[-96:] if len(c) >= 96 else c, 12)
    ema_slow = _ema(c[-96:] if len(c) >= 96 else c, 26)
    trend_ema = float((ema_fast - ema_slow) / c[-1]) if (not math.isnan(ema_fast)) and (not math.isnan(ema_slow)) and c[-1] != 0 else float("nan")

    slope_48 = _rolling_slope(c[-48:]) if len(c) >= 48 else float("nan")
    slope_96 = _rolling_slope(c[-96:]) if len(c) >= 96 else float("nan")

    rsi14 = _calc_rsi(c, 14)

    ret_12 = _ret(c, 12)
    ret_48 = _ret(c, 48)
    ret_96 = _ret(c, 96)

    hh48 = float(np.max(h[-48:])) if len(h) >= 48 else float(np.max(h))
    ll48 = float(np.min(l[-48:])) if len(l) >= 48 else float(np.min(l))
    range48 = float((hh48 - ll48) / c[-1]) if c[-1] != 0 else float("nan")

    if not math.isnan(atrp48):
        if atrp48 < 0.01:
            vol_cluster = 0.0
        elif atrp48 < 0.03:
            vol_cluster = 1.0
        else:
            vol_cluster = 2.0
    else:
        vol_cluster = float("nan")

    if not math.isnan(ret_48):
        if ret_48 > 0.05:
            regime = 1.0
        elif ret_48 < -0.05:
            regime = -1.0
        else:
            regime = 0.0
    else:
        regime = float("nan")

    if not math.isnan(trend_ema) and not math.isnan(atrp48) and atrp48 > 0:
        trend_strength_atr = float(trend_ema / atrp48)
    else:
        trend_strength_atr = float("nan")

    vol_z20 = float("nan")
    if v is not None and len(v) >= 20:
        vol_z20 = _zscore_last(v[-20:])

    return {
        "ctx_ret_12": ret_12,
        "ctx_ret_48": ret_48,
        "ctx_ret_96": ret_96,
        "ctx_slope_48": slope_48,
        "ctx_slope_96": slope_96,
        "ctx_ema_trend": trend_ema,
        "ctx_trend_strength_atr": trend_strength_atr,
        "ctx_rsi14": rsi14,
        "ctx_atr14": atr14,
        "ctx_atr48": atr48,
        "ctx_atrp14": atrp14,
        "ctx_atrp48": atrp48,
        "ctx_range48": range48,
        "ctx_vol_cluster3": vol_cluster,
        "ctx_regime_48": regime,
        "ctx_vol_z20": vol_z20,
    }


# ============================================================
# PA FEATURES V3 LONG+SHORT
# ============================================================

def _compute_pa_from_window(win: pd.DataFrame) -> Dict[str, float]:
    o = win["open"].to_numpy(dtype=float)
    h = win["high"].to_numpy(dtype=float)
    l = win["low"].to_numpy(dtype=float)
    c = win["close"].to_numpy(dtype=float)
    v = win["volume"].to_numpy(dtype=float) if "volume" in win.columns else None

    ret1 = _ret(c, 1)
    ret4 = _ret(c, 4)
    ret12 = _ret(c, 12)
    ret48 = _ret(c, 48)

    atr14 = _atr(h, l, c, 14)
    atr48 = _atr(h, l, c, 48)
    atrp14 = float(atr14 / c[-1]) if (not math.isnan(atr14)) and c[-1] != 0 else float("nan")
    atrp48 = float(atr48 / c[-1]) if (not math.isnan(atr48)) and c[-1] != 0 else float("nan")

    slope_12 = _rolling_slope(c[-12:]) if len(c) >= 12 else float("nan")
    slope_48 = _rolling_slope(c[-48:]) if len(c) >= 48 else float("nan")

    rsi14 = _calc_rsi(c, 14)

    cf = _candle_features(o[-1], h[-1], l[-1], c[-1])
    prev_cf = _candle_features(o[-2], h[-2], l[-2], c[-2]) if len(c) >= 2 else {}

    engulf_bull, engulf_bear = _engulfing(o[-2], c[-2], o[-1], c[-1]) if len(c) >= 2 else (float("nan"), float("nan"))
    inside, outside = _inside_outside(h[-2], l[-2], h[-1], l[-1]) if len(c) >= 2 else (float("nan"), float("nan"))

    hh48 = float(np.max(h[-48:])) if len(h) >= 48 else float(np.max(h))
    ll48 = float(np.min(l[-48:])) if len(l) >= 48 else float(np.min(l))
    dist_to_hh = float((hh48 - c[-1]) / c[-1]) if c[-1] != 0 else float("nan")
    dist_to_ll = float((c[-1] - ll48) / c[-1]) if c[-1] != 0 else float("nan")

    swing_hi, swing_lo = _swing_counts(h, l, 12)

    vol_z = float("nan")
    if v is not None and len(v) >= 20:
        vol_z = _zscore_last(v[-20:])

    # -------- directional PA --------
    prev_hi_12 = _roll_prev_max(h, 12)
    prev_lo_12 = _roll_prev_min(l, 12)
    prev_hi_24 = _roll_prev_max(h, 24)
    prev_lo_24 = _roll_prev_min(l, 24)
    prev_hi_48 = _roll_prev_max(h, 48)
    prev_lo_48 = _roll_prev_min(l, 48)

    prev2_hi_12 = _roll_prev_max(h[:-1], 12) if len(h) >= 14 else float("nan")
    prev2_lo_12 = _roll_prev_min(l[:-1], 12) if len(l) >= 14 else float("nan")

    slope_12_prev1 = _rolling_slope(c[-13:-1]) if len(c) >= 13 else float("nan")
    ret12_prev1 = float(c[-2] / c[-14] - 1.0) if len(c) >= 14 and c[-14] != 0 else float("nan")

    prev_range_48 = float(prev_hi_48 - prev_lo_48) if np.isfinite(prev_hi_48) and np.isfinite(prev_lo_48) else float(
        "nan")
    prev_range_48_pct = float(prev_range_48 / c[-1]) if np.isfinite(prev_range_48) and c[-1] != 0 else float("nan")
    breakout_margin_pct = float(max(0.15 * atrp14, 0.0025)) if np.isfinite(atrp14) else float("nan")
    body_pct = cf["pa_body_pct"]
    up_wick_pct = cf["pa_up_wick_pct"]
    dn_wick_pct = cf["pa_dn_wick_pct"]

    vol_med20 = _median_safe(v[-20:]) if v is not None and len(v) >= 20 else float("nan")
    vol_boost = float(v[-1] > vol_med20) if v is not None and np.isfinite(vol_med20) else 0.0

    rng = float(h[-1] - l[-1])
    tr_arr = _true_range(h, l, c)
    tr_med20 = _median_safe(tr_arr[-20:]) if len(tr_arr) >= 20 else float("nan")

    is_squeeze = 1.0 if np.isfinite(atrp14) and np.isfinite(atrp48) and atrp14 < atrp48 * 0.85 else 0.0

    bos_up_12 = 1.0 if np.isfinite(prev_hi_12) and c[-1] > prev_hi_12 else 0.0
    bos_dn_12 = 1.0 if np.isfinite(prev_lo_12) and c[-1] < prev_lo_12 else 0.0

    bos_up_24 = 1.0 if np.isfinite(prev_hi_24) and c[-1] > prev_hi_24 else 0.0
    bos_dn_24 = 1.0 if np.isfinite(prev_lo_24) and c[-1] < prev_lo_24 else 0.0

    choch_up_48_12 = 1.0 if (
            bos_up_12 == 1.0
            and np.isfinite(ret12_prev1)
            and np.isfinite(slope_12_prev1)
            and ret12_prev1 < 0.0
            and slope_12_prev1 <= 0.0
    ) else 0.0
    choch_dn_48_12 = 1.0 if (
            bos_dn_12 == 1.0
            and np.isfinite(ret12_prev1)
            and np.isfinite(slope_12_prev1)
            and ret12_prev1 > 0.0
            and slope_12_prev1 >= 0.0
    ) else 0.0

    sweep_hi_reject_dn_12 = 1.0 if np.isfinite(prev_hi_12) and (h[-1] > prev_hi_12) and (c[-1] < prev_hi_12) and (
                up_wick_pct >= 0.35) else 0.0
    sweep_lo_reject_up_12 = 1.0 if np.isfinite(prev_lo_12) and (l[-1] < prev_lo_12) and (c[-1] > prev_lo_12) and (
                dn_wick_pct >= 0.35) else 0.0

    range_break_up_48 = 1.0 if (
            np.isfinite(prev_hi_48)
            and c[-1] > prev_hi_48
            and np.isfinite(prev_range_48_pct)
            and np.isfinite(atrp14)
            and np.isfinite(breakout_margin_pct)
            and prev_range_48_pct <= max(6.0 * atrp14, 0.08)
            and ((c[-1] / prev_hi_48) - 1.0) >= breakout_margin_pct
            and len(c) >= 4
            and float(np.max(c[-4:-1])) <= prev_hi_48
    ) else 0.0
    range_break_dn_48 = 1.0 if (
            np.isfinite(prev_lo_48)
            and c[-1] < prev_lo_48
            and np.isfinite(prev_range_48_pct)
            and np.isfinite(atrp14)
            and np.isfinite(breakout_margin_pct)
            and prev_range_48_pct <= max(6.0 * atrp14, 0.08)
            and ((prev_lo_48 / c[-1]) - 1.0) >= breakout_margin_pct
            and len(c) >= 4
            and float(np.min(c[-4:-1])) >= prev_lo_48
    ) else 0.0

    ema_fast = _ema(c[-48:] if len(c) >= 48 else c, 12)
    ema_slow = _ema(c[-48:] if len(c) >= 48 else c, 26)

    trend_pullback_up = 1.0 if (np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast > ema_slow and c[-1] > ema_slow and l[-1] <= ema_fast) else 0.0
    trend_pullback_dn = 1.0 if (np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast < ema_slow and c[-1] < ema_slow and h[-1] >= ema_fast) else 0.0

    atr_squeeze_break_up = 1.0 if (is_squeeze == 1.0 and np.isfinite(prev_hi_12) and c[-1] > prev_hi_12) else 0.0
    atr_squeeze_break_dn = 1.0 if (is_squeeze == 1.0 and np.isfinite(prev_lo_12) and c[-1] < prev_lo_12) else 0.0

    bos_up_12_close_top20 = 1.0 if (bos_up_12 == 1.0 and body_pct >= 0.35 and c[-1] >= (h[-1] - 0.2 * max(rng, 1e-12))) else 0.0
    bos_dn_12_close_bot20 = 1.0 if (bos_dn_12 == 1.0 and body_pct >= 0.35 and c[-1] <= (l[-1] + 0.2 * max(rng, 1e-12))) else 0.0

    bos_up_12_volz20_high = 1.0 if (bos_up_12 == 1.0 and vol_boost == 1.0) else 0.0
    bos_dn_12_volz20_high = 1.0 if (bos_dn_12 == 1.0 and vol_boost == 1.0) else 0.0

    range_break_up_48_ret4pos = 1.0 if (range_break_up_48 == 1.0 and np.isfinite(ret4) and ret4 > 0) else 0.0
    range_break_dn_48_ret4neg = 1.0 if (range_break_dn_48 == 1.0 and np.isfinite(ret4) and ret4 < 0) else 0.0

    sweep_hi_dn_12_confirm1 = 1.0 if (sweep_hi_reject_dn_12 == 1.0 and c[-1] < o[-1]) else 0.0
    sweep_lo_up_12_confirm1 = 1.0 if (sweep_lo_reject_up_12 == 1.0 and c[-1] > o[-1]) else 0.0

    sweep_hi_dn_12_reclaim2 = 1.0 if (
            np.isfinite(prev2_hi_12)
            and len(c) >= 2
            and h[-2] > prev2_hi_12
            and c[-1] < prev2_hi_12
    ) else 0.0
    sweep_lo_up_12_reclaim2 = 1.0 if (
            np.isfinite(prev2_lo_12)
            and len(c) >= 2
            and l[-2] < prev2_lo_12
            and c[-1] > prev2_lo_12
    ) else 0.0

    squeeze_and_rngatr_comp = 1.0 if (is_squeeze == 1.0 and np.isfinite(tr_med20) and rng < tr_med20 * 0.8) else 0.0
    squeeze_break_up_trend = 1.0 if (atr_squeeze_break_up == 1.0 and np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast > ema_slow) else 0.0
    squeeze_break_dn_trend = 1.0 if (atr_squeeze_break_dn == 1.0 and np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast < ema_slow) else 0.0

    choch_up_48_12_ema_cross = 1.0 if (choch_up_48_12 == 1.0 and np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast > ema_slow) else 0.0
    choch_dn_48_12_ema_cross = 1.0 if (choch_dn_48_12 == 1.0 and np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast < ema_slow) else 0.0

    choch_up_48_12_ret12pos = 1.0 if (choch_up_48_12 == 1.0 and np.isfinite(ret12) and ret12 > 0) else 0.0
    choch_dn_48_12_ret12neg = 1.0 if (choch_dn_48_12 == 1.0 and np.isfinite(ret12) and ret12 < 0) else 0.0

    # extra mirrored / short-supportive columns
    bos_up_48 = 1.0 if np.isfinite(prev_hi_48) and c[-1] > prev_hi_48 else 0.0
    bos_dn_48 = 1.0 if np.isfinite(prev_lo_48) and c[-1] < prev_lo_48 else 0.0

    sweep_hi_confirm_dn = 1.0 if (sweep_hi_reject_dn_12 == 1.0 and body_pct >= 0.25 and cf["pa_is_bear"] == 1.0) else 0.0
    sweep_lo_confirm_up = 1.0 if (sweep_lo_reject_up_12 == 1.0 and body_pct >= 0.25 and cf["pa_is_bull"] == 1.0) else 0.0

    pullback_sell_from_ema = 1.0 if (trend_pullback_dn == 1.0 and up_wick_pct >= 0.25) else 0.0
    pullback_buy_from_ema = 1.0 if (trend_pullback_up == 1.0 and dn_wick_pct >= 0.25) else 0.0

    out = {
        "pa_ret_1": ret1,
        "pa_ret_4": ret4,
        "pa_ret_12": ret12,
        "pa_ret_48": ret48,
        "pa_atr14": atr14,
        "pa_atr48": atr48,
        "pa_atrp14": atrp14,
        "pa_atrp48": atrp48,
        "pa_slope_12": slope_12,
        "pa_slope_48": slope_48,
        "pa_rsi14": rsi14,
        "pa_engulf_bull": float(engulf_bull),
        "pa_engulf_bear": float(engulf_bear),
        "pa_inside": float(inside),
        "pa_outside": float(outside),
        "pa_dist_to_hh48": dist_to_hh,
        "pa_dist_to_ll48": dist_to_ll,
        "pa_swing_hi_12": swing_hi,
        "pa_swing_lo_12": swing_lo,
        "pa_vol_z20": vol_z,

        # old directional block - MUST stay
        "pa_bos_up_12": bos_up_12,
        "pa_bos_dn_12": bos_dn_12,
        "pa_bos_up_24": bos_up_24,
        "pa_bos_dn_24": bos_dn_24,
        "pa_choch_up_48_12": choch_up_48_12,
        "pa_choch_dn_48_12": choch_dn_48_12,
        "pa_sweep_hi_reject_dn_12": sweep_hi_reject_dn_12,
        "pa_sweep_lo_reject_up_12": sweep_lo_reject_up_12,
        "pa_range_break_up_48": range_break_up_48,
        "pa_range_break_dn_48": range_break_dn_48,
        "pa_trend_pullback_up": trend_pullback_up,
        "pa_trend_pullback_dn": trend_pullback_dn,
        "pa_atr_squeeze": is_squeeze,
        "pa_atr_squeeze_break_up": atr_squeeze_break_up,
        "pa_atr_squeeze_break_dn": atr_squeeze_break_dn,
        "pa_bos_up_12_close_top20": bos_up_12_close_top20,
        "pa_bos_dn_12_close_bot20": bos_dn_12_close_bot20,
        "pa_bos_up_12_volz20_high": bos_up_12_volz20_high,
        "pa_bos_dn_12_volz20_high": bos_dn_12_volz20_high,
        "pa_range_break_up_48_ret4pos": range_break_up_48_ret4pos,
        "pa_range_break_dn_48_ret4neg": range_break_dn_48_ret4neg,
        "pa_sweep_hi_dn_12_confirm1": sweep_hi_dn_12_confirm1,
        "pa_sweep_lo_up_12_confirm1": sweep_lo_up_12_confirm1,
        "pa_sweep_hi_dn_12_reclaim2": sweep_hi_dn_12_reclaim2,
        "pa_sweep_lo_up_12_reclaim2": sweep_lo_up_12_reclaim2,
        "pa_squeeze_and_rngatr_comp": squeeze_and_rngatr_comp,
        "pa_squeeze_break_up_trend": squeeze_break_up_trend,
        "pa_squeeze_break_dn_trend": squeeze_break_dn_trend,
        "pa_choch_up_48_12_ema_cross": choch_up_48_12_ema_cross,
        "pa_choch_dn_48_12_ema_cross": choch_dn_48_12_ema_cross,
        "pa_choch_up_48_12_ret12pos": choch_up_48_12_ret12pos,
        "pa_choch_dn_48_12_ret12neg": choch_dn_48_12_ret12neg,

        # new extra columns for future short/long expansion
        "pa_bos_up_48": bos_up_48,
        "pa_bos_dn_48": bos_dn_48,
        "pa_sweep_hi_confirm_dn": sweep_hi_confirm_dn,
        "pa_sweep_lo_confirm_up": sweep_lo_confirm_up,
        "pa_pullback_sell_from_ema": pullback_sell_from_ema,
        "pa_pullback_buy_from_ema": pullback_buy_from_ema,
    }

    out.update(cf)
    out["pa_prev_body_pct"] = float(prev_cf.get("pa_body_pct", float("nan"))) if prev_cf else float("nan")
    out["pa_prev_is_doji"] = float(prev_cf.get("pa_is_doji", float("nan"))) if prev_cf else float("nan")

    return out


# ============================================================
# STATEFUL ACTIVE ENGINE
# ============================================================

def _build_stateful_active_series(
    event: np.ndarray,
    close: np.ndarray,
    atr14_series: np.ndarray,
    side: str,
    max_active_bars: int,
    stop_atr_mult: float,
) -> np.ndarray:
    x = np.asarray(event, dtype=int)
    c = np.asarray(close, dtype=float)
    a = np.asarray(atr14_series, dtype=float)

    out = np.zeros(len(x), dtype=int)

    active = False
    bars_in = 0
    start_close = np.nan
    start_atr = np.nan

    for i in range(len(x)):
        if x[i] == 1:
            active = True
            bars_in = 0
            start_close = c[i]
            start_atr = a[i] if np.isfinite(a[i]) and a[i] > 0 else np.nan

        if active:
            out[i] = 1
            bars_in += 1

            if bars_in >= max_active_bars:
                active = False
                continue

            if np.isfinite(start_close) and np.isfinite(start_atr) and start_atr > 0:
                if side == "long":
                    stop_level = start_close - stop_atr_mult * start_atr
                    if c[i] < stop_level:
                        active = False
                        continue
                else:
                    stop_level = start_close + stop_atr_mult * start_atr
                    if c[i] > stop_level:
                        active = False
                        continue

    return out


def _safe_int(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").fillna(0).astype(int)
def _event_from_df(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=int)
    return _safe_int(df[col]).to_numpy(dtype=int)

def _pattern_policy(raw_col: str, default_max_active_bars: int, default_stop_atr_mult: float) -> Tuple[int, float]:
    cfg = ACTIVE_PATTERN_POLICY.get(raw_col, None)
    if cfg is None:
        return int(default_max_active_bars), float(default_stop_atr_mult)
    return int(cfg["max_active_bars"]), float(cfg["stop_atr_mult"])

def _compute_stateful_active_block(
    df: pd.DataFrame,
    max_active_bars: int,
    stop_atr_mult: float,
) -> pd.DataFrame:
    close = df["close"].to_numpy(dtype=float)
    atr14 = df["ctx_atr14"].to_numpy(dtype=float) if "ctx_atr14" in df.columns else np.full(len(df), np.nan)

    specs = [
        ("active_pa_atr_squeeze_break_up", "pa_atr_squeeze_break_up", "long"),
        ("active_pa_atr_squeeze_break_dn", "pa_atr_squeeze_break_dn", "short"),

        ("active_pa_bos_up_12", "pa_bos_up_12", "long"),
        ("active_pa_bos_dn_12", "pa_bos_dn_12", "short"),
        ("active_pa_bos_up_24", "pa_bos_up_24", "long"),
        ("active_pa_bos_dn_24", "pa_bos_dn_24", "short"),
        ("active_pa_bos_up_48", "pa_bos_up_48", "long"),
        ("active_pa_bos_dn_48", "pa_bos_dn_48", "short"),

        ("active_pa_choch_up_48_12", "pa_choch_up_48_12", "long"),
        ("active_pa_choch_dn_48_12", "pa_choch_dn_48_12", "short"),

        ("active_pa_range_break_up_48", "pa_range_break_up_48", "long"),
        ("active_pa_range_break_dn_48", "pa_range_break_dn_48", "short"),

        ("active_pa_trend_pullback_up", "pa_trend_pullback_up", "long"),
        ("active_pa_trend_pullback_dn", "pa_trend_pullback_dn", "short"),

        ("active_pa_sweep_lo_reject_up_12", "pa_sweep_lo_reject_up_12", "long"),
        ("active_pa_sweep_hi_reject_dn_12", "pa_sweep_hi_reject_dn_12", "short"),

        ("active_pa_sweep_lo_confirm_up", "pa_sweep_lo_confirm_up", "long"),
        ("active_pa_sweep_hi_confirm_dn", "pa_sweep_hi_confirm_dn", "short"),

        ("active_pa_sweep_lo_up_12_confirm1", "pa_sweep_lo_up_12_confirm1", "long"),
        ("active_pa_sweep_hi_dn_12_confirm1", "pa_sweep_hi_dn_12_confirm1", "short"),

        ("active_pa_sweep_lo_up_12_reclaim2", "pa_sweep_lo_up_12_reclaim2", "long"),
        ("active_pa_sweep_hi_dn_12_reclaim2", "pa_sweep_hi_dn_12_reclaim2", "short"),

        ("active_pa_squeeze_break_up_trend", "pa_squeeze_break_up_trend", "long"),
        ("active_pa_squeeze_break_dn_trend", "pa_squeeze_break_dn_trend", "short"),

        ("active_pa_pullback_buy_from_ema", "pa_pullback_buy_from_ema", "long"),
        ("active_pa_pullback_sell_from_ema", "pa_pullback_sell_from_ema", "short"),
    ]

    out = {}
    for active_col, raw_col, side in specs:
        event = _event_from_df(df, raw_col)
        pat_max_active_bars, pat_stop_atr_mult = _pattern_policy(
            raw_col=raw_col,
            default_max_active_bars=max_active_bars,
            default_stop_atr_mult=stop_atr_mult,
        )
        out[active_col] = _build_stateful_active_series(
            event=event,
            close=close,
            atr14_series=atr14,
            side=side,
            max_active_bars=pat_max_active_bars,
            stop_atr_mult=pat_stop_atr_mult,
        )

    return pd.DataFrame(out, index=df.index)

# ============================================================
# EXACT OLD 96 COLS
# ============================================================

OLD_96_COLUMNS = [
    "symbol",
    "entry_ts",
    "exit_ts",
    "entry_px",
    "exit_px",
    "exit_reason",
    "side",
    "y_fast",
    "pa_valid",
    "pa_missing_ratio",
    "pa_missing_bars",
    "pa_ret_1",
    "pa_ret_4",
    "pa_ret_12",
    "pa_ret_48",
    "pa_atr14",
    "pa_atr48",
    "pa_atrp14",
    "pa_atrp48",
    "pa_slope_12",
    "pa_slope_48",
    "pa_rsi14",
    "pa_engulf_bull",
    "pa_engulf_bear",
    "pa_inside",
    "pa_outside",
    "pa_dist_to_hh48",
    "pa_dist_to_ll48",
    "pa_swing_hi_12",
    "pa_swing_lo_12",
    "pa_vol_z20",
    "pa_rng",
    "pa_body",
    "pa_up_wick",
    "pa_dn_wick",
    "pa_body_pct",
    "pa_up_wick_pct",
    "pa_dn_wick_pct",
    "pa_is_bull",
    "pa_is_bear",
    "pa_is_doji",
    "pa_pin_up",
    "pa_pin_dn",
    "pa_prev_body_pct",
    "pa_prev_is_doji",
    "ctx_ret_12",
    "ctx_ret_48",
    "ctx_ret_96",
    "ctx_slope_48",
    "ctx_slope_96",
    "ctx_ema_trend",
    "ctx_trend_strength_atr",
    "ctx_rsi14",
    "ctx_atr14",
    "ctx_atr48",
    "ctx_atrp14",
    "ctx_atrp48",
    "ctx_range48",
    "ctx_vol_cluster3",
    "ctx_regime_48",
    "ctx_vol_z20",
    "pa_bos_up_12",
    "pa_bos_dn_12",
    "pa_bos_up_24",
    "pa_bos_dn_24",
    "pa_choch_up_48_12",
    "pa_choch_dn_48_12",
    "pa_sweep_hi_reject_dn_12",
    "pa_sweep_lo_reject_up_12",
    "pa_range_break_up_48",
    "pa_range_break_dn_48",
    "pa_trend_pullback_up",
    "pa_trend_pullback_dn",
    "pa_atr_squeeze",
    "pa_atr_squeeze_break_up",
    "pa_atr_squeeze_break_dn",
    "pa_bos_up_12_close_top20",
    "pa_bos_dn_12_close_bot20",
    "pa_bos_up_12_volz20_high",
    "pa_bos_dn_12_volz20_high",
    "pa_range_break_up_48_ret4pos",
    "pa_range_break_dn_48_ret4neg",
    "pa_sweep_hi_dn_12_confirm1",
    "pa_sweep_lo_up_12_confirm1",
    "pa_sweep_hi_dn_12_reclaim2",
    "pa_sweep_lo_up_12_reclaim2",
    "pa_squeeze_and_rngatr_comp",
    "pa_squeeze_break_up_trend",
    "pa_squeeze_break_dn_trend",
    "pa_choch_up_48_12_ema_cross",
    "pa_choch_dn_48_12_ema_cross",
    "pa_choch_up_48_12_ret12pos",
    "pa_choch_dn_48_12_ret12neg",
    "active_pa_atr_squeeze_break_up",
    "active_pa_atr_squeeze_break_dn",
    "active_pa_bos_up_24",
]
ACTIVE_V3_EXTRA_COLUMNS = [
    "active_pa_bos_up_12",
    "active_pa_bos_dn_12",
    "active_pa_bos_dn_24",
    "active_pa_bos_up_48",
    "active_pa_bos_dn_48",
    "active_pa_choch_up_48_12",
    "active_pa_choch_dn_48_12",
    "active_pa_range_break_up_48",
    "active_pa_range_break_dn_48",
    "active_pa_trend_pullback_up",
    "active_pa_trend_pullback_dn",
    "active_pa_sweep_lo_reject_up_12",
    "active_pa_sweep_hi_reject_dn_12",
    "active_pa_sweep_lo_confirm_up",
    "active_pa_sweep_hi_confirm_dn",
    "active_pa_sweep_lo_up_12_confirm1",
    "active_pa_sweep_hi_dn_12_confirm1",
    "active_pa_sweep_lo_up_12_reclaim2",
    "active_pa_sweep_hi_dn_12_reclaim2",
    "active_pa_squeeze_break_up_trend",
    "active_pa_squeeze_break_dn_trend",
    "active_pa_pullback_buy_from_ema",
    "active_pa_pullback_sell_from_ema",
]
ACTIVE_PATTERN_POLICY = {
    "pa_atr_squeeze_break_up": {"max_active_bars": 8, "stop_atr_mult": 1.35},
    "pa_atr_squeeze_break_dn": {"max_active_bars": 8, "stop_atr_mult": 1.35},

    "pa_bos_up_12": {"max_active_bars": 5, "stop_atr_mult": 1.10},
    "pa_bos_dn_12": {"max_active_bars": 5, "stop_atr_mult": 1.10},
    "pa_bos_up_24": {"max_active_bars": 6, "stop_atr_mult": 1.20},
    "pa_bos_dn_24": {"max_active_bars": 6, "stop_atr_mult": 1.20},
    "pa_bos_up_48": {"max_active_bars": 7, "stop_atr_mult": 1.25},
    "pa_bos_dn_48": {"max_active_bars": 7, "stop_atr_mult": 1.25},

    "pa_choch_up_48_12": {"max_active_bars": 6, "stop_atr_mult": 1.20},
    "pa_choch_dn_48_12": {"max_active_bars": 6, "stop_atr_mult": 1.20},

    "pa_range_break_up_48": {"max_active_bars": 7, "stop_atr_mult": 1.25},
    "pa_range_break_dn_48": {"max_active_bars": 7, "stop_atr_mult": 1.25},

    "pa_trend_pullback_up": {"max_active_bars": 5, "stop_atr_mult": 1.00},
    "pa_trend_pullback_dn": {"max_active_bars": 5, "stop_atr_mult": 1.00},
    "pa_pullback_buy_from_ema": {"max_active_bars": 5, "stop_atr_mult": 0.95},
    "pa_pullback_sell_from_ema": {"max_active_bars": 5, "stop_atr_mult": 0.95},

    "pa_sweep_lo_reject_up_12": {"max_active_bars": 3, "stop_atr_mult": 0.90},
    "pa_sweep_hi_reject_dn_12": {"max_active_bars": 3, "stop_atr_mult": 0.90},
    "pa_sweep_lo_confirm_up": {"max_active_bars": 4, "stop_atr_mult": 0.95},
    "pa_sweep_hi_confirm_dn": {"max_active_bars": 4, "stop_atr_mult": 0.95},
    "pa_sweep_lo_up_12_confirm1": {"max_active_bars": 4, "stop_atr_mult": 0.95},
    "pa_sweep_hi_dn_12_confirm1": {"max_active_bars": 4, "stop_atr_mult": 0.95},
    "pa_sweep_lo_up_12_reclaim2": {"max_active_bars": 4, "stop_atr_mult": 0.95},
    "pa_sweep_hi_dn_12_reclaim2": {"max_active_bars": 4, "stop_atr_mult": 0.95},

    "pa_squeeze_break_up_trend": {"max_active_bars": 7, "stop_atr_mult": 1.25},
    "pa_squeeze_break_dn_trend": {"max_active_bars": 7, "stop_atr_mult": 1.25},
}

# ============================================================
# MAIN BUILDER
# ============================================================

def build_gate3_v3_long_short(
    gate1_root: str,
    h4_root: str,
    out_root: str,
    context_bars: int,
    active_max_bars: int,
    active_stop_atr_mult: float,
    max_symbols: Optional[int] = None,
    symbols: Optional[List[str]] = None,
) -> None:
    out_by_symbol = os.path.join(out_root, "pa_gate3_v3_long_short_by_symbol")
    _safe_mkdir(out_by_symbol)

    files = _list_symbol_parquets(gate1_root)
    if symbols:
        wanted = set(symbols)
        files = [p for p in files if _symbol_from_path(p) in wanted]
    if max_symbols is not None:
        files = files[:max_symbols]

    audit_rows = []
    all_rows = []
    policy_rows = []

    for fp in files:
        symbol = _symbol_from_path(fp)

        df = _read_parquet(fp)
        if "symbol" not in df.columns:
            df["symbol"] = symbol

        if "entry_ts" not in df.columns:
            raise RuntimeError(f"ks file missing entry_ts: {fp}")

        if len(df) == 0:
            print(f"[SKIP] {symbol}: empty gate1 dataset")
            continue

        if df["entry_ts"].isna().all():
            print(f"[SKIP] {symbol}: entry_ts all NaT")
            continue

        df["entry_ts"] = _ensure_dt_utc(df["entry_ts"])
        # === STRICT TIME ALIGNMENT CHECK ===
        df = df.sort_values("entry_ts").reset_index(drop=True)

        dt = df["entry_ts"].diff().dropna().dt.total_seconds()
        bad = (dt != 4 * 3600).sum()

        if bad > 0:
            print(f"[WARNING] {symbol}: {bad} non-4h gaps in gate1 dataset")
        if "exit_ts" in df.columns:
            df["exit_ts"] = _ensure_dt_utc(df["exit_ts"])

        df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts")

        label_cols = _pick_label_cols(list(df.columns))

        keep_cols = ["symbol", "entry_ts"]
        for c in ["exit_ts", "entry_px", "exit_px", "exit_reason", "side"]:
            if c in df.columns:
                keep_cols.append(c)

        keep_cols += [c for c in label_cols if c not in keep_cols]


        base = df.copy()
        base = df[keep_cols].copy()
        h4 = _load_h4_symbol(h4_root, symbol)

        rows = []
        ok_cnt = 0
        miss_cnt = 0
        miss_bars_sum = 0

        # сначала посчитаем все "события"
        for r in base.itertuples(index=False):
            entry_ts = pd.Timestamp(getattr(r, "entry_ts"))

            win, chk = _get_window_strict(h4, entry_ts, context_bars)

            out = {k: getattr(r, k) for k in base.columns}
            out["entry_ts"] = getattr(r, "entry_ts")
            out["pa_valid"] = 1 if chk.ok else 0
            out["pa_missing_ratio"] = float(chk.missing_ratio)
            out["pa_missing_bars"] = int(chk.missing_bars)
            out["close"] = np.nan

            ratio = float(chk.missing_ratio)

            out["pa_quality"] = 1.0 - ratio

            out["pa_quality_bucket"] = (
                2 if ratio == 0.0 else
                1 if ratio <= 0.1 else
                0
            )

            if chk.ok and win is not None:
                pa = _compute_pa_from_window(win)
                ctx = _market_context_from_window(win)
                out.update(pa)
                out.update(ctx)
                out["close"] = float(win["close"].iloc[-1])
                ok_cnt += 1
            else:
                miss_cnt += 1
                miss_bars_sum += int(chk.missing_bars)

            rows.append(out)

        if len(rows) == 0:
            raise RuntimeError(f"{symbol}: rows is empty after processing")

        out_df = pd.DataFrame(rows).sort_values("entry_ts").reset_index(drop=True)

        # stateful active block по entry_ts-оси текущего symbol dataset
        if "ctx_atr14" not in out_df.columns:
            out_df["ctx_atr14"] = np.nan
        if "close" not in out_df.columns:
            out_df["close"] = np.nan

        active_block = _compute_stateful_active_block(
            df=out_df,
            max_active_bars=active_max_bars,
            stop_atr_mult=active_stop_atr_mult,
        )
        out_df = pd.concat([out_df, active_block.reset_index(drop=True)], axis=1)

        # гарантируем старые 96 колонок
        for c in OLD_96_COLUMNS:
            if c not in out_df.columns:
                if c.startswith("active_") or c.startswith("pa_"):
                    out_df[c] = 0
                else:
                    out_df[c] = np.nan

        # обязательные новые active-колонки long+short
        for c in ACTIVE_V3_EXTRA_COLUMNS:
            if c not in out_df.columns:
                out_df[c] = 0

        # порядок: сначала old 96, потом новые extras
        extra_cols = [c for c in out_df.columns if c not in OLD_96_COLUMNS]
        out_df = out_df[OLD_96_COLUMNS + sorted(extra_cols)].copy()

        # close — внутренний техстолбец, наружу не нужен
        if "close" in out_df.columns:
            out_df = out_df.drop(columns=["close"], errors="ignore")

        out_path = os.path.join(out_by_symbol, f"{symbol}.parquet")
        out_df.to_parquet(out_path, index=False)
        for raw_col, cfg in ACTIVE_PATTERN_POLICY.items():
            policy_rows.append({
                "symbol": symbol,
                "raw_col": raw_col,
                "max_active_bars": int(cfg["max_active_bars"]),
                "stop_atr_mult": float(cfg["stop_atr_mult"]),
                "raw_present": int(raw_col in out_df.columns),
                "raw_sum": float(pd.to_numeric(out_df[raw_col], errors="coerce").fillna(0).sum()) if raw_col in out_df.columns else 0.0,
            })

        n = len(out_df)
        ok_rate = ok_cnt / n if n > 0 else 0.0
        avg_missing_bars = miss_bars_sum / max(1, miss_cnt)

        old_cols_present = int(sum(int(c in out_df.columns) for c in OLD_96_COLUMNS))
        audit_rows.append({
            "symbol": symbol,
            "n_rows": int(n),
            "context_bars": int(context_bars),
            "pa_valid_rate": float(ok_rate),
            "pa_missing_rows": int(miss_cnt),
            "avg_missing_bars_when_missing": float(avg_missing_bars),
            "old_96_cols_present": int(old_cols_present),
            "n_total_cols": int(len(out_df.columns)),
            "n_active_v3_extra_present": int(sum(int(c in out_df.columns) for c in ACTIVE_V3_EXTRA_COLUMNS)),
            "has_active_pa_bos_dn_24": int("active_pa_bos_dn_24" in out_df.columns),
            "out_path": out_path,
        })
        all_rows.append(out_df)

    audit = pd.DataFrame(audit_rows).sort_values(
        ["old_96_cols_present", "pa_valid_rate", "n_rows"],
        ascending=[False, False, False]
    )
    audit_path = os.path.join(out_root, "_AUDIT_GATE3_PA_V3_LONG_SHORT.csv")
    audit.to_csv(audit_path, index=False)

    policy_audit_path = os.path.join(out_root, "_AUDIT_GATE3_PA_V3_ACTIVE_POLICY.csv")
    if len(policy_rows):
        pd.DataFrame(policy_rows).to_csv(policy_audit_path, index=False)
    else:
        pd.DataFrame(
            columns=["symbol", "raw_col", "max_active_bars", "stop_atr_mult", "raw_present", "raw_sum"]).to_csv(
            policy_audit_path, index=False)

    if all_rows:
        merged = pd.concat(all_rows, ignore_index=True)
        merged_path = os.path.join(out_root, "pa_gate3_v3_long_short_all.parquet")
        merged.to_parquet(merged_path, index=False)

    print("WROTE", audit_path)
    print("WROTE", policy_audit_path)
    if all_rows:
        print("WROTE", os.path.join(out_root, "pa_gate3_v3_long_short_all.parquet"))
    print("WROTE", out_by_symbol)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--gate1-root", "--gate1_root", type=str, default="production/dataset/gate1")
    ap.add_argument("--h4-root", "--h4_root", type=str, default="data/h4_3")
    ap.add_argument("--out-root", "--out_root", type=str, default="production/dataset")
    ap.add_argument("--context-bars", "--context_bars", type=int, default=96)

    ap.add_argument("--active-max-bars", "--active_max_bars", type=int, default=6)
    ap.add_argument("--active-stop-atr-mult", "--active_stop_atr_mult", type=float, default=1.25)

    ap.add_argument("--max-symbols", "--max_symbols", type=int, default=0)
    ap.add_argument("--symbols", type=str, default="")

    args = ap.parse_args()

    max_symbols = args.max_symbols if args.max_symbols and args.max_symbols > 0 else None
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    build_gate3_v3_long_short(
        gate1_root=args.gate1_root,
        h4_root=args.h4_root,
        out_root=args.out_root,
        context_bars=args.context_bars,
        active_max_bars=args.active_max_bars,
        active_stop_atr_mult=args.active_stop_atr_mult,
        max_symbols=max_symbols,
        symbols=symbols,
    )


if __name__ == "__main__":
    main()