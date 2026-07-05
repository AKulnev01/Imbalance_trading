from __future__ import annotations
from online.trading import config
import os

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from online.oos_context import append_oos_sql_filters, get_online_oos_context
import argparse
import json
import math
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)
import psycopg2
from psycopg2.extras import execute_values


warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(os.environ.get("IMB_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DB_DSN = config.DB_DSN

ONLINE_GATE2_FEATURES_TABLE = "public.online_gate2_features"
ONLINE_GATE3_FEATURES_TABLE = "public.online_gate3_features"

GATE3_POLICY_CSV = ROOT / "production/models/ks/gate3_symbol_policy.csv.updated"

REPORT_DIR = ROOT / "online" / "_reports_gate3"
REPORT_CSV = REPORT_DIR / "online_gate3_features_report.csv"
REPORT_JSON = REPORT_DIR / "online_gate3_features_report.json"

H4_STEP_SECONDS = 4 * 3600
ENTRY_DELAY_SECONDS = 90

DEFAULT_CONTEXT_BARS = 96
DEFAULT_ACTIVE_MAX_BARS = 6
DEFAULT_ACTIVE_STOP_ATR_MULT = 1.25
DEFAULT_ACTIVE_WARMUP_BARS = int(os.environ.get("IMB_GATE3_ACTIVE_WARMUP_BARS", "48"))
DEFAULT_MAX_WORKERS = int(os.environ.get("IMB_GATE3_BUILD_WORKERS", "8"))

ONLINE_FORBIDDEN_OUTPUT_COLUMNS = {
    "exit_ts",
    "exit_px",
    "exit_reason",
}

POLICY_FEATURE_COLUMNS = [
    "gate3_score_long",
    "gate3_score_short",
    "gate3_score_long_z",
    "gate3_score_short_z",
    "gate3_rank_long",
    "gate3_rank_short",
    "gate3_side_bias",
]

POLICY_NORM_COLUMNS = [
    "gate3_rank_long_norm",
    "gate3_rank_short_norm",
]

TRAIN_CONTEXT_EXTRA_COLUMNS = [
    "ctx_ret1",
    "ctx_ret2",
    "ctx_ret4",
    "ctx_ret8",
    "ctx_atrp14",
    "ctx_range_atr",
]

GATE3_GENERIC_ENGINEERED_COLUMNS = [
    "gate1_proba",

    "gate3_any_active",
    "gate3_active_count",
    "gate3_active_primary",
    "gate3_active_secondary",
    "gate3_active_overlap_primary_secondary",
    "gate3_max_active_age",

    "active_age_norm",
    "active_type_count",
    "active_any",
    "active_is_single",
    "active_is_combo",
    "active_density",

    "active_count_x_quality",
    "active_combo_x_quality",
    "active_count_x_gate1",
    "active_combo_x_gate1",

    "pa_quality_sq",
    "pa_bos_up_12_x_quality",
    "pa_bos_dn_12_x_quality",

    "gate1_proba_lag1",
    "gate1_proba_lag2",
    "gate3_active_count_lag1",
    "gate3_active_count_lag2",
    "ctx_ret1_lag1",
    "ctx_ret1_lag2",
    "ctx_ret2_lag1",
    "ctx_ret2_lag2",
    "ctx_ret4_lag1",
    "ctx_ret4_lag2",
    "ctx_ret8_lag1",
    "ctx_ret8_lag2",
    "ctx_atrp14_lag1",
    "ctx_atrp14_lag2",
    "ctx_range_atr_lag1",
    "ctx_range_atr_lag2",
]

GATE3_SIDE_ENGINEERED_COLUMNS = [
    "g3_long_any_active",
    "g3_long_active_count",
    "g3_long_active_primary",
    "g3_long_active_secondary",
    "g3_long_active_overlap_primary_secondary",
    "g3_long_max_active_age",
    "g3_long_active_age_norm",
    "g3_long_active_type_count",
    "g3_long_active_any",
    "g3_long_active_is_single",
    "g3_long_active_is_combo",
    "g3_long_active_density",
    "g3_long_active_count_x_quality",
    "g3_long_active_combo_x_quality",
    "g3_long_active_count_x_gate1",
    "g3_long_active_combo_x_gate1",

    "g3_short_any_active",
    "g3_short_active_count",
    "g3_short_active_primary",
    "g3_short_active_secondary",
    "g3_short_active_overlap_primary_secondary",
    "g3_short_max_active_age",
    "g3_short_active_age_norm",
    "g3_short_active_type_count",
    "g3_short_active_any",
    "g3_short_active_is_single",
    "g3_short_active_is_combo",
    "g3_short_active_density",
    "g3_short_active_count_x_quality",
    "g3_short_active_combo_x_quality",
    "g3_short_active_count_x_gate1",
    "g3_short_active_combo_x_gate1",
]

GATE3_POLICY_CACHE = None

MISSING_GATE3_KEYS_BATCH: Optional[Dict[str, pd.DataFrame]] = None
GATE2_EXTRA_FEATURES_BATCH: Optional[Dict[str, pd.DataFrame]] = None
H4_DB_BATCH: Optional[Dict[str, pd.DataFrame]] = None


# ============================================================
# FEATURE COLUMNS
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

PA_EXTRA_COLUMNS = [
    "pa_bos_up_48",
    "pa_bos_dn_48",
    "pa_sweep_hi_confirm_dn",
    "pa_sweep_lo_confirm_up",
    "pa_pullback_sell_from_ema",
    "pa_pullback_buy_from_ema",
    "pa_quality",
    "pa_quality_bucket",
    "signal_ts",
    "entry_bar_open_ts",
    "entry_ts_exec",
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
# DB / TIME HELPERS
# ============================================================

def connect_db():
    return psycopg2.connect(DB_DSN)


def utc_now_floor_second() -> pd.Timestamp:
    return pd.Timestamp.utcnow().floor("s")

def latest_closed_h4_open_ts() -> pd.Timestamp:
    """
    Safe wall-clock fallback only.
    Returns timezone-aware UTC timestamp for DB timestamptz comparisons.
    """
    now = pd.Timestamp.now(tz="UTC")
    current_h4_open = now.floor("4h")
    latest_closed = current_h4_open - pd.Timedelta(seconds=H4_STEP_SECONDS)
    return latest_closed


def latest_available_gate3_source_ts() -> pd.Timestamp:
    """
    Gate3 must not decide latest H4 from wall clock only.

    Correct online cap:
    - candle must exist in public.candles_h4
    - upstream Gate2 features must exist in public.online_gate2_features

    Important:
    return timezone-aware UTC timestamp. Do not use tz_convert(None) here,
    because naive datetime passed into PostgreSQL timestamptz can shift the cap
    by the local DB/session timezone and silently select the previous H4.
    """
    query = f"""
        SELECT
            LEAST(
                (SELECT MAX(entry_ts) FROM public.candles_h4),
                (SELECT MAX(entry_ts) FROM {ONLINE_GATE2_FEATURES_TABLE})
            ) AS max_ts
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty or pd.isna(df["max_ts"].iloc[0]):
        return latest_closed_h4_open_ts()

    ts = pd.to_datetime(df["max_ts"].iloc[0], utc=True, errors="coerce")
    if pd.isna(ts):
        return latest_closed_h4_open_ts()

    return ts

def split_table_name(table_name: str) -> Tuple[str, str]:
    if "." not in table_name:
        return "public", table_name
    schema, name = table_name.split(".", 1)
    return schema, name


def table_exists(table_name: str) -> bool:
    schema, name = split_table_name(table_name)
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (schema, name))
            return bool(cur.fetchone()[0])


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def to_naive_utc_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)




# ============================================================
# NUMERIC HELPERS
# ============================================================

def rolling_slope(y: np.ndarray) -> float:
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


def ema(x: np.ndarray, span: int) -> float:
    if len(x) < 2:
        return float("nan")
    alpha = 2.0 / (span + 1.0)
    v = float(x[0])
    for i in range(1, len(x)):
        v = alpha * float(x[i]) + (1.0 - alpha) * v
    return v


def calc_rsi(close: np.ndarray, period: int = 14) -> float:
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


def true_range_np(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return tr.astype(float)


def atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    tr = true_range_np(high, low, close)
    if len(tr) < period:
        return float("nan")
    return float(np.mean(tr[-period:]))


def zscore_last(x: np.ndarray) -> float:
    if len(x) < 5:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if sd == 0:
        return 0.0
    return float((x[-1] - mu) / sd)


def ret_np(c: np.ndarray, n: int) -> float:
    if len(c) < n + 1:
        return float("nan")
    if c[-(n + 1)] == 0:
        return float("nan")
    return float(c[-1] / c[-(n + 1)] - 1.0)


def median_safe(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.median(x.astype(float)))


def roll_prev_max(x: np.ndarray, w: int) -> float:
    if len(x) < w + 1:
        return float("nan")
    return float(np.max(x[-(w + 1):-1]))


def roll_prev_min(x: np.ndarray, w: int) -> float:
    if len(x) < w + 1:
        return float("nan")
    return float(np.min(x[-(w + 1):-1]))


# ============================================================
# CANDLES / PA
# ============================================================

def candle_features(o: float, h: float, l: float, c: float) -> Dict[str, float]:
    rng = float(h - l)
    body = float(abs(c - o))
    up_wick = float(h - max(o, c))
    dn_wick = float(min(o, c) - l)

    body_pct = body / rng if rng > 0 else 0.0
    up_wick_pct = up_wick / rng if rng > 0 else 0.0
    dn_wick_pct = dn_wick / rng if rng > 0 else 0.0

    return {
        "pa_rng": rng,
        "pa_body": body,
        "pa_up_wick": up_wick,
        "pa_dn_wick": dn_wick,
        "pa_body_pct": float(body_pct),
        "pa_up_wick_pct": float(up_wick_pct),
        "pa_dn_wick_pct": float(dn_wick_pct),
        "pa_is_bull": 1.0 if c > o else 0.0,
        "pa_is_bear": 1.0 if c < o else 0.0,
        "pa_is_doji": 1.0 if rng > 0 and body_pct <= 0.1 else 0.0,
        "pa_pin_up": 1.0 if rng > 0 and up_wick_pct >= 0.55 and body_pct <= 0.35 else 0.0,
        "pa_pin_dn": 1.0 if rng > 0 and dn_wick_pct >= 0.55 and body_pct <= 0.35 else 0.0,
    }


def engulfing(prev_o: float, prev_c: float, o: float, c: float) -> Tuple[float, float]:
    prev_hi = max(prev_o, prev_c)
    prev_lo = min(prev_o, prev_c)
    cur_hi = max(o, c)
    cur_lo = min(o, c)

    bull = 1.0 if (c > o) and (prev_c < prev_o) and (cur_hi >= prev_hi) and (cur_lo <= prev_lo) else 0.0
    bear = 1.0 if (c < o) and (prev_c > prev_o) and (cur_hi >= prev_hi) and (cur_lo <= prev_lo) else 0.0

    return bull, bear


def inside_outside(prev_h: float, prev_l: float, h: float, l: float) -> Tuple[float, float]:
    inside = 1.0 if (h <= prev_h) and (l >= prev_l) else 0.0
    outside = 1.0 if (h >= prev_h) and (l <= prev_l) else 0.0
    return inside, outside


def swing_counts(high: np.ndarray, low: np.ndarray, lookback: int = 12) -> Tuple[float, float]:
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


@dataclass
class WindowCheck:
    ok: bool
    missing_ratio: float
    missing_bars: int


def get_window_strict(h4: pd.DataFrame, entry_ts: pd.Timestamp, context_bars: int) -> Tuple[Optional[pd.DataFrame], WindowCheck]:
    if entry_ts not in h4.index:
        return None, WindowCheck(False, 1.0, context_bars)

    pos = h4.index.get_loc(entry_ts)

    if isinstance(pos, slice) or isinstance(pos, np.ndarray):
        try:
            pos = int(np.asarray(pos).ravel()[0])
        except Exception:
            return None, WindowCheck(False, 1.0, context_bars)

    start = int(pos) - (context_bars - 1)

    if start < 0:
        return None, WindowCheck(False, 1.0, context_bars)

    win = h4.iloc[start:int(pos) + 1]

    if len(win) != context_bars:
        miss = context_bars - len(win)
        return None, WindowCheck(False, miss / context_bars, miss)

    diffs = win.index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    bad = int(np.sum(diffs != H4_STEP_SECONDS))

    if bad > 0:
        return None, WindowCheck(False, bad / max(1, len(diffs)), bad)

    return win, WindowCheck(True, 0.0, 0)


def market_context_from_window(win: pd.DataFrame) -> Dict[str, float]:
    h = win["high"].to_numpy(dtype=float)
    l = win["low"].to_numpy(dtype=float)
    c = win["close"].to_numpy(dtype=float)
    v = win["volume"].to_numpy(dtype=float) if "volume" in win.columns else None

    atr14 = atr_np(h, l, c, 14)
    atr48 = atr_np(h, l, c, 48)

    atrp14 = float(atr14 / c[-1]) if (not math.isnan(atr14)) and c[-1] != 0 else float("nan")
    atrp48 = float(atr48 / c[-1]) if (not math.isnan(atr48)) and c[-1] != 0 else float("nan")

    ema_fast = ema(c[-96:] if len(c) >= 96 else c, 12)
    ema_slow = ema(c[-96:] if len(c) >= 96 else c, 26)

    trend_ema = (
        float((ema_fast - ema_slow) / c[-1])
        if (not math.isnan(ema_fast)) and (not math.isnan(ema_slow)) and c[-1] != 0
        else float("nan")
    )

    ret_48 = ret_np(c, 48)

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

    trend_strength_atr = (
        float(trend_ema / atrp48)
        if not math.isnan(trend_ema) and not math.isnan(atrp48) and atrp48 > 0
        else float("nan")
    )

    vol_z20 = float("nan")
    if v is not None and len(v) >= 20:
        vol_z20 = zscore_last(v[-20:])

    hh48 = float(np.max(h[-48:])) if len(h) >= 48 else float(np.max(h))
    ll48 = float(np.min(l[-48:])) if len(l) >= 48 else float(np.min(l))
    range48 = float((hh48 - ll48) / c[-1]) if c[-1] != 0 else float("nan")

    return {
        "ctx_ret_12": ret_np(c, 12),
        "ctx_ret_48": ret_48,
        "ctx_ret_96": ret_np(c, 96),
        "ctx_slope_48": rolling_slope(c[-48:]) if len(c) >= 48 else float("nan"),
        "ctx_slope_96": rolling_slope(c[-96:]) if len(c) >= 96 else float("nan"),
        "ctx_ema_trend": trend_ema,
        "ctx_trend_strength_atr": trend_strength_atr,
        "ctx_rsi14": calc_rsi(c, 14),
        "ctx_atr14": atr14,
        "ctx_atr48": atr48,
        "ctx_atrp14": atrp14,
        "ctx_atrp48": atrp48,
        "ctx_range48": range48,
        "ctx_vol_cluster3": vol_cluster,
        "ctx_regime_48": regime,
        "ctx_vol_z20": vol_z20,
    }


def compute_pa_from_window(win: pd.DataFrame) -> Dict[str, float]:
    o = win["open"].to_numpy(dtype=float)
    h = win["high"].to_numpy(dtype=float)
    l = win["low"].to_numpy(dtype=float)
    c = win["close"].to_numpy(dtype=float)
    v = win["volume"].to_numpy(dtype=float) if "volume" in win.columns else None

    ret1 = ret_np(c, 1)
    ret4 = ret_np(c, 4)
    ret12 = ret_np(c, 12)
    ret48 = ret_np(c, 48)

    atr14 = atr_np(h, l, c, 14)
    atr48 = atr_np(h, l, c, 48)

    atrp14 = float(atr14 / c[-1]) if (not math.isnan(atr14)) and c[-1] != 0 else float("nan")
    atrp48 = float(atr48 / c[-1]) if (not math.isnan(atr48)) and c[-1] != 0 else float("nan")

    slope_12 = rolling_slope(c[-12:]) if len(c) >= 12 else float("nan")
    slope_48 = rolling_slope(c[-48:]) if len(c) >= 48 else float("nan")

    rsi14 = calc_rsi(c, 14)

    cf = candle_features(o[-1], h[-1], l[-1], c[-1])
    prev_cf = candle_features(o[-2], h[-2], l[-2], c[-2]) if len(c) >= 2 else {}

    engulf_bull, engulf_bear = engulfing(o[-2], c[-2], o[-1], c[-1]) if len(c) >= 2 else (float("nan"), float("nan"))
    inside, outside = inside_outside(h[-2], l[-2], h[-1], l[-1]) if len(c) >= 2 else (float("nan"), float("nan"))

    hh48 = float(np.max(h[-48:])) if len(h) >= 48 else float(np.max(h))
    ll48 = float(np.min(l[-48:])) if len(l) >= 48 else float(np.min(l))

    dist_to_hh = float((hh48 - c[-1]) / c[-1]) if c[-1] != 0 else float("nan")
    dist_to_ll = float((c[-1] - ll48) / c[-1]) if c[-1] != 0 else float("nan")

    swing_hi, swing_lo = swing_counts(h, l, 12)

    vol_z = float("nan")
    if v is not None and len(v) >= 20:
        vol_z = zscore_last(v[-20:])

    prev_hi_12 = roll_prev_max(h, 12)
    prev_lo_12 = roll_prev_min(l, 12)
    prev_hi_24 = roll_prev_max(h, 24)
    prev_lo_24 = roll_prev_min(l, 24)
    prev_hi_48 = roll_prev_max(h, 48)
    prev_lo_48 = roll_prev_min(l, 48)

    prev2_hi_12 = roll_prev_max(h[:-1], 12) if len(h) >= 14 else float("nan")
    prev2_lo_12 = roll_prev_min(l[:-1], 12) if len(l) >= 14 else float("nan")

    slope_12_prev1 = rolling_slope(c[-13:-1]) if len(c) >= 13 else float("nan")
    ret12_prev1 = float(c[-2] / c[-14] - 1.0) if len(c) >= 14 and c[-14] != 0 else float("nan")

    prev_range_48 = (
        float(prev_hi_48 - prev_lo_48)
        if np.isfinite(prev_hi_48) and np.isfinite(prev_lo_48)
        else float("nan")
    )

    prev_range_48_pct = (
        float(prev_range_48 / c[-1])
        if np.isfinite(prev_range_48) and c[-1] != 0
        else float("nan")
    )

    breakout_margin_pct = float(max(0.15 * atrp14, 0.0025)) if np.isfinite(atrp14) else float("nan")

    body_pct = cf["pa_body_pct"]
    up_wick_pct = cf["pa_up_wick_pct"]
    dn_wick_pct = cf["pa_dn_wick_pct"]

    vol_med20 = median_safe(v[-20:]) if v is not None and len(v) >= 20 else float("nan")
    vol_boost = float(v[-1] > vol_med20) if v is not None and np.isfinite(vol_med20) else 0.0

    rng = float(h[-1] - l[-1])
    tr_arr = true_range_np(h, l, c)
    tr_med20 = median_safe(tr_arr[-20:]) if len(tr_arr) >= 20 else float("nan")

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

    sweep_hi_reject_dn_12 = 1.0 if (
        np.isfinite(prev_hi_12)
        and h[-1] > prev_hi_12
        and c[-1] < prev_hi_12
        and up_wick_pct >= 0.35
    ) else 0.0

    sweep_lo_reject_up_12 = 1.0 if (
        np.isfinite(prev_lo_12)
        and l[-1] < prev_lo_12
        and c[-1] > prev_lo_12
        and dn_wick_pct >= 0.35
    ) else 0.0

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

    ema_fast = ema(c[-48:] if len(c) >= 48 else c, 12)
    ema_slow = ema(c[-48:] if len(c) >= 48 else c, 26)

    trend_pullback_up = 1.0 if (
        np.isfinite(ema_fast)
        and np.isfinite(ema_slow)
        and ema_fast > ema_slow
        and c[-1] > ema_slow
        and l[-1] <= ema_fast
    ) else 0.0

    trend_pullback_dn = 1.0 if (
        np.isfinite(ema_fast)
        and np.isfinite(ema_slow)
        and ema_fast < ema_slow
        and c[-1] < ema_slow
        and h[-1] >= ema_fast
    ) else 0.0

    atr_squeeze_break_up = 1.0 if is_squeeze == 1.0 and np.isfinite(prev_hi_12) and c[-1] > prev_hi_12 else 0.0
    atr_squeeze_break_dn = 1.0 if is_squeeze == 1.0 and np.isfinite(prev_lo_12) and c[-1] < prev_lo_12 else 0.0

    bos_up_12_close_top20 = 1.0 if (
        bos_up_12 == 1.0 and body_pct >= 0.35 and c[-1] >= (h[-1] - 0.2 * max(rng, 1e-12))
    ) else 0.0

    bos_dn_12_close_bot20 = 1.0 if (
        bos_dn_12 == 1.0 and body_pct >= 0.35 and c[-1] <= (l[-1] + 0.2 * max(rng, 1e-12))
    ) else 0.0

    bos_up_12_volz20_high = 1.0 if bos_up_12 == 1.0 and vol_boost == 1.0 else 0.0
    bos_dn_12_volz20_high = 1.0 if bos_dn_12 == 1.0 and vol_boost == 1.0 else 0.0

    range_break_up_48_ret4pos = 1.0 if range_break_up_48 == 1.0 and np.isfinite(ret4) and ret4 > 0 else 0.0
    range_break_dn_48_ret4neg = 1.0 if range_break_dn_48 == 1.0 and np.isfinite(ret4) and ret4 < 0 else 0.0

    sweep_hi_dn_12_confirm1 = 1.0 if sweep_hi_reject_dn_12 == 1.0 and c[-1] < o[-1] else 0.0
    sweep_lo_up_12_confirm1 = 1.0 if sweep_lo_reject_up_12 == 1.0 and c[-1] > o[-1] else 0.0

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

    squeeze_and_rngatr_comp = 1.0 if (
        is_squeeze == 1.0 and np.isfinite(tr_med20) and rng < tr_med20 * 0.8
    ) else 0.0

    squeeze_break_up_trend = 1.0 if (
        atr_squeeze_break_up == 1.0 and np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast > ema_slow
    ) else 0.0

    squeeze_break_dn_trend = 1.0 if (
        atr_squeeze_break_dn == 1.0 and np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast < ema_slow
    ) else 0.0

    choch_up_48_12_ema_cross = 1.0 if (
        choch_up_48_12 == 1.0 and np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast > ema_slow
    ) else 0.0

    choch_dn_48_12_ema_cross = 1.0 if (
        choch_dn_48_12 == 1.0 and np.isfinite(ema_fast) and np.isfinite(ema_slow) and ema_fast < ema_slow
    ) else 0.0

    choch_up_48_12_ret12pos = 1.0 if choch_up_48_12 == 1.0 and np.isfinite(ret12) and ret12 > 0 else 0.0
    choch_dn_48_12_ret12neg = 1.0 if choch_dn_48_12 == 1.0 and np.isfinite(ret12) and ret12 < 0 else 0.0

    bos_up_48 = 1.0 if np.isfinite(prev_hi_48) and c[-1] > prev_hi_48 else 0.0
    bos_dn_48 = 1.0 if np.isfinite(prev_lo_48) and c[-1] < prev_lo_48 else 0.0

    sweep_hi_confirm_dn = 1.0 if sweep_hi_reject_dn_12 == 1.0 and body_pct >= 0.25 and cf["pa_is_bear"] == 1.0 else 0.0
    sweep_lo_confirm_up = 1.0 if sweep_lo_reject_up_12 == 1.0 and body_pct >= 0.25 and cf["pa_is_bull"] == 1.0 else 0.0

    pullback_sell_from_ema = 1.0 if trend_pullback_dn == 1.0 and up_wick_pct >= 0.25 else 0.0
    pullback_buy_from_ema = 1.0 if trend_pullback_up == 1.0 and dn_wick_pct >= 0.25 else 0.0

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
# ACTIVE ENGINE
# ============================================================

def safe_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def event_from_df(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=int)
    return safe_int(df[col]).to_numpy(dtype=int)


def pattern_policy(raw_col: str, default_max_active_bars: int, default_stop_atr_mult: float) -> Tuple[int, float]:
    cfg = ACTIVE_PATTERN_POLICY.get(raw_col)
    if cfg is None:
        return int(default_max_active_bars), float(default_stop_atr_mult)
    return int(cfg["max_active_bars"]), float(cfg["stop_atr_mult"])


def build_stateful_active_series(
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



def add_policy_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    q_raw = (
        pd.to_numeric(out["pa_quality"], errors="coerce")
        if "pa_quality" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    q_raw = q_raw.fillna(0.0)

    q_mean = q_raw.shift(1).rolling(200).mean()
    q_std = q_raw.shift(1).rolling(200).std()
    q = (q_raw - q_mean) / (q_std + 1e-6)

    bos_up = (
        pd.to_numeric(out["pa_bos_up_12"], errors="coerce")
        if "pa_bos_up_12" in out.columns
        else pd.Series(0.0, index=out.index)
    ).fillna(0.0)

    bos_dn = (
        pd.to_numeric(out["pa_bos_dn_12"], errors="coerce")
        if "pa_bos_dn_12" in out.columns
        else pd.Series(0.0, index=out.index)
    ).fillna(0.0)

    out["pa_quality_sq"] = q ** 2
    out["pa_bos_up_12_x_quality"] = bos_up * q
    out["pa_bos_dn_12_x_quality"] = bos_dn * q

    return out


def compute_stateful_active_block(
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
        event = event_from_df(df, raw_col)
        pat_max_active_bars, pat_stop_atr_mult = pattern_policy(
            raw_col=raw_col,
            default_max_active_bars=max_active_bars,
            default_stop_atr_mult=stop_atr_mult,
        )

        out[active_col] = build_stateful_active_series(
            event=event,
            close=close,
            atr14_series=atr14,
            side=side,
            max_active_bars=pat_max_active_bars,
            stop_atr_mult=pat_stop_atr_mult,
        )

    return pd.DataFrame(out, index=df.index)


def active_feature_columns() -> List[str]:
    cols = []
    for c in OLD_96_COLUMNS + ACTIVE_V3_EXTRA_COLUMNS:
        if c.startswith("active_pa_") and c not in cols:
            cols.append(c)
    return sorted(cols)


def active_persistence_columns() -> List[str]:
    out = []
    for c in active_feature_columns():
        out.extend([
            f"{c}__age",
            f"{c}__fresh",
            f"{c}__mid",
            f"{c}__late",
        ])
    return out


def all_gate3_extra_engineered_columns() -> List[str]:
    cols = []
    for c in (
        POLICY_FEATURE_COLUMNS
        + POLICY_NORM_COLUMNS
        + TRAIN_CONTEXT_EXTRA_COLUMNS
        + GATE3_GENERIC_ENGINEERED_COLUMNS
        + GATE3_SIDE_ENGINEERED_COLUMNS
        + active_persistence_columns()
    ):
        if c not in cols:
            cols.append(c)
    return cols


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).gt(0.5).astype(int)


def select_side_pattern_cols(cols: List[str], side: str) -> List[str]:
    side = str(side).strip().lower()
    out = []

    for c in cols:
        c_low = str(c).lower()

        is_long = ("_up" in c_low) or ("buy" in c_low) or ("_lo_" in c_low)
        is_short = ("_dn" in c_low) or ("sell" in c_low) or ("_hi_" in c_low)

        if side == "long" and is_long and not is_short:
            out.append(c)

        if side == "short" and is_short and not is_long:
            out.append(c)

    return sorted(set(out))


def load_gate3_policy() -> pd.DataFrame:
    global GATE3_POLICY_CACHE

    if GATE3_POLICY_CACHE is not None:
        return GATE3_POLICY_CACHE

    if not GATE3_POLICY_CSV.exists():
        raise RuntimeError(f"not found: {GATE3_POLICY_CSV}")

    p = pd.read_csv(GATE3_POLICY_CSV)
    if "symbol" not in p.columns:
        raise RuntimeError(f"policy missing symbol column: {GATE3_POLICY_CSV}")

    p["symbol"] = p["symbol"].astype(str).str.upper()

    for c in POLICY_FEATURE_COLUMNS:
        if c not in p.columns:
            p[c] = 0.0
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)

    GATE3_POLICY_CACHE = p.drop_duplicates("symbol", keep="last").reset_index(drop=True)
    return GATE3_POLICY_CACHE


def policy_row_for_symbol(symbol: str) -> Dict[str, float]:
    p = load_gate3_policy()
    sym = str(symbol).upper()
    row = p[p["symbol"] == sym]

    out = {}
    for c in POLICY_FEATURE_COLUMNS:
        out[c] = 0.0

    if row.empty:
        return out

    r = row.iloc[-1]
    for c in POLICY_FEATURE_COLUMNS:
        out[c] = float(pd.to_numeric(r.get(c, 0.0), errors="coerce") or 0.0)

    return out


def load_gate2_extra_features(symbol: str, rebuild: bool, limit_latest: Optional[int]) -> pd.DataFrame:
    global GATE2_EXTRA_FEATURES_BATCH

    symbol = str(symbol).upper()

    if GATE2_EXTRA_FEATURES_BATCH is not None:
        df = GATE2_EXTRA_FEATURES_BATCH.get(
            symbol,
            pd.DataFrame(columns=["symbol", "entry_ts", "gate1_proba"]),
        ).copy()

        if limit_latest is not None and int(limit_latest) > 0 and len(df) > int(limit_latest):
            df = df.tail(int(limit_latest)).reset_index(drop=True)

        return df

    existing = set(get_table_columns(ONLINE_GATE2_FEATURES_TABLE))

    wanted = [
        "symbol",
        "entry_ts",
        "gate1_proba",
    ]

    select_cols = []
    for c in wanted:
        if c in existing:
            select_cols.append(f"f.{quote_ident(c)} AS {quote_ident(c)}")
        else:
            select_cols.append(f"NULL AS {quote_ident(c)}")

    if rebuild or not table_exists(ONLINE_GATE3_FEATURES_TABLE):
        where_missing = ""
    else:
        where_missing = f"""
            AND NOT EXISTS (
                SELECT 1
                FROM {ONLINE_GATE3_FEATURES_TABLE} g3
                WHERE g3.symbol = f.symbol
                  AND g3.entry_ts = f.entry_ts
            )
        """

    limit_clause = ""
    if limit_latest is not None and int(limit_latest) > 0:
        limit_clause = f"LIMIT {int(limit_latest)}"

    query = f"""
        SELECT
            {", ".join(select_cols)}
        FROM {ONLINE_GATE2_FEATURES_TABLE} f
        WHERE f.symbol = %s
        {where_missing}
        ORDER BY f.entry_ts DESC
        {limit_clause}
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=(symbol,))

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = to_naive_utc_series(df["entry_ts"])
    df["gate1_proba"] = pd.to_numeric(df["gate1_proba"], errors="coerce")

    df = (
        df.dropna(subset=["entry_ts"])
        .sort_values("entry_ts")
        .drop_duplicates(["symbol", "entry_ts"], keep="last")
        .reset_index(drop=True)
    )

    return df

def add_train_context_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = pd.to_numeric(out["close"], errors="coerce") if "close" in out.columns else pd.Series(np.nan, index=out.index)
    high = pd.to_numeric(out["high"], errors="coerce") if "high" in out.columns else pd.Series(np.nan, index=out.index)
    low = pd.to_numeric(out["low"], errors="coerce") if "low" in out.columns else pd.Series(np.nan, index=out.index)

    out["ctx_ret1"] = close.pct_change(1).replace([np.inf, -np.inf], np.nan)
    out["ctx_ret2"] = close.pct_change(2).replace([np.inf, -np.inf], np.nan)
    out["ctx_ret4"] = close.pct_change(4).replace([np.inf, -np.inf], np.nan)
    out["ctx_ret8"] = close.pct_change(8).replace([np.inf, -np.inf], np.nan)

    atr14 = pd.to_numeric(out["ctx_atr14"], errors="coerce") if "ctx_atr14" in out.columns else pd.Series(np.nan, index=out.index)
    out["ctx_atrp14"] = atr14 / close.replace(0.0, np.nan)
    out["ctx_range_atr"] = (high - low) / atr14.replace(0.0, np.nan)

    return out


def add_policy_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    prow = policy_row_for_symbol(symbol)

    for c in POLICY_FEATURE_COLUMNS:
        out[c] = float(prow.get(c, 0.0))

    for c in POLICY_FEATURE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).shift(1)

    for c in POLICY_FEATURE_COLUMNS:
        out[c] = np.clip(pd.to_numeric(out[c], errors="coerce").fillna(0.0), -5.0, 5.0)

    long_abs_max = float(pd.to_numeric(out["gate3_rank_long"], errors="coerce").abs().max() or 0.0)
    short_abs_max = float(pd.to_numeric(out["gate3_rank_short"], errors="coerce").abs().max() or 0.0)

    out["gate3_rank_long_norm"] = pd.to_numeric(out["gate3_rank_long"], errors="coerce").fillna(0.0) / (long_abs_max + 1e-6)
    out["gate3_rank_short_norm"] = pd.to_numeric(out["gate3_rank_short"], errors="coerce").fillna(0.0) / (short_abs_max + 1e-6)

    return out


def add_active_persistence_features(df: pd.DataFrame, active_cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for c in active_cols:
        if c not in out.columns:
            out[c] = 0

        x = safe_bool_series(out[c]).to_numpy(dtype=int)
        age = np.zeros(len(x), dtype=int)
        run = 0

        for i in range(len(x)):
            if x[i] == 1:
                run += 1
            else:
                run = 0
            age[i] = run

        out[f"{c}__age"] = age
        out[f"{c}__fresh"] = (age == 1).astype(int)
        out[f"{c}__mid"] = ((age >= 2) & (age <= 3)).astype(int)
        out[f"{c}__late"] = (age >= 4).astype(int)

    return out


def add_side_active_set_features(
    df: pd.DataFrame,
    active_cols: List[str],
    side: str,
    prefix: str,
) -> pd.DataFrame:
    out = df.copy()

    if side == "long":
        primary_col = "active_pa_atr_squeeze_break_up"
        secondary_col = "active_pa_bos_up_24"
    else:
        primary_col = "active_pa_atr_squeeze_break_dn"
        secondary_col = "active_pa_bos_dn_24"

    if not active_cols:
        out[f"{prefix}_any_active"] = 0
        out[f"{prefix}_active_count"] = 0
        out[f"{prefix}_active_primary"] = 0
        out[f"{prefix}_active_secondary"] = 0
        out[f"{prefix}_active_overlap_primary_secondary"] = 0
        out[f"{prefix}_max_active_age"] = 0
        out[f"{prefix}_active_age_norm"] = 0.0
        out[f"{prefix}_active_type_count"] = 0
        out[f"{prefix}_active_any"] = 0
        out[f"{prefix}_active_is_single"] = 0
        out[f"{prefix}_active_is_combo"] = 0
        out[f"{prefix}_active_density"] = 0.0
        out[f"{prefix}_active_count_x_quality"] = 0.0
        out[f"{prefix}_active_combo_x_quality"] = 0.0
        out[f"{prefix}_active_count_x_gate1"] = 0.0
        out[f"{prefix}_active_combo_x_gate1"] = 0.0
        return out

    act = pd.DataFrame(index=out.index)
    for c in active_cols:
        if c not in out.columns:
            out[c] = 0
        act[c] = safe_bool_series(out[c])

    act_sum = act.sum(axis=1).astype(int)

    out[f"{prefix}_any_active"] = (act_sum > 0).astype(int)
    out[f"{prefix}_active_count"] = act_sum
    out[f"{prefix}_active_primary"] = act[primary_col].astype(int) if primary_col in act.columns else 0
    out[f"{prefix}_active_secondary"] = act[secondary_col].astype(int) if secondary_col in act.columns else 0
    out[f"{prefix}_active_overlap_primary_secondary"] = (
        (pd.to_numeric(out[f"{prefix}_active_primary"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(out[f"{prefix}_active_secondary"], errors="coerce").fillna(0) > 0)
    ).astype(int)

    max_age = np.zeros(len(out), dtype=int)
    for c in active_cols:
        age_col = f"{c}__age"
        if age_col in out.columns:
            max_age = np.maximum(max_age, pd.to_numeric(out[age_col], errors="coerce").fillna(0).to_numpy(dtype=int))

    out[f"{prefix}_max_active_age"] = max_age
    out[f"{prefix}_active_age_norm"] = out[f"{prefix}_max_active_age"] / 10.0

    out[f"{prefix}_active_type_count"] = act_sum
    out[f"{prefix}_active_any"] = (act_sum > 0).astype(int)
    out[f"{prefix}_active_is_single"] = (act_sum == 1).astype(int)
    out[f"{prefix}_active_is_combo"] = (act_sum >= 2).astype(int)
    out[f"{prefix}_active_density"] = act_sum / max(1, len(active_cols))

    q = pd.to_numeric(out["pa_quality"], errors="coerce").fillna(0.0) if "pa_quality" in out.columns else pd.Series(0.0, index=out.index)
    g1 = pd.to_numeric(out["gate1_proba"], errors="coerce").fillna(0.0) if "gate1_proba" in out.columns else pd.Series(0.0, index=out.index)

    out[f"{prefix}_active_count_x_quality"] = out[f"{prefix}_active_type_count"] * q
    out[f"{prefix}_active_combo_x_quality"] = out[f"{prefix}_active_is_combo"] * q
    out[f"{prefix}_active_count_x_gate1"] = out[f"{prefix}_active_type_count"] * g1
    out[f"{prefix}_active_combo_x_gate1"] = out[f"{prefix}_active_is_combo"] * g1

    return out


def add_generic_all_active_features(df: pd.DataFrame, active_cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    act = pd.DataFrame(index=out.index)
    for c in active_cols:
        if c not in out.columns:
            out[c] = 0
        act[c] = safe_bool_series(out[c])

    act_sum = act.sum(axis=1).astype(int) if len(act.columns) else pd.Series(0, index=out.index)

    out["gate3_any_active"] = (act_sum > 0).astype(int)
    out["gate3_active_count"] = act_sum

    out["gate3_active_primary"] = (
        safe_bool_series(out["active_pa_atr_squeeze_break_up"]) |
        safe_bool_series(out["active_pa_atr_squeeze_break_dn"])
    ).astype(int)

    out["gate3_active_secondary"] = (
        safe_bool_series(out["active_pa_bos_up_24"]) |
        safe_bool_series(out["active_pa_bos_dn_24"])
    ).astype(int)

    out["gate3_active_overlap_primary_secondary"] = (
        (out["gate3_active_primary"] == 1) &
        (out["gate3_active_secondary"] == 1)
    ).astype(int)

    age_cols = [f"{c}__age" for c in active_cols if f"{c}__age" in out.columns]
    if age_cols:
        out["gate3_max_active_age"] = out[age_cols].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1)
    else:
        out["gate3_max_active_age"] = 0

    out["active_age_norm"] = pd.to_numeric(out["gate3_max_active_age"], errors="coerce").fillna(0.0) / 10.0
    out["active_type_count"] = act_sum
    out["active_any"] = (act_sum > 0).astype(int)
    out["active_is_single"] = (act_sum == 1).astype(int)
    out["active_is_combo"] = (act_sum >= 2).astype(int)
    out["active_density"] = act_sum / max(1, len(active_cols))

    q = pd.to_numeric(out["pa_quality"], errors="coerce").fillna(0.0) if "pa_quality" in out.columns else pd.Series(0.0, index=out.index)
    g1 = pd.to_numeric(out["gate1_proba"], errors="coerce").fillna(0.0) if "gate1_proba" in out.columns else pd.Series(0.0, index=out.index)

    out["active_count_x_quality"] = out["active_type_count"] * q
    out["active_combo_x_quality"] = out["active_is_combo"] * q
    out["active_count_x_gate1"] = out["active_type_count"] * g1
    out["active_combo_x_gate1"] = out["active_is_combo"] * g1

    return out


def add_quality_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    q_raw = pd.to_numeric(out["pa_quality"], errors="coerce") if "pa_quality" in out.columns else pd.Series(0.0, index=out.index)
    q_raw = q_raw.fillna(0.0)

    q_mean = q_raw.shift(1).rolling(200).mean()
    q_std = q_raw.shift(1).rolling(200).std()
    q = (q_raw - q_mean) / (q_std + 1e-6)

    bos_up = pd.to_numeric(out["pa_bos_up_12"], errors="coerce").fillna(0.0) if "pa_bos_up_12" in out.columns else pd.Series(0.0, index=out.index)
    bos_dn = pd.to_numeric(out["pa_bos_dn_12"], errors="coerce").fillna(0.0) if "pa_bos_dn_12" in out.columns else pd.Series(0.0, index=out.index)

    out["pa_quality_sq"] = q ** 2
    out["pa_bos_up_12_x_quality"] = bos_up * q
    out["pa_bos_dn_12_x_quality"] = bos_dn * q

    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cols = [
        "gate1_proba",
        "gate3_active_count",
        "ctx_ret1",
        "ctx_ret2",
        "ctx_ret4",
        "ctx_ret8",
        "ctx_atrp14",
        "ctx_range_atr",
    ]

    for c in cols:
        if c not in out.columns:
            out[c] = np.nan

        out[f"{c}_lag1"] = pd.to_numeric(out[c], errors="coerce").shift(1)
        out[f"{c}_lag2"] = pd.to_numeric(out[c], errors="coerce").shift(2)

    return out


def add_online_gate3_train_compatible_features(
    df: pd.DataFrame,
    symbol: str,
    gate2_extra: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    out["entry_ts"] = to_naive_utc_series(out["entry_ts"])
    gate2_extra = gate2_extra.copy()
    if not gate2_extra.empty:
        gate2_extra["entry_ts"] = to_naive_utc_series(gate2_extra["entry_ts"])
        gate2_extra = gate2_extra[["entry_ts", "gate1_proba"]].drop_duplicates("entry_ts", keep="last")
        out = out.merge(gate2_extra, on="entry_ts", how="left", suffixes=("", "_g2"))
        if "gate1_proba_g2" in out.columns:
            out["gate1_proba"] = out["gate1_proba"].combine_first(out["gate1_proba_g2"]) if "gate1_proba" in out.columns else out["gate1_proba_g2"]
            out = out.drop(columns=["gate1_proba_g2"], errors="ignore")

    if "gate1_proba" not in out.columns:
        out["gate1_proba"] = np.nan

    out["gate1_proba"] = pd.to_numeric(out["gate1_proba"], errors="coerce")

    out = add_train_context_extra_features(out)
    out = add_policy_features(out, symbol=symbol)

    active_cols = active_feature_columns()
    long_active_cols = select_side_pattern_cols(active_cols, "long")
    short_active_cols = select_side_pattern_cols(active_cols, "short")

    out = add_active_persistence_features(out, active_cols=active_cols)

    out = add_side_active_set_features(out, active_cols=long_active_cols, side="long", prefix="g3_long")
    out = add_side_active_set_features(out, active_cols=short_active_cols, side="short", prefix="g3_short")
    out = add_generic_all_active_features(out, active_cols=active_cols)

    out = add_lag_features(out)

    return out


# ============================================================
# LOADERS
# ============================================================



def load_h4_db(symbol: str) -> pd.DataFrame:
    global H4_DB_BATCH

    symbol = str(symbol).upper()

    if H4_DB_BATCH is not None:
        batch_df = H4_DB_BATCH.get(
            symbol,
            pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"]),
        ).copy()

        if not batch_df.empty:
            return batch_df

    query = """
        SELECT symbol, entry_ts AS ts, open, high, low, close, volume
        FROM public.candles_h4
        WHERE symbol = %s
        ORDER BY entry_ts
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=(symbol,))

    if df.empty:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["ts"] = to_naive_utc_series(df["ts"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    return df[["symbol", "ts", "open", "high", "low", "close", "volume"]].copy()


def load_h4_context(symbol: str) -> pd.DataFrame:
    df = load_h4_db(symbol)

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["ts"] = to_naive_utc_series(df["ts"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ts", "open", "high", "low", "close"])
    df = df.sort_values("ts").drop_duplicates(["symbol", "ts"], keep="last").reset_index(drop=True)

    df = df.set_index("ts", drop=True)

    return df[["open", "high", "low", "close", "volume"]].copy()


def get_symbols_from_gate2_features() -> List[str]:
    query = f"""
        SELECT DISTINCT symbol
        FROM {ONLINE_GATE2_FEATURES_TABLE}
        ORDER BY symbol
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn)

    return [str(x).upper() for x in df["symbol"].tolist()]


def build_symbol_time_bounds(
    symbols: List[str],
    missing_by_symbol: Optional[Dict[str, pd.DataFrame]],
    context_bars: int,
) -> List[Tuple[str, object, object]]:
    symbols_clean = sorted(set(str(s).upper() for s in symbols))
    if not symbols_clean:
        return []

    if missing_by_symbol is None:
        return []

    extra_bars = int(DEFAULT_ACTIVE_WARMUP_BARS) + int(context_bars) + 5
    bounds: List[Tuple[str, object, object]] = []

    for symbol in symbols_clean:
        missing = missing_by_symbol.get(symbol)
        if missing is None or missing.empty or "entry_ts" not in missing.columns:
            continue

        ts = pd.to_datetime(missing["entry_ts"], utc=True, errors="coerce").dropna()
        if ts.empty:
            continue

        min_ts = pd.Timestamp(ts.min()) - pd.Timedelta(seconds=H4_STEP_SECONDS * extra_bars)
        max_ts = pd.Timestamp(ts.max())

        bounds.append((
            symbol,
            min_ts.to_pydatetime(),
            max_ts.to_pydatetime(),
        ))

    return bounds



def load_missing_gate3_keys_batch(
    symbols: List[str],
    rebuild: bool,
    limit_latest: Optional[int],
) -> Dict[str, pd.DataFrame]:
    symbols_clean = sorted(set(str(s).upper() for s in symbols))
    if not symbols_clean:
        return {}

    max_allowed_entry_ts = latest_available_gate3_source_ts()

    if rebuild or not table_exists(ONLINE_GATE3_FEATURES_TABLE):
        where_missing = ""
    else:
        where_missing = f"""
            AND NOT EXISTS (
                SELECT 1
                FROM {ONLINE_GATE3_FEATURES_TABLE} g3
                WHERE g3.symbol = f.symbol
                  AND g3.entry_ts = f.entry_ts
            )
        """

    where_parts = [
        "f.symbol = ANY(%s)",
        "f.entry_ts <= %s",
    ]
    params: List[object] = [
        symbols_clean,
        max_allowed_entry_ts.to_pydatetime(),
    ]

    append_oos_sql_filters(
        where_parts=where_parts,
        params=params,
        table_alias="f",
        ts_column="entry_ts",
        symbol_column="symbol",
    )

    where_sql = " AND ".join(where_parts)

    limit_filter = ""
    if limit_latest is not None and int(limit_latest) > 0:
        limit_filter = "WHERE rn <= {}".format(int(limit_latest))

    query = f"""
        WITH src AS (
            SELECT
                f.symbol,
                f.entry_ts,
                f.signal_ts,
                f.entry_bar_open_ts,
                f.entry_ts_exec,
                f.entry_px_exec,
                ROW_NUMBER() OVER (
                    PARTITION BY f.symbol
                    ORDER BY f.entry_ts DESC
                ) AS rn
            FROM {ONLINE_GATE2_FEATURES_TABLE} f
            WHERE {where_sql}
              AND EXISTS (
                  SELECT 1
                  FROM public.candles_h4 c
                  WHERE c.symbol = f.symbol
                    AND c.entry_ts = f.entry_ts
              )
            {where_missing}
        )
        SELECT
            symbol,
            entry_ts,
            signal_ts,
            entry_bar_open_ts,
            entry_ts_exec,
            entry_px_exec
        FROM src
        {limit_filter}
        ORDER BY symbol ASC, entry_ts ASC
    """

    with connect_db() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=params,
        )

    result: Dict[str, pd.DataFrame] = {
        s: pd.DataFrame(
            columns=[
                "symbol",
                "entry_ts",
                "signal_ts",
                "entry_bar_open_ts",
                "entry_ts_exec",
                "entry_px_exec",
            ]
        )
        for s in symbols_clean
    }

    if df.empty:
        return result

    df["symbol"] = df["symbol"].astype(str).str.upper()

    for c in ["entry_ts", "signal_ts", "entry_bar_open_ts", "entry_ts_exec"]:
        if c in df.columns:
            df[c] = to_naive_utc_series(df[c])

    df = (
        df.dropna(subset=["entry_ts"])
        .sort_values(["symbol", "entry_ts"])
        .drop_duplicates(["symbol", "entry_ts"], keep="last")
        .reset_index(drop=True)
    )

    for symbol, g in df.groupby("symbol", sort=False):
        result[str(symbol).upper()] = g.reset_index(drop=True)

    return result
def load_gate2_extra_features_batch(
    symbols: List[str],
    missing_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    context_bars: int = DEFAULT_CONTEXT_BARS,
) -> Dict[str, pd.DataFrame]:
    symbols_clean = sorted(set(str(s).upper() for s in symbols))
    if not symbols_clean:
        return {}

    existing = set(get_table_columns(ONLINE_GATE2_FEATURES_TABLE))

    wanted = [
        "symbol",
        "entry_ts",
        "gate1_proba",
    ]

    select_cols = []
    for c in wanted:
        if c in existing:
            select_cols.append(f"f.{quote_ident(c)} AS {quote_ident(c)}")
        else:
            select_cols.append(f"NULL AS {quote_ident(c)}")

    result: Dict[str, pd.DataFrame] = {
        s: pd.DataFrame(columns=["symbol", "entry_ts", "gate1_proba"])
        for s in symbols_clean
    }

    bounds = build_symbol_time_bounds(
        symbols=symbols_clean,
        missing_by_symbol=missing_by_symbol,
        context_bars=context_bars,
    )
    if not bounds:
        return result

    query = f"""
        SELECT
            {", ".join(select_cols)}
        FROM {ONLINE_GATE2_FEATURES_TABLE} f
        INNER JOIN (VALUES %s) AS b(symbol, min_ts, max_ts)
            ON f.symbol = b.symbol
           AND f.entry_ts >= b.min_ts::timestamptz
           AND f.entry_ts <= b.max_ts::timestamptz
        ORDER BY f.symbol ASC, f.entry_ts ASC
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            rows = execute_values(cur, query, bounds, fetch=True)
            cols = [desc[0] for desc in cur.description] if cur.description else []

    if not rows:
        return result

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return result

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = to_naive_utc_series(df["entry_ts"])
    df["gate1_proba"] = pd.to_numeric(df["gate1_proba"], errors="coerce")

    df = (
        df.dropna(subset=["entry_ts"])
        .sort_values(["symbol", "entry_ts"])
        .drop_duplicates(["symbol", "entry_ts"], keep="last")
        .reset_index(drop=True)
    )

    for symbol, g in df.groupby("symbol", sort=False):
        result[str(symbol).upper()] = g.reset_index(drop=True)

    return result


def load_h4_db_batch(
    symbols: List[str],
    missing_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    context_bars: int = DEFAULT_CONTEXT_BARS,
) -> Dict[str, pd.DataFrame]:
    symbols_clean = sorted(set(str(s).upper() for s in symbols))
    if not symbols_clean:
        return {}

    result: Dict[str, pd.DataFrame] = {
        s: pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        for s in symbols_clean
    }

    bounds = build_symbol_time_bounds(
        symbols=symbols_clean,
        missing_by_symbol=missing_by_symbol,
        context_bars=context_bars,
    )
    if not bounds:
        return result

    query = """
        SELECT
            c.symbol,
            c.entry_ts AS ts,
            c.open,
            c.high,
            c.low,
            c.close,
            c.volume
        FROM public.candles_h4 c
        INNER JOIN (VALUES %s) AS b(symbol, min_ts, max_ts)
            ON c.symbol = b.symbol
           AND c.entry_ts >= b.min_ts::timestamptz
           AND c.entry_ts <= b.max_ts::timestamptz
        ORDER BY c.symbol ASC, c.entry_ts ASC
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            rows = execute_values(cur, query, bounds, fetch=True)
            cols = [desc[0] for desc in cur.description] if cur.description else []

    if not rows:
        return result

    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        return result

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["ts"] = to_naive_utc_series(df["ts"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["ts"])
        .sort_values(["symbol", "ts"])
        .drop_duplicates(["symbol", "ts"], keep="last")
        .reset_index(drop=True)
    )

    for symbol, g in df.groupby("symbol", sort=False):
        result[str(symbol).upper()] = g[["symbol", "ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    return result



def load_missing_gate3_keys(symbol: str, rebuild: bool, limit_latest: Optional[int]) -> pd.DataFrame:
    global MISSING_GATE3_KEYS_BATCH

    symbol = str(symbol).upper()
    max_allowed_entry_ts = latest_available_gate3_source_ts()

    if MISSING_GATE3_KEYS_BATCH is not None:
        return MISSING_GATE3_KEYS_BATCH.get(
            symbol,
            pd.DataFrame(
                columns=[
                    "symbol",
                    "entry_ts",
                    "signal_ts",
                    "entry_bar_open_ts",
                    "entry_ts_exec",
                    "entry_px_exec",
                ]
            ),
        ).copy()

    if rebuild or not table_exists(ONLINE_GATE3_FEATURES_TABLE):
        where_missing = ""
    else:
        where_missing = f"""
            AND NOT EXISTS (
                SELECT 1
                FROM {ONLINE_GATE3_FEATURES_TABLE} g3
                WHERE g3.symbol = f.symbol
                  AND g3.entry_ts = f.entry_ts
            )
        """

    where_parts = [
        "f.symbol = %s",
        "f.entry_ts <= %s",
    ]
    params: List[object] = [
        symbol,
        max_allowed_entry_ts.to_pydatetime(),
    ]

    append_oos_sql_filters(
        where_parts=where_parts,
        params=params,
        table_alias="f",
        ts_column="entry_ts",
        symbol_column="symbol",
    )

    where_sql = " AND ".join(where_parts)

    limit_clause = ""
    if limit_latest is not None and int(limit_latest) > 0:
        limit_clause = f"LIMIT {int(limit_latest)}"

    query = f"""
        SELECT
            f.symbol,
            f.entry_ts,
            f.signal_ts,
            f.entry_bar_open_ts,
            f.entry_ts_exec,
            f.entry_px_exec
        FROM {ONLINE_GATE2_FEATURES_TABLE} f
        WHERE {where_sql}
        {where_missing}
        ORDER BY f.entry_ts DESC
        {limit_clause}
    """

    with connect_db() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=params,
        )

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()

    for c in ["entry_ts", "signal_ts", "entry_bar_open_ts", "entry_ts_exec"]:
        if c in df.columns:
            df[c] = to_naive_utc_series(df[c])

    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").drop_duplicates(["symbol", "entry_ts"], keep="last")
    df = df.reset_index(drop=True)

    return df
def all_output_columns() -> List[str]:
    cols = []

    for c in (
        OLD_96_COLUMNS
        + ACTIVE_V3_EXTRA_COLUMNS
        + PA_EXTRA_COLUMNS
        + all_gate3_extra_engineered_columns()
    ):
        if c in ONLINE_FORBIDDEN_OUTPUT_COLUMNS:
            continue
        if c not in cols:
            cols.append(c)

    service = [
        "online_source",
        "online_feature_builder",
        "online_created_at",
        "online_updated_at",
    ]

    for c in service:
        if c not in cols:
            cols.append(c)

    return cols


def column_sql_type(col: str) -> str:
    if col in {"symbol", "side", "online_source", "online_feature_builder"}:
        return "TEXT"

    if col in {"entry_ts", "signal_ts", "entry_bar_open_ts", "entry_ts_exec", "online_created_at", "online_updated_at"}:
        return "TIMESTAMPTZ"

    integer_cols = {
        "pa_valid",
        "pa_missing_bars",
        "pa_quality_bucket",
        "gate3_any_active",
        "gate3_active_count",
        "gate3_active_primary",
        "gate3_active_secondary",
        "gate3_active_overlap_primary_secondary",
        "gate3_max_active_age",
        "active_type_count",
        "active_any",
        "active_is_single",
        "active_is_combo",
        "g3_long_any_active",
        "g3_long_active_count",
        "g3_long_active_primary",
        "g3_long_active_secondary",
        "g3_long_active_overlap_primary_secondary",
        "g3_long_max_active_age",
        "g3_long_active_type_count",
        "g3_long_active_any",
        "g3_long_active_is_single",
        "g3_long_active_is_combo",
        "g3_short_any_active",
        "g3_short_active_count",
        "g3_short_active_primary",
        "g3_short_active_secondary",
        "g3_short_active_overlap_primary_secondary",
        "g3_short_max_active_age",
        "g3_short_active_type_count",
        "g3_short_active_any",
        "g3_short_active_is_single",
        "g3_short_active_is_combo",
    }

    if col in integer_cols:
        return "INTEGER"

    if col.startswith("active_pa_") and "__" not in col:
        return "INTEGER"

    if col.startswith("active_pa_") and (
        col.endswith("__age")
        or col.endswith("__fresh")
        or col.endswith("__mid")
        or col.endswith("__late")
    ):
        return "INTEGER"

    return "DOUBLE PRECISION"


def drop_forbidden_online_columns() -> None:
    if not table_exists(ONLINE_GATE3_FEATURES_TABLE):
        return

    existing = set(get_table_columns(ONLINE_GATE3_FEATURES_TABLE))
    drop_cols = [c for c in sorted(ONLINE_FORBIDDEN_OUTPUT_COLUMNS) if c in existing]

    if not drop_cols:
        return

    with connect_db() as conn:
        with conn.cursor() as cur:
            for c in drop_cols:
                cur.execute(
                    f"ALTER TABLE {ONLINE_GATE3_FEATURES_TABLE} "
                    f"DROP COLUMN IF EXISTS {quote_ident(c)}"
                )
        conn.commit()


def ensure_features_table() -> None:
    cols = all_output_columns()

    col_defs = []
    for c in cols:
        if c == "symbol":
            col_defs.append("symbol TEXT NOT NULL")
        elif c == "entry_ts":
            col_defs.append("entry_ts TIMESTAMPTZ NOT NULL")
        elif c == "online_created_at":
            col_defs.append("online_created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        elif c == "online_updated_at":
            col_defs.append("online_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        else:
            col_defs.append(f"{quote_ident(c)} {column_sql_type(c)}")

    query = f"""
        CREATE TABLE IF NOT EXISTS {ONLINE_GATE3_FEATURES_TABLE} (
            {", ".join(col_defs)},
            PRIMARY KEY (symbol, entry_ts)
        )
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()

    existing = get_table_columns(ONLINE_GATE3_FEATURES_TABLE)
    missing = [c for c in cols if c not in existing]

    if missing:
        with connect_db() as conn:
            with conn.cursor() as cur:
                for c in missing:
                    cur.execute(
                        f"ALTER TABLE {ONLINE_GATE3_FEATURES_TABLE} "
                        f"ADD COLUMN {quote_ident(c)} {column_sql_type(c)}"
                    )
            conn.commit()

    drop_forbidden_online_columns()


def get_table_columns(table_name: str) -> List[str]:
    schema, name = split_table_name(table_name)

    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=(schema, name))

    return [str(x) for x in df["column_name"].tolist()]


def clear_features_table() -> None:
    if not table_exists(ONLINE_GATE3_FEATURES_TABLE):
        return

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {ONLINE_GATE3_FEATURES_TABLE}")
        conn.commit()


# ============================================================
# BUILD
# ============================================================

def build_features_for_symbol(
    symbol: str,
    rebuild: bool,
    limit_latest: Optional[int],
    context_bars: int,
    active_max_bars: int,
    active_stop_atr_mult: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    report = {
        "symbol": symbol,
        "status": "unknown",
        "source_rows": 0,
        "built_rows": 0,
        "pa_valid_rows": 0,
        "pa_invalid_rows": 0,
        "h4_rows": 0,
        "active_warmup_bars": int(DEFAULT_ACTIVE_WARMUP_BARS),
        "err": "",
    }

    keys = load_missing_gate3_keys(symbol=symbol, rebuild=rebuild, limit_latest=limit_latest)
    gate2_extra = load_gate2_extra_features(symbol=symbol, rebuild=rebuild, limit_latest=limit_latest)
    report["source_rows"] = int(len(keys))

    if keys.empty:
        report["status"] = "no_missing"
        return pd.DataFrame(), report

    h4 = load_h4_context(symbol)
    report["h4_rows"] = int(len(h4))

    if h4.empty:
        report["status"] = "missing_h4"
        report["err"] = "h4 context from public.candles_h4 is empty"
        return pd.DataFrame(), report

    h4 = h4.sort_index()
    h4 = h4[~h4.index.duplicated(keep="last")].copy()

    target_ts = pd.Series(keys["entry_ts"]).dropna().map(pd.Timestamp)
    if target_ts.empty:
        report["status"] = "empty_target_ts"
        report["err"] = "no valid entry_ts in source keys"
        return pd.DataFrame(), report

    target_set = set(target_ts.tolist())
    min_target_ts = pd.Timestamp(target_ts.min())
    max_target_ts = pd.Timestamp(target_ts.max())

    warmup_start_ts = min_target_ts - pd.Timedelta(seconds=H4_STEP_SECONDS * int(DEFAULT_ACTIVE_WARMUP_BARS))

    build_index = h4.loc[
        (h4.index >= warmup_start_ts) &
        (h4.index <= max_target_ts)
    ].index

    if len(build_index) == 0:
        report["status"] = "empty_h4_build_window"
        report["err"] = f"no h4 rows in build window {warmup_start_ts}..{max_target_ts}"
        return pd.DataFrame(), report

    key_meta = keys.copy()
    key_meta["entry_ts"] = pd.to_datetime(key_meta["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    key_meta = key_meta.dropna(subset=["entry_ts"]).drop_duplicates(["symbol", "entry_ts"], keep="last")
    key_meta = key_meta.set_index("entry_ts", drop=False)

    rows = []
    target_ok_cnt = 0
    target_miss_cnt = 0

    for entry_ts in build_index:
        entry_ts = pd.Timestamp(entry_ts)

        win, chk = get_window_strict(h4=h4, entry_ts=entry_ts, context_bars=context_bars)

        is_target = entry_ts in target_set

        if is_target and entry_ts in key_meta.index:
            meta_row = key_meta.loc[entry_ts]
            signal_ts = meta_row["signal_ts"] if "signal_ts" in meta_row.index else entry_ts
            entry_bar_open_ts = meta_row["entry_bar_open_ts"] if "entry_bar_open_ts" in meta_row.index else entry_ts
            entry_ts_exec = meta_row["entry_ts_exec"] if "entry_ts_exec" in meta_row.index else entry_ts + pd.Timedelta(seconds=ENTRY_DELAY_SECONDS)
            entry_px_exec = meta_row["entry_px_exec"] if "entry_px_exec" in meta_row.index else np.nan
        else:
            signal_ts = entry_ts
            entry_bar_open_ts = entry_ts
            entry_ts_exec = entry_ts + pd.Timedelta(seconds=ENTRY_DELAY_SECONDS)
            entry_px_exec = np.nan

        out = {
            "symbol": symbol,
            "entry_ts": entry_ts,
            "signal_ts": signal_ts,
            "entry_bar_open_ts": entry_bar_open_ts,
            "entry_ts_exec": entry_ts_exec,
            "entry_px": np.nan,
            "entry_px_exec": entry_px_exec,
            "exit_ts": pd.NaT,
            "exit_px": np.nan,
            "exit_reason": "",
            "side": "",
            "y_fast": np.nan,
            "pa_valid": 1 if chk.ok else 0,
            "pa_missing_ratio": float(chk.missing_ratio),
            "pa_missing_bars": int(chk.missing_bars),
            "pa_quality": 1.0 - float(chk.missing_ratio),
            "pa_quality_bucket": 2 if float(chk.missing_ratio) == 0.0 else (1 if float(chk.missing_ratio) <= 0.1 else 0),
            "close": np.nan,
            "_is_target_row": int(is_target),
        }

        if chk.ok and win is not None:
            pa = compute_pa_from_window(win)
            ctx = market_context_from_window(win)
            out.update(pa)
            out.update(ctx)
            out["close"] = float(win["close"].iloc[-1])

            if is_target:
                target_ok_cnt += 1
        else:
            if is_target:
                target_miss_cnt += 1

        rows.append(out)

    if not rows:
        report["status"] = "empty_after_build"
        return pd.DataFrame(), report

    full_df = pd.DataFrame(rows).sort_values("entry_ts").reset_index(drop=True)

    if "ctx_atr14" not in full_df.columns:
        full_df["ctx_atr14"] = np.nan
    if "close" not in full_df.columns:
        full_df["close"] = np.nan

    active_block = compute_stateful_active_block(
        df=full_df,
        max_active_bars=active_max_bars,
        stop_atr_mult=active_stop_atr_mult,
    )

    full_df = pd.concat([full_df, active_block.reset_index(drop=True)], axis=1)

    

    full_df = add_online_gate3_train_compatible_features(
        df=full_df,
        symbol=symbol,
        gate2_extra=gate2_extra,
    )

    out_df = full_df[full_df["_is_target_row"] == 1].copy()
    out_df = out_df.drop(columns=["_is_target_row"], errors="ignore")
    out_df = out_df.sort_values("entry_ts").drop_duplicates(["symbol", "entry_ts"], keep="last").reset_index(drop=True)

    for c in (
        OLD_96_COLUMNS
        + ACTIVE_V3_EXTRA_COLUMNS
        + PA_EXTRA_COLUMNS
        + all_gate3_extra_engineered_columns()
    ):
        if c not in out_df.columns:
            if c.startswith("active_pa_") or c in {
                "pa_valid",
                "pa_missing_bars",
                "pa_quality_bucket",
                "gate3_any_active",
                "gate3_active_count",
                "gate3_active_primary",
                "gate3_active_secondary",
                "gate3_active_overlap_primary_secondary",
                "gate3_max_active_age",
                "active_type_count",
                "active_any",
                "active_is_single",
                "active_is_combo",
            }:
                out_df[c] = 0
            else:
                out_df[c] = np.nan

    out_df["online_source"] = ONLINE_GATE2_FEATURES_TABLE
    out_df["online_feature_builder"] = "online.gate3.build_online_gate3_features"
    out_df["online_created_at"] = utc_now_floor_second()
    out_df["online_updated_at"] = out_df["online_created_at"]

    out_df = out_df.drop(columns=["close"], errors="ignore")

    out_df["pa_quality"] = pd.to_numeric(out_df.get("pa_quality", 0.0), errors="coerce").fillna(0.0)
    out_df["pa_quality_sq"] = out_df["pa_quality"] * out_df["pa_quality"]

    out_df["pa_bos_up_12"] = pd.to_numeric(out_df.get("pa_bos_up_12", 0.0), errors="coerce").fillna(0.0)
    out_df["pa_bos_dn_12"] = pd.to_numeric(out_df.get("pa_bos_dn_12", 0.0), errors="coerce").fillna(0.0)

    out_df["pa_bos_up_12_x_quality"] = out_df["pa_bos_up_12"] * out_df["pa_quality"]
    out_df["pa_bos_dn_12_x_quality"] = out_df["pa_bos_dn_12"] * out_df["pa_quality"]

    cols = all_output_columns()
    out_df = out_df[cols].copy()

    report["status"] = "ok"
    report["built_rows"] = int(len(out_df))
    report["pa_valid_rows"] = int(target_ok_cnt)
    report["pa_invalid_rows"] = int(target_miss_cnt)
    report["warmup_start_ts"] = str(warmup_start_ts)
    report["min_target_ts"] = str(min_target_ts)
    report["max_target_ts"] = str(max_target_ts)
    report["full_sequence_rows_used_for_active"] = int(len(full_df))

    return out_df, report

def clean_value(v):
    if isinstance(v, (pd.Timestamp, datetime, np.datetime64)):
        ts = pd.to_datetime(v, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()

    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, np.generic):
        return v.item()

    return v


def insert_features(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    cols = all_output_columns()

    rows = []
    for row in df[cols].itertuples(index=False, name=None):
        rows.append(tuple(clean_value(v) for v in row))

    insert_cols = ", ".join([quote_ident(c) for c in cols])

    update_cols = [c for c in cols if c not in {"symbol", "entry_ts", "online_created_at"}]
    set_clause = ", ".join([f"{quote_ident(c)} = EXCLUDED.{quote_ident(c)}" for c in update_cols])

    query = f"""
        INSERT INTO {ONLINE_GATE3_FEATURES_TABLE} ({insert_cols})
        VALUES %s
        ON CONFLICT (symbol, entry_ts)
        DO UPDATE SET {set_clause}
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows, page_size=1000)
        conn.commit()

    return int(len(rows))


# ============================================================
# CLI
# ============================================================

def parse_args() -> Tuple[Optional[str], bool, Optional[int], bool, int, int, float]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--symbol", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit-latest", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--context-bars", type=int, default=DEFAULT_CONTEXT_BARS)
    parser.add_argument("--active-max-bars", type=int, default=DEFAULT_ACTIVE_MAX_BARS)
    parser.add_argument("--active-stop-atr-mult", type=float, default=DEFAULT_ACTIVE_STOP_ATR_MULT)

    args = parser.parse_args()

    symbol = None
    if args.symbol:
        symbol = str(args.symbol).upper().strip()

    return (
        symbol,
        bool(args.rebuild),
        args.limit_latest,
        bool(args.dry_run),
        int(args.context_bars),
        int(args.active_max_bars),
        float(args.active_stop_atr_mult),
    )


def main() -> None:
    symbol_arg, rebuild, limit_latest, dry_run, context_bars, active_max_bars, active_stop_atr_mult = parse_args()

    verbose_symbol_logs = str(os.environ.get("IMB_VERBOSE_SYMBOL_LOGS", "0")).strip().lower() in {"1", "true", "yes", "y"}
    max_workers = max(1, int(DEFAULT_MAX_WORKERS))

    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("ONLINE_GATE2_FEATURES_TABLE:", ONLINE_GATE2_FEATURES_TABLE)
    print("ONLINE_GATE3_FEATURES_TABLE:", ONLINE_GATE3_FEATURES_TABLE)
    print("LATEST_CLOSED_H4_OPEN_TS:", latest_closed_h4_open_ts())
    print("LATEST_AVAILABLE_GATE3_SOURCE_TS:", latest_available_gate3_source_ts())
    print("REBUILD:", rebuild)
    print("LIMIT_LATEST:", limit_latest)
    print("DRY_RUN:", dry_run)
    print("CONTEXT_BARS:", context_bars)
    print("ACTIVE_MAX_BARS:", active_max_bars)
    print("ACTIVE_STOP_ATR_MULT:", active_stop_atr_mult)
    print("MAX_WORKERS:", max_workers)
    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_features_table()
    oos_ctx = get_online_oos_context()

    if rebuild and not dry_run and limit_latest is None:
        if oos_ctx.enabled:
            print("REBUILD_WITH_OOS: full features table truncate disabled; OOS rows will be overwritten by upsert")
        else:
            clear_features_table()
    elif rebuild and not dry_run and limit_latest is not None:
        print("REBUILD_WITH_LIMIT_LATEST: full table truncate disabled; latest rows will be overwritten by upsert")

    if symbol_arg:
        symbols = [symbol_arg]
    elif oos_ctx.enabled:
        symbols = list(oos_ctx.symbols)
    else:
        symbols = get_symbols_from_gate2_features()

    print("OOS_MODE:", oos_ctx.enabled)
    print("OOS_SYMBOLS:", ",".join(oos_ctx.symbols))
    print("OOS_START:", oos_ctx.start_text)
    print("OOS_END:", oos_ctx.end_text)
    print("SYMBOLS:", len(symbols))
    print("DB_BATCH_LOAD: missing gate3 keys + gate2 extra features + candles_h4")
    print()

    global MISSING_GATE3_KEYS_BATCH
    global GATE2_EXTRA_FEATURES_BATCH
    global H4_DB_BATCH

    MISSING_GATE3_KEYS_BATCH = load_missing_gate3_keys_batch(
        symbols=symbols,
        rebuild=rebuild,
        limit_latest=limit_latest,
    )

    symbols_to_process = [
        s for s in symbols
        if s in MISSING_GATE3_KEYS_BATCH and len(MISSING_GATE3_KEYS_BATCH[s]) > 0
    ]

    GATE2_EXTRA_FEATURES_BATCH = load_gate2_extra_features_batch(
        symbols=symbols_to_process,
        missing_by_symbol=MISSING_GATE3_KEYS_BATCH,
        context_bars=context_bars,
    )
    H4_DB_BATCH = load_h4_db_batch(
        symbols=symbols_to_process,
        missing_by_symbol=MISSING_GATE3_KEYS_BATCH,
        context_bars=context_bars,
    )

    missing_total = int(sum(len(x) for x in MISSING_GATE3_KEYS_BATCH.values()))
    missing_symbols = int(sum(1 for x in MISSING_GATE3_KEYS_BATCH.values() if len(x) > 0))
    gate2_extra_total = int(sum(len(x) for x in GATE2_EXTRA_FEATURES_BATCH.values()))
    h4_db_total = int(sum(len(x) for x in H4_DB_BATCH.values()))

    print("MISSING_GATE3_KEYS_TOTAL:", missing_total)
    print("MISSING_GATE3_SYMBOLS:", missing_symbols)
    print("SYMBOLS_TO_PROCESS:", len(symbols_to_process))
    print("GATE2_EXTRA_ROWS_BATCH:", gate2_extra_total)
    print("H4_DB_ROWS_BATCH:", h4_db_total)
    print()

    reports = []
    built_frames = []

    if symbols_to_process:
        workers = min(max_workers, len(symbols_to_process))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_symbol = {}

            for symbol in symbols_to_process:
                fut = pool.submit(
                    build_features_for_symbol,
                    symbol,
                    rebuild,
                    limit_latest,
                    context_bars,
                    active_max_bars,
                    active_stop_atr_mult,
                )
                future_to_symbol[fut] = symbol

            for fut in as_completed(future_to_symbol):
                symbol = future_to_symbol[fut]

                try:
                    built, rep = fut.result()
                    built_frames.append(built)
                    reports.append(rep)

                    if verbose_symbol_logs:
                        print(
                            "{} | status={} | source={} | built={} | pa_valid={} | pa_invalid={} | h4_rows={}".format(
                                symbol,
                                rep.get("status"),
                                rep.get("source_rows", 0),
                                len(built),
                                rep.get("pa_valid_rows", 0),
                                rep.get("pa_invalid_rows", 0),
                                rep.get("h4_rows", 0),
                            )
                        )

                except Exception as e:
                    rep = {
                        "symbol": symbol,
                        "status": "error",
                        "source_rows": 0,
                        "built_rows": 0,
                        "inserted_rows": 0,
                        "pa_valid_rows": 0,
                        "pa_invalid_rows": 0,
                        "h4_rows": 0,
                        "err": repr(e),
                    }
                    reports.append(rep)
                    print("{} | ERROR: {}".format(symbol, rep["err"]))

    no_missing_symbols = [s for s in symbols if s not in symbols_to_process]
    for symbol in no_missing_symbols:
        reports.append({
            "symbol": symbol,
            "status": "no_missing",
            "source_rows": 0,
            "built_rows": 0,
            "inserted_rows": 0,
            "pa_valid_rows": 0,
            "pa_invalid_rows": 0,
            "h4_rows": 0,
            "err": "",
        })

    non_empty_built_frames = [x for x in built_frames if x is not None and not x.empty]

    if non_empty_built_frames:
        all_built = pd.concat(non_empty_built_frames, ignore_index=True)
    else:
        all_built = pd.DataFrame()

    total_built = int(len(all_built))

    if dry_run or all_built.empty:
        total_inserted = 0
    else:
        total_inserted = insert_features(all_built)

    for rep in reports:
        if rep.get("status") == "ok":
            rep["inserted_rows"] = int(rep.get("built_rows", 0))

    rep_df = pd.DataFrame(reports)
    if len(rep_df):
        rep_df = rep_df.sort_values("symbol").reset_index(drop=True)

    rep_df.to_csv(REPORT_CSV, index=False)

    summary = {
        "created_at_utc": str(utc_now_floor_second()),
        "root": str(ROOT),
        "online_gate2_features_table": ONLINE_GATE2_FEATURES_TABLE,
        "online_gate3_features_table": ONLINE_GATE3_FEATURES_TABLE,
        "h4_source": "public.candles_h4",
        "latest_closed_h4_open_ts": str(latest_closed_h4_open_ts()),
        "latest_available_gate3_source_ts": str(latest_available_gate3_source_ts()),
        "symbols_count": int(len(symbols)),
        "symbols_to_process": int(len(symbols_to_process)),
        "rebuild": bool(rebuild),
        "limit_latest": limit_latest,
        "dry_run": bool(dry_run),
        "context_bars": int(context_bars),
        "active_max_bars": int(active_max_bars),
        "active_stop_atr_mult": float(active_stop_atr_mult),
        "max_workers": int(max_workers),
        "total_built": int(total_built),
        "total_inserted": int(total_inserted),
        "status_counts": rep_df["status"].value_counts(dropna=False).to_dict() if len(rep_df) else {},
        "report_csv": str(REPORT_CSV),
    }

    REPORT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 120)
    print("DONE")
    print("STATUS COUNTS:", summary["status_counts"])
    print("SYMBOLS TO PROCESS:", len(symbols_to_process))
    print("TOTAL BUILT:", total_built)
    print("TOTAL INSERTED:", total_inserted)
    print("WROTE:", REPORT_CSV)
    print("WROTE:", REPORT_JSON)


if __name__ == "__main__":
    main()
