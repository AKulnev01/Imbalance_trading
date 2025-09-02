# evaluate_signals.py
import os
import sys
import math
import pandas as pd
from datetime import timedelta, datetime, timezone
from typing import Tuple

import config as CFG
from utils.fetch_data import get_bybit_klines

# ===== helpers to read config/env without hardcoding =====
def _get_cfg(name, *, required=True, cast=None, default=None):
    val = getattr(CFG, name, None)
    if val is None:
        val = os.getenv(name, None)
    if val is None:
        if required and default is None:
            raise RuntimeError(
                f"Missing required setting '{name}' in config.py or .env. "
                f"Please define {name} (example: see README/.env.example)."
            )
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

# ===== core settings =====
INITIAL_CAPITAL   = _get_cfg("INITIAL_CAPITAL",   required=True, cast=float)
POSITION_FRACTION = _get_cfg("POSITION_FRACTION", required=True, cast=float)
FEE_TAKER         = _get_cfg("FEE_TAKER",         required=True, cast=float)
SLIPPAGE_PCT      = _get_cfg("SLIPPAGE_PCT",      required=True, cast=float)
MAX_FILL_DAYS     = _get_cfg("MAX_FILL_DAYS",     required=True, cast=int)

ENTRY_MODE        = _get_cfg("ENTRY_MODE",        required=True, cast=str).upper()

# momentum execution realism
MOMENTUM_EXEC           = _get_cfg("MOMENTUM_EXEC",           required=True, cast=str).lower()      # market | aggr_limit
AGGR_LIMIT_EPS_PCT      = _get_cfg("AGGR_LIMIT_EPS_PCT",      required=True, cast=float)
MOMENTUM_FALLBACK       = _get_cfg("MOMENTUM_FALLBACK",       required=True, cast=str).lower()      # market | none
MAX_ACCEPT_SLIPPAGE_PCT = _get_cfg("MAX_ACCEPT_SLIPPAGE_PCT", required=True, cast=float)
MOMENTUM_FILL_WINDOW_MIN= _get_cfg("MOMENTUM_FILL_WINDOW_MIN",required=True, cast=int)
MAX_CONCURRENT_POSITIONS= _get_cfg("MAX_CONCURRENT_POSITIONS",required=True, cast=int)
MOMENTUM_MIN_LTF_BARS   = _get_cfg("MOMENTUM_MIN_LTF_BARS",   required=True, cast=int)

# === DEEP_RETEST settings ===
DEEP_RETEST_DYNAMIC    = _get_cfg("DEEP_RETEST_DYNAMIC",    required=True, cast=bool)
DEEP_RETEST_PCT        = _get_cfg("DEEP_RETEST_PCT",        required=True, cast=float, default=0.05)
DEEP_STRENGTH_MIN      = _get_cfg("DEEP_STRENGTH_MIN",      required=True, cast=float, default=2.0)
DEEP_STRENGTH_MAX      = _get_cfg("DEEP_STRENGTH_MAX",      required=True, cast=float, default=6.0)
DEEP_DEPTH_MIN_PCT     = _get_cfg("DEEP_DEPTH_MIN_PCT",     required=True, cast=float, default=0.02)
DEEP_DEPTH_MAX_PCT     = _get_cfg("DEEP_DEPTH_MAX_PCT",     required=True, cast=float, default=0.08)
DEEP_TP_MODE           = _get_cfg("DEEP_TP_MODE",           required=True, cast=str, default="rr").lower()
DEEP_RR                = _get_cfg("DEEP_RR",                required=True, cast=float, default=3.0)
DEEP_LADDER = os.getenv("DEEP_LADDER", "").strip()  # "0.008:0.5,0.016:0.3,0.024:0.2"

FVG_TOP_COL            = getattr(CFG, "FVG_TOP_COL", "fvg_top")
FVG_BOTTOM_COL         = getattr(CFG, "FVG_BOTTOM_COL", "fvg_bottom")

DEFAULT_TTL_DAYS  = _get_cfg("DEFAULT_TTL_DAYS",  required=True, cast=int)

# momentum RR/TP/SL
MOMENTUM_TP_PCT   = _get_cfg("MOMENTUM_TP_PCT",   required=True, cast=float)
MOMENTUM_SL_PCT   = _get_cfg("MOMENTUM_SL_PCT",   required=True, cast=float)
STOP_PCT = MOMENTUM_SL_PCT
TAKE_PCT = MOMENTUM_TP_PCT

# LTF windows
INTRABAR_INTERVALS              = _get_cfg("INTRABAR_INTERVALS",              required=True, cast=list)
INTRABAR_LOOKBACK_DAYS_FALLBACK = _get_cfg("INTRABAR_LOOKBACK_DAYS_FALLBACK", required=True, cast=int)

RISK_PCT = getattr(CFG, "RISK_PCT", None)
RISK_REWARD_RATIO = getattr(CFG, "RISK_REWARD_RATIO", None)

VARIANT_COL_CANDIDATES = ["variant", "mode", "entry_mode", "strategy"]

MOMENTUM_FILL_WINDOW_MIN= _get_cfg("MOMENTUM_FILL_WINDOW_MIN",required=True, cast=int)
MAX_CONCURRENT_POSITIONS= _get_cfg("MAX_CONCURRENT_POSITIONS",required=True, cast=int)
MOMENTUM_MIN_LTF_BARS   = _get_cfg("MOMENTUM_MIN_LTF_BARS",   required=True, cast=int)
# ===================== Утилиты =====================
def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
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

def _to_num(x):
    return pd.to_numeric(str(x).replace(',', '.').strip(), errors='coerce')

def _find_variant_col(df: pd.DataFrame) -> str:
    for c in VARIANT_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _row_variant(row, variant_col: str) -> str:
    if variant_col and pd.notna(row.get(variant_col)):
        return str(row[variant_col]).strip().upper()
    return ENTRY_MODE

def _calc_sl_tp(entry: float, side: str, risk_pct_price: float, rr: float) -> Tuple[float, float]:
    k = float(risk_pct_price)
    if side == "SELL":
        sl = entry * (1.0 + k)
        tp = entry - (sl - entry) * rr
    else:
        sl = entry * (1.0 - k)
        tp = entry + (entry - sl) * rr
    return float(sl), float(tp)

def _first_touch_after(df: pd.DataFrame, entry: float, t0: pd.Timestamp) -> pd.Timestamp:
    if df is None or df.empty or pd.isna(entry) or pd.isna(t0):
        return pd.NaT
    win = df[(df.index > t0)]
    for ts, row in win.iterrows():
        if float(row['low']) <= entry <= float(row['high']):
            return ts
    return pd.NaT

def _repair_levels(side, entry, stop_eval, tp_eval):
    ok = True
    if side == 'BUY':
        ok = (stop_eval < entry) and (tp_eval > entry)
    else:
        ok = (tp_eval < entry) and (stop_eval > entry)
    if ok:
        return float(stop_eval), float(tp_eval), False
    rr = TAKE_PCT / max(STOP_PCT, 1e-9)
    s, t = _calc_sl_tp(float(entry), side, STOP_PCT, rr)
    return float(s), float(t), True

def _momentum_entry(symbol: str, side: str, df_baseTF: pd.DataFrame, t0: pd.Timestamp) -> Tuple[pd.Timestamp, float, str]:
    """
    Реалистичный маркет-вход:
      • last_px = close бара t0 (или ближайшего)
      • берём ПЕРВОЕ закрытие LTF после t0 в окне MOMENTUM_FILL_WINDOW_MIN
      • проверяем слиппедж относительно last_px
      • возврат: (entry_at, entry_px, entry_exec) или (NaT, NaN, 'skip_*')
    """
    # last_px по базовому ТФ
    bar = df_baseTF[df_baseTF.index == t0]
    if bar.empty:
        try:
            nearest_idx = df_baseTF.index.get_indexer([t0], method="nearest")[0]
            bar = df_baseTF.iloc[[nearest_idx]]
        except Exception:
            return (pd.NaT, float("nan"), "no_bar")
    last_px = float(bar.iloc[0]["close"])

    # окно LTF
    t_start = t0
    t_end   = t0 + pd.Timedelta(minutes=int(MOMENTUM_FILL_WINDOW_MIN))
    ltf = _fetch_ltf_window(symbol, t_start, t_end, candidates=INTRABAR_INTERVALS)
    if ltf.empty or len(ltf) < max(1, MOMENTUM_MIN_LTF_BARS):
        return (pd.NaT, float("nan"), "no_ltf")

    # первое закрытие после t0
    first_ts = ltf.index[0]
    close_px = float(ltf.iloc[0]["close"])

    # защита по слиппеджу
    if side == "BUY":
        rel = (close_px - last_px) / max(last_px, 1e-9)
        if rel > MAX_ACCEPT_SLIPPAGE_PCT:
            return (pd.NaT, float("nan"), "skip_slippage")
    else:
        rel = (last_px - close_px) / max(last_px, 1e-9)
        if rel > MAX_ACCEPT_SLIPPAGE_PCT:
            return (pd.NaT, float("nan"), "skip_slippage")

    return (first_ts, close_px, "market_ltf_close")

def _safe_group_exit_reason(df_res: pd.DataFrame) -> pd.DataFrame:
    df = df_res.copy()
    if 'skipped' in df.columns:
        df = df[df['skipped'] == False].copy()
    for col, default in [
        ('exit_reason', 'unknown'),
        ('win', False),
        ('pnl_pct', 0.0),
        ('pnl_usd', 0.0),
        ('exit_days', pd.NA),
    ]:
        if col not in df.columns:
            df[col] = default
    df['win'] = df['win'].astype('bool')
    df['pnl_pct'] = pd.to_numeric(df['pnl_pct'], errors='coerce').fillna(0.0)
    df['pnl_usd'] = pd.to_numeric(df['pnl_usd'], errors='coerce').fillna(0.0)
    df['exit_days'] = pd.to_numeric(df['exit_days'], errors='coerce')

    def _winrate_safe(s):
        n = int(s.size) if s is not None else 0
        return round(100.0 * float(s.sum()) / float(n), 2) if n > 0 else 0.0

    if df.empty:
        return pd.DataFrame(columns=[
            'exit_reason','trades','wins','winrate_pct','pnl_pct','pnl_usd',
            'avg_exit_days','med_exit_days'
        ])
    try:
        by_exit_reason = (
            df.groupby('exit_reason', dropna=False)
              .agg(
                  trades=('win', 'size'),
                  wins=('win', 'sum'),
                  winrate_pct=('win', _winrate_safe),
                  pnl_pct=('pnl_pct', 'sum'),
                  pnl_usd=('pnl_usd', 'sum'),
                  avg_exit_days=('exit_days', 'mean'),
                  med_exit_days=('exit_days', 'median'),
              )
              .reset_index()
              .sort_values(['pnl_usd', 'winrate_pct'], ascending=[False, False])
        )
        return by_exit_reason
    except TypeError:
        grouped = (
            df.groupby('exit_reason', dropna=False)
              .agg({
                  'win': ['size', 'sum'],
                  'pnl_pct': ['sum'],
                  'pnl_usd': ['sum'],
                  'exit_days': ['mean', 'median'],
              })
        )
        grouped.columns = ['_'.join([c for c in col if c]) for c in grouped.columns.to_flat_index()]
        rename_map = {
            'win_size': 'trades',
            'win_sum': 'wins',
            'pnl_pct_sum': 'pnl_pct',
            'pnl_usd_sum': 'pnl_usd',
            'exit_days_mean': 'avg_exit_days',
            'exit_days_median': 'med_exit_days',
        }
        by_exit_reason = grouped.rename(columns=rename_map).reset_index()
        by_exit_reason['winrate_pct'] = (
            (by_exit_reason['wins'] / by_exit_reason['trades'])
            .replace([pd.NA, pd.NaT], 0).fillna(0).astype(float) * 100.0
        ).round(2)
        by_exit_reason = by_exit_reason.sort_values(['pnl_usd', 'winrate_pct'], ascending=[False, False])
        return by_exit_reason

def _enforce_one_at_a_time_per_symbol(df_res: pd.DataFrame) -> pd.DataFrame:
    if df_res is None or df_res.empty:
        return df_res
    df = df_res.copy()
    for c in ["imb_time", "t_start", "close_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df.sort_values(
        ["symbol", "imb_time", "t_start", "close_time"],
        kind="mergesort"
    ).reset_index(drop=True)

    out_rows = []
    last_close_by_sym = {}
    for _, r in df.iterrows():
        sym = str(r.get("symbol"))
        t_start = r.get("t_start")
        t_close = r.get("close_time")
        prev_close = last_close_by_sym.get(sym, pd.Timestamp.min.tz_localize("UTC"))

        if pd.isna(t_start):
            if pd.notna(t_close) and t_close > prev_close:
                last_close_by_sym[sym] = t_close
            out_rows.append(r.to_dict())
            continue

        if t_start < prev_close:
            rr = r.to_dict()
            rr["skipped"] = True
            rr["exit_reason"] = "skipped_overlap"
            rr["pnl_usd"] = pd.NA
            rr["pnl_pct"] = pd.NA
            rr["alloc_usd_comp"] = pd.NA
            rr["pnl_usd_comp"] = pd.NA
            rr["equity_after"] = pd.NA
            out_rows.append(rr)
            continue

        if pd.notna(t_close) and t_close > prev_close:
            last_close_by_sym[sym] = t_close
        out_rows.append(r.to_dict())
    return pd.DataFrame(out_rows).reset_index(drop=True)

def _clamp(x, a, b):
    return max(a, min(b, x))

def _depth_from_strength(strength: float) -> float:
    s_min = float(DEEP_STRENGTH_MIN)
    s_max = float(DEEP_STRENGTH_MAX)
    d_min = float(DEEP_DEPTH_MIN_PCT)
    d_max = float(DEEP_DEPTH_MAX_PCT)
    if math.isnan(strength):
        return d_min
    if s_max <= s_min:
        return d_min
    t = (float(strength) - s_min) / (s_max - s_min)
    t = _clamp(t, 0.0, 1.0)
    return d_min + t * (d_max - d_min)

def _to_utc_safe(ts):
    if pd.isna(ts):
        return pd.NaT
    t = pd.to_datetime(ts, errors='coerce')
    if t is pd.NaT:
        return pd.NaT
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize('UTC')
    return t.tz_convert('UTC')

def _parse_ladder(spec: str):
    steps = []
    for chunk in str(spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            d,w = chunk.split(":")
            depth = float(d); weight = float(w)
            if depth > 0 and weight > 0:
                steps.append((depth, weight))
        except Exception:
            continue
    if not steps:
        return []
    s = sum(w for _,w in steps)
    return [(d, w/s) for d,w in steps if s > 0]

# ===== LTF helpers =====
def _fetch_ltf_window(symbol: str, t_start: pd.Timestamp, t_end: pd.Timestamp, candidates=None) -> pd.DataFrame:
    if candidates is None:
        candidates = INTRABAR_INTERVALS
    days = max(1, int((t_end - t_start).total_seconds() // 86400) + 2)
    days = max(days, INTRABAR_LOOKBACK_DAYS_FALLBACK)
    for iv in [c.strip() for c in candidates if c.strip()]:
        try:
            df_ltf = get_bybit_klines(symbol=symbol, interval=iv, lookback_days=days)
        except Exception:
            continue
        df_ltf = _ensure_dt_index(df_ltf)
        if df_ltf is None or df_ltf.empty:
            continue
        win = df_ltf[(df_ltf.index >= t_start) & (df_ltf.index <= t_end)].copy()
        if not win.empty:
            return win
    return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))

def _resolve_tp_sl_order_ltf(symbol: str, side: str, entry_at: pd.Timestamp, bar_close_time: pd.Timestamp,
                             stop_eval: float, tp_eval: float) -> Tuple[bool, pd.Timestamp, float, str]:
    t0 = _to_utc_safe(entry_at)
    t1 = _to_utc_safe(bar_close_time)
    if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
        return (False, t1, float(stop_eval), 'sl')
    ltf = _fetch_ltf_window(symbol, t0, t1, candidates=INTRABAR_INTERVALS)
    if ltf.empty:
        return (False, t1, float(stop_eval), 'sl')
    for ts, c in ltf.iterrows():
        hi, lo = float(c['high']), float(c['low'])
        if side == 'BUY':
            hit_tp = (hi >= tp_eval); hit_sl = (lo <= stop_eval)
        else:
            hit_tp = (lo <= tp_eval);  hit_sl = (hi >= stop_eval)
        if hit_tp and not hit_sl:
            return (True, ts, float(tp_eval), 'tp')
        if hit_sl and not hit_tp:
            return (False, ts, float(stop_eval), 'sl')
        if hit_tp and hit_sl:
            return (False, ts, float(stop_eval), 'sl')
    return (False, t1, float(stop_eval), 'uncertain')

# ===== Реалистичная симуляция капитала =====
def _simulate_capital_notional(
        df_res: pd.DataFrame,
        initial_capital: float,
        position_fraction: float,
        stop_pct: float,
        take_pct: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_res is None or df_res.empty:
        return df_res, pd.DataFrame()
    df = df_res.copy()
    for c in ("t_start", "close_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df.sort_values(
        ["symbol", "imb_time", "t_start", "close_time"],
        kind="mergesort"
    ).reset_index(drop=True)

    equity = float(initial_capital)
    free_cash = equity
    active = []  # [{idx,symbol,size,size_weight,close_time,exit_reason,pnl_usd}]
    out_rows = []
    eq_rows = []
    fees_slip_pos = (float(FEE_TAKER) * 2.0) + (float(SLIPPAGE_PCT) * 2.0)

    def _close_until(ts):
        nonlocal equity, free_cash, active, eq_rows
        still = []
        for pos in active:
            ctime = pos["close_time"]
            if pd.notna(ctime) and ctime <= ts:
                pnl = float(pos["pnl_usd"])
                eq_before = equity
                equity += pnl
                free_cash += pos["size"] + pnl
                eq_rows.append({
                    "i": pos["idx"] + 1,
                    "time": ctime,
                    "symbol": pos["symbol"],
                    "alloc_usd": round(pos["size"], 2),
                    "pnl_usd_comp": round(pnl, 2),
                    "equity_before": round(eq_before, 2),
                    "equity_after": round(equity, 2),
                    "exit_reason": pos["exit_reason"],
                    "size_weight": pos.get("size_weight", pd.NA),
                })
            else:
                still.append(pos)
        active = still

    for i, r in df.iterrows():
        t_start = r.get("t_start")
        t_close = r.get("close_time")
        symbol = r.get("symbol")

        if pd.notna(t_start):
            _close_until(t_start)

        if pd.isna(t_start):
            row = r.to_dict()
            row.update({
                "skipped": True,
                "alloc_usd_comp": pd.NA,
                "pnl_usd_comp": pd.NA,
                "equity_after": pd.NA,
                "pnl_usd": pd.NA,
                "pnl_pct": pd.NA,
            })
            out_rows.append(row)
            continue

        # жёсткий лимит по слотам
        if MAX_CONCURRENT_POSITIONS is not None and int(MAX_CONCURRENT_POSITIONS) > 0:
            if len(active) >= int(MAX_CONCURRENT_POSITIONS):
                row = r.to_dict()
                row.update({
                    "skipped": True,
                    "exit_reason": "skipped_slots_full",
                    "alloc_usd_comp": pd.NA,
                    "pnl_usd_comp": pd.NA,
                    "equity_after": pd.NA,
                    "pnl_usd": pd.NA,
                    "pnl_pct": pd.NA,
                })
                out_rows.append(row)
                continue

        size_weight = r.get("size_weight")
        try:
            size_weight = float(size_weight) if pd.notna(size_weight) else 1.0
        except Exception:
            size_weight = 1.0
        size_weight = max(0.0, min(1.0, size_weight))

        size = max(0.0, float(position_fraction) * float(equity) * size_weight)

        if free_cash < size or size <= 0:
            row = r.to_dict()
            row.update({
                "skipped": True,
                "alloc_usd_comp": pd.NA,
                "pnl_usd_comp": pd.NA,
                "equity_after": pd.NA,
                "pnl_usd": pd.NA,
                "pnl_pct": pd.NA,
                "exit_reason": "skipped_no_capital"
            })
            out_rows.append(row)
            continue

        free_cash -= size

        er = str(r.get("exit_reason") or "").lower()
        price_pnl_pct = r.get("pnl_pct")
        if er == "tp":
            pnl_usd = (float(take_pct) - fees_slip_pos) * size
        elif er == "sl":
            pnl_usd = -(float(stop_pct) + fees_slip_pos) * size
        else:
            try:
                pnl_usd = (float(price_pnl_pct) / 100.0) * size if pd.notna(price_pnl_pct) else 0.0
            except Exception:
                pnl_usd = 0.0

        active.append({
            "idx": i,
            "symbol": symbol,
            "size": size,
            "size_weight": size_weight,
            "close_time": t_close if pd.notna(t_close) else t_start,
            "exit_reason": r.get("exit_reason"),
            "pnl_usd": float(pnl_usd),
        })

        row = r.to_dict()
        row.update({
            "skipped": False,
            "pnl_usd": round(pnl_usd, 2),
            "alloc_usd_comp": pd.NA,
            "pnl_usd_comp": pd.NA,
            "equity_after": pd.NA,
        })
        out_rows.append(row)

    active.sort(key=lambda x: x["close_time"] if pd.notna(x["close_time"]) else pd.Timestamp.max.tz_localize("UTC"))
    for pos in active:
        pnl = float(pos["pnl_usd"])
        eq_before = float(equity)
        equity += pnl
        free_cash += pos["size"] + pnl
        eq_rows.append({
            "i": pos["idx"] + 1,
            "time": pos["close_time"],
            "symbol": pos["symbol"],
            "alloc_usd": round(pos["size"], 2),
            "pnl_usd_comp": round(pnl, 2),
            "equity_before": round(eq_before, 2),
            "equity_after": round(equity, 2),
            "exit_reason": pos["exit_reason"],
            "size_weight": pos.get("size_weight", pd.NA),
        })

    df_out = pd.DataFrame(out_rows).reset_index(drop=True)
    for e in eq_rows:
        idx = e["i"] - 1
        if 0 <= idx < len(df_out):
            df_out.at[idx, "alloc_usd_comp"] = e["alloc_usd"]
            df_out.at[idx, "pnl_usd_comp"] = e["pnl_usd_comp"]
            df_out.at[idx, "equity_after"] = e["equity_after"]

    if not eq_rows:
        eq_curve = pd.DataFrame(eq_rows)
        if not eq_curve.empty and 'time' in eq_curve.columns:
            eq_curve = eq_curve.sort_values("time").reset_index(drop=True)
        else:
            eq_curve = pd.DataFrame(
                columns=["i", "time", "symbol", "alloc_usd", "pnl_usd_comp", "equity_before", "equity_after",
                         "exit_reason", "size_weight"])
    else:
        eq_curve = (
            pd.DataFrame(eq_rows)
            .sort_values("time")
            .reset_index(drop=True)
        )
    return df_out, eq_curve

def _norm_ts_utc(x):
    try:
        return pd.to_datetime(x, utc=True, errors='coerce')
    except Exception:
        return pd.NaT

# ===== Новое: реалистичный вход для MOMENTUM =====
def _momentum_entry(symbol: str, side: str, df_baseTF: pd.DataFrame, t0: pd.Timestamp) -> Tuple[pd.Timestamp, float, str]:
    """
    Возвращает (entry_at, entry_px, entry_exec).
    Логика:
      • market: берём ПЕРВОЕ закрытие LTF после t0; проверяем слиппедж vs last_px (закрытие бара t0).
      • aggr_limit: лимитка по last_px*(1±eps); считаем fill, только если LTF в окне коснулся цены.
         - если не коснулся и fallback=market → как market (с лимитом по слиппеджу);
         - иначе → (NaT, NaN, 'no_fill').
    """
    # last_px = close бара t0 (или ближайшего)
    bar = df_baseTF[df_baseTF.index == t0]
    if bar.empty:
        try:
            nearest_idx = df_baseTF.index.get_indexer([t0], method="nearest")[0]
            bar = df_baseTF.iloc[[nearest_idx]]
        except Exception:
            return (pd.NaT, float("nan"), "no_bar")
    last_px = float(bar.iloc[0]["close"])
    exec_mode = str(MOMENTUM_EXEC).lower()
    fallback = str(MOMENTUM_FALLBACK).lower()

    # окно LTF
    t_start = t0
    t_end   = t0 + pd.Timedelta(minutes=int(MOMENTUM_FILL_WINDOW_MIN))
    ltf = _fetch_ltf_window(symbol, t_start, t_end, candidates=INTRABAR_INTERVALS)
    if ltf.empty or len(ltf) < max(1, MOMENTUM_MIN_LTF_BARS):
        return (pd.NaT, float("nan"), "no_ltf")

    if exec_mode == "market":
        # берём ПЕРВОЕ закрытие LTF после t0
        first_ts = ltf.index[0]
        close_px = float(ltf.iloc[0]["close"])
        # защита по слиппеджу
        if side == "BUY":
            rel = (close_px - last_px) / max(last_px, 1e-9)
            if rel > MAX_ACCEPT_SLIPPAGE_PCT:
                return (pd.NaT, float("nan"), "skip_slippage")
        else:
            rel = (last_px - close_px) / max(last_px, 1e-9)
            if rel > MAX_ACCEPT_SLIPPAGE_PCT:
                return (pd.NaT, float("nan"), "skip_slippage")
        return (first_ts, close_px, "market_ltf_close")

    # aggr_limit
    if side == "BUY":
        limit_px = last_px * (1.0 + float(AGGR_LIMIT_EPS_PCT))
    else:
        limit_px = last_px * (1.0 - float(AGGR_LIMIT_EPS_PCT))

    # считаем fill, если LTF коснулся
    touched = False
    touched_ts = None
    for ts, c in ltf.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if side == "BUY":
            if lo <= limit_px <= hi:
                touched = True; touched_ts = ts; break
        else:
            if lo <= limit_px <= hi:
                touched = True; touched_ts = ts; break

    if touched:
        return (touched_ts, float(limit_px), "aggr_limit_filled")

    if fallback == "market":
        # fallback как market (первое закрытие), но с лимитом по слиппеджу
        first_ts = ltf.index[0]
        close_px = float(ltf.iloc[0]["close"])
        if side == "BUY":
            rel = (close_px - last_px) / max(last_px, 1e-9)
            if rel > MAX_ACCEPT_SLIPPAGE_PCT:
                return (pd.NaT, float("nan"), "skip_slippage")
        else:
            rel = (last_px - close_px) / max(last_px, 1e-9)
            if rel > MAX_ACCEPT_SLIPPAGE_PCT:
                return (pd.NaT, float("nan"), "skip_slippage")
        return (first_ts, close_px, "aggr_limit_fallback_market")
    else:
        return (pd.NaT, float("nan"), "no_fill")

# ===================== Основной пайплайн =====================
def evaluate_signals(
    signals_path: str,
    result_path: str,
    lookback_days: int = 360,
    interval: str = '4h',
    max_days: int = None,
    include_open: bool = False,
    compounding: bool = True,
    initial_capital: float = None,
    capital_aware: bool = True,
    only_filled: bool = False,
    dedup: bool = False,
):
    try:
        mtime = os.path.getmtime(signals_path)
        as_of = datetime.fromtimestamp(mtime, tz=timezone.utc)
    except Exception:
        as_of = datetime.now(tz=timezone.utc)

    head = pd.read_excel(signals_path, nrows=0)
    parse_dates = [c for c in ['imb_time', 'entry_at'] if c in head.columns]
    df_sig = pd.read_excel(signals_path, parse_dates=parse_dates)

    if "symbol" in df_sig.columns:
        df_sig["symbol"] = df_sig["symbol"].astype(str).str.upper().str.strip()
    df_sig = df_sig[df_sig.get("symbol").notna()]

    if "imb_time" in df_sig.columns:
        df_sig["imb_time"] = pd.to_datetime(df_sig["imb_time"], utc=True, errors="coerce")
        df_sig = df_sig[df_sig["imb_time"].notna()]

    for c in ("entry", "stop", "tp", "strength"):
        if c in df_sig.columns:
            df_sig[c] = pd.to_numeric(df_sig[c], errors="coerce")

    if "filled" in df_sig.columns:
        df_sig["filled"] = df_sig["filled"].map(
            lambda x: str(x).strip().lower() in ("1", "true", "yes", "y", "да", "истина"))

    sort_cols = [c for c in ["symbol", "imb_time"] if c in df_sig.columns]
    if sort_cols:
        df_sig = df_sig.sort_values(sort_cols).reset_index(drop=True)

    if only_filled and "filled" in df_sig.columns:
        df_sig = df_sig[df_sig["filled"] == True].copy()

    if dedup and set(["symbol", "imb_time"]).issubset(df_sig.columns):
        df_sig = (df_sig.sort_values(["symbol", "imb_time"])
                        .drop_duplicates(subset=["symbol", "imb_time"], keep="first")
                        .reset_index(drop=True))
    if df_sig.empty:
        print("⚠️ После фильтров сигналов не осталось — выходим.")
        return

    results = []
    price_cache = {}
    symbols = df_sig['symbol'].dropna().unique().tolist()
    for symbol in symbols:
        try:
            df_hist = get_bybit_klines(symbol=symbol, interval=interval, lookback_days=lookback_days)
        except Exception:
            df_hist = pd.DataFrame()
        df_hist = _ensure_dt_index(df_hist)
        price_cache[symbol] = df_hist

    variant_col = _find_variant_col(df_sig)

    for _, row in df_sig.iterrows():
        symbol   = row['symbol']
        t0       = _to_utc_safe(row['imb_time'])
        side     = str(row['type']).upper().strip()
        entry    = _to_num(row['entry'])
        df       = price_cache.get(symbol)
        mode     = _row_variant(row, variant_col)

        if df is None or df.empty or pd.isna(entry) or side not in ('BUY', 'SELL'):
            continue

        entry_at = None
        entry_px = None
        _deep_depth_used = None
        _deep_tp_mode = None
        _size_weight_used = 1.0

        if mode == "BREAKOUT":
            bar = df[df.index == t0]
            if bar.empty:
                try:
                    nearest_idx = df.index.get_indexer([t0], method="nearest")[0]
                    bar = df.iloc[[nearest_idx]]
                except Exception:
                    continue
            entry_px = float(bar.iloc[0]['close'])
            rr = TAKE_PCT / max(STOP_PCT, 1e-9)
            stop_eval, tp_eval = _calc_sl_tp(entry_px, side, STOP_PCT, rr)
            entry_at = t0


        elif mode == "MOMENTUM":

            # реалистичный вход с LTF-валидацией

            e_at, e_px, e_exec = _momentum_entry(symbol, side, df, t0)

            entry_at, entry_px = e_at, e_px

            if pd.isna(entry_at) or (isinstance(entry_px, float) and math.isnan(entry_px)):

                # не смогли войти реалистично — сделки нет

                stop_eval = float('nan');
                tp_eval = float('nan')

            else:

                rr = float(MOMENTUM_TP_PCT) / max(float(MOMENTUM_SL_PCT), 1e-9)

                stop_eval, tp_eval = _calc_sl_tp(entry_px, side, float(MOMENTUM_SL_PCT), rr)


        elif mode == "DEEP_RETEST":

            entry_base = float(entry)

            ladder = _parse_ladder(DEEP_LADDER)

            depth_pct_used = None

            size_weight_used = 1.0

            if ladder:

                # лестница: берём первую сработавшую ступень в TTL

                candidates = []

                for depth_pct, w in ladder:

                    if side == "BUY":

                        px = entry_base * (1.0 - depth_pct)

                    else:

                        px = entry_base * (1.0 + depth_pct)

                    ft = _first_touch_after(df, float(px), t0)

                    if pd.notna(ft):
                        candidates.append((ft, depth_pct, w, px))

                ttl_deadline = t0 + pd.Timedelta(days=DEFAULT_TTL_DAYS)

                candidates = [(ft, dp, w, px) for ft, dp, w, px in candidates if ft <= ttl_deadline]

                if candidates:

                    candidates.sort(key=lambda x: x[0])

                    entry_at, depth_pct_used, size_weight_used, entry_px = candidates[0]

                else:

                    entry_at = pd.NaT;
                    entry_px = float('nan')

            else:

                # одиночная глубина: динамическая от strength или фиксированная

                if bool(DEEP_RETEST_DYNAMIC):

                    strength_val = _to_num(row.get("strength", None))

                    depth_pct = _depth_from_strength(strength_val)

                else:

                    depth_pct = float(DEEP_RETEST_PCT)

                entry_px = entry_base * (1.0 - depth_pct) if side == "BUY" else entry_base * (1.0 + depth_pct)

                first_touch = _first_touch_after(df, float(entry_px), t0)

                ttl_deadline = t0 + pd.Timedelta(days=DEFAULT_TTL_DAYS)

                entry_at = first_touch if (pd.notna(first_touch) and first_touch <= ttl_deadline) else pd.NaT

                depth_pct_used = float(depth_pct)

            # TP/SL режим

            tp_mode = str(DEEP_TP_MODE).lower()

            _deep_tp_mode = tp_mode

            if tp_mode == "rr":

                rr = float(DEEP_RR)

                stop_eval, tp_eval = _calc_sl_tp(float(entry_px), side, float(STOP_PCT), float(rr))

            else:

                fvg_top = _to_num(row.get(FVG_TOP_COL, pd.NA))

                fvg_bot = _to_num(row.get(FVG_BOTTOM_COL, pd.NA))

                have_zone = (pd.notna(fvg_top) and pd.notna(fvg_bot))

                if have_zone:

                    if tp_mode == "zone_top":

                        tp_target = float(max(fvg_top, fvg_bot)) if side == "BUY" else float(min(fvg_top, fvg_bot))

                    else:  # zone_mid

                        tp_target = float((float(fvg_top) + float(fvg_bot)) / 2.0)

                    sl_tmp, _ = _calc_sl_tp(float(entry_px), side, float(STOP_PCT),

                                            float(TAKE_PCT) / max(float(STOP_PCT), 1e-9))

                    stop_eval = float(sl_tmp)

                    tp_eval = float(tp_target)

                else:

                    rr = float(DEEP_RR)

                    stop_eval, tp_eval = _calc_sl_tp(float(entry_px), side, float(STOP_PCT), float(rr))

            _deep_depth_used = float(depth_pct_used if depth_pct_used is not None else 0.0)
            _size_weight_used = float(size_weight_used)
        else:
            # RETEST
            entry_px = float(entry)
            first_touch = _first_touch_after(df, entry_px, t0)
            if pd.notna(first_touch) and first_touch <= (t0 + pd.Timedelta(days=DEFAULT_TTL_DAYS)):
                entry_at = first_touch
            else:
                entry_at = pd.NaT
            rr = TAKE_PCT / max(STOP_PCT, 1e-9)
            stop_eval, tp_eval = _calc_sl_tp(entry_px, side, STOP_PCT, rr)

        stop_eval, tp_eval, _ = _repair_levels(side, float(entry_px) if entry_px is not None else float('nan'),
                                               float(stop_eval) if stop_eval is not None else float('nan'),
                                               float(tp_eval) if tp_eval is not None else float('nan'))

        t_start = entry_at
        ttl_days = int(DEFAULT_TTL_DAYS if max_days is None else max_days)
        window_end = min(t0 + pd.Timedelta(days=ttl_days), as_of)
        if pd.notna(t_start):
            window = df[(df.index > t_start) & (df.index <= window_end)]
        else:
            window = pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))

        win = False
        close_time = None
        close_price = None
        exit_reason = None

        if pd.notna(entry_at) and not window.empty and not (isinstance(stop_eval, float) and math.isnan(stop_eval)) and not (isinstance(tp_eval, float) and math.isnan(tp_eval)):
            last_checked = t_start
            for ts, c in window.iterrows():
                hi, lo = float(c['high']), float(c['low'])
                if side == 'BUY':
                    hit_tp = (hi >= tp_eval)
                    hit_sl = (lo <= stop_eval)
                else:
                    hit_tp = (lo <= tp_eval)
                    hit_sl = (hi >= stop_eval)
                if hit_tp and hit_sl:
                    w, ct, cp, er = _resolve_tp_sl_order_ltf(
                        symbol=symbol, side=side, entry_at=last_checked, bar_close_time=ts,
                        stop_eval=float(stop_eval), tp_eval=float(tp_eval),
                    )
                    if er in ('tp', 'sl'):
                        win = w; close_time = ct; close_price = cp; exit_reason = er; break
                    else:
                        last_checked = ts; continue
                if hit_tp:
                    win = True;  close_time = ts;  close_price = tp_eval;  exit_reason = 'tp';  break
                if hit_sl:
                    win = False; close_time = ts;  close_price = stop_eval; exit_reason = 'sl'; break
                last_checked = ts
            if close_time is None:
                close_time = window.index[-1]
                close_price = float(window.iloc[-1]['close'])
                exit_reason = 'timeout_last_close'
        else:
            close_time = t0 + pd.Timedelta(days=ttl_days)
            close_price = float('nan')
            exit_reason = 'timeout_no_fill'

        fee_in  = float(FEE_TAKER)
        fee_out = float(FEE_TAKER)
        if pd.notna(entry_at) and pd.notna(close_time) and not (entry_px is None or (isinstance(entry_px, float) and math.isnan(entry_px))) and not (isinstance(close_price, float) and math.isnan(close_price)):
            if side == 'BUY':
                move_pct = (float(close_price) - float(entry_px)) / float(entry_px) * 100.0
            else:
                move_pct = (float(entry_px) - float(close_price)) / float(entry_px) * 100.0
            fees_slip_pct = (fee_in + fee_out) * 100.0 + (2.0 * float(SLIPPAGE_PCT) * 100.0)
            pnl_pct_price = float(move_pct) - float(fees_slip_pct)
        else:
            move_pct = float('nan')
            pnl_pct_price = float('nan')

        out = row.to_dict()
        out.update({
            'variant': mode,
            'as_of': as_of,
            'stop_eval': float(stop_eval) if stop_eval is not None else pd.NA,
            'tp_eval': float(tp_eval) if tp_eval is not None else pd.NA,
            'win': True if exit_reason == 'tp' else (False if exit_reason in ('sl', 'timeout_last_close') else False),
            'risk_pct': STOP_PCT * 100.0,
            'profit_pct': TAKE_PCT * 100.0,
            'move_pct': move_pct,
            'pnl_pct': pnl_pct_price,
            'pnl_usd': pd.NA,
            'close_time': close_time,
            'close_price': float(close_price) if close_price is not None and not (isinstance(close_price, float) and math.isnan(close_price)) else pd.NA,
            'exit_reason': exit_reason if exit_reason is not None else 'unknown',
            'is_open_mark': False,
            't_start': entry_at,
            'deep_depth_pct': float(_deep_depth_used) if _deep_depth_used is not None else pd.NA,
            'deep_tp_mode': str(_deep_tp_mode) if _deep_tp_mode is not None else pd.NA,
            'size_weight': float(_size_weight_used) if _size_weight_used is not None else 1.0,
        })
        results.append(out)

    df_res = pd.DataFrame(results)
    df_res = _enforce_one_at_a_time_per_symbol(df_res)
    if df_res.empty:
        print("⚠️ После оценки сделок нет данных.")
        return

    for c in ['close_time','imb_time','t_start']:
        if c in df_res.columns:
            df_res[c] = df_res[c].map(_to_utc_safe)
    # безопасная разница по времени в днях (tz-aware)
    t_start_utc = pd.to_datetime(df_res['t_start'], utc=True, errors='coerce')
    t_exit_utc  = pd.to_datetime(df_res['close_time'], utc=True, errors='coerce')
    df_res['exit_time'] = t_exit_utc
    df_res['exit_days'] = ((t_exit_utc - t_start_utc).dt.total_seconds() / 86400.0).round(3)

    init_cap = float(initial_capital) if initial_capital is not None else float(INITIAL_CAPITAL or 0.0)
    eq_sheet = pd.DataFrame()
    if init_cap <= 0:
        print("⚠️ INITIAL_CAPITAL <= 0 — симуляция будет пропущена.")
        df_out = df_res.copy(); df_out['skipped'] = False
    elif capital_aware:
        df_out, eq_sheet = _simulate_capital_notional(
            df_res, init_cap, position_fraction=POSITION_FRACTION,
            stop_pct=STOP_PCT, take_pct=TAKE_PCT,
        )
    else:
        df_out = df_res.copy(); df_out['skipped'] = False

    df_exec = df_out[df_out['skipped'] == False].copy()
    try:
        by_variant = (
            df_exec.groupby('variant')
                  .agg(trades=('win','size'),
                       wins=('win','sum'),
                       winrate_pct=('win', lambda s: round(100.0*float(s.sum())/max(int(s.size),1),2)),
                       pnl_pct=('pnl_pct','sum'),
                       pnl_usd=('pnl_usd','sum'))
                  .reset_index()
                  .sort_values(['pnl_usd','winrate_pct'], ascending=[False, False])
        )
    except Exception:
        by_variant = pd.DataFrame()

    by_exit_reason = _safe_group_exit_reason(df_out)

    equity_summary = pd.DataFrame()
    if not eq_sheet.empty:
        start_eq = float(eq_sheet['equity_before'].iloc[0])
        end_eq   = float(eq_sheet['equity_after'].iloc[-1])
        total_ret_pct = (end_eq / start_eq - 1.0) * 100.0 if start_eq > 0 else 0.0
        equity_summary = pd.DataFrame({
            'metric': ['start_equity','end_equity','total_return_pct','closed_trades'],
            'value':  [round(start_eq,2), round(end_eq,2), round(total_ret_pct,2), int(len(eq_sheet))]
        })

    for col in ['imb_time', 'close_time', 'exit_time', 'as_of', 't_start']:
        if col in df_out.columns:
            ser = pd.to_datetime(df_out[col], errors='coerce')
            if getattr(ser.dt, 'tz', None) is not None:
                df_out[col] = ser.dt.tz_convert(None)
            else:
                df_out[col] = ser
    if not eq_sheet.empty:
        ser = pd.to_datetime(eq_sheet['time'], errors='coerce')
        if getattr(ser.dt, 'tz', None) is not None:
            eq_sheet['time'] = ser.dt.tz_convert(None)
        else:
            eq_sheet['time'] = ser

    if 'skipped' in df_out.columns:
        for c in ('move_pct','pnl_pct','pnl_usd','alloc_usd_comp','pnl_usd_comp','equity_after'):
            if c in df_out.columns:
                df_out.loc[df_out['skipped'] == True, c] = pd.NA

    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    try:
        with pd.ExcelWriter(result_path, engine='xlsxwriter') as wr:
            df_out.to_excel(wr, sheet_name='results', index=False)
            if not by_exit_reason.empty:
                by_exit_reason.to_excel(wr, sheet_name='by_exit_reason', index=False)
            if not by_variant.empty:
                by_variant.to_excel(wr, sheet_name='by_variant', index=False)
            if not eq_sheet.empty:
                eq_sheet.to_excel(wr, sheet_name='equity_curve', index=False)
            if not equity_summary.empty:
                equity_summary.to_excel(wr, sheet_name='equity_summary', index=False)
        print(f"✅ Результаты сохранены в {result_path}")
    except ModuleNotFoundError:
        csv_fallback = os.path.splitext(result_path)[0] + ".csv"
        df_out.to_csv(csv_fallback, index=False)
        print(f"💾 Сохранил в CSV: {csv_fallback}")

def _default_reports_dir() -> str:
    return os.path.expanduser("~/Documents/отчеты")

def _derive_default_result_path(signals_path: str) -> str:
    reports_dir = _default_reports_dir()
    base = os.path.splitext(os.path.basename(signals_path))[0]
    out_name = f"{base}_eval.xlsx"
    return os.path.join(reports_dir, out_name)

if __name__ == "__main__":
    import argparse
    def _str2bool(v: str) -> bool:
        return str(v).strip().lower() in ("1", "true", "yes", "y", "t", "on")

    p = argparse.ArgumentParser(
        description="Evaluate imbalance/momentum signals with capital-aware simulation and intrabar TP/SL resolution."
    )
    p.add_argument("signals")
    p.add_argument("--out", default=None)
    p.add_argument("--lookback-days", type=int, default=360)
    p.add_argument("--ttl-days", type=int, default=None)
    p.add_argument("--include-open", type=_str2bool, default=False)
    p.add_argument("--interval", default="4h")
    p.add_argument("--compounding", type=_str2bool, default=True)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--capital-aware", type=_str2bool, default=True)
    p.add_argument("--intrabar", default=None)
    p.add_argument("--intrabar-lookback-days", type=int, default=None)
    p.add_argument("--only-filled", action="store_true")
    p.add_argument("--dedup", action="store_true")
    p.add_argument("--deep-rr", type=float, default=None)
    p.add_argument("--deep-tp-mode", type=str, default=None, choices=["rr", "zone_mid", "zone_top"])
    p.add_argument("--deep-ladder", type=str, default=None)
    args = p.parse_args()

    sig_path = os.path.expanduser(args.signals)
    if args.deep_rr is not None:
        os.environ["DEEP_RR"] = str(float(args.deep_rr))
        globals()["DEEP_RR"] = _get_cfg("DEEP_RR", required=True, cast=float)
    if args.deep_tp_mode is not None:
        os.environ["DEEP_TP_MODE"] = args.deep_tp_mode
        globals()["DEEP_TP_MODE"] = _get_cfg("DEEP_TP_MODE", required=True, cast=str).lower()
    if args.deep_ladder is not None:
        os.environ["DEEP_LADDER"] = args.deep_ladder
    if args.out:
        res_path = os.path.expanduser(args.out)
    else:
        res_path = _derive_default_result_path(sig_path)
        os.makedirs(os.path.dirname(res_path) or ".", exist_ok=True)

    if args.intrabar is not None:
        os.environ["INTRABAR_INTERVALS"] = args.intrabar
        globals()["INTRABAR_INTERVALS"] = _get_cfg("INTRABAR_INTERVALS", required=True, cast=list)
    if args.intrabar_lookback_days is not None:
        os.environ["INTRABAR_LOOKBACK_DAYS_FALLBACK"] = str(int(args.intrabar_lookback_days))
        globals()["INTRABAR_LOOKBACK_DAYS_FALLBACK"] = _get_cfg("INTRABAR_LOOKBACK_DAYS_FALLBACK", required=True, cast=int)

    ttl_days = int(args.ttl_days) if args.ttl_days is not None else None

    evaluate_signals(
        signals_path=sig_path,
        result_path=res_path,
        lookback_days=int(args.lookback_days),
        interval=str(args.interval),
        max_days=ttl_days,
        include_open=bool(args.include_open),
        compounding=bool(args.compounding),
        initial_capital=(float(args.initial_capital) if args.initial_capital is not None else None),
        capital_aware=bool(args.capital_aware),
        only_filled=bool(args.only_filled),
        dedup=bool(args.dedup),
    )