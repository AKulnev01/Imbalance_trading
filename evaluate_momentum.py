import os
import math
import json
import pandas as pd
from datetime import datetime, timezone

from evaluate_common import (
    get_cfg, INITIAL_CAPITAL, POSITION_FRACTION, DEFAULT_TTL_DAYS,
    to_utc_safe, fetch_ltf_window, load_price_cache, load_signals,
    enforce_one_at_a_time_per_symbol, simulate_capital_notional,
    safe_group_exit_reason, finalize_write, calc_sl_tp,
)

ENABLE_BUY  = bool(get_cfg("ENABLE_BUY",  cast=bool,  default=True))
ENABLE_SELL = bool(get_cfg("ENABLE_SELL", cast=bool,  default=True))
MAX_CONCURRENT = int(get_cfg("MAX_CONCURRENT_POSITIONS", cast=int, default=0))
ENFORCE_ONE_AT_A_TIME = bool(get_cfg("EVAL_ENFORCE_ONE_AT_A_TIME", cast=bool, default=True))

TAKE_PCT = float(get_cfg("MOMENTUM_TP_PCT", cast=float))
STOP_PCT = float(get_cfg("MOMENTUM_SL_PCT", cast=float))

TP_SL_MODE_RAW = str(get_cfg("MOMENTUM_TP_SL_MODE", cast=str, default="entry")).strip().lower()
FORCE_ENTRY = bool(get_cfg("MOMENTUM_FORCE_ENTRY", cast=bool, default=False))
TP_SL_MODE = "entry" if FORCE_ENTRY else TP_SL_MODE_RAW

DISABLE_MINUTE_FALLBACK = bool(get_cfg("MOMENTUM_DISABLE_MINUTE_FALLBACK", cast=bool, default=False))
MINUTE_EXIT_FOR_SINGLE = bool(get_cfg("MOMENTUM_MINUTE_EXIT_FOR_SINGLE_HIT", cast=bool, default=True))

FEE_TAKER = float(get_cfg("FEE_TAKER", cast=float, default=0.0))
ENTRY_SLIPPAGE_PCT = float(get_cfg("ENTRY_SLIPPAGE_PCT", cast=float, default=0.0))
ENTRY_LIMIT_SLIPPAGE_PCT = float(get_cfg("ENTRY_LIMIT_SLIPPAGE_PCT", cast=float, default=0.0))
EXIT_SLIPPAGE_PCT  = float(get_cfg("EXIT_SLIPPAGE_PCT",  cast=float, default=0.0))
STOP_SLIPPAGE_PCT  = float(get_cfg("STOP_SLIPPAGE_PCT",  cast=float, default=EXIT_SLIPPAGE_PCT))

ENABLE_EARLY_CHECK            = bool(get_cfg("ENABLE_EARLY_CHECK",            cast=bool,  default=False))
EARLY_CHECK_MIN_BEFORE_CLOSE  = int(get_cfg("EARLY_CHECK_MIN_BEFORE_CLOSE",  cast=int,   default=5))
EARLY_MOVE_PCT                = float(get_cfg("EARLY_MOVE_PCT",               cast=float, default=0.0))
EARLY_VOL_MULT                = float(get_cfg("EARLY_VOL_MULT",               cast=float, default=0.0))
EARLY_VOL_SMA_N               = int(get_cfg("EARLY_VOL_SMA_N",                cast=int,   default=20))
EARLY_REQUIRE_BOTH            = bool(get_cfg("EARLY_REQUIRE_BOTH",            cast=bool,  default=False))
EARLY_USE_LAST_MINUTE         = bool(get_cfg("EARLY_USE_LAST_MINUTE",         cast=bool,  default=True))

ENABLE_SCALE_OUT              = bool(get_cfg("ENABLE_SCALE_OUT",              cast=bool,  default=True))
SCALE_TP_LEVELS_RAW           = str(get_cfg("SCALE_TP_LEVELS",                cast=str,   default="0.02@0.5,0.03@0.25,0.05@0.25"))
SCALE_SL_LEVELS_RAW           = str(get_cfg("SCALE_SL_LEVELS",                cast=str,   default="0.008@0.5,0.010@0.5"))
SCALE_BREAKEVEN_AFTER_TP1     = bool(get_cfg("SCALE_BREAKEVEN_AFTER_TP1",     cast=bool,  default=True))
BREAKEVEN_OFFSET_PCT          = float(get_cfg("BREAKEVEN_OFFSET_PCT",         cast=float, default=0.0))
CAP_AT_LAST_TP                = bool(get_cfg("CAP_AT_LAST_TP",                cast=bool,  default=True))

SLOT_FREE_AT_REMAINING_PCT    = float(get_cfg("SLOT_FREE_AT_REMAINING_PCT",   cast=float, default=0.25))

LIMIT_BACKFILL_ENABLE         = bool(get_cfg("LIMIT_BACKFILL_ENABLE",         cast=bool,  default=True))
LIMIT_BACKFILL_MAX_AGE_HOURS  = int(get_cfg("LIMIT_BACKFILL_MAX_AGE_HOURS",   cast=int,   default=4))
LIMIT_BACKFILL_BAND_PCT       = float(get_cfg("LIMIT_BACKFILL_BAND_PCT",      cast=float, default=0.01))
LIMIT_BACKFILL_AT             = str(get_cfg("LIMIT_BACKFILL_AT",              cast=str,   default="entry")).strip().lower()

def _normalize_side(v: str) -> str:
    s = str(v).strip().upper()
    if s in ("BUY", "LONG"): return "BUY"
    if s in ("SELL", "SHORT"): return "SELL"
    return s

def _apply_slippage(price: float, pct: float, action: str) -> float:
    sp = float(pct or 0.0)
    if not isinstance(price, (int, float)) or math.isnan(price):
        return float('nan')
    return float(price) * (1.0 + sp) if action == "BUY" else float(price) * (1.0 - sp)

def _pnl_pct_with_fees(entry_px_adj: float, exit_px_adj: float, side: str, fee_taker_pct: float) -> float:
    if any(not isinstance(x, (int, float)) or math.isnan(x) for x in (entry_px_adj, exit_px_adj)):
        return float('nan')
    move_pct = ((exit_px_adj - entry_px_adj) / entry_px_adj * 100.0) if side == "BUY" \
               else ((entry_px_adj - exit_px_adj) / entry_px_adj * 100.0)
    fees_pct = float(fee_taker_pct or 0.0) * 2.0 * 100.0
    return float(move_pct) - float(fees_pct)

def _entry_from_4h_close(df_4h: pd.DataFrame, t0) -> tuple:
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

def _early_entry_from_ltf(symbol: str, side: str, t0: pd.Timestamp, minutes_before: int) -> tuple:
    t0 = to_utc_safe(t0)
    if pd.isna(t0) or minutes_before is None or minutes_before <= 0:
        return (pd.NaT, float("nan"), "early_disabled")
    t_start = t0 - pd.Timedelta(minutes=int(minutes_before))
    ltf = fetch_ltf_window(symbol, t_start, t0, candidates=["1m"])
    if ltf is None or ltf.empty:
        return (pd.NaT, float("nan"), "no_1m_data")
    first_row = ltf.iloc[0]; last_row  = ltf.iloc[-1]
    open0 = float(first_row.get("open", first_row.get("close", float("nan"))))
    last_close = float(last_row.get("close", float("nan")))
    last_vol = float(last_row.get("volume", last_row.get("vol", 0.0)))
    move_ok = True; move_note = "move_na"
    if EARLY_MOVE_PCT and EARLY_MOVE_PCT > 0:
        if side == "BUY":
            move = (last_close - open0) / max(open0, 1e-12)
        else:
            move = (open0 - last_close) / max(open0, 1e-12)
        move_ok = (move >= EARLY_MOVE_PCT); move_note = f"move={move:.6f} req>={EARLY_MOVE_PCT:.6f}"
    vol_ok = True; vol_note = "vol_na"
    if EARLY_VOL_MULT and EARLY_VOL_MULT > 0:
        v = ltf.get("volume", ltf.get("vol"))
        if v is None:
            vol_ok = False; vol_note = "no_vol_col"
        else:
            sma = v.rolling(EARLY_VOL_SMA_N, min_periods=1).mean().iloc[-1]
            thresh = float(sma) * float(EARLY_VOL_MULT)
            vol_ok = (last_vol >= thresh)
            vol_note = f"vol_last={last_vol:.6f} thresh={thresh:.6f}"
    cond = (move_ok and vol_ok) if EARLY_REQUIRE_BOTH else (move_ok or vol_ok)
    if not cond:
        return (pd.NaT, float("nan"), f"no_early_cond__{move_note}__{vol_note}")
    entry_px_ref = last_close if EARLY_USE_LAST_MINUTE else float(first_row.get("close", open0))
    which = "last1m" if EARLY_USE_LAST_MINUTE else "first1m"
    note = f"early_ok[{which}]__{move_note}__{vol_note}"
    return (t0, float(entry_px_ref), note)

def _resolve_exit_minute(symbol: str, side: str, entry_at: pd.Timestamp, stop_eval: float, tp_eval: float, t_end: pd.Timestamp):
    t0 = to_utc_safe(entry_at); t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end):
        return (None, pd.NaT, float('nan'), "bad_ts")
    ltf = fetch_ltf_window(symbol, t0, t_end, candidates=["1m"])
    if ltf is None or ltf.empty:
        return (None, pd.NaT, float('nan'), "uncertain_no_ltf")
    for ts, c in ltf.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if side == "BUY":
            hit_tp = (hi >= float(tp_eval)); hit_sl = (lo <= float(stop_eval))
        else:
            hit_tp = (lo <= float(tp_eval)); hit_sl = (hi >= float(stop_eval))
        if hit_tp and hit_sl:
            return (False, ts, float(stop_eval), "sl")
        if hit_tp:
            return (True, ts, float(tp_eval), "tp")
        if hit_sl:
            return (False, ts, float(stop_eval), "sl")
    last_ts = ltf.index[-1]; last_close = float(ltf.iloc[-1]["close"])
    return (False, last_ts, last_close, "timeout_last_close")

def _resolve_exit_hybrid(symbol: str, side: str, entry_at: pd.Timestamp, stop_eval: float, tp_eval: float, t_end: pd.Timestamp, df_4h: pd.DataFrame):
    t0 = to_utc_safe(entry_at); t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end) or df_4h is None or df_4h.empty:
        return (None, pd.NaT, float("nan"), "bad_ts", "4h")
    bars = df_4h[(df_4h.index >= t0) & (df_4h.index < t_end)]
    for ts, c in bars.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if side == "BUY":
            hit_tp = (hi >= float(tp_eval)); hit_sl = (lo <= float(stop_eval))
        else:
            hit_tp = (lo <= float(tp_eval)); hit_sl = (hi >= float(stop_eval))
        bar_start = ts; bar_end = ts + pd.Timedelta(hours=4)
        if hit_tp and hit_sl:
            if DISABLE_MINUTE_FALLBACK:
                return (None, ts + pd.Timedelta(hours=4), float('nan'), "uncertain_both_hit_4h", "4h")
            return _resolve_exit_minute(symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end)) + ("1m",)
        if hit_tp:
            if MINUTE_EXIT_FOR_SINGLE:
                win_m, close_t, trig_px, reason = _resolve_exit_minute(symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end))
                if reason in ("tp", "sl", "timeout_last_close"):
                    return (win_m, close_t, trig_px, reason, "1m")
            return (True, ts + pd.Timedelta(hours=4), float(tp_eval), "tp", "4h")
        if hit_sl:
            if MINUTE_EXIT_FOR_SINGLE:
                win_m, close_t, trig_px, reason = _resolve_exit_minute(symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end))
                if reason in ("tp", "sl", "timeout_last_close"):
                    return (win_m, close_t, trig_px, reason, "1m")
            return (False, ts + pd.Timedelta(hours=4), float(stop_eval), "sl", "4h")
    if not bars.empty:
        last_ts = bars.index[-1] + pd.Timedelta(hours=4); last_close = float(bars.iloc[-1]["close"])
    else:
        last_ts = t0; last_close = float("nan")
    return (False, last_ts, last_close, "timeout_last_close", "4h")

def _parse_levels(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return []
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "@" in chunk:
            chunk = chunk.replace("”","@").replace("“","@")
        parts = chunk.split("@")
        if len(parts)!=2:
            continue
        try:
            lvl = float(parts[0]); frac = float(parts[1])
            out.append((lvl, frac))
        except:
            continue
    return out

def _simulate_scaleout_4h(symbol, side, entry_at, entry_px_adj, df_4h, as_of, max_days, tp_prices, sl_prices, sl_eval, tp_eval):
    ttl_days = int(DEFAULT_TTL_DAYS or 3) if max_days is None else int(max_days)
    t0 = to_utc_safe(entry_at)
    t_end = min(t0 + pd.Timedelta(days=ttl_days), as_of)
    if df_4h is None or df_4h.empty:
        return {"fills": [], "closed": False, "close_time": pd.NaT, "close_price": float('nan'),
                "exit_reason": "uncertain_no_4h", "slot_free_time": pd.NaT,
                "pnl_pct": float('nan'), "win": pd.NA, "lt_resolution": "4h"}
    bars = df_4h[(df_4h.index >= t0) & (df_4h.index < t_end)].copy()
    if bars.empty:
        return {"fills": [], "closed": False, "close_time": pd.NaT, "close_price": float('nan'),
                "exit_reason": "uncertain_no_4h", "slot_free_time": pd.NaT,
                "pnl_pct": float('nan'), "win": pd.NA, "lt_resolution": "4h"}
    remaining = 1.0
    fills = []
    tp_hit_flags = [False]*len(tp_prices)
    sl_hit_flags = [False]*len(sl_prices)
    slot_free_time = pd.NaT
    for ts, c in bars.iterrows():
        hi, lo, close = float(c["high"]), float(c["low"]), float(c["close"])
        for i,(px, frac, tag) in enumerate(tp_prices):
            if tp_hit_flags[i]: continue
            hit = (hi >= px) if side=="BUY" else (lo <= px)
            if hit and remaining>0:
                take_frac = min(frac, remaining)
                exit_action = "SELL" if side=="BUY" else "BUY"
                px_adj = _apply_slippage(px, EXIT_SLIPPAGE_PCT, exit_action)
                fills.append({"ts": ts+pd.Timedelta(hours=4), "price": px_adj, "frac": take_frac, "reason": tag})
                remaining = max(0.0, remaining - take_frac)
                tp_hit_flags[i] = True
                if pd.isna(slot_free_time) and remaining <= SLOT_FREE_AT_REMAINING_PCT:
                    slot_free_time = ts+pd.Timedelta(hours=4)
        for i,(px, frac, tag) in enumerate(sorted(sl_prices, key=lambda x: (x[0] if side=="BUY" else -x[0]))):
            idx = None
            for j,(opx, ofrac, otag) in enumerate(sl_prices):
                if opx==px and ofrac==frac and otag==tag: idx=j; break
            if sl_hit_flags[idx]: continue
            hit = (lo <= px) if side=="BUY" else (hi >= px)
            if hit and remaining>0:
                take_frac = min(frac, remaining)
                exit_action = "BUY" if side=="SELL" else "SELL"
                px_adj = _apply_slippage(px, STOP_SLIPPAGE_PCT, exit_action)
                fills.append({"ts": ts+pd.Timedelta(hours=4), "price": px_adj, "frac": take_frac, "reason": tag})
                remaining = max(0.0, remaining - take_frac)
                sl_hit_flags[idx] = True
                if pd.isna(slot_free_time) and remaining <= SLOT_FREE_AT_REMAINING_PCT:
                    slot_free_time = ts+pd.Timedelta(hours=4)
        hit_tp_final = (hi >= float(tp_eval)) if side=="BUY" else (lo <= float(tp_eval))
        hit_sl_final = (lo <= float(sl_eval)) if side=="BUY" else (hi >= float(sl_eval))
        if hit_tp_final and remaining>0:
            exit_action = "SELL" if side=="BUY" else "BUY"
            px_adj = _apply_slippage(float(tp_eval), EXIT_SLIPPAGE_PCT, exit_action)
            fills.append({"ts": ts+pd.Timedelta(hours=4), "price": px_adj, "frac": remaining, "reason": "tp"})
            remaining = 0.0
            if pd.isna(slot_free_time): slot_free_time = ts+pd.Timedelta(hours=4)
            break
        if hit_sl_final and remaining>0:
            exit_action = "BUY" if side=="SELL" else "SELL"
            px_adj = _apply_slippage(float(sl_eval), STOP_SLIPPAGE_PCT, exit_action)
            fills.append({"ts": ts+pd.Timedelta(hours=4), "price": px_adj, "frac": remaining, "reason": "sl"})
            remaining = 0.0
            if pd.isna(slot_free_time): slot_free_time = ts+pd.Timedelta(hours=4)
            break
    if remaining > 0.0:
        last_ts = bars.index[-1] + pd.Timedelta(hours=4); last_close = float(bars.iloc[-1]["close"])
        px_timeout = last_close
        if CAP_AT_LAST_TP and len(tp_prices) > 0:
            last_tp_px = tp_prices[-1][0]
            if side == "BUY":
                px_timeout = min(px_timeout, last_tp_px)
            else:
                px_timeout = max(px_timeout, last_tp_px)
        exit_action = "SELL" if side=="BUY" else "BUY"
        px_adj = _apply_slippage(px_timeout, EXIT_SLIPPAGE_PCT, exit_action)
        fills.append({"ts": last_ts, "price": px_adj, "frac": remaining, "reason": "scaleout_timeout"})
        remaining = 0.0
        if pd.isna(slot_free_time): slot_free_time = last_ts
        close_reason = "scaleout_timeout"; close_time = last_ts; close_price = px_adj
    else:
        last_fill = fills[-1]; close_time = last_fill["ts"]; close_price = last_fill["price"]; close_reason = last_fill["reason"]
    pnl_pct = 0.0
    for f in fills:
        leg = _pnl_pct_with_fees(entry_px_adj, float(f["price"]), side, FEE_TAKER) * float(f["frac"])
        pnl_pct += leg
    win = (close_reason.startswith("tp") or any(fr["reason"].startswith("tp") for fr in fills))
    return {
        "fills": fills,
        "closed": True,
        "close_time": close_time,
        "close_price": close_price,
        "exit_reason": close_reason,
        "slot_free_time": slot_free_time,
        "pnl_pct": pnl_pct,
        "win": win,
        "lt_resolution": "4h"
    }

def _simulate_scaleout(symbol, side, entry_at, entry_px_adj, df_4h, as_of, max_days):
    rr = float(TAKE_PCT) / max(float(STOP_PCT), 1e-9)
    mode = TP_SL_MODE
    if mode == "anchored":
        anchor = float(entry_px_adj)
        k = float(STOP_PCT)
        width = anchor * k * (1.0 + rr)
        if side == "BUY":
            sl_eval = anchor - width / 2.0
            tp_eval = anchor + width / 2.0
        else:
            sl_eval = anchor + width / 2.0
            tp_eval = anchor - width / 2.0
    else:
        sl_eval, tp_eval = calc_sl_tp(float(entry_px_adj), side, float(STOP_PCT), rr)

    ttl_days = int(DEFAULT_TTL_DAYS or 3) if max_days is None else int(max_days)
    t_end = min(to_utc_safe(entry_at) + pd.Timedelta(days=ttl_days), as_of)

    tp_levels = _parse_levels(SCALE_TP_LEVELS_RAW) if ENABLE_SCALE_OUT else []
    sl_levels = _parse_levels(SCALE_SL_LEVELS_RAW) if ENABLE_SCALE_OUT else []

    tp_prices = []
    for pct, frac in tp_levels:
        if side == "BUY":
            tp_prices.append((float(entry_px_adj)*(1.0+pct), frac, f"tp@{pct}"))
        else:
            tp_prices.append((float(entry_px_adj)*(1.0-pct), frac, f"tp@{pct}"))
    sl_prices = []
    for pct, frac in sl_levels:
        if side == "BUY":
            sl_prices.append((float(entry_px_adj)*(1.0-pct), frac, f"sl@{pct}"))
        else:
            sl_prices.append((float(entry_px_adj)*(1.0+pct), frac, f"sl@{pct}"))

    ltf = fetch_ltf_window(symbol, to_utc_safe(entry_at), t_end, candidates=["1m"])
    if ltf is None or ltf.empty:
        return _simulate_scaleout_4h(symbol, side, entry_at, entry_px_adj, df_4h, as_of, max_days, tp_prices, sl_prices, sl_eval, tp_eval)

    remaining = 1.0
    fills = []
    tp_hit_flags = [False]*len(tp_prices)
    sl_hit_flags = [False]*len(sl_prices)
    breakeven_active = False
    slot_free_time = pd.NaT

    for ts, c in ltf.iterrows():
        hi, lo, close = float(c["high"]), float(c["low"]), float(c["close"])

        if ENABLE_SCALE_OUT and SCALE_BREAKEVEN_AFTER_TP1 and (not breakeven_active):
            if any(tp_hit_flags):
                breakeven_active = True
                if side == "BUY":
                    sl_eval = float(entry_px_adj) * (1.0 + BREAKEVEN_OFFSET_PCT)
                else:
                    sl_eval = float(entry_px_adj) * (1.0 - BREAKEVEN_OFFSET_PCT)

        for i,(px, frac, tag) in enumerate(tp_prices):
            if tp_hit_flags[i]: continue
            hit = (hi >= px) if side=="BUY" else (lo <= px)
            if hit and remaining>0:
                take_frac = min(frac, remaining)
                exit_action = "SELL" if side=="BUY" else "BUY"
                px_adj = _apply_slippage(px, EXIT_SLIPPAGE_PCT, exit_action)
                fills.append({"ts": ts, "price": px_adj, "frac": take_frac, "reason": tag})
                remaining = max(0.0, remaining - take_frac)
                tp_hit_flags[i] = True
                if pd.isna(slot_free_time) and remaining <= SLOT_FREE_AT_REMAINING_PCT:
                    slot_free_time = ts

        for i,(px, frac, tag) in enumerate(sorted(sl_prices, key=lambda x: (x[0] if side=="BUY" else -x[0]))):
            idx = None
            for j,(opx, ofrac, otag) in enumerate(sl_prices):
                if opx==px and ofrac==frac and otag==tag: idx=j; break
            if sl_hit_flags[idx]: continue
            hit = (lo <= px) if side=="BUY" else (hi >= px)
            if hit and remaining>0:
                take_frac = min(frac, remaining)
                exit_action = "BUY" if side=="SELL" else "SELL"
                px_adj = _apply_slippage(px, STOP_SLIPPAGE_PCT, exit_action)
                fills.append({"ts": ts, "price": px_adj, "frac": take_frac, "reason": tag})
                remaining = max(0.0, remaining - take_frac)
                sl_hit_flags[idx] = True
                if pd.isna(slot_free_time) and remaining <= SLOT_FREE_AT_REMAINING_PCT:
                    slot_free_time = ts

        hit_tp_final = (hi >= float(tp_eval)) if side=="BUY" else (lo <= float(tp_eval))
        hit_sl_final = (lo <= float(sl_eval)) if side=="BUY" else (hi >= float(sl_eval))
        if hit_tp_final and remaining>0:
            exit_action = "SELL" if side=="BUY" else "BUY"
            px_adj = _apply_slippage(float(tp_eval), EXIT_SLIPPAGE_PCT, exit_action)
            fills.append({"ts": ts, "price": px_adj, "frac": remaining, "reason": "tp"})
            remaining = 0.0
            if pd.isna(slot_free_time): slot_free_time = ts
            break
        if hit_sl_final and remaining>0:
            exit_action = "BUY" if side=="SELL" else "SELL"
            px_adj = _apply_slippage(float(sl_eval), STOP_SLIPPAGE_PCT, exit_action)
            fills.append({"ts": ts, "price": px_adj, "frac": remaining, "reason": "sl"})
            remaining = 0.0
            if pd.isna(slot_free_time): slot_free_time = ts
            break

    if remaining > 0.0:
        last_ts = ltf.index[-1]; last_close = float(ltf.iloc[-1]["close"])
        px_timeout = last_close
        if CAP_AT_LAST_TP and len(tp_prices) > 0:
            last_tp_px = tp_prices[-1][0]
            if side == "BUY":
                px_timeout = min(px_timeout, last_tp_px)
            else:
                px_timeout = max(px_timeout, last_tp_px)
        exit_action = "SELL" if side=="BUY" else "BUY"
        px_adj = _apply_slippage(px_timeout, EXIT_SLIPPAGE_PCT, exit_action)
        fills.append({"ts": last_ts, "price": px_adj, "frac": remaining, "reason": "scaleout_timeout"})
        remaining = 0.0
        if pd.isna(slot_free_time): slot_free_time = last_ts
        close_reason = "scaleout_timeout"; close_time = last_ts; close_price = px_adj
    else:
        last_fill = fills[-1]; close_time = last_fill["ts"]; close_price = last_fill["price"]; close_reason = last_fill["reason"]

    pnl_pct = 0.0
    for f in fills:
        leg = _pnl_pct_with_fees(entry_px_adj, float(f["price"]), side, FEE_TAKER) * float(f["frac"])
        pnl_pct += leg
    win = (close_reason.startswith("tp") or any(fr["reason"].startswith("tp") for fr in fills))
    return {
        "fills": fills,
        "closed": True,
        "close_time": close_time,
        "close_price": close_price,
        "exit_reason": close_reason,
        "slot_free_time": slot_free_time,
        "pnl_pct": pnl_pct,
        "win": win,
        "lt_resolution": "1m"
    }

def _evaluate_one_variant(row: pd.Series, *, variant_name: str, symbol: str, side: str, entry_at: pd.Timestamp, entry_px_ref: float, df4h: pd.DataFrame, as_of: pd.Timestamp, max_days: int):
    if pd.isna(entry_at) or not isinstance(entry_px_ref, (int, float)) or math.isnan(entry_px_ref):
        out = row.to_dict(); out.update({
            "variant": variant_name, "as_of": as_of,
            "stop_eval": pd.NA, "tp_eval": pd.NA, "win": pd.NA,
            "pnl_pct": pd.NA, "pnl_usd": pd.NA, "move_pct": pd.NA,
            "close_time": pd.NaT, "close_price": pd.NA,
            "exit_reason": "no_entry_ref", "is_open_mark": False,
            "t_start": pd.NaT, "size_weight": float(row.get("size_weight", 1.0)) if pd.notna(row.get("size_weight", 1.0)) else 1.0,
            "skipped": True, "lt_resolution": "none",
            "entry_note": "no_entry_ref",
            "strength": float(row.get("strength", pd.NA)) if pd.notna(row.get("strength", pd.NA)) else pd.NA,
            "type": side,
            "scaleout": ENABLE_SCALE_OUT,
            "slot_free_time": pd.NaT,
            "fills_json": "[]",
        }); return out

    entry_action = "BUY" if side == "BUY" else "SELL"
    entry_px_adj = _apply_slippage(entry_px_ref, ENTRY_SLIPPAGE_PCT, entry_action)

    rr = float(TAKE_PCT) / max(float(STOP_PCT), 1e-9)
    mode = TP_SL_MODE
    if mode == "anchored":
        anchor = float(entry_px_ref)
        k = float(STOP_PCT)
        width = anchor * k * (1.0 + rr)
        if side == "BUY":
            sl_eval = anchor - width / 2.0
            tp_eval = anchor + width / 2.0
        else:
            sl_eval = anchor + width / 2.0
            tp_eval = anchor - width / 2.0
    else:
        sl_eval, tp_eval = calc_sl_tp(float(entry_px_adj), side, float(STOP_PCT), rr)

    ttl_days = int(DEFAULT_TTL_DAYS if max_days is None else max_days)
    window_end = min(to_utc_safe(row.get("imb_time")) + pd.Timedelta(days=ttl_days), as_of)

    if ENABLE_SCALE_OUT:
        sim = _simulate_scaleout(symbol, side, entry_at, float(entry_px_adj), df4h, as_of, max_days)
        if not sim["closed"]:
            out = row.to_dict(); out.update({
                "variant": variant_name, "as_of": as_of,
                "stop_eval": float(sl_eval), "tp_eval": float(tp_eval),
                "win": pd.NA, "pnl_pct": pd.NA, "pnl_usd": pd.NA, "move_pct": pd.NA,
                "close_time": sim["close_time"], "close_price": pd.NA,
                "exit_reason": "uncertain", "is_open_mark": False,
                "t_start": entry_at, "size_weight": float(row.get("size_weight", 1.0)) if pd.notna(row.get("size_weight", 1.0)) else 1.0,
                "skipped": True,
                "lt_resolution": sim.get("lt_resolution","1m"),
                "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),
                "exit_trigger": pd.NA,
                "exit_reason_raw": "uncertain",
                "strength": float(row.get("strength", pd.NA)) if pd.notna(row.get("strength", pd.NA)) else pd.NA,
                "tpsl_mode": mode, "type": side,
                "fee_taker_pct": float(FEE_TAKER),
                "entry_slip_pct": float(ENTRY_SLIPPAGE_PCT),
                "exit_slip_pct": float(EXIT_SLIPPAGE_PCT),
                "scaleout": True,
                "slot_free_time": sim.get("slot_free_time", pd.NaT),
                "fills_json": json.dumps(sim.get("fills", []), default=str),
            }); return out

        out = row.to_dict()
        out.update({
            "variant": variant_name, "as_of": as_of,
            "stop_eval": float(sl_eval), "tp_eval": float(tp_eval),
            "win": bool(sim["win"]),
            "pnl_pct": float(sim["pnl_pct"]), "pnl_usd": pd.NA, "move_pct": float(sim["pnl_pct"]),
            "close_time": sim["close_time"], "close_price": float(sim["close_price"]),
            "exit_reason": sim["exit_reason"],
            "is_open_mark": False,
            "t_start": entry_at, "size_weight": float(row.get("size_weight", 1.0)) if pd.notna(row.get("size_weight", 1.0)) else 1.0,
            "skipped": False,
            "lt_resolution": sim.get("lt_resolution","1m"),
            "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),
            "exit_px_adj": float(sim["close_price"]),
            "exit_trigger": float('nan'),
            "exit_reason_raw": sim["exit_reason"],
            "strength": float(row.get("strength", pd.NA)) if pd.notna(row.get("strength", pd.NA)) else pd.NA,
            "tpsl_mode": mode, "type": side,
            "fee_taker_pct": float(FEE_TAKER),
            "entry_slip_pct": float(ENTRY_SLIPPAGE_PCT),
            "exit_slip_pct": float(EXIT_SLIPPAGE_PCT),
            "scaleout": True,
            "slot_free_time": sim.get("slot_free_time", pd.NaT),
            "fills_json": json.dumps(sim.get("fills", []), default=str),
        })
        return out

    win, close_time, trigger_price, exit_reason, tf_used = _resolve_exit_hybrid(
        symbol=symbol, side=side, entry_at=entry_at,
        stop_eval=float(sl_eval), tp_eval=float(tp_eval),
        t_end=window_end, df_4h=df4h
    )
    if win is None or (isinstance(exit_reason, str) and exit_reason.startswith("uncertain")):
        out = row.to_dict(); out.update({
            "variant": variant_name, "as_of": as_of,
            "stop_eval": float(sl_eval), "tp_eval": float(tp_eval),
            "win": pd.NA, "pnl_pct": pd.NA, "pnl_usd": pd.NA, "move_pct": pd.NA,
            "close_time": close_time, "close_price": pd.NA,
            "exit_reason": exit_reason, "is_open_mark": False,
            "t_start": entry_at, "size_weight": float(row.get("size_weight", 1.0)) if pd.notna(row.get("size_weight", 1.0)) else 1.0,
            "skipped": True,
            "lt_resolution": tf_used or "none",
            "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),
            "exit_trigger": pd.NA,
            "exit_reason_raw": exit_reason,
            "strength": float(row.get("strength", pd.NA)) if pd.notna(row.get("strength", pd.NA)) else pd.NA,
            "tpsl_mode": mode, "type": side,
            "fee_taker_pct": float(FEE_TAKER),
            "entry_slip_pct": float(ENTRY_SLIPPAGE_PCT),
            "exit_slip_pct": float(EXIT_SLIPPAGE_PCT),
            "scaleout": False,
            "slot_free_time": close_time,
            "fills_json": "[]",
        }); return out

    exit_action = "SELL" if side == "BUY" else "BUY"
    exit_trigger = float(trigger_price) if isinstance(trigger_price, (int, float)) else float('nan')
    exit_slip = STOP_SLIPPAGE_PCT if exit_reason == "sl" else EXIT_SLIPPAGE_PCT
    exit_px_adj  = _apply_slippage(exit_trigger, exit_slip, exit_action)
    pnl_pct_net  = _pnl_pct_with_fees(entry_px_adj, exit_px_adj, side, FEE_TAKER)

    out = row.to_dict()
    out.update({
        "variant": variant_name, "as_of": as_of,
        "stop_eval": float(sl_eval), "tp_eval": float(tp_eval),
        "win": True if exit_reason == "tp" else False,
        "pnl_pct": pnl_pct_net, "pnl_usd": pd.NA, "move_pct": pnl_pct_net,
        "close_time": close_time, "close_price": exit_px_adj,
        "exit_reason": exit_reason,
        "is_open_mark": False,
        "t_start": entry_at, "size_weight": float(row.get("size_weight", 1.0)) if pd.notna(row.get("size_weight", 1.0)) else 1.0,
        "skipped": False,
        "lt_resolution": tf_used or "4h",
        "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),
        "exit_px_adj": float(exit_px_adj),
        "exit_trigger": float(exit_trigger),
        "exit_reason_raw": exit_reason,
        "strength": float(row.get("strength", pd.NA)) if pd.notna(row.get("strength", pd.NA)) else pd.NA,
        "tpsl_mode": mode,
        "type": side,
        "fee_taker_pct": float(FEE_TAKER),
        "entry_slip_pct": float(ENTRY_SLIPPAGE_PCT),
        "exit_slip_pct": float(exit_slip),
        "scaleout": False,
        "slot_free_time": close_time,
        "fills_json": "[]",
    })
    return out

def _slot_free_concurrency_with_backfill_markers(df: pd.DataFrame, k: int):
    if k is None or k <= 0 or df is None or df.empty:
        df["_backfill_start"] = pd.NaT
        return df.copy(), {}
    work = df.copy()
    work["_t0"] = pd.to_datetime(work.get("t_start"), utc=True, errors="coerce")
    sft = pd.to_datetime(work.get("slot_free_time"), utc=True, errors="coerce")
    ct  = pd.to_datetime(work.get("close_time"),      utc=True, errors="coerce")
    work["_t1"] = sft.fillna(ct)
    strength = pd.to_numeric(work.get("strength"), errors="coerce")
    work["_strength"] = strength.fillna(-1e18)

    kept_idx = []
    active_ends = []
    backfill = {}
    for t0, grp in work.sort_values("_t0").groupby("_t0", sort=True):
        active_ends = [t for t in active_ends if (pd.isna(t) or t > t0)]
        slots = k - len(active_ends)
        if slots <= 0:
            next_free = min(active_ends) if len(active_ends)>0 else pd.NaT
            for idx in grp.index:
                backfill[idx] = next_free
            continue
        sel = grp.sort_values("_strength", ascending=False).head(slots)
        kept_idx.extend(sel.index.tolist())
        active_ends.extend(sel["_t1"].tolist())
        rejected = grp.drop(sel.index, errors="ignore")
        next_free = min(active_ends) if len(active_ends)>0 else pd.NaT
        for idx in rejected.index:
            backfill[idx] = next_free

    work["_backfill_start"] = pd.NaT
    for idx, ts in backfill.items():
        work.loc[idx, "_backfill_start"] = ts

    kept = work.loc[sorted(set(kept_idx))].copy()
    kept.drop(columns=[c for c in ["_t0", "_t1", "_strength"] if c in kept.columns], inplace=True)
    return kept.sort_values(["t_start", "symbol"]).reset_index(drop=True), {idx: ts for idx, ts in backfill.items()}

def _simulate_limit_backfill(row_out_dict, row_sig: pd.Series, df4h: pd.DataFrame, as_of: pd.Timestamp):
    if not LIMIT_BACKFILL_ENABLE:
        return None
    symbol = row_sig["symbol"]
    side   = _normalize_side(row_sig.get("type"))
    if side not in ("BUY","SELL"):
        return None
    t0     = to_utc_safe(row_sig.get("imb_time"))
    start  = row_out_dict.get("_backfill_start", pd.NaT)
    if pd.isna(start):
        return None
    deadline = min(t0 + pd.Timedelta(hours=int(LIMIT_BACKFILL_MAX_AGE_HOURS)), as_of)

    if LIMIT_BACKFILL_AT == "entry":
        entry_at_ref, limit_px_ref = _entry_from_4h_close(df4h, t0)
    else:
        entry_at_ref, limit_px_ref = _entry_from_4h_close(df4h, t0)
    if pd.isna(limit_px_ref) or not isinstance(limit_px_ref, (int,float)):
        return None

    band = float(LIMIT_BACKFILL_BAND_PCT or 0.0)
    band_hi = limit_px_ref * (1.0 + band)
    band_lo = limit_px_ref * (1.0 - band)

    ltf = fetch_ltf_window(symbol, to_utc_safe(start), to_utc_safe(deadline), candidates=["1m"])
    if ltf is None or ltf.empty:
        return {"status":"nofill_noltf"}

    filled = False
    fill_ts = pd.NaT
    for ts, c in ltf.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if side == "BUY":
            if lo <= limit_px_ref:
                filled = True; fill_ts = ts; break
            if hi >= band_hi or lo <= band_lo:
                return {"status":"cancel_band"}
        else:
            if hi >= limit_px_ref:
                filled = True; fill_ts = ts; break
            if lo <= band_lo or hi >= band_hi:
                return {"status":"cancel_band"}
    if not filled:
        return {"status":"cancel_timeout"}

    entry_action = "BUY" if side=="BUY" else "SELL"
    entry_px_adj = _apply_slippage(limit_px_ref, ENTRY_LIMIT_SLIPPAGE_PCT, entry_action)

    sim = _simulate_scaleout(symbol, side, fill_ts, float(entry_px_adj), df4h, as_of, max_days=None)
    if not sim["closed"]:
        return {"status":"uncertain_after_fill"}
    out = row_sig.to_dict()
    out.update({
        "variant": f"MOMENTUM_LMT_BACKFILL",
        "as_of": as_of,
        "stop_eval": pd.NA,
        "tp_eval": pd.NA,
        "win": bool(sim["win"]),
        "pnl_pct": float(sim["pnl_pct"]), "pnl_usd": pd.NA, "move_pct": float(sim["pnl_pct"]),
        "close_time": sim["close_time"], "close_price": float(sim["close_price"]),
        "exit_reason": f"backfill_{sim['exit_reason']}",
        "is_open_mark": False,
        "t_start": fill_ts,
        "size_weight": float(row_sig.get("size_weight", 1.0)) if pd.notna(row_sig.get("size_weight", 1.0)) else 1.0,
        "skipped": False,
        "lt_resolution": sim.get("lt_resolution","1m"),
        "entry_px_ref": float(limit_px_ref), "entry_px_adj": float(entry_px_adj),
        "exit_px_adj": float(sim["close_price"]),
        "exit_trigger": float('nan'),
        "exit_reason_raw": f"backfill_{sim['exit_reason']}",
        "strength": float(row_sig.get("strength", pd.NA)) if pd.notna(row_sig.get("strength", pd.NA)) else pd.NA,
        "tpsl_mode": TP_SL_MODE, "type": side,
        "fee_taker_pct": float(FEE_TAKER),
        "entry_slip_pct": float(ENTRY_LIMIT_SLIPPAGE_PCT),
        "exit_slip_pct": float(EXIT_SLIPPAGE_PCT),
        "scaleout": True,
        "slot_free_time": sim.get("slot_free_time", pd.NaT),
        "fills_json": json.dumps(sim.get("fills", []), default=str),
        "_is_backfill": True,
    })
    return out

def evaluate_momentum(signals_path: str, result_path: str, lookback_days: int = 360, interval: str = "4h", max_days: int = None, only_filled: bool = False, dedup: bool = False, initial_capital: float = None, capital_aware: bool = True):
    try:
        as_of = datetime.fromtimestamp(os.path.getmtime(signals_path), tz=timezone.utc)
    except Exception:
        as_of = datetime.now(tz=timezone.utc)

    df_sig = load_signals(signals_path, only_filled=only_filled, dedup=dedup, require_entry=False)
    if df_sig.empty:
        print("⚠️ Сигналов нет после фильтров."); return

    if "type" in df_sig.columns:
        t = df_sig["type"].astype(str).str.upper()
        mask = pd.Series([True] * len(df_sig), index=df_sig.index)
        if not ENABLE_BUY:
            mask &= (t != "BUY")
        if not ENABLE_SELL:
            mask &= (t != "SELL")
        df_sig = df_sig[mask]
    if df_sig.empty:
        print("⚠️ После фильтров ENABLE_BUY/SELL сигналов нет."); return

    df_sig = _dedup_strongest_per_symbol_time(df_sig)
    if df_sig.empty:
        print("⚠️ После дедупа по strongest сигналов нет."); return

    symbols = df_sig["symbol"].dropna().unique().tolist()
    price_cache = load_price_cache(symbols, interval=interval, lookback_days=lookback_days)

    results = []
    for _, row in df_sig.iterrows():
        symbol = row["symbol"]
        side   = _normalize_side(row.get("type"))
        t0     = to_utc_safe(row.get("imb_time"))
        df4h   = price_cache.get(symbol)
        if df4h is None or df4h.empty or side not in ("BUY", "SELL") or pd.isna(t0):
            continue

        entry_at_std, entry_px_ref_std = _entry_from_4h_close(df4h, t0)
        out_std = _evaluate_one_variant(
            row, variant_name="MOMENTUM", symbol=symbol, side=side,
            entry_at=entry_at_std, entry_px_ref=entry_px_ref_std,
            df4h=df4h, as_of=as_of, max_days=max_days
        )
        results.append(out_std)

        if ENABLE_EARLY_CHECK and EARLY_CHECK_MIN_BEFORE_CLOSE and EARLY_CHECK_MIN_BEFORE_CLOSE > 0:
            entry_at_e, entry_px_ref_e, note = _early_entry_from_ltf(symbol, side, t0, EARLY_CHECK_MIN_BEFORE_CLOSE)
            if pd.notna(entry_at_e) and isinstance(entry_px_ref_e, (int, float)) and not math.isnan(entry_px_ref_e):
                variant_name = f"MOMENTUM_EARLY{int(EARLY_CHECK_MIN_BEFORE_CLOSE)}m"
                out_e = _evaluate_one_variant(
                    row, variant_name=variant_name, symbol=symbol, side=side,
                    entry_at=entry_at_e, entry_px_ref=entry_px_ref_e,
                    df4h=df4h, as_of=as_of, max_days=max_days
                )
                note_old = str(out_e.get("entry_note") or "")
                out_e["entry_note"] = (note_old + ("; " if note_old else "") + str(note)).strip()
                results.append(out_e)
            else:
                results.append({
                    **row.to_dict(),
                    "variant": f"MOMENTUM_EARLY{int(EARLY_CHECK_MIN_BEFORE_CLOSE)}m",
                    "as_of": as_of,
                    "stop_eval": pd.NA, "tp_eval": pd.NA,
                    "win": pd.NA, "pnl_pct": pd.NA, "pnl_usd": pd.NA, "move_pct": pd.NA,
                    "close_time": pd.NaT, "close_price": pd.NA,
                    "exit_reason": "early_not_triggered",
                    "is_open_mark": False,
                    "t_start": pd.NaT, "size_weight": float(row.get("size_weight", 1.0)) if pd.notna(row.get("size_weight", 1.0)) else 1.0,
                    "skipped": True, "lt_resolution": "none",
                    "entry_note": str(note),
                    "strength": float(row.get("strength", pd.NA)) if pd.notna(row.get("strength", pd.NA)) else pd.NA,
                    "type": side,
                    "scaleout": ENABLE_SCALE_OUT,
                    "slot_free_time": pd.NaT,
                    "fills_json": "[]",
                })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("⚠️ Ничего не оценили."); return

    for c in ["t_start", "close_time", "imb_time", "slot_free_time", "exit_time"]:
        if c in df_res.columns:
            df_res[c] = df_res[c].map(to_utc_safe)
    df_res["exit_time"] = df_res["close_time"]
    t_start_utc = pd.to_datetime(df_res["t_start"], utc=True, errors="coerce")
    t_exit_utc  = pd.to_datetime(df_res["exit_time"], utc=True, errors="coerce")
    df_res["exit_days"] = ((t_exit_utc - t_start_utc).dt.total_seconds() / 86400.0).round(3)

    before_cnt = len(df_res)
    if ENFORCE_ONE_AT_A_TIME and MAX_CONCURRENT and MAX_CONCURRENT > 0:
        kept, backfill_map = _slot_free_concurrency_with_backfill_markers(df_res, MAX_CONCURRENT)
        kept["_backfill_start"] = pd.NaT
        df_res["_backfill_start"] = df_res.index.map(backfill_map).astype("datetime64[ns, UTC]")
        df_res = kept
    else:
        df_res["_backfill_start"] = pd.NaT
    after_cnt = len(df_res)
    if MAX_CONCURRENT and MAX_CONCURRENT > 0:
        print(f"🔒 Applied global concurrency cap = {MAX_CONCURRENT}: kept {after_cnt} of {before_cnt}")

    backfill_rows = []
    if LIMIT_BACKFILL_ENABLE and ENFORCE_ONE_AT_A_TIME and MAX_CONCURRENT and MAX_CONCURRENT > 0:
        full_idx = set(df_res.index.tolist())
        orig_idx = set(range(len(results)))
        skipped_idx = sorted(list(orig_idx - full_idx))
        if skipped_idx:
            sig_indexed = df_sig.reset_index(drop=True)
            for i in skipped_idx:
                if i < 0 or i >= len(sig_indexed):
                    continue
                sig_row = sig_indexed.iloc[i]
                symbol = sig_row["symbol"]
                df4h = price_cache.get(symbol)
                if df4h is None or df4h.empty:
                    continue
                row_out_dict = {"_backfill_start": df_res["_backfill_start"].iloc[0] if "_backfill_start" in df_res.columns and len(df_res)>0 else pd.NaT}
                if i < len(df_res.index) and "_backfill_start" in df_res.columns:
                    try:
                        row_out_dict["_backfill_start"] = df_res.loc[i, "_backfill_start"]
                    except Exception:
                        pass
                res = _simulate_limit_backfill(row_out_dict, sig_row, df4h, as_of)
                if isinstance(res, dict) and res.get("status") is None:
                    backfill_rows.append(res)

    if backfill_rows:
        df_back = pd.DataFrame(backfill_rows)
        df_res = pd.concat([df_res, df_back], ignore_index=True, sort=False)

    df_res_adj = df_res.copy()
    mask_exec = (df_res_adj.get("skipped") == False)
    df_res_adj.loc[mask_exec, "exit_reason"] = df_res_adj.loc[mask_exec, "exit_reason"].fillna("price")

    init_cap = float(initial_capital) if initial_capital is not None else float(INITIAL_CAPITAL or 0.0)
    if init_cap <= 0 or not capital_aware:
        df_out = df_res_adj.copy(); df_out["skipped"] = df_res_adj.get("skipped", False); eq_sheet = pd.DataFrame()
    else:
        df_out, eq_sheet = simulate_capital_notional(
            df_res_adj, init_cap, POSITION_FRACTION, stop_pct=STOP_PCT, take_pct=TAKE_PCT
        )

    df_exec = df_out[df_out.get("skipped") == False].copy()
    by_variant = (df_exec.groupby("variant")
                        .agg(trades=("win", "size"),
                             wins=("win", "sum"),
                             winrate_pct=("win", lambda s: round(100.0 * float(s.sum()) / max(int(s.size), 1), 2)),
                             pnl_pct=("pnl_pct", "sum"),
                             pnl_usd=("pnl_usd", "sum"))
                        .reset_index()) if not df_exec.empty else pd.DataFrame()
    by_exit_reason = safe_group_exit_reason(df_out)

    reason = df_out.get("exit_reason_raw", df_out.get("exit_reason", pd.Series([], dtype=str))).astype(str).fillna("")
    lt_res = df_out.get("lt_resolution", pd.Series([], dtype=str)).astype(str).fillna("")

    n_total = int(len(df_out))
    n_exec  = int((df_out.get("skipped") == False).sum()) if "skipped" in df_out.columns else n_total
    n_skip  = int((df_out.get("skipped") == True).sum()) if "skipped" in df_out.columns else 0

    uncertain_mask = reason.str.startswith("uncertain")
    n_uncertain = int(uncertain_mask.sum())
    uncertain_pct = round(100.0 * n_uncertain / max(n_total, 1), 2)

    exec_mask = (df_out.get("skipped") == False) if "skipped" in df_out.columns else pd.Series([True]*n_total)
    ltf_1m_exec = int((lt_res[exec_mask] == "1m").sum())
    ltf_1m_exec_pct = round(100.0 * ltf_1m_exec / max(n_exec, 1), 2)

    quality_summary = pd.DataFrame([
        {"metric": "n_total", "value": n_total},
        {"metric": "n_executed", "value": n_exec},
        {"metric": "n_skipped", "value": n_skip},
        {"metric": "n_uncertain", "value": n_uncertain},
        {"metric": "uncertain_pct", "value": uncertain_pct},
        {"metric": "ltf_fallback_1m_exec", "value": ltf_1m_exec},
        {"metric": "ltf_fallback_1m_exec_pct", "value": ltf_1m_exec_pct},
        {"metric": "max_concurrent_used", "value": MAX_CONCURRENT},
        {"metric": "kept_after_concurrency", "value": len(df_res)},
        {"metric": "slot_free_threshold", "value": SLOT_FREE_AT_REMAINING_PCT},
        {"metric": "backfill_enabled", "value": int(LIMIT_BACKFILL_ENABLE)},
        {"metric": "backfill_band_pct", "value": LIMIT_BACKFILL_BAND_PCT},
        {"metric": "backfill_max_age_h", "value": LIMIT_BACKFILL_MAX_AGE_HOURS},
    ])

    lt_resolution_all  = lt_res.value_counts(dropna=False).rename("trades").reset_index().rename(columns={"index": "lt_resolution"})
    lt_resolution_exec = lt_res[exec_mask].value_counts(dropna=False).rename("trades").reset_index().rename(columns={"index": "lt_resolution"})
    uncertain_breakdown = (reason[uncertain_mask]
                            .value_counts(dropna=False)
                            .rename("trades").reset_index()
                            .rename(columns={"index": "exit_reason"}))

    extra = {
        "quality_summary": quality_summary,
        "lt_resolution_all": lt_resolution_all,
        "lt_resolution_exec": lt_resolution_exec,
        "uncertain_breakdown": uncertain_breakdown,
    }

    time_cols = [c for c in ["t_start","close_time","imb_time","exit_time","slot_free_time","as_of"] if c in df_out.columns]
    for c in time_cols:
        try:
            s = pd.to_datetime(df_out[c], errors="coerce", utc=True)
            df_out[c] = s.dt.tz_convert(None)
        except Exception:
            pass
    if isinstance(eq_sheet, pd.DataFrame):
        eq_time_cols = [c for c in ["time","as_of"] if c in eq_sheet.columns]
        for c in eq_time_cols:
            try:
                s = pd.to_datetime(eq_sheet[c], errors="coerce", utc=True)
                eq_sheet[c] = s.dt.tz_convert(None)
            except Exception:
                pass

    finalize_write(result_path, df_out, eq_sheet, by_variant, by_exit_reason, extra_sheets=extra)
    print(f"✅ MOMENTUM eval saved → {result_path}")

def _dedup_strongest_per_symbol_time(df_sig: pd.DataFrame) -> pd.DataFrame:
    if df_sig is None or df_sig.empty or "imb_time" not in df_sig.columns:
        return df_sig
    df = df_sig.copy()
    if "strength" not in df.columns:
        return df.sort_values(["symbol", "imb_time"]).drop_duplicates(["symbol", "imb_time"], keep="first")
    df = df.sort_values(["symbol", "imb_time", "strength"], ascending=[True, True, False])
    return df.drop_duplicates(["symbol", "imb_time"], keep="first")

if __name__ == "__main__":
    import argparse
    def _b(v): return str(v).strip().lower() in ("1","true","yes","y","on")
    p = argparse.ArgumentParser(description="Evaluate MOMENTUM with early entry, scale-out TP/SL, breakeven, cap-at-last-TP, slot-free concurrency and limit backfill.")
    p.add_argument("signals")
    p.add_argument("--out", default=None)
    p.add_argument("--interval", default="4h")
    p.add_argument("--lookback-days", type=int, default=360)
    p.add_argument("--ttl-days", type=int, default=None)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--capital-aware", type=_b, default=True)
    p.add_argument("--intrabar", default=None)
    p.add_argument("--intrabar-lookback-days", type=int, default=None)
    p.add_argument("--only-filled", action="store_true")
    p.add_argument("--dedup", action="store_true")

    p.add_argument("--fee-taker-pct", type=float, default=None)
    p.add_argument("--entry-slippage-pct", type=float, default=None)
    p.add_argument("--entry-limit-slippage-pct", type=float, default=None)
    p.add_argument("--exit-slippage-pct", type=float, default=None)
    p.add_argument("--stop-slippage-pct", type=float, default=None)

    p.add_argument("--enable-early-check", type=_b, default=None)
    p.add_argument("--early-minutes", type=int, default=None)
    p.add_argument("--early-move-pct", type=float, default=None)
    p.add_argument("--early-vol-mult", type=float, default=None)
    p.add_argument("--early-vol-sma-n", type=int, default=None)
    p.add_argument("--early-require-both", type=_b, default=None)
    p.add_argument("--early-use-last-minute", type=_b, default=None)

    p.add_argument("--enable-scale-out", type=_b, default=None)
    p.add_argument("--scale-tp-levels", type=str, default=None)
    p.add_argument("--scale-sl-levels", type=str, default=None)
    p.add_argument("--breakeven-after-tp1", type=_b, default=None)
    p.add_argument("--breakeven-offset-pct", type=float, default=None)
    p.add_argument("--cap-at-last-tp", type=_b, default=None)

    p.add_argument("--slot-free-at-remaining-pct", type=float, default=None)

    p.add_argument("--limit-backfill-enable", type=_b, default=None)
    p.add_argument("--limit-backfill-max-age-hours", type=int, default=None)
    p.add_argument("--limit-backfill-band-pct", type=float, default=None)
    p.add_argument("--limit-backfill-at", type=str, default=None)

    p.add_argument("--no-one-at-a-time", action="store_true")

    args = p.parse_args()

    if args.fee_taker_pct        is not None: os.environ["FEE_TAKER"]                  = str(args.fee_taker_pct)
    if args.entry_slippage_pct   is not None: os.environ["ENTRY_SLIPPAGE_PCT"]         = str(args.entry_slippage_pct)
    if args.entry_limit_slippage_pct is not None: os.environ["ENTRY_LIMIT_SLIPPAGE_PCT"]= str(args.entry_limit_slippage_pct)
    if args.exit_slippage_pct    is not None: os.environ["EXIT_SLIPPAGE_PCT"]          = str(args.exit_slippage_pct)
    if args.stop_slippage_pct    is not None: os.environ["STOP_SLIPPAGE_PCT"]          = str(args.stop_slippage_pct)

    if args.enable_early_check   is not None: os.environ["ENABLE_EARLY_CHECK"]           = "1" if args.enable_early_check else "0"
    if args.early_minutes        is not None: os.environ["EARLY_CHECK_MIN_BEFORE_CLOSE"] = str(int(args.early_minutes))
    if args.early_move_pct       is not None: os.environ["EARLY_MOVE_PCT"]               = str(float(args.early_move_pct))
    if args.early_vol_mult       is not None: os.environ["EARLY_VOL_MULT"]               = str(float(args.early_vol_mult))
    if args.early_vol_sma_n      is not None: os.environ["EARLY_VOL_SMA_N"]              = str(int(args.early_vol_sma_n))
    if args.early_require_both   is not None: os.environ["EARLY_REQUIRE_BOTH"]           = "1" if args.early_require_both else "0"
    if args.early_use_last_minute is not None: os.environ["EARLY_USE_LAST_MINUTE"]       = "1" if args.early_use_last_minute else "0"

    if args.enable_scale_out     is not None: os.environ["ENABLE_SCALE_OUT"]             = "1" if args.enable_scale_out else "0"
    if args.scale_tp_levels      is not None: os.environ["SCALE_TP_LEVELS"]              = str(args.scale_tp_levels)
    if args.scale_sl_levels      is not None: os.environ["SCALE_SL_LEVELS"]              = str(args.scale_sl_levels)
    if args.breakeven_after_tp1  is not None: os.environ["SCALE_BREAKEVEN_AFTER_TP1"]    = "1" if args.breakeven_after_tp1 else "0"
    if args.breakeven_offset_pct is not None: os.environ["BREAKEVEN_OFFSET_PCT"]         = str(float(args.breakeven_offset_pct))
    if args.cap_at_last_tp       is not None: os.environ["CAP_AT_LAST_TP"]               = "1" if args.cap_at_last_tp else "0"

    if args.slot_free_at_remaining_pct is not None:
        os.environ["SLOT_FREE_AT_REMAINING_PCT"] = str(float(args.slot_free_at_remaining_pct))

    if args.limit_backfill_enable is not None: os.environ["LIMIT_BACKFILL_ENABLE"] = "1" if args.limit_backfill_enable else "0"
    if args.limit_backfill_max_age_hours is not None: os.environ["LIMIT_BACKFILL_MAX_AGE_HOURS"] = str(int(args.limit_backfill_max_age_hours))
    if args.limit_backfill_band_pct is not None: os.environ["LIMIT_BACKFILL_BAND_PCT"] = str(float(args.limit_backfill_band_pct))
    if args.limit_backfill_at is not None: os.environ["LIMIT_BACKFILL_AT"] = str(args.limit_backfill_at)

    if args.no_one_at_a_time: os.environ["EVAL_ENFORCE_ONE_AT_A_TIME"] = "0"

    sig_path = os.path.expanduser(args.signals)
    res_path = os.path.expanduser(args.out) if args.out else os.path.splitext(sig_path)[0] + "_momentum_eval.xlsx"

    if args.intrabar is not None:
        os.environ["INTRABAR_INTERVALS"] = args.intrabar
    if args.intrabar_lookback_days is not None:
        os.environ["INTRABAR_LOOKBACK_DAYS_FALLBACK"] = str(int(args.intrabar_lookback_days))

    evaluate_momentum(
        signals_path=sig_path, result_path=res_path,
        lookback_days=int(args.lookback_days), interval=str(args.interval),
        max_days=(int(args.ttl_days) if args.ttl_days is not None else None),
        only_filled=bool(args.only_filled), dedup=bool(args.dedup),
        initial_capital=(float(args.initial_capital) if args.initial_capital is not None else None),
        capital_aware=bool(args.capital_aware),
    )