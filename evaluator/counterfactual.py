import numpy as np, pandas as pd
from typing import List, Dict, Tuple
def _fee_mult_bps(bps: float, side: str) -> float:
    sgn = +1.0 if side.lower()=="long" else -1.0
    return 1.0 + sgn * (bps * 1e-4)
def _apply_spread_slip(price: float, side: str, half_spread_bps: float, slip_bps: float) -> float:
    sgn = +1.0 if side.lower()=="long" else -1.0
    adj = (half_spread_bps + slip_bps) * 1e-4
    return price * (1.0 + sgn * adj)
def _intra_bar_hit(high: float, low: float, tp_px: float, sl_px: float, side: str, policy: str) -> str:
    if side=="long":
        tp_hit = high >= tp_px
        sl_hit = low  <= sl_px
    else:
        tp_hit = low  <= tp_px
        sl_hit = high >= sl_px
    if not tp_hit and not sl_hit: return "none"
    if policy == "optimistic": return "tp" if tp_hit else "sl"
    if policy == "pessimistic": return "sl" if sl_hit else "tp"
    if side=="long":
        if high >= tp_px: return "tp"
        if low  <= sl_px: return "sl"
        return "none"
    else:
        if low  <= tp_px: return "tp"
        if high >= sl_px: return "sl"
        return "none"
def _minute_iter(df: pd.DataFrame, start_ms: int):
    for _, r in df.iterrows():
        if r["ts"] < start_ms: continue
        yield int(r["ts"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
def _target_prices(entry_fill: float, rr_tp: float, rr_sl: float, side: str) -> Tuple[float,float]:
    if side=="long":
        tp = entry_fill * (1.0 + rr_tp/100.0)
        sl = entry_fill * (1.0 - rr_sl/100.0)
    else:
        tp = entry_fill * (1.0 - rr_tp/100.0)
        sl = entry_fill * (1.0 + rr_sl/100.0)
    return tp, sl
def evaluate_counterfactuals(m1: pd.DataFrame, entry_ts_ms: int, side: str, entry_price: float,
    rr_tp_list: List[float], rr_sl_list: List[float], ttl_hours_list: List[int],
    fees_bps: float, half_spread_bps: float, slip_bps: float, intra_bar_policy: str) -> List[Dict]:
    res = []
    for rr_tp in rr_tp_list:
        for rr_sl in rr_sl_list:
            for ttl_h in ttl_hours_list:
                deadline = entry_ts_ms + int(ttl_h*3600*1000)
                best = {"first_touch": "ttl", "touch_ts": deadline, "exit_price": None}
                tp_px, sl_px = _target_prices(entry_price, rr_tp, rr_sl, side)
                for ts, o, h, l, c in _minute_iter(m1, entry_ts_ms):
                    if ts > deadline: break
                    hit = _intra_bar_hit(h, l, tp_px, sl_px, side, intra_bar_policy)
                    if hit == "tp":
                        best = {"first_touch":"tp","touch_ts":ts,"exit_price":tp_px}; break
                    elif hit == "sl":
                        best = {"first_touch":"sl","touch_ts":ts,"exit_price":sl_px}; break
                if best["exit_price"] is None:
                    last = m1[m1["ts"]<=deadline].tail(1)
                    if len(last): best["exit_price"] = float(last["close"].values[0])
                px_in  = _apply_spread_slip(entry_price, side, half_spread_bps, slip_bps) * _fee_mult_bps(fees_bps, side)
                side_exit = "short" if side=="long" else "long"
                px_out = _apply_spread_slip(best["exit_price"], side_exit, half_spread_bps, slip_bps) * _fee_mult_bps(fees_bps, side_exit)
                pnl_pct = (px_out - px_in)/px_in if side=="long" else (px_in - px_out)/px_in
                res.append({
                    "rr_tp": rr_tp, "rr_sl": rr_sl, "ttl_hours": ttl_h,
                    "first_touch": best["first_touch"], "touch_ts": int(best["touch_ts"]),
                    "entry_adj": px_in, "exit_adj": px_out, "pnl_pct_after_cost": pnl_pct
                })
    return res
