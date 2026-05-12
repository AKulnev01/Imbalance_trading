# predict/tp_entry/outcomes_shared.py
import math
import numpy as np
import pandas as pd

NOTIONAL_USD = 100.0

def apply_entry_slip(px: float, side: str, slip_pct: float) -> float:
    return px * (1.0 + slip_pct) if side == "BUY" else px * (1.0 - slip_pct)

def apply_exit_slip(px: float, side: str, slip_pct: float) -> float:
    return px * (1.0 - slip_pct) if side == "BUY" else px * (1.0 + slip_pct)

def side_of_bar(o: float, c: float) -> str:
    return "BUY" if c >= o else "SELL"

def resolve_first_hit(df_m1: pd.DataFrame, side: str, entry_ts: pd.Timestamp,
                      tp: float, sl: float, ttl_h: int, slip_pct: float,
                      tie_break: str = "sl"):
    """
    Возвращает: outcome, close_ts, exit_px_adj
      outcome ∈ {"tp","sl","timeout","none","both"}  (both=в одной минуте задеты и tp, и sl)
    """
    t_end = entry_ts + pd.Timedelta(hours=int(ttl_h))
    m = df_m1[(df_m1.index >= entry_ts) & (df_m1.index <= t_end)]
    if m.empty:
        return "none", t_end, float("nan")

    for ts, r in m.iterrows():
        hi = float(r["high"]); lo = float(r["low"])
        hit_tp = (hi >= tp) if side == "BUY" else (lo <= tp)
        hit_sl = (lo <= sl) if side == "BUY" else (hi >= sl)
        if hit_tp and hit_sl:
            # в реале порядок неизвестен → используем policy
            if tie_break == "tp":
                return "tp", ts, apply_exit_slip(tp, side, slip_pct)
            elif tie_break == "sl":
                return "sl", ts, apply_exit_slip(sl, side, slip_pct)
            else:
                return "both", ts, float("nan")
        if hit_sl:
            return "sl", ts, apply_exit_slip(sl, side, slip_pct)
        if hit_tp:
            return "tp", ts, apply_exit_slip(tp, side, slip_pct)

    # таймаут: закроемся по последней минуте
    last_ts = m.index[-1]
    last_close = float(m.iloc[-1]["close"])
    return "timeout", last_ts, apply_exit_slip(last_close, side, slip_pct)

def compute_path_stats(df_m1: pd.DataFrame, side: str, entry_ts: pd.Timestamp,
                       entry_px: float, ttl_h: int):
    """
    MFE/MAE в % относительно entry_px за весь TTL (по минуткам).
    """
    t_end = entry_ts + pd.Timedelta(hours=int(ttl_h))
    m = df_m1[(df_m1.index >= entry_ts) & (df_m1.index <= t_end)]
    if m.empty:
        return np.nan, np.nan, np.nan, np.nan

    if side == "BUY":
        # рост = хорошо
        mfe = (m["high"].max()  - entry_px) / entry_px
        mae = (m["low"].min()   - entry_px) / entry_px
        close_ret = (m.iloc[-1]["close"] - entry_px) / entry_px
    else:
        # падение = хорошо
        mfe = (entry_px - m["low"].min())  / entry_px
        mae = (entry_px - m["high"].max()) / entry_px
        close_ret = (entry_px - m.iloc[-1]["close"]) / entry_px

    # «моментум» внутри окна как макс. Δ за минуту в пользу позиции
    # (просто эвристика — полезно как фича)
    mrets = m["close"].pct_change().fillna(0.0).to_numpy()
    mom_peak = float(np.maximum.accumulate(np.where(side=="BUY", mrets, -mrets)).max()) if len(mrets)>0 else 0.0

    return float(mfe), float(mae), float(close_ret), mom_peak

def pnl_pct_from_outcome(side: str, entry_px: float, exit_px: float) -> float:
    if side == "BUY":
        return (exit_px - entry_px) / entry_px * 100.0
    else:
        return (entry_px - exit_px) / entry_px * 100.0

def evaluate_tp_sl_grid(df_m1: pd.DataFrame, side: str, entry_ts: pd.Timestamp,
                        entry_px: float, ttl_h: int, slip_pct: float,
                        tp_grid, sl_grid, tie_break: str = "sl"):
    """
    Возвращает DataFrame по сетке TP/SL с метриками:
      outcome, pnl_pct, tp_pct, sl_pct
    """
    rows = []
    for tp_pct in tp_grid:
        tp_level = entry_px * (1 + tp_pct) if side == "BUY" else entry_px * (1 - tp_pct)
        for sl_pct in sl_grid:
            sl_level = entry_px * (1 - sl_pct) if side == "BUY" else entry_px * (1 + sl_pct)
            outcome, ct, ex = resolve_first_hit(df_m1, side, entry_ts, tp_level, sl_level, ttl_h, slip_pct, tie_break)
            if outcome in ("tp","sl","timeout"):
                pnl_pct = pnl_pct_from_outcome(side, entry_px, ex)
            else:
                pnl_pct = float("nan")
            rows.append({
                "tp_pct": float(tp_pct),
                "sl_pct": float(sl_pct),
                "outcome": outcome,
                "pnl_pct": float(pnl_pct) if not math.isnan(pnl_pct) else np.nan,
            })
    return pd.DataFrame(rows)

def pick_best_tp_sl(grid_df: pd.DataFrame, objective: str = "exp_pnl",
                    min_rr: float = 1.5, min_tp: float = 0.01, max_tp: float = 0.40,
                    min_sl: float = 0.005, max_sl: float = 0.30):
    """
    Выбор «лучшего» TP/SL на сетке.
      objective:
        - "exp_pnl"       : ожидание PNL на сделку (tp_rate*tp - sl_rate*sl)
        - "f1_like"       : 2*tp_rate*rr / (tp_rate+rr+eps) — просто как альтернатива
        - "tp_rate_rr"    : сначала максимизируем tp_rate при RR>=min_rr, потом RR
    Возвращает dict(best_tp_pct,best_sl_pct, rr, tp_rate, sl_rate, exp_pnl, objective_used)
    """
    x = grid_df.copy()
    x = x[(x["tp_pct"].between(min_tp, max_tp)) & (x["sl_pct"].between(min_sl, max_sl))].copy()
    if x.empty:
        return None

    # сводим по (tp,sl)
    sv = (
        x.groupby(["tp_pct","sl_pct"])["outcome"]
         .value_counts().unstack(fill_value=0).reset_index()
    )
    for col in ("tp","sl","timeout","none","both"):
        if col not in sv.columns:
            sv[col] = 0
    sv["n"] = sv[["tp","sl","timeout"]].sum(axis=1).clip(lower=1)
    sv["tp_rate"] = sv["tp"]/sv["n"]
    sv["sl_rate"] = sv["sl"]/sv["n"]
    sv["rr"] = sv["tp_pct"]/sv["sl_pct"]  # risk-reward
    sv = sv[sv["rr"] >= float(min_rr)].copy()
    if sv.empty:
        return None

    if objective == "exp_pnl":
        sv["exp_pnl"] = sv["tp_rate"]*sv["tp_pct"] - sv["sl_rate"]*sv["sl_pct"]
        sv = sv.sort_values(["exp_pnl","tp_rate","rr"], ascending=False)
    elif objective == "f1_like":
        eps = 1e-9
        sv["score"] = 2*sv["tp_rate"]*sv["rr"] / (sv["tp_rate"] + sv["rr"] + eps)
        sv = sv.sort_values(["score","tp_rate","rr"], ascending=False)
    else:  # "tp_rate_rr"
        sv = sv.sort_values(["tp_rate","rr","sl_pct"], ascending=[False, False, True])

    b = sv.iloc[0]
    return {
        "best_tp_pct": float(b["tp_pct"]),
        "best_sl_pct": float(b["sl_pct"]),
        "rr": float(b["rr"]),
        "tp_rate": float(b["tp_rate"]),
        "sl_rate": float(b["sl_rate"]),
        "exp_pnl": float(b.get("exp_pnl", np.nan)),
        "objective_used": objective
    }