import os, sys, argparse, math, json
import numpy as np, pandas as pd
from joblib import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from .features_shared import build_4h_features, resample_4h

NOTIONAL_USD = 100.0

def ensure_utc(ts):
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t): return t
    try:
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except: return pd.Timestamp(t).tz_localize("UTC")

def load_m1(symbol, m1_dir):
    p = os.path.join(os.path.expanduser(m1_dir), f"{symbol}_m1.parquet")
    if not os.path.exists(p): return pd.DataFrame()
    df = pd.read_parquet(p)
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True) if "ts" in df.columns else pd.to_datetime(df["timestamp"], utc=True)
    df = df.assign(ts=ts).set_index("ts").sort_index()
    cols = ["open","high","low","close","volume"]
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cols].dropna()

def apply_entry_slip(px, side, slip_pct): return px*(1+slip_pct) if side=="BUY" else px*(1-slip_pct)
def apply_exit_slip(px, side, slip_pct):  return px*(1-slip_pct) if side=="BUY" else px*(1+slip_pct)
def side_of_bar(o, c): return "BUY" if c>=o else "SELL"

def passes_dir_rules(side, fr):
    try:
        if side=="BUY":
            return (fr["ema_diff_pct"]>=0.0 and fr["vol_z"]>=1.0 and fr["body_ratio"]>=0.3 and fr["lower_wick_ratio"]<=0.5)
        else:
            return (fr["ema_diff_pct"]<=0.0 and fr["vol_z"]>=1.0 and fr["body_ratio"]>=0.3 and fr["upper_wick_ratio"]<=0.5)
    except: return False

def after_proba(bundle, fr):
    if bundle is None: return None
    model, feats = bundle.get("model"), bundle.get("features")
    if model is None: return None
    cols = feats or [c for c,v in fr.items() if isinstance(v, (int,float,np.floating))]
    x = {c: pd.to_numeric(fr.get(c,0.0), errors="coerce") for c in cols}
    X = pd.DataFrame([x]).fillna(0.0)
    if hasattr(model,"predict_proba"): return float(model.predict_proba(X.values)[:,1][0])
    if hasattr(model,"decision_function"):
        z = float(model.decision_function(X.values)[0]); return 1.0/(1.0+math.exp(-z))
    return None

def resolve_exit_minutes(df_m1, side, entry_ts, tp, sl, ttl_h, slip):
    t_end = entry_ts + pd.Timedelta(hours=int(ttl_h))
    m = df_m1[(df_m1.index >= entry_ts) & (df_m1.index <= t_end)]
    if m.empty: return False, t_end, float("nan"), "no_m1"
    for ts, r in m.iterrows():
        hi, lo = float(r["high"]), float(r["low"])
        if side=="BUY":
            if lo <= sl: return False, ts, apply_exit_slip(sl, side, slip), "sl"
            if hi >= tp: return True,  ts, apply_exit_slip(tp, side, slip), "tp"
        else:
            if hi >= sl: return False, ts, apply_exit_slip(sl, side, slip), "sl"
            if lo <= tp: return True,  ts, apply_exit_slip(tp, side, slip), "tp"
    last_ts = m.index[-1]; last_close = float(m.iloc[-1]["close"])
    return False, last_ts, apply_exit_slip(last_close, side, slip), "timeout_last_close"

def load_after_bundle(path):
    try:
        if not path or not os.path.exists(path): return None
        obj = load(path)
        if isinstance(obj, dict) and "model" in obj: return obj
        return {"model": obj, "features": None}
    except: return None

def main():
    ap = argparse.ArgumentParser(description="Backtest TP-first entries with dir rules and AFTER models.")
    ap.add_argument("--model-dir", default="./predict/tp_entry/models")
    ap.add_argument("--m1-dir", default="./data/m1")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--tp-pct", type=float, default=0.135)
    ap.add_argument("--sl-pct", type=float, default=0.04)
    ap.add_argument("--slippage-pct", type=float, default=0.004)
    ap.add_argument("--ttl-hours", type=int, default=80)
    ap.add_argument("--apply-dir-rules", type=int, default=1)
    ap.add_argument("--after-buy", default="./after_predict/models/after_buy.pkl")
    ap.add_argument("--after-sell", default="./after_predict/models/after_sell.pkl")
    ap.add_argument("--after-thr-buy", type=float, default=0.48)
    ap.add_argument("--after-thr-sell", type=float, default=0.55)
    ap.add_argument("--out", default="./predict/tp_entry/tp_backtest.xlsx")
    ap.add_argument("--debug", type=int, default=0)
    args = ap.parse_args()

    bundle = load(os.path.join(args.model_dir, "model.pkl"))
    model, features = bundle["model"], bundle["features"]
    thr = args.threshold
    if thr is None:
        tpath = os.path.join(args.model_dir, "threshold.txt")
        thr = float(open(tpath).read().strip()) if os.path.exists(tpath) else 0.5

    after_buy = load_after_bundle(args.after_buy)
    after_sell = load_after_bundle(args.after_sell)
    use_after = (after_buy is not None) or (after_sell is not None)

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        try:
            from config import TRADE_UNIVERSE, filter_universe
            symbols = filter_universe(TRADE_UNIVERSE or [])
            symbols = [s.upper() for s in symbols]
        except: 
            print("no symbols"); return

    rows, eq_points = [], []
    eq = 0.0
    c_bars = c_pred1 = c_dir = c_after = c_buy = c_sell = 0

    for sym in symbols:
        m1 = load_m1(sym, args.m1_dir)
        if m1.empty: 
            if args.debug: print(f"{sym}: no m1")
            continue
        h4 = resample_4h(m1)
        if h4.shape[0] < 60: 
            if args.debug: print(f"{sym}: few bars")
            continue
        feats = build_4h_features(h4)

        X = pd.DataFrame(index=feats.index)
        for c in features:
            X[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0) if c in feats.columns else 0.0

        try: proba = model.predict_proba(X.values)[:,1]
        except:
            z = model.decision_function(X.values); proba = (z-z.min())/(z.max()-z.min()+1e-12)

        pred = (proba >= float(thr)).astype(int)
        c_bars += len(pred)
        sig_idx = X.index[pred==1]
        c_pred1 += len(sig_idx)

        for t_open in sig_idx:
            fr = feats.loc[t_open].to_dict()
            side = "BUY" if float(h4.loc[t_open,"close"] - h4.loc[t_open,"open"]) >= 0 else "SELL"

            if args.apply_dir_rules:
                if not passes_dir_rules(side, fr): 
                    continue
                c_dir += 1

            if use_after:
                pa = after_proba(after_buy if side=="BUY" else after_sell, fr)
                if pa is None or pa < float(args.after_thr_buy if side=="BUY" else args.after_thr_sell): 
                    continue
                c_after += 1

            entry_ts = ensure_utc(t_open) + pd.Timedelta(hours=4)
            entry_ref = float(h4.loc[t_open,"close"])
            entry = apply_entry_slip(entry_ref, side, args.slippage_pct)
            if side=="BUY":
                sl = entry*(1-args.sl_pct); tp = entry*(1+args.tp_pct); c_buy += 1
            else:
                sl = entry*(1+args.sl_pct); tp = entry*(1-args.tp_pct); c_sell += 1

            win, close_time, exit_px, reason = resolve_exit_minutes(m1, side, entry_ts, tp, sl, args.ttl_hours, args.slippage_pct)
            ct = ensure_utc(close_time) if pd.notna(close_time) else entry_ts
            move = (exit_px - entry)/max(entry,1e-12)*100.0 if side=="BUY" else (entry - exit_px)/max(entry,1e-12)*100.0
            pnl = NOTIONAL_USD*(move/100.0); eq += pnl; eq_points.append({"ts":ct,"equity":eq})
            rows.append({
                "symbol": sym, "type": side, "proba": float(proba[X.index.get_loc(t_open)]),
                "time_open": ensure_utc(t_open), "t_start": entry_ts,
                "entry": float(entry), "tp": float(tp), "sl": float(sl),
                "close_time": ct, "close_price": float(exit_px), "exit_reason": reason,
                "win": bool(win), "pnl_pct": float(move), "pnl_usd": float(pnl), "variant": "TP_MODEL"
            })

    trades = pd.DataFrame(rows).sort_values(["t_start","symbol"]).reset_index(drop=True)
    if trades.empty:
        print("no trades"); 
        print(f"bars={c_bars} pred1={c_pred1} dir={c_dir} after={c_after}")
        return

    by_variant = trades.groupby("variant").agg(
        trades=("win","size"), wins=("win","sum"),
        winrate_pct=("win", lambda s: round(100.0*float(s.sum())/max(int(s.size),1),2)),
        pnl_pct=("pnl_pct","sum"), pnl_usd=("pnl_usd","sum")
    ).reset_index()

    eqdf = pd.DataFrame(eq_points).sort_values("ts") if eq_points else pd.DataFrame()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    def drop_tz(df):
        x=df.copy()
        for c in x.columns:
            if pd.api.types.is_datetime64_any_dtype(x[c]): x[c]=pd.to_datetime(x[c], utc=True, errors="coerce").dt.tz_localize(None)
        return x
    with pd.ExcelWriter(out, engine="openpyxl") as wr:
        drop_tz(trades).to_excel(wr, index=False, sheet_name="trades")
        by_variant.to_excel(wr, index=False, sheet_name="by_variant")
        if not eqdf.empty: drop_tz(eqdf).to_excel(wr, index=False, sheet_name="equity")

    print("— pipeline summary —")
    print(f"bars={c_bars} pred1={c_pred1} dir={c_dir} after={c_after} BUY={c_buy} SELL={c_sell}")
    print(f"saved: {out}")
    print(by_variant.to_string(index=False))

if __name__=="__main__":
    main()
