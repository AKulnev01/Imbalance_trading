# predict/tp_entry/backtest_tp_sl_dynamic.py
import os, sys, argparse, json, math, time, warnings
import numpy as np, pandas as pd
from joblib import load
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

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

def load_bundle(path):
    obj = load(path)
    if isinstance(obj, dict) and "model" in obj: return obj
    return {"model": obj, "features": None}

def load_symbol_te(model_dir):
    path = os.path.join(model_dir, "symbol_te.json")
    if not os.path.exists(path):
        return {"map": {}, "global": {
            "sym_tp_mean":0.0,"sym_sl_mean":0.0,"sym_winrate":0.0,"sym_atr14_mean":0.0,"sym_volz_mean":0.0
        }}
    with open(path, "r") as f:
        return json.load(f)

def attach_te(df, te):
    te_map = te.get("map", {})
    te_global = te.get("global", {})
    def _get(sym, key):
        d = te_map.get(sym)
        if d is None: return te_global.get(key, 0.0)
        v = d.get(key, te_global.get(key, 0.0))
        return float(v if np.isfinite(v) else te_global.get(key, 0.0))
    out = df.copy()
    out["sym_tp_mean"]    = out["symbol"].map(lambda s: _get(s,"sym_tp_mean"))
    out["sym_sl_mean"]    = out["symbol"].map(lambda s: _get(s,"sym_sl_mean"))
    out["sym_winrate"]    = out["symbol"].map(lambda s: _get(s,"sym_winrate"))
    out["sym_atr14_mean"] = out["symbol"].map(lambda s: _get(s,"sym_atr14_mean"))
    out["sym_volz_mean"]  = out["symbol"].map(lambda s: _get(s,"sym_volz_mean"))
    return out

def clamp_tp_sl(tp, sl, min_rr=1.5, min_tp=0.01, max_tp=0.35, min_sl=0.005, max_sl=0.28):
    tp = float(np.clip(tp, min_tp, max_tp))
    sl = float(np.clip(sl, min_sl, max_sl))
    if tp / max(sl,1e-9) < min_rr:
        sl = max(tp / max(min_rr,1e-9), min_sl)
        sl = float(np.clip(sl, min_sl, max_sl))
    return tp, sl

def passes_dir_rules(side, fr):
    # простые направленные правила как в старом пайплайне
    try:
        if side=="BUY":
            return (fr.get("ema_diff_pct",0.0) >= 0.0 and
                    fr.get("vol_z",0.0)      >= 1.0 and
                    fr.get("body_ratio",0.0) >= 0.3 and
                    fr.get("lower_wick_ratio",1.0) <= 0.5)
        else:
            return (fr.get("ema_diff_pct",0.0) <= 0.0 and
                    fr.get("vol_z",0.0)      >= 1.0 and
                    fr.get("body_ratio",0.0) >= 0.3 and
                    fr.get("upper_wick_ratio",1.0) <= 0.5)
    except: 
        return False

def main():
    ap = argparse.ArgumentParser(description="Backtest dynamic TP/SL with symbol target encoding + dir-rules + progress.")
    ap.add_argument("--data", default="./predict/tp_entry/tp_training_base_fullrecalc.parquet")
    ap.add_argument("--model-dir", default="./predict/tp_entry/models_tp_sl")
    ap.add_argument("--m1-dir", default="./data/m1")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--ttl-hours", type=int, default=80)
    ap.add_argument("--slippage-pct", type=float, default=0.004)
    ap.add_argument("--after-min-pp", type=float, default=0.0, help="min margin over 0.5 for tie classifier")
    ap.add_argument("--apply-dir-rules", type=int, default=0, help="1=enable simple directional rules")
    ap.add_argument("--notional-usd", type=float, default=100.0)
    ap.add_argument("--out", default="./predict/tp_entry/tp_sl_backtest.parquet")
    args = ap.parse_args()

    t0 = time.perf_counter()

    tp_bundle = load_bundle(os.path.join(args.model_dir, "tp_reg.pkl"))
    sl_bundle = load_bundle(os.path.join(args.model_dir, "sl_reg.pkl"))
    tie_bundle= load_bundle(os.path.join(args.model_dir, "tie_cls.pkl"))
    te = load_symbol_te(args.model_dir)

    tp_model, feats = tp_bundle["model"], tp_bundle["features"] or []
    sl_model = sl_bundle["model"]; min_rr = float(tp_bundle.get("min_rr", 1.5))
    tie_model = tie_bundle["model"]; tie_feats = tie_bundle["features"] or feats

    df = pd.read_parquet(args.data).replace([np.inf,-np.inf], np.nan)

    if args.symbols.strip():
        keep = set([s.strip().upper() for s in args.symbols.split(",") if s.strip()])
        df = df[df["symbol"].str.upper().isin(keep)]

    # добавим TE к исходным данным
    df = attach_te(df, te)

    rows = []
    uniq_syms = df["symbol"].nunique()
    for sym, grp in tqdm(df.groupby("symbol"), total=uniq_syms, desc="Symbols"):
        m1 = load_m1(sym, args.m1_dir)
        if m1.empty: 
            continue

        # подготовим X с точным набором фичей (чтобы не было варнингов)
        Xt = pd.DataFrame(index=grp.index)
        for c in feats:
            Xt[c] = pd.to_numeric(grp.get(c, 0.0), errors="coerce").fillna(0.0)
        # убедимся, что TE-колонки есть, если они фигурируют среди фичей
        for c in ["sym_tp_mean","sym_sl_mean","sym_winrate","sym_atr14_mean","sym_volz_mean"]:
            if c in feats and c not in Xt.columns:
                Xt[c] = pd.to_numeric(grp.get(c, 0.0), errors="coerce").fillna(0.0)

        # предсказания без предупреждений: передаём DataFrame с именами колонок
        tp_pred = tp_model.predict(Xt)
        sl_pred = sl_model.predict(Xt)

        # proba TP-first
        try:
            Xtie = Xt[tie_feats] if isinstance(tie_feats, (list, tuple)) and len(tie_feats)>0 else Xt
            p_tp = tie_model.predict_proba(Xtie)[:,1]
        except Exception:
            z = tie_model.decision_function(Xt.values); p_tp = (z - z.min())/(z.max()-z.min()+1e-12)

        grp_reset = grp.reset_index(drop=True)
        for i, r in grp_reset.iterrows():
            # направление бара по датасету
            side = "BUY" if float(r.get("close", r.get("open", 0)) - r.get("open", 0)) >= 0 else "SELL"
            t_open = ensure_utc(r["time_open"]); entry_ts = t_open + pd.Timedelta(hours=4)
            entry_ref = float(r.get("close", np.nan))
            if not np.isfinite(entry_ref): 
                continue
            entry = apply_entry_slip(entry_ref, side, args.slippage_pct)

            # dir-rules (по желанию)
            if args.apply_dir_rules:
                fr = {k: float(r.get(k, 0.0)) for k in [
                    "ema_diff_pct","vol_z","body_ratio","upper_wick_ratio","lower_wick_ratio"
                ]}
                if not passes_dir_rules(side, fr):
                    continue

            tp = float(tp_pred[i]); sl = float(sl_pred[i])
            tp, sl = clamp_tp_sl(tp, sl, min_rr=min_rr)

            # AFTER: порог по уверенности модели tie
            if (float(p_tp[i]) - 0.5) < float(args.after_min_pp): 
                continue

            # абсолютные уровни
            if side=="BUY":
                tp_px = entry*(1+tp); sl_px = entry*(1-sl)
            else:
                tp_px = entry*(1-tp); sl_px = entry*(1+sl)

            win, ct, exit_px, reason = resolve_exit_minutes(m1, side, entry_ts, tp_px, sl_px, args.ttl_hours, args.slippage_pct)
            pct = (exit_px - entry)/max(entry,1e-12)*100.0 if side=="BUY" else (entry - exit_px)/max(entry,1e-12)*100.0
            pnl_usd = float(args.notional_usd) * (pct / 100.0)

            rows.append({
                "symbol": sym, "time_open": t_open.tz_localize(None), "side": side,
                "tp_pct_pred": float(tp), "sl_pct_pred": float(sl), "p_tp_first": float(p_tp[i]),
                "t_start": entry_ts.tz_localize(None),
                "exit_time": pd.to_datetime(ct, utc=True, errors="coerce").tz_localize(None) if pd.notna(ct) else None,
                "exit_reason": reason, "pnl_pct": float(pct), "pnl_usd": pnl_usd, "win": bool(win),
            })

    if not rows:
        print("no trades produced"); 
        return

    trades = pd.DataFrame(rows).sort_values(["t_start","symbol"]).reset_index(drop=True)
    out = os.path.expanduser(args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    trades.to_parquet(out, index=False)

    # по символам
    bysym = trades.groupby("symbol").agg(
        n=("win","size"),
        winrate=("win", lambda s: float(100.0*s.mean() if len(s)>0 else 0.0)),
        pnl_pct=("pnl_pct","sum"),
        pnl_usd=("pnl_usd","sum")
    ).reset_index().sort_values("pnl_usd", ascending=False)
    bysym.to_parquet(os.path.splitext(out)[0] + "_bysymbol.parquet", index=False)

    # эквити кривая суммарная (по времени)
    eq = trades.sort_values("t_start")[["t_start","pnl_usd"]].copy()
    eq["equity_usd"] = eq["pnl_usd"].cumsum()
    eq.to_parquet(os.path.splitext(out)[0] + "_equity.parquet", index=False)

    total = {
        "trades": int(len(trades)),
        "wins": int(trades["win"].sum()),
        "winrate_pct": float(100.0*trades["win"].mean()),
        "pnl_pct_sum": float(trades["pnl_pct"].sum()),
        "pnl_usd_sum": float(trades["pnl_usd"].sum()),
        "notional_usd": float(args.notional_usd),
        "symbols": int(trades["symbol"].nunique()),
        "ttl_hours": int(args.ttl_hours),
        "after_min_pp": float(args.after_min_pp),
        "apply_dir_rules": int(args.apply_dir_rules),
    }
    with open(os.path.splitext(out)[0] + "_summary.json", "w") as f:
        json.dump({"total": total}, f, ensure_ascii=False, indent=2)

    dt = time.perf_counter() - t0
    print("=== BACKTEST DONE ===")
    print(f"trades={total['trades']}  winrate={total['winrate_pct']:.1f}%  "
          f"pnl_pct_sum={total['pnl_pct_sum']:.1f}  pnl_usd_sum={total['pnl_usd_sum']:.2f}  "
          f"symbols={total['symbols']}  time={dt:.1f}s")
    print(f"saved → {out}")

if __name__=="__main__":
    main()