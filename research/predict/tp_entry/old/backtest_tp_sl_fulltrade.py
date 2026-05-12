# predict/tp_entry/backtest_tp_sl_fulltrade.py
import os, sys, argparse, json, math, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np, pandas as pd
from joblib import load
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TE_COLS = ["sym_tp_mean", "sym_sl_mean", "sym_winrate", "sym_atr14_mean", "sym_volz_mean"]

def ensure_utc(ts):
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return t
    try:
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except:
        return pd.Timestamp(t).tz_localize("UTC")

def load_m1(symbol, m1_dir):
    p = os.path.join(os.path.expanduser(m1_dir), f"{symbol}_m1.parquet")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_parquet(p)
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True) if "ts" in df.columns else pd.to_datetime(df["timestamp"], utc=True)
    df = df.assign(ts=ts).set_index("ts").sort_index()
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cols].dropna()

def apply_entry_slip(px, side, slip_pct):
    return px * (1 + slip_pct) if side == "BUY" else px * (1 - slip_pct)

def apply_exit_slip(px, side, slip_pct):
    return px * (1 - slip_pct) if side == "BUY" else px * (1 + slip_pct)

# --- заменяет старую версию ---
def _apply_tie_calibration(p_raw, calib, how="avg"):
    """
    Корректное применение калибровки вероятностей.
    ВАЖНО: каждая калибровка применяется к НЕкалиброванным p_raw, а не последовательно.
      how = "avg"  -> среднее(platt(p_raw), isotonic(p_raw)) если обе есть
           "platt" -> только Platt
           "isotonic" -> только isotonic
    """
    if not calib:
        return np.asarray(p_raw, dtype=float)

    p_raw = np.asarray(p_raw, dtype=float)
    outs = []

    if "platt" in calib:
        a = float(calib["platt"]["a"]); b = float(calib["platt"]["b"])
        pr = np.clip(p_raw, 1e-6, 1-1e-6)
        z  = np.log(pr/(1-pr))
        p_platt = 1.0/(1.0 + np.exp(-(a*z + b)))
        outs.append(p_platt)

    if "isotonic" in calib:
        xs = np.asarray(calib["isotonic"]["x"], dtype=float)
        ys = np.asarray(calib["isotonic"]["y"], dtype=float)
        p_iso = np.interp(p_raw, xs, ys)
        outs.append(p_iso)

    if not outs:
        return p_raw

    if how == "platt" and "platt" in calib:
        return outs[0] if len(outs)==1 else p_platt
    if how == "isotonic" and "isotonic" in calib:
        # если обе есть, вернём именно изотонику
        return outs[-1] if len(outs)>0 else p_raw

    # по умолчанию усредняем все доступные калибровки
    return np.mean(np.vstack(outs), axis=0)

def resolve_exit_minutes(df_m1, side, entry_ts, tp, sl, ttl_h, slip):
    t_end = entry_ts + pd.Timedelta(hours=int(ttl_h))
    m = df_m1[(df_m1.index >= entry_ts) & (df_m1.index <= t_end)]
    if m.empty:
        return False, t_end, float("nan"), "no_m1"
    for ts, r in m.iterrows():
        hi, lo = float(r["high"]), float(r["low"])
        if side == "BUY":
            if lo <= sl:
                return False, ts, apply_exit_slip(sl, side, slip), "sl"
            if hi >= tp:
                return True, ts, apply_exit_slip(tp, side, slip), "tp"
        else:
            if hi >= sl:
                return False, ts, apply_exit_slip(sl, side, slip), "sl"
            if lo <= tp:
                return True, ts, apply_exit_slip(tp, side, slip), "tp"
    last_ts = m.index[-1]
    last_close = float(m.iloc[-1]["close"])
    return False, last_ts, apply_exit_slip(last_close, side, slip), "timeout_last_close"

def load_bundle(path):
    obj = load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj, "features": None}

def load_symbol_te(model_dir):
    path = os.path.join(model_dir, "symbol_te.json")
    if not os.path.exists(path):
        return {"map": {}, "global": {c: 0.0 for c in TE_COLS}}
    with open(path, "r") as f:
        return json.load(f)

def attach_te(df, te):
    te_map = te.get("map", {})
    te_gl = te.get("global", {})
    def _get(sym, key):
        d = te_map.get(str(sym))
        v = (d or {}).get(key, te_gl.get(key, 0.0))
        try:
            v = float(v)
        except:
            v = te_gl.get(key, 0.0)
        return v
    out = df.copy()
    for c in TE_COLS:
        out[c] = out["symbol"].map(lambda s: _get(s, c))
    return out

def clamp_tp_sl(tp, sl, min_rr=1.5, min_tp=0.01, max_tp=0.35, min_sl=0.005, max_sl=0.28):
    tp = float(np.clip(tp, min_tp, max_tp))
    sl = float(np.clip(sl, min_sl, max_sl))
    if tp / max(sl, 1e-9) < min_rr:
        sl = max(tp / max(min_rr, 1e-9), min_sl)
        sl = float(np.clip(sl, min_sl, max_sl))
    return tp, sl

def main():
    ap = argparse.ArgumentParser(description="Fulltrade backtest with date filtering, TE and calibrated TIE prob")
    ap.add_argument("--data", default="./predict/tp_entry/tp_training_base_fullrecalc.parquet")
    ap.add_argument("--model-dir", default="./predict/tp_entry/models_tp_sl_oot")
    ap.add_argument("--m1-dir", default="./data/m1")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--date-from", dest="date_from", default="")
    ap.add_argument("--date-to", dest="date_to", default="")
    ap.add_argument("--ttl-hours", type=int, default=80)
    ap.add_argument("--fee-pct", type=float, default=0.0015)
    ap.add_argument("--slip-exit-pct", type=float, default=0.006)
    ap.add_argument("--after-min-pp", type=float, default=0.0, help="legacy: use (p-0.5) >= after_min_pp")
    ap.add_argument("--p-thresh-abs", type=float, default=None, help="absolute threshold on calibrated prob p_cal (overrides after_min_pp)")
    ap.add_argument("--min-rr", type=float, default=1.5)
    ap.add_argument("--out", default="./predict/tp_entry/tp_sl_fulltrade_oot.parquet")
    ap.add_argument("--progress", type=int, default=1)
    ap.add_argument("--tie-calib", default="", help="path to tie_calibration.json (from calibrate_tie*_*.py)")
    args = ap.parse_args()

    print("📥 Loading data:", args.data, flush=True)
    df = pd.read_parquet(args.data).replace([np.inf, -np.inf], np.nan)
    df["symbol"] = df["symbol"].astype(str)
    # делаем UTC-aware, чтобы срезы по датам были корректны
    df["time_open"] = pd.to_datetime(df["time_open"], utc=True, errors="coerce")

    if args.symbols.strip():
        keep = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        df = df[df["symbol"].str.upper().isin(keep)]

    # фильтры по дате (в UTC)
    if args.date_from or args.date_to:
        dfrom = pd.Timestamp(args.date_from, tz="UTC") if args.date_from else None
        dto   = pd.Timestamp(args.date_to,   tz="UTC") if args.date_to   else None
        if dfrom is not None:
            df = df[df["time_open"] >= dfrom]
        if dto is not None:
            df = df[df["time_open"] < dto]

    # читаем калибровку вероятности (опционально)
    tie_calib = None
    if args.tie_calib and os.path.exists(args.tie_calib):
        with open(args.tie_calib, "r") as f:
            tie_calib = json.load(f)

    print(f"after filters: rows={len(df)} symbols={df['symbol'].nunique()}", flush=True)
    if len(df) == 0:
        print("no rows after filters — exit", flush=True)
        return

    print("📦 Loading models:", args.model_dir, flush=True)
    tp_bundle  = load_bundle(os.path.join(args.model_dir, "tp_reg.pkl"))
    sl_bundle  = load_bundle(os.path.join(args.model_dir, "sl_reg.pkl"))
    tie_bundle = load_bundle(os.path.join(args.model_dir, "tie_cls.pkl"))
    te = load_symbol_te(args.model_dir)

    tp_model, feats = tp_bundle["model"], tp_bundle["features"]
    sl_model = sl_bundle["model"]
    min_rr = float(tp_bundle.get("min_rr", args.min_rr))
    tie_model = tie_bundle["model"]
    tie_feats = tie_bundle["features"] or feats

    # добавляем target-encoding признаки по символам
    df = attach_te(df, te)

    # проверим наличие m1
    syms = sorted(df["symbol"].unique())
    missing = []
    for s in syms:
        p = os.path.join(os.path.expanduser(args.m1_dir), f"{s}_m1.parquet")
        if not os.path.exists(p):
            missing.append(s)
    print(f"m1 missing: {len(missing)} → {missing[:5]}", flush=True)

    rows = []
    it = tqdm(syms, desc="Symbols", mininterval=0.5) if args.progress else syms
    print(f"min_rr={min_rr}, after_min_pp={args.after_min_pp}, p_thresh_abs={args.p_thresh_abs}", flush=True)

    for sym in it:
        grp = df[df["symbol"] == sym]
        m1 = load_m1(sym, args.m1_dir)
        if m1.empty:
            continue

        # сборка матрицы признаков под порядок обучения
        Xt = pd.DataFrame(index=grp.index)
        for c in feats:
            Xt[c] = pd.to_numeric(grp.get(c, 0.0), errors="coerce").fillna(0.0)
        for c in TE_COLS:
            if c not in Xt.columns:
                Xt[c] = pd.to_numeric(grp.get(c, 0.0), errors="coerce").fillna(0.0)

        # предсказания TP/SL процентов
        tp_pred = tp_model.predict(Xt.values)  # доли (0.01 = +1%)
        sl_pred = sl_model.predict(Xt.values)

        # вероятность "TP наступит раньше SL" (TIE), затем калибровка
        try:
            Xtie = Xt[tie_feats] if isinstance(tie_feats, list) else Xt
            p_raw = tie_model.predict_proba(Xtie.values)[:, 1]
        except Exception:
            z = tie_model.decision_function(Xt.values)
            p_raw = (z - z.min()) / (z.max() - z.min() + 1e-12)
        p_cal = _apply_tie_calibration(p_raw, tie_calib)

        grp2 = grp.reset_index(drop=True)
        for i, r in grp2.iterrows():
            # направление по бару как раньше
            side = "BUY" if float(r.get("close", r.get("open", 0)) - r.get("open", 0)) >= 0 else "SELL"

            t_open = ensure_utc(r["time_open"])
            entry_ts = t_open + pd.Timedelta(hours=4)  # вход на следующей 4h
            entry_ref = float(r.get("close", np.nan))
            if not np.isfinite(entry_ref):
                continue

            # комиссия на входе как слип
            entry = apply_entry_slip(entry_ref, side, args.fee_pct)

            # TP/SL в долях, кламп + RR
            tp = float(tp_pred[i])
            sl = float(sl_pred[i])
            tp, sl = clamp_tp_sl(tp, sl, min_rr=min_rr)

            # фильтр по вероятности
            p_use = float(p_cal[i])
            if args.p_thresh_abs is not None:
                if p_use < float(args.p_thresh_abs):
                    continue
            else:
                if (p_use - 0.5) < float(args.after_min_pp):
                    continue

            if side == "BUY":
                tp_px = entry * (1 + tp)
                sl_px = entry * (1 - sl)
            else:
                tp_px = entry * (1 - tp)
                sl_px = entry * (1 + sl)

            win, ct, exit_px_raw, reason = resolve_exit_minutes(
                m1, side, entry_ts, tp_px, sl_px, args.ttl_hours, args.slip_exit_pct
            )
            # комиссия на выходе
            exit_px = apply_exit_slip(exit_px_raw, side, args.fee_pct)

            # PnL в процентах (как раньше)
            if side == "BUY":
                pct = (exit_px - entry) / max(entry, 1e-12) * 100.0
            else:
                pct = (entry - exit_px) / max(entry, 1e-12) * 100.0

            rows.append({
                "symbol": sym,
                "time_open": (t_open.tz_convert(None) if t_open.tzinfo else t_open),
                "side": side,
                "tp_pct_pred": tp,
                "sl_pct_pred": sl,
                "p_tp_first": float(p_raw[i]),
                "p_tp_cal": p_use,  # сохраняем калиброванную вероятность
                "t_start": (entry_ts.tz_convert(None) if entry_ts.tzinfo else entry_ts),
                "exit_time": (pd.to_datetime(ct, utc=True, errors="coerce").tz_convert(None) if pd.notna(ct) else None),
                "exit_reason": reason,
                "pnl_pct": float(pct),
                "win": bool(win),
            })

    if not rows:
        print("no trades produced", flush=True)
        return

    trades = pd.DataFrame(rows).sort_values(["t_start", "symbol"]).reset_index(drop=True)
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    trades.to_parquet(out, index=False)

    total = {
        "trades": int(len(trades)),
        "wins": int(trades["win"].sum()),
        "winrate_pct": float(100.0 * trades["win"].mean()),
        "pnl_pct_sum": float(trades["pnl_pct"].sum()),
        "symbols": int(trades["symbol"].nunique()),
    }
    with open(os.path.splitext(out)[0] + "_summary.json", "w") as f:
        json.dump(total, f, ensure_ascii=False, indent=2)

    print("=== FULLTRADE BACKTEST DONE ===", flush=True)
    print(
        f"trades={total['trades']}  winrate={total['winrate_pct']:.1f}%  "
        f"pnl_pct_sum={total['pnl_pct_sum']:.1f}  symbols={total['symbols']}",
        flush=True,
    )
    print(f"saved → {out}", flush=True)