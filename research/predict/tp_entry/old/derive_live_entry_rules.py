# predict/tp_entry/derive_live_entry_rules.py
import os, sys, json, argparse
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FEATURES = [
    "atr14","vol_z","body_ratio","upper_wick_ratio","lower_wick_ratio",
    "ema_diff_pct","rsi14"
]

def ensure_dt(df, cols=("time_open",)):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_localize(None)
    return df

def load_trades(path):
    df = pd.read_parquet(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    # нормализуем столбцы
    for c in ["symbol"]:
        if c in df.columns: df[c] = df[c].astype(str)
    ensure_dt(df, ["time_open","t_start","exit_time"])
    return df

def load_base(path):
    base = pd.read_parquet(path)
    for c in ["symbol"]:
        if c in base.columns: base[c] = base[c].astype(str)
    ensure_dt(base, ["time_open","time_close"])
    return base

def summarize(df):
    if len(df)==0:
        return {"n":0,"winrate_pct":0.0,"pnl_mean":0.0,"pnl_sum":0.0}
    return {
        "n": int(len(df)),
        "winrate_pct": float(100.0*df["win"].mean()) if "win" in df.columns else float("nan"),
        "pnl_mean": float(df["pnl_pct"].mean()) if "pnl_pct" in df.columns else float("nan"),
        "pnl_sum": float(df["pnl_pct"].sum()) if "pnl_pct" in df.columns else float("nan"),
    }

def main():
    ap = argparse.ArgumentParser(description="Derive live entry rules from profitable trades and export JSON config + offline validation.")
    ap.add_argument("--trades", required=True, help="parquet with trades (fulltrade / strong / seq_equity_*_trades.parquet)")
    ap.add_argument("--base", default="./predict/tp_entry/tp_training_base_fullrecalc.parquet")
    ap.add_argument("--min-pnl", type=float, default=8.0, help="минимальный pnl_pct сделки, чтобы попасть в эталон")
    ap.add_argument("--win-only", type=int, default=1, help="использовать только win-сделки (1/0)")
    ap.add_argument("--q-strong", type=float, default=0.60, help="квантиль для нижних порогов (vol_z, body_ratio, rsi)")
    ap.add_argument("--q-upper-max", type=float, default=0.40, help="квантиль для верхнего фитиля (макс)")
    ap.add_argument("--q-lower-max", type=float, default=0.40, help="квантиль для нижнего фитиля (макс)")
    ap.add_argument("--q-atr-max", type=float, default=0.70, help="квантиль для atr14 (макс, т.к. на сильных он был ниже)")
    ap.add_argument("--tp-q", type=float, default=0.50, help="квантиль для tp_pct_pred (мин-порог)")
    ap.add_argument("--sl-q", type=float, default=0.50, help="квантиль для sl_pct_pred (макс-порог)")
    ap.add_argument("--out-json", default="./predict/tp_entry/live_rules.json")
    ap.add_argument("--out-md", default="./predict/tp_entry/live_rules_summary.md")
    args = ap.parse_args()

    trades = load_trades(args.trades)
    base = load_base(args.base)

    # мерджим признаки бара
    keep_cols = ["symbol","time_open","tp_pct_pred","sl_pct_pred","pnl_pct","win"] + FEATURES
    merged = trades.merge(base[["symbol","time_open"] + [c for c in FEATURES if c in base.columns]],
                          on=["symbol","time_open"], how="left")

    # отберём эталонные сделки
    flt = pd.Series(True, index=merged.index)
    if args.win_only:
        flt &= (merged.get("win", True) == True)
    if "pnl_pct" in merged.columns:
        flt &= (merged["pnl_pct"] >= float(args.min_pnl))
    strong = merged[flt].copy()

    base_summary = summarize(merged)
    strong_summary = summarize(strong)

    # посчитаем пороги по фичам из strong
    def q(df, col, qv, default=None):
        if col not in df.columns or df[col].dropna().empty:
            return default
        return float(df[col].quantile(qv))

    rules = {
        "version": 1,
        "source_trades": os.path.abspath(args.trades),
        "criteria": {
            "vol_z_min": q(strong, "vol_z", args.q-strong, 0.0),
            "body_ratio_min": q(strong, "body_ratio", args.q-strong, 0.0),
            "rsi14_min": q(strong, "rsi14", args.q-strong, 0.0),
            "upper_wick_ratio_max": q(strong, "upper_wick_ratio", args.q_upper_max, 1.0),
            "lower_wick_ratio_max": q(strong, "lower_wick_ratio", args.q_lower_max, 1.0),
            "atr14_max": q(strong, "atr14", args.q_atr_max, 999.0),
            # ema_diff_pct — слабый признак, но можно ограничить «слишком бычьи/медвежьи» шумы:
            "ema_diff_min": None,  # оставим None — не включаем по умолчанию
            "ema_diff_max": None
        },
        "tp_sl_pred_thresholds": {
            "tp_pct_pred_min": q(strong, "tp_pct_pred", args.tp_q, 0.02),
            "sl_pct_pred_max": q(strong, "sl_pct_pred", args.sl_q, 0.02)
        },
        "selection_stats": {
            "base": base_summary,
            "strong": strong_summary
        }
    }

    # оффлайн-валидация: применим правила ко ВСЕМ сделкам trades (на мердженных признаках)
    def passes(row):
        c = rules["criteria"]
        t = rules["tp_sl_pred_thresholds"]

        if pd.isna(row.get("vol_z")) or row["vol_z"] < c["vol_z_min"]: return False
        if pd.isna(row.get("body_ratio")) or row["body_ratio"] < c["body_ratio_min"]: return False
        if pd.isna(row.get("rsi14")) or row["rsi14"] < c["rsi14_min"]: return False
        if pd.isna(row.get("upper_wick_ratio")) or row["upper_wick_ratio"] > c["upper_wick_ratio_max"]: return False
        if pd.isna(row.get("lower_wick_ratio")) or row["lower_wick_ratio"] > c["lower_wick_ratio_max"]: return False
        if pd.isna(row.get("atr14")) or row["atr14"] > c["atr14_max"]: return False

        # tp/sl пороги, если есть предсказания в трейдах
        if "tp_pct_pred" in row and pd.notna(row["tp_pct_pred"]):
            if row["tp_pct_pred"] < t["tp_pct_pred_min"]: return False
        if "sl_pct_pred" in row and pd.notna(row["sl_pct_pred"]):
            if row["sl_pct_pred"] > t["sl_pct_pred_max"]: return False

        # ema_diff опционален
        if c["ema_diff_min"] is not None and row.get("ema_diff_pct", 0) < c["ema_diff_min"]:
            return False
        if c["ema_diff_max"] is not None and row.get("ema_diff_pct", 0) > c["ema_diff_max"]:
            return False
        return True

    sel = merged.dropna(subset=["symbol","time_open"]).copy()
    sel["selected"] = sel.apply(passes, axis=1)
    picked = sel[sel["selected"]].copy()
    picked_summary = summarize(picked)

    # список символов, где выборка достаточна и прибыль положительная
    bysym = picked.groupby("symbol").agg(
        n=("pnl_pct","size"),
        winrate=("win","mean"),
        pnl_sum=("pnl_pct","sum")
    ).reset_index()
    universe = bysym[(bysym["n"]>=20) & (bysym["winrate"]>=0.6) & (bysym["pnl_sum"]>0)]["symbol"].tolist()

    export = {
        "rules": rules,
        "validation": {
            "picked": picked_summary,
            "bysymbol_top": bysym.sort_values("pnl_sum", ascending=False).head(50).to_dict(orient="records"),
            "universe": universe
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    # короткое резюме в MD
    with open(args.out_md, "w") as f:
        f.write("# Live entry rules (auto-derived)\n\n")
        f.write(f"**source trades:** `{os.path.abspath(args.trades)}`\n\n")
        f.write("## Criteria\n")
        for k,v in rules["criteria"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## TP/SL predicted thresholds\n")
        for k,v in rules["tp_sl_pred_thresholds"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Selection\n")
        f.write(f"- base: {base_summary}\n")
        f.write(f"- strong: {strong_summary}\n")
        f.write(f"- picked_by_rules: {picked_summary}\n")
        f.write("\n## Universe (symbols)\n")
        for s in universe: f.write(f"- {s}\n")

    print("— EXPORT —")
    print("Rules JSON →", args.out_json)
    print("Summary MD →", args.out_md)
    print("Picked:", picked_summary)
    print("Universe size:", len(universe))