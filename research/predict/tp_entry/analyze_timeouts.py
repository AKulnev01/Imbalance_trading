# predict/tp_entry/analyze_timeouts.py
import argparse, pandas as pd, numpy as np
from pathlib import Path

def bins_by_hours(max_hours=48, step=2):
    edges = list(range(0, max_hours + step, step))
    labels = [f"{edges[i]}-{edges[i+1]}h" for i in range(len(edges)-1)]
    return edges, labels

def summarize(df, use_pnl=True, pay_tp=1.0, pay_sl=-1.0, max_hours=48, step=2):
    df = df.copy()
    if "ttm_min" not in df.columns:
        raise ValueError("В feats нет колонки 'ttm_min' (нужна разметка label_entries с временем до выхода).")

    # оставим сделки, где известна длительность
    before = len(df)
    df = df[~df["ttm_min"].isna()].copy()
    after = len(df)

    df["ttm_h"] = df["ttm_min"] / 60.0

    edges, labels = bins_by_hours(max_hours, step)
    df["bin"] = pd.cut(df["ttm_h"], bins=edges, labels=labels, right=True, include_lowest=True)

    # дискретные корзины
    if df.empty:
        by_bin = pd.DataFrame({"bin": labels, "n": 0, "tp": 0, "tp_rate": np.nan})
    else:
        grp = df.groupby("bin", observed=False)
        by_bin = grp.agg(n=("y","count"), tp=("y","sum")).reset_index()
        by_bin["tp_rate"] = by_bin.apply(
            lambda r: (r["tp"]/r["n"]) if r["n"] > 0 else np.nan, axis=1
        )
        # заполнить отсутствующие корзины нулями
        full = pd.DataFrame({"bin": labels})
        by_bin = full.merge(by_bin, on="bin", how="left").fillna({"n":0, "tp":0})
        # tp_rate оставляем NaN при n=0

    # накопленные метрики "до H"
    out_rows = []
    for H in edges[1:]:
        if df.empty:
            out_rows.append((H, 0, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        sub = df[df["ttm_h"] <= H]
        if len(sub) == 0:
            out_rows.append((H, 0, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        n = int(len(sub))
        tp = int(sub["y"].sum())
        tp_rate = float(tp / n) if n > 0 else np.nan

        if use_pnl and ("pnl_net" in sub.columns) and ("ref_close" in sub.columns):
            ev = float(sub["pnl_net"].mean()) if n > 0 else np.nan
            pnl_sum = float(sub["pnl_net"].sum()) if n > 0 else np.nan
        else:
            # бинарная аппроксимация
            if np.isnan(tp_rate):
                ev = np.nan
            else:
                ev = float(tp_rate*pay_tp + (1.0 - tp_rate)*pay_sl)
            pnl_sum = np.nan

        out_rows.append((H, n, tp, tp_rate, ev, pnl_sum, float(sub["ttm_h"].max())))

    cum = pd.DataFrame(out_rows, columns=[
        "H_hours","n","tp","cum_tp_rate","EV_per_trade","PnL_sum","max_ttm_h_included"
    ])
    return by_bin, cum, before, after

def pick_timeout(cum: pd.DataFrame, ev_delta_frac=0.01):
    """Возвращает timeout_hours. Устойчив к all-NaN."""
    cum = cum.copy()

    # если есть EV — используем его
    if not cum["EV_per_trade"].dropna().empty:
        best_idx = cum["EV_per_trade"].idxmax()
        best_ev = float(cum.loc[best_idx, "EV_per_trade"])
        best_H  = float(cum.loc[best_idx, "H_hours"])
        # ранний выбор на плато
        threshold = best_ev * (1 - float(ev_delta_frac))
        near = cum[cum["EV_per_trade"] >= threshold]
        if not near.empty:
            return float(near["H_hours"].min())
        return best_H

    # иначе по cum_tp_rate
    if not cum["cum_tp_rate"].dropna().empty:
        best_idx = cum["cum_tp_rate"].idxmax()
        return float(cum.loc[best_idx, "H_hours"])

    # полный fallback: середина диапазона
    Hs = cum["H_hours"].values.tolist()
    return float(Hs[len(Hs)//2]) if len(Hs) else 24.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="*_feats.parquet (BUY или SELL)")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (entry_ts >= start)")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (entry_ts <= end)")
    ap.add_argument("--side", type=str, default=None, help="optional filter: BUY/SELL")
    ap.add_argument("--use-pnl", action="store_true", help="использовать pnl_net для EV (иначе бинарный EV)")
    ap.add_argument("--pay-tp", type=float, default=1.0)
    ap.add_argument("--pay-sl", type=float, default=-1.0)
    ap.add_argument("--max-hours", type=int, default=48)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--ev-delta-frac", type=float, default=0.01, help="1% плато для раннего выбора timeout")
    ap.add_argument("--outdir", default="reports/timeout_analysis")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.feats).copy()

    need = {"entry_ts","y","ttm_min"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"в {args.feats} нет колонок: {miss}")

    if args.side:
        df = df[df["side"].astype(str).str.upper() == args.side.upper()]
    if args.start:
        df = df[df["entry_ts"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["entry_ts"] <= pd.Timestamp(args.end)]

    # Диагностика
    total_rows = len(df)
    tp_share = float(df["y"].mean()) if total_rows else float("nan")
    print(f"[INFO] rows after filters: {total_rows} | TP_rate={tp_share:.3f} | side={args.side or 'ANY'}")

    by_bin, cum, before_ttm, after_ttm = summarize(
        df, use_pnl=args.use_pnl, pay_tp=args.pay_tp, pay_sl=args.pay_sl,
        max_hours=args.max_hours, step=args.step
    )

    if after_ttm == 0:
        print("[WARN] после фильтров нет сделок с известной длительностью (ttm_min). "
              "Проверь окно дат/сторону или повторно разметь данные с tmax_hours.")
    else:
        dropped = before_ttm - after_ttm
        print(f"[INFO] deals with ttm_min: {after_ttm} (dropped {dropped} rows without ttm_min)")

    timeout_H = pick_timeout(cum, ev_delta_frac=args.ev_delta_frac)

    sym = (df["symbol"].iloc[0] if "symbol" in df.columns and not df.empty else "SYMBOL")
    side = (df["side"].iloc[0] if "side" in df.columns and not df.empty else args.side or "BOTH")
    base = Path(args.outdir) / f"{sym}_{side}"

    by_bin.to_csv(base.with_suffix(".bins.csv"), index=False)
    cum.to_csv(base.with_suffix(".cum.csv"), index=False)

    print("=== DISCRETE by time-bin ===")
    print(by_bin.head(20).to_string(index=False))
    print("\n=== CUMULATIVE ≤ H ===")
    print(cum.head(20).to_string(index=False))
    print(f"\n[RECOMMEND] timeout_hours ≈ {timeout_H:.1f}")
    print(f"[OK] saved: {base.with_suffix('.bins.csv')} and {base.with_suffix('.cum.csv')}")

if __name__ == "__main__":
    main()