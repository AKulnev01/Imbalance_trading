import argparse, pandas as pd, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--feats")
    ap.add_argument("--join-keys", default="entry_ts")
    ap.add_argument("--use-pnl")
    ap.add_argument("--pay-tp", type=float, default=1.0)
    ap.add_argument("--pay-sl", type=float, default=-1.0)
    ap.add_argument("--grid-from", type=float, default=0.3)
    ap.add_argument("--grid-to", type=float, default=0.95)
    ap.add_argument("--grid-n", type=int, default=20)
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--coverage-max", type=float, default=0.10, help="максимальная доля сделок")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-picks", help="путь для сохранения отобранных сделок лучшего порога")
    args = ap.parse_args()

    df = pd.read_parquet(args.scored).copy()

    if args.feats:
        join_keys = [k.strip() for k in args.join_keys.split(",") if k.strip()]
        feats = pd.read_parquet(args.feats)
        for k in join_keys:
            if k not in df.columns or k not in feats.columns:
                raise SystemExit(f"[ERR] join-key '{k}' not found in both dataframes")
        if args.use_pnl and args.use_pnl in feats.columns:
            df = df.merge(feats[join_keys + [args.use_pnl]], on=join_keys, how="left")
        else:
            raise SystemExit(f"[ERR] --use-pnl '{args.use_pnl}' not in feats")
    else:
        if "y" not in df.columns:
            raise SystemExit("[ERR] no feats and no y column for binary payoff")

    N = len(df)
    grid = np.linspace(args.grid_from, args.grid_to, args.grid_n)
    rows = []

    for thr in grid:
        pick = df[df["p"] >= thr]
        if len(pick) < args.min_trades:
            continue
        if len(pick) / N > args.coverage_max:
            continue

        if args.use_pnl and args.use_pnl in pick.columns:
            ev = pick[args.use_pnl].mean()
            hit_rate = (pick[args.use_pnl] > 0).mean()
            pnl_sum = pick[args.use_pnl].sum()
        else:
            hit_rate = pick["y"].mean()
            ev = hit_rate * args.pay_tp + (1 - hit_rate) * args.pay_sl
            pnl_sum = np.nan

        rows.append((thr, len(pick), len(pick)/N, hit_rate, pick["p"].mean(), ev, pnl_sum))

    if not rows:
        print("[WARN] порогов под ограничения не найдено")
        return

    rep = pd.DataFrame(rows, columns=["thr","n","coverage","hit_rate","p_mean","EV_per_trade","PnL_sum"])
    rep = rep.sort_values(["EV_per_trade","hit_rate","p_mean"], ascending=False)

    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out_p, index=False)

    # Кривая cum-EV (по всем сделкам, отсортированным по p)
    if args.use_pnl and args.use_pnl in df.columns:
        d_sorted = df.sort_values("p", ascending=False).reset_index(drop=True)
        d_sorted["cum_EV"] = d_sorted[args.use_pnl].cumsum()
        d_sorted["coverage"] = (np.arange(len(d_sorted)) + 1) / len(d_sorted)
        cum = d_sorted[["cum_EV", "coverage"]].reset_index().rename(columns={"index":"k"})
        cum.to_csv(out_p.with_name(out_p.stem + "_cum.csv"), index=False)
        print("\n=== CUM-EV (top) ==="); print(cum.head(10))
        print("\n=== CUM-EV (tail) ==="); print(cum.tail(3))

    # Сохраним сделки для лучшего порога (если просят)
    if args.save_picks:
        best_thr = rep.iloc[0]["thr"]
        picks = df[df["p"] >= best_thr].copy()
        Path(args.save_picks).parent.mkdir(parents=True, exist_ok=True)
        picks.to_parquet(args.save_picks, index=False)
        print(f"\n[OK] picks({best_thr:.4f}) saved -> {args.save_picks}")

    print("\n=== TOP thresholds by EV_per_trade ===")
    print(rep.head(10))
    print(f"[OK] saved: {out_p}")

if __name__ == "__main__":
    main()