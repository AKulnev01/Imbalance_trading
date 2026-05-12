# predict/tp_entry/build_tp_labels.py
import os, argparse
import pandas as pd

from .label_triple_barrier import label_entries
from .data_utils import load_m1, make_4h

def _best_pair(path_rand: str, path_grid: str):
    p = path_rand if os.path.exists(path_rand) else path_grid
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p).sort_values("utility_val", ascending=False)
    if df.empty: return None
    r = df.iloc[0]
    return float(r["k_tp"]), float(r["k_sl"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, required=True)
    ap.add_argument("--m1-dir", type=str, default="./data/m1")
    ap.add_argument("--opt-dir", type=str, default="./reports/tp_opt_rand", help="где лежат *_rand.parquet; fallbacks: *_grid.parquet в той же папке")
    ap.add_argument("--out", type=str, default="./data/tp_labels.parquet")
    ap.add_argument("--tmax-hours", type=int, default=80)
    ap.add_argument("--fee-pct", type=float, default=0.001)
    ap.add_argument("--slip-exit-pct", type=float, default=0.004)
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    all_rows = []
    for s in syms:
        m1 = load_m1(s, args.m1_dir)
        if m1.empty:
            print(f"[SKIP] {s}: no m1");
            continue
        h4 = make_4h(m1).dropna(subset=["atr14"])
        if h4.empty:
            print(f"[SKIP] {s}: no 4h/atr");
            continue

        p_buy_rand  = os.path.join(args.opt_dir, f"{s}_BUY_rand.parquet")
        p_sell_rand = os.path.join(args.opt_dir, f"{s}_SELL_rand.parquet")
        p_buy_grid  = os.path.join(args.opt_dir, f"{s}_BUY_grid.parquet")
        p_sell_grid = os.path.join(args.opt_dir, f"{s}_SELL_grid.parquet")

        best_buy  = _best_pair(p_buy_rand,  p_buy_grid)
        best_sell = _best_pair(p_sell_rand, p_sell_grid)

        parts = []
        if best_buy is not None:
            k_tp, k_sl = best_buy
            lab = label_entries(m1, h4.assign(side=+1), "side", k_tp=k_tp, k_sl=k_sl,
                                tmax_hours=args.tmax_hours, fee_pct=args.fee_pct, slip_exit_pct=args.slip_exit_pct,
                                atr_col="atr14", atr_n=14)
            if not lab.empty:
                lab["symbol"] = s; lab["side"] = +1; lab["k_tp"] = k_tp; lab["k_sl"] = k_sl
                lab["label_tp_first"] = (lab["reason"]=="tp").astype(int)
                lab["label_exit"] = lab["reason"].map({"tp":1,"sl":0,"timeout":-1}).astype(int)
                parts.append(lab.reset_index())

        if best_sell is not None:
            k_tp, k_sl = best_sell
            lab = label_entries(m1, h4.assign(side=-1), "side", k_tp=k_tp, k_sl=k_sl,
                                tmax_hours=args.tmax_hours, fee_pct=args.fee_pct, slip_exit_pct=args.slip_exit_pct,
                                atr_col="atr14", atr_n=14)
            if not lab.empty:
                lab["symbol"] = s; lab["side"] = -1; lab["k_tp"] = k_tp; lab["k_sl"] = k_sl
                lab["label_tp_first"] = (lab["reason"]=="tp").astype(int)
                lab["label_exit"] = lab["reason"].map({"tp":1,"sl":0,"timeout":-1}).astype(int)
                parts.append(lab.reset_index())

        if parts:
            all_rows.append(pd.concat(parts, ignore_index=True))
            print(f"[OK] {s}: labels {sum(len(x) for x in parts):,}")
        else:
            print(f"[SKIP] {s}: no best pairs or empty labels")

    if not all_rows:
        print("[DONE] nothing to save"); return

    out = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[DONE] labels → {args.out} rows={len(out):,}")

if __name__ == "__main__":
    main()