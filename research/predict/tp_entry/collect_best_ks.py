import os, glob
import pandas as pd
from pathlib import Path
import numpy as np

def best_from_parquet(p):
    df = pd.read_parquet(p)
    df = df.sort_values("utility_val", ascending=False)
    top = df.iloc[0]
    return float(top["k_tp"]), float(top["k_sl"]), float(top["utility_val"]), int(top["n_val"])

def main():
    reports_dir = "./reports/tp_opt_rand"
    rows = []
    for f in glob.glob(os.path.join(reports_dir, "*_rand.parquet")):
        fn = Path(f).name  # e.g., BTCUSDT_BUY_rand.parquet
        try:
            sym, side, _ = fn.split("_")
        except ValueError:
            continue
        k_tp, k_sl, u, n = best_from_parquet(f)
        rows.append({"symbol": sym, "side": side, "k_tp": k_tp, "k_sl": k_sl, "utility": u, "n": n, "file": fn})

    out = pd.DataFrame(rows)
    out.sort_values(["symbol","side"], inplace=True)
    os.makedirs("./reports/summary", exist_ok=True)
    out_path = "./reports/summary/best_ks.csv"
    out.to_csv(out_path, index=False)

    # агрегаты по сторонам — robust baseline
    summary = out.groupby("side").apply(
        lambda g: pd.Series({
            "k_tp_median": np.median(g["k_tp"]),
            "k_sl_median": np.median(g["k_sl"]),
            "k_tp_wmed": np.average(g["k_tp"], weights=np.clip(g["n"],1,None)),
            "k_sl_wmed": np.average(g["k_sl"], weights=np.clip(g["n"],1,None)),
            "symbols": g["symbol"].nunique()
        })
    ).reset_index()
    summary.to_csv("./reports/summary/best_ks_side_stats.csv", index=False)

    print(out_path)
    print("./reports/summary/best_ks_side_stats.csv")

if __name__ == "__main__":
    main()