import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_one(path: str, out_dir: str):
    df = pd.read_parquet(path)
    if df.empty:
        print(f"[SKIP] {path}: empty file")
        return None

    # --- Автоматический поиск основных колонок ---
    col_k_tp = next((c for c in ["k_tp", "tp_k", "tp_mult"] if c in df.columns), None)
    col_k_sl = next((c for c in ["k_sl", "sl_k", "sl_mult"] if c in df.columns), None)
    col_u = next((c for c in ["u_mean", "u", "utility", "u_sum", "best_u", "utility_val"] if c in df.columns), None)

    # Число сделок и попаданий в TP/SL
    col_n = next((c for c in ["n", "n_total", "total_trades", "count", "n_val"] if c in df.columns), None)
    col_tp_hits = next((c for c in ["n_tp", "tp_hits", "tp_count"] if c in df.columns), None)
    col_sl_hits = next((c for c in ["n_sl", "sl_hits", "sl_count"] if c in df.columns), None)

    if not all([col_k_tp, col_k_sl, col_u]):
        print(f"[WARN] {path}: missing k_tp/k_sl/u")
        return None

    # --- Расчёт derived метрик ---
    df["k_tp"] = df[col_k_tp].astype(float)
    df["k_sl"] = df[col_k_sl].astype(float)
    df["utility"] = df[col_u].astype(float)

    if col_tp_hits and col_n in df and (df[col_n] > 0).any():
        df["tp_rate"] = df[col_tp_hits] / df[col_n]
    else:
        df["tp_rate"] = np.nan

    if col_sl_hits and col_n in df and (df[col_n] > 0).any():
        df["sl_rate"] = df[col_sl_hits] / df[col_n]
    else:
        df["sl_rate"] = np.nan

    df["rr_ratio"] = df["tp_rate"] / (df["sl_rate"] + 1e-6)
    df["score"] = df["utility"] * (df["tp_rate"] - df["sl_rate"]).fillna(0)

    # --- Топ-10 и лучший результат ---
    top = df.sort_values("score", ascending=False).head(10)
    best = top.iloc[0]

    print(f"\n=== {os.path.basename(path)} ===")
    print(top[["k_tp", "k_sl", "tp_rate", "sl_rate", "utility", "score"]].round(4))

    # --- Визуализация поверхности ---
    if len(df["k_tp"].unique()) > 2 and len(df["k_sl"].unique()) > 2:
        plt.figure(figsize=(6, 5))
        pivot = df.pivot_table(index="k_sl", columns="k_tp", values="score", aggfunc="mean")
        plt.imshow(
            pivot, origin="lower", aspect="auto", cmap="magma",
            extent=[
                pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max()
            ]
        )
        plt.colorbar(label="score")
        plt.xlabel("k_tp")
        plt.ylabel("k_sl")
        plt.title(os.path.basename(path).replace("_rand.parquet", ""))
        out_png = Path(out_dir) / (Path(path).stem + "_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    # --- Сохранение топ-10 ---
    out_csv = Path(out_dir) / (Path(path).stem + "_top10.csv")
    top.to_csv(out_csv, index=False)

    # --- Возврат итогового best ---
    sym, side = Path(path).stem.replace("_rand", "").split("_")
    return {
        "symbol": sym,
        "side": side,
        "k_tp": best["k_tp"],
        "k_sl": best["k_sl"],
        "tp_rate": best["tp_rate"],
        "sl_rate": best["sl_rate"],
        "utility": best["utility"],
        "score": best["score"],
    }

def main():
    in_dir = "./reports/tp_opt_rand"
    out_dir = "./reports/tp_opt_analysis"
    os.makedirs(out_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(in_dir, "*_rand.parquet")))
    results = []
    for f in all_files:
        r = analyze_one(f, out_dir)
        if r is not None:
            results.append(r)

    if results:
        bests = pd.DataFrame(results)
        bests = bests.sort_values("score", ascending=False).reset_index(drop=True)

        best_csv = Path(out_dir) / "best_ks.csv"
        bests.to_csv(best_csv, index=False)

        print(f"\n[OK] Saved best per symbol → {best_csv}")
        print(bests.head(20).round(4))
    else:
        print("[WARN] No valid results processed.")

if __name__ == "__main__":
    main()