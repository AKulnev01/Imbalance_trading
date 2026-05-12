import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def load_quality_files(in_dir="./reports/tp_opt_rand"):
    paths = sorted(glob.glob(os.path.join(in_dir, "*_rand.parquet")))
    dfs = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[ERR] {p}: {e}")
            continue

        if df.empty:
            continue

        # находим подходящие колонки
        col_u = next((c for c in ["utility", "u", "u_mean", "utility_val"] if c in df.columns), None)
        if not col_u or "k_tp" not in df or "k_sl" not in df:
            print(f"[WARN] {p}: нет нужных колонок (k_tp/k_sl/utility)")
            continue

        df = df[["k_tp", "k_sl", col_u]].copy()
        df = df.rename(columns={col_u: "utility"})
        df["utility"] = df["utility"].astype(float)

        # вычисляем z-score и медиану
        med = df["utility"].median()
        std = df["utility"].std(ddof=0) or 1.0
        df["zscore"] = (df["utility"] - med) / std

        df["symbol"], df["side"] = Path(p).stem.replace("_rand", "").split("_")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def synthesize_signals(df, n_per_symbol=300):
    out_rows = []
    now = datetime.now()

    for (sym, side), sub in df.groupby(["symbol", "side"]):
        # берём только лучшие 50% по utility
        good = sub[sub["utility"] > sub["utility"].median()]
        if good.empty:
            continue

        good = good.sample(min(len(good), n_per_symbol), random_state=42)
        base_ts = now - timedelta(hours=len(good) * 4)

        for i, row in enumerate(good.itertuples(index=False)):
            bar_ts = base_ts + timedelta(hours=i * 4)
            detect_px = round(0.1 + np.random.random() * 0.05, 6)
            out_rows.append({
                "symbol": sym,
                "side": side,
                "bar_ts": bar_ts,
                "detect_px": detect_px,
                "k_tp": row.k_tp,
                "k_sl": row.k_sl,
                "utility": row.utility,
                "zscore": row.zscore,
            })

    return pd.DataFrame(out_rows)

def main():
    in_dir = "./reports/tp_opt_rand"
    out_path = "./reports/signals_from_quality.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = load_quality_files(in_dir)
    if df.empty:
        print("[ERR] Не найдено данных в tp_opt_rand/")
        return

    signals = synthesize_signals(df)
    signals.to_parquet(out_path, index=False)
    print(f"[OK] Сохранено {len(signals)} сигналов → {out_path}")

if __name__ == "__main__":
    main()