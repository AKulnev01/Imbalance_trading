#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATA_PATH = Path("reports/features/dataset_ks_v11_symbol_split/dataset_ks_v11_full.parquet")

OUT_DIR = Path("reports/features/debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = OUT_DIR / "ks_sensitivity_sample_1000.parquet"
OUT_CSV = OUT_DIR / "ks_sensitivity_sample_1000.csv"

N_SAMPLE_TRADES = 1000  # сколько разных (symbol, side, entry_ts)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print("=== KS sensitivity sample ===")
    print("DATA:", DATA_PATH)

    pf = pq.ParquetFile(DATA_PATH)

    # ---------- шаг 1: читаем только ключи трейдов ----------
    print("[STEP 1] Read trade keys (symbol, side, entry_ts)...")
    table_keys = pf.read(columns=["symbol", "side", "entry_ts"])
    df_keys = table_keys.to_pandas()
    print("  total rows:", len(df_keys))

    df_trades = df_keys.drop_duplicates(subset=["symbol", "side", "entry_ts"])
    n_trades = len(df_trades)
    print("  unique trades:", n_trades)

    n_sample = min(N_SAMPLE_TRADES, n_trades)
    print("  sampling trades:", n_sample)

    df_sample_trades = df_trades.sample(n=n_sample, random_state=42).reset_index(drop=True)

    # ---------- шаг 2: читаем только нужные колонки для ks ----------
    cols_needed = [
        "symbol",
        "side",
        "entry_ts",
        "ks_ret_adj",
        "ks_tp_scale",
        "ks_sl_scale",
        "ks_ttl_hours",
        "ks_tp_abs",
        "ks_sl_abs",
    ]

    print("[STEP 2] Read KS columns for all rows...")
    table_all = pf.read(columns=cols_needed)
    df_all = table_all.to_pandas()
    print("  df_all shape:", df_all.shape)

    # ---------- шаг 3: отфильтровываем только выбранные трейды ----------
    print("[STEP 3] Filter rows to sampled trades...")
    df_sub = df_all.merge(
        df_sample_trades,
        on=["symbol", "side", "entry_ts"],
        how="inner",
    )
    print("  df_sub shape (only sampled trades):", df_sub.shape)

    # ---------- шаг 4: группировка и вычисление метрик ----------
    print("[STEP 4] Group by trade and compute stats...")

    rows_out = []
    grouped = df_sub.groupby(["symbol", "side", "entry_ts"], sort=False)

    for (sym, side, entry_ts), g in grouped:
        # защита: если по какой-то причине меньше 2 строк — пропустим
        if len(g) == 0:
            continue

        # best/worst
        idx_best = g["ks_ret_adj"].idxmax()
        idx_worst = g["ks_ret_adj"].idxmin()

        best = g.loc[idx_best]
        worst = g.loc[idx_worst]

        ks_ret_best = float(best["ks_ret_adj"])
        ks_ret_worst = float(worst["ks_ret_adj"])

        ks_ret_median = float(g["ks_ret_adj"].median())
        ks_ret_mean = float(g["ks_ret_adj"].mean())

        # baseline = медиана по ks для сделки
        baseline = ks_ret_median

        q05 = float(g["ks_ret_adj"].quantile(0.05))
        q95 = float(g["ks_ret_adj"].quantile(0.95))
        q25 = float(g["ks_ret_adj"].quantile(0.25))
        q75 = float(g["ks_ret_adj"].quantile(0.75))
        iqr = q75 - q25
        std = float(g["ks_ret_adj"].std(ddof=0))

        rows_out.append(
            {
                "symbol": sym,
                "side": side,
                "entry_ts": entry_ts,

                "ks_ret_best": ks_ret_best,
                "ks_tp_scale_best": float(best["ks_tp_scale"]),
                "ks_sl_scale_best": float(best["ks_sl_scale"]),
                "ks_ttl_hours_best": int(best["ks_ttl_hours"]),
                "ks_tp_abs_best": float(best["ks_tp_abs"]),
                "ks_sl_abs_best": float(best["ks_sl_abs"]),

                "ks_ret_worst": ks_ret_worst,
                "ks_tp_scale_worst": float(worst["ks_tp_scale"]),
                "ks_sl_scale_worst": float(worst["ks_sl_scale"]),
                "ks_ttl_hours_worst": int(worst["ks_ttl_hours"]),
                "ks_tp_abs_worst": float(worst["ks_tp_abs"]),
                "ks_sl_abs_worst": float(worst["ks_sl_abs"]),

                "ks_ret_median": ks_ret_median,
                "ks_ret_mean": ks_ret_mean,

                "best_minus_median": ks_ret_best - ks_ret_median,
                "best_minus_mean": ks_ret_best - ks_ret_mean,
                "range_best_minus_worst": ks_ret_best - ks_ret_worst,

                "q05": q05,
                "q25": q25,
                "q75": q75,
                "q95": q95,
                "iqr": iqr,
                "std": std,
                "n_grid": int(len(g)),
            }
        )

    df_out = pd.DataFrame(rows_out)
    print("[STEP 5] Result trades:", len(df_out))

    print("  global mean ks_ret_best          :", df_out["ks_ret_best"].mean())
    print("  global mean ks_ret_median        :", df_out["ks_ret_median"].mean())
    print("  global mean best_minus_median    :", df_out["best_minus_median"].mean())
    print("  global mean range_best_minus_worst:", df_out["range_best_minus_worst"].mean())

    print("[SAVE] →", OUT_PARQUET)
    df_out.to_parquet(OUT_PARQUET, index=False)

    print("[SAVE] →", OUT_CSV)
    df_out.to_csv(OUT_CSV, index=False)

    print("=== DONE ks_sensitivity_sample ===")


if __name__ == "__main__":
    main()