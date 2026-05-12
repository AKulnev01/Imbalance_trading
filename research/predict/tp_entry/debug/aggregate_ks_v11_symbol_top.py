import os
from pathlib import Path

import pandas as pd


SRC_DIR = Path("reports/features/dataset_ks_v11_by_symbol")
OUT_DIR = Path("reports/features/dataset_ks_v11_symbol_top")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(SRC_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"[ERR] No .parquet files found in {SRC_DIR}")
        return

    print("=== AGGREGATE KS V11 BY SYMBOL → 1 ROW PER TRADE ===")
    print(f"SRC_DIR : {SRC_DIR}")
    print(f"OUT_DIR : {OUT_DIR}")
    print(f"FILES   : {len(parquet_files)}")

    group_cols = ["symbol", "side", "entry_ts"]

    for src_path in parquet_files:
        symbol = src_path.stem
        print(f"\n[{symbol}] Load: {src_path}")

        df = pd.read_parquet(src_path)

        expected_cols = {
            "symbol",
            "side",
            "entry_ts",
            "ks_tp_scale",
            "ks_sl_scale",
            "ks_ttl_hours",
            "ks_tp_abs",
            "ks_sl_abs",
            "ks_ret_adj",
        }
        missing = expected_cols - set(df.columns)
        if missing:
            print(f"[{symbol}] ERROR: missing columns: {missing}")
            continue

        # На всякий случай приводим entry_ts к datetime
        if not pd.api.types.is_datetime64_any_dtype(df["entry_ts"]):
            df["entry_ts"] = pd.to_datetime(df["entry_ts"])

        n_rows = len(df)
        print(f"[{symbol}] rows total: {n_rows}")

        # Группа = одна реальная сделка
        grp = df.groupby(group_cols, observed=True)

        # 1) Аггрегаты по ks_ret_adj
        agg = grp["ks_ret_adj"].agg(
            ks_ret_best="max",
            ks_ret_worst="min",
            ks_ret_median="median",
            ks_ret_mean="mean",
            n_ks="count",
        ).reset_index()

        # 2) Лучшие параметры (argmax ks_ret_adj)
        idx_best = grp["ks_ret_adj"].idxmax()
        df_best = df.loc[
            idx_best,
            group_cols
            + [
                "ks_ret_adj",
                "ks_tp_scale",
                "ks_sl_scale",
                "ks_ttl_hours",
                "ks_tp_abs",
                "ks_sl_abs",
            ],
        ].copy()

        df_best = df_best.rename(
            columns={
                "ks_ret_adj": "ks_ret_best",
                "ks_tp_scale": "ks_tp_scale_best",
                "ks_sl_scale": "ks_sl_scale_best",
                "ks_ttl_hours": "ks_ttl_hours_best",
                "ks_tp_abs": "ks_tp_abs_best",
                "ks_sl_abs": "ks_sl_abs_best",
            }
        )

        # 3) Джойним аггрегаты и лучшие параметры
        df_out = pd.merge(
            agg,
            df_best,
            on=group_cols + ["ks_ret_best"],
            how="left",
            validate="one_to_one",
        )

        # 4) Диапазон между лучшим и худшим исходом по сетке
        df_out["ks_ret_range"] = df_out["ks_ret_best"] - df_out["ks_ret_worst"]

        # Сортировка по времени входа — удобно для time-split
        df_out = df_out.sort_values("entry_ts").reset_index(drop=True)

        n_trades = len(df_out)
        mean_n_ks = df_out["n_ks"].mean()
        print(
            f"[{symbol}] trades: {n_trades} | mean n_ks per trade: {mean_n_ks:.1f}"
        )

        dst_path = OUT_DIR / f"{symbol}.parquet"
        df_out.to_parquet(dst_path, index=False)
        print(f"[{symbol}] SAVED → {dst_path}")

    print("\n=== DONE: AGGREGATE KS V11 BY SYMBOL ===")


if __name__ == "__main__":
    main()