#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np

# путь к большому датасету KS
KS_DATA_PATH = Path("reports/features/dataset_ks_v11_symbol_split/dataset_ks_v11_full.parquet")

SYMBOL = "BTCUSDT"
DATE_FROM = "2024-01-01"


def find_first_trade_key(pf: pq.ParquetFile, symbol: str, date_from: str):
    """
    Стримингово ищем ПЕРВУЮ сделку по SYMBOL после DATE_FROM.
    Возвращаем (symbol, side, entry_ts) или бросаем RuntimeError.
    """
    ts_from = pd.Timestamp(date_from, tz="UTC")

    print(f"[SEARCH] symbol={symbol}, entry_ts >= {ts_from}")

    first_row = None

    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(
            rg_idx,
            columns=["entry_ts", "symbol", "side"]
        )
        df_rg = tbl.to_pandas()

        # нормализуем типы
        df_rg["entry_ts"] = pd.to_datetime(df_rg["entry_ts"], utc=True)
        mask = (df_rg["symbol"] == symbol) & (df_rg["entry_ts"] >= ts_from)
        sub = df_rg[mask]

        if sub.empty:
            continue

        # в этом row group есть подходящие строки — берём минимальный entry_ts
        sub_sorted = sub.sort_values("entry_ts")
        cand = sub_sorted.iloc[0]

        if first_row is None or cand["entry_ts"] < first_row["entry_ts"]:
            first_row = cand

    if first_row is None:
        raise RuntimeError("No trades found for given SYMBOL/DATE_FROM")

    key = (
        str(first_row["symbol"]),
        str(first_row["side"]),
        pd.Timestamp(first_row["entry_ts"]).tz_convert("UTC"),
    )

    print("\n[FOUND TRADE KEY]")
    print("  symbol   :", key[0])
    print("  side     :", key[1])
    print("  entry_ts :", key[2])

    return key


def collect_trade_variants(pf: pq.ParquetFile, key):
    """
    Вторым проходом собираем все строки с этим (symbol, side, entry_ts)
    по всем row group’ам.
    """
    sym_key, side_key, ts_key = key

    dfs = []
    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx)
        df_rg = tbl.to_pandas()
        df_rg["entry_ts"] = pd.to_datetime(df_rg["entry_ts"], utc=True)

        mask = (
            (df_rg["symbol"] == sym_key) &
            (df_rg["side"] == side_key) &
            (df_rg["entry_ts"] == ts_key)
        )
        sub = df_rg[mask]
        if not sub.empty:
            dfs.append(sub)

    if not dfs:
        raise RuntimeError("Trade key found, but no rows collected on second pass")

    g = pd.concat(dfs, ignore_index=True)
    print(f"\n[COLLECT] rows for this trade: {len(g)}")
    return g


def main():
    if not KS_DATA_PATH.exists():
        raise FileNotFoundError(f"KS dataset not found: {KS_DATA_PATH}")

    print(f"[LOAD ARROW] {KS_DATA_PATH}")
    pf = pq.ParquetFile(KS_DATA_PATH)

    # 1) находим одну сделку-ключ
    key = find_first_trade_key(pf, SYMBOL, DATE_FROM)

    # 2) собираем все её варианты
    g = collect_trade_variants(pf, key)

    # убеждаемся, что необходимые колонки есть
    need_cols = [
        "ks_tp_scale", "ks_sl_scale", "ks_ttl_hours",
        "ks_tp_abs", "ks_sl_abs", "ks_ret_adj",
    ]
    missing = [c for c in need_cols if c not in g.columns]
    if missing:
        raise RuntimeError(f"Missing columns in trade slice: {missing}")

    # проверка сетки
    uniq_tp = np.sort(g["ks_tp_scale"].unique())
    uniq_sl = np.sort(g["ks_sl_scale"].unique())
    uniq_ttl = np.sort(g["ks_ttl_hours"].unique())

    print("\n=== KS GRID SHAPES FOR THIS TRADE ===")
    print("unique ks_tp_scale :", uniq_tp, " ->", len(uniq_tp))
    print("unique ks_sl_scale :", uniq_sl, " ->", len(uniq_sl))
    print("unique ks_ttl_hours:", uniq_ttl, " ->", len(uniq_ttl))
    print("expected combos    :", len(uniq_tp) * len(uniq_sl) * len(uniq_ttl))
    print("actual rows        :", len(g))

    # базовая статистика по ks_ret_adj
    print("\n=== ks_ret_adj STATS FOR THIS TRADE ===")
    print(g["ks_ret_adj"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    print("\n=== TOP 5 KS COMBOS (max ks_ret_adj) ===")
    top5 = g.sort_values("ks_ret_adj", ascending=False).head(5)
    print(top5[[
        "ks_tp_scale", "ks_sl_scale", "ks_ttl_hours",
        "ks_tp_abs", "ks_sl_abs", "ks_ret_adj"
    ]])

    print("\n=== BOTTOM 5 KS COMBOS (min ks_ret_adj) ===")
    bot5 = g.sort_values("ks_ret_adj", ascending=True).head(5)
    print(bot5[[
        "ks_tp_scale", "ks_sl_scale", "ks_ttl_hours",
        "ks_tp_abs", "ks_sl_abs", "ks_ret_adj"
    ]])

    # дубликаты по сетке
    dup_mask = g.duplicated(subset=["ks_tp_scale", "ks_sl_scale", "ks_ttl_hours"])
    n_dup = int(dup_mask.sum())
    print("\n=== DUPLICATE CHECK ===")
    print("duplicates by (tp_scale, sl_scale, ttl_hours):", n_dup)

    if n_dup:
        print(g[dup_mask][[
            "ks_tp_scale", "ks_sl_scale", "ks_ttl_hours", "ks_ret_adj"
        ]])

    print("\n=== DONE check_one_ks_trade_light ===")


if __name__ == "__main__":
    main()