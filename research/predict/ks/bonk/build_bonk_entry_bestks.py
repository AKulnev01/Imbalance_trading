from pathlib import Path
import numpy as np
import pandas as pd

KS_DF_PATH = Path("reports/features/bonk/bonk_v15_weighted_q.parquet")
OUT_PATH   = Path("reports/features/bonk/bonk_v15_entry_bestks.parquet")

TP_BASE = 0.13
SL_BASE = 0.04

PNL_GOOD = 3.0
PNL_BAD  = -3.0


def main():
    print("[LOAD KS DF]", KS_DF_PATH)
    df = pd.read_parquet(KS_DF_PATH)

    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df = df.sort_values(
        ["entry_ts", "side", "ks_tp_mult", "ks_sl_mult"]
    ).reset_index(drop=True)

    print("[INFO] rows total:", len(df))

    # --- best KS по среднему pnl_net за весь период ---
    grid = (
        df.groupby(["side", "ks_tp_mult", "ks_sl_mult"], as_index=False)["pnl_net"]
          .mean()
    )
    best_rows = (
        grid.sort_values("pnl_net", ascending=False)
            .groupby("side", as_index=False)
            .first()
    )

    best_ks = {}
    print("\n[STATIC BEST_KS by side]:")
    for _, row in best_rows.iterrows():
        side = int(row["side"])
        tp_mult = float(row["ks_tp_mult"])
        sl_mult = float(row["ks_sl_mult"])
        best_ks[side] = (tp_mult, sl_mult)
        rr = (TP_BASE * tp_mult) / (SL_BASE * sl_mult)
        print(f"  side={side:+d}: tp_mult={tp_mult:.4f}, sl_mult={sl_mult:.4f}, RR≈{rr:.3f}")

    # --- берём только строки с этими KS ---
    rows = []
    for side, (tp, sl) in best_ks.items():
        mask = (
            (df["side"] == side)
            & (df["ks_tp_mult"] == tp)
            & (df["ks_sl_mult"] == sl)
        )
        sub = df.loc[mask].copy()
        rows.append(sub)

    best_df = pd.concat(rows, axis=0)
    best_df = best_df.sort_values(["entry_ts", "side"]).reset_index(drop=True)

    pnl = best_df["pnl_net"].astype(float)

    # --- таргет ---
    label = np.full(len(best_df), np.nan)
    label[pnl >= PNL_GOOD] = 1
    label[pnl <= PNL_BAD]  = 0

    best_df["label"] = label

    print("\n[INFO] label counts (incl. NaN):")
    print(best_df["label"].value_counts(dropna=False))

    ds = best_df.dropna(subset=["label"]).copy()
    ds["label"] = ds["label"].astype(int)

    print("[INFO] bars kept after label filter:", len(ds))
    print(ds["label"].value_counts(normalize=True))

    # --- фичи: только числовые, без служебных/KS и без label ---
    numeric_cols = ds.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "pnl_net",
        "ks_tp_mult",
        "ks_sl_mult",
        "ks_ttl_hours",
        "sample_weight",
        "is_focus",
        "label",
    }
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # формируем список колонок без дублей
    cols = ["entry_ts", "side"] + feature_cols + ["label"]
    cols = list(dict.fromkeys(cols))  # удаляем повторы, порядок сохраняем

    entry_ds = ds[cols].copy()

    print("\n[INFO] final entry dataset:")
    print("rows:", len(entry_ds))
    print("num features:", len(feature_cols))
    print("label distribution:")
    print(entry_ds["label"].value_counts(normalize=True))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry_ds.to_parquet(OUT_PATH)
    print("\nSaved ->", OUT_PATH)


if __name__ == "__main__":
    main()