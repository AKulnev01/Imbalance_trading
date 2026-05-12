from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRanker

DATA_PATH = Path("reports/features/bonk/bonk_v15_weighted_q.parquet")
MODEL_RANK_PATH = Path("models/bonk_v15_rank_q_gpu.cbm")

TP_BASE = 0.13
SL_BASE = 0.04


def load_data():
    print("[LOAD]", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df = df.sort_values(["entry_ts", "side", "ks_tp_mult", "ks_sl_mult"]).reset_index(drop=True)

    bars = (
        df[["entry_ts", "side"]]
        .drop_duplicates()
        .sort_values(["entry_ts", "side"])
        .reset_index(drop=True)
    )
    bars["group_id"] = np.arange(len(bars), dtype=np.int64)
    df = df.merge(bars, on=["entry_ts", "side"], how="left")

    if df["group_id"].isna().any():
        raise SystemExit("group_id не проставились для части строк")

    print("[INFO] total rows:", len(df))
    print("[INFO] unique bars:", bars.shape[0])
    return df, bars


def get_feature_cols(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "pnl_net",
        "ks_ttl_hours",
        "sample_weight",
        "is_focus",
        "group_id",
    }
    feature_cols = [c for c in num_cols if c not in exclude]
    print("[INFO] num features:", len(feature_cols))
    print("[INFO] first 20 features:", feature_cols[:20])
    return feature_cols


def compute_static_best_ks(df: pd.DataFrame):
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

    return best_ks


def apply_static_strategy(df: pd.DataFrame, bars: pd.DataFrame, best_ks: dict):
    rows = []
    for side, (tp, sl) in best_ks.items():
        mask = (df["side"] == side) & (df["ks_tp_mult"] == tp) & (df["ks_sl_mult"] == sl)
        sub = df.loc[mask, ["group_id", "pnl_net"]]
        rows.append(sub)
    static_df = pd.concat(rows, axis=0)

    if static_df["group_id"].nunique() != bars["group_id"].nunique():
        print("[WARN] static: covered:",
              static_df["group_id"].nunique(), "bars:", bars["group_id"].nunique())

    static_df = static_df.groupby("group_id", as_index=False)["pnl_net"].first()
    static_df = static_df.rename(columns={"pnl_net": "pnl_static"})
    return static_df


def apply_catboost_ranker(df: pd.DataFrame, bars: pd.DataFrame, feature_cols):
    print("\n[LOAD RANKER]", MODEL_RANK_PATH)
    ranker = CatBoostRanker()
    ranker.load_model(MODEL_RANK_PATH)

    print("[PREDICT] CatBoostRanker scores...")
    df["score_rank"] = ranker.predict(df[feature_cols])

    idx_max = df.groupby("group_id")["score_rank"].idxmax()
    rank_df = (
        df.loc[idx_max, ["group_id", "pnl_net"]]
        .rename(columns={"pnl_net": "pnl_ranker"})
        .reset_index(drop=True)
    )

    if rank_df["group_id"].nunique() != bars["group_id"].nunique():
        print("[WARN] ranker: covered:",
              rank_df["group_id"].nunique(), "bars:", bars["group_id"].nunique())

    return rank_df


def join_bar_results(df: pd.DataFrame, static_df: pd.DataFrame, rank_df: pd.DataFrame):
    bar_meta = (
        df.sort_values("group_id")
        .drop_duplicates("group_id")[
            ["group_id", "entry_ts", "side",
             "is_focus", "vol_rel", "rng_norm", "quality_score"]
        ]
    )

    res = bar_meta.merge(static_df, on="group_id", how="left")
    res = res.merge(rank_df, on="group_id", how="left")
    return res


def summarize_subset(res: pd.DataFrame, mask: pd.Series, name: str):
    sub = res.loc[mask].copy()
    n = len(sub)
    if n == 0:
        print(f"\n=== SUBSET {name}: 0 bars ===")
        return

    print(f"\n=== SUBSET {name}: bars={n} ===")

    for col, label in [("pnl_static", "STATIC"),
                       ("pnl_ranker", "RANKER")]:
        s = sub[col].dropna()
        if s.empty:
            print(f"{label}: no data")
            continue
        mean = float(s.mean())
        med = float(s.median())
        pos = float((s > 0).mean())
        neg = float((s < 0).mean())
        print(f"{label}: mean={mean:.4f}, median={med:.4f}, "
              f"pos={pos:.3%}, neg={neg:.3%}")

    both = sub.dropna(subset=["pnl_static", "pnl_ranker"])
    if len(both):
        win = float((both["pnl_ranker"] > both["pnl_static"]).mean())
        tie = float((both["pnl_ranker"] == both["pnl_static"]).mean())
        lose = float((both["pnl_ranker"] < both["pnl_static"]).mean())
        print(f"RANKER vs STATIC (per bar): win={win:.3%}, tie={tie:.3%}, lose={lose:.3%}")


def main():
    df, bars = load_data()
    feature_cols = get_feature_cols(df)

    best_ks = compute_static_best_ks(df)
    static_df = apply_static_strategy(df, bars, best_ks)
    rank_df = apply_catboost_ranker(df, bars, feature_cols)

    res = join_bar_results(df, static_df, rank_df)

    mask_all = res["pnl_static"].notna() & res["pnl_ranker"].notna()
    mask_focus = mask_all & (res["is_focus"] == 1)
    mask_entry = mask_all & (
        (res["quality_score"] > 0.4) &
        (res["vol_rel"] > 1.0) &
        (res["rng_norm"] > 1.0)
    )

    summarize_subset(res, mask_all,   "ALL BARS")
    summarize_subset(res, mask_focus, "FOCUS BARS")
    summarize_subset(res, mask_entry, "ENTRY-ZONE")

    out_path = Path("reports/features/bonk/bonk_v15_eval_ks_cat_vs_static.parquet")
    res.to_parquet(out_path)
    print("\n[SAVED per-bar results] ->", out_path)


if __name__ == "__main__":
    main()