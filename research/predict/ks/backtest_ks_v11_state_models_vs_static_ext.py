from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

DATA_DIR = Path("reports/features/dataset_ks_v11_by_symbol_with_states_175")

FOCUS_MODELS  = Path("models/ks_v11_state_per_symbol_cat_focus_only")
WEIGHT_MODELS = Path("models/ks_v11_state_per_symbol_cat_weighted_gpu")

OUT_CSV = Path("reports/features/ks_v11_state_models_backtest_compare_ext.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

results = []

def best_static(df):
    """Находим лучший KS-комбо по среднему pnl_net."""
    ks_cols = ["ks_tp_scale", "ks_sl_scale", "ks_ttl_hours"]
    if not set(ks_cols).issubset(df.columns):
        return None

    ks_grp = df.groupby(ks_cols)["pnl_net"].mean().reset_index()
    best_idx = ks_grp["pnl_net"].idxmax()
    row = ks_grp.loc[best_idx]
    return row["ks_tp_scale"], row["ks_sl_scale"], row["ks_ttl_hours"]


for p in sorted(DATA_DIR.glob("*.parquet")):
    symbol = p.stem

    focus_path  = FOCUS_MODELS  / f"{symbol}.cbm"
    weight_path = WEIGHT_MODELS / f"{symbol}.cbm"

    if not focus_path.exists() or not weight_path.exists():
        print(f"SKIP {symbol} (no focus/weight model)")
        continue

    print(f"\n=== {symbol} ===")

    df = pd.read_parquet(p)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df = df.sort_values("entry_ts").reset_index(drop=True)

    # trade_id
    df["_trade_id"] = df.groupby(["entry_ts", "side"]).ngroup()
    all_trade_ids = sorted(df["_trade_id"].unique())
    n_trades = len(all_trade_ids)

    # features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "pnl_net","label_focus","state_focus","is_focus","sample_weight",
        "pnl_max","pnl_min","pnl_spread"
    }
    feature_cols = [c for c in num_cols if c not in exclude]

    if not feature_cols:
        print("  SKIP: no features")
        continue

    X = df[feature_cols]

    # ---------------- STATIC KS ----------------
    ks_static = best_static(df)
    if ks_static is None:
        print("  SKIP: missing KS-scale columns")
        continue

    tp_s, sl_s, ttl_s = ks_static

    df_static = df[
        (df["ks_tp_scale"] == tp_s) &
        (df["ks_sl_scale"] == sl_s) &
        (df["ks_ttl_hours"] == ttl_s)
    ].copy()

    df_static_u = df_static.drop_duplicates("_trade_id").set_index("_trade_id")

    pnl_static = df_static_u["pnl_net"].sum()

    # ---------------- MODELS ----------------
    # focus
    model_focus = CatBoostRegressor()
    model_focus.load_model(str(focus_path))
    df["pred_focus"] = model_focus.predict(X)

    idx_focus = df.groupby("_trade_id")["pred_focus"].idxmax()
    df_focus = df.loc[idx_focus]
    df_focus_u = df_focus.drop_duplicates("_trade_id").set_index("_trade_id")

    pnl_focus = df_focus_u["pnl_net"].sum()

    # weight
    model_weight = CatBoostRegressor()
    model_weight.load_model(str(weight_path))
    df["pred_weight"] = model_weight.predict(X)

    idx_weight = df.groupby("_trade_id")["pred_weight"].idxmax()
    df_weight = df.loc[idx_weight]
    df_weight_u = df_weight.drop_duplicates("_trade_id").set_index("_trade_id")

    pnl_weight = df_weight_u["pnl_net"].sum()

    # ---------------- ALIGN индексов ----------------
    ks_cols = ["ks_tp_scale", "ks_sl_scale", "ks_ttl_hours"]

    df_static_u = df_static_u.reindex(all_trade_ids)
    df_focus_u  = df_focus_u.reindex(all_trade_ids)
    df_weight_u = df_weight_u.reindex(all_trade_ids)

    # ---------------- DIFFERENCE MASKS ----------------
    diff_focus_mask = (df_static_u[ks_cols] != df_focus_u[ks_cols]).any(axis=1)
    diff_weight_mask = (df_static_u[ks_cols] != df_weight_u[ks_cols]).any(axis=1)

    diff_focus_rate  = diff_focus_mask.mean()
    diff_weight_rate = diff_weight_mask.mean()

    # ---------------- EDGE НА ИЗМЕНЕННЫХ KS ----------------
    if diff_focus_mask.sum() > 0:
        edge_focus_diff = (
            df_focus_u.loc[diff_focus_mask, "pnl_net"].sum()
            - df_static_u.loc[diff_focus_mask, "pnl_net"].sum()
        )
    else:
        edge_focus_diff = 0.0

    if diff_weight_mask.sum() > 0:
        edge_weight_diff = (
            df_weight_u.loc[diff_weight_mask, "pnl_net"].sum()
            - df_static_u.loc[diff_weight_mask, "pnl_net"].sum()
        )
    else:
        edge_weight_diff = 0.0

    # ---------------- save ----------------
    results.append(dict(
        symbol=symbol,
        n_trades=n_trades,

        pnl_static=pnl_static,
        pnl_focus=pnl_focus,
        pnl_weight=pnl_weight,

        edge_focus_vs_static=pnl_focus - pnl_static,
        edge_weight_vs_static=pnl_weight - pnl_static,

        focus_diff_rate=float(diff_focus_rate),
        weight_diff_rate=float(diff_weight_rate),

        edge_focus_diff_trades=float(edge_focus_diff),
        edge_weight_diff_trades=float(edge_weight_diff),

        n_focus_diff=int(diff_focus_mask.sum()),
        n_weight_diff=int(diff_weight_mask.sum()),
    ))

    print(
        f"  trades={n_trades} | "
        f"static={pnl_static:.2f} | focus={pnl_focus:.2f} | weight={pnl_weight:.2f} | "
        f"focus_diff_rate={diff_focus_rate:.3f} | weight_diff_rate={diff_weight_rate:.3f}"
    )


# SAVE CSV
out_df = pd.DataFrame(results)
out_df.to_csv(OUT_CSV, index=False)

print("\nSaved EXT backtest ->", OUT_CSV)
print(out_df.head())