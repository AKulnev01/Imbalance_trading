from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

DATA = Path("reports/features/dataset_ks_v12_bonk_merged.parquet")

MODEL_RAW     = Path("models/bonk_v14_raw.cbm")
MODEL_FOCUS   = Path("models/bonk_v12_focus.cbm")
MODEL_WEIGHT  = Path("models/bonk_v12_weighted.cbm")

OUT = Path("reports/features/bonk_v12_model_compare.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading BONK dataset...")
df = pd.read_parquet(DATA)
df["entry_ts"] = pd.to_datetime(df["entry_ts"])
df = df.sort_values("entry_ts").reset_index(drop=True)

# trade id per (entry_ts, side)
df["_trade_id"] = df.groupby(["entry_ts", "side"]).ngroup()
n_trades = df["_trade_id"].nunique()
print("Trades:", n_trades)

ks_cols = ["ks_tp_mult", "ks_sl_mult"]

# ---------- STATIC BASELINE ----------
static_grp = df.groupby(ks_cols)["pnl_net"].mean().reset_index()
best_idx = static_grp["pnl_net"].idxmax()
best_tp, best_sl = static_grp.loc[best_idx, ks_cols]

df_static = df[(df["ks_tp_mult"] == best_tp) &
               (df["ks_sl_mult"] == best_sl)].copy()

pnl_static = df_static.drop_duplicates("_trade_id")["pnl_net"].sum()
print("Static PNL:", pnl_static)

# ---------- FEATURES ----------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = {"pnl_net", "ks_ttl_hours", "sample_weight", "is_focus"}
feature_cols = [c for c in num_cols if c not in exclude]

X = df[feature_cols]

# ---------- LOAD MODELS ----------
m_raw = CatBoostRegressor()
m_raw.load_model(str(MODEL_RAW))

m_focus = CatBoostRegressor()
m_focus.load_model(str(MODEL_FOCUS))

m_weight = CatBoostRegressor()
m_weight.load_model(str(MODEL_WEIGHT))

print("Predicting...")

df["pred_raw"]    = m_raw.predict(X)
df["pred_focus"]  = m_focus.predict(X)
df["pred_weight"] = m_weight.predict(X)

# ---------- KS SELECTION ----------
idx_raw    = df.groupby("_trade_id")["pred_raw"].idxmax()
idx_focus  = df.groupby("_trade_id")["pred_focus"].idxmax()
idx_weight = df.groupby("_trade_id")["pred_weight"].idxmax()

df_raw    = df.loc[idx_raw]
df_focus  = df.loc[idx_focus]
df_weight = df.loc[idx_weight]

pnl_raw    = df_raw["pnl_net"].sum()
pnl_focus  = df_focus["pnl_net"].sum()
pnl_weight = df_weight["pnl_net"].sum()

print("\n=== PNL totals ===")
print("static :", pnl_static)
print("raw    :", pnl_raw)
print("focus  :", pnl_focus)
print("weight :", pnl_weight)

# ---------- Extended Analysis ----------
static_u = df_static.drop_duplicates("_trade_id").set_index("_trade_id")
raw_u    = df_raw.drop_duplicates("_trade_id").set_index("_trade_id")
focus_u  = df_focus.drop_duplicates("_trade_id").set_index("_trade_id")
weight_u = df_weight.drop_duplicates("_trade_id").set_index("_trade_id")

def diff_mask(a, b):
    return (a[ks_cols] != b[ks_cols]).any(axis=1)

mask_raw    = diff_mask(static_u, raw_u)
mask_focus  = diff_mask(static_u, focus_u)
mask_weight = diff_mask(static_u, weight_u)

def edge_diff(a, b, mask):
    if mask.sum() == 0:
        return 0.0
    return b.loc[mask, "pnl_net"].sum() - a.loc[mask, "pnl_net"].sum()

edge_raw_diff    = edge_diff(static_u, raw_u, mask_raw)
edge_focus_diff  = edge_diff(static_u, focus_u, mask_focus)
edge_weight_diff = edge_diff(static_u, weight_u, mask_weight)

out = pd.DataFrame([
    dict(
        pnl_static=pnl_static,
        pnl_raw=pnl_raw,
        pnl_focus=pnl_focus,
        pnl_weight=pnl_weight,

        edge_raw_vs_static=pnl_raw - pnl_static,
        edge_focus_vs_static=pnl_focus - pnl_static,
        edge_weight_vs_static=pnl_weight - pnl_static,

        diff_raw_rate=float(mask_raw.mean()),
        diff_focus_rate=float(mask_focus.mean()),
        diff_weight_rate=float(mask_weight.mean()),

        diff_raw_trades=int(mask_raw.sum()),
        diff_focus_trades=int(mask_focus.sum()),
        diff_weight_trades=int(mask_weight.sum()),

        edge_raw_diff_trades=edge_raw_diff,
        edge_focus_diff_trades=edge_focus_diff,
        edge_weight_diff_trades=edge_weight_diff,
    )
])

out.to_csv(OUT, index=False)
print("\nSaved →", OUT)
print("\nDONE.")