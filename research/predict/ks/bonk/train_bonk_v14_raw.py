from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

DATA = Path("reports/features/dataset_ks_v14_bonk_merged.parquet")
OUT  = Path("models/bonk_v14_raw.cbm")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading:", DATA)
df = pd.read_parquet(DATA)
df = df.sort_values("entry_ts").reset_index(drop=True)

y = df["pnl_net"].astype(float)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = {"pnl_net", "sample_weight", "is_focus", "ks_ttl_hours"}
feature_cols = [c for c in num_cols if c not in exclude]

split = int(len(df)*0.8)
X_train, y_train = df[feature_cols].iloc[:split], y.iloc[:split]
X_valid, y_valid = df[feature_cols].iloc[split:], y.iloc[split:]

print("Train rows:", len(X_train), "Valid rows:", len(X_valid))

model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=8000,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
    random_seed=42,
    od_type="Iter",
    od_wait=400,
    task_type="CPU",
    verbose=False,
)

model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
)

model.save_model(OUT)
print("Saved →", OUT)