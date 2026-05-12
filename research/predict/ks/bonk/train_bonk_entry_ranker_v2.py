from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

DATA_PATH = Path("reports/features/bonk/bonk_v15_entry_dataset.parquet")
MODEL_PATH = Path("models/bonk_v15_entry_v2_cat.cbm")

print("[LOAD]", DATA_PATH)
df = pd.read_parquet(DATA_PATH)
df["entry_ts"] = pd.to_datetime(df["entry_ts"])
df = df.sort_values("entry_ts").reset_index(drop=True)

print("[INFO] rows:", len(df))

if "label" not in df.columns:
    raise SystemExit("ERROR: no 'label' column in entry dataset")

y = df["label"].astype(int)
pos_rate = float((y == 1).mean())
print("[INFO] pos rate:", pos_rate)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c != "label"]

print("[INFO] num features:", len(feature_cols))

X = df[feature_cols]

split = int(len(df) * 0.8)
X_train = X.iloc[:split]
y_train = y.iloc[:split]
X_valid = X.iloc[split:]
y_valid = y.iloc[split:]

print("Train rows:", len(X_train), "Valid rows:", len(X_valid))

train_pool = Pool(X_train, label=y_train)
valid_pool = Pool(X_valid, label=y_valid)

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=2000,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    od_type="Iter",
    od_wait=200,
    task_type="CPU",
    verbose=False,
)

model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

p_train = model.predict_proba(train_pool)[:, 1]
p_valid = model.predict_proba(valid_pool)[:, 1]

auc_train = roc_auc_score(y_train, p_train)
auc_valid = roc_auc_score(y_valid, p_valid)

print("\n=== ENTRY MODEL (CatBoostClassifier, v2 on entry_dataset) ===")
print("AUC train:", auc_train)
print("AUC valid:", auc_valid)

for thr in [0.3, 0.5, 0.7]:
    pred = (p_valid >= thr).astype(int)
    acc = float((pred == y_valid).mean())
    share_entry = float(pred.mean())
    print(f"thr={thr:.2f} | acc={acc:.3f} | share_entry={share_entry:.3f}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save_model(MODEL_PATH)
print("Saved ->", MODEL_PATH)
