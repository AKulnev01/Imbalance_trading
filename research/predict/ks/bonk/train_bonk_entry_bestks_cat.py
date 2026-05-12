from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

DATA_PATH = Path("reports/features/bonk/bonk_v15_entry_bestks.parquet")
OUT_PATH  = Path("models/bonk_v15_entry_bestks_cat.cbm")


def main():
    print("Loading:", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df = df.sort_values(["entry_ts", "side"]).reset_index(drop=True)

    y = df["label"].astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != "label"]

    print("Rows:", len(df))
    print("Num features:", len(feature_cols))
    print("Pos rate:", float(y.mean()))

    split = int(len(df) * 0.80)
    X_train = df[feature_cols].iloc[:split]
    y_train = y.iloc[:split]
    X_valid = df[feature_cols].iloc[split:]
    y_valid = y.iloc[split:]

    print("Train rows:", len(X_train), "Valid rows:", len(X_valid))

    train_pool = Pool(X_train, label=y_train)
    valid_pool = Pool(X_valid, label=y_valid)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        random_seed=42,
        od_type="Iter",
        od_wait=200,
        task_type="CPU",
        verbose=False,
        class_weights=[1.0, 3.0],
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    p_train = model.predict_proba(X_train)[:, 1]
    p_valid = model.predict_proba(X_valid)[:, 1]

    auc_train = roc_auc_score(y_train, p_train)
    auc_valid = roc_auc_score(y_valid, p_valid)

    print("\n=== ENTRY MODEL (CatBoostClassifier, best_ks only) ===")
    print("AUC train:", auc_train)
    print("AUC valid:", auc_valid)

    for thr in [0.3, 0.5, 0.7]:
        pred = (p_valid >= thr).astype(int)
        acc = float((pred == y_valid).mean())
        share_entry = float(pred.mean())
        print(f"thr={thr:.2f} | acc={acc:.3f} | share_entry={share_entry:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(OUT_PATH)
    print("Saved ->", OUT_PATH)


if __name__ == "__main__":
    main()