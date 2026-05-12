import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


def parse_args():
    p = argparse.ArgumentParser(description="Train CatBoost regressor for KS PnL (ks_ret_adj).")
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--test", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)

    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--l2-leaf-reg", type=float, default=3.0)
    p.add_argument("--bagging-temperature", type=float, default=1.0)
    p.add_argument("--border-count", type=int, default=128)

    p.add_argument("--early-stopping", type=int, default=200)
    p.add_argument("--random-strength", type=float, default=1.0)
    p.add_argument("--leaf-estimation-iterations", type=int, default=4)

    p.add_argument("--loss-function", type=str, default="RMSE")
    p.add_argument("--eval-metric", type=str, default="RMSE")

    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def select_numeric_features(df: pd.DataFrame, target: str):
    bad_cols = {"entry_ts", "side", "symbol", "set"}
    feats = []
    for c in df.columns:
        if c == target:
            continue
        if c in bad_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats


def main():
    args = parse_args()
    t0 = time.time()

    train_path = Path(args.train)
    test_path = Path(args.test)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] train:", train_path)
    df_train = pd.read_parquet(train_path)

    print("[LOAD] test :", test_path)
    df_test = pd.read_parquet(test_path)

    target = "ks_ret_adj"
    if target not in df_train.columns:
        raise RuntimeError(f"target '{target}' not in train columns")

    if args.max_rows is not None and len(df_train) > args.max_rows:
        print(f"[SUBSAMPLE] train rows {len(df_train)} → {args.max_rows}")
        df_train = df_train.sample(args.max_rows, random_state=args.seed).sort_values("entry_ts")

    print("[INFO] train rows:", len(df_train), "| test rows:", len(df_test))

    feature_cols = select_numeric_features(df_train, target)
    print(f"[FEATURE SELECT] numeric feature cols: {len(feature_cols)}")
    print("first 40:", feature_cols[:40])

    X_train = df_train[feature_cols]
    y_train = df_train[target].astype("float32")

    X_valid = df_test[feature_cols]
    y_valid = df_test[target].astype("float32")

    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_valid shape:", X_valid.shape)
    print("NaNs train:", np.isnan(X_train.to_numpy()).mean(), "| NaNs target:", np.isnan(y_train.to_numpy()).mean())
    print("NaNs valid:", np.isnan(X_valid.to_numpy()).mean(), "| NaNs target:", np.isnan(y_valid.to_numpy()).mean())

    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    params = {
        "loss_function": args.loss_function,
        "eval_metric": args.eval_metric,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "bagging_temperature": args.bagging_temperature,
        "border_count": args.border_count,
        "random_strength": args.random_strength,
        "leaf_estimation_iterations": args.leaf_estimation_iterations,
        "od_type": "Iter",
        "od_wait": args.early_stopping,
        "thread_count": args.num_threads,
        "random_seed": args.seed,
        "verbose": 50,
    }

    print("\n[TRAIN] starting CatBoost...")
    print(json.dumps(params, indent=2))

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    best_it = model.get_best_iteration()
    print("[TRAIN] finished. best_iteration=", best_it)

    train_preds = model.predict(train_pool)
    valid_preds = model.predict(valid_pool)

    def metrics(y, y_hat, name):
        rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
        mae = float(np.mean(np.abs(y_hat - y)))
        mean_y = float(np.mean(y))
        std_y = float(np.std(y))
        print(f"[METRIC] {name}: rmse={rmse:.6f}  mae={mae:.6f}  mean_y={mean_y:.6f}  std_y={std_y:.6f}")
        return {"rmse": rmse, "mae": mae, "mean_y": mean_y, "std_y": std_y}

    m_train = metrics(y_train.to_numpy(), train_preds, "train")
    m_valid = metrics(y_valid.to_numpy(), valid_preds, "valid")

    model_path = outdir / "model_ks_pnl_cat.cbm"
    feats_path = outdir / "feature_cols_cat.txt"
    meta_path = outdir / "meta_ks_pnl_cat.json"

    model.save_model(str(model_path))
    with open(feats_path, "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")

    meta = {
        "model_path": str(model_path),
        "feature_cols_path": str(feats_path),
        "target": target,
        "train_rows": int(len(df_train)),
        "valid_rows": int(len(df_test)),
        "params": params,
        "metrics_train": m_train,
        "metrics_valid": m_valid,
        "best_iteration": int(best_it),
        "created_utc": time.time(),
        "git_commit": os.getenv("GIT_COMMIT", None),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[SAVE] model:", model_path)
    print("[SAVE] feature cols:", feats_path)
    print("[SAVE] meta:", meta_path)
    print("[DONE] elapsed_sec:", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()