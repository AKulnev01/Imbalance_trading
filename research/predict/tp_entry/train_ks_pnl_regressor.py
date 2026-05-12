import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


def parse_args():
    p = argparse.ArgumentParser(description="Train KS PnL regressor (variant B: fat KS grid)")
    p.add_argument(
        "--train",
        type=str,
        default="reports/features/2025-11-12_fullscan/dataset_ks_fat_train.parquet",
        help="Path to train parquet",
    )
    p.add_argument(
        "--test",
        type=str,
        default="reports/features/2025-11-12_fullscan/dataset_ks_fat_test.parquet",
        help="Path to test parquet",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="predict/tp_entry/models_ks_pnl",
        help="Output dir for model and metadata",
    )
    p.add_argument(
        "--target",
        type=str,
        default="ks_ret_adj",
        help="Target column (adjusted PnL)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows for train/test (for quick runs)",
    )
    p.add_argument("--n-estimators", type=int, default=2500)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=255)
    p.add_argument("--min-child-samples", type=int, default=40)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--early-stopping", type=int, default=200)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def select_feature_cols(df: pd.DataFrame, target: str):
    all_cols = df.columns.tolist()

    # то, что точно не должно идти в признаки
    exclude_exact = {
        target,
        "ks_ret_gross",
        "set",
        "entry_ts",
        "symbol",
        "side",
    }

    feature_cols = []
    bad_cols = []

    for c in all_cols:
        if c in exclude_exact:
            continue
        # отсекаем всё ks_ret_*, даже если что-то ещё появится
        if c.startswith("ks_ret_"):
            continue

        dt = df[c].dtype

        # оставляем только числовые типы
        if dt.kind not in ("b", "i", "u", "f"):
            bad_cols.append((c, str(dt)))
            continue

        feature_cols.append(c)

    print("[FEATURE SELECT] numeric feature cols:", len(feature_cols))
    print("first 40:", feature_cols[:40])
    if bad_cols:
        print("\n[FEATURE SELECT] skipped non-numeric cols:")
        for name, dt in bad_cols:
            print(f"  - {name}: {dt}")

    return feature_cols


def eval_split(name: str, model: lgb.Booster, X: pd.DataFrame, y: pd.Series):
    pred = model.predict(X, num_iteration=model.best_iteration or model.current_iteration())
    err = pred - y.to_numpy()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mean_y = float(np.mean(y))
    std_y = float(np.std(y))
    print(f"[METRIC] {name}: rmse={rmse:.6f}  mae={mae:.6f}  mean_y={mean_y:.6f}  std_y={std_y:.6f}")
    return {"rmse": rmse, "mae": mae, "mean_y": mean_y, "std_y": std_y}


def main():
    args = parse_args()

    print("[LOAD] train:", args.train)
    df_train = pd.read_parquet(args.train)
    print("[LOAD] test :", args.test)
    df_test = pd.read_parquet(args.test)

    # sanity по таргету
    target = args.target
    if target not in df_train.columns:
        raise RuntimeError(f"Target '{target}' not in train columns")
    if target not in df_test.columns:
        raise RuntimeError(f"Target '{target}' not in test columns")

    # опциональное ограничение по строкам
    if args.max_rows is not None:
        if len(df_train) > args.max_rows:
            df_train = df_train.sample(n=args.max_rows, random_state=args.seed)
        if len(df_test) > args.max_rows:
            df_test = df_test.sample(n=args.max_rows, random_state=args.seed)

    print("[INFO] train rows:", len(df_train), "| test rows:", len(df_test))

    # выбираем признаки
    feature_cols = select_feature_cols(df_train, target=target)

    X_train = df_train[feature_cols]
    y_train = df_train[target]

    X_valid = df_test[feature_cols]
    y_valid = df_test[target]

    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_valid shape:", X_valid.shape)

    # lightgbm dataset
    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": ["l2", "l1"],
        "verbosity": -1,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_child_samples,
        "bagging_fraction": args.subsample,
        "feature_fraction": args.colsample_bytree,
        "lambda_l1": args.reg_alpha,
        "lambda_l2": args.reg_lambda,
        "num_threads": args.num_threads,
        "seed": args.seed,
    }

    print("\n[TRAIN] starting LightGBM...")
    print(json.dumps(params, indent=2))

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
    ]

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=args.n_estimators,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration or model.current_iteration()
    print(f"[TRAIN] finished. best_iteration={best_iter}")

    # метрики
    metrics = {}
    metrics["train"] = eval_split("train", model, X_train, y_train)
    metrics["valid"] = eval_split("valid", model, X_valid, y_valid)

    # сохранение
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model_ks_pnl.txt"
    model.save_model(str(model_path))
    print("[SAVE] model:", model_path)

    feat_path = outdir / "feature_cols.txt"
    with open(feat_path, "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")
    print("[SAVE] feature cols:", feat_path)

    meta = {
        "train_path": args.train,
        "test_path": args.test,
        "target": target,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "params": params,
        "best_iteration": int(best_iter),
        "metrics": metrics,
    }
    meta_path = outdir / "meta_ks_pnl.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[SAVE] meta:", meta_path)


if __name__ == "__main__":
    main()