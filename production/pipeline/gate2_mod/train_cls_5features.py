from __future__ import annotations

import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from production.pipeline.gate2_mod.common_5features import (
    OUT_ROOT,
    REACH_ALL_PATH,
    build_feature_cols,
    ensure_dir,
    make_top_bucket_report,
    prepare_xy,
    save_json,
    split_train_valid,
)


CATBOOST_PARAMS_BASE = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 20000,
    "depth": 6,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10.0,
    "random_seed": 42,
    "verbose": 100,
    "allow_writing_files": False,
    "od_type": "Iter",
    "od_wait": 800,
    "use_best_model": True,
}

TASKS = [
    {
        "name": "up_reach_high",
        "target_col": "gate2_up_reach_high",
    },
    {
        "name": "dn_reach_high",
        "target_col": "gate2_dn_reach_high",
    },
]


def train_one(df: pd.DataFrame, task: dict) -> None:
    name = str(task["name"])
    target_col = str(task["target_col"])

    out_dir = os.path.join(OUT_ROOT, "cls", name)
    ensure_dir(out_dir)

    work = df.copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col]).copy()
    work[target_col] = work[target_col].astype(int)

    feature_cols = build_feature_cols(work)
    if len(feature_cols) == 0:
        raise ValueError(f"{name}: no feature cols")

    forbidden_now = {
        "gate2_up_reach_mid",
        "gate2_dn_reach_mid",
        "gate2_up_reach_high",
        "gate2_dn_reach_high",
        "gate2_up_impulse_8h_010",
        "gate2_dn_impulse_8h_010",
        "gate2_up_impulse_8h_015",
        "gate2_dn_impulse_8h_015",
        "gate2_up_clean_impulse_8h",
        "gate2_dn_clean_impulse_8h",
        "gate2_up_impulse_8h_2atr",
        "gate2_dn_impulse_8h_2atr",
        "mfe_up_atr_8h",
        "mfe_dn_atr_8h",
        "mfe_up_atr_h",
        "mfe_dn_atr_h",
        "mfe_up_pct_8h",
        "mfe_dn_pct_8h",
        "mfe_up_pct_16h",
        "mfe_dn_pct_16h",
        "mae_up_pct_8h",
        "mae_dn_pct_8h",
        "mae_up_pct_16h",
        "mae_dn_pct_16h",
        "first_up_mid_ts",
        "first_dn_mid_ts",
        "first_up_high_ts",
        "first_dn_high_ts",
        "first_up_impulse_8h_010_ts",
        "first_dn_impulse_8h_010_ts",
        "first_up_impulse_8h_015_ts",
        "first_dn_impulse_8h_015_ts",
        "mid_first_side",
        "high_first_side",
        "impulse_010_first_side_8h",
        "impulse_015_first_side_8h",
        "entry_px_exec",
        "atr14_at_signal",
        "ttl_hours",
        "impulse_hours",
        "upstream_split",
        "upstream_valid_start_ts",
    }

    leaked_features = sorted(set(feature_cols) & forbidden_now)
    if leaked_features:
        raise ValueError(f"{name}: leaked features in feature_cols: {leaked_features}")

    train_df, valid_df, split_source = split_train_valid(work)

    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError(f"{name}: empty train or valid split")

    if train_df[target_col].nunique() < 2:
        raise ValueError(f"{name}: train split has single class")

    if valid_df[target_col].nunique() < 2:
        raise ValueError(f"{name}: valid split has single class")

    x_train, x_valid, y_train, y_valid = prepare_xy(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    y_train = pd.to_numeric(y_train, errors="coerce").astype(int)
    y_valid = pd.to_numeric(y_valid, errors="coerce").astype(int)

    model_params = dict(CATBOOST_PARAMS_BASE)

    pos_rate = float(train_df[target_col].mean())
    neg_rate = 1.0 - pos_rate

    model_params["class_weights"] = {
        0: 1.0,
        1: neg_rate / max(pos_rate, 1e-6),
    }

    model = CatBoostClassifier(**model_params)

    model.fit(
        x_train,
        y_train,
        eval_set=(x_valid, y_valid),
        cat_features=["symbol"] if "symbol" in x_train.columns else None,
        use_best_model=True,
    )

    valid_proba = model.predict_proba(x_valid)[:, 1]
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(0.1, 0.99, 50):
        pred = (valid_proba >= thr).astype(int)
        f1 = f1_score(y_valid, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    valid_pred = (valid_proba >= best_thr).astype(int)

    auc = float(roc_auc_score(y_valid, valid_proba))
    ap = float(average_precision_score(y_valid, valid_proba))
    precision_best = float(precision_score(y_valid, valid_pred, zero_division=0))
    recall_best = float(recall_score(y_valid, valid_pred, zero_division=0))
    f1_best = float(f1_score(y_valid, valid_pred, zero_division=0))
    cm = confusion_matrix(y_valid, valid_pred, labels=[0, 1])

    valid_pred_df = valid_df[
        [c for c in ["symbol", "signal_ts", "upstream_split", target_col] if c in valid_df.columns]
    ].copy()
    valid_pred_df["proba"] = valid_proba
    valid_pred_df["pred"] = valid_pred
    valid_pred_df["is_correct"] = (valid_pred_df["pred"] == valid_pred_df[target_col]).astype(int)

    top_bucket = make_top_bucket_report(
        df_valid=valid_pred_df,
        proba_col="proba",
        y_col=target_col,
    )

    used_features = list(x_train.columns)

    feature_importance = pd.DataFrame(
        {
            "feature": used_features,
            "importance": model.get_feature_importance(),
        }
    )

    model_path = os.path.join(out_dir, f"{name}.cbm")
    fi_path = os.path.join(out_dir, "features.csv")
    features_path = os.path.join(out_dir, "features_used.csv")
    pred_path = os.path.join(out_dir, "valid_predictions.csv")
    bucket_path = os.path.join(out_dir, "top_bucket_report.csv")
    report_path = os.path.join(out_dir, "report.json")

    model.save_model(model_path)
    feature_importance.to_csv(fi_path, index=False)
    pd.DataFrame({"feature": used_features}).to_csv(features_path, index=False)
    valid_pred_df.to_csv(pred_path, index=False)
    top_bucket.to_csv(bucket_path, index=False)

    report = {
        "task_name": name,
        "target_col": target_col,
        "rows_total": int(len(work)),
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
        "split_source": split_source,
        "feature_count": int(len(feature_cols)),
        "class_rate_total": float(work[target_col].mean()),
        "class_rate_train": float(train_df[target_col].mean()),
        "class_rate_valid": float(valid_df[target_col].mean()),
        "best_threshold": float(best_thr),
        "metrics": {
            "auc": auc,
            "ap": ap,
            "precision_best": precision_best,
            "recall_best": recall_best,
            "f1_best": f1_best,
        },
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm.tolist(),
        "files": {
            "model": model_path,
            "features": fi_path,
            "features_used": features_path,
            "valid_predictions": pred_path,
            "top_bucket_report": bucket_path,
        },
    }
    save_json(report_path, report)

    print("=" * 120)
    print("TASK:", name)
    print("TARGET:", target_col)
    print("ROWS TOTAL:", len(work))
    print("ROWS TRAIN:", len(train_df))
    print("ROWS VALID:", len(valid_df))
    print("FEATURE COUNT:", len(feature_cols))
    print("POS RATE VALID:", round(float(valid_df[target_col].mean()), 6))
    print("AUC:", round(auc, 6))
    print("AP:", round(ap, 6))
    print("PRECISION@BEST:", round(precision_best, 6))
    print("RECALL@BEST:", round(recall_best, 6))
    print("F1@BEST:", round(f1_best, 6))
    print("BEST THRESHOLD:", round(best_thr, 6))
    print("TOP BUCKET REPORT")
    print(top_bucket.to_string(index=False))
    print("TOP FEATURES")
    print(feature_importance.head(25).to_string(index=False))
    print("WROTE", model_path)
    print("WROTE", fi_path)
    print("WROTE", features_path)
    print("WROTE", pred_path)
    print("WROTE", bucket_path)
    print("WROTE", report_path)
    print()


def main() -> None:
    ensure_dir(OUT_ROOT)

    if not os.path.exists(REACH_ALL_PATH):
        raise SystemExit(f"not found: {REACH_ALL_PATH}")

    df = pd.read_parquet(REACH_ALL_PATH)

    for task in TASKS:
        train_one(df, task)


if __name__ == "__main__":
    main()