from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

DATASET_PARQUET = "production/dataset/gate4/gate4_1_side_builder/gate4_1_side_dataset.parquet"
OUT_ROOT = "production/models/gate4/gate4_y_side_clean_multiclass"

OUT_MODEL = os.path.join(OUT_ROOT, "gate4_y_side_clean_multiclass.cbm")
OUT_FEATURES_CSV = os.path.join(OUT_ROOT, "features.csv")
OUT_VALID_PRED_CSV = os.path.join(OUT_ROOT, "valid_predictions.csv")
OUT_REPORT_JSON = os.path.join(OUT_ROOT, "report.json")
OUT_METRICS_CSV = os.path.join(OUT_ROOT, "metrics.csv")


# ============================================================
# CONFIG
# ============================================================

TARGET_COL = "y_side_clean"


VALID_TAIL_SHARE = 0.20

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 30000,
    "depth": 6,
    "learning_rate": 0.02,
    "l2_leaf_reg": 10.0,
    "random_seed": 42,
    "verbose": 100,
    "allow_writing_files": False,
    "auto_class_weights": "Balanced",
    "od_type": "Iter",
    "od_wait": 600,
}

TARGET_BIN_COL = "y_side_bin"

TARGET_MAP_BIN = {
    "SHORT": 0,
    "LONG": 1,
}

TARGET_NAMES_BIN = ["SHORT", "LONG"]


# ============================================================
# DROP COLS
# ============================================================

DROP_COLS_EXACT = {
    # meta
    "symbol",
    "ts",
    "upstream_split",
    "upstream_valid_start_ts",
    "upstream_is_oos",

    # obvious trading leakage / old leftovers
    "pnl_net",
    "entry_px",
    "exit_px",
    "tp_px",
    "sl_px",
    "y",
    "y_fast",
    "ret",
    "ret_feat",
    "ret_l1",
    "ret_l1_feat",
    "ret_l2",
    "ret_l2_feat",
    "ks_ret_adj",
    "side_num",
    "side_num_feat",
    "symbol_id",
    "symbol_id_feat",
    "gate4_margin_atr",
    "gate4_ttl_bars",
    "long_score_threshold",
    "short_score_threshold",
    "long_score_model_used",
    "short_score_model_used",
    "ks_tp_abs_best",
    "ks_sl_abs_best",
    "ks_ttl_hours_best",

    # raw OHLC if захочешь можно потом вернуть, пока убираю консервативно
    "open",
    "high",
    "low",
    "close",
    "close_g2",
    "open_g2",
    "high_g2",
    "low_g2",

    # y_dir block = альтернативный future target
    "y_dir",
    "y_dir_int",
    "y_dir_mfe",
    "y_dir_first",
    "y_dir_mfe_int",
    "y_dir_first_int",

    # direct future-path leakage
    "mfe_up_atr_16h",
    "mfe_dn_atr_16h",
    "first_up_hit_bar",
    "first_dn_hit_bar",
    "first_hit_atr",

    # primary target aliases
    "y_side_clean",
    "y_side_clean_int",
    "edge_atr_clean",
    "abs_edge_atr_clean",
    "is_side_clean_long",
    "is_side_clean_short",
    "is_side_clean_ambig",

    "base_long_candidate",
    "base_short_candidate",
    "extreme_long_candidate",
    "extreme_short_candidate",

    "pass_long",
    "pass_short",
    "pass_any",
    "pass_both",
    "pass_long_only",
    "pass_short_only",

    "g2_cls_spread",
    "g2_cls_abs_spread",
    "g2_up_dominant",
    "g2_dn_dominant",

    "g2_g3_side_agree",
    "g2_g3_side_conflict",
    "g1_g2_strength",
    "g1_g3_strength",

    "g2g3_joint_long",
    "g2g3_joint_short",
    "g2g3_joint_long_minus_short",
    "g2g3_joint_abs_spread",

    "g3_any_active",
    "g3_long_any_active",
    "g3_short_any_active",
    "g3_long_active_overlap_primary_secondary",
    "g3_short_active_overlap_primary_secondary",
    "gate3_active_overlap_primary_secondary",
    "has_any_gate3_bundle",
    "has_gate3_long_bundle",
    "has_gate3_short_bundle",
    "has_full_gate3_bundle",

    # target holder
    TARGET_BIN_COL,
}

DROP_COLS_PREFIX = (
    "mfe_",
    "sym_",
    "edge_delta_",
    "abs_edge_delta_",
    "y_side_clean_delta_",
    "is_long_delta_",
    "is_short_delta_",
    "is_ambig_delta_",
)

DROP_COLS_CONTAINS = (
    "target",
    "label",
    "future",
    "fwd",
    "outcome",
    "realized",
    "pnl",
)


KEEP_META_COLS_FOR_VALID = [
    "symbol",
    "ts",
    "upstream_split",

    "y_side_clean",
    "y_side_clean_int",
    "y_dir",
    "y_dir_int",

    "gate1_proba",
    "gate1_pass",

    "g2_cls_up_reach_high_proba",
    "g2_cls_dn_reach_high_proba",
    "g2_cls_spread",
    "g2_cls_abs_spread",
    "g2_cls_max",
    "g2_up_dominant",
    "g2_dn_dominant",

    "g2_up_dominant",
    "g2_dn_dominant",

    "g2_cls_spread",
    "g2_cls_abs_spread",

    "base_long_candidate",
    "base_short_candidate",
    "extreme_long_candidate",
    "extreme_short_candidate",
    "pass_long",
    "pass_short",
    "pass_any",
    "pass_both",
    "pass_long_only",
    "pass_short_only",

    "g3_long_active",
    "g3_short_active",
    "g3_any_active",
    "g3_both_active",
    "g3_long_score_proba",
    "g3_short_score_proba",
    "g3_long_score_pass",
    "g3_short_score_pass",
    "g3_score_spread",
    "g3_score_abs_spread",
    "g3_score_max",

    "g2_g3_side_agree",
    "g2_g3_side_conflict",
    "g1_g2_strength",
    "g1_g3_strength",
    "g2g3_joint_long",
    "g2g3_joint_short",
    "g2g3_joint_long_minus_short",
    "g2g3_joint_abs_spread",

    "has_gate3_long_bundle",
    "has_gate3_short_bundle",
    "has_any_gate3_bundle",
    "has_full_gate3_bundle",

    "gate3_pass_long",
    "gate3_pass_short",
    "gate3_proba_long",
    "gate3_proba_short",
    "gate3_margin_long",
    "gate3_margin_short",
    "gate3_threshold_long",
    "gate3_threshold_short",

    "gate3_precision_meta_long",
    "gate3_wilson_meta_long",
    "gate3_delta_wilson_meta_long",
    "gate3_pvalue_meta_long",
    "gate3_kept_n_meta_long",
    "gate3_valid_pos_rate_meta_long",
    "gate3_thr_kept_lift_meta_long",

    "gate3_precision_meta_short",
    "gate3_wilson_meta_short",
    "gate3_delta_wilson_meta_short",
    "gate3_pvalue_meta_short",
    "gate3_kept_n_meta_short",
    "gate3_valid_pos_rate_meta_short",
    "gate3_thr_kept_lift_meta_short",

    "gate3_precision_meta",
    "gate3_wilson_meta",
    "gate3_delta_wilson_meta",
    "gate3_pvalue_meta",
    "gate3_kept_n_meta",
    "gate3_valid_pos_rate_meta",
    "gate3_thr_kept_lift_meta",

    "gate3_active_count",
    "gate3_active_primary",
    "gate3_active_secondary",
    "gate3_active_overlap_primary_secondary",
    "gate3_max_active_age",
    "gate3_side_bias",
    "gate3_score_long",
    "gate3_score_short",
    "gate3_rank_long",
    "gate3_rank_short",

    "g2_g3_side_agree",
    "g2_g3_side_conflict",
    "g1_g2_strength",
    "g1_g3_strength",
    "g2g3_joint_long",
    "g2g3_joint_short",
    "g2g3_joint_long_minus_short",
    "g2g3_joint_abs_spread",

    "pass_long",
    "pass_short",
    "pass_any",
    "pass_both",
    "pass_long_only",
    "pass_short_only",
]


# ============================================================
# HELPERS
# ============================================================

def build_feature_cols(df: pd.DataFrame) -> List[str]:
    feature_cols: List[str] = []

    for c in df.columns:
        if c in DROP_COLS_EXACT:
            continue
        if any(c.startswith(prefix) for prefix in DROP_COLS_PREFIX):
            continue
        c_low = str(c).lower()
        if any(token in c_low for token in DROP_COLS_CONTAINS):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        feature_cols.append(c)

    return sorted(feature_cols)


def split_train_valid(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if "upstream_split" in df.columns:
        train_df = df[df["upstream_split"] == "train"].copy()
        valid_df = df[df["upstream_split"] == "valid"].copy()
        if len(train_df) > 0 and len(valid_df) > 0:
            return train_df, valid_df, "upstream_split"

    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    n_valid = max(1, int(round(n * VALID_TAIL_SHARE)))
    n_train = n - n_valid

    train_df = df.iloc[:n_train].copy()
    valid_df = df.iloc[n_train:].copy()
    return train_df, valid_df, "tail_split"


def prepare_xy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    med = x_train.median(numeric_only=True)
    x_train = x_train.fillna(med).fillna(0.0)
    x_valid = x_valid.fillna(med).fillna(0.0)

    y_train = train_df[target_col].astype(int)
    y_valid = valid_df[target_col].astype(int)

    return x_train, x_valid, y_train, y_valid


def build_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": macro_f1,
        "recall_short": float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "recall_long": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
    }


# ============================================================
# MAIN
# ============================================================

os.makedirs(OUT_ROOT, exist_ok=True)

if not os.path.exists(DATASET_PARQUET):
    raise SystemExit(f"not found: {DATASET_PARQUET}")

df = pd.read_parquet(DATASET_PARQUET)

if len(df) == 0:
    raise SystemExit("dataset is empty")

required_cols = {
    "symbol",
    "ts",
    TARGET_COL,
}
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise SystemExit(f"dataset missing required columns: {missing_required}")

df = df[df[TARGET_COL].isin(TARGET_MAP_BIN.keys())].copy()
df[TARGET_BIN_COL] = df[TARGET_COL].map(TARGET_MAP_BIN)
df = df.dropna(subset=[TARGET_BIN_COL]).copy()
df[TARGET_BIN_COL] = df[TARGET_BIN_COL].astype(int)

if len(df) == 0:
    raise SystemExit(f"no rows after filtering {TARGET_COL}")

df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

feature_cols = build_feature_cols(df)

SAFE_SUSPICIOUS_EXACT = {
    "body_pct_rng",
    "body_to_prev",
    "day_of_week",
    "prev_day_close",
    "prev_day_range",
    "prev_day_ret",
}

SAFE_SUSPICIOUS_PREFIX = (
    "prev_day_",
)

SAFE_SUSPICIOUS_CONTAINS = (
    "day_of_week",
    "body_pct_rng",
    "body_to_prev",
)

SUSPICIOUS_EXACT = {
    "y",
    "y_fast",
    "pnl_net",
    "entry_px",
    "exit_px",
    "tp_px",
    "sl_px",
    "edge_atr_clean",
    "abs_edge_atr_clean",
    "first_up_hit_bar",
    "first_dn_hit_bar",
    "mfe_up_atr_16h",
    "mfe_dn_atr_16h",
    "y_dir",
    "y_dir_int",
    "y_dir_mfe",
    "y_dir_first",
    "y_side_clean",
    "y_side_clean_int",
}

SUSPICIOUS_PREFIX = (
    "mfe_",
    "mae_",
    "edge_delta_",
    "abs_edge_delta_",
    "y_side_clean_delta_",
    "is_long_delta_",
    "is_short_delta_",
    "is_ambig_delta_",
    "future_",
    "fwd_",
    "target_",
    "label_",
    "realized_",
    "outcome_",
)

SUSPICIOUS_CONTAINS = (
    "future",
    "fwd",
    "target",
    "label",
    "realized",
    "outcome",
    "lookahead",
    "leak",
    "pnl",
)

suspicious_features = []
for c in feature_cols:
    c_low = str(c).lower()

    if c in SAFE_SUSPICIOUS_EXACT:
        continue
    if any(c.startswith(prefix) for prefix in SAFE_SUSPICIOUS_PREFIX):
        continue
    if any(token in c_low for token in SAFE_SUSPICIOUS_CONTAINS):
        continue

    is_suspicious = (
        (c in SUSPICIOUS_EXACT)
        or any(c.startswith(prefix) for prefix in SUSPICIOUS_PREFIX)
        or any(token in c_low for token in SUSPICIOUS_CONTAINS)
    )

    if is_suspicious:
        suspicious_features.append(c)

if suspicious_features:
    raise SystemExit(
        "suspicious features survived leak filter: "
        + ", ".join(sorted(suspicious_features))
    )

if len(feature_cols) == 0:
    raise SystemExit("no numeric feature columns after leak filter")

print("FINAL FEATURE COUNT AFTER LEAK FILTER:", len(feature_cols))
print("FIRST 80 FEATURES:")
for c in feature_cols[:80]:
    print(c)
print()

train_df, valid_df, split_source = split_train_valid(df)

if len(train_df) == 0 or len(valid_df) == 0:
    raise SystemExit("empty train or valid split")

if train_df[TARGET_BIN_COL].nunique() < 2:
    raise SystemExit("train split has fewer than 2 classes")

if valid_df[TARGET_BIN_COL].nunique() < 2:
    raise SystemExit("valid split has fewer than 2 classes")

x_train, x_valid, y_train, y_valid = prepare_xy(
    train_df=train_df,
    valid_df=valid_df,
    feature_cols=feature_cols,
    target_col=TARGET_BIN_COL,
)

model = CatBoostClassifier(**CATBOOST_PARAMS)
model.fit(
    x_train,
    y_train,
    eval_set=(x_valid, y_valid),
    use_best_model=True,
)

valid_proba = model.predict_proba(x_valid)[:, 1]  # вероятность LONG
valid_pred_int = (valid_proba >= 0.5).astype(int)

label_map = {0: "SHORT", 1: "LONG"}
valid_pred_label = pd.Series(valid_pred_int).map(label_map).to_numpy()

best_metrics = build_metrics(y_valid, valid_pred_int)

cm = confusion_matrix(y_valid, valid_pred_int, labels=[0, 1])
cls_report = classification_report(
    y_valid,
    valid_pred_int,
    target_names=TARGET_NAMES_BIN,
    labels=[0, 1],
    zero_division=0,
    output_dict=True,
)

valid_pred = valid_df[[c for c in KEEP_META_COLS_FOR_VALID if c in valid_df.columns]].copy()
valid_pred["true_y_side_clean"] = valid_df[TARGET_COL].to_numpy()
valid_pred["true_y_int"] = y_valid.to_numpy(dtype=int)
valid_pred["proba_long"] = valid_proba
valid_pred["proba_short"] = 1.0 - valid_proba
valid_pred["confidence"] = np.abs(valid_pred["proba_long"] - 0.5) * 2.0
valid_pred["pred_y_int"] = valid_pred_int
valid_pred["pred_y_side_clean"] = valid_pred_label
valid_pred["is_correct"] = (valid_pred["pred_y_int"] == valid_pred["true_y_int"]).astype(int)

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False).reset_index(drop=True)

model.save_model(OUT_MODEL)
feature_importance.to_csv(OUT_FEATURES_CSV, index=False)
valid_pred.to_csv(OUT_VALID_PRED_CSV, index=False)
pd.DataFrame([best_metrics]).to_csv(OUT_METRICS_CSV, index=False)

report = {
    "dataset_path": DATASET_PARQUET,
    "target_col": TARGET_COL,
    "target_int_col": TARGET_BIN_COL,
    "rows_total": int(len(df)),
    "rows_train": int(len(train_df)),
    "rows_valid": int(len(valid_df)),
    "split_source": split_source,
    "feature_count": int(len(feature_cols)),
    "features": feature_cols,
    "class_distribution_total": {
    "SHORT": int((df[TARGET_COL] == "SHORT").sum()),
    "LONG": int((df[TARGET_COL] == "LONG").sum()),
},
    "class_distribution_train": {
    "SHORT": int((train_df[TARGET_COL] == "SHORT").sum()),
    "LONG": int((train_df[TARGET_COL] == "LONG").sum()),
},
    "class_distribution_valid": {
    "SHORT": int((valid_df[TARGET_COL] == "SHORT").sum()),
    "LONG": int((valid_df[TARGET_COL] == "LONG").sum()),
},
    "best_metrics": best_metrics,
    "confusion_matrix_labels": TARGET_NAMES_BIN,
    "confusion_matrix": cm.tolist(),
    "classification_report": cls_report,
    "catboost_params": CATBOOST_PARAMS,
    "files": {
        "model": OUT_MODEL,
        "features": OUT_FEATURES_CSV,
        "valid_predictions": OUT_VALID_PRED_CSV,
        "metrics": OUT_METRICS_CSV,
    },
}

with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("WROTE", OUT_MODEL)
print("WROTE", OUT_FEATURES_CSV)
print("WROTE", OUT_VALID_PRED_CSV)
print("WROTE", OUT_METRICS_CSV)
print("WROTE", OUT_REPORT_JSON)
print()

print("ROWS TOTAL:", len(df))
print("ROWS TRAIN:", len(train_df))
print("ROWS VALID:", len(valid_df))
print("SPLIT SOURCE:", split_source)
print("FEATURE COUNT:", len(feature_cols))
print()

print("TARGET TOTAL")
print(df[TARGET_COL].value_counts(dropna=False).to_string())
print()

print("BEST METRICS")
print(pd.DataFrame([best_metrics]).to_string(index=False))
print()

print("CONFUSION MATRIX 2x2 [SHORT, LONG]")
print(pd.DataFrame(
    cm,
    index=["true_SHORT", "true_LONG"],
    columns=["pred_SHORT", "pred_LONG"],
).to_string())
print()

print()

print("VALID PRED DISTRIBUTION")
print(valid_pred["pred_y_side_clean"].value_counts(dropna=False).to_string())
print()

print("TOP FEATURES")
print(feature_importance.head(50).to_string(index=False))