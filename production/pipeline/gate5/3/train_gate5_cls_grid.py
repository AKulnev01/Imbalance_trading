from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


ROOT = Path(".")
DATA_DIR = ROOT / "production/dataset/gate5/gate5_3"
OUT_DIR = ROOT / "production/models/gate5/gate5_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "summary.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns: {missing}")


def build_feature_cols(df: pd.DataFrame) -> list[str]:
    drop_exact = {
        "signal_id",
        "ts",
        "symbol",
        "gate5_split",
        "gate5_is_oos",
        "y",
        "delta_score",

        "safe_target_score",
        "agg_target_score",
        "safe_is_better",
        "agg_is_better",

        "safe_grid_name",
        "agg_grid_name",

        "entry_ts_exec",
        "entry_px_ref",
        "entry_delay_seconds",
    }

    drop_contains = (
        "target",
        "label",
        "future",
        "lookahead",
        "realized",
        "oracle",
        "winner",
        "pnl",
        "profit",
        "loss",
        "ret_",
        "_ret",
        "mfe",
        "mae",
        "first_tp",
        "first_sl",
        "tp_hit",
        "sl_hit",
        "tp_before_sl",
        "sl_before_tp",
        "ambiguous",
        "no_hit",
        "exit_",
        "_exit",
        "hold_minutes",
        "duration",
        "minute",
        "time_to",
        "bars_to",
        "exec",
    )

    cols: list[str] = []

    for c in df.columns:
        c_low = c.lower()

        if c in drop_exact:
            continue

        if any(token in c_low for token in drop_contains):
            continue

        if c_low.startswith("g5_"):
            continue
        if c_low.startswith("safe_g5_"):
            continue
        if c_low.startswith("agg_g5_"):
            continue

        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue

        if not (
            pd.api.types.is_bool_dtype(df[c])
            or pd.api.types.is_numeric_dtype(df[c])
            or pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
        ):
            continue

        cols.append(c)

    return sorted(cols)


def compute_economic_metrics(valid_pred_df: pd.DataFrame) -> dict:
    require_cols(
        valid_pred_df,
        [
            "pred_label",
            "safe_target_score",
            "agg_target_score",
        ],
        "valid_pred_df",
    )

    chosen_score = np.where(
        valid_pred_df["pred_label"].to_numpy(dtype=int) == 1,
        valid_pred_df["agg_target_score"].to_numpy(dtype=float),
        valid_pred_df["safe_target_score"].to_numpy(dtype=float),
    )

    return {
        "mean_score_model_choice": float(np.mean(chosen_score)),
        "mean_score_always_safe": float(valid_pred_df["safe_target_score"].mean()),
        "mean_score_always_agg": float(valid_pred_df["agg_target_score"].mean()),
        "uplift_vs_safe": float(np.mean(chosen_score) - valid_pred_df["safe_target_score"].mean()),
        "uplift_vs_agg": float(np.mean(chosen_score) - valid_pred_df["agg_target_score"].mean()),
        "model_choose_agg_share": float(valid_pred_df["pred_label"].mean()),
    }


def train_one(path: Path) -> dict:
    df = pd.read_parquet(path)

    require_cols(
        df,
        [
            "signal_id",
            "ts",
            "symbol",
            "side",
            "gate5_split",
            "safe_target_score",
            "agg_target_score",
            "y",
            "safe_grid_name",
            "agg_grid_name",
        ],
        f"dataset[{path.name}]",
    )

    train = df[df["gate5_split"] == "train"].copy()
    valid = df[df["gate5_split"] == "valid"].copy()

    if len(train) == 0 or len(valid) == 0:
        raise RuntimeError(f"{path.name}: empty train/valid split")

    train_max_ts = pd.to_datetime(train["ts"], errors="coerce").max()
    valid_min_ts = pd.to_datetime(valid["ts"], errors="coerce").min()
    if not (pd.notna(train_max_ts) and pd.notna(valid_min_ts) and train_max_ts < valid_min_ts):
        raise RuntimeError(
            f"{path.name}: bad time split train_max_ts={train_max_ts} valid_min_ts={valid_min_ts}"
        )
    if train["y"].nunique() < 2:
        raise RuntimeError(f"{path.name}: train target has <2 classes")
    if valid["y"].nunique() < 2:
        raise RuntimeError(f"{path.name}: valid target has <2 classes")

    X_cols = build_feature_cols(df)
    if len(X_cols) == 0:
        raise RuntimeError(f"{path.name}: no features left after leakage filter")
    suspicious_cols = [
        c for c in X_cols
        if (
                c == "y"
                or "target" in c.lower()
                or "label" in c.lower()
                or "future" in c.lower()
                or "lookahead" in c.lower()
                or "realized" in c.lower()
                or "oracle" in c.lower()
                or "winner" in c.lower()
                or "pnl" in c.lower()
                or "profit" in c.lower()
                or "loss" in c.lower()
                or "mfe" in c.lower()
                or "mae" in c.lower()
                or "first_tp" in c.lower()
                or "first_sl" in c.lower()
                or "tp_hit" in c.lower()
                or "sl_hit" in c.lower()
                or "tp_before_sl" in c.lower()
                or "sl_before_tp" in c.lower()
                or "ambiguous" in c.lower()
                or "no_hit" in c.lower()
                or "minute" in c.lower()
                or "time_to" in c.lower()
                or "bars_to" in c.lower()
                or "exec" in c.lower()
                or c.lower().startswith("g5_")
                or c.lower().startswith("safe_g5_")
                or c.lower().startswith("agg_g5_")
                or c in {"entry_ts_exec", "entry_px_ref", "entry_delay_seconds"}
        )
    ]
    if suspicious_cols:
        raise RuntimeError(
            "Leakage columns survived feature filter: " + ", ".join(sorted(suspicious_cols))
        )

    cat_cols = [c for c in ["side"] if c in X_cols]

    X_train = train[X_cols].copy()
    y_train = train["y"].astype(int)

    X_valid = valid[X_cols].copy()
    y_valid = valid["y"].astype(int)

    for c in X_cols:
        if c not in cat_cols:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_valid[c] = pd.to_numeric(X_valid[c], errors="coerce")

    X_train = X_train.fillna(0.0)
    X_valid = X_valid.fillna(0.0)

    model = CatBoostClassifier(
        iterations=1500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=200,
        od_type="Iter",
        od_wait=150,
        random_seed=42,
        thread_count=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_cols,
    )

    proba_valid = model.predict_proba(X_valid)[:, 1]
    pred_valid = (proba_valid >= 0.5).astype(int)

    auc_valid = roc_auc_score(y_valid, proba_valid)

    valid_pred_df = valid[
        [
            "signal_id",
            "ts",
            "symbol",
            "side",
            "safe_grid_name",
            "agg_grid_name",
            "safe_target_score",
            "agg_target_score",
            "delta_score",
            "y",
        ]
    ].copy()
    valid_pred_df["pred_proba"] = proba_valid
    valid_pred_df["pred_label"] = pred_valid

    econ = compute_economic_metrics(valid_pred_df)

    name = path.stem
    model_dir = OUT_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{name}.cbm"
    valid_pred_path = model_dir / "valid_predictions.parquet"
    features_path = model_dir / "features.csv"
    report_path = model_dir / "report.json"

    model.save_model(model_path)
    valid_pred_df.to_parquet(valid_pred_path, index=False)
    pd.DataFrame({"feature": X_cols}).to_csv(features_path, index=False)
    print("FEATURES USED:", len(X_cols))
    print(pd.Series(X_cols).to_string(index=False))

    report = {
        "pair_name": name,
        "safe_grid": str(df["safe_grid_name"].iloc[0]),
        "agg_grid": str(df["agg_grid_name"].iloc[0]),
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_valid": int(len(valid)),
        "target_mean_train": float(y_train.mean()),
        "target_mean_valid": float(y_valid.mean()),
        "auc_valid": float(auc_valid),
        "feature_count": int(len(X_cols)),
        "model_path": str(model_path),
        "valid_predictions_path": str(valid_pred_path),
        "features_path": str(features_path),
        **econ,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"DONE: {name} | "
        f"AUC={auc_valid:.6f} | "
        f"model={econ['mean_score_model_choice']:.6f} | "
        f"safe={econ['mean_score_always_safe']:.6f} | "
        f"agg={econ['mean_score_always_agg']:.6f} | "
        f"uplift_vs_safe={econ['uplift_vs_safe']:.6f} | "
        f"uplift_vs_agg={econ['uplift_vs_agg']:.6f}"
    )

    return report


def main() -> None:
    paths = sorted(
        p for p in DATA_DIR.glob("*.parquet")
        if not p.name.startswith("_")
    )

    if not paths:
        raise RuntimeError(f"No datasets found in {DATA_DIR}")

    reports = []

    for path in paths:
        print("TRAIN:", path.name)
        try:
            rep = train_one(path)
            reports.append(rep)
        except Exception as e:
            print(f"FAIL: {path.name} | {e}")
            reports.append({
                "pair_name": path.stem,
                "error": str(e),
            })

    summary_df = pd.DataFrame(reports)

    sort_cols = [c for c in ["uplift_vs_safe", "auc_valid"] if c in summary_df.columns]
    if sort_cols:
        summary_df = summary_df.sort_values(
            sort_cols,
            ascending=[False, False][:len(sort_cols)],
        ).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_CSV, index=False)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print()
    print("WROTE:", SUMMARY_CSV)
    print("WROTE:", SUMMARY_JSON)


if __name__ == "__main__":
    main()