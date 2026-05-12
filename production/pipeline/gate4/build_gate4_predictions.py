from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

DATASET_PARQUET = ROOT / "production/dataset/gate4/gate4_1_side_builder/gate4_1_side_dataset.parquet"

MODEL_DIR = ROOT / "production/models/gate4/gate4_y_side_clean_multiclass"
MODEL_PATH = MODEL_DIR / "gate4_y_side_clean_multiclass.cbm"
FEATURES_CSV = MODEL_DIR / "features.csv"
REPORT_JSON = MODEL_DIR / "report.json"

OUT_PREDICTIONS_CSV = MODEL_DIR / "all_predictions_raw.csv"
OUT_BUILD_REPORT_JSON = MODEL_DIR / "all_predictions_raw_build_report.json"


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")


def normalize_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)


def load_feature_list(path: Path) -> list[str]:
    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise RuntimeError(f"{path}: missing column 'feature', columns={list(df.columns)}")

    features = df["feature"].dropna().astype(str).tolist()
    features = list(dict.fromkeys(features))

    if not features:
        raise RuntimeError(f"{path}: empty feature list")

    return features


def prepare_x(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"dataset missing model features: {missing[:50]} total={len(missing)}")

    x = df[feature_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(x[c]):
            raise RuntimeError(f"non-numeric feature in Gate4 feature list: {c}, dtype={x[c].dtype}")

    x = x.fillna(0.0)
    return x


def detect_model_mode(model: CatBoostClassifier, proba: np.ndarray) -> str:
    if proba.ndim != 2:
        raise RuntimeError(f"bad predict_proba shape: {proba.shape}")

    if proba.shape[1] == 2:
        return "binary"

    if proba.shape[1] == 3:
        return "multiclass_3"

    raise RuntimeError(f"unsupported class count in predict_proba: {proba.shape[1]}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    require_file(DATASET_PARQUET)
    require_file(MODEL_PATH)
    require_file(FEATURES_CSV)

    df = pd.read_parquet(DATASET_PARQUET)

    required_meta = ["ts", "symbol"]
    missing_meta = [c for c in required_meta if c not in df.columns]
    if missing_meta:
        raise RuntimeError(f"dataset missing meta cols: {missing_meta}")

    if "upstream_split" not in df.columns:
        df["upstream_split"] = "unknown"

    if "upstream_is_oos" not in df.columns:
        df["upstream_is_oos"] = np.nan

    df["ts"] = normalize_ts(df["ts"])
    df["symbol"] = df["symbol"].astype(str)

    bad_ts = int(df["ts"].isna().sum())
    if bad_ts:
        raise RuntimeError(f"bad ts rows: {bad_ts}")

    feature_cols = load_feature_list(FEATURES_CSV)
    x = prepare_x(df, feature_cols)

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    model_feats = list(dict.fromkeys(model.feature_names_ or []))
    if not model_feats:
        raise RuntimeError("model.feature_names_ is empty")

    missing = [c for c in model_feats if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing model features in dataset: {missing[:50]} total={len(missing)}")

    x = df[model_feats].copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    med = x.median(numeric_only=True)
    x = x.fillna(med).fillna(0.0)

    proba = model.predict_proba(x)
    mode = detect_model_mode(model, proba)

    if mode == "binary":
        proba_short = proba[:, 0]
        proba_long = proba[:, 1]
        pred_label = pd.Series(model.predict(x)).astype(str)

        out = pd.DataFrame({
            "ts": df["ts"],
            "symbol": df["symbol"],
            "upstream_split": df["upstream_split"],
            "upstream_is_oos": df["upstream_is_oos"],
            "pred_label": pred_label,
            "proba_0": proba_short,
            "proba_1": proba_long,
            "proba_long": proba_long,
            "proba_short": proba_short,
        })

    else:
        proba_short = proba[:, 0]
        proba_ambig = proba[:, 1]
        proba_long = proba[:, 2]
        pred_label = np.argmax(proba, axis=1).astype(int)

        out = pd.DataFrame({
            "ts": df["ts"],
            "symbol": df["symbol"],
            "upstream_split": df["upstream_split"],
            "upstream_is_oos": df["upstream_is_oos"],
            "pred_label": pred_label,
            "proba_0": proba_short,
            "proba_1": proba_ambig,
            "proba_2": proba_long,
            "proba_short": proba_short,
            "proba_ambig": proba_ambig,
            "proba_long": proba_long,
        })

    dup_count = int(out.duplicated(["symbol", "ts"]).sum())
    if dup_count:
        raise RuntimeError(f"duplicated symbol+ts rows in output: {dup_count}")

    OUT_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PREDICTIONS_CSV, index=False)

    report = {
        "dataset_parquet": str(DATASET_PARQUET),
        "model_path": str(MODEL_PATH),
        "features_csv": str(FEATURES_CSV),
        "out_predictions_csv": str(OUT_PREDICTIONS_CSV),
        "rows": int(len(out)),
        "feature_count": int(len(model_feats)),
        "model_mode": mode,
        "proba_shape": list(proba.shape),
        "duplicates_symbol_ts": dup_count,
        "ts_min": str(out["ts"].min()),
        "ts_max": str(out["ts"].max()),
        "columns": list(out.columns),
    }

    with open(OUT_BUILD_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("WROTE:", OUT_PREDICTIONS_CSV)
    print("WROTE:", OUT_BUILD_REPORT_JSON)
    print()
    print("ROWS:", len(out))
    print("MODEL MODE:", mode)
    print("PROBA SHAPE:", proba.shape)
    print("FEATURE COUNT:", len(model_feats))
    print("COLUMNS:")
    print(list(out.columns))
    print()
    print("SPLIT:")
    print(out["upstream_split"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()