from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

DATASETS = sorted(
    p.stem.replace("gate5_dataset_", "")
    for p in (ROOT / "production/dataset/gate5/gate5_pair_datasets").glob("gate5_dataset_*.parquet")
)

print("DATASETS FOUND:", len(DATASETS))
print("\n".join(DATASETS))

DATA_DIR = ROOT / "production/dataset/gate5/gate5_pair_datasets"
OUT_DIR = ROOT / "production/models/gate5/v1"

OUT_REPORT = OUT_DIR / "report.json"
OUT_SUMMARY_CSV = OUT_DIR / "summary.csv"
OUT_SWEEP_CSV = OUT_DIR / "threshold_sweep.csv"


# ============================================================
# CONFIG
# ============================================================

TARGET_TEMPLATE = "g5_target_{pair}"

CAT_COLS = []

SPLIT_COL = "gate5_split"
THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.991, 0.01)]
MIN_KEPT = 20


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_feature_cols(df: pd.DataFrame, pair: str) -> list[str]:
    target_col = TARGET_TEMPLATE.format(pair=pair)

    DROP_EXACT = {
        "ts",
        "symbol",
        "pred_side",
        "gate5_split",
        "gate5_is_oos",
        "upstream_split",
        "upstream_valid_start_ts",
        "upstream_is_oos",
        "g3_any_active",
        "g3_both_active",
        "g3_long_active",
        "g3_short_active",
        "gate1_pass",
        target_col,
    }

    DROP_CONTAINS = (
        ".1",
        "pred_y",
        "true_y",
        "is_correct",
        "future",
        "fwd",
        "lookahead",
        "outcome",
        "realized",
        "label",
    )

    pair_prefixes = (
        f"g5_target_{pair}",
        f"g5_mfe_side_atr_{pair}",
        f"g5_mae_side_atr_{pair}",
        f"g5_ttl_ret_side_atr_{pair}",
        f"g5_first_tp_minute_{pair}",
        f"g5_first_sl_minute_{pair}",
        f"g5_first_tp_bar_{pair}",
        f"g5_first_sl_bar_{pair}",
        f"g5_tp_hit_{pair}",
        f"g5_sl_hit_{pair}",
        f"g5_tp_before_sl_{pair}",
        f"g5_sl_before_tp_{pair}",
        f"g5_ambiguous_same_bar_{pair}",
        f"g5_no_hit_{pair}",
    )

    cols: list[str] = []

    for c in df.columns:
        c_low = c.lower()

        if c in DROP_EXACT:
            continue

        if c.endswith("_g4"):
            continue

        if c.startswith(pair_prefixes):
            continue

        if any(token in c_low for token in DROP_CONTAINS):
            continue

        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue

        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            continue

        if not (
            pd.api.types.is_bool_dtype(df[c]) or
            pd.api.types.is_numeric_dtype(df[c])
        ):
            continue

        cols.append(c)

    return sorted(cols)


def split_df(df: pd.DataFrame):
    if SPLIT_COL not in df.columns:
        raise RuntimeError(f"Missing required split column: {SPLIT_COL}")

    train = df[df[SPLIT_COL] == "train"].copy()
    valid = df[df[SPLIT_COL] == "valid"].copy()

    if len(train) == 0 or len(valid) == 0:
        raise RuntimeError(
            f"Bad split by {SPLIT_COL}: rows_train={len(train)} rows_valid={len(valid)}"
        )

    train_max_ts = pd.to_datetime(train["ts"], errors="coerce").max()
    valid_min_ts = pd.to_datetime(valid["ts"], errors="coerce").min()

    if not (pd.notna(train_max_ts) and pd.notna(valid_min_ts) and train_max_ts < valid_min_ts):
        raise RuntimeError(
            f"Time overlap in {SPLIT_COL}: train_max_ts={train_max_ts}, valid_min_ts={valid_min_ts}"
        )

    return train, valid
def parse_pair_tp_sl(pair: str) -> tuple[float, float]:
    left, right = pair.split("_")
    tp_atr = float(left.replace("tp", "")) / 100.0
    sl_atr = float(right.replace("sl", "")) / 100.0
    return tp_atr, sl_atr

def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2*n)
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)
    return (centre - margin) / denom

def build_threshold_sweep(
    pair: str,
    y_valid: pd.Series,
    proba_valid: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    tp_atr, sl_atr = parse_pair_tp_sl(pair)

    rows = []

    y_valid_np = y_valid.to_numpy(dtype=int)
    total_n = len(y_valid_np)

    for thr in THRESHOLDS:
        mask = proba_valid >= thr
        kept = int(mask.sum())

        if kept == 0:
            rows.append({
                "pair": pair,
                "tp_atr": tp_atr,
                "sl_atr": sl_atr,
                "rr": float(tp_atr / sl_atr) if sl_atr > 0 else np.nan,
                "thr": float(thr),
                "kept": 0,
                "coverage": 0.0,
                "precision": np.nan,
                "ev_proxy": np.nan,
            })
            continue

        y_kept = y_valid_np[mask]
        k = int(y_kept.sum())
        n = int(len(y_kept))

        precision = float(y_kept.mean())
        precision_lb = float(wilson_lower_bound(k, n))
        coverage = float(kept / total_n)
        ev_proxy = float(precision_lb * tp_atr - (1.0 - precision_lb) * sl_atr)

        rows.append({
            "pair": pair,
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
            "rr": float(tp_atr / sl_atr) if sl_atr > 0 else np.nan,
            "thr": float(thr),
            "kept": kept,
            "coverage": coverage,
            "precision": precision,
            "ev_proxy": ev_proxy,
        })

    sweep_df = pd.DataFrame(rows)

    eligible = sweep_df[
        (sweep_df["kept"] >= MIN_KEPT) &
        (sweep_df["precision"].notna()) &
        (sweep_df["ev_proxy"].notna())
    ].copy()

    if len(eligible) == 0:
        best_row = {
            "best_thr_by_ev": None,
            "best_kept_by_ev": None,
            "best_coverage_by_ev": None,
            "best_precision_by_ev": None,
            "best_ev_proxy": None,
        }
    else:
        eligible = eligible.sort_values(
            ["ev_proxy", "precision", "kept", "thr"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        top = eligible.iloc[0]

        best_row = {
            "best_thr_by_ev": float(top["thr"]),
            "best_kept_by_ev": int(top["kept"]),
            "best_coverage_by_ev": float(top["coverage"]),
            "best_precision_by_ev": float(top["precision"]),
            "best_ev_proxy": float(top["ev_proxy"]),
        }

    return sweep_df, best_row


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUT_DIR)

    report = []
    all_sweeps = []

    print("DATASETS TO TRAIN:", len(DATASETS))
    for ds in DATASETS:
        print(ds)

    for pair in DATASETS:
        print(f"\n=== TRAIN {pair} ===")

        path = DATA_DIR / f"gate5_dataset_{pair}.parquet"
        if not path.exists():
            print("SKIP: no file", path)
            continue

        df = pd.read_parquet(path)

        target_col = TARGET_TEMPLATE.format(pair=pair)

        required_core_cols = {
            "ts",
            "symbol",
            "close",
            "atr14",
            "proba_long",
            "proba_short",
            "pred_side_int",
            "pred_side_confidence",
            "pred_side_gap",
            "pred_side_ratio",
            "gate4_confidence",
            "gate5_split",
            "gate5_is_oos",
            target_col,
        }
        missing_core_cols = sorted(c for c in required_core_cols if c not in df.columns)
        if missing_core_cols:
            raise RuntimeError(
                f"{pair}: missing required columns: {missing_core_cols}"
            )

        if target_col not in df.columns:
            print("SKIP: no target", target_col)
            continue

        df = df.dropna(subset=[target_col]).copy()

        y = df[target_col].astype(int)

        # sanity
        pos_rate = y.mean()
        print("rows:", len(df), "pos_rate:", round(pos_rate, 4))

        if pos_rate < 0.05 or pos_rate > 0.95:
            print("SKIP: degenerate target")
            continue

        X_cols = build_feature_cols(df, pair)

        suspicious_cols = [
            c for c in X_cols
            if (
                    c == target_col
                    or c == "pred_side"
                    or c == "gate1_pass"
                    or c in {"g3_any_active", "g3_both_active", "g3_long_active", "g3_short_active"}
                    or c.startswith(f"g5_target_{pair}")
                    or c.startswith(f"g5_mfe_side_atr_{pair}")
                    or c.startswith(f"g5_mae_side_atr_{pair}")
                    or c.startswith(f"g5_ttl_ret_side_atr_{pair}")
                    or c.startswith(f"g5_first_tp_minute_{pair}")
                    or c.startswith(f"g5_first_sl_minute_{pair}")
                    or c.startswith(f"g5_first_tp_bar_{pair}")
                    or c.startswith(f"g5_first_sl_bar_{pair}")
                    or c.startswith(f"g5_tp_hit_{pair}")
                    or c.startswith(f"g5_sl_hit_{pair}")
                    or c.startswith(f"g5_tp_before_sl_{pair}")
                    or c.startswith(f"g5_sl_before_tp_{pair}")
                    or c.startswith(f"g5_ambiguous_same_bar_{pair}")
                    or c.startswith(f"g5_no_hit_{pair}")
                    or ".1" in c
                    or "pred_y" in c.lower()
                    or "true_y" in c.lower()
                    or "is_correct" in c.lower()
            )
        ]

        if suspicious_cols:
            raise RuntimeError(
                "Suspicious feature columns survived filter: "
                + ", ".join(sorted(suspicious_cols))
            )

        train_df, valid_df = split_df(df)

        print("rows_train:", len(train_df), "rows_valid:", len(valid_df))
        print("train_pos_rate:", round(train_df[target_col].mean(), 4) if len(train_df) else None)
        print("valid_pos_rate:", round(valid_df[target_col].mean(), 4) if len(valid_df) else None)

        if len(X_cols) == 0:
            raise RuntimeError(f"{pair}: no features left after filtering")

        print("FEATURE COUNT:", len(X_cols))
        print("FEATURES USED:")
        for f in X_cols:
            print(f)

        X_train = train_df[X_cols].copy()
        y_train = train_df[target_col].astype(int)

        X_valid = valid_df[X_cols].copy()
        y_valid = valid_df[target_col].astype(int)

        X_train = X_train.fillna(0.0)
        X_valid = X_valid.fillna(0.0)

        if len(X_train) == 0 or len(y_train) == 0:
            print("SKIP: empty train split")
            continue

        if len(X_valid) == 0 or len(y_valid) == 0:
            print("SKIP: empty valid split")
            continue

        if y_train.nunique() < 2:
            print("SKIP: train target has <2 classes")
            continue

        if y_valid.nunique() < 2:
            print("SKIP: valid target has <2 classes")
            continue

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=6.0,
            random_seed=42,
            verbose=200,
            od_type="Iter",
            od_wait=200,
            thread_count=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=[c for c in CAT_COLS if c in X_cols],
        )

        proba_valid = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, proba_valid)

        valid_pred_df = valid_df[["ts", "symbol", "gate5_split", "gate5_is_oos"]].copy()
        valid_pred_df["target"] = y_valid.to_numpy(dtype=int)
        valid_pred_df["proba"] = proba_valid
        valid_pred_df["pred"] = (valid_pred_df["proba"] >= 0.5).astype(int)

        valid_pred_path = OUT_DIR / f"valid_predictions_{pair}.csv"
        valid_pred_df.to_csv(valid_pred_path, index=False)

        print("AUC:", round(auc, 4))

        sweep_df, best_by_ev = build_threshold_sweep(
            pair=pair,
            y_valid=y_valid,
            proba_valid=proba_valid,
        )
        all_sweeps.append(sweep_df)

        print(
            "BEST BY EV:",
            {
                "thr": best_by_ev["best_thr_by_ev"],
                "kept": best_by_ev["best_kept_by_ev"],
                "coverage": best_by_ev["best_coverage_by_ev"],
                "precision": best_by_ev["best_precision_by_ev"],
                "ev_proxy": best_by_ev["best_ev_proxy"],
            }
        )

        model_path = OUT_DIR / f"model_{pair}.cbm"
        model.save_model(model_path)

        report.append({
            "pair": pair,
            "rows": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_valid": int(len(valid_df)),
            "pos_rate": float(pos_rate),
            "auc_valid": float(auc),
            "best_thr_by_ev": best_by_ev["best_thr_by_ev"],
            "best_kept_by_ev": best_by_ev["best_kept_by_ev"],
            "best_coverage_by_ev": best_by_ev["best_coverage_by_ev"],
            "best_precision_by_ev": best_by_ev["best_precision_by_ev"],
            "best_ev_proxy": best_by_ev["best_ev_proxy"],
            "model_path": str(model_path),
            "valid_predictions_path": str(valid_pred_path),
        })

    report_df = pd.DataFrame(report).sort_values(
        ["best_ev_proxy", "auc_valid"],
        ascending=[False, False],
    ).reset_index(drop=True)
    report_df.to_csv(OUT_SUMMARY_CSV, index=False)

    if all_sweeps:
        sweep_all_df = pd.concat(all_sweeps, ignore_index=True)
        sweep_all_df = sweep_all_df.sort_values(
            ["pair", "thr"],
            ascending=[True, True],
        ).reset_index(drop=True)
        sweep_all_df.to_csv(OUT_SWEEP_CSV, index=False)
    else:
        sweep_all_df = pd.DataFrame()

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("\nWROTE:", OUT_REPORT)
    print("WROTE:", OUT_SUMMARY_CSV)
    print("WROTE:", OUT_SWEEP_CSV)

    print()
    print("=== TOP 20 PAIRS BY BEST EV ===")
    print(report_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()