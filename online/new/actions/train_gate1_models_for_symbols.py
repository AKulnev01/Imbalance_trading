from __future__ import annotations

import argparse
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = ROOT / "production" / "dataset" / "gate1"
OUT_ROOT = ROOT / "production" / "models" / "final_gate1"

REPORT_JSON = ROOT / "online" / "new" / "actions" / "_train_gate1_models_for_symbols_report.json"
REPORT_CSV = ROOT / "online" / "new" / "actions" / "_train_gate1_models_for_symbols_report.csv"

MODEL_FILENAME = "gate1_impulse_abs_move_atr_16h.cbm"
META_FILENAME = "meta.json"

RANDOM_SEED = 42

THR_GRID = [round(float(x), 3) for x in np.linspace(0.10, 0.90, 17)]
KEPT_MIN = 0.03
KEPT_MAX = 0.20

DEFAULT_ITERATIONS = 1200
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_DEPTH = 8
DEFAULT_L2_LEAF_REG = 6.0
DEFAULT_OD_WAIT = 100

MIN_TOTAL_ROWS = 200
MIN_TRAIN_ROWS = 100
MIN_VALID_ROWS = 20


def fail(message: str) -> None:
    raise SystemExit(message)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_floor_minute() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("min")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)



def backup_existing_artifact(path: Path, artifact_group: str, symbol: str, report=None) -> str:
    """
    Backup only real data/model artifacts before overwrite.
    This function is intentionally used by new online/new/actions wrappers
    when they are about to overwrite existing production/data artifacts.
    It does NOT create backups of the wrapper script itself.
    """
    p = Path(path)

    if not p.exists():
        return ""

    symbol_safe = str(symbol).upper().replace("/", "_").replace("\\", "_")
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")

    backup_dir = (
        ROOT
        / "online"
        / "new"
        / "actions"
        / "_artifact_backups"
        / str(artifact_group)
        / symbol_safe
        / run_id
    )
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_path = backup_dir / p.name

    suffix_i = 1
    while backup_path.exists():
        backup_path = backup_dir / "{}.dup{}{}".format(p.stem, suffix_i, p.suffix)
        suffix_i += 1

    shutil.copy2(str(p), str(backup_path))

    if report is not None:
        try:
            backup_paths = report.setdefault("backup_paths", [])
            backup_paths.append(str(backup_path))
        except Exception:
            pass

    print("    BACKUP_ARTIFACT:", p, "->", backup_path, flush=True)
    return str(backup_path)

def parse_symbols(raw_values: List[str]) -> List[str]:
    out: List[str] = []

    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            symbol = part.strip().upper()
            if symbol:
                out.append(symbol)

    seen = set()
    unique: List[str] = []

    for symbol in out:
        if symbol in seen:
            continue
        unique.append(symbol)
        seen.add(symbol)

    if not unique:
        fail("no symbols provided")

    return unique


def parse_required_ts(value: str, name: str) -> pd.Timestamp:
    ts = pd.to_datetime(str(value).strip(), utc=True, errors="coerce")
    if pd.isna(ts):
        fail("bad {} value: {}".format(name, value))
    return pd.Timestamp(ts)


def parse_optional_ts(value: str, name: str) -> Optional[pd.Timestamp]:
    raw = str(value or "").strip()
    if not raw:
        return None

    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        fail("bad {} value: {}".format(name, value))

    return pd.Timestamp(ts)


def parse_train_end(value: str) -> pd.Timestamp:
    return parse_required_ts(value, "--train-end")


def validate_split_window(
    train_end: pd.Timestamp,
    valid_start: Optional[pd.Timestamp],
    valid_end: Optional[pd.Timestamp],
) -> Tuple[pd.Timestamp, Optional[pd.Timestamp]]:
    if valid_start is None and valid_end is None:
        return train_end, None

    if valid_start is None or valid_end is None:
        fail("--valid-start and --valid-end must be provided together")

    if train_end > valid_start:
        fail(
            "--train-end must be <= --valid-start, got train_end={} valid_start={}".format(
                train_end,
                valid_start,
            )
        )

    if valid_start >= valid_end:
        fail(
            "--valid-start must be < --valid-end, got valid_start={} valid_end={}".format(
                valid_start,
                valid_end,
            )
        )

    return valid_start, valid_end


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = y_true.astype(int)

    if len(y) == 0:
        return float("nan")

    if y.min() == y.max():
        return float("nan")

    order = np.argsort(proba)
    y_sorted = y[order]

    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = np.arange(1, len(y_sorted) + 1)
    s_pos = ranks[y_sorted == 1].sum()
    auc = (s_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)

    return float(auc)


def pick_features(df: pd.DataFrame) -> List[str]:
    drop_exact = {
        "symbol",
        "symbol_id",
        "entry_ts",
        "side",
        "pnl_net",
        "y",
        "y_fast",
        "ks_ret_adj",
        "exit_px",
        "tp_px",
        "sl_px",
        "ttm_min",
        "exit_reason",
        "label_gate1",
        "g1_label",
    }

    drop_name_substr = (
        "ts",
        "time",
        "date",
        "datetime",
        "open_time",
        "close_time",
        "start",
        "end",
    )

    cols: List[str] = []

    for col in df.columns:
        if col in drop_exact:
            continue

        col_l = str(col).lower()

        if any(s in col_l for s in drop_name_substr) and col not in ("day_of_week", "hour_of_day"):
            continue

        if str(col).startswith("ks_"):
            continue

        if str(col).startswith("p_"):
            continue

        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            continue

        if pd.api.types.is_timedelta64_dtype(s):
            continue

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            continue

        if not pd.api.types.is_numeric_dtype(s):
            continue

        if s.nunique(dropna=False) <= 1 and col not in ("atr_to_price",):
            continue

        cols.append(str(col))

    if not cols:
        cols = [
            str(c)
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in ("y",)
        ]

    if not cols:
        raise RuntimeError("feature list is empty")

    return cols


def time_split_by_train_end(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_start: Optional[pd.Timestamp],
    valid_end: Optional[pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = df.copy()
    x["entry_ts"] = pd.to_datetime(x["entry_ts"], utc=True, errors="coerce")
    x = x.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    valid_start_eff, valid_end_eff = validate_split_window(
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
    )

    train = x[x["entry_ts"] < train_end].copy()

    valid_mask = x["entry_ts"] >= valid_start_eff
    if valid_end_eff is not None:
        valid_mask = valid_mask & (x["entry_ts"] < valid_end_eff)

    valid = x[valid_mask].copy()

    return train, valid


def eval_thresholds(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    n = len(y_true)

    y_mean = float(np.mean(y_true)) if n else 0.0

    for thr in THR_GRID:
        kept = proba >= float(thr)
        kept_n = int(kept.sum())
        kept_share = float(kept_n / n) if n else 0.0

        if kept_n > 0:
            precision = float(np.mean(y_true[kept]))
            lift = float(precision / (y_mean + 1e-9))
        else:
            precision = 0.0
            lift = 0.0

        rows.append(
            {
                "thr": float(thr),
                "kept_n": int(kept_n),
                "kept_share": float(kept_share),
                "precision": float(precision),
                "lift": float(lift),
            }
        )

    best = None

    for row in rows:
        if KEPT_MIN <= row["kept_share"] <= KEPT_MAX and row["kept_n"] >= 20:
            if best is None or row["lift"] > best["lift"]:
                best = row

    if best is None:
        target = (KEPT_MIN + KEPT_MAX) / 2.0
        best = min(rows, key=lambda r: abs(r["kept_share"] - target))

    return {
        "best_thr": float(best["thr"]),
        "grid": rows,
    }


def prepare_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "entry_ts" not in df.columns:
        raise RuntimeError("dataset missing entry_ts")

    if "y" not in df.columns:
        raise RuntimeError("dataset missing target y")

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts"]).copy()
    df = df.sort_values("entry_ts", kind="mergesort")
    df = df.drop_duplicates(subset=["entry_ts"], keep="last").reset_index(drop=True)

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    if len(df) < MIN_TOTAL_ROWS:
        raise RuntimeError("too few rows total: {}".format(len(df)))

    return df


def build_sample_weight(train_df: pd.DataFrame) -> np.ndarray:
    if "atr_to_price" not in train_df.columns:
        return np.ones(len(train_df), dtype=float)

    w = pd.to_numeric(train_df["atr_to_price"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    med = np.nanmedian(w)
    if med == 0.0 or np.isnan(med):
        med = 1.0

    w = np.clip(w / med, 0.5, 2.0)

    return w


def train_one_symbol(
    symbol: str,
    train_end: pd.Timestamp,
    valid_start: Optional[pd.Timestamp],
    valid_end: Optional[pd.Timestamp],
    overwrite: bool,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    od_wait: int,
) -> Dict[str, Any]:
    started_at = time.time()

    ds_path = SRC_DIR / "{}.parquet".format(symbol)
    out_dir = OUT_ROOT / symbol / "gate1"
    model_path = out_dir / MODEL_FILENAME
    meta_path = out_dir / META_FILENAME

    report: Dict[str, Any] = {
        "symbol": symbol,
        "status": "OK",
        "error": "",
        "dataset_path": str(ds_path),
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "train_end": str(train_end),
        "valid_start": str(validate_split_window(train_end, valid_start, valid_end)[0]),
        "valid_end": "" if validate_split_window(train_end, valid_start, valid_end)[1] is None else str(validate_split_window(train_end, valid_start, valid_end)[1]),
        "overwrite": bool(overwrite),
    }

    if not ds_path.exists():
        raise FileNotFoundError(str(ds_path))

    if model_path.exists() and not overwrite:
        raise RuntimeError("model already exists; pass --overwrite to replace: {}".format(model_path))

    df = prepare_dataset(ds_path)
    feats = pick_features(df)

    valid_start_eff, valid_end_eff = validate_split_window(
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
    )

    train_df, valid_df = time_split_by_train_end(
        df,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
    )

    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError("too few train rows: {}".format(len(train_df)))

    if len(valid_df) < MIN_VALID_ROWS:
        raise RuntimeError("too few valid rows: {}".format(len(valid_df)))

    if train_df["y"].nunique() < 2:
        raise RuntimeError("train target has one class only")

    if valid_df["y"].nunique() < 2:
        raise RuntimeError("valid target has one class only")

    X_train = train_df[feats]
    y_train = train_df["y"].astype(int).to_numpy()

    X_valid = valid_df[feats]
    y_valid = valid_df["y"].astype(int).to_numpy()

    sample_weight = build_sample_weight(train_df)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        l2_leaf_reg=float(l2_leaf_reg),
        random_seed=RANDOM_SEED,
        verbose=200,
        od_type="Iter",
        od_wait=int(od_wait),
        auto_class_weights="Balanced",
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    p_train = model.predict_proba(X_train)[:, 1]
    p_valid = model.predict_proba(X_valid)[:, 1]

    auc_train = safe_auc(y_train, p_train)
    auc_valid = safe_auc(y_valid, p_valid)

    thr_info = eval_thresholds(y_true=y_valid, proba=p_valid)
    best_thr = float(thr_info["best_thr"])

    kept = p_valid >= best_thr
    kept_n = int(kept.sum())
    kept_share = float(kept_n / len(p_valid)) if len(p_valid) else 0.0
    precision = float(y_valid[kept].mean()) if kept_n else 0.0

    ensure_dir(out_dir)
    backup_existing_artifact(model_path, "gate1_model", symbol, report)
    model.save_model(str(model_path))

    meta = {
        "symbol": symbol,
        "gate": "gate1",
        "trainer": "online.new.actions.train_gate1_models_for_symbols",
        "dataset_path": str(ds_path),
        "model_path": str(model_path),
        "split": {
            "type": "fixed_time",
            "train_condition": "entry_ts < train_end",
            "valid_condition": "valid_start <= entry_ts < valid_end" if valid_end_eff is not None else "entry_ts >= valid_start",
            "train_end": str(train_end),
            "valid_start_ts": str(valid_start_eff),
            "valid_end_ts": None if valid_end_eff is None else str(valid_end_eff),
            "train_min_entry_ts": str(train_df["entry_ts"].min()),
            "train_max_entry_ts": str(train_df["entry_ts"].max()),
            "valid_min_entry_ts": str(valid_df["entry_ts"].min()),
            "valid_max_entry_ts": str(valid_df["entry_ts"].max()),
        },
        "label": {
            "source": "dataset_column_y",
            "type": "next_bar_range",
            "threshold": 0.01,
            "formula": "max((high_next-close)/close, (close-low_next)/close) >= 1%",
        },
        "train": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_valid": int(len(valid_df)),
            "pos_rate_total": float(df["y"].mean()),
            "pos_rate_train": float(train_df["y"].mean()),
            "pos_rate_valid": float(valid_df["y"].mean()),
            "auc_train": float(auc_train),
            "auc_valid": float(auc_valid),
            "random_seed": RANDOM_SEED,
        },
        "catboost": {
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "depth": int(depth),
            "l2_leaf_reg": float(l2_leaf_reg),
            "od_wait": int(od_wait),
            "auto_class_weights": "Balanced",
        },
        "features": {
            "n_features": int(len(feats)),
            "feature_names": feats,
        },
        "thresholding": {
            "best_thr": float(best_thr),
            "kept_min": KEPT_MIN,
            "kept_max": KEPT_MAX,
            "kept_n_valid": int(kept_n),
            "kept_share_valid": float(kept_share),
            "precision_valid": float(precision),
            "thr_grid": THR_GRID,
            "grid_stats": thr_info["grid"],
        },
    }

    backup_existing_artifact(meta_path, "gate1_meta", symbol, report)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    report.update(
        {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_valid": int(len(valid_df)),
            "train_end": str(train_end),
            "valid_start": str(valid_start_eff),
            "valid_end": "" if valid_end_eff is None else str(valid_end_eff),
            "train_min_entry_ts": str(train_df["entry_ts"].min()),
            "train_max_entry_ts": str(train_df["entry_ts"].max()),
            "valid_min_entry_ts": str(valid_df["entry_ts"].min()),
            "valid_max_entry_ts": str(valid_df["entry_ts"].max()),
            "pos_rate_total": float(df["y"].mean()),
            "pos_rate_train": float(train_df["y"].mean()),
            "pos_rate_valid": float(valid_df["y"].mean()),
            "feature_count": int(len(feats)),
            "auc_train": float(auc_train),
            "auc_valid": float(auc_valid),
            "best_thr": float(best_thr),
            "kept_n_valid": int(kept_n),
            "kept_share_valid": float(kept_share),
            "precision_valid": float(precision),
            "elapsed_sec": round(time.time() - started_at, 3),
        }
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Gate1 Models For Symbols. "
            "Reads production/dataset/gate1/<SYMBOL>.parquet and writes "
            "production/models/final_gate1/<SYMBOL>/gate1/gate1_impulse_abs_move_atr_16h.cbm. "
            "Split is controlled by required --train-end and optional --valid-start/--valid-end."
        )
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--valid-start", default="")
    parser.add_argument("--valid-end", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--l2-leaf-reg", type=float, default=DEFAULT_L2_LEAF_REG)
    parser.add_argument("--od-wait", type=int, default=DEFAULT_OD_WAIT)
    args = parser.parse_args()

    started_at = time.time()

    global SRC_DIR
    global OUT_ROOT

    if str(args.dataset_dir).strip():
        dataset_dir_arg = Path(str(args.dataset_dir).strip())
        SRC_DIR = dataset_dir_arg if dataset_dir_arg.is_absolute() else ROOT / dataset_dir_arg

    if str(args.out_root).strip():
        out_root_arg = Path(str(args.out_root).strip())
        OUT_ROOT = out_root_arg if out_root_arg.is_absolute() else ROOT / out_root_arg

    symbols = parse_symbols(args.symbols)
    train_end = parse_train_end(args.train_end)
    valid_start_arg = parse_optional_ts(args.valid_start, "--valid-start")
    valid_end_arg = parse_optional_ts(args.valid_end, "--valid-end")
    valid_start, valid_end = validate_split_window(
        train_end=train_end,
        valid_start=valid_start_arg,
        valid_end=valid_end_arg,
    )

    ensure_dir(OUT_ROOT)
    ensure_dir(REPORT_JSON.parent)

    print("Train Gate1 Models For Symbols")
    print("ROOT:", ROOT)
    print("SRC_DIR:", SRC_DIR)
    print("OUT_ROOT:", OUT_ROOT)
    print("REPORT_JSON:", REPORT_JSON)
    print("REPORT_CSV:", REPORT_CSV)
    print("SYMBOLS:", symbols)
    print("TRAIN_END:", train_end)
    print("VALID_START:", valid_start)
    print("VALID_END:", valid_end)
    print("OVERWRITE:", bool(args.overwrite))
    print("ITERATIONS:", int(args.iterations))
    print("LEARNING_RATE:", float(args.learning_rate))
    print("DEPTH:", int(args.depth))
    print("L2_LEAF_REG:", float(args.l2_leaf_reg))
    print("OD_WAIT:", int(args.od_wait))
    print("=" * 120)

    reports: List[Dict[str, Any]] = []

    for idx, symbol in enumerate(symbols, start=1):
        print("[{}/{}] {}: train gate1".format(idx, len(symbols), symbol), flush=True)

        try:
            report = train_one_symbol(
                symbol=symbol,
                train_end=train_end,
                valid_start=valid_start_arg,
                valid_end=valid_end_arg,
                overwrite=bool(args.overwrite),
                iterations=int(args.iterations),
                learning_rate=float(args.learning_rate),
                depth=int(args.depth),
                l2_leaf_reg=float(args.l2_leaf_reg),
                od_wait=int(args.od_wait),
            )

            print(
                "    OK | rows={} | train={} | valid={} | auc_valid={:.6f} | best_thr={:.3f} | kept={:.3f} | precision={:.3f} | features={} | elapsed_sec={}".format(
                    report["rows_total"],
                    report["rows_train"],
                    report["rows_valid"],
                    report["auc_valid"],
                    report["best_thr"],
                    report["kept_share_valid"],
                    report["precision_valid"],
                    report["feature_count"],
                    report["elapsed_sec"],
                ),
                flush=True,
            )
            print("    MODEL:", report["model_path"], flush=True)
            print("    META:", report["meta_path"], flush=True)

        except Exception as exc:
            report = {
                "symbol": symbol,
                "status": "ERR",
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
            print("    ERR:", report["error"], flush=True)

        reports.append(report)

    rep_df = pd.DataFrame(reports)
    rep_df.to_csv(REPORT_CSV, index=False)

    summary = {
        "created_at_utc": str(utc_now_floor_minute()),
        "name": "Train Gate1 Models For Symbols",
        "root": str(ROOT),
        "src_dir": str(SRC_DIR),
        "out_root": str(OUT_ROOT),
        "symbols": symbols,
        "train_end": str(train_end),
        "valid_start": str(valid_start),
        "valid_end": "" if valid_end is None else str(valid_end),
        "overwrite": bool(args.overwrite),
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "od_wait": int(args.od_wait),
        "status_counts": dict(rep_df["status"].value_counts().sort_index()) if "status" in rep_df.columns else {},
        "elapsed_sec": round(time.time() - started_at, 3),
        "report_csv": str(REPORT_CSV),
        "reports": reports,
    }

    REPORT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("=" * 120)
    print("DONE")
    print("STATUS_COUNTS:", summary["status_counts"])
    print("ELAPSED_SEC:", summary["elapsed_sec"])
    print("WROTE_JSON:", REPORT_JSON)
    print("WROTE_CSV:", REPORT_CSV)

    if any(r.get("status") == "ERR" for r in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
