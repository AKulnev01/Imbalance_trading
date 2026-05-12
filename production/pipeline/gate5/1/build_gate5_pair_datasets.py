from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from joblib import Parallel, delayed

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

GATE4_DATASET_PARQUET = ROOT / "production/dataset/gate4/gate4_1_side_builder/gate4_1_side_dataset.parquet"
GATE4_PREDICTIONS_CSV = ROOT / "production/models/gate4/gate4_y_side_clean_multiclass/all_predictions.csv"
M1_DATA_DIR = ROOT / "data/m1_4"

OUT_ROOT = ROOT / "production/dataset/gate5/gate5_pair_datasets"
OUT_SUMMARY_CSV = OUT_ROOT / "_SUMMARY.csv"
OUT_REPORT_JSON = OUT_ROOT / "_REPORT.json"


# ============================================================
# CONFIG
# ============================================================

GATE4_CONFIDENCE_THRESHOLD = 0.90
USE_ONLY_VALID_SPLIT = False
TTL_BARS = 4
GATE5_VALID_PCT = 0.10

# если хочешь жестче — поставь 3.0
# тогда LONG берем только если proba_long >= proba_short * 3
# и SHORT только если proba_short >= proba_long * 3
SIDE_RATIO_MIN = 2.0

TP_SL_GRID: List[Tuple[float, float]] = [
    # RR = 2
    (0.40, 0.20),
    (0.60, 0.30),
    (0.80, 0.40),
    (1.00, 0.50),
    (1.00, 0.75),
    (1.20, 0.60),
    (1.50, 0.75),

    # RR = 2.5
    (0.50, 0.20),
    (0.75, 0.30),
    (1.00, 0.40),
    (1.25, 0.50),
    (1.50, 0.60),

    # RR = 3
    (0.60, 0.20),
    (0.90, 0.30),
    (1.20, 0.40),
    (1.50, 0.50),
    (1.80, 0.60),
    (2.25, 0.75),

    # RR = 4
    (0.80, 0.20),
    (1.20, 0.30),
    (1.60, 0.40),
    (2.00, 0.50),
    (2.40, 0.60),
    (3.00, 0.75),

    # RR = 5
    (1.00, 0.20),
    (1.50, 0.30),
    (2.00, 0.40),
    (2.50, 0.50),
    (3.00, 0.60),

    # RR = 6
    (1.20, 0.20),
    (1.80, 0.30),
    (2.40, 0.40),
    (3.00, 0.50),

    # RR = 7.5
    (1.50, 0.20),
    (2.25, 0.30),

    # RR = 10
    (2.00, 0.20),
]

# сохранять строки, где side не определился по ratio?
KEEP_UNRESOLVED_SIDE = False

GATE4_REQUIRED_META_COLS = {
    "symbol",
    "ts",
    "close",
    "atr14",
}

GATE4_ALLOWED_MODEL_OUTPUT_COLS = {
    # Gate4 verdict
    "proba_long",
    "proba_short",
    "pred_side",
    "pred_side_int",
    "pred_side_confidence",
    "pred_side_gap",
    "pred_side_ratio",
    "gate4_confidence",

    # Gate1
    "gate1_proba",
    "gate1_pass",

    # Gate2
    "g2_cls_up_reach_high_proba",
    "g2_cls_dn_reach_high_proba",
    "g2_cls_spread",
    "g2_cls_abs_spread",
    "g2_cls_max",
    "g2_up_dominant",
    "g2_dn_dominant",

    # Gate3 dynamic outputs only
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
    "gate3_pass_long",
    "gate3_pass_short",
    "gate3_proba_long",
    "gate3_proba_short",
    "gate3_margin_long",
    "gate3_margin_short",

    # Cross / joint
    "g2_g3_side_agree",
    "g2_g3_side_conflict",
    "g1_g2_strength",
    "g1_g3_strength",
    "g2g3_joint_long",
    "g2g3_joint_short",
    "g2g3_joint_long_minus_short",
    "g2g3_joint_abs_spread",
}

GATE4_BANNED_PREFIXES = (
    "y_",
    "mfe_",
    "mae_",
    "first_",
    "edge_",
    "g5_",
)

GATE4_BANNED_EXACT_COLS = {
    "first_up_hit_bar",
    "first_dn_hit_bar",
    "mfe_up_atr_16h",
    "mfe_dn_atr_16h",
    "edge_atr_clean",
    "abs_edge_atr_clean",
    "edge_delta_050",
    "edge_delta_060",
    "edge_delta_075",
    "abs_edge_delta_050",
    "abs_edge_delta_060",
    "abs_edge_delta_075",
}


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_ts(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce", utc=True).dt.tz_localize(None)


def pair_name(tp_atr: float, sl_atr: float) -> str:
    tp = int(round(tp_atr * 100))
    sl = int(round(sl_atr * 100))
    return f"tp{tp:03d}_sl{sl:03d}"



def resolve_pred_side(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["proba_long"] = pd.to_numeric(out["proba_long"], errors="coerce")
    out["proba_short"] = pd.to_numeric(out["proba_short"], errors="coerce")

    long_ok = out["proba_long"] >= out["proba_short"] * SIDE_RATIO_MIN
    short_ok = out["proba_short"] >= out["proba_long"] * SIDE_RATIO_MIN

    out["pred_side"] = np.where(
        long_ok & ~short_ok,
        "LONG",
        np.where(short_ok & ~long_ok, "SHORT", "UNRESOLVED"),
    )

    out["pred_side_int"] = np.where(
        out["pred_side"] == "LONG",
        1,
        np.where(out["pred_side"] == "SHORT", 0, np.nan),
    )

    out["pred_side_confidence"] = np.maximum(out["proba_long"], out["proba_short"])
    out["pred_side_gap"] = (out["proba_long"] - out["proba_short"]).abs()
    out["pred_side_ratio"] = np.where(
        np.minimum(out["proba_long"], out["proba_short"]) > 0,
        np.maximum(out["proba_long"], out["proba_short"]) / np.minimum(out["proba_long"], out["proba_short"]),
        np.nan,
    )

    if not KEEP_UNRESOLVED_SIDE:
        out = out[out["pred_side"].isin(["LONG", "SHORT"])].copy()

    return out


def strip_forbidden_gate4_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols: list[str] = []

    for c in df.columns:
        if c in GATE4_REQUIRED_META_COLS:
            keep_cols.append(c)
            continue

        if c in GATE4_ALLOWED_MODEL_OUTPUT_COLS:
            keep_cols.append(c)
            continue

        if c in GATE4_BANNED_EXACT_COLS:
            continue

        if c.startswith(GATE4_BANNED_PREFIXES):
            continue

    return df.loc[:, sorted(dict.fromkeys(keep_cols))].copy()


def find_ts_col(df: pd.DataFrame) -> str:
    for c in ["ts", "timestamp", "open_time", "time", "datetime", "dt"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"timestamp column not found; cols={list(df.columns)[:30]}")


def load_m1_symbol(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    ts_col = find_ts_col(df)
    if ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})

    need_cols = ["ts", "open", "high", "low", "close"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing M1 cols {missing} in {path}")

    df = df[need_cols].copy()
    df["ts"] = to_ts(df["ts"])
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = (
        df.dropna(subset=["ts", "open", "high", "low", "close"])
          .sort_values("ts")
          .drop_duplicates(subset=["ts"], keep="last")
          .reset_index(drop=True)
    )
    return df


def compute_side_pair_targets_m1(
    df: pd.DataFrame,
    m1_map: Dict[str, pd.DataFrame],
    tp_atr: float,
    sl_atr: float,
) -> pd.DataFrame:
    out = df.copy()
    pair = pair_name(tp_atr, sl_atr)

    tp_hit = np.zeros(len(out), dtype=np.int8)
    sl_hit = np.zeros(len(out), dtype=np.int8)
    tp_before_sl = np.zeros(len(out), dtype=np.int8)
    sl_before_tp = np.zeros(len(out), dtype=np.int8)
    ambiguous_same_bar = np.zeros(len(out), dtype=np.int8)
    no_hit = np.zeros(len(out), dtype=np.int8)

    first_tp_minute = np.full(len(out), np.nan, dtype=float)
    first_sl_minute = np.full(len(out), np.nan, dtype=float)

    mfe_side_atr = np.full(len(out), np.nan, dtype=float)
    mae_side_atr = np.full(len(out), np.nan, dtype=float)
    ttl_ret_side_atr = np.full(len(out), np.nan, dtype=float)

    for i in range(len(out)):
        symbol = str(out.iloc[i]["symbol"])
        if symbol not in m1_map:
            continue

        entry_ts = pd.Timestamp(out.iloc[i]["ts"])
        side = str(out.iloc[i]["pred_side"])
        entry_px = pd.to_numeric(out.iloc[i]["close"], errors="coerce")
        atr14 = pd.to_numeric(out.iloc[i]["atr14"], errors="coerce")

        if not (pd.notna(entry_ts) and np.isfinite(entry_px) and np.isfinite(atr14) and atr14 > 0):
            continue

        end_ts = entry_ts + pd.Timedelta(hours=4 * TTL_BARS)

        m1 = m1_map[symbol]
        path = m1[(m1["ts"] > entry_ts) & (m1["ts"] <= end_ts)].copy()

        if path.empty:
            continue

        high_arr = path["high"].to_numpy(dtype=float)
        low_arr = path["low"].to_numpy(dtype=float)
        close_arr = path["close"].to_numpy(dtype=float)

        if side == "LONG":
            up_path = (high_arr - entry_px) / atr14
            dn_path = (entry_px - low_arr) / atr14

            mfe_side_atr[i] = np.nanmax(up_path) if len(up_path) else np.nan
            mae_side_atr[i] = np.nanmax(dn_path) if len(dn_path) else np.nan
            ttl_ret_side_atr[i] = (close_arr[-1] - entry_px) / atr14 if len(close_arr) else np.nan

            tp_idx = np.where(up_path >= tp_atr)[0]
            sl_idx = np.where(dn_path >= sl_atr)[0]

        elif side == "SHORT":
            dn_path = (entry_px - low_arr) / atr14
            up_path = (high_arr - entry_px) / atr14

            mfe_side_atr[i] = np.nanmax(dn_path) if len(dn_path) else np.nan
            mae_side_atr[i] = np.nanmax(up_path) if len(up_path) else np.nan
            ttl_ret_side_atr[i] = (entry_px - close_arr[-1]) / atr14 if len(close_arr) else np.nan

            tp_idx = np.where(dn_path >= tp_atr)[0]
            sl_idx = np.where(up_path >= sl_atr)[0]

        else:
            continue

        tp_first = int(tp_idx[0]) if len(tp_idx) else None
        sl_first = int(sl_idx[0]) if len(sl_idx) else None

        if tp_first is not None:
            tp_hit[i] = 1
            first_tp_minute[i] = float(tp_first + 1)

        if sl_first is not None:
            sl_hit[i] = 1
            first_sl_minute[i] = float(sl_first + 1)

        if tp_first is not None and sl_first is None:
            tp_before_sl[i] = 1
        elif sl_first is not None and tp_first is None:
            sl_before_tp[i] = 1
        elif tp_first is not None and sl_first is not None:
            if tp_first < sl_first:
                tp_before_sl[i] = 1
            elif sl_first < tp_first:
                sl_before_tp[i] = 1
            else:
                ambiguous_same_bar[i] = 1
        else:
            no_hit[i] = 1

    out[f"g5_mfe_side_atr_{pair}"] = mfe_side_atr
    out[f"g5_mae_side_atr_{pair}"] = mae_side_atr
    out[f"g5_ttl_ret_side_atr_{pair}"] = ttl_ret_side_atr

    out[f"g5_first_tp_minute_{pair}"] = first_tp_minute
    out[f"g5_first_sl_minute_{pair}"] = first_sl_minute

    out[f"g5_tp_hit_{pair}"] = tp_hit
    out[f"g5_sl_hit_{pair}"] = sl_hit
    out[f"g5_tp_before_sl_{pair}"] = tp_before_sl
    out[f"g5_sl_before_tp_{pair}"] = sl_before_tp
    out[f"g5_ambiguous_same_bar_{pair}"] = ambiguous_same_bar
    out[f"g5_no_hit_{pair}"] = no_hit

    out[f"g5_target_{pair}"] = tp_before_sl.astype(int)
    return out


def compute_side_pair_targets(
    df: pd.DataFrame,
    tp_atr: float,
    sl_atr: float,
) -> pd.DataFrame:
    out = df.copy()

    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    atr14 = pd.to_numeric(out["atr14_src"], errors="coerce").to_numpy(dtype=float)

    pred_side = out["pred_side"].astype(str).to_numpy()

    tp_hit = np.zeros(len(out), dtype=np.int8)
    sl_hit = np.zeros(len(out), dtype=np.int8)
    tp_before_sl = np.zeros(len(out), dtype=np.int8)
    sl_before_tp = np.zeros(len(out), dtype=np.int8)
    ambiguous_same_bar = np.zeros(len(out), dtype=np.int8)
    no_hit = np.zeros(len(out), dtype=np.int8)

    first_tp_bar = np.full(len(out), np.nan, dtype=float)
    first_sl_bar = np.full(len(out), np.nan, dtype=float)

    mfe_side_atr = np.full(len(out), np.nan, dtype=float)
    mae_side_atr = np.full(len(out), np.nan, dtype=float)
    ttl_ret_side_atr = np.full(len(out), np.nan, dtype=float)

    for i in range(len(out)):
        start = i + 1
        end = min(len(out), i + 1 + TTL_BARS)

        if start >= end:
            continue

        if not (np.isfinite(close[i]) and np.isfinite(atr14[i]) and atr14[i] > 0):
            continue

        side = pred_side[i]
        if side not in {"LONG", "SHORT"}:
            continue

        if side == "LONG":
            side_mfe = -np.inf
            side_mae = -np.inf
            ttl_ret_val = (close[end - 1] - close[i]) / atr14[i]

            tp_idx = None
            sl_idx = None

            for j in range(start, end):
                up_j = (high[j] - close[i]) / atr14[i]
                dn_j = (close[i] - low[j]) / atr14[i]

                if np.isfinite(up_j):
                    side_mfe = max(side_mfe, up_j)
                if np.isfinite(dn_j):
                    side_mae = max(side_mae, dn_j)

                hit_tp_now = up_j >= tp_atr if np.isfinite(up_j) else False
                hit_sl_now = dn_j >= sl_atr if np.isfinite(dn_j) else False

                if hit_tp_now and hit_sl_now:
                    ambiguous_same_bar[i] = 1
                    tp_idx = j - i if tp_idx is None else tp_idx
                    sl_idx = j - i if sl_idx is None else sl_idx
                    break

                if tp_idx is None and hit_tp_now:
                    tp_idx = j - i

                if sl_idx is None and hit_sl_now:
                    sl_idx = j - i

                if tp_idx is not None and sl_idx is not None:
                    break

        else:
            side_mfe = -np.inf
            side_mae = -np.inf
            ttl_ret_val = (close[i] - close[end - 1]) / atr14[i]

            tp_idx = None
            sl_idx = None

            for j in range(start, end):
                dn_j = (close[i] - low[j]) / atr14[i]
                up_j = (high[j] - close[i]) / atr14[i]

                if np.isfinite(dn_j):
                    side_mfe = max(side_mfe, dn_j)
                if np.isfinite(up_j):
                    side_mae = max(side_mae, up_j)

                hit_tp_now = dn_j >= tp_atr if np.isfinite(dn_j) else False
                hit_sl_now = up_j >= sl_atr if np.isfinite(up_j) else False

                if hit_tp_now and hit_sl_now:
                    ambiguous_same_bar[i] = 1
                    tp_idx = j - i if tp_idx is None else tp_idx
                    sl_idx = j - i if sl_idx is None else sl_idx
                    break

                if tp_idx is None and hit_tp_now:
                    tp_idx = j - i

                if sl_idx is None and hit_sl_now:
                    sl_idx = j - i

                if tp_idx is not None and sl_idx is not None:
                    break

        mfe_side_atr[i] = side_mfe if np.isfinite(side_mfe) else np.nan
        mae_side_atr[i] = side_mae if np.isfinite(side_mae) else np.nan
        ttl_ret_side_atr[i] = ttl_ret_val if np.isfinite(ttl_ret_val) else np.nan

        if tp_idx is not None:
            tp_hit[i] = 1
            first_tp_bar[i] = float(tp_idx)

        if sl_idx is not None:
            sl_hit[i] = 1
            first_sl_bar[i] = float(sl_idx)

        if tp_idx is not None and sl_idx is None:
            tp_before_sl[i] = 1
        elif sl_idx is not None and tp_idx is None:
            sl_before_tp[i] = 1
        elif tp_idx is not None and sl_idx is not None:
            if tp_idx < sl_idx:
                tp_before_sl[i] = 1
            elif sl_idx < tp_idx:
                sl_before_tp[i] = 1
        else:
            no_hit[i] = 1

    pair = pair_name(tp_atr, sl_atr)

    out[f"g5_mfe_side_atr_{pair}"] = mfe_side_atr
    out[f"g5_mae_side_atr_{pair}"] = mae_side_atr
    out[f"g5_ttl_ret_side_atr_{pair}"] = ttl_ret_side_atr

    out[f"g5_first_tp_bar_{pair}"] = first_tp_bar
    out[f"g5_first_sl_bar_{pair}"] = first_sl_bar

    out[f"g5_tp_hit_{pair}"] = tp_hit
    out[f"g5_sl_hit_{pair}"] = sl_hit
    out[f"g5_tp_before_sl_{pair}"] = tp_before_sl
    out[f"g5_sl_before_tp_{pair}"] = sl_before_tp
    out[f"g5_ambiguous_same_bar_{pair}"] = ambiguous_same_bar
    out[f"g5_no_hit_{pair}"] = no_hit

    out[f"g5_target_{pair}"] = tp_before_sl.astype(int)

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dir(OUT_ROOT)

    if not GATE4_DATASET_PARQUET.exists():
        raise SystemExit(f"not found: {GATE4_DATASET_PARQUET}")

    if not GATE4_PREDICTIONS_CSV.exists():
        raise SystemExit(f"not found: {GATE4_PREDICTIONS_CSV}")

    gate4_df = pd.read_parquet(GATE4_DATASET_PARQUET)
    pred_df = pd.read_csv(GATE4_PREDICTIONS_CSV)

    required_pred_cols = ["symbol", "ts", "proba_long", "proba_short"]
    missing_pred = [c for c in required_pred_cols if c not in pred_df.columns]
    if missing_pred:
        raise SystemExit(f"valid_predictions missing columns: {missing_pred}")

    gate4_df["ts"] = to_ts(gate4_df["ts"])
    pred_df["ts"] = to_ts(pred_df["ts"])

    gate4_merge_cols = ["symbol", "ts"] + sorted(
        c for c in gate4_df.columns
        if c in GATE4_REQUIRED_META_COLS or c in GATE4_ALLOWED_MODEL_OUTPUT_COLS
    )

    gate4_merge_cols = list(dict.fromkeys(gate4_merge_cols))

    base = pred_df.merge(
        gate4_df[gate4_merge_cols],
        on=["symbol", "ts"],
        how="left",
        suffixes=("", "_g4"),
    )

    base["proba_long"] = pd.to_numeric(base["proba_long"], errors="coerce")
    base["proba_short"] = pd.to_numeric(base["proba_short"], errors="coerce")
    base["gate4_confidence"] = np.maximum(base["proba_long"], base["proba_short"])

    base = base[
        base["gate4_confidence"] >= GATE4_CONFIDENCE_THRESHOLD
        ].copy()

    base = resolve_pred_side(base)
    base = strip_forbidden_gate4_cols(base)

    for c in ["pred_side_int", "pred_side_confidence", "pred_side_gap", "pred_side_ratio"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    if len(base) == 0:
        raise SystemExit("no rows after gate4 confidence filter and pred_side resolution")

    print("ROWS AFTER GATE4 MERGE:", len(pred_df))
    print("ROWS AFTER GATE4 CONF FILTER:", len(base))
    print("GATE4 CONF THRESHOLD:", GATE4_CONFIDENCE_THRESHOLD)
    print("SIDE RATIO MIN:", SIDE_RATIO_MIN)
    print("PRED SIDE DISTRIBUTION")
    print(base["pred_side"].value_counts(dropna=False).to_string())
    print()

    rows_all: List[pd.DataFrame] = []
    symbol_audit: List[dict] = []

    for symbol in sorted(base["symbol"].dropna().astype(str).unique()):
        fp_src = M1_DATA_DIR / f"{symbol}.parquet"
        if not fp_src.exists():
            fp_src = M1_DATA_DIR / f"{symbol}_m1.parquet"

        if not fp_src.exists():
            symbol_audit.append({
                "symbol": symbol,
                "status": "missing_m1_source",
                "rows_gate4_pred": int((base["symbol"] == symbol).sum()),
                "rows_merged": 0,
            })
            continue

        try:
            m1 = load_m1_symbol(fp_src)
        except Exception as e:
            symbol_audit.append({
                "symbol": symbol,
                "status": f"bad_m1_source:{type(e).__name__}",
                "rows_gate4_pred": int((base["symbol"] == symbol).sum()),
                "rows_merged": 0,
            })
            continue

        xs = base[base["symbol"] == symbol].copy()

        need_gate4_cols = [
            "symbol",
            "ts",
            "close",
            "atr14",

            "proba_long",
            "proba_short",
            "pred_side",
            "pred_side_int",
            "pred_side_confidence",
            "pred_side_gap",
            "pred_side_ratio",
            "gate4_confidence",

            "gate1_proba",
            "gate1_pass",

            "g2_cls_up_reach_high_proba",
            "g2_cls_dn_reach_high_proba",
            "g2_cls_spread",
            "g2_cls_abs_spread",
            "g2_cls_max",
            "g2_up_dominant",
            "g2_dn_dominant",

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
            "gate3_pass_long",
            "gate3_pass_short",
            "gate3_proba_long",
            "gate3_proba_short",
            "gate3_margin_long",
            "gate3_margin_short",

            "g2_g3_side_agree",
            "g2_g3_side_conflict",
            "g1_g2_strength",
            "g1_g3_strength",
            "g2g3_joint_long",
            "g2g3_joint_short",
            "g2g3_joint_long_minus_short",
            "g2g3_joint_abs_spread",
        ]

        extra_numeric_cols = [
            "gate4_confidence",

            "gate1_proba",
            "gate1_pass",

            "g2_cls_up_reach_high_proba",
            "g2_cls_dn_reach_high_proba",
            "g2_cls_spread",
            "g2_cls_abs_spread",
            "g2_cls_max",
            "g2_up_dominant",
            "g2_dn_dominant",

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
            "gate3_pass_long",
            "gate3_pass_short",
            "gate3_proba_long",
            "gate3_proba_short",
            "gate3_margin_long",
            "gate3_margin_short",

            "g2_g3_side_agree",
            "g2_g3_side_conflict",
            "g1_g2_strength",
            "g1_g3_strength",
            "g2g3_joint_long",
            "g2g3_joint_short",
            "g2g3_joint_long_minus_short",
            "g2g3_joint_abs_spread",
        ]

        for c in extra_numeric_cols:
            if c in xs.columns:
                xs[c] = pd.to_numeric(xs[c], errors="coerce")
        missing_gate4 = [c for c in need_gate4_cols if c not in xs.columns]
        if missing_gate4:
            symbol_audit.append({
                "symbol": symbol,
                "status": f"missing_gate4_cols:{','.join(missing_gate4)}",
                "rows_gate4_pred": int(len(xs)),
                "rows_merged": 0,
            })
            continue

        xs = xs.dropna(subset=["ts", "close", "atr14"]).copy()
        xs["symbol"] = symbol
        xs["close"] = pd.to_numeric(xs["close"], errors="coerce")
        xs["atr14"] = pd.to_numeric(xs["atr14"], errors="coerce")
        xs["proba_long"] = pd.to_numeric(xs["proba_long"], errors="coerce")
        xs["proba_short"] = pd.to_numeric(xs["proba_short"], errors="coerce")
        xs["pred_side_int"] = pd.to_numeric(xs["pred_side_int"], errors="coerce")
        xs["pred_side_confidence"] = pd.to_numeric(xs["pred_side_confidence"], errors="coerce")
        xs["pred_side_gap"] = pd.to_numeric(xs["pred_side_gap"], errors="coerce")
        xs["pred_side_ratio"] = pd.to_numeric(xs["pred_side_ratio"], errors="coerce")

        symbol_audit.append({
            "symbol": symbol,
            "status": "ok",
            "rows_gate4_pred": int(len(xs)),
            "rows_merged": int(len(xs)),
        })

        if len(xs):
            rows_all.append(xs)

    if not rows_all:
        audit_df = pd.DataFrame(symbol_audit)
        print("=== SYMBOL AUDIT ===")
        if len(audit_df):
            print(audit_df.to_string(index=False))
            print()
            print("=== STATUS COUNTS ===")
            print(audit_df["status"].value_counts(dropna=False).to_string())
        else:
            print("symbol_audit is empty")
        raise SystemExit("no rows collected into rows_all; see SYMBOL AUDIT above")

    full = pd.concat(rows_all, ignore_index=True)
    full = full.sort_values(["ts", "symbol"]).reset_index(drop=True)

    # ============================================================
    # GLOBAL TIME SPLIT (CRITICAL FIX)
    # ============================================================

    full = full.sort_values("ts").reset_index(drop=True)

    split_idx = int(len(full) * (1.0 - GATE5_VALID_PCT))
    split_ts = full.iloc[split_idx]["ts"]

    full["gate5_split"] = np.where(full["ts"] < split_ts, "train", "valid")
    full["gate5_is_oos"] = (full["ts"] >= split_ts).astype(int)

    print("GATE5 GLOBAL SPLIT TS:", split_ts)
    print(full["gate5_split"].value_counts())
    print()

    m1_map: Dict[str, pd.DataFrame] = {}
    for symbol in sorted(full["symbol"].dropna().astype(str).unique()):
        fp_m1 = M1_DATA_DIR / f"{symbol}.parquet"
        if not fp_m1.exists():
            fp_m1 = M1_DATA_DIR / f"{symbol}_m1.parquet"
        if not fp_m1.exists():
            continue
        try:
            m1_map[symbol] = load_m1_symbol(fp_m1)
        except Exception:
            continue

    if not m1_map:
        raise SystemExit(f"no valid M1 files loaded from {M1_DATA_DIR}")

    summary_rows = []

    def process_pair(tp_atr, sl_atr):
        pair = pair_name(tp_atr, sl_atr)

        out = compute_side_pair_targets_m1(
            df=full,
            m1_map=m1_map,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
        )

        target_col = f"g5_target_{pair}"
        tp_col = f"g5_tp_before_sl_{pair}"
        sl_col = f"g5_sl_before_tp_{pair}"
        amb_col = f"g5_ambiguous_same_bar_{pair}"
        nohit_col = f"g5_no_hit_{pair}"

        out_path = OUT_ROOT / f"gate5_dataset_{pair}.parquet"
        out.to_parquet(out_path, index=False)

        tp_before_sl_rate = float(pd.to_numeric(out[tp_col], errors="coerce").mean())
        sl_before_tp_rate = float(pd.to_numeric(out[sl_col], errors="coerce").mean())
        ambiguous_same_bar_rate = float(pd.to_numeric(out[amb_col], errors="coerce").mean())
        no_hit_rate = float(pd.to_numeric(out[nohit_col], errors="coerce").mean())

        rr = float(tp_atr / sl_atr) if sl_atr > 0 else np.nan
        expectancy_hit_proxy = float(tp_atr * tp_before_sl_rate - sl_atr * sl_before_tp_rate)

        return {
            "pair": pair,
            "tp_atr": float(tp_atr),
            "sl_atr": float(sl_atr),
            "rr": rr,
            "rows_total": int(len(out)),
            "target_pos_rate": float(pd.to_numeric(out[target_col], errors="coerce").mean()),
            "tp_before_sl_rate": tp_before_sl_rate,
            "sl_before_tp_rate": sl_before_tp_rate,
            "ambiguous_same_bar_rate": ambiguous_same_bar_rate,
            "no_hit_rate": no_hit_rate,
            "expectancy_hit_proxy": expectancy_hit_proxy,
            "long_share": float((out["pred_side"] == "LONG").mean()),
            "short_share": float((out["pred_side"] == "SHORT").mean()),
            "rows_long": int((out["pred_side"] == "LONG").sum()),
            "rows_short": int((out["pred_side"] == "SHORT").sum()),
            "file": str(out_path),
        }

    summary_rows = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_pair)(tp, sl)
        for tp, sl in TP_SL_GRID
    )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["expectancy_hit_proxy", "tp_before_sl_rate", "rr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    report = {
        "gate4_dataset_parquet": str(GATE4_DATASET_PARQUET),
        "gate4_predictions_csv": str(GATE4_PREDICTIONS_CSV),
        "m1_data_dir": str(M1_DATA_DIR),
        "ttl_bars": int(TTL_BARS),
        "side_ratio_min": float(SIDE_RATIO_MIN),
        "keep_unresolved_side": bool(KEEP_UNRESOLVED_SIDE),
        "tp_sl_grid": [{"tp_atr": float(tp), "sl_atr": float(sl)} for tp, sl in TP_SL_GRID],
        "rows_after_gate4_conf_filter_and_side_resolution": int(len(base)),
        "gate4_confidence_threshold": float(GATE4_CONFIDENCE_THRESHOLD),
        "gate5_split_ts": str(split_ts),
        "rows_final_for_gate5_m1": int(len(full)),
        "rows_train": int((full["gate5_split"] == "train").sum()),
        "rows_valid": int((full["gate5_split"] == "valid").sum()),
        "summary_csv": str(OUT_SUMMARY_CSV),
        "symbols_audit": symbol_audit,
    }

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("WROTE", OUT_SUMMARY_CSV)
    print("WROTE", OUT_REPORT_JSON)
    print()
    print("=== SUMMARY SORTED BY EXPECTANCY HIT PROXY ===")
    print(summary_df.to_string(index=False))
    print()

    print("=== TOP 15 BY TP_BEFORE_SL_RATE ===")
    print(
        summary_df.sort_values(
            ["tp_before_sl_rate", "expectancy_hit_proxy"],
            ascending=[False, False],
        ).head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()