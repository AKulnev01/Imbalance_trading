from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")

GRID_LIST = [
    "tp150_sl075",
    "tp100_sl075",
    "tp150_sl060",
    "tp120_sl060",
    "tp160_sl040",
    "tp225_sl075",
    "tp120_sl040",
    "tp180_sl060",
    "tp150_sl050",
    "tp125_sl050",
    "tp200_sl050",
    "tp240_sl060",
]

PRED_DIR = ROOT / "production/models/gate5/v1_1"
GRID_DATA_DIR = ROOT / "production/dataset/gate5/gate5_pair_datasets"
META_PATH = PRED_DIR / "summary.csv"

OUT_DIR = ROOT / "production/dataset/gate5/gate5_2"
OUT_DATASET = OUT_DIR / "gate5_grid_ranker_dataset.parquet"
OUT_REPORT = OUT_DIR / "gate5_grid_ranker_report.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# teacher-score weights
PROFIT_REWARD_MULT = 1.0
LOSS_PENALTY_MULT = 3.0
NO_HIT_PENALTY_MULT = 1.0
AMBIG_PENALTY_MULT = 1.5

# contextual penalty:
# if signal has at least one profitable grid, then all non-profitable grids
# inside this signal become much more toxic
POS_EXISTS_PENALTY = 1.5

# safety
REQUIRE_FULL_PROBA_COVERAGE = True


# ============================================================
# HELPERS
# ============================================================

def parse_tp_sl(pair: str) -> tuple[float, float]:
    left, right = pair.split("_")
    tp_atr = float(left.replace("tp", "")) / 100.0
    sl_atr = float(right.replace("sl", "")) / 100.0
    return tp_atr, sl_atr


def pair_to_idx_map(grid_list: list[str]) -> dict[str, int]:
    return {pair: i for i, pair in enumerate(grid_list)}


def safe_rank_desc(values: np.ndarray) -> np.ndarray:
    """
    Desc rank: 1 = best.
    Stable and deterministic.
    """
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int32)
    return ranks


def compute_teacher_score(
    tp_before_sl: pd.Series,
    sl_before_tp: pd.Series,
    no_hit: pd.Series,
    ambiguous_same_bar: pd.Series,
    tp_atr: float,
    sl_atr: float,
) -> pd.Series:
    tp_before_sl = pd.to_numeric(tp_before_sl, errors="coerce").fillna(0.0)
    sl_before_tp = pd.to_numeric(sl_before_tp, errors="coerce").fillna(0.0)
    no_hit = pd.to_numeric(no_hit, errors="coerce").fillna(0.0)
    ambiguous_same_bar = pd.to_numeric(ambiguous_same_bar, errors="coerce").fillna(0.0)

    score = np.zeros(len(tp_before_sl), dtype=float)

    score += tp_before_sl * (PROFIT_REWARD_MULT * tp_atr)
    score -= sl_before_tp * (LOSS_PENALTY_MULT * sl_atr)
    score -= no_hit * (NO_HIT_PENALTY_MULT * sl_atr)
    score -= ambiguous_same_bar * (AMBIG_PENALTY_MULT * sl_atr)

    return pd.Series(score, index=tp_before_sl.index, dtype=float)


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns: {missing}")


def load_meta_map(meta_path: Path) -> dict[str, dict]:
    if not meta_path.exists():
        raise FileNotFoundError(f"not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)
    require_cols(
        meta_df,
        [
            "pair",
            "auc_valid",
            "best_thr_by_ev",
            "best_kept_by_ev",
            "best_coverage_by_ev",
            "best_precision_by_ev",
            "best_ev_proxy",
        ],
        "meta_df",
    )

    out: dict[str, dict] = {}
    for _, row in meta_df.iterrows():
        pair = str(row["pair"])
        out[pair] = {
            "auc_valid": row["auc_valid"],
            "best_thr_by_ev": row["best_thr_by_ev"],
            "best_kept_by_ev": row["best_kept_by_ev"],
            "best_coverage_by_ev": row["best_coverage_by_ev"],
            "best_precision_by_ev": row["best_precision_by_ev"],
            "best_ev_proxy": row["best_ev_proxy"],
        }
    return out


# ============================================================
# LOAD BASE SIGNAL FRAME
# ============================================================

def build_base_signal_frame() -> pd.DataFrame:
    base_pair = GRID_LIST[0]
    base_path = GRID_DATA_DIR / f"gate5_dataset_{base_pair}.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"not found: {base_path}")

    base_df = pd.read_parquet(base_path)
    require_cols(
        base_df,
        ["ts", "symbol", "pred_side", "gate5_split", "gate5_is_oos"],
        "base_df",
    )

    out = base_df[["ts", "symbol", "pred_side", "gate5_split", "gate5_is_oos"]].copy()
    out = out.rename(columns={"pred_side": "side"})
    out["side"] = out["side"].astype(str).str.upper()
    bad_side_mask = ~out["side"].isin(["LONG", "SHORT"])
    if bad_side_mask.any():
        bad_n = int(bad_side_mask.sum())
        raise RuntimeError(f"base_df: invalid side values, rows={bad_n}")
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts", "symbol"]).copy()
    out = out.sort_values(["ts", "symbol"]).reset_index(drop=True)
    out["signal_id"] = np.arange(len(out), dtype=np.int64)
    return out


# ============================================================
# ADD GRID PRETRADE FEATURES TO SIGNAL FRAME
# ============================================================

def add_grid_prediction_block(
    signal_df: pd.DataFrame,
    pair: str,
    meta_map: dict[str, dict],
) -> pd.DataFrame:
    pred_path = PRED_DIR / f"full_predictions_{pair}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"not found: {pred_path}")

    pred_df = pd.read_parquet(pred_path)
    require_cols(pred_df, ["ts", "symbol", "proba"], f"pred_df[{pair}]")

    pred_df = pred_df[["ts", "symbol", "proba"]].copy()
    pred_df["ts"] = pd.to_datetime(pred_df["ts"], errors="coerce")
    pred_df = pred_df.dropna(subset=["ts", "symbol"]).copy()

    pred_col = f"grid_{pair}_proba"
    pred_df = pred_df.rename(columns={"proba": pred_col})

    out = signal_df.merge(
        pred_df,
        on=["ts", "symbol"],
        how="left",
    )

    if REQUIRE_FULL_PROBA_COVERAGE:
        miss = int(out[pred_col].isna().sum())
        if miss != 0:
            raise RuntimeError(f"{pair}: missing proba after merge: {miss}")

    meta = meta_map[pair]

    thr = meta["best_thr_by_ev"]
    if pd.isna(thr):
        thr = np.nan
    else:
        thr = float(thr)

    tp_atr, sl_atr = parse_tp_sl(pair)

    out[f"grid_{pair}_best_thr"] = thr
    out[f"grid_{pair}_ev_meta"] = meta["best_ev_proxy"]
    out[f"grid_{pair}_precision_meta"] = meta["best_precision_by_ev"]
    out[f"grid_{pair}_kept_n_meta"] = meta["best_kept_by_ev"]
    out[f"grid_{pair}_coverage_meta"] = meta["best_coverage_by_ev"]
    out[f"grid_{pair}_auc_meta"] = meta["auc_valid"]

    out[f"grid_{pair}_tp_atr"] = tp_atr
    out[f"grid_{pair}_sl_atr"] = sl_atr
    out[f"grid_{pair}_rr"] = tp_atr / sl_atr

    if pd.isna(thr):
        out[f"grid_{pair}_pass"] = 0
        out[f"grid_{pair}_margin_to_thr"] = np.nan
    else:
        out[f"grid_{pair}_pass"] = (pd.to_numeric(out[pred_col], errors="coerce") >= thr).astype(int)
        out[f"grid_{pair}_margin_to_thr"] = pd.to_numeric(out[pred_col], errors="coerce") - thr

    return out


# ============================================================
# SIGNAL-LEVEL CROSS FEATURES FROM PRETRADE OUTPUTS
# ============================================================

def add_signal_level_cross_features(signal_df: pd.DataFrame) -> pd.DataFrame:
    out = signal_df.copy()

    proba_cols = [f"grid_{p}_proba" for p in GRID_LIST]
    margin_cols = [f"grid_{p}_margin_to_thr" for p in GRID_LIST]
    pass_cols = [f"grid_{p}_pass" for p in GRID_LIST]
    rr_cols = [f"grid_{p}_rr" for p in GRID_LIST]
    tp_cols = [f"grid_{p}_tp_atr" for p in GRID_LIST]
    sl_cols = [f"grid_{p}_sl_atr" for p in GRID_LIST]
    ev_cols = [f"grid_{p}_ev_meta" for p in GRID_LIST]
    wilson_like_cols = [f"grid_{p}_precision_meta" for p in GRID_LIST]

    proba_mat = out[proba_cols].to_numpy(dtype=float)
    margin_mat = out[margin_cols].to_numpy(dtype=float)
    pass_mat = out[pass_cols].to_numpy(dtype=float)
    rr_mat = out[rr_cols].to_numpy(dtype=float)
    tp_mat = out[tp_cols].to_numpy(dtype=float)
    sl_mat = out[sl_cols].to_numpy(dtype=float)
    ev_mat = out[ev_cols].to_numpy(dtype=float)
    precision_mat = out[wilson_like_cols].to_numpy(dtype=float)

    n = len(out)

    # fill margin NaN for ranking purposes only
    margin_rank_mat = np.where(np.isfinite(margin_mat), margin_mat, -1e18)

    proba_order = np.argsort(-proba_mat, axis=1)
    margin_order = np.argsort(-margin_rank_mat, axis=1)

    top1_proba_idx = proba_order[:, 0]
    top2_proba_idx = proba_order[:, 1]

    top1_margin_idx = margin_order[:, 0]
    top2_margin_idx = margin_order[:, 1]

    out["sig_top1_proba"] = proba_mat[np.arange(n), top1_proba_idx]
    out["sig_top2_proba"] = proba_mat[np.arange(n), top2_proba_idx]
    out["sig_top1_minus_top2_proba"] = out["sig_top1_proba"] - out["sig_top2_proba"]

    out["sig_top1_margin"] = margin_rank_mat[np.arange(n), top1_margin_idx]
    out["sig_top2_margin"] = margin_rank_mat[np.arange(n), top2_margin_idx]
    out["sig_top1_minus_top2_margin"] = out["sig_top1_margin"] - out["sig_top2_margin"]

    out["sig_num_passed_grids"] = pass_mat.sum(axis=1).astype(np.int32)

    pass_mask = pass_mat == 1

    out["sig_max_rr_among_passed"] = np.where(
        pass_mask.any(axis=1),
        np.max(np.where(pass_mask, rr_mat, -np.inf), axis=1),
        0.0,
    )

    out["sig_max_tp_atr_among_passed"] = np.where(
        pass_mask.any(axis=1),
        np.max(np.where(pass_mask, tp_mat, -np.inf), axis=1),
        0.0,
    )

    out["sig_min_sl_atr_among_passed"] = np.where(
        pass_mask.any(axis=1),
        np.min(np.where(pass_mask, sl_mat, np.inf), axis=1),
        0.0,
    )

    out["sig_mean_proba"] = np.nanmean(proba_mat, axis=1)
    out["sig_std_proba"] = np.nanstd(proba_mat, axis=1)

    out["sig_mean_margin"] = np.nanmean(margin_mat, axis=1)
    out["sig_std_margin"] = np.nanstd(margin_mat, axis=1)

    out["sig_mean_ev_meta"] = np.nanmean(ev_mat, axis=1)
    out["sig_std_ev_meta"] = np.nanstd(ev_mat, axis=1)

    out["sig_mean_precision_meta"] = np.nanmean(precision_mat, axis=1)
    out["sig_std_precision_meta"] = np.nanstd(precision_mat, axis=1)

    return out


# ============================================================
# BUILD LONG DATASET
# ============================================================

def build_long_pretrade_dataset(signal_df: pd.DataFrame) -> pd.DataFrame:
    pair_idx = pair_to_idx_map(GRID_LIST)
    rows: list[pd.DataFrame] = []

    for pair in GRID_LIST:
        tp_atr, sl_atr = parse_tp_sl(pair)

        block = signal_df[
            [
                "signal_id",
                "ts",
                "symbol",
                "side",
                "gate5_split",
                "gate5_is_oos",

                "sig_top1_proba",
                "sig_top2_proba",
                "sig_top1_minus_top2_proba",
                "sig_top1_margin",
                "sig_top2_margin",
                "sig_top1_minus_top2_margin",
                "sig_num_passed_grids",
                "sig_max_rr_among_passed",
                "sig_max_tp_atr_among_passed",
                "sig_min_sl_atr_among_passed",
                "sig_mean_proba",
                "sig_std_proba",
                "sig_mean_margin",
                "sig_std_margin",
                "sig_mean_ev_meta",
                "sig_std_ev_meta",
                "sig_mean_precision_meta",
                "sig_std_precision_meta",

                f"grid_{pair}_proba",
                f"grid_{pair}_best_thr",
                f"grid_{pair}_ev_meta",
                f"grid_{pair}_precision_meta",
                f"grid_{pair}_kept_n_meta",
                f"grid_{pair}_coverage_meta",
                f"grid_{pair}_auc_meta",
                f"grid_{pair}_tp_atr",
                f"grid_{pair}_sl_atr",
                f"grid_{pair}_rr",
                f"grid_{pair}_pass",
                f"grid_{pair}_margin_to_thr",
            ]
        ].copy()

        block = block.rename(
            columns={
                f"grid_{pair}_proba": "grid_proba",
                f"grid_{pair}_best_thr": "grid_best_thr",
                f"grid_{pair}_ev_meta": "grid_ev_meta",
                f"grid_{pair}_precision_meta": "grid_precision_meta",
                f"grid_{pair}_kept_n_meta": "grid_kept_n_meta",
                f"grid_{pair}_coverage_meta": "grid_coverage_meta",
                f"grid_{pair}_auc_meta": "grid_auc_meta",
                f"grid_{pair}_tp_atr": "grid_tp_atr",
                f"grid_{pair}_sl_atr": "grid_sl_atr",
                f"grid_{pair}_rr": "grid_rr",
                f"grid_{pair}_pass": "grid_pass",
                f"grid_{pair}_margin_to_thr": "grid_margin_to_thr",
            }
        )

        block["grid_name"] = pair
        block["grid_idx"] = pair_idx[pair]
        block["grid_proba_to_top1_ratio"] = np.where(
            pd.to_numeric(block["sig_top1_proba"], errors="coerce") > 0,
            pd.to_numeric(block["grid_proba"], errors="coerce") / pd.to_numeric(block["sig_top1_proba"], errors="coerce"),
            0.0,
        )

        block["grid_margin_to_top1_ratio"] = np.where(
            pd.to_numeric(block["sig_top1_margin"], errors="coerce") != 0,
            pd.to_numeric(block["grid_margin_to_thr"], errors="coerce") / pd.to_numeric(block["sig_top1_margin"], errors="coerce"),
            0.0,
        )

        block["grid_tp_minus_sig_max_tp_passed"] = block["grid_tp_atr"] - block["sig_max_tp_atr_among_passed"]
        block["grid_sl_minus_sig_min_sl_passed"] = block["grid_sl_atr"] - block["sig_min_sl_atr_among_passed"]
        block["grid_rr_minus_sig_max_rr_passed"] = block["grid_rr"] - block["sig_max_rr_among_passed"]

        # z-like relative positions
        block["grid_proba_vs_sig_mean"] = block["grid_proba"] - block["sig_mean_proba"]
        block["grid_margin_vs_sig_mean"] = block["grid_margin_to_thr"] - block["sig_mean_margin"]
        block["grid_ev_meta_vs_sig_mean"] = block["grid_ev_meta"] - block["sig_mean_ev_meta"]
        block["grid_precision_meta_vs_sig_mean"] = block["grid_precision_meta"] - block["sig_mean_precision_meta"]


        rows.append(block)

    long_df = pd.concat(rows, ignore_index=True)
    if "side" not in long_df.columns:
        raise RuntimeError("long_df: side column is missing")

    long_df["side"] = long_df["side"].astype(str).str.upper()

    bad_side_mask = ~long_df["side"].isin(["LONG", "SHORT"])
    if bad_side_mask.any():
        bad_n = int(bad_side_mask.sum())
        raise RuntimeError(f"long_df: invalid side values, rows={bad_n}")

    long_df = long_df.sort_values(["ts", "symbol", "side", "grid_idx"]).reset_index(drop=True)
    return long_df


# ============================================================
# ADD TARGETS FROM FUTURE OUTCOMES
# ============================================================

def add_teacher_targets(long_df: pd.DataFrame) -> pd.DataFrame:
    outcome_frames: list[pd.DataFrame] = []

    for pair in GRID_LIST:
        grid_path = GRID_DATA_DIR / f"gate5_dataset_{pair}.parquet"
        if not grid_path.exists():
            raise FileNotFoundError(f"not found: {grid_path}")

        grid_df = pd.read_parquet(grid_path)
        require_cols(
            grid_df,
            [
                "ts",
                "symbol",
                "pred_side",
                f"g5_tp_before_sl_{pair}",
                f"g5_sl_before_tp_{pair}",
                f"g5_no_hit_{pair}",
                f"g5_ambiguous_same_bar_{pair}",
            ],
            f"grid_df[{pair}]",
        )

        grid_df["ts"] = pd.to_datetime(grid_df["ts"], errors="coerce")

        tp_atr, sl_atr = parse_tp_sl(pair)

        out = grid_df[
            [
                "ts",
                "symbol",
                "pred_side",
                f"g5_tp_before_sl_{pair}",
                f"g5_sl_before_tp_{pair}",
                f"g5_no_hit_{pair}",
                f"g5_ambiguous_same_bar_{pair}",
            ]
        ].copy()
        out = out.rename(columns={"pred_side": "side"})
        out["side"] = out["side"].astype(str).str.upper()

        out = out.rename(
            columns={
                f"g5_tp_before_sl_{pair}": "target_tp_before_sl",
                f"g5_sl_before_tp_{pair}": "target_sl_before_tp",
                f"g5_no_hit_{pair}": "target_no_hit",
                f"g5_ambiguous_same_bar_{pair}": "target_ambiguous_same_bar",
            }
        )

        out["grid_name"] = pair
        out["target_score"] = compute_teacher_score(
            tp_before_sl=out["target_tp_before_sl"],
            sl_before_tp=out["target_sl_before_tp"],
            no_hit=out["target_no_hit"],
            ambiguous_same_bar=out["target_ambiguous_same_bar"],
            tp_atr=tp_atr,
            sl_atr=sl_atr,
        )

        outcome_frames.append(out)

    outcome_df = pd.concat(outcome_frames, ignore_index=True)
    outcome_df = outcome_df.sort_values(["ts", "symbol", "grid_name"]).reset_index(drop=True)

    out = long_df.merge(
        outcome_df,
        on=["ts", "symbol", "side", "grid_name"],
        how="left",
    )

    target_cols = [
        "target_tp_before_sl",
        "target_sl_before_tp",
        "target_no_hit",
        "target_ambiguous_same_bar",
        "target_score",
    ]
    miss = {c: int(out[c].isna().sum()) for c in target_cols}
    bad = {k: v for k, v in miss.items() if v != 0}
    if bad:
        raise RuntimeError(f"target merge produced NaN: {bad}")

    return out

def add_contextual_teacher_score(long_df: pd.DataFrame) -> pd.DataFrame:
    out = long_df.copy()

    grp = out.groupby("signal_id", sort=False)["target_score"]

    out["signal_best_base_target_score"] = grp.transform("max")
    out["signal_has_positive_base_target"] = (
        out["signal_best_base_target_score"] > 0
    ).astype(int)

    out["target_score_ctx"] = np.where(
        out["signal_has_positive_base_target"] == 1,
        np.where(
            pd.to_numeric(out["target_score"], errors="coerce") > 0,
            pd.to_numeric(out["target_score"], errors="coerce"),
            pd.to_numeric(out["target_score"], errors="coerce") - POS_EXISTS_PENALTY,
        ),
        pd.to_numeric(out["target_score"], errors="coerce"),
    )

    out["target_is_positive_base"] = (
        pd.to_numeric(out["target_score"], errors="coerce") > 0
    ).astype(int)

    out["target_is_non_positive_base"] = (
        pd.to_numeric(out["target_score"], errors="coerce") <= 0
    ).astype(int)

    return out


# ============================================================
# ADD ORACLE WINNER INFO (FOR ANALYSIS / LATER TRAINING)
# ============================================================

def add_signal_level_oracle_info(long_df: pd.DataFrame) -> pd.DataFrame:
    out = long_df.copy()

    grp = out.groupby("signal_id", sort=False)["target_score_ctx"]
    out["signal_best_target_score"] = grp.transform("max")
    out["signal_worst_target_score"] = grp.transform("min")

    # deterministic winner: sort by target_score desc, then tp_atr desc, then sl_atr asc, then grid_idx asc
    order_df = out[
        ["signal_id", "grid_idx", "target_score_ctx", "grid_tp_atr", "grid_sl_atr"]
    ].copy()

    order_df = order_df.sort_values(
        ["signal_id", "target_score_ctx", "grid_tp_atr", "grid_sl_atr", "grid_idx"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)

    winner_map = order_df.groupby("signal_id", sort=False).first().reset_index()
    winner_map = winner_map.rename(
        columns={
            "grid_idx": "signal_best_grid_idx",
            "target_score_ctx": "signal_best_grid_score",
            "grid_tp_atr": "signal_best_grid_tp_atr",
            "grid_sl_atr": "signal_best_grid_sl_atr",
        }
    )

    out = out.merge(
        winner_map,
        on="signal_id",
        how="left",
    )

    idx_to_pair = {i: pair for i, pair in enumerate(GRID_LIST)}
    out["signal_best_grid_name"] = out["signal_best_grid_idx"].map(idx_to_pair)
    out["is_oracle_winner"] = (out["grid_idx"] == out["signal_best_grid_idx"]).astype(int)

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("LOAD META")
    meta_map = load_meta_map(META_PATH)

    for pair in GRID_LIST:
        if pair not in meta_map:
            raise RuntimeError(f"pair not found in meta: {pair}")

    print("BUILD BASE SIGNAL FRAME")
    signal_df = build_base_signal_frame()

    print("BASE SIGNALS:", len(signal_df))
    print(signal_df["gate5_split"].value_counts(dropna=False).to_string())
    print()

    print("MERGE GRID PRETRADE FEATURES")
    for pair in GRID_LIST:
        print("LOAD GRID:", pair)
        signal_df = add_grid_prediction_block(
            signal_df=signal_df,
            pair=pair,
            meta_map=meta_map,
        )

    print("ADD SIGNAL-LEVEL CROSS FEATURES")
    signal_df = add_signal_level_cross_features(signal_df)

    print("BUILD LONG DATASET")
    long_df = build_long_pretrade_dataset(signal_df)

    print("ADD TEACHER TARGETS")
    long_df = add_teacher_targets(long_df)

    print("ADD CONTEXTUAL TEACHER TARGET")
    long_df = add_contextual_teacher_score(long_df)

    print("ADD SIGNAL-LEVEL ORACLE INFO")
    long_df = add_signal_level_oracle_info(long_df)

    long_df = long_df.sort_values(["ts", "symbol", "side", "grid_idx"]).reset_index(drop=True)
    long_df.to_parquet(OUT_DATASET, index=False)

    report = {
        "grid_list": GRID_LIST,
        "rows_signal_level": int(len(signal_df)),
        "rows_long_level": int(len(long_df)),
        "expected_rows_long_level": int(len(signal_df) * len(GRID_LIST)),
        "unique_signals": int(long_df["signal_id"].nunique()),
        "rows_per_signal_mean": float(long_df.groupby("signal_id").size().mean()),
        "side_counts_signal": {
            k: int(v)
            for k, v in signal_df["side"].value_counts(dropna=False).to_dict().items()
        },
        "side_counts_long": {
            k: int(v)
            for k, v in long_df["side"].value_counts(dropna=False).to_dict().items()
        },
        "split_counts_signal": {
            k: int(v)
            for k, v in signal_df["gate5_split"].value_counts(dropna=False).to_dict().items()
        },
        "split_counts_long": {
            k: int(v)
            for k, v in long_df["gate5_split"].value_counts(dropna=False).to_dict().items()
        },
        "target_score_total_mean": float(long_df["target_score"].mean()),
        "target_score_train_mean": float(long_df.loc[long_df["gate5_split"] == "train", "target_score"].mean()),
        "target_score_valid_mean": float(long_df.loc[long_df["gate5_split"] == "valid", "target_score"].mean()),
        "target_score_ctx_total_mean": float(long_df["target_score_ctx"].mean()),
        "target_score_ctx_train_mean": float(long_df.loc[long_df["gate5_split"] == "train", "target_score_ctx"].mean()),
        "target_score_ctx_valid_mean": float(long_df.loc[long_df["gate5_split"] == "valid", "target_score_ctx"].mean()),
        "signal_has_positive_base_target_rate_total": float(long_df["signal_has_positive_base_target"].mean()),
        "target_tp_before_sl_rate_total": float(long_df["target_tp_before_sl"].mean()),
        "target_sl_before_tp_rate_total": float(long_df["target_sl_before_tp"].mean()),
        "target_no_hit_rate_total": float(long_df["target_no_hit"].mean()),
        "target_ambiguous_rate_total": float(long_df["target_ambiguous_same_bar"].mean()),
        "dataset_path": str(OUT_DATASET),
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("TARGET SCORE SUMMARY TOTAL:")
    print(long_df["target_score"].describe().to_string())

    print()
    print("TARGET SCORE SUMMARY TRAIN:")
    print(long_df.loc[long_df["gate5_split"] == "train", "target_score"].describe().to_string())

    print()
    print("TARGET SCORE SUMMARY VALID:")
    print(long_df.loc[long_df["gate5_split"] == "valid", "target_score"].describe().to_string())

    print()
    print("TARGET SCORE CTX SUMMARY TOTAL:")
    print(long_df["target_score_ctx"].describe().to_string())

    print()
    print("TARGET SCORE CTX SUMMARY TRAIN:")
    print(long_df.loc[long_df["gate5_split"] == "train", "target_score_ctx"].describe().to_string())

    print()
    print("TARGET SCORE CTX SUMMARY VALID:")
    print(long_df.loc[long_df["gate5_split"] == "valid", "target_score_ctx"].describe().to_string())


if __name__ == "__main__":
    main()