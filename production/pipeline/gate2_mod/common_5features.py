from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


REACH_ALL_PATH = "production/dataset/final_gate2_2_directional_reach_5features_all.parquet"
STRENGTH_ALL_PATH = "production/dataset/final_gate2_3_directional_strength_5features_all.parquet"
OUT_ROOT = "production/models/gate2_mod_5features"


DROP_COLS_EXACT = {
    "entry_ts",
    "signal_ts",
    "entry_bar_open_ts",
    "entry_ts_exec",
    "entry_px_exec",
    "upstream_split",
    "upstream_valid_start_ts",

    "entry_px",
    "y",
    "y_fast",
    "pnl_net",
    "exit_px",
    "exit_ts",
    "exit_reason",
    "tp_px",
    "sl_px",
    "side",

    "ks_ret_adj",
    "ks_ttl_hours_best",
    "ks_tp_abs_best",
    "ks_sl_abs_best",
    "ks_tp_abs",
    "ks_sl_abs",
    "ks_tp_scale",
    "ks_sl_scale",
    "ks_ttl_hours",

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

    "ttl_hours",
    "impulse_hours",
    "atr14_at_signal",

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
}

DROP_COLS_PREFIX = (
    "Unnamed:",
)

BANNED_SUBSTRINGS = [
    "future",
    "target",
    "label",
    "outcome",
    "realized",
    "exit",
    "valid_start",
    "first_",
    "_ts",
    "forward",
    "next_",
]

BANNED_EXACT = {
    "y",
    "y_fast",
    "pnl_net",
    "exit_px",
    "exit_ts",
    "exit_reason",
    "tp_px",
    "sl_px",
    "side",
    "entry_px_exec",
    "atr14_at_signal",
    "ttl_hours",
    "impulse_hours",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_feature_cols(df: pd.DataFrame) -> List[str]:
    feature_cols: List[str] = []

    for c in df.columns:
        c_low = str(c).lower()

        if c == "symbol":
            feature_cols.append(c)
            continue

        if c in DROP_COLS_EXACT:
            continue
        if c in BANNED_EXACT:
            continue
        if c in {"ref_close", "ref_btc_close", "ref_eth_close"}:
            continue
        if any(str(c).startswith(p) for p in DROP_COLS_PREFIX):
            continue
        if any(x in c_low for x in BANNED_SUBSTRINGS):
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

    if "signal_ts" not in df.columns:
        raise ValueError("split_train_valid: missing signal_ts and missing upstream_split")

    tmp = df.copy()
    tmp["signal_ts"] = pd.to_datetime(tmp["signal_ts"], errors="coerce")
    tmp = tmp.dropna(subset=["signal_ts"]).sort_values("signal_ts").reset_index(drop=True)

    n = len(tmp)
    n_valid = max(1, int(round(n * 0.20)))
    n_train = n - n_valid

    train_df = tmp.iloc[:n_train].copy()
    valid_df = tmp.iloc[n_train:].copy()
    return train_df, valid_df, "tail_split"


def prepare_xy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
):
    cat_cols = ["symbol"] if "symbol" in feature_cols else []
    num_cols = [c for c in feature_cols if c not in cat_cols]

    x_train_num = train_df[num_cols].replace([np.inf, -np.inf], np.nan)
    x_valid_num = valid_df[num_cols].replace([np.inf, -np.inf], np.nan)

    med = x_train_num.median(numeric_only=True)
    x_train_num = x_train_num.fillna(med)
    x_valid_num = x_valid_num.fillna(med)

    valid_cols = x_train_num.columns[x_train_num.notna().sum() > 0]
    x_train_num = x_train_num[valid_cols]
    x_valid_num = x_valid_num[valid_cols]

    const_cols = [c for c in x_train_num.columns if x_train_num[c].nunique(dropna=True) <= 1]
    x_train_num = x_train_num.drop(columns=const_cols)
    x_valid_num = x_valid_num.drop(columns=const_cols)

    med = x_train_num.median()
    x_train_num = x_train_num.fillna(med)
    x_valid_num = x_valid_num.fillna(med)

    if cat_cols:
        x_train = pd.concat(
            [train_df[cat_cols].astype(str).reset_index(drop=True), x_train_num.reset_index(drop=True)],
            axis=1,
        )
        x_valid = pd.concat(
            [valid_df[cat_cols].astype(str).reset_index(drop=True), x_valid_num.reset_index(drop=True)],
            axis=1,
        )
    else:
        x_train = x_train_num
        x_valid = x_valid_num

    y_train = train_df[target_col]
    y_valid = valid_df[target_col]

    return x_train, x_valid, y_train, y_valid


def make_top_bucket_report(
    df_valid: pd.DataFrame,
    proba_col: str,
    y_col: str,
    thresholds: tuple = (0.90, 0.95, 0.97, 0.99),
) -> pd.DataFrame:
    rows = []

    work = df_valid.copy()
    work[proba_col] = pd.to_numeric(work[proba_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[proba_col, y_col]).copy()

    for thr in thresholds:
        part = work[work[proba_col] >= thr].copy()
        rows.append(
            {
                "threshold": float(thr),
                "rows": int(len(part)),
                "share": float(len(part) / len(work)) if len(work) else 0.0,
                "precision": float(part[y_col].mean()) if len(part) else np.nan,
            }
        )

    return pd.DataFrame(rows)