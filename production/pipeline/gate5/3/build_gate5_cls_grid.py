from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(".")
DATA_PATH = ROOT / "production/dataset/gate5/gate5_2/gate5_grid_ranker_dataset.parquet"
PAIR_DATA_DIR = ROOT / "production/dataset/gate5/gate5_pair_datasets"

OUT_DIR = ROOT / "production/dataset/gate5/gate5_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT = OUT_DIR / "_build_report.json"

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

SAFE_LIST = GRID_LIST
AGGR_LIST = GRID_LIST

DELTA_MIN = 0.25
ENTRY_DELAY_SECONDS = 90


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns: {missing}")


def load_pair_source(pair: str) -> pd.DataFrame:
    path = PAIR_DATA_DIR / f"gate5_dataset_{pair}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"pair source not found: {path}")

    df = pd.read_parquet(path)

    required = [
        "ts",
        "symbol",
        "pred_side",
        "close",
        "atr14",
        f"g5_mfe_side_atr_{pair}",
        f"g5_mae_side_atr_{pair}",
        f"g5_ttl_ret_side_atr_{pair}",
        f"g5_first_tp_minute_{pair}",
        f"g5_first_sl_minute_{pair}",
        f"g5_tp_hit_{pair}",
        f"g5_sl_hit_{pair}",
        f"g5_tp_before_sl_{pair}",
        f"g5_sl_before_tp_{pair}",
        f"g5_ambiguous_same_bar_{pair}",
        f"g5_no_hit_{pair}",
        f"g5_target_{pair}",
    ]
    require_cols(df, required, f"pair_source[{pair}]")

    out = df[required].copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["pred_side"] = out["pred_side"].astype(str).str.upper()

    out = out.rename(
        columns={
            "pred_side": "side",
            "close": f"{pair}_close",
            "atr14": f"{pair}_atr14",
            f"g5_mfe_side_atr_{pair}": f"{pair}_g5_mfe_side_atr",
            f"g5_mae_side_atr_{pair}": f"{pair}_g5_mae_side_atr",
            f"g5_ttl_ret_side_atr_{pair}": f"{pair}_g5_ttl_ret_side_atr",
            f"g5_first_tp_minute_{pair}": f"{pair}_g5_first_tp_minute",
            f"g5_first_sl_minute_{pair}": f"{pair}_g5_first_sl_minute",
            f"g5_tp_hit_{pair}": f"{pair}_g5_tp_hit",
            f"g5_sl_hit_{pair}": f"{pair}_g5_sl_hit",
            f"g5_tp_before_sl_{pair}": f"{pair}_g5_tp_before_sl",
            f"g5_sl_before_tp_{pair}": f"{pair}_g5_sl_before_tp",
            f"g5_ambiguous_same_bar_{pair}": f"{pair}_g5_ambiguous_same_bar",
            f"g5_no_hit_{pair}": f"{pair}_g5_no_hit",
            f"g5_target_{pair}": f"{pair}_g5_target",
        }
    )

    return out


def build_pair(
    df: pd.DataFrame,
    safe_pair: str,
    agg_pair: str,
    safe_src: pd.DataFrame,
    agg_src: pd.DataFrame,
) -> pd.DataFrame:
    safe = df[df["grid_name"] == safe_pair].copy()
    agg = df[df["grid_name"] == agg_pair].copy()

    base_cols = [
        "signal_id",
        "ts",
        "symbol",
        "side",
        "gate5_split",
        "gate5_is_oos",
    ]

    signal_feature_cols = [
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
    ]

    grid_cols = [
        "grid_proba",
        "grid_best_thr",
        "grid_ev_meta",
        "grid_precision_meta",
        "grid_kept_n_meta",
        "grid_coverage_meta",
        "grid_auc_meta",
        "grid_tp_atr",
        "grid_sl_atr",
        "grid_rr",
        "grid_pass",
        "grid_margin_to_thr",
        "grid_proba_to_top1_ratio",
        "grid_margin_to_top1_ratio",
        "grid_tp_minus_sig_max_tp_passed",
        "grid_sl_minus_sig_min_sl_passed",
        "grid_rr_minus_sig_max_rr_passed",
        "grid_proba_vs_sig_mean",
        "grid_margin_vs_sig_mean",
        "grid_ev_meta_vs_sig_mean",
        "grid_precision_meta_vs_sig_mean",
        "target_score",
    ]

    safe = safe[base_cols + signal_feature_cols + grid_cols].copy()
    agg = agg[base_cols + grid_cols].copy()

    safe = safe.rename(columns={c: f"safe_{c}" for c in grid_cols})
    agg = agg.rename(columns={c: f"agg_{c}" for c in grid_cols})

    df_pair = safe.merge(
        agg,
        on=base_cols,
        how="inner",
    )

    df_pair["ts"] = pd.to_datetime(df_pair["ts"], errors="coerce")
    df_pair["symbol"] = df_pair["symbol"].astype(str)
    df_pair["side"] = df_pair["side"].astype(str).str.upper()

    df_pair = df_pair.merge(
        safe_src,
        on=["ts", "symbol", "side"],
        how="left",
    )
    df_pair = df_pair.merge(
        agg_src,
        on=["ts", "symbol", "side"],
        how="left",
    )

    require_cols(
        df_pair,
        [
            f"{safe_pair}_close",
            f"{safe_pair}_atr14",
            f"{agg_pair}_close",
            f"{agg_pair}_atr14",
        ],
        f"df_pair_sources[{safe_pair}__vs__{agg_pair}]",
    )

    same_close_mask = (
        pd.to_numeric(df_pair[f"{safe_pair}_close"], errors="coerce")
        == pd.to_numeric(df_pair[f"{agg_pair}_close"], errors="coerce")
    )
    same_atr_mask = (
        pd.to_numeric(df_pair[f"{safe_pair}_atr14"], errors="coerce")
        == pd.to_numeric(df_pair[f"{agg_pair}_atr14"], errors="coerce")
    )

    bad_close = int((~same_close_mask).sum())
    bad_atr = int((~same_atr_mask).sum())

    if bad_close != 0:
        raise RuntimeError(f"{safe_pair}__vs__{agg_pair}: close mismatch rows={bad_close}")
    if bad_atr != 0:
        raise RuntimeError(f"{safe_pair}__vs__{agg_pair}: atr14 mismatch rows={bad_atr}")

    df_pair["signal_ts_h4"] = df_pair["ts"]
    df_pair["entry_ts_exec"] = df_pair["ts"] + pd.to_timedelta(ENTRY_DELAY_SECONDS, unit="s")
    df_pair["entry_delay_seconds"] = ENTRY_DELAY_SECONDS
    df_pair["entry_px_ref"] = pd.to_numeric(df_pair[f"{safe_pair}_close"], errors="coerce")
    df_pair["atr14_ref"] = pd.to_numeric(df_pair[f"{safe_pair}_atr14"], errors="coerce")

    df_pair = df_pair.rename(
        columns={
            f"{safe_pair}_g5_mfe_side_atr": "safe_g5_mfe_side_atr",
            f"{safe_pair}_g5_mae_side_atr": "safe_g5_mae_side_atr",
            f"{safe_pair}_g5_ttl_ret_side_atr": "safe_g5_ttl_ret_side_atr",
            f"{safe_pair}_g5_first_tp_minute": "safe_g5_first_tp_minute",
            f"{safe_pair}_g5_first_sl_minute": "safe_g5_first_sl_minute",
            f"{safe_pair}_g5_tp_hit": "safe_g5_tp_hit",
            f"{safe_pair}_g5_sl_hit": "safe_g5_sl_hit",
            f"{safe_pair}_g5_tp_before_sl": "safe_g5_tp_before_sl",
            f"{safe_pair}_g5_sl_before_tp": "safe_g5_sl_before_tp",
            f"{safe_pair}_g5_ambiguous_same_bar": "safe_g5_ambiguous_same_bar",
            f"{safe_pair}_g5_no_hit": "safe_g5_no_hit",
            f"{safe_pair}_g5_target": "safe_g5_target",

            f"{agg_pair}_g5_mfe_side_atr": "agg_g5_mfe_side_atr",
            f"{agg_pair}_g5_mae_side_atr": "agg_g5_mae_side_atr",
            f"{agg_pair}_g5_ttl_ret_side_atr": "agg_g5_ttl_ret_side_atr",
            f"{agg_pair}_g5_first_tp_minute": "agg_g5_first_tp_minute",
            f"{agg_pair}_g5_first_sl_minute": "agg_g5_first_sl_minute",
            f"{agg_pair}_g5_tp_hit": "agg_g5_tp_hit",
            f"{agg_pair}_g5_sl_hit": "agg_g5_sl_hit",
            f"{agg_pair}_g5_tp_before_sl": "agg_g5_tp_before_sl",
            f"{agg_pair}_g5_sl_before_tp": "agg_g5_sl_before_tp",
            f"{agg_pair}_g5_ambiguous_same_bar": "agg_g5_ambiguous_same_bar",
            f"{agg_pair}_g5_no_hit": "agg_g5_no_hit",
            f"{agg_pair}_g5_target": "agg_g5_target",
        }
    )

    df_pair = df_pair.drop(
        columns=[
            f"{safe_pair}_close",
            f"{safe_pair}_atr14",
            f"{agg_pair}_close",
            f"{agg_pair}_atr14",
        ],
        errors="ignore",
    )

    df_pair["delta_score"] = df_pair["agg_target_score"] - df_pair["safe_target_score"]

    df_pair["y"] = np.where(
        df_pair["delta_score"] > DELTA_MIN,
        1,
        np.where(df_pair["delta_score"] < -DELTA_MIN, 0, np.nan),
    )

    df_pair = df_pair.dropna(subset=["y"]).copy()
    df_pair["y"] = df_pair["y"].astype(int)

    df_pair["safe_grid_name"] = safe_pair
    df_pair["agg_grid_name"] = agg_pair

    df_pair["proba_diff"] = df_pair["agg_grid_proba"] - df_pair["safe_grid_proba"]
    df_pair["margin_diff"] = df_pair["agg_grid_margin_to_thr"] - df_pair["safe_grid_margin_to_thr"]
    df_pair["rr_diff"] = df_pair["agg_grid_rr"] - df_pair["safe_grid_rr"]
    df_pair["tp_diff"] = df_pair["agg_grid_tp_atr"] - df_pair["safe_grid_tp_atr"]
    df_pair["sl_diff"] = df_pair["agg_grid_sl_atr"] - df_pair["safe_grid_sl_atr"]

    df_pair["proba_ratio"] = np.where(
        pd.to_numeric(df_pair["safe_grid_proba"], errors="coerce") != 0,
        pd.to_numeric(df_pair["agg_grid_proba"], errors="coerce") / pd.to_numeric(df_pair["safe_grid_proba"], errors="coerce"),
        np.nan,
    )
    df_pair["margin_ratio"] = np.where(
        pd.to_numeric(df_pair["safe_grid_margin_to_thr"], errors="coerce") != 0,
        pd.to_numeric(df_pair["agg_grid_margin_to_thr"], errors="coerce") / pd.to_numeric(df_pair["safe_grid_margin_to_thr"], errors="coerce"),
        np.nan,
    )
    df_pair["rr_ratio"] = np.where(
        pd.to_numeric(df_pair["safe_grid_rr"], errors="coerce") != 0,
        pd.to_numeric(df_pair["agg_grid_rr"], errors="coerce") / pd.to_numeric(df_pair["safe_grid_rr"], errors="coerce"),
        np.nan,
    )

    df_pair["safe_is_better"] = (df_pair["safe_target_score"] > df_pair["agg_target_score"]).astype(int)
    df_pair["agg_is_better"] = (df_pair["agg_target_score"] > df_pair["safe_target_score"]).astype(int)

    return df_pair


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    require_cols(
        df,
        [
            "signal_id",
            "ts",
            "symbol",
            "side",
            "grid_name",
            "gate5_split",
            "gate5_is_oos",
            "target_score",
        ],
        "dataset",
    )

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()

    source_map = {}
    for pair in GRID_LIST:
        source_map[pair] = load_pair_source(pair)

    reports = []
    total_built = 0

    for safe in SAFE_LIST:
        for agg in AGGR_LIST:
            if safe == agg:
                continue

            name = f"{safe}__vs__{agg}"
            print("BUILD:", name)

            out = build_pair(
                df=df,
                safe_pair=safe,
                agg_pair=agg,
                safe_src=source_map[safe],
                agg_src=source_map[agg],
            )

            out_path = OUT_DIR / f"{name}.parquet"
            out.to_parquet(out_path, index=False)

            row = {
                "pair_name": name,
                "safe_grid": safe,
                "agg_grid": agg,
                "rows_total": int(len(out)),
                "rows_train": int((out["gate5_split"] == "train").sum()),
                "rows_valid": int((out["gate5_split"] == "valid").sum()),
                "target_mean_total": float(out["y"].mean()) if len(out) else None,
                "target_mean_train": float(out.loc[out["gate5_split"] == "train", "y"].mean()) if len(out.loc[out["gate5_split"] == "train"]) else None,
                "target_mean_valid": float(out.loc[out["gate5_split"] == "valid", "y"].mean()) if len(out.loc[out["gate5_split"] == "valid"]) else None,
                "mean_safe_score_total": float(out["safe_target_score"].mean()) if len(out) else None,
                "mean_agg_score_total": float(out["agg_target_score"].mean()) if len(out) else None,
                "mean_delta_score_total": float(out["delta_score"].mean()) if len(out) else None,
                "dataset_path": str(out_path),
            }
            reports.append(row)

            total_built += 1
            print("ROWS:", len(out))

    report_df = pd.DataFrame(reports).sort_values(
        ["rows_valid", "rows_total", "pair_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    report_csv = OUT_DIR / "_build_report.csv"
    report_df.to_csv(report_csv, index=False)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print()
    print("TOTAL BUILT:", total_built)
    print("WROTE:", report_csv)
    print("WROTE:", OUT_REPORT)


if __name__ == "__main__":
    main()