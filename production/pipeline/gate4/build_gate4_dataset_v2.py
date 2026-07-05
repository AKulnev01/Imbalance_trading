from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DATA_DIR = "production/dataset/gate1"
GATE3_DATA_DIR = "production/dataset/pa_gate3_v3_long_short_by_symbol"

GATE1_MODELS_DIR = "production/models/final_gate1"
GATE2_MOD_DIR = "production/models/gate2_mod_5features"
GATE3_SCORE_ROOT = "production/models/final_gate3_score_long_short"
POLICY_CSV = "production/models/ks/gate3_symbol_policy.csv.updated"

OUT_ROOT = "production/dataset/gate4/gate4_1_side_builder"
OUT_RAW_PARQUET = os.path.join(OUT_ROOT, "gate4_1_candidates_raw.parquet")
OUT_DATASET_PARQUET = os.path.join(OUT_ROOT, "gate4_1_side_dataset.parquet")
OUT_AUDIT_CSV = os.path.join(OUT_ROOT, "_AUDIT.csv")
OUT_REPORT_JSON = os.path.join(OUT_ROOT, "_REPORT.json")


# ============================================================
# CONFIG
# ============================================================

TTL_BARS = 4
DIR_MARGIN_ATR = 0.15
DIR_FIRST_HIT_ATR = 0.35
DIR_LABEL_MODE = "first_hit"
UPSTREAM_VALID_TAIL_SHARE = 0.20

TRAIN_END = os.environ.get("IMB_OFFLINE_TRAIN_END", "")
VALID_START = os.environ.get("IMB_OFFLINE_VALID_START", "")
VALID_END = os.environ.get("IMB_OFFLINE_VALID_END", "")

GATE1_PROBA_MIN = 0.50

G2_CLS_BASE_MIN = 0.50
G2_CLS_EXTREME_MIN = 0.90

G3_SCORE_EXTREME_MIN = 0.90
G3_SCORE_FALLBACK_MIN = 0.50

REQUIRE_GATE1_PASS = True
REQUIRE_FULL_GATE3_BUNDLE = False

PRIMARY_DELTA = 0.50
AUX_DELTAS = [0.60, 0.75]
ALL_DELTAS = [PRIMARY_DELTA, *AUX_DELTAS]


# ============================================================
# УЗКИЙ WHITELIST ДЛЯ MARKET CONTEXT
# Только базовые колонки, без _feat, без старых target/side/ks/pnl/exit
# ============================================================

BASE_CONTEXT_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",


    "atr14",
    "atr4h",
    "atr_ratio6",
    "atr_to_price",
    "atr_rank_48",
    "volat_ret12",
    "vol_ratio6",
    "vol_regime",
    "regime_index",
    "market_heat",
    "amihud20",

    "ret_l1",
    "ret_l2",
    "rng_pct",
    "prev_day_close",
    "prev_day_range",
    "prev_day_ret",

    "ref_close",
    "ref_btc_close",
    "ref_eth_close",
    "ret_vs_btc",
    "ret_vs_eth",
    "ret_vs_ref",
    "ret_vs_btc_z",
    "ret_vs_eth_z",
    "ret_vs_ref_z",

    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_monday",
    "hod_sin",
    "hod_cos",
    "sess_asia",
    "sess_eu",
    "sess_us",

    "body",
    "body_pct_rng",
    "upper_wick",
    "lower_wick",
    "wick_asymmetry",
    "body_to_prev",
    "doji_score",
    "hammer_like",
    "pinbar_like",
    "engulf_bull",
    "engulf_bear",
    "fvg_bull",
    "fvg_bear",
    "fvg_size",
    "bar_sequence_len",
    "candle_entropy",

    "sma20",
    "sma50",
    "sma100",
    "ema12",
    "ema26",
    "slope6",
    "slope12",
    "momentum6",
    "momentum12",
    "trend_strength",
    "cross_fast_slow",

    "rsi14",
    "rsi_z",
    "cci20",
    "mfi14",
    "adx14",
    "plus_di",
    "minus_di",
    "macd",
    "macd_sig",
    "macd_hist",
    "bb_width",
    "bbp",
    "range_z",

    "dist_to_high",
    "dist_to_low",
    "hl_spread_ratio",
    "hl_spread_med48",
    "gap_to_prev_close",
    "local_high_break",
    "local_low_break",
    "price_distance_ma20",
    "price_vs_vwap",
    "price_vol_corr12",
    "momentum_vol_corr",
]

FORBIDDEN_EXACT = {
    "symbol",
    "side",
    "side_num",
    "entry_ts",
    "entry_px",
    "exit_ts",
    "exit_px",
    "exit_reason",
    "pnl_net",
    "y",
    "y_fast",
    "tp_px",
    "sl_px",
    "ret",
    "dir_prev",
    "ks_tp_scale",
    "ks_sl_scale",
    "ks_ttl_hours",
    "ks_tp_abs",
    "ks_sl_abs",
    "ks_ret_adj",
    "ks_tp_abs_best",
    "ks_sl_abs_best",
    "ks_ttl_hours_best",
}


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json_safe(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def parse_optional_ts(value: str, name: str) -> Optional[pd.Timestamp]:
    raw = str(value or "").strip()
    if not raw:
        return None

    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        raise SystemExit("bad {}: {}".format(name, value))

    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        out = out.tz_convert("UTC").tz_localize(None)

    return pd.Timestamp(out).floor("min")


def parse_runtime_args():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--valid-start", default=VALID_START)
    parser.add_argument("--valid-end", default=VALID_END)

    args, _ = parser.parse_known_args()
    return args


def apply_runtime_args(args) -> None:
    global TRAIN_END
    global VALID_START
    global VALID_END

    TRAIN_END = str(args.train_end or "").strip()
    VALID_START = str(args.valid_start or "").strip()
    VALID_END = str(args.valid_end or "").strip()


def refresh_output_paths() -> None:
    global OUT_RAW_PARQUET
    global OUT_DATASET_PARQUET
    global OUT_AUDIT_CSV
    global OUT_REPORT_JSON

    OUT_RAW_PARQUET = os.path.join(OUT_ROOT, "gate4_1_candidates_raw.parquet")
    OUT_DATASET_PARQUET = os.path.join(OUT_ROOT, "gate4_1_side_dataset.parquet")
    OUT_AUDIT_CSV = os.path.join(OUT_ROOT, "_AUDIT.csv")
    OUT_REPORT_JSON = os.path.join(OUT_ROOT, "_REPORT.json")


def build_gate4_split_config() -> dict:
    train_end = parse_optional_ts(TRAIN_END, "--train-end")
    valid_start = parse_optional_ts(VALID_START, "--valid-start")
    valid_end = parse_optional_ts(VALID_END, "--valid-end")

    provided = [train_end is not None, valid_start is not None, valid_end is not None]

    if any(provided) and not all(provided):
        raise SystemExit(
            "split args must be provided together: --train-end --valid-start --valid-end"
        )

    if train_end is None:
        return {
            "mode": "legacy_meta_or_tail",
            "train_end": None,
            "valid_start": None,
            "valid_end": None,
            "train_safe_cutoff": None,
        }

    if train_end > valid_start:
        raise SystemExit(
            "--train-end must be <= --valid-start, got train_end={} valid_start={}".format(
                train_end,
                valid_start,
            )
        )

    if valid_start >= valid_end:
        raise SystemExit(
            "--valid-start must be < --valid-end, got valid_start={} valid_end={}".format(
                valid_start,
                valid_end,
            )
        )

    train_safe_cutoff = train_end - pd.Timedelta(hours=4 * int(TTL_BARS))

    return {
        "mode": "fixed_time_train_safe",
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "train_safe_cutoff": train_safe_cutoff,
    }


def apply_gate4_split(
    df: pd.DataFrame,
    split_config: dict,
    legacy_valid_start_ts: Optional[pd.Timestamp],
) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")

    if split_config.get("mode") == "fixed_time_train_safe":
        train_end = pd.Timestamp(split_config["train_end"])
        valid_start = pd.Timestamp(split_config["valid_start"])
        valid_end = pd.Timestamp(split_config["valid_end"])
        train_safe_cutoff = pd.Timestamp(split_config["train_safe_cutoff"])

        train_mask = out["ts"] < train_safe_cutoff
        valid_mask = (out["ts"] >= valid_start) & (out["ts"] < valid_end)

        out["upstream_train_end_ts"] = train_end
        out["upstream_train_safe_cutoff_ts"] = train_safe_cutoff
        out["upstream_valid_start_ts"] = valid_start
        out["upstream_valid_end_ts"] = valid_end
        out["upstream_split"] = np.select(
            [train_mask, valid_mask],
            ["train", "valid"],
            default="gap",
        )
        out["upstream_is_oos"] = (out["upstream_split"] == "valid").astype(int)
        return out

    out["upstream_train_end_ts"] = pd.NaT
    out["upstream_train_safe_cutoff_ts"] = pd.NaT
    out["upstream_valid_end_ts"] = pd.NaT
    out["upstream_valid_start_ts"] = legacy_valid_start_ts

    if legacy_valid_start_ts is not None:
        out["upstream_split"] = np.where(out["ts"] >= legacy_valid_start_ts, "valid", "train")
        out["upstream_is_oos"] = (out["ts"] >= legacy_valid_start_ts).astype(int)
    else:
        out["upstream_split"] = ""
        out["upstream_is_oos"] = np.nan

    return out

def extract_gate3_meta_features(meta: dict, fallback_thr: float) -> dict:
    stats = meta.get("stats", {}) if isinstance(meta.get("stats", {}), dict) else {}

    best_thr = pd.to_numeric(meta.get("best_threshold", fallback_thr), errors="coerce")
    if not np.isfinite(best_thr):
        best_thr = fallback_thr

    return {
        "threshold": float(best_thr) if np.isfinite(best_thr) else np.nan,
        "precision_meta": float(pd.to_numeric(stats.get("precision", np.nan), errors="coerce")),
        "wilson_meta": float(pd.to_numeric(stats.get("wilson_lower", np.nan), errors="coerce")),
        "delta_wilson_meta": float(pd.to_numeric(stats.get("delta_wilson", np.nan), errors="coerce")),
        "pvalue_meta": float(pd.to_numeric(stats.get("p_value", np.nan), errors="coerce")),
        "kept_n_meta": float(pd.to_numeric(meta.get("best_threshold_kept_n", np.nan), errors="coerce")),
        "valid_pos_rate_meta": float(pd.to_numeric(meta.get("best_threshold_kept_pos_rate", np.nan), errors="coerce")),
        "thr_kept_lift_meta": float(pd.to_numeric(meta.get("best_threshold_kept_lift", np.nan), errors="coerce")),
    }


def to_naive_utc_auto(x) -> pd.Series:
    s = pd.Series(x)

    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").astype("float64")
        finite = vals[np.isfinite(vals)]
        vmax = float(finite.max()) if len(finite) else 0.0

        if vmax > 1e18:
            out = pd.to_datetime(vals, unit="ns", utc=True, errors="coerce")
        elif vmax > 1e14:
            out = pd.to_datetime(vals, unit="us", utc=True, errors="coerce")
        elif vmax > 1e11:
            out = pd.to_datetime(vals, unit="ms", utc=True, errors="coerce")
        else:
            out = pd.to_datetime(vals, unit="s", utc=True, errors="coerce")
    else:
        out = pd.to_datetime(s, utc=True, errors="coerce")

    return pd.Series(out).dt.tz_localize(None)


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).gt(0.5).astype(int)


def build_base_feature_cols(df: pd.DataFrame) -> List[str]:
    out: List[str] = []

    for c in BASE_CONTEXT_COLS:
        if c not in df.columns:
            continue
        if c in FORBIDDEN_EXACT:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        out.append(c)

    extra_feat_cols = []
    for c in df.columns:
        if c in FORBIDDEN_EXACT:
            continue
        if not str(c).endswith("_feat"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        extra_feat_cols.append(c)

    out.extend(sorted(set(extra_feat_cols)))
    return sorted(set(out))

def find_first_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def try_parse_ts(x) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    try:
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
    except Exception:
        return None


def infer_valid_start_ts_from_meta(
    meta: dict,
    ts_series: pd.Series,
    fallback_tail_share: float,
) -> Optional[pd.Timestamp]:
    ts_sorted = pd.Series(ts_series).dropna().sort_values().reset_index(drop=True)
    if len(ts_sorted) == 0:
        return None

    roots = [
        meta,
        meta.get("split", {}) if isinstance(meta.get("split", {}), dict) else {},
        meta.get("validation", {}) if isinstance(meta.get("validation", {}), dict) else {},
        meta.get("oos", {}) if isinstance(meta.get("oos", {}), dict) else {},
        meta.get("dataset_split", {}) if isinstance(meta.get("dataset_split", {}), dict) else {},
        meta.get("train_config", {}) if isinstance(meta.get("train_config", {}), dict) else {},
    ]

    for root in roots:
        for k in [
            "valid_start_ts",
            "validation_start_ts",
            "valid_from",
            "oos_start_ts",
            "oos_from",
            "test_start_ts",
            "split_valid_start_ts",
            "valid_start",
        ]:
            if k in root:
                ts = try_parse_ts(root.get(k))
                if ts is not None:
                    return ts

    train_candidates = [
        meta.get("rows_train"),
        meta.get("train_rows"),
        meta.get("n_train"),
        meta.get("episodes_train"),
        meta.get("train_episodes"),
    ]

    train_n = None
    for v in train_candidates:
        if v is None:
            continue
        try:
            train_n = int(v)
            break
        except Exception:
            pass

    if train_n is not None and 0 < train_n < len(ts_sorted):
        return pd.Timestamp(ts_sorted.iloc[train_n])

    valid_share = None
    for root in roots:
        for k in ["valid_share", "validation_share", "valid_frac", "validation_frac"]:
            if k in root:
                try:
                    valid_share = float(root.get(k))
                    break
                except Exception:
                    pass
        if valid_share is not None:
            break

    if valid_share is None:
        valid_share = float(fallback_tail_share)

    if not np.isfinite(valid_share):
        return None

    valid_share = min(max(valid_share, 0.01), 0.95)
    train_n = int(np.floor(len(ts_sorted) * (1.0 - valid_share)))
    train_n = max(1, min(train_n, len(ts_sorted) - 1))

    if 0 <= train_n < len(ts_sorted):
        return pd.Timestamp(ts_sorted.iloc[train_n])

    return None


def resolve_validation_start_ts(
    ts_series: pd.Series,
    meta_paths: List[str],
    fallback_tail_share: float,
) -> Optional[pd.Timestamp]:
    vals = []
    for p in meta_paths:
        meta = load_json_safe(p)
        ts = infer_valid_start_ts_from_meta(
            meta=meta,
            ts_series=ts_series,
            fallback_tail_share=fallback_tail_share,
        )
        if ts is not None:
            vals.append(ts)

    if not vals:
        return None
    return max(vals)


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)

    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    return pd.Series(tr).rolling(14).mean()

def rolling_wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)

    def _wma(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(window, min_periods=window).apply(_wma, raw=True)


def add_symbol_dummies(df: pd.DataFrame, symbol: str, required_features: List[str]) -> pd.DataFrame:
    sym_cols = list(dict.fromkeys([c for c in required_features if c.startswith("sym_")]))
    if not sym_cols:
        return df

    values = {}

    for c in sym_cols:
        if c in df.columns:
            continue

        name = c[4:]
        if c == "sym_is_other":
            continue

        values[c] = np.full(len(df), int(symbol == name), dtype=np.int8)

    if "sym_is_other" in sym_cols and "sym_is_other" not in df.columns:
        known_symbols = {c[4:] for c in sym_cols if c != "sym_is_other"}
        values["sym_is_other"] = np.full(len(df), int(symbol not in known_symbols), dtype=np.int8)

    if not values:
        return df

    block = pd.DataFrame(values, index=df.index)
    return pd.concat([df, block], axis=1).copy()



def add_lag_features_if_needed(df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    lag_feats = [c for c in required_features if c.endswith("_lag1") or c.endswith("_lag2")]
    if not lag_feats:
        return df

    block = pd.DataFrame(index=df.index)

    for feat in lag_feats:
        if feat.endswith("_lag1"):
            base = feat[:-5]
            if base in df.columns:
                block[feat] = pd.to_numeric(df[base], errors="coerce").shift(1)

        if feat.endswith("_lag2"):
            base = feat[:-5]
            if base in df.columns:
                block[feat] = pd.to_numeric(df[base], errors="coerce").shift(2)

    if block.empty:
        return df

    return pd.concat([df, block], axis=1).copy()


def prepare_model_input(
    base_df: pd.DataFrame,
    symbol: str,
    required_features: List[str],
    active_cols: Optional[List[str]] = None,
    active_prefix: Optional[str] = None,
) -> pd.DataFrame:
    df = base_df.copy()
    required_features = list(dict.fromkeys(required_features))
    df = add_common_missing_features(df=df, required_features=required_features)
    df = add_symbol_dummies(df=df, symbol=symbol, required_features=required_features)

    if active_cols and active_prefix:
        need_active = False
        for feat in required_features:
            if feat.startswith(f"{active_prefix}_"):
                need_active = True
                break
            if "__age" in feat or "__fresh" in feat or "__mid" in feat or "__late" in feat:
                need_active = True
                break

        if need_active:
            active_block = add_active_set_features(df=df, active_cols=active_cols, prefix=active_prefix)
            df = active_block.copy()

    df = add_lag_features_if_needed(df=df, required_features=required_features)
    df = df.copy()

    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        missing_block = pd.DataFrame({col: np.nan for col in missing_cols}, index=df.index)
        df = pd.concat([df, missing_block], axis=1).copy()

    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df


def add_common_missing_features(df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    need = set(required_features)
    out = df.copy()

    close = pd.to_numeric(out["close"], errors="coerce") if "close" in out.columns else pd.Series(np.nan, index=out.index)
    open_ = pd.to_numeric(out["open"], errors="coerce") if "open" in out.columns else pd.Series(np.nan, index=out.index)
    high = pd.to_numeric(out["high"], errors="coerce") if "high" in out.columns else pd.Series(np.nan, index=out.index)
    low = pd.to_numeric(out["low"], errors="coerce") if "low" in out.columns else pd.Series(np.nan, index=out.index)
    volume = pd.to_numeric(out["volume"], errors="coerce") if "volume" in out.columns else pd.Series(np.nan, index=out.index)

    if "ret" in need and "ret" not in out.columns:
        out["ret"] = close.pct_change().replace([np.inf, -np.inf], np.nan)

    if "side_num" in need and "side_num" not in out.columns:
        out["side_num"] = 0.0

    if "dir_prev" in need and "dir_prev" not in out.columns:
        prev_body = (close.shift(1) - open_.shift(1)).fillna(0.0)
        out["dir_prev"] = np.sign(prev_body).astype(float)

    if "sma5" in need and "sma5" not in out.columns:
        out["sma5"] = close.rolling(5, min_periods=5).mean()

    if "sma10" in need and "sma10" not in out.columns:
        out["sma10"] = close.rolling(10, min_periods=10).mean()

    if "wma10" in need and "wma10" not in out.columns:
        out["wma10"] = rolling_wma(close, 10)

    if "vol_delta" in need and "vol_delta" not in out.columns:
        out["vol_delta"] = volume.diff()

    if "vol_med20" in need and "vol_med20" not in out.columns:
        out["vol_med20"] = volume.rolling(20, min_periods=20).median()

    if "vol_med48" in need and "vol_med48" not in out.columns:
        out["vol_med48"] = volume.rolling(48, min_periods=48).median()

    if "vol_z" in need and "vol_z" not in out.columns:
        vol_mean20 = volume.rolling(20, min_periods=20).mean()
        vol_std20 = volume.rolling(20, min_periods=20).std()
        out["vol_z"] = (volume - vol_mean20) / vol_std20.replace(0.0, np.nan)

    if "obv" in need and "obv" not in out.columns:
        direction = np.sign(close.diff().fillna(0.0))
        out["obv"] = (direction * volume.fillna(0.0)).cumsum()

    if "vwap20" in need and "vwap20" not in out.columns:
        typical = (high + low + close) / 3.0
        num = (typical * volume).rolling(20, min_periods=20).sum()
        den = volume.rolling(20, min_periods=20).sum()
        out["vwap20"] = num / den.replace(0.0, np.nan)

    if "atr14" not in out.columns and {"high", "low", "close"}.issubset(out.columns):
        out["atr14"] = compute_atr14(out)

    if "atr_slope" in need and "atr_slope" not in out.columns:
        atr14 = pd.to_numeric(out["atr14"], errors="coerce")
        out["atr_slope"] = atr14.diff()

    if "body_vs_wick" in need and "body_vs_wick" not in out.columns:
        body = (close - open_).abs()
        upper = high - np.maximum(open_, close)
        lower = np.minimum(open_, close) - low
        wick = upper + lower
        out["body_vs_wick"] = body / wick.replace(0.0, np.nan)

    if "zero_vol_share48" in need and "zero_vol_share48" not in out.columns:
        out["zero_vol_share48"] = (volume.fillna(0.0) <= 0.0).astype(float).rolling(48, min_periods=1).mean()

    if "ctx_ret1" in need and "ctx_ret1" not in out.columns:
        out["ctx_ret1"] = close.pct_change(1).replace([np.inf, -np.inf], np.nan)

    if "ctx_ret2" in need and "ctx_ret2" not in out.columns:
        out["ctx_ret2"] = close.pct_change(2).replace([np.inf, -np.inf], np.nan)

    if "ctx_ret4" in need and "ctx_ret4" not in out.columns:
        out["ctx_ret4"] = close.pct_change(4).replace([np.inf, -np.inf], np.nan)

    if "ctx_ret8" in need and "ctx_ret8" not in out.columns:
        out["ctx_ret8"] = close.pct_change(8).replace([np.inf, -np.inf], np.nan)

    if "ctx_atrp14" in need and "ctx_atrp14" not in out.columns:
        atr14 = pd.to_numeric(out["atr14"], errors="coerce")
        out["ctx_atrp14"] = atr14 / close.replace(0.0, np.nan)

    if "ctx_range_atr" in need and "ctx_range_atr" not in out.columns:
        atr14 = pd.to_numeric(out["atr14"], errors="coerce")
        out["ctx_range_atr"] = (high - low) / atr14.replace(0.0, np.nan)

    return out


def compute_future_direction_labels(
    df: pd.DataFrame,
    ttl_bars: int,
    dir_margin_atr: float,
    first_hit_atr: float,
    label_mode: str,
) -> pd.DataFrame:
    out = df[["ts", "open", "high", "low", "close"]].copy()

    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    atr14 = compute_atr14(out).to_numpy(dtype=float)

    mfe_up = np.full(len(out), np.nan, dtype=float)
    mfe_dn = np.full(len(out), np.nan, dtype=float)
    first_up_bar = np.full(len(out), np.nan, dtype=float)
    first_dn_bar = np.full(len(out), np.nan, dtype=float)

    y_dir_mfe = np.full(len(out), "DROP", dtype=object)
    y_dir_first = np.full(len(out), "DROP", dtype=object)

    for i in range(len(out)):
        start = i + 1
        end = min(len(out), i + 1 + ttl_bars)

        if start >= end:
            continue

        if not (np.isfinite(close[i]) and np.isfinite(atr14[i]) and atr14[i] > 0):
            continue

        hh = np.max(high[start:end])
        ll = np.min(low[start:end])

        up = (hh - close[i]) / atr14[i]
        dn = (close[i] - ll) / atr14[i]

        mfe_up[i] = up
        mfe_dn[i] = dn

        if up > dn + dir_margin_atr:
            y_dir_mfe[i] = "BUY"
        elif dn > up + dir_margin_atr:
            y_dir_mfe[i] = "SELL"
        else:
            y_dir_mfe[i] = "DROP"

        up_hit_idx = None
        dn_hit_idx = None

        for j in range(start, end):
            up_j = (high[j] - close[i]) / atr14[i]
            dn_j = (close[i] - low[j]) / atr14[i]

            if up_hit_idx is None and up_j >= first_hit_atr:
                up_hit_idx = j - i

            if dn_hit_idx is None and dn_j >= first_hit_atr:
                dn_hit_idx = j - i

            if up_hit_idx is not None and dn_hit_idx is not None:
                break

        first_up_bar[i] = float(up_hit_idx) if up_hit_idx is not None else np.nan
        first_dn_bar[i] = float(dn_hit_idx) if dn_hit_idx is not None else np.nan

        if up_hit_idx is not None and dn_hit_idx is None:
            y_dir_first[i] = "BUY"
        elif dn_hit_idx is not None and up_hit_idx is None:
            y_dir_first[i] = "SELL"
        elif up_hit_idx is not None and dn_hit_idx is not None:
            if up_hit_idx < dn_hit_idx:
                y_dir_first[i] = "BUY"
            elif dn_hit_idx < up_hit_idx:
                y_dir_first[i] = "SELL"
            else:
                y_dir_first[i] = "DROP"
        else:
            y_dir_first[i] = "DROP"

    y_dir = y_dir_first if label_mode == "first_hit" else y_dir_mfe

    out["mfe_up_atr_16h"] = mfe_up
    out["mfe_dn_atr_16h"] = mfe_dn
    out["first_up_hit_bar"] = first_up_bar
    out["first_dn_hit_bar"] = first_dn_bar
    out["y_dir_mfe"] = y_dir_mfe
    out["y_dir_first"] = y_dir_first
    out["y_dir"] = y_dir
    out["y_dir_int"] = np.where(out["y_dir"] == "BUY", 1, np.where(out["y_dir"] == "SELL", 0, 2))

    return out

def delta_to_suffix(delta: float) -> str:
    return f"{int(round(delta * 100)):03d}"


def build_side_clean_labels(
    df: pd.DataFrame,
    deltas: List[float],
) -> pd.DataFrame:
    out = df.copy()

    mfe_up = pd.to_numeric(out["mfe_up_atr_16h"], errors="coerce")
    mfe_dn = pd.to_numeric(out["mfe_dn_atr_16h"], errors="coerce")

    for delta in deltas:
        suffix = delta_to_suffix(delta)

        edge_long = mfe_up - mfe_dn
        edge_short = mfe_dn - mfe_up

        side_col = f"y_side_clean_delta_{suffix}"
        side_int_col = f"y_side_clean_delta_{suffix}_int"
        edge_col = f"edge_delta_{suffix}"
        abs_edge_col = f"abs_edge_delta_{suffix}"
        is_long_col = f"is_long_delta_{suffix}"
        is_short_col = f"is_short_delta_{suffix}"
        is_ambig_col = f"is_ambig_delta_{suffix}"

        out[edge_col] = edge_long
        out[abs_edge_col] = edge_long.abs()

        out[side_col] = np.where(
            edge_long > delta,
            "LONG",
            np.where(edge_short > delta, "SHORT", "AMBIG"),
        )

        out[side_int_col] = np.where(
            out[side_col] == "LONG",
            1,
            np.where(out[side_col] == "SHORT", -1, 0),
        )

        out[is_long_col] = (out[side_col] == "LONG").astype(int)
        out[is_short_col] = (out[side_col] == "SHORT").astype(int)
        out[is_ambig_col] = (out[side_col] == "AMBIG").astype(int)

    primary_suffix = delta_to_suffix(PRIMARY_DELTA)

    out["y_side_clean"] = out[f"y_side_clean_delta_{primary_suffix}"]
    out["y_side_clean_int"] = out[f"y_side_clean_delta_{primary_suffix}_int"]
    out["edge_atr_clean"] = out[f"edge_delta_{primary_suffix}"]
    out["abs_edge_atr_clean"] = out[f"abs_edge_delta_{primary_suffix}"]
    out["is_side_clean_long"] = (out["y_side_clean"] == "LONG").astype(int)
    out["is_side_clean_short"] = (out["y_side_clean"] == "SHORT").astype(int)
    out["is_side_clean_ambig"] = (out["y_side_clean"] == "AMBIG").astype(int)

    return out

def model_predict_proba(
    model_path: str,
    prepared_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], List[str]]:
    model = CatBoostClassifier()
    model.load_model(model_path)

    feats = list(dict.fromkeys(model.feature_names_ or []))
    missing = [c for c in feats if c not in prepared_df.columns]
    if missing:
        return None, feats, missing

    prepared_df = prepared_df.loc[:, ~prepared_df.columns.duplicated(keep="first")].copy()
    x = prepared_df[feats].replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, ~x.columns.duplicated(keep="first")].copy()
    med = x.median(numeric_only=True)
    x = x.fillna(med)
    x = x.fillna(0.0)

    proba = model.predict_proba(x)[:, 1]
    return proba, feats, []




def select_side_pattern_cols(cols: List[str], side: str) -> List[str]:
    side = str(side).strip().lower()
    out = []

    for c in cols:
        c_low = str(c).lower()

        is_long = ("_up" in c_low) or ("buy" in c_low) or ("_lo_" in c_low)
        is_short = ("_dn" in c_low) or ("sell" in c_low) or ("_hi_" in c_low)

        if side == "long" and is_long and not is_short:
            out.append(c)
        if side == "short" and is_short and not is_long:
            out.append(c)

    return sorted(set(out))


def add_active_set_features(df: pd.DataFrame, active_cols: List[str], prefix: str) -> pd.DataFrame:
    block = pd.DataFrame(index=df.index)

    if prefix == "g3_long":
        primary_col = "active_pa_atr_squeeze_break_up"
        secondary_col = "active_pa_bos_up_24"
    elif prefix == "g3_short":
        primary_col = "active_pa_atr_squeeze_break_dn"
        secondary_col = "active_pa_bos_dn_24"
    else:
        primary_col = ""
        secondary_col = ""

    if not active_cols:
        block[f"{prefix}_any_active"] = 0
        block[f"{prefix}_active_count"] = 0
        block[f"{prefix}_active_primary"] = 0
        block[f"{prefix}_active_secondary"] = 0
        block[f"{prefix}_active_overlap_primary_secondary"] = 0
        block[f"{prefix}_max_active_age"] = 0
        return pd.concat([df, block], axis=1).copy()

    act = df[active_cols].copy()
    for c in active_cols:
        act[c] = safe_bool_series(act[c])

    act_sum = act.sum(axis=1)

    block[f"{prefix}_any_active"] = (act_sum > 0).astype(int)
    block[f"{prefix}_active_count"] = act_sum.astype(int)

    if primary_col and primary_col in act.columns:
        block[f"{prefix}_active_primary"] = act[primary_col].astype(int)
    else:
        block[f"{prefix}_active_primary"] = 0

    if secondary_col and secondary_col in act.columns:
        block[f"{prefix}_active_secondary"] = act[secondary_col].astype(int)
    else:
        block[f"{prefix}_active_secondary"] = 0

    block[f"{prefix}_active_overlap_primary_secondary"] = (
        (block[f"{prefix}_active_primary"] == 1) &
        (block[f"{prefix}_active_secondary"] == 1)
    ).astype(int)

    max_age = np.zeros(len(df), dtype=int)

    for c in active_cols:
        x = act[c].to_numpy(dtype=int)
        age = np.zeros(len(x), dtype=int)
        run = 0
        for i in range(len(x)):
            if x[i] == 1:
                run += 1
            else:
                run = 0
            age[i] = run
        max_age = np.maximum(max_age, age)

    block[f"{prefix}_max_active_age"] = max_age

    return pd.concat([df, block], axis=1).copy()

# ============================================================
# MODEL PATHS
# ============================================================

def gate1_model_path(symbol: str) -> str:
    return os.path.join(GATE1_MODELS_DIR, symbol, "gate1", "gate1_impulse_abs_move_atr_16h.cbm")


def gate1_meta_path(symbol: str) -> str:
    return os.path.join(GATE1_MODELS_DIR, symbol, "gate1", "meta.json")


def g2_cls_up_path() -> str:
    return os.path.join(GATE2_MOD_DIR, "cls", "up_reach_high", "up_reach_high.cbm")


def g2_cls_dn_path() -> str:
    return os.path.join(GATE2_MOD_DIR, "cls", "dn_reach_high", "dn_reach_high.cbm")


def g2_cls_up_meta_path() -> str:
    return os.path.join(GATE2_MOD_DIR, "cls", "up_reach_high", "report.json")


def g2_cls_dn_meta_path() -> str:
    return os.path.join(GATE2_MOD_DIR, "cls", "dn_reach_high", "report.json")

def g3_long_model_path(symbol: str) -> str:
    return os.path.join(GATE3_SCORE_ROOT, symbol, "long", "gate3_score", "gate3_score.cbm")


def g3_short_model_path(symbol: str) -> str:
    return os.path.join(GATE3_SCORE_ROOT, symbol, "short", "gate3_score", "gate3_score.cbm")


def g3_long_meta_path(symbol: str) -> str:
    return os.path.join(GATE3_SCORE_ROOT, symbol, "long", "gate3_score", "meta.json")


def g3_short_meta_path(symbol: str) -> str:
    return os.path.join(GATE3_SCORE_ROOT, symbol, "short", "gate3_score", "meta.json")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_runtime_args()
    apply_runtime_args(args)
    refresh_output_paths()
    split_config = build_gate4_split_config()

    ensure_dir(OUT_ROOT)

    print("Gate4 Dataset Builder")
    print("OUT_ROOT:", OUT_ROOT)
    print("TRAIN_END:", TRAIN_END)
    print("VALID_START:", VALID_START)
    print("VALID_END:", VALID_END)
    print("SPLIT_CONFIG:", {k: str(v) for k, v in split_config.items()})
    print("=" * 120)

    if not os.path.exists(POLICY_CSV):
        raise SystemExit(f"not found: {POLICY_CSV}")

    required_global_models = [
        g2_cls_up_path(),
        g2_cls_dn_path(),
    ]
    for p in required_global_models:
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}")

    policy = pd.read_csv(POLICY_CSV)
    policy["symbol"] = policy["symbol"].astype(str)
    policy["gate3_enabled"] = pd.to_numeric(policy.get("gate3_enabled", 0), errors="coerce").fillna(0).astype(int)

    raw_rows = []
    final_rows = []
    audit_rows = []

    for _, prow in policy.iterrows():
        symbol = str(prow["symbol"])
        gate3_enabled = int(prow["gate3_enabled"])

        long_pattern = str(prow.get("gate3_pattern_long", "") or "")
        short_pattern = str(prow.get("gate3_pattern_short", "") or "")

        long_use_score = int(pd.to_numeric(prow.get("gate3_use_score_model_long", 0), errors="coerce") or 0)
        short_use_score = int(pd.to_numeric(prow.get("gate3_use_score_model_short", 0), errors="coerce") or 0)

        long_thr = pd.to_numeric(prow.get("gate3_score_threshold_long", np.nan), errors="coerce")
        short_thr = pd.to_numeric(prow.get("gate3_score_threshold_short", np.nan), errors="coerce")

        policy_gate3_score_long = float(pd.to_numeric(prow.get("gate3_score_long", np.nan), errors="coerce"))
        policy_gate3_score_short = float(pd.to_numeric(prow.get("gate3_score_short", np.nan), errors="coerce"))
        policy_gate3_rank_long = float(pd.to_numeric(prow.get("gate3_rank_long", np.nan), errors="coerce"))
        policy_gate3_rank_short = float(pd.to_numeric(prow.get("gate3_rank_short", np.nan), errors="coerce"))
        policy_gate3_side_bias = float(pd.to_numeric(prow.get("gate3_side_bias", np.nan), errors="coerce"))

        fp_base = os.path.join(BASE_DATA_DIR, f"{symbol}.parquet")
        fp_g3 = os.path.join(GATE3_DATA_DIR, f"{symbol}.parquet")

        audit = {
            "symbol": symbol,
            "gate3_enabled": gate3_enabled,
            "rows_total_base": 0,
            "rows_after_merge": 0,
            "rows_raw_candidates": 0,
            "rows_labeled": 0,
            "split_mode": str(split_config.get("mode")),
            "train_end": "" if split_config.get("train_end") is None else str(split_config.get("train_end")),
            "train_safe_cutoff": "" if split_config.get("train_safe_cutoff") is None else str(split_config.get("train_safe_cutoff")),
            "valid_start": "" if split_config.get("valid_start") is None else str(split_config.get("valid_start")),
            "valid_end": "" if split_config.get("valid_end") is None else str(split_config.get("valid_end")),
            "status": "init",
        }

        if not os.path.exists(fp_base):
            audit["status"] = "missing_base_dataset"
            audit_rows.append(audit)
            continue


        if not os.path.exists(gate1_model_path(symbol)):
            audit["status"] = "missing_gate1_model"
            audit_rows.append(audit)
            continue

        long_model_exists = os.path.exists(g3_long_model_path(symbol))
        short_model_exists = os.path.exists(g3_short_model_path(symbol))
        gate3_dataset_exists = os.path.exists(fp_g3)

        has_gate3_long_bundle = (
                gate3_enabled == 1
                and gate3_dataset_exists
                and long_use_score == 1
                and np.isfinite(long_thr)
                and long_model_exists
        )

        has_gate3_short_bundle = (
                gate3_enabled == 1
                and gate3_dataset_exists
                and short_use_score == 1
                and np.isfinite(short_thr)
                and short_model_exists
        )

        has_any_gate3_bundle = has_gate3_long_bundle or has_gate3_short_bundle
        has_full_gate3_bundle = has_gate3_long_bundle and has_gate3_short_bundle

        base_df = pd.read_parquet(fp_base)
        if "entry_ts" not in base_df.columns:
            audit["status"] = "missing_entry_ts_base"
            audit_rows.append(audit)
            continue

        base_df["entry_ts"] = to_naive_utc_auto(base_df["entry_ts"])
        base_df = (
            base_df.dropna(subset=["entry_ts"])
            .sort_values("entry_ts")
            .drop_duplicates(subset=["entry_ts"], keep="last")
            .reset_index(drop=True)
        )

        audit["rows_total_base"] = int(len(base_df))

        base_cols = build_base_feature_cols(base_df)
        if not base_cols:
            audit["status"] = "empty_base_whitelist"
            audit_rows.append(audit)
            continue

        work = base_df[["entry_ts"] + base_cols].copy().rename(columns={"entry_ts": "ts"})
        work["symbol"] = symbol

        long_active_cols: List[str] = []
        short_active_cols: List[str] = []

        if gate3_enabled == 1 and gate3_dataset_exists:
            g3_df = pd.read_parquet(fp_g3)

            if "entry_ts" not in g3_df.columns:
                audit["status"] = "missing_entry_ts_gate3"
                audit_rows.append(audit)
                continue

            g3_df["entry_ts"] = to_naive_utc_auto(g3_df["entry_ts"])
            g3_df = (
                g3_df.dropna(subset=["entry_ts"])
                .sort_values("entry_ts")
                .drop_duplicates(subset=["entry_ts"], keep="last")
                .reset_index(drop=True)
            )

            all_active_cols = sorted([c for c in g3_df.columns if c.startswith("active_pa_")])
            long_active_cols = select_side_pattern_cols(all_active_cols, "long")
            short_active_cols = select_side_pattern_cols(all_active_cols, "short")

            g3_keep = ["entry_ts"]

            if long_pattern and long_pattern in g3_df.columns:
                g3_keep.append(long_pattern)
            if short_pattern and short_pattern in g3_df.columns and short_pattern not in g3_keep:
                g3_keep.append(short_pattern)

            for c in long_active_cols + short_active_cols:
                if c not in g3_keep:
                    g3_keep.append(c)

            g3m = g3_df[g3_keep].copy().rename(columns={"entry_ts": "ts"})
            work = work.merge(g3m, on="ts", how="left")

        audit["rows_after_merge"] = int(len(work))
        work["gate3_threshold_long"] = float(long_thr) if np.isfinite(long_thr) else np.nan
        work["gate3_threshold_short"] = float(short_thr) if np.isfinite(short_thr) else np.nan
        work["gate3_score_long"] = policy_gate3_score_long
        work["gate3_score_short"] = policy_gate3_score_short
        work["gate3_rank_long"] = policy_gate3_rank_long
        work["gate3_rank_short"] = policy_gate3_rank_short
        work["gate3_side_bias"] = policy_gate3_side_bias

        if has_gate3_long_bundle:
            g3_long_meta = extract_gate3_meta_features(
                meta=load_json_safe(g3_long_meta_path(symbol)),
                fallback_thr=float(long_thr),
            )
        else:
            g3_long_meta = {
                "threshold": np.nan,
                "precision_meta": np.nan,
                "wilson_meta": np.nan,
                "delta_wilson_meta": np.nan,
                "pvalue_meta": np.nan,
                "kept_n_meta": np.nan,
                "valid_pos_rate_meta": np.nan,
                "thr_kept_lift_meta": np.nan,
            }

        if has_gate3_short_bundle:
            g3_short_meta = extract_gate3_meta_features(
                meta=load_json_safe(g3_short_meta_path(symbol)),
                fallback_thr=float(short_thr),
            )
        else:
            g3_short_meta = {
                "threshold": np.nan,
                "precision_meta": np.nan,
                "wilson_meta": np.nan,
                "delta_wilson_meta": np.nan,
                "pvalue_meta": np.nan,
                "kept_n_meta": np.nan,
                "valid_pos_rate_meta": np.nan,
                "thr_kept_lift_meta": np.nan,
            }

        for c in work.columns:
            if c in {"ts", "symbol"}:
                continue
            work[c] = pd.to_numeric(work[c], errors="coerce")

        work = add_active_set_features(work, long_active_cols, prefix="g3_long")
        work = add_active_set_features(work, short_active_cols, prefix="g3_short")

        g3_long_active = safe_bool_series(
            work[long_pattern]) if long_pattern and long_pattern in work.columns else pd.Series(0, index=work.index,
                                                                                                dtype=int)
        g3_short_active = safe_bool_series(
            work[short_pattern]) if short_pattern and short_pattern in work.columns else pd.Series(0, index=work.index,
                                                                                                   dtype=int)

        g3_block = pd.DataFrame({
            "g3_long_active": g3_long_active.astype(int),
            "g3_short_active": g3_short_active.astype(int),
        }, index=work.index)

        g3_block["g3_any_active"] = ((g3_block["g3_long_active"] == 1) | (g3_block["g3_short_active"] == 1)).astype(int)
        g3_block["g3_both_active"] = ((g3_block["g3_long_active"] == 1) & (g3_block["g3_short_active"] == 1)).astype(int)

        g3_block["gate3_active_count"] = (
            pd.to_numeric(work["g3_long_active_count"], errors="coerce").fillna(0).astype(int)
            + pd.to_numeric(work["g3_short_active_count"], errors="coerce").fillna(0).astype(int)
        )

        g3_block["gate3_active_primary"] = (
            (pd.to_numeric(work["g3_long_active_primary"], errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(work["g3_short_active_primary"], errors="coerce").fillna(0) > 0)
        ).astype(int)

        g3_block["gate3_active_secondary"] = (
            (pd.to_numeric(work["g3_long_active_secondary"], errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(work["g3_short_active_secondary"], errors="coerce").fillna(0) > 0)
        ).astype(int)

        g3_block["gate3_active_overlap_primary_secondary"] = (
            (pd.to_numeric(work["g3_long_active_overlap_primary_secondary"], errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(work["g3_short_active_overlap_primary_secondary"], errors="coerce").fillna(0) > 0)
        ).astype(int)

        g3_block["gate3_max_active_age"] = np.maximum(
            pd.to_numeric(work["g3_long_max_active_age"], errors="coerce").fillna(0),
            pd.to_numeric(work["g3_short_max_active_age"], errors="coerce").fillna(0),
        )

        work = pd.concat([work, g3_block], axis=1).copy()
        # ------------------------------------------------------------
        # Gate1
        # ------------------------------------------------------------
        gate1_model = CatBoostClassifier()
        gate1_model.load_model(gate1_model_path(symbol))
        gate1_feats = list(gate1_model.feature_names_ or [])

        gate1_input = prepare_model_input(
            base_df=work,
            symbol=symbol,
            required_features=gate1_feats,
            active_cols=None,
            active_prefix=None,
        )

        g1_proba, _, miss = model_predict_proba(gate1_model_path(symbol), gate1_input)
        if g1_proba is None:
            audit["status"] = f"gate1_missing_features:{','.join(miss[:5])}"
            audit_rows.append(audit)
            continue

        work["gate1_proba"] = g1_proba
        work["gate1_pass"] = (work["gate1_proba"] >= GATE1_PROBA_MIN).astype(int)
        # ------------------------------------------------------------
        # Новый Gate2 classification
        # ------------------------------------------------------------
        g2_up_cls_model = CatBoostClassifier()
        g2_up_cls_model.load_model(g2_cls_up_path())
        g2_up_cls_feats = list(g2_up_cls_model.feature_names_ or [])

        g2_up_cls_input = prepare_model_input(
            base_df=work,
            symbol=symbol,
            required_features=g2_up_cls_feats,
            active_cols=None,
            active_prefix=None,
        )

        g2_up_cls, _, miss = model_predict_proba(g2_cls_up_path(), g2_up_cls_input)
        if g2_up_cls is None:
            audit["status"] = f"g2_up_cls_missing_features:{','.join(miss[:5])}"
            audit_rows.append(audit)
            continue
        work["g2_cls_up_reach_high_proba"] = g2_up_cls

        g2_dn_cls_model = CatBoostClassifier()
        g2_dn_cls_model.load_model(g2_cls_dn_path())
        g2_dn_cls_feats = list(g2_dn_cls_model.feature_names_ or [])

        g2_dn_cls_input = prepare_model_input(
            base_df=work,
            symbol=symbol,
            required_features=g2_dn_cls_feats,
            active_cols=None,
            active_prefix=None,
        )

        g2_dn_cls, _, miss = model_predict_proba(g2_cls_dn_path(), g2_dn_cls_input)
        if g2_dn_cls is None:
            audit["status"] = f"g2_dn_cls_missing_features:{','.join(miss[:5])}"
            audit_rows.append(audit)
            continue
        work["g2_cls_dn_reach_high_proba"] = g2_dn_cls

        # derived from Gate2
        g2_block = pd.DataFrame(index=work.index)

        g2_block["g2_cls_spread"] = work["g2_cls_up_reach_high_proba"] - work["g2_cls_dn_reach_high_proba"]
        g2_block["g2_cls_abs_spread"] = np.abs(g2_block["g2_cls_spread"])
        g2_block["g2_cls_max"] = np.maximum(work["g2_cls_up_reach_high_proba"], work["g2_cls_dn_reach_high_proba"])
        g2_block["g2_up_dominant"] = (g2_block["g2_cls_spread"] > 0).astype(int)
        g2_block["g2_dn_dominant"] = (g2_block["g2_cls_spread"] < 0).astype(int)


        work = pd.concat([work, g2_block], axis=1).copy()
        work["gate2_proba"] = np.maximum(work["g2_cls_up_reach_high_proba"], work["g2_cls_dn_reach_high_proba"])
        # ------------------------------------------------------------
        # Gate3 score
        # ------------------------------------------------------------
        work["g3_long_score_proba"] = np.nan
        work["g3_short_score_proba"] = np.nan
        work["g3_long_score_pass"] = 0
        work["g3_short_score_pass"] = 0

        if has_gate3_long_bundle:
            long_model = g3_long_model_path(symbol)

            long_cbm = CatBoostClassifier()
            long_cbm.load_model(long_model)
            long_feats = list(long_cbm.feature_names_ or [])

            long_input = prepare_model_input(
                base_df=work,
                symbol=symbol,
                required_features=long_feats,
                active_cols=long_active_cols,
                active_prefix="g3_long",
            )

            long_score, _, miss = model_predict_proba(long_model, long_input)
            if long_score is None:
                audit["status"] = f"g3_long_missing_features:{','.join(miss[:5])}"
                audit_rows.append(audit)
                continue

            work["g3_long_score_proba"] = long_score
            work["g3_long_score_pass"] = (work["g3_long_score_proba"] >= float(long_thr)).astype(int)

        if has_gate3_short_bundle:
            short_model = g3_short_model_path(symbol)

            short_cbm = CatBoostClassifier()
            short_cbm.load_model(short_model)
            short_feats = list(short_cbm.feature_names_ or [])

            short_input = prepare_model_input(
                base_df=work,
                symbol=symbol,
                required_features=short_feats,
                active_cols=short_active_cols,
                active_prefix="g3_short",
            )

            short_score, _, miss = model_predict_proba(short_model, short_input)
            if short_score is None:
                audit["status"] = f"g3_short_missing_features:{','.join(miss[:5])}"
                audit_rows.append(audit)
                continue

            work["g3_short_score_proba"] = short_score
            work["g3_short_score_pass"] = (work["g3_short_score_proba"] >= float(short_thr)).astype(int)
        # derived from Gate3
        work["gate3_proba_long"] = pd.to_numeric(work["g3_long_score_proba"], errors="coerce")
        work["gate3_proba_short"] = pd.to_numeric(work["g3_short_score_proba"], errors="coerce")
        work["gate3_pass_long"] = pd.to_numeric(work["g3_long_score_pass"], errors="coerce").fillna(0).astype(int)
        work["gate3_pass_short"] = pd.to_numeric(work["g3_short_score_pass"], errors="coerce").fillna(0).astype(int)
        work["gate3_margin_long"] = work["gate3_proba_long"] - work["gate3_threshold_long"]
        work["gate3_margin_short"] = work["gate3_proba_short"] - work["gate3_threshold_short"]

        work["gate3_precision_meta_long"] = g3_long_meta["precision_meta"]
        work["gate3_wilson_meta_long"] = g3_long_meta["wilson_meta"]
        work["gate3_delta_wilson_meta_long"] = g3_long_meta["delta_wilson_meta"]
        work["gate3_pvalue_meta_long"] = g3_long_meta["pvalue_meta"]
        work["gate3_kept_n_meta_long"] = g3_long_meta["kept_n_meta"]
        work["gate3_valid_pos_rate_meta_long"] = g3_long_meta["valid_pos_rate_meta"]
        work["gate3_thr_kept_lift_meta_long"] = g3_long_meta["thr_kept_lift_meta"]

        work["gate3_precision_meta_short"] = g3_short_meta["precision_meta"]
        work["gate3_wilson_meta_short"] = g3_short_meta["wilson_meta"]
        work["gate3_delta_wilson_meta_short"] = g3_short_meta["delta_wilson_meta"]
        work["gate3_pvalue_meta_short"] = g3_short_meta["pvalue_meta"]
        work["gate3_kept_n_meta_short"] = g3_short_meta["kept_n_meta"]
        work["gate3_valid_pos_rate_meta_short"] = g3_short_meta["valid_pos_rate_meta"]
        work["gate3_thr_kept_lift_meta_short"] = g3_short_meta["thr_kept_lift_meta"]

        work["gate3_precision_meta"] = work[["gate3_precision_meta_long", "gate3_precision_meta_short"]].max(axis=1, skipna=True)
        work["gate3_wilson_meta"] = work[["gate3_wilson_meta_long", "gate3_wilson_meta_short"]].max(axis=1, skipna=True)
        work["gate3_delta_wilson_meta"] = work[["gate3_delta_wilson_meta_long", "gate3_delta_wilson_meta_short"]].max(axis=1, skipna=True)
        work["gate3_pvalue_meta"] = work[["gate3_pvalue_meta_long", "gate3_pvalue_meta_short"]].min(axis=1, skipna=True)
        work["gate3_kept_n_meta"] = work[["gate3_kept_n_meta_long", "gate3_kept_n_meta_short"]].max(axis=1, skipna=True)
        work["gate3_valid_pos_rate_meta"] = work[["gate3_valid_pos_rate_meta_long", "gate3_valid_pos_rate_meta_short"]].max(axis=1, skipna=True)
        work["gate3_thr_kept_lift_meta"] = work[["gate3_thr_kept_lift_meta_long", "gate3_thr_kept_lift_meta_short"]].max(axis=1, skipna=True)
        g3_score_short = pd.to_numeric(work["g3_short_score_proba"], errors="coerce").fillna(0.0)
        work["has_gate3_long_bundle"] = np.full(len(work), int(has_gate3_long_bundle), dtype=np.int8)
        work["has_gate3_short_bundle"] = np.full(len(work), int(has_gate3_short_bundle), dtype=np.int8)
        work["has_any_gate3_bundle"] = np.full(len(work), int(has_any_gate3_bundle), dtype=np.int8)
        work["has_full_gate3_bundle"] = np.full(len(work), int(has_full_gate3_bundle), dtype=np.int8)

        g3_score_long_raw = pd.to_numeric(work["g3_long_score_proba"], errors="coerce")
        g3_score_short_raw = pd.to_numeric(work["g3_short_score_proba"], errors="coerce")

        g3_score_long_filled = g3_score_long_raw.fillna(0.0)
        g3_score_short_filled = g3_score_short_raw.fillna(0.0)

        g3_score_block = pd.DataFrame(index=work.index)
        g3_score_block["g3_score_spread"] = np.where(
            work["has_any_gate3_bundle"] == 1,
            g3_score_long_filled - g3_score_short_filled,
            np.nan,
        )
        g3_score_block["g3_score_abs_spread"] = np.abs(g3_score_block["g3_score_spread"])
        g3_score_block["g3_score_max"] = np.where(
            work["has_any_gate3_bundle"] == 1,
            np.maximum(g3_score_long_filled, g3_score_short_filled),
            np.nan,
        )

        work = pd.concat([work, g3_score_block], axis=1).copy()

        meta_block = pd.DataFrame(index=work.index)

        meta_block["g2_side_sign"] = np.where(
            work["g2_cls_spread"] > 0,
            1,
            np.where(work["g2_cls_spread"] < 0, -1, 0),
        )

        meta_block["g3_side_sign"] = np.where(
            work["g3_score_spread"] > 0,
            1,
            np.where(work["g3_score_spread"] < 0, -1, 0),
        )

        meta_block["g2_g3_side_agree"] = (
                (work["has_any_gate3_bundle"] == 1) &
                (meta_block["g2_side_sign"] != 0) &
                (meta_block["g3_side_sign"] != 0) &
                (meta_block["g2_side_sign"] == meta_block["g3_side_sign"])
        ).astype(int)

        meta_block["g2_g3_side_conflict"] = (
                (work["has_any_gate3_bundle"] == 1) &
                (meta_block["g2_side_sign"] != 0) &
                (meta_block["g3_side_sign"] != 0) &
                (meta_block["g2_side_sign"] != meta_block["g3_side_sign"])
        ).astype(int)

        meta_block["g1_g2_strength"] = work["gate1_proba"] * work["g2_cls_max"]
        meta_block["g1_g3_strength"] = np.where(
            work["has_any_gate3_bundle"] == 1,
            work["gate1_proba"] * work["g3_score_max"],
            np.nan,
        )

        meta_block["g2g3_joint_long"] = np.where(
            work["has_gate3_long_bundle"] == 1,
            work["g2_cls_up_reach_high_proba"] *
            pd.to_numeric(work["g3_long_score_proba"], errors="coerce").fillna(0.0),
            np.nan,
        )

        meta_block["g2g3_joint_short"] = np.where(
            work["has_gate3_short_bundle"] == 1,
            work["g2_cls_dn_reach_high_proba"] *
            pd.to_numeric(work["g3_short_score_proba"], errors="coerce").fillna(0.0),
            np.nan,
        )

        meta_block["g2g3_joint_long_minus_short"] = (
            meta_block["g2g3_joint_long"] - meta_block["g2g3_joint_short"]
        )

        meta_block["g2g3_joint_abs_spread"] = meta_block["g2g3_joint_long_minus_short"].abs()

        work = pd.concat([work, meta_block], axis=1).copy()

        # ------------------------------------------------------------
        # ЛОГИКА КАНДИДАТОВ ДЛЯ GATE4
        # ------------------------------------------------------------

        gate1_ok = work["gate1_pass"] == 1 if REQUIRE_GATE1_PASS else pd.Series(True, index=work.index)

        # базовый проход
        long_g3_score = pd.to_numeric(work["g3_long_score_proba"], errors="coerce").fillna(0.0)
        short_g3_score = pd.to_numeric(work["g3_short_score_proba"], errors="coerce").fillna(0.0)

        long_g3_extreme = (
                (work["g3_long_active"] == 1)
                & (long_g3_score >= G3_SCORE_EXTREME_MIN)
        )

        short_g3_extreme = (
                (work["g3_short_active"] == 1)
                & (short_g3_score >= G3_SCORE_EXTREME_MIN)
        )

        cand_block = pd.DataFrame(index=work.index)

        cand_block["base_long_candidate"] = (
                gate1_ok
                & (
                        (work["g2_cls_up_reach_high_proba"] >= G2_CLS_BASE_MIN)
                        | (
                                (work["has_gate3_long_bundle"] == 1)
                                & (
                                        (work["g3_long_active"] == 1)
                                        | (work["g3_long_score_pass"] == 1)
                                )
                        )
                )
        ).astype(int)

        cand_block["base_short_candidate"] = (
                gate1_ok
                & (
                        (work["g2_cls_dn_reach_high_proba"] >= G2_CLS_BASE_MIN)
                        | (
                                (work["has_gate3_short_bundle"] == 1)
                                & (
                                        (work["g3_short_active"] == 1)
                                        | (work["g3_short_score_pass"] == 1)
                                )
                        )
                )
        ).astype(int)

        cand_block["extreme_long_candidate"] = (
                gate1_ok
                & (
                        (work["g2_cls_up_reach_high_proba"] >= G2_CLS_EXTREME_MIN)
                        | (
                                (work["has_gate3_long_bundle"] == 1)
                                & long_g3_extreme
                        )
                )
        ).astype(int)

        cand_block["extreme_short_candidate"] = (
                gate1_ok
                & (
                        (work["g2_cls_dn_reach_high_proba"] >= G2_CLS_EXTREME_MIN)
                        | (
                                (work["has_gate3_short_bundle"] == 1)
                                & short_g3_extreme
                        )
                )
        ).astype(int)

        cand_block["pass_long"] = (
                    (cand_block["base_long_candidate"] == 1) | (cand_block["extreme_long_candidate"] == 1)).astype(int)
        cand_block["pass_short"] = (
                    (cand_block["base_short_candidate"] == 1) | (cand_block["extreme_short_candidate"] == 1)).astype(
            int)
        cand_block["pass_any"] = ((cand_block["pass_long"] == 1) | (cand_block["pass_short"] == 1)).astype(int)
        cand_block["pass_both"] = ((cand_block["pass_long"] == 1) & (cand_block["pass_short"] == 1)).astype(int)
        cand_block["pass_long_only"] = ((cand_block["pass_long"] == 1) & (cand_block["pass_short"] == 0)).astype(int)
        cand_block["pass_short_only"] = ((cand_block["pass_short"] == 1) & (cand_block["pass_long"] == 0)).astype(int)

        work = pd.concat([work, cand_block], axis=1).copy()

        # ------------------------------------------------------------
        # TARGET FOR GATE4
        # ------------------------------------------------------------
        labels = compute_future_direction_labels(
            df=base_df.rename(columns={"entry_ts": "ts"}),
            ttl_bars=TTL_BARS,
            dir_margin_atr=DIR_MARGIN_ATR,
            first_hit_atr=DIR_FIRST_HIT_ATR,
            label_mode=DIR_LABEL_MODE,
        )

        work = work.merge(
            labels[[
                "ts",
                "mfe_up_atr_16h",
                "mfe_dn_atr_16h",
                "first_up_hit_bar",
                "first_dn_hit_bar",
                "y_dir_mfe",
                "y_dir_first",
                "y_dir",
                "y_dir_int",
            ]],
            on="ts",
            how="left",
        )

        work = build_side_clean_labels(
            df=work,
            deltas=ALL_DELTAS,
        )

        legacy_valid_start_ts = resolve_validation_start_ts(
            ts_series=work["ts"],
            meta_paths=[
                gate1_meta_path(symbol),
                g2_cls_up_meta_path(),
                g2_cls_dn_meta_path(),
                g3_long_meta_path(symbol),
                g3_short_meta_path(symbol),
            ],
            fallback_tail_share=UPSTREAM_VALID_TAIL_SHARE,
        )

        work = apply_gate4_split(
            df=work,
            split_config=split_config,
            legacy_valid_start_ts=legacy_valid_start_ts,
        )

        gate4_keep_cols = [
            "ts",
            "symbol",
            *base_cols,

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
                        "g3_long_any_active",
            "g3_long_active_count",
            "g3_long_active_primary",
            "g3_long_active_secondary",
            "g3_long_active_overlap_primary_secondary",
            "g3_long_max_active_age",
            "g3_short_any_active",
            "g3_short_active_count",
            "g3_short_active_primary",
            "g3_short_active_secondary",
            "g3_short_active_overlap_primary_secondary",
            "g3_short_max_active_age",

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

            "has_gate3_long_bundle",
            "has_gate3_short_bundle",
            "has_any_gate3_bundle",
            "has_full_gate3_bundle",

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

            "mfe_up_atr_16h",
            "mfe_dn_atr_16h",
            "first_up_hit_bar",
            "first_dn_hit_bar",
            "y_dir_mfe",
            "y_dir_first",
            "y_dir",
            "y_dir_int",

            "y_side_clean",
            "y_side_clean_int",
            "edge_atr_clean",
            "abs_edge_atr_clean",
            "is_side_clean_long",
            "is_side_clean_short",
            "is_side_clean_ambig",

            "edge_delta_050",
            "abs_edge_delta_050",
            "y_side_clean_delta_050",
            "y_side_clean_delta_050_int",
            "is_long_delta_050",
            "is_short_delta_050",
            "is_ambig_delta_050",

            "edge_delta_060",
            "abs_edge_delta_060",
            "y_side_clean_delta_060",
            "y_side_clean_delta_060_int",
            "is_long_delta_060",
            "is_short_delta_060",
            "is_ambig_delta_060",

            "edge_delta_075",
            "abs_edge_delta_075",
            "y_side_clean_delta_075",
            "y_side_clean_delta_075_int",
            "is_long_delta_075",
            "is_short_delta_075",
            "is_ambig_delta_075",

            "upstream_train_end_ts",
            "upstream_train_safe_cutoff_ts",
            "upstream_valid_start_ts",
            "upstream_valid_end_ts",
            "upstream_split",
            "upstream_is_oos",
        ]

        gate4_keep_cols = [c for c in gate4_keep_cols if c in work.columns]
        work = work.loc[:, gate4_keep_cols].copy()

        cand_raw = work[work["pass_any"] == 1].copy()

        audit["rows_raw_candidates"] = int(len(cand_raw))
        audit["train_rows"] = int((work["upstream_split"] == "train").sum()) if "upstream_split" in work.columns else 0
        audit["valid_rows"] = int((work["upstream_split"] == "valid").sum()) if "upstream_split" in work.columns else 0
        audit["gap_rows"] = int((work["upstream_split"] == "gap").sum()) if "upstream_split" in work.columns else 0

        if len(cand_raw) == 0:
            audit["status"] = "no_candidates"
            audit_rows.append(audit)
            continue

        cand_final = cand_raw[cand_raw["y_side_clean"].isin(["LONG", "SHORT", "AMBIG"])].copy()
        if len(cand_final):
            cand_final["y_dir_int"] = pd.to_numeric(cand_final["y_dir_int"], errors="coerce")
            cand_final["y_side_clean_int"] = pd.to_numeric(cand_final["y_side_clean_int"], errors="coerce")

        audit["rows_labeled"] = int(len(cand_final))
        audit["status"] = "ok"

        raw_rows.append(cand_raw)
        final_rows.append(cand_final)
        audit_rows.append(audit)

    raw_df = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    dataset_df = pd.concat(final_rows, ignore_index=True) if final_rows else pd.DataFrame()
    audit_df = pd.DataFrame(audit_rows)

    if len(raw_df):
        raw_df = raw_df.sort_values(["ts", "symbol"]).reset_index(drop=True)

    if len(dataset_df):
        dataset_df = dataset_df.sort_values(["ts", "symbol"]).reset_index(drop=True)

    if len(audit_df):
        audit_df = audit_df.sort_values(["status", "rows_labeled", "symbol"], ascending=[True, False, True]).reset_index(drop=True)

    raw_df.to_parquet(OUT_RAW_PARQUET, index=False)
    dataset_df.to_parquet(OUT_DATASET_PARQUET, index=False)
    audit_df.to_csv(OUT_AUDIT_CSV, index=False)

    report = {
        "rows_raw_candidates": int(len(raw_df)),
        "rows_final_labeled": int(len(dataset_df)),
        "symbols_total_in_policy": int(len(policy)),
        "symbols_ok": int((audit_df["status"] == "ok").sum()) if len(audit_df) else 0,
        "ttl_bars": int(TTL_BARS),
        "dir_margin_atr": float(DIR_MARGIN_ATR),
        "dir_first_hit_atr": float(DIR_FIRST_HIT_ATR),
        "dir_label_mode": str(DIR_LABEL_MODE),
        "gate1_proba_min": float(GATE1_PROBA_MIN),
        "g2_cls_base_min": float(G2_CLS_BASE_MIN),
        "g2_cls_extreme_min": float(G2_CLS_EXTREME_MIN),
        "g3_score_extreme_min": float(G3_SCORE_EXTREME_MIN),
        "require_gate1_pass": bool(REQUIRE_GATE1_PASS),
        "gate3_bundle_logic": "independent_per_side",
        "split": {
            "mode": str(split_config.get("mode")),
            "train_end": "" if split_config.get("train_end") is None else str(split_config.get("train_end")),
            "train_safe_cutoff": "" if split_config.get("train_safe_cutoff") is None else str(split_config.get("train_safe_cutoff")),
            "valid_start": "" if split_config.get("valid_start") is None else str(split_config.get("valid_start")),
            "valid_end": "" if split_config.get("valid_end") is None else str(split_config.get("valid_end")),
            "train_rule": "ts < train_end - TTL_BARS * 4h" if split_config.get("mode") == "fixed_time_train_safe" else "legacy meta/fallback split",
            "valid_rule": "valid_start <= ts < valid_end" if split_config.get("mode") == "fixed_time_train_safe" else "ts >= upstream_valid_start_ts",
        },
        "base_context_cols_count": int(len(BASE_CONTEXT_COLS)),
        "primary_target": "y_side_clean_delta_050",
        "aux_target_columns": [
            "y_side_clean_delta_060",
            "y_side_clean_delta_075",
            "y_dir",
        ],
        "primary_delta": float(PRIMARY_DELTA),
        "aux_deltas": [float(x) for x in AUX_DELTAS],
        "files": {
            "raw_candidates": OUT_RAW_PARQUET,
            "dataset": OUT_DATASET_PARQUET,
            "audit": OUT_AUDIT_CSV,
        },
    }

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("WROTE", OUT_RAW_PARQUET)
    print("WROTE", OUT_DATASET_PARQUET)
    print("WROTE", OUT_AUDIT_CSV)
    print("WROTE", OUT_REPORT_JSON)
    print()

    if len(dataset_df):
        print("DATASET SHAPE")
        print(dataset_df.shape)
        print()

        print("TARGET DISTRIBUTION: y_side_clean")
        print(dataset_df["y_side_clean"].value_counts(dropna=False).to_string())
        print()

        print("TARGET DISTRIBUTION: y_dir")
        print(dataset_df["y_dir"].value_counts(dropna=False).to_string())
        print()

        print("UPSTREAM SPLIT DISTRIBUTION")
        if "upstream_split" in dataset_df.columns:
            print(dataset_df["upstream_split"].value_counts(dropna=False).to_string())
        else:
            print("no upstream_split")
    else:
        print("Final Gate4 dataset is empty")


if __name__ == "__main__":
    main()