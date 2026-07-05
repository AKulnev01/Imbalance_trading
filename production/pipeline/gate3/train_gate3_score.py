import os
import json
import argparse
from typing import Dict, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import binomtest
# ============================================================
# PATHS
# ============================================================

#python production/pipeline/train_gate3_score.py

H4_DIR = os.environ.get("IMB_GATE3_H4_DIR", "data/h4_3")
BASE_DATA_DIR = os.environ.get("IMB_GATE3_BASE_DATA_DIR", "production/dataset/gate1")

GATE3_DATA_DIR = os.environ.get("IMB_GATE3_DATA_DIR", "production/dataset/pa_gate3_v3_long_short_by_symbol")
GATE3_AUDIT_CSV = os.environ.get("IMB_GATE3_AUDIT_CSV", "production/dataset/_AUDIT_GATE3_PA_V3_LONG_SHORT.csv")

GATE1_MODELS_DIR = os.environ.get("IMB_GATE3_GATE1_MODELS_DIR", "production/models/final_gate1")

POLICY_CSV = os.environ.get("IMB_GATE3_POLICY_CSV", "production/models/ks/gate3_symbol_policy.csv")

OUT_ROOT = os.environ.get("IMB_GATE3_OUT_ROOT", "production/models/final_gate3_score_long_short")
OUT_MANIFEST_CSV = os.path.join(OUT_ROOT, "_MANIFEST.csv")
OUT_REPORT_JSON = os.path.join(OUT_ROOT, "_REPORT.json")
OUT_AUDIT_CSV = os.path.join(OUT_ROOT, "_AUDIT_TRAINSETS.csv")
# ============================================================
# TARGET / HORIZON
# ============================================================

TTL_BARS = 4
TARGET_MFE_ATR = 0.8
TARGET_MFE_DN_ATR = 0.8

SIDES = ("long", "short")


# ============================================================
# ENTRY / COST ASSUMPTIONS
# ============================================================

SLIPPAGE_BPS = 20.0
FEE_BPS = 10.0

# ============================================================
# GATE FILTERS
# ============================================================

GATE1_PROBA_MIN = 0.50

# ============================================================
# TRAINING CONFIG
# ============================================================

VALID_SHARE = 0.20
MIN_EDGE_THRESHOLD = 0.3  # например

MIN_ROWS_TOTAL = 160
MIN_TRAIN_ROWS = 100
MIN_VALID_ROWS = 40
MIN_SYMBOL_PA_VALID_RATE = 0.95

THR_GRID = np.round(np.arange(0.50, 0.91, 0.02), 2)

CB_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 1200,
    "depth": 6,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}

# ============================================================
# HELPERS
# ============================================================

SPLIT_ENV_TRAIN_END = "IMB_OFFLINE_TRAIN_END"
SPLIT_ENV_VALID_START = "IMB_OFFLINE_VALID_START"
SPLIT_ENV_VALID_END = "IMB_OFFLINE_VALID_END"
SPLIT_ENV_SYMBOLS = "IMB_OFFLINE_SYMBOLS"


def parse_optional_split_ts(raw: Optional[str], name: str) -> Optional[pd.Timestamp]:
    text = str(raw or "").strip()

    if not text:
        return None

    ts = pd.to_datetime(text, utc=True, errors="coerce")

    if pd.isna(ts):
        raise SystemExit("bad {} value: {}".format(name, raw))

    out = pd.Timestamp(ts)

    if out.tzinfo is not None:
        out = out.tz_convert("UTC").tz_localize(None)

    return pd.Timestamp(out).floor("min")


def parse_symbol_filter(raw: Optional[str]) -> Optional[set[str]]:
    text = str(raw or "").strip()

    if not text:
        return None

    out = set()

    for part in text.replace(";", ",").replace(" ", ",").split(","):
        symbol = part.strip().upper()

        if not symbol:
            continue

        if not symbol.endswith("USDT"):
            symbol = symbol + "USDT"

        out.add(symbol)

    return out if out else None


def parse_gate3_split_args() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-end", default="")
    parser.add_argument("--valid-start", default="")
    parser.add_argument("--valid-end", default="")
    parser.add_argument("--symbols", default="")
    args, _ = parser.parse_known_args()

    train_end = parse_optional_split_ts(
        args.train_end or os.environ.get(SPLIT_ENV_TRAIN_END, ""),
        "--train-end",
    )
    valid_start = parse_optional_split_ts(
        args.valid_start or os.environ.get(SPLIT_ENV_VALID_START, ""),
        "--valid-start",
    )
    valid_end = parse_optional_split_ts(
        args.valid_end or os.environ.get(SPLIT_ENV_VALID_END, ""),
        "--valid-end",
    )

    symbols = parse_symbol_filter(
        args.symbols or os.environ.get(SPLIT_ENV_SYMBOLS, "")
    )

    provided = [train_end is not None, valid_start is not None, valid_end is not None]

    if any(provided) and not all(provided):
        raise SystemExit(
            "split args must be provided together: --train-end --valid-start --valid-end"
        )

    if train_end is None:
        return {
            "mode": "legacy_tail_share",
            "train_end": None,
            "valid_start": None,
            "valid_end": None,
            "symbols": symbols,
            "valid_share": float(VALID_SHARE),
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

    return {
        "mode": "fixed_time",
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "symbols": symbols,
        "valid_share": None,
    }


def gate3_split_config_for_json(split_config: dict) -> dict:
    symbols = split_config.get("symbols")

    return {
        "mode": str(split_config.get("mode") or ""),
        "train_end": None if split_config.get("train_end") is None else str(split_config.get("train_end")),
        "valid_start": None if split_config.get("valid_start") is None else str(split_config.get("valid_start")),
        "valid_end": None if split_config.get("valid_end") is None else str(split_config.get("valid_end")),
        "symbols": None if symbols is None else sorted(list(symbols)),
        "valid_share": split_config.get("valid_share"),
    }


def split_gate3_train_valid(candidates: pd.DataFrame, split_config: dict) -> dict:
    x = candidates.copy()
    x["ts"] = pd.to_datetime(x["ts"], errors="coerce")
    x = x.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    mode = str(split_config.get("mode") or "legacy_tail_share")

    if mode == "fixed_time":
        train_end = pd.Timestamp(split_config["train_end"])
        valid_start = pd.Timestamp(split_config["valid_start"])
        valid_end = pd.Timestamp(split_config["valid_end"])

        gap_bars = int(TTL_BARS) + 1
        gap_delta = pd.Timedelta(hours=4 * gap_bars)
        train_cutoff = train_end - gap_delta

        train_df = x[x["ts"] < train_cutoff].copy()
        valid_df = x[(x["ts"] >= valid_start) & (x["ts"] < valid_end)].copy()

        return {
            "split_source": "fixed_time",
            "rows_total": int(len(x)),
            "rows_train": int(len(train_df)),
            "rows_valid": int(len(valid_df)),
            "train_df": train_df,
            "valid_df": valid_df,
            "gap_bars": int(gap_bars),
            "gap_delta_hours": float(gap_delta.total_seconds() / 3600.0),
            "train_end": str(train_end),
            "valid_start": str(valid_start),
            "valid_end": str(valid_end),
            "train_cutoff": str(train_cutoff),
            "train_min_ts": None if train_df.empty else str(train_df["ts"].min()),
            "train_max_ts": None if train_df.empty else str(train_df["ts"].max()),
            "valid_min_ts": None if valid_df.empty else str(valid_df["ts"].min()),
            "valid_max_ts": None if valid_df.empty else str(valid_df["ts"].max()),
            "train_condition": "ts < train_end - gap_delta",
            "valid_condition": "valid_start <= ts < valid_end",
        }

    n_total = len(x)
    n_valid_plan = max(MIN_VALID_ROWS, int(round(n_total * VALID_SHARE)))
    n_valid_plan = min(n_valid_plan, max(0, n_total - MIN_TRAIN_ROWS))
    n_train_plan = n_total - n_valid_plan

    if n_train_plan < MIN_TRAIN_ROWS and n_total >= (MIN_TRAIN_ROWS + MIN_VALID_ROWS):
        n_train_plan = MIN_TRAIN_ROWS
        n_valid_plan = n_total - n_train_plan

    gap_bars = int(TTL_BARS) + 1

    train_df = x.iloc[:max(0, n_train_plan - gap_bars)].copy()
    valid_df = x.iloc[n_train_plan:].copy()

    split_ts = None
    if len(valid_df):
        split_ts = valid_df["ts"].min()

    return {
        "split_source": "legacy_tail_share",
        "rows_total": int(len(x)),
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
        "train_df": train_df,
        "valid_df": valid_df,
        "gap_bars": int(gap_bars),
        "gap_delta_hours": float(4 * gap_bars),
        "train_end": None if split_ts is None else str(split_ts),
        "valid_start": None if split_ts is None else str(split_ts),
        "valid_end": None,
        "train_cutoff": None,
        "train_min_ts": None if train_df.empty else str(train_df["ts"].min()),
        "train_max_ts": None if train_df.empty else str(train_df["ts"].max()),
        "valid_min_ts": None if valid_df.empty else str(valid_df["ts"].min()),
        "valid_max_ts": None if valid_df.empty else str(valid_df["ts"].max()),
        "train_condition": "legacy: candidates.iloc[:n_train - gap]",
        "valid_condition": "legacy: candidates.iloc[n_train:]",
    }



def normalize_side_name(side: str) -> str:
    side = str(side).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError(f"unsupported side: {side}")
    return side


def select_side_pattern_cols(cols: list[str], side: str) -> list[str]:
    side = normalize_side_name(side)

    out = []
    for c in cols:
        c_low = str(c).lower()

        is_long = (
            ("_up" in c_low)
            or ("buy" in c_low)
            or ("_lo_" in c_low)
        )
        is_short = (
            ("_dn" in c_low)
            or ("sell" in c_low)
            or ("_hi_" in c_low)
        )

        if side == "long" and is_long and not is_short:
            out.append(c)
        if side == "short" and is_short and not is_long:
            out.append(c)

    return sorted(set(out))


def load_quality_map(path: str) -> Dict[str, float]:
    if not path:
        return {}

    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)

    if "symbol" not in df.columns or "pa_valid_rate" not in df.columns:
        print(
            "GATE3_QUALITY_CSV_SKIP_BAD_FORMAT: {} | columns={}".format(
                path,
                sorted(list(df.columns)),
            )
        )
        return {}

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["pa_valid_rate"] = pd.to_numeric(df["pa_valid_rate"], errors="coerce")
    df = df.dropna(subset=["symbol", "pa_valid_rate"])

    return dict(zip(df["symbol"], df["pa_valid_rate"]))


def ensure_dt_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def find_first_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).gt(0.5).astype(int)


def load_h4(symbol: str) -> Optional[pd.DataFrame]:
    fp = os.path.join(H4_DIR, f"{symbol}.parquet")
    if not os.path.exists(fp):
        return None

    df = pd.read_parquet(fp)
    tcol = find_first_col(df.columns, ["ts", "open_time", "time", "timestamp", "datetime"])
    if tcol is None:
        return None

    df[tcol] = ensure_dt_utc(df[tcol])
    df = (
        df.dropna(subset=[tcol])
          .sort_values(tcol)
          .rename(columns={tcol: "ts"})
          .drop_duplicates(subset=["ts"], keep="last")
          .reset_index(drop=True)
    )

    need = ["open", "high", "low", "close"]
    if any(c not in df.columns for c in need):
        return None

    return df[["ts", "open", "high", "low", "close"]].copy()


def ema_np(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(dtype=float)


def atr14_from_hlc(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).rolling(period).mean().to_numpy(dtype=float)


def roll_max_prev_np(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).shift(1).rolling(w, min_periods=w).max().to_numpy(dtype=float)


def roll_min_prev_np(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).shift(1).rolling(w, min_periods=w).min().to_numpy(dtype=float)


def load_gate1_model_path(symbol: str) -> Optional[str]:
    p = os.path.join(GATE1_MODELS_DIR, symbol, "gate1", "gate1_impulse_abs_move_atr_16h.cbm")
    return p if os.path.exists(p) else None

def load_base_dataset(symbol: str) -> Optional[pd.DataFrame]:
    fp = os.path.join(BASE_DATA_DIR, f"{symbol}.parquet")
    if not os.path.exists(fp):
        return None

    df = pd.read_parquet(fp)
    if "entry_ts" not in df.columns:
        return None

    df["entry_ts"] = ensure_dt_utc(df["entry_ts"])
    df = (
        df.dropna(subset=["entry_ts"])
          .sort_values("entry_ts")
          .drop_duplicates(subset=["entry_ts"], keep="last")
          .reset_index(drop=True)
    )
    return df

def to_naive_ts(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, utc=True, errors="coerce")
    return pd.Series(x).dt.tz_localize(None)

def catboost_predict_proba_from_file(model_path: str, df: pd.DataFrame):
    model = CatBoostClassifier()
    model.load_model(model_path)
    feats = list(model.feature_names_)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        return None, feats, missing

    X = df[feats].replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    proba = model.predict_proba(X)[:, 1]
    return proba, feats, []

def get_gate3_available_symbols(data_dir: str) -> set[str]:
    if not os.path.exists(data_dir):
        return set()

    out = set()
    for name in os.listdir(data_dir):
        if not name.endswith(".parquet"):
            continue
        if name.startswith("_"):
            continue
        symbol = name[:-8]
        if symbol:
            out.add(symbol)
    return out

def build_full_market_context(h4: pd.DataFrame) -> pd.DataFrame:
    out = h4[["ts", "open", "high", "low", "close"]].copy()

    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    atr14 = atr14_from_hlc(h, l, c, 14)
    ema20 = ema_np(c, 20)
    ema50 = ema_np(c, 50)
    ema100 = ema_np(c, 100)

    ret1 = pd.Series(c).pct_change(1).to_numpy(dtype=float)
    ret2 = pd.Series(c).pct_change(2).to_numpy(dtype=float)
    ret4 = pd.Series(c).pct_change(4).to_numpy(dtype=float)
    ret8 = pd.Series(c).pct_change(8).to_numpy(dtype=float)
    ret12 = pd.Series(c).pct_change(12).to_numpy(dtype=float)

    hl_range = h - l
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    atrp14 = atr14 / np.where(c != 0, c, np.nan)
    range_atr = hl_range / np.where(atr14 > 0, atr14, np.nan)
    body_atr = body / np.where(atr14 > 0, atr14, np.nan)

    trend_up = (ema20 > ema50).astype(int)
    trend_dn = (ema20 < ema50).astype(int)

    vol_q80 = pd.Series(atrp14).rolling(200, min_periods=50).quantile(0.80).to_numpy(dtype=float)
    vol_q20 = pd.Series(atrp14).rolling(200, min_periods=50).quantile(0.20).to_numpy(dtype=float)
    is_high_vol = ((atrp14 >= vol_q80) & np.isfinite(vol_q80)).astype(int)
    is_low_vol = ((atrp14 <= vol_q20) & np.isfinite(vol_q20)).astype(int)

    hi12_prev = roll_max_prev_np(h, 12)
    lo12_prev = roll_min_prev_np(l, 12)
    hi24_prev = roll_max_prev_np(h, 24)
    lo24_prev = roll_min_prev_np(l, 24)

    out["ctx_atr14"] = atr14
    out["ctx_atrp14"] = atrp14
    out["ctx_ema20"] = ema20
    out["ctx_ema50"] = ema50
    out["ctx_ema100"] = ema100

    out["ctx_ret1"] = ret1
    out["ctx_ret2"] = ret2
    out["ctx_ret4"] = ret4
    out["ctx_ret8"] = ret8
    out["ctx_ret12"] = ret12

    out["ctx_range"] = hl_range
    out["ctx_body"] = body
    out["ctx_upper_wick"] = upper_wick
    out["ctx_lower_wick"] = lower_wick

    out["ctx_range_atr"] = range_atr
    out["ctx_body_atr"] = body_atr

    out["ctx_trend_up"] = trend_up
    out["ctx_trend_dn"] = trend_dn
    out["ctx_high_vol"] = is_high_vol
    out["ctx_low_vol"] = is_low_vol

    out["ctx_close_above_ema20"] = (c > ema20).astype(int)
    out["ctx_close_above_ema50"] = (c > ema50).astype(int)
    out["ctx_close_above_ema100"] = (c > ema100).astype(int)

    out["ctx_hi12_prev"] = hi12_prev
    out["ctx_lo12_prev"] = lo12_prev
    out["ctx_hi24_prev"] = hi24_prev
    out["ctx_lo24_prev"] = lo24_prev

    out["ctx_regime"] = (
            out["ctx_trend_up"] * 2 +
            out["ctx_high_vol"]
    )

    return out


def add_active_set_features(df: pd.DataFrame, active_cols: list[str], side: str) -> pd.DataFrame:
    side = normalize_side_name(side)

    if not active_cols:
        df["gate3_any_active"] = 0
        df["active_type_count"] = 0
        df["gate3_active_primary"] = 0
        df["gate3_active_secondary"] = 0
        df["gate3_active_overlap_primary_secondary"] = 0
        return df

    act = df[active_cols].copy()
    for c in active_cols:
        act[c] = safe_bool_series(act[c])

    df["gate3_any_active"] = (act.sum(axis=1) > 0).astype(int)
    df["gate3_active_count"] = act.sum(axis=1).astype(int)

    if side == "long":
        primary_col = "active_pa_atr_squeeze_break_up"
        secondary_col = "active_pa_bos_up_24"
    else:
        primary_col = "active_pa_atr_squeeze_break_dn"
        secondary_col = "active_pa_bos_dn_24"

    if primary_col in act.columns:
        df["gate3_active_primary"] = act[primary_col].astype(int)
    else:
        df["gate3_active_primary"] = 0

    if secondary_col in act.columns:
        df["gate3_active_secondary"] = act[secondary_col].astype(int)
    else:
        df["gate3_active_secondary"] = 0

    df["gate3_active_overlap_primary_secondary"] = (
        (df["gate3_active_primary"] == 1) &
        (df["gate3_active_secondary"] == 1)
    ).astype(int)

    return df


def add_active_persistence_features(df: pd.DataFrame, active_cols: list[str]) -> pd.DataFrame:
    for c in active_cols:
        x = safe_bool_series(df[c]).to_numpy(dtype=int)
        age = np.full(len(x), 0, dtype=int)
        run = 0
        for i in range(len(x)):
            if x[i] == 1:
                run += 1
            else:
                run = 0
            age[i] = run

        df[f"{c}__age"] = age
        df[f"{c}__fresh"] = (age == 1).astype(int)
        df[f"{c}__mid"] = ((age >= 2) & (age <= 3)).astype(int)
        df[f"{c}__late"] = (age >= 4).astype(int)

    if active_cols:
        age_cols = [f"{c}__age" for c in active_cols]
        df["gate3_max_active_age"] = df[age_cols].max(axis=1)
    else:
        df["gate3_max_active_age"] = 0

    return df


def add_lag_features(df: pd.DataFrame, cols: list[str], lags=(1, 2)) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def wilson_lower_bound(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2*n)
    adj = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)
    return (centre - adj) / denom


def compute_future_edge_labels(
    h4: pd.DataFrame,
    ttl_bars: int,
    target_mfe_atr: float,
    target_mfe_dn_atr: float,
) -> pd.DataFrame:
    out = h4[["ts", "open", "high", "low", "close"]].copy().reset_index(drop=True)

    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    open_ = out["open"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)

    atr14 = atr14_from_hlc(high, low, close, 14)

    entry_px = np.full(len(out), np.nan, dtype=float)
    mfe_up = np.full(len(out), np.nan, dtype=float)
    mae_dn = np.full(len(out), np.nan, dtype=float)
    mfe_dn = np.full(len(out), np.nan, dtype=float)
    mae_up = np.full(len(out), np.nan, dtype=float)
    y_edge_long = np.full(len(out), np.nan, dtype=float)
    y_edge_short = np.full(len(out), np.nan, dtype=float)

    for i in range(len(out)):
        entry_idx = i + 1
        end = min(len(out), entry_idx + ttl_bars)

        if entry_idx >= len(out) or entry_idx >= end:
            continue

        if not np.isfinite(open_[entry_idx]) or open_[entry_idx] <= 0:
            continue

        if not np.isfinite(atr14[i]) or atr14[i] <= 0:
            continue

        ep = open_[entry_idx]
        hh = np.max(high[entry_idx:end])
        ll = np.min(low[entry_idx:end])

        entry_px[i] = ep
        mfe_up[i] = (hh - ep) / atr14[i]
        mae_dn[i] = (ep - ll) / atr14[i]
        mfe_dn[i] = (ep - ll) / atr14[i]
        mae_up[i] = (hh - ep) / atr14[i]

        y_edge_long[i] = 1.0 if mfe_up[i] >= float(target_mfe_atr) else 0.0
        y_edge_short[i] = 1.0 if mfe_dn[i] >= float(target_mfe_dn_atr) else 0.0

    out["entry_px_next_open"] = entry_px
    out["mfe_up_atr_16h"] = mfe_up
    out["mae_dn_atr_16h"] = mae_dn
    out["mfe_dn_atr_16h"] = mfe_dn
    out["mae_up_atr_16h"] = mae_up
    out["y_edge_long"] = y_edge_long
    out["y_edge_short"] = y_edge_short

    return out[
        [
            "ts",
            "entry_px_next_open",
            "mfe_up_atr_16h",
            "mae_dn_atr_16h",
            "mfe_dn_atr_16h",
            "mae_up_atr_16h",
            "y_edge_long",
            "y_edge_short",
        ]
    ]

def threshold_search(valid_df: pd.DataFrame, prob_col: str = "proba", edge_col: str = "edge_main"):
    if len(valid_df) == 0:
        return None, pd.DataFrame()

    valid_df = valid_df.sort_values("ts").reset_index(drop=True)
    base_pos = float(valid_df["y_edge"].mean())
    rows = []

    for thr in THR_GRID:
        if thr > 0.85:
            continue

        kept = valid_df[valid_df[prob_col] >= thr].copy()
        if len(kept) == 0:
            continue

        kept = kept.sort_values("ts").reset_index(drop=True)

        n = 0
        k = 0

        for i in range(1, len(kept)):
            prev = kept.iloc[i - 1]
            n += 1
            k += int(prev["y_edge"])

        min_online_n = max(25, int(np.ceil(0.10 * len(valid_df))))
        if n < min_online_n:
            continue

        precision = k / n if n > 0 else 0.0
        wilson = wilson_lower_bound(k, n) if n > 0 else 0.0
        delta = wilson - base_pos

        if delta <= 0.0:
            continue

        kept_pos = float(kept["y_edge"].mean())
        kept_lift = kept_pos / base_pos if base_pos > 0 else np.nan

        if not np.isfinite(kept_lift) or kept_lift < 1.01:
            continue

        kept_edge_mean = float(kept[edge_col].mean())
        kept_edge_med = float(kept[edge_col].median())
        if not np.isfinite(kept_edge_mean) or not np.isfinite(kept_edge_med):
            continue

        if kept_edge_mean > 6.0:
            continue

        edge_risk = float(kept["edge_aux"].mean())
        risk_penalty = edge_risk / (abs(kept_edge_mean) + 1e-6)

        score = (delta ** 1.2) * np.log1p(n) * (1 + 0.20 * max(0.0, kept_lift - 1.0)) / (1 + risk_penalty)

        rows.append({
            "thr": float(thr),
            "n": int(n),
            "k": int(k),
            "kept_n": int(len(kept)),
            "valid_n": int(len(valid_df)),
            "precision": float(precision),
            "wilson": float(wilson),
            "delta_wilson": float(delta),
            "base_pos_rate": float(base_pos),
            "kept_pos_rate": float(kept_pos),
            "kept_lift": float(kept_lift) if np.isfinite(kept_lift) else np.nan,
            "kept_edge_mean": float(kept_edge_mean),
            "kept_edge_med": float(kept_edge_med),
            "score": float(score),
        })

    grid = pd.DataFrame(rows)
    if len(grid) == 0:
        return None, grid

    best = grid.sort_values(
        ["score", "delta_wilson", "kept_n", "thr"],
        ascending=[False, False, False, True]
    ).iloc[0].to_dict()

    return best, grid


# ============================================================
# MAIN
# ============================================================

# ============================================================
# MAIN
# ============================================================

GATE3_SPLIT_CONFIG = parse_gate3_split_args()

print("=" * 120)
print("GATE3_SCORE_SPLIT_CONFIG")
print(json.dumps(gate3_split_config_for_json(GATE3_SPLIT_CONFIG), ensure_ascii=False, indent=2))
print("=" * 120)

os.makedirs(OUT_ROOT, exist_ok=True)

if not os.path.exists(POLICY_CSV):
    raise SystemExit(f"not found: {POLICY_CSV}")

policy = pd.read_csv(POLICY_CSV)
policy["symbol"] = policy["symbol"].astype(str)

if GATE3_SPLIT_CONFIG.get("symbols") is not None:
    before_policy_rows = len(policy)
    requested_symbols = sorted([str(x).upper() for x in GATE3_SPLIT_CONFIG["symbols"]])
    policy = policy[policy["symbol"].astype(str).str.upper().isin(requested_symbols)].copy()
    print("SYMBOL_FILTER:", requested_symbols)
    print("POLICY_ROWS_BEFORE_FILTER:", before_policy_rows)
    print("POLICY_ROWS_AFTER_FILTER:", len(policy))

    if len(policy) == 0:
        print("GATE3_EMPTY_POLICY_FALLBACK_ENABLED")
        print("GATE3_EMPTY_POLICY_FALLBACK_SYMBOLS:", requested_symbols)

        fallback_rows = []

        for symbol in requested_symbols:
            row = {col: "" for col in policy.columns}
            row["symbol"] = symbol

            for col in [
                "gate3_enabled",
                "gate3_enabled_long",
                "gate3_enabled_short",
                "gate3_use_score_model_long",
                "gate3_use_score_model_short",
            ]:
                if col in row:
                    row[col] = 0

            for col in [
                "gate3_score_long",
                "gate3_score_short",
                "gate3_score_long_z",
                "gate3_score_short_z",
                "gate3_rank_long",
                "gate3_rank_short",
                "gate3_top_pattern_long",
                "gate3_top_pattern_short",
                "gate3_side_bias",
                "gate3_score_threshold_long",
                "gate3_score_threshold_short",
            ]:
                if col in row:
                    row[col] = 0.0

            for col in list(row.keys()):
                if str(col).startswith("active_pa_"):
                    row[col] = 0.0

            if "gate3_mode_long" in row:
                row["gate3_mode_long"] = "disabled"

            if "gate3_mode_short" in row:
                row["gate3_mode_short"] = "disabled"

            if "reason_long" in row:
                row["reason_long"] = "no_active_edge_for_symbol"

            if "reason_short" in row:
                row["reason_short"] = "no_active_edge_for_symbol"

            if "gate3_pattern_long" in row:
                row["gate3_pattern_long"] = ""

            if "gate3_pattern_short" in row:
                row["gate3_pattern_short"] = ""

            if "gate3_score_model_name_long" in row:
                row["gate3_score_model_name_long"] = ""

            if "gate3_score_model_name_short" in row:
                row["gate3_score_model_name_short"] = ""

            fallback_rows.append(row)

        policy = pd.DataFrame(fallback_rows, columns=list(policy.columns))
        print("GATE3_EMPTY_POLICY_FALLBACK_ROWS:", len(policy))


for col in [
    "gate3_enabled",
    "gate3_enabled_long",
    "gate3_enabled_short",
    "gate3_use_score_model_long",
    "gate3_use_score_model_short",
]:
    if col in policy.columns:
        policy[col] = pd.to_numeric(policy[col], errors="coerce").fillna(0).astype(int)

for col in [
    "gate3_pattern_long",
    "gate3_pattern_short",
    "gate3_mode_long",
    "gate3_mode_short",
    "gate3_score_model_name_long",
    "gate3_score_model_name_short",
]:
    if col in policy.columns:
        policy[col] = policy[col].fillna("").astype(str)

for col in [
    "gate3_score_threshold_long",
    "gate3_score_threshold_short",
]:
    if col in policy.columns:
        policy[col] = pd.to_numeric(policy[col], errors="coerce")

gate3_available_symbols = get_gate3_available_symbols(GATE3_DATA_DIR)

if len(gate3_available_symbols) == 0:
    raise SystemExit(f"no parquet symbols found in: {GATE3_DATA_DIR}")

quality_map = load_quality_map(GATE3_AUDIT_CSV) if os.path.exists(GATE3_AUDIT_CSV) else {}

if os.path.exists(GATE3_AUDIT_CSV):
    gate3_audit_df = pd.read_csv(GATE3_AUDIT_CSV)

    if "symbol" not in gate3_audit_df.columns or "pa_valid_rate" not in gate3_audit_df.columns:
        print(
            "GATE3_QUALITY_CSV_SKIP_BAD_FORMAT: {} | columns={}".format(
                GATE3_AUDIT_CSV,
                sorted(list(gate3_audit_df.columns)),
            )
        )
        gate3_audit_df = pd.DataFrame(columns=["symbol", "pa_valid_rate"])
        good_symbols_by_quality = set(gate3_available_symbols)
    else:
        gate3_audit_df["symbol"] = gate3_audit_df["symbol"].astype(str)
        gate3_audit_df["pa_valid_rate"] = pd.to_numeric(
            gate3_audit_df["pa_valid_rate"],
            errors="coerce",
        )

        good_symbols_by_quality = set(
            gate3_audit_df.loc[
                gate3_audit_df["pa_valid_rate"] >= MIN_SYMBOL_PA_VALID_RATE,
                "symbol"
            ].tolist()
        )
else:
    gate3_audit_df = None
    good_symbols_by_quality = set(gate3_available_symbols)

manifest_rows = []
audit_rows = []

for _, prow in policy.iterrows():
    symbol = str(prow["symbol"])
    if symbol not in gate3_available_symbols:
        audit_rows.append({
            "symbol": symbol,
            "pattern": "",
            "rows_total": 0,
            "rows_train": 0,
            "rows_valid": 0,
            "status": "symbol_not_in_gate3_dataset",
        })
        continue

    if symbol not in good_symbols_by_quality:
        audit_rows.append({
            "symbol": symbol,
            "pattern": "",
            "rows_total": 0,
            "rows_train": 0,
            "rows_valid": 0,
            "status": f"filtered_by_pa_valid_rate_lt_{MIN_SYMBOL_PA_VALID_RATE:.2f}",
        })
        continue

    quality_valid_rate = quality_map.get(symbol, np.nan) if quality_map else 1.0
    if not np.isfinite(quality_valid_rate):
        for side in SIDES:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "",
                "pa_valid_rate": np.nan,
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "missing_quality_row",
            })
        continue

    if quality_valid_rate < MIN_SYMBOL_PA_VALID_RATE:
        for side in SIDES:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "filtered_by_quality",
            })
        continue

    fp3 = os.path.join(GATE3_DATA_DIR, f"{symbol}.parquet")

    if not os.path.exists(fp3):
        for side in SIDES:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "missing_gate3_dataset",
            })
        continue

    base_df = load_base_dataset(symbol)
    if base_df is None:
        for side in SIDES:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "missing_base_dataset_for_gate1",
            })
        continue

    h4 = load_h4(symbol)
    if h4 is None or len(h4) < 250:
        for side in SIDES:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "missing_h4",
            })
        continue

    gate1_model_path = load_gate1_model_path(symbol)
    if gate1_model_path is None:
        for side in SIDES:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "missing_gate1_model",
            })
        continue

    d3 = pd.read_parquet(fp3)

    if "entry_ts" not in d3.columns:
        for side in SIDES:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "missing_entry_ts",
            })
        continue

    d3["entry_ts"] = ensure_dt_utc(d3["entry_ts"])

    d3 = (
        d3.dropna(subset=["entry_ts"])
        .sort_values("entry_ts")
        .drop_duplicates("entry_ts", keep="last")
        .reset_index(drop=True)
    )

    all_active_cols = sorted([c for c in d3.columns if c.startswith("active_pa_")])
    all_pa_cols = sorted([c for c in d3.columns if c.startswith("pa_") and not c.startswith("active_pa_")])

    if not all_active_cols:
        for side in SIDES:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "no_active_columns",
            })
        continue

    full_df = h4[["ts", "open", "high", "low", "close"]].copy()
    full_df["ts"] = to_naive_ts(full_df["ts"])

    d3m = d3[["entry_ts"] + all_active_cols + all_pa_cols].copy().rename(columns={"entry_ts": "ts"})
    d3m["ts"] = to_naive_ts(d3m["ts"])

    full_df = full_df.merge(d3m, on="ts", how="left", suffixes=("", "_g3"))
    # =========================
    # MERGE POLICY FEATURES
    # =========================
    prow_df = pd.DataFrame([prow])

    merge_cols = [
        "gate3_score_long",
        "gate3_score_short",
        "gate3_score_long_z",
        "gate3_score_short_z",
        "gate3_rank_long",
        "gate3_rank_short",
        "gate3_side_bias",
    ]

    # ===== MERGE POLICY (CORRECT) =====
    for c in merge_cols:
        if c in prow_df.columns:
            val = float(prow_df[c].values[0])
            full_df[c] = val
        else:
            full_df[c] = 0.0

    # ===== SHIFT (ANTI-LEAKAGE) =====
    for c in merge_cols:
        full_df[c] = full_df[c].shift(1)

    # ===== CLIP (STABILITY) =====
    for c in merge_cols:
        full_df[c] = np.clip(full_df[c], -5, 5)

    # ===== RANK NORMALIZATION =====
    if "gate3_rank_long" in full_df.columns:
        full_df["gate3_rank_long_norm"] = full_df["gate3_rank_long"] / (
                full_df["gate3_rank_long"].abs().max() + 1e-6
        )
    else:
        full_df["gate3_rank_long_norm"] = 0.0

    if "gate3_rank_short" in full_df.columns:
        full_df["gate3_rank_short_norm"] = full_df["gate3_rank_short"] / (
                full_df["gate3_rank_short"].abs().max() + 1e-6
        )
    else:
        full_df["gate3_rank_short_norm"] = 0.0

    for c in all_active_cols + all_pa_cols:
        if c in full_df.columns:
            full_df[c] = pd.to_numeric(full_df[c], errors="coerce").fillna(0.0)

    # ===== QUALITY FEATURES (CRITICAL BOOST) =====
    q = pd.to_numeric(full_df["pa_quality"], errors="coerce") if "pa_quality" in full_df.columns else pd.Series(0.0,
                                                                                                          index=full_df.index)
    q = q.fillna(0.0)
    q_mean = q.shift(1).rolling(200).mean()
    q_std = q.shift(1).rolling(200).std()

    q = (q - q_mean) / (q_std + 1e-6)

    bos_up = pd.to_numeric(full_df["pa_bos_up_12"],
                           errors="coerce") if "pa_bos_up_12" in full_df.columns else pd.Series(0.0,
                                                                                                index=full_df.index)
    bos_up = bos_up.fillna(0.0)

    bos_dn = pd.to_numeric(full_df["pa_bos_dn_12"],
                           errors="coerce") if "pa_bos_dn_12" in full_df.columns else pd.Series(0.0,
                                                                                                index=full_df.index)
    bos_dn = bos_dn.fillna(0.0)

    new_cols = pd.DataFrame({
        "pa_quality_sq": q ** 2,
        "pa_bos_up_12_x_quality": bos_up * q,
        "pa_bos_dn_12_x_quality": bos_dn * q,
    }, index=full_df.index)

    full_df = pd.concat([full_df, new_cols], axis=1)
    ctx = build_full_market_context(h4)
    ctx["ts"] = to_naive_ts(ctx["ts"])
    full_df = full_df.drop(columns=["open", "high", "low", "close"], errors="ignore")
    full_df = full_df.merge(ctx, on="ts", how="left")

    full_df = full_df.copy()

    gate1_input = base_df.copy()
    gate1_input = gate1_input.rename(columns={"entry_ts": "ts"})
    gate1_input["ts"] = to_naive_ts(gate1_input["ts"])

    gate1_proba, _, miss1 = catboost_predict_proba_from_file(gate1_model_path, gate1_input)
    if gate1_proba is None:
        for side in SIDES:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": f"gate1_missing_features:{','.join(miss1[:5])}",
            })
        continue

    gate1_block = gate1_input[["ts"]].copy()
    gate1_block["gate1_proba"] = gate1_proba
    gate1_block["gate1_pass"] = (gate1_block["gate1_proba"] >= GATE1_PROBA_MIN).astype(int)

    full_df = full_df.merge(gate1_block, on="ts", how="left")

    edge_df = compute_future_edge_labels(h4, TTL_BARS, TARGET_MFE_ATR, TARGET_MFE_DN_ATR)
    edge_df["ts"] = to_naive_ts(edge_df["ts"])
    full_df = full_df.merge(edge_df, on="ts", how="left")
    full_df = full_df.dropna(subset=["y_edge_long", "y_edge_short"])

    h4_pos_ts = to_naive_ts(h4["ts"])
    ts_to_pos = {ts: i for i, ts in enumerate(h4_pos_ts)}


    for side in SIDES:
        gate3_enabled_side = 1

        side_active_cols = select_side_pattern_cols(all_active_cols, side)
        side_pa_cols = select_side_pattern_cols(all_pa_cols, side)

        if not side_active_cols:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "no_side_active_columns",
            })
            continue

        side_df = full_df.copy()

        if side_pa_cols:
            pa_vals = side_df[side_pa_cols].copy()
            pa_vals = pa_vals.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            side_df["pa_sum"] = pa_vals.sum(axis=1)
            side_df["pa_mean"] = pa_vals.mean(axis=1)
            side_df["pa_max"] = pa_vals.max(axis=1)
            side_df["pa_std"] = pa_vals.std(axis=1)

            q_local = pd.to_numeric(side_df["pa_quality"], errors="coerce").fillna(
                0.0) if "pa_quality" in side_df.columns else 0.0
            side_df["pa_strength_weighted"] = side_df["pa_sum"] * q_local
            side_df["pa_strength_x_gate1"] = side_df["pa_strength_weighted"] * side_df["gate1_proba"]
            side_df["ctx_ret1_z"] = (side_df["ctx_ret1"] - side_df["ctx_ret1"].rolling(100).mean()) / (
                    side_df["ctx_ret1"].rolling(100).std() + 1e-6)

        side_df = add_active_set_features(side_df, side_active_cols, side=side)
        side_df = add_active_persistence_features(side_df, side_active_cols)
        side_df["active_age_norm"] = side_df["gate3_max_active_age"] / 10.0
        # ===== TYPE-AWARE FEATURES (CRITICAL BOOST) =====
        active_matrix = side_df[side_active_cols].copy()
        for c in side_active_cols:
            active_matrix[c] = safe_bool_series(active_matrix[c])

        # количество активных паттернов
        side_df["active_type_count"] = active_matrix.sum(axis=1).astype(int)

        # есть ли вообще сигнал
        side_df["active_any"] = (side_df["active_type_count"] > 0).astype(int)

        # ровно один тип сигнала
        side_df["active_is_single"] = (side_df["active_type_count"] == 1).astype(int)

        # комбинация (2+ сигналов)
        side_df["active_is_combo"] = (side_df["active_type_count"] >= 2).astype(int)

        # нормализованная плотность сигналов
        side_df["active_density"] = side_df["active_type_count"] / max(1, len(side_active_cols))

        # энтропия (разнообразие паттернов)
        p = active_matrix.div(active_matrix.sum(axis=1).replace(0, np.nan), axis=0)

        # взаимодействие с качеством
        if "pa_quality" in side_df.columns:
            q = pd.to_numeric(side_df["pa_quality"], errors="coerce").fillna(0.0)
            side_df["active_count_x_quality"] = side_df["active_type_count"] * q
            side_df["active_combo_x_quality"] = side_df["active_is_combo"] * q


        # взаимодействие с gate1 (ОЧЕНЬ ВАЖНО)
        side_df["active_count_x_gate1"] = side_df["active_type_count"] * side_df["gate1_proba"]
        side_df["active_combo_x_gate1"] = side_df["active_is_combo"] * side_df["gate1_proba"]

        side_df = add_lag_features(
            side_df,
            cols=[
                "gate1_proba",
                "gate3_active_count",
                "ctx_ret1",
                "ctx_ret2",
                "ctx_ret4",
                "ctx_ret8",
                "ctx_atrp14",
                "ctx_range_atr",
            ],
            lags=(1, 2),
        )

        if side == "long":
            side_df["y_edge"] = pd.to_numeric(side_df["y_edge_long"], errors="coerce").fillna(0).astype(int)
            side_df["edge_main"] = pd.to_numeric(side_df["mfe_up_atr_16h"], errors="coerce")
            side_df["edge_aux"] = pd.to_numeric(side_df["mae_dn_atr_16h"], errors="coerce")
        else:
            side_df["y_edge"] = pd.to_numeric(side_df["y_edge_short"], errors="coerce").fillna(0).astype(int)
            side_df["edge_main"] = pd.to_numeric(side_df["mfe_dn_atr_16h"], errors="coerce")
            side_df["edge_aux"] = pd.to_numeric(side_df["mae_up_atr_16h"], errors="coerce")

        candidates = side_df[
            (
                    (side_df["gate3_active_count"] >= 1)
                    | (side_df["gate3_score_long"] > side_df["gate3_score_long"].quantile(0.7) if side == "long" else side_df["gate3_score_short"] > side_df["gate3_score_short"].quantile(0.7))
            ) &
            (side_df["ctx_atrp14"] > 0.01) &
            (side_df["gate1_pass"] == 1)
            ].copy()

        candidates["signal_pos"] = candidates["ts"].map(ts_to_pos)
        candidates = candidates.dropna(subset=["signal_pos"]).copy()
        candidates["signal_pos"] = candidates["signal_pos"].astype(int)
        candidates = candidates[(candidates["signal_pos"] + 1) < len(h4)].copy()

        if len(candidates) == 0:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": 0,
                "rows_train": 0,
                "rows_valid": 0,
                "status": "no_candidates_after_gate1_and_any_active",
            })
            continue

        candidates = candidates.sort_values("ts").reset_index(drop=True)

        split_info = split_gate3_train_valid(candidates, GATE3_SPLIT_CONFIG)

        n_total = int(split_info["rows_total"])
        n_train = int(split_info["rows_train"])
        n_valid = int(split_info["rows_valid"])
        split_source = str(split_info["split_source"])

        train_df = split_info["train_df"]
        valid_df = split_info["valid_df"]

        if n_total < MIN_ROWS_TOTAL or n_train < MIN_TRAIN_ROWS or n_valid < MIN_VALID_ROWS:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": int(n_total),
                "rows_train": int(n_train),
                "rows_valid": int(n_valid),
                "split_source": split_source,
                "train_min_ts": split_info.get("train_min_ts"),
                "train_max_ts": split_info.get("train_max_ts"),
                "valid_min_ts": split_info.get("valid_min_ts"),
                "valid_max_ts": split_info.get("valid_max_ts"),
                "status": "too_few_rows",
            })
            continue
        if train_df["y_edge"].nunique() < 2 or valid_df["y_edge"].nunique() < 2:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": int(n_total),
                "rows_train": int(n_train),
                "rows_valid": int(n_valid),
                "status": "single_class_train_or_valid",
            })
            continue

        extra_feature_candidates = [
            c for c in candidates.columns
            if (
                    c in side_active_cols
                    or c in side_pa_cols
                    or c.startswith("gate3_")
                    or c.startswith("active_")
                    or c.endswith("_x_quality")
                    or c == "pa_quality_sq"
            )
        ]



        banned = {
            "ts",
            "y_edge",
            "y_edge_long",
            "y_edge_short",
            "mfe_up_atr_16h",
            "mae_dn_atr_16h",
            "mfe_dn_atr_16h",
            "mae_up_atr_16h",
            "edge_main",
            "edge_aux",
            "signal_pos",
            "edge_main",
            "edge_aux",
        }

        feature_cols = []
        seen = set()
        base_feature_candidates = ["gate1_proba"]

        for c in base_feature_candidates + extra_feature_candidates:
            if c in banned:
                continue
            if c not in candidates.columns:
                continue
            if c in seen:
                continue
            if not pd.api.types.is_numeric_dtype(candidates[c]):
                continue
            seen.add(c)
            feature_cols.append(c)

        if not feature_cols:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": int(n_total),
                "rows_train": int(n_train),
                "rows_valid": int(n_valid),
                "status": "no_features",
            })
            continue

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan)

        med = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med)
        X_valid = X_valid.fillna(med)

        y_train = train_df["y_edge"].astype(int)
        y_valid = valid_df["y_edge"].astype(int)

        pos_rate = y_train.mean()
        class_weights = {
            0: 1.0,
            1: min(5.0, 1.0 / (pos_rate + 1e-6))
        }

        model = CatBoostClassifier(**CB_PARAMS, class_weights=class_weights)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
        )

        valid_df = valid_df.copy()
        valid_df["proba"] = model.predict_proba(X_valid)[:, 1]

        best_thr, thr_grid = threshold_search(valid_df, prob_col="proba", edge_col="edge_main")
        if best_thr is None:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": int(n_total),
                "rows_train": int(n_train),
                "rows_valid": int(n_valid),
                "status": "threshold_search_failed",
            })
            continue

        # ===== ONLINE n/k SIMULATION =====

        kept = valid_df[valid_df["proba"] >= best_thr["thr"]].copy()

        kept = kept.sort_values("ts").reset_index(drop=True)

        n = 0
        k = 0

        online_stats = []

        for i in range(len(kept)):
            row = kept.iloc[i]

            # обновляем только ПОСЛЕ сделки (как в проде)
            if i > 0:
                prev = kept.iloc[i - 1]
                n += 1
                k += int(prev["y_edge"])

            precision_online = k / n if n > 0 else 0.0
            wilson_online = wilson_lower_bound(k, n) if n > 0 else 0.0

            online_stats.append({
                "n": n,
                "k": k,
                "precision": precision_online,
                "wilson": wilson_online,
            })

        online_df = pd.DataFrame(online_stats)

        # финальные значения (как в проде на текущий момент)
        if len(online_df) == 0:
            continue

        n = int(online_df.iloc[-1]["n"])
        k = int(online_df.iloc[-1]["k"])
        precision = float(online_df.iloc[-1]["precision"])
        wilson = float(online_df.iloc[-1]["wilson"])
        base_pos = float(valid_df["y_edge"].mean())

        delta = wilson - base_pos

        if delta <= 0.0:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": int(n_total),
                "rows_train": int(n_train),
                "rows_valid": int(n_valid),
                "status": "no_stat_edge",
            })
            continue

        p_value = binomtest(k, n, p=base_pos, alternative="greater").pvalue

        if precision >= 0.98 and n < 120:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": int(n_total),
                "rows_train": int(n_train),
                "rows_valid": int(n_valid),
                "status": "overfit_ultra_high_precision_small_n",
            })
            continue

        if p_value > 0.05:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": int(n_total),
                "rows_train": int(n_train),
                "rows_valid": int(n_valid),
                "status": "not_significant",
            })
            continue

        if n < 30:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": int(n_total),
                "rows_train": int(n_train),
                "rows_valid": int(n_valid),
                "status": "too_few_kept_after_threshold",
            })
            continue

        out_dir = os.path.join(OUT_ROOT, symbol, side, "gate3_score")
        os.makedirs(out_dir, exist_ok=True)

        model_path = os.path.join(out_dir, "gate3_score.cbm")
        meta_path = os.path.join(out_dir, "meta.json")
        thr_grid_path = os.path.join(out_dir, "threshold_grid.csv")

        model.save_model(model_path)
        thr_grid.to_csv(thr_grid_path, index=False)

        meta = {
            "symbol": symbol,
            "side": side,
            "primary_policy_pattern": "multi",
            "training_mode": "multi_active_patterns_full_sequence_long_short",
            "rows_total": int(n_total),
            "rows_train": int(n_train),
            "rows_valid": int(n_valid),
            "split": {
                "type": split_source,
                "train_end": split_info.get("train_end"),
                "valid_start_ts": split_info.get("valid_start"),
                "valid_end_ts": split_info.get("valid_end"),
                "train_cutoff": split_info.get("train_cutoff"),
                "gap_bars": split_info.get("gap_bars"),
                "gap_delta_hours": split_info.get("gap_delta_hours"),
                "train_condition": split_info.get("train_condition"),
                "valid_condition": split_info.get("valid_condition"),
                "train_min_ts": split_info.get("train_min_ts"),
                "train_max_ts": split_info.get("train_max_ts"),
                "valid_min_ts": split_info.get("valid_min_ts"),
                "valid_max_ts": split_info.get("valid_max_ts"),
            },
            "feature_count": int(len(feature_cols)),
            "feature_names": feature_cols,
            "quality_valid_rate": float(quality_valid_rate),
                        "target": {
                "name": "y_edge",
                "ttl_bars": int(TTL_BARS),
                "target_mfe_atr_long": float(TARGET_MFE_ATR),
                "target_mfe_atr_short": float(TARGET_MFE_DN_ATR),
                "long_definition": f"mfe_up_atr_16h >= {TARGET_MFE_ATR}",
                "short_definition": f"mfe_dn_atr_16h >= {TARGET_MFE_DN_ATR}",
            },
            "stats": {
                "n": int(n),
                "k": int(k),
                "precision": float(precision),
                "wilson_lower": float(wilson),
                "base_rate": float(base_pos),
                "delta_wilson": float(delta),
                "p_value": float(p_value)
            },
            "best_threshold": float(best_thr["thr"]),
            "best_threshold_kept_n": int(best_thr["kept_n"]),
            "best_threshold_kept_pos_rate": float(best_thr["kept_pos_rate"]),
            "best_threshold_kept_lift": float(best_thr["kept_lift"]) if np.isfinite(best_thr["kept_lift"]) else None,
            "best_threshold_kept_edge_mean": float(best_thr["kept_edge_mean"]),
            "best_threshold_kept_edge_med": float(best_thr["kept_edge_med"]),
            "entry_logic": {
                "signal_bar": "closed_h4_t",
                "entry_bar": "next_h4_open_t_plus_1",
                "execution_delay_inside_next_bar": "allowed",
                "slippage_bps": float(SLIPPAGE_BPS),
                "fee_bps_one_way": float(FEE_BPS),
                "full_sequence_context_preserved": True,
                "multiple_active_patterns_used": True,
                "gate1_hard_filter_in_training": True,
                "gate2_used_as_feature_only_in_training": False,
                "gate2_hard_filter_in_production": False,
                "gate3_hard_filter_in_production": True,
            },
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if best_thr["kept_edge_mean"] <= 0:
            audit_rows.append({
                "symbol": symbol,
                "side": side,
                "pattern": "multi",
                "pa_valid_rate": float(quality_valid_rate),
                "rows_total": int(n_total),
                "rows_train": int(n_train),
                "rows_valid": int(n_valid),
                "status": "low_edge_filtered",
            })
            continue




        manifest_rows.append({
            "symbol": symbol,
            "side": side,
            "pattern": "multi",
            "pa_valid_rate": float(quality_valid_rate),
            "model_path": model_path,
            "meta_path": meta_path,
            "threshold_grid_path": thr_grid_path,
            "threshold": float(best_thr["thr"]),
            "rows_total": int(n_total),
            "rows_train": int(n_train),
            "rows_valid": int(n_valid),
            "split_source": split_source,
            "train_end": split_info.get("train_end"),
            "valid_start_ts": split_info.get("valid_start"),
            "valid_end_ts": split_info.get("valid_end"),
            "train_min_ts": split_info.get("train_min_ts"),
            "train_max_ts": split_info.get("train_max_ts"),
            "valid_min_ts": split_info.get("valid_min_ts"),
            "valid_max_ts": split_info.get("valid_max_ts"),
            "valid_pos_rate": float(y_valid.mean()),
            "thr_kept_n": int(best_thr["kept_n"]),
            "thr_kept_pos_rate": float(best_thr["kept_pos_rate"]),
            "thr_kept_lift": float(best_thr["kept_lift"]) if np.isfinite(best_thr["kept_lift"]) else np.nan,
            "thr_kept_edge_mean": float(best_thr["kept_edge_mean"]),
            "thr_kept_edge_med": float(best_thr["kept_edge_med"]),
            "feature_count": int(len(feature_cols)),
            "n": int(n),
            "k": int(k),
            "precision": float(precision),
            "wilson_lower": float(wilson),
            "delta_wilson": float(delta),
            "p_value": float(p_value),
            "status": "ok",
        })

        audit_rows.append({
            "symbol": symbol,
            "side": side,
            "pattern": "multi",
            "pa_valid_rate": float(quality_valid_rate),
            "rows_total": int(n_total),
            "rows_train": int(n_train),
            "rows_valid": int(n_valid),
            "split_source": split_source,
            "train_end": split_info.get("train_end"),
            "valid_start_ts": split_info.get("valid_start"),
            "valid_end_ts": split_info.get("valid_end"),
            "train_min_ts": split_info.get("train_min_ts"),
            "train_max_ts": split_info.get("train_max_ts"),
            "valid_min_ts": split_info.get("valid_min_ts"),
            "valid_max_ts": split_info.get("valid_max_ts"),
            "status": "ok",
        })

manifest_df = pd.DataFrame(manifest_rows)
audit_df = pd.DataFrame(audit_rows)

if len(manifest_df):
    manifest_df = manifest_df.sort_values(
        ["side", "thr_kept_lift", "thr_kept_pos_rate", "thr_kept_n", "symbol"],
        ascending=[True, False, False, False, True]
    ).reset_index(drop=True)

if len(audit_df):
    for col, default_value in [
        ("status", ""),
        ("side", ""),
        ("rows_total", 0),
        ("symbol", ""),
    ]:
        if col not in audit_df.columns:
            audit_df[col] = default_value

    audit_df = audit_df.sort_values(
        ["status", "side", "rows_total", "symbol"],
        ascending=[True, True, False, True]
    ).reset_index(drop=True)

manifest_df.to_csv(OUT_MANIFEST_CSV, index=False)
audit_df.to_csv(OUT_AUDIT_CSV, index=False)

report = {
    "models_trained": int(len(manifest_df)),
    "symbols_in_policy_enabled_long": int((policy["gate3_enabled_long"] == 1).sum()) if "gate3_enabled_long" in policy.columns else 0,
    "symbols_in_policy_enabled_short": int((policy["gate3_enabled_short"] == 1).sum()) if "gate3_enabled_short" in policy.columns else 0,
    "ttl_bars": int(TTL_BARS),
    "target_mfe_atr_long": float(TARGET_MFE_ATR),
    "target_mfe_atr_short": float(TARGET_MFE_DN_ATR),
    "slippage_bps": float(SLIPPAGE_BPS),
    "fee_bps_one_way": float(FEE_BPS),
    "train_config": CB_PARAMS,
    "training_mode": "multi_active_patterns_full_sequence_long_short",
    "quality_min_valid_rate": float(MIN_SYMBOL_PA_VALID_RATE),
    "sides": list(SIDES),
    "split_config": gate3_split_config_for_json(GATE3_SPLIT_CONFIG),
}

with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# ============================================================
# UPDATE POLICY
# ============================================================

policy_upd = policy.copy()

new_policy_cols = [
    "gate3_use_score_model_long",
    "gate3_score_model_name_long",
    "gate3_score_threshold_long",
    "gate3_use_score_model_short",
    "gate3_score_model_name_short",
    "gate3_score_threshold_short",
]

for col in new_policy_cols:
    if col not in policy_upd.columns:
        if col.startswith("gate3_use_"):
            policy_upd[col] = 0
        elif col.startswith("gate3_score_threshold_"):
            policy_upd[col] = np.nan
        else:
            policy_upd[col] = ""

policy_upd["gate3_use_score_model_long"] = pd.to_numeric(policy_upd["gate3_use_score_model_long"], errors="coerce").fillna(0).astype(int)
policy_upd["gate3_use_score_model_short"] = pd.to_numeric(policy_upd["gate3_use_score_model_short"], errors="coerce").fillna(0).astype(int)
policy_upd["gate3_score_model_name_long"] = policy_upd["gate3_score_model_name_long"].fillna("").astype(str)
policy_upd["gate3_score_model_name_short"] = policy_upd["gate3_score_model_name_short"].fillna("").astype(str)
policy_upd["gate3_score_threshold_long"] = pd.to_numeric(policy_upd["gate3_score_threshold_long"], errors="coerce")
policy_upd["gate3_score_threshold_short"] = pd.to_numeric(policy_upd["gate3_score_threshold_short"], errors="coerce")
policy_upd["gate3_use_score_model_long"] = 0
policy_upd["gate3_score_model_name_long"] = ""
policy_upd["gate3_score_threshold_long"] = np.nan

policy_upd["gate3_use_score_model_short"] = 0
policy_upd["gate3_score_model_name_short"] = ""
policy_upd["gate3_score_threshold_short"] = np.nan

if len(manifest_df):
    by_symbol_side = {}
    for _, r in manifest_df.iterrows():
        by_symbol_side[(str(r["symbol"]), str(r["side"]))] = r

    for i in range(len(policy_upd)):
        sym = str(policy_upd.at[i, "symbol"])

        r_long = by_symbol_side.get((sym, "long"))
        if r_long is not None:
            policy_upd.at[i, "gate3_use_score_model_long"] = 1
            policy_upd.at[i, "gate3_score_model_name_long"] = "gate3_score.cbm"
            policy_upd.at[i, "gate3_score_threshold_long"] = float(r_long["threshold"])

        r_short = by_symbol_side.get((sym, "short"))
        if r_short is not None:
            policy_upd.at[i, "gate3_use_score_model_short"] = 1
            policy_upd.at[i, "gate3_score_model_name_short"] = "gate3_score.cbm"
            policy_upd.at[i, "gate3_score_threshold_short"] = float(r_short["threshold"])

policy_upd.to_csv(POLICY_CSV + ".updated", index=False)

# ============================================================
# PRINT
# ============================================================

print("WROTE", OUT_MANIFEST_CSV)
print("WROTE", OUT_AUDIT_CSV)
print("WROTE", OUT_REPORT_JSON)
print("UPDATED", POLICY_CSV)
print()

if len(manifest_df):
    print("TOP TRAINED MODELS")
    print(manifest_df.head(100).to_string(index=False))
else:
    print("No models trained")
print()

if len(audit_df):
    print("AUDIT STATUS")
    print(audit_df.groupby(["side", "status"]).size().to_string())
else:
    print("Audit is empty")
