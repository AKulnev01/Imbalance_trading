from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SRC_GATE2_DIR = "production/dataset/gate1"
H4_DIR = "data/h4_3"
M1_DIR = "data/m1_4"
OLD_GATE2_MODELS_DIR = "production/models/final_gate2"

OUT_REACH_DIR = "production/dataset/final_gate2_2_directional_reach_5features_by_symbol"
OUT_STRENGTH_DIR = "production/dataset/final_gate2_3_directional_strength_5features_by_symbol"

OUT_REACH_ALL = "production/dataset/final_gate2_2_directional_reach_5features_all.parquet"
OUT_STRENGTH_ALL = "production/dataset/final_gate2_3_directional_strength_5features_all.parquet"

OUT_AUDIT_CSV = "production/dataset/final_gate2_directional_5features__AUDIT.csv"
OUT_REPORT_JSON = "production/dataset/final_gate2_directional_5features__REPORT.json"

# ============================================================
# CONFIG
# ============================================================

H_HOURS = 16
IMPULSE_HOURS = 8
ENTRY_DELAY_SECONDS = 90

REACH_HIGH_ATR = 1.00

IMPULSE_ATR_MAIN = 2.0

VALID_TAIL_SHARE_FALLBACK = 0.20

DROP_COLS_FROM_SOURCE = {
    "side",
    "side_num",
    "side_num_feat",

    "symbol_id",
    "symbol_id_feat",

    "tp_px",
    "sl_px",
    "exit_ts",
    "exit_px",
    "exit_reason",
    "pnl_net",

    "y",
    "y_fast",

    "upstream_split",
    "upstream_valid_start_ts",

    "entry_px",   # дубликат entry_px_exec
    "ref_close",
    "ref_close_feat",
    "ref_btc_close",
    "ref_btc_close_feat",
    "ref_eth_close",
    "ref_eth_close_feat",
}

DROP_PREFIXES_FROM_SOURCE = (
    "ks_",
    "sym_",
)

META_COLS_COMMON = [
    "symbol",
    "entry_ts",
    "signal_ts",
    "entry_bar_open_ts",
    "entry_ts_exec",
    "entry_px_exec",
    "upstream_valid_start_ts",
    "upstream_split",
]

REACH_TARGET_COLS = [
    "gate2_up_reach_high",
    "gate2_dn_reach_high",
]

STRENGTH_TARGET_COLS = [
    "gate2_up_impulse_8h_2atr",
    "gate2_dn_impulse_8h_2atr",
]

TIME_COL_CANDIDATES = ["entry_ts", "ts", "open_time", "time", "datetime", "timestamp"]
M1_TIME_COL_CANDIDATES = ["ts", "time", "datetime", "open_time", "timestamp"]
H4_TIME_COL_CANDIDATES = ["ts", "open_time", "time", "datetime", "timestamp"]


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_naive_utc_auto(x) -> pd.Series:
    s = pd.Series(x)

    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").astype("float64")
        finite = vals[np.isfinite(vals)]

        if len(finite) == 0:
            out = pd.to_datetime(vals, utc=True, errors="coerce")
            return pd.Series(out).dt.tz_localize(None)

        vmax = float(finite.max())

        if vmax > 1e18:
            out = pd.to_datetime(vals, unit="ns", utc=True, errors="coerce")
        elif vmax > 1e15:
            out = pd.to_datetime(vals, unit="us", utc=True, errors="coerce")
        elif vmax > 1e11:
            out = pd.to_datetime(vals, unit="ms", utc=True, errors="coerce")
        else:
            out = pd.to_datetime(vals, unit="s", utc=True, errors="coerce")
    else:
        out = pd.to_datetime(s, utc=True, errors="coerce")

    return pd.Series(out).dt.tz_localize(None)


def find_first_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_json_safe(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


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


def load_h4(symbol: str) -> Optional[pd.DataFrame]:
    path = os.path.join(H4_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    tcol = find_first_col(df.columns, H4_TIME_COL_CANDIDATES)
    if tcol is None:
        return None

    df["ts"] = to_naive_utc_auto(df[tcol])
    df = df.dropna(subset=["ts"]).copy()
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    need = ["open", "high", "low", "close"]
    if any(c not in df.columns for c in need):
        return None

    return df[["ts", "open", "high", "low", "close"]].copy()


def load_m1(symbol: str) -> Optional[pd.DataFrame]:
    path_plain = os.path.join(M1_DIR, f"{symbol}.parquet")
    path_m1 = os.path.join(M1_DIR, f"{symbol}_m1.parquet")

    if os.path.exists(path_plain):
        path = path_plain
    elif os.path.exists(path_m1):
        path = path_m1
    else:
        return None

    df = pd.read_parquet(path)

    if isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        else:
            idx = idx.tz_localize(None)
        df.index = idx
    else:
        tcol = find_first_col(df.columns, M1_TIME_COL_CANDIDATES)
        if tcol is None:
            return None
        idx = to_naive_utc_auto(df[tcol])
        df.index = pd.DatetimeIndex(idx)

    df = df[~df.index.isna()].copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    need = ["open", "high", "low", "close"]
    if any(c not in df.columns for c in need):
        return None

    return df[["open", "high", "low", "close"]].copy()


def atr14(df: pd.DataFrame) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev).abs(), (l - prev).abs()))
    return tr.rolling(14).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev).abs(), (l - prev).abs()))
    return pd.Series(tr, index=df.index, dtype=float)


def add_gate2_5features(h4: pd.DataFrame) -> pd.DataFrame:
    out = h4.copy()

    o = out["open"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)

    tr = true_range(out)

    out["atr4h"] = tr.rolling(4, min_periods=4).mean()

    out["ret_l1"] = c.pct_change(1)
    out["ret_l2"] = c.pct_change(2)

    body = (c - o).abs()
    rng = (h - l).replace(0.0, np.nan)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)

    out["hammer_like"] = (
        (lower_wick >= 2.0 * body) &
        (upper_wick <= 0.35 * rng) &
        ((body / rng) <= 0.45)
    ).astype(float)

    atr14_to_price = out["atr14"] / c.replace(0.0, np.nan)
    q33 = atr14_to_price.rolling(96, min_periods=32).quantile(0.33)
    q66 = atr14_to_price.rolling(96, min_periods=32).quantile(0.66)

    out["vol_regime"] = np.where(
        atr14_to_price <= q33,
        0.0,
        np.where(atr14_to_price <= q66, 1.0, 2.0)
    )
    out.loc[q33.isna() | q66.isna(), "vol_regime"] = np.nan

    return out


def first_m1_bar_at_or_after(m1: pd.DataFrame, ts_exec: pd.Timestamp):
    idx = m1.index.searchsorted(ts_exec, side="left")
    if idx >= len(m1):
        return None
    return m1.index[idx], m1.iloc[idx]


def dedupe_source_gate2(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    dup_rows_removed = 0

    tcol = find_first_col(df.columns, TIME_COL_CANDIDATES)
    if tcol is None:
        raise ValueError("missing time column in gate2 source")

    df["entry_ts"] = to_naive_utc_auto(df[tcol])
    df = df.dropna(subset=["entry_ts"]).copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)

    if "side" in df.columns:
        before = len(df)
        df = df.drop(columns=["side"])
        dup_rows_removed += before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["entry_ts"], keep="first").reset_index(drop=True)
    dup_rows_removed += before - len(df)

    return df, dup_rows_removed

def select_safe_source_cols(df: pd.DataFrame) -> list[str]:
    cols = []

    for c in df.columns:
        if c in DROP_COLS_FROM_SOURCE:
            continue

        if any(str(c).startswith(p) for p in DROP_PREFIXES_FROM_SOURCE):
            continue

        # ЖЁСТКО убираем *_feat (никаких дублей признаков)
        if str(c).endswith("_feat"):
            continue

        # также убираем служебные дубли symbol_id и side_num
        if str(c) in ("symbol_id_feat", "side_num_feat"):
            continue

        cols.append(c)

    if "entry_ts" not in cols:
        cols.append("entry_ts")

    return cols

def compute_directional_labels_for_symbol(base_df: pd.DataFrame, h4: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    # === REF CACHE (BTC / ETH) ===
    if not hasattr(compute_directional_labels_for_symbol, "_ref_cache"):
        compute_directional_labels_for_symbol._ref_cache = {}

    ref_cache = compute_directional_labels_for_symbol._ref_cache

    def get_ref_h4(sym: str):
        if sym not in ref_cache:
            ref_cache[sym] = load_h4(sym)
        return ref_cache[sym]

    ref_btc_h4 = get_ref_h4("BTCUSDT")
    ref_eth_h4 = get_ref_h4("ETHUSDT")
    h4 = h4.copy()
    h4["atr14"] = atr14(h4)
    h4 = add_gate2_5features(h4)

    ts_to_pos = {ts: i for i, ts in enumerate(h4["ts"])}

    rows = []

    for _, r in base_df.iterrows():
        signal_ts = pd.Timestamp(r["entry_ts"])

        if signal_ts not in ts_to_pos:
            continue

        i = ts_to_pos[signal_ts]

        if i + 1 >= len(h4):
            continue

        atr = float(h4.iloc[i]["atr14"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        entry_bar_open_ts = pd.Timestamp(h4.iloc[i + 1]["ts"])
        entry_ts_exec = entry_bar_open_ts + pd.Timedelta(seconds=ENTRY_DELAY_SECONDS)

        entry_bar = first_m1_bar_at_or_after(m1, entry_ts_exec)
        if entry_bar is None:
            continue

        real_entry_ts, entry_bar_row = entry_bar
        entry_px = float(entry_bar_row["open"])

        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        ttl_ts_8h = real_entry_ts + pd.Timedelta(hours=IMPULSE_HOURS)
        ttl_ts_16h = real_entry_ts + pd.Timedelta(hours=H_HOURS)

        m1_slice_8h = m1.loc[real_entry_ts:ttl_ts_8h]
        m1_slice_16h = m1.loc[real_entry_ts:ttl_ts_16h]

        if len(m1_slice_8h) == 0 or len(m1_slice_16h) == 0:
            continue

        hi_8h = float(m1_slice_8h["high"].max())
        lo_8h = float(m1_slice_8h["low"].min())
        hi_16h = float(m1_slice_16h["high"].max())
        lo_16h = float(m1_slice_16h["low"].min())

        mfe_up_atr_8h = np.clip((hi_8h - entry_px) / atr, 0, 5)
        mfe_dn_atr_8h = np.clip((entry_px - lo_8h) / atr, 0, 5)
        mfe_up_atr_h = np.clip((hi_16h - entry_px) / atr, 0, 7)
        mfe_dn_atr_h = np.clip((entry_px - lo_16h) / atr, 0, 7)

        mfe_up_pct_8h = np.clip((hi_8h - entry_px) / entry_px, 0, 0.5)
        mfe_dn_pct_8h = np.clip((entry_px - lo_8h) / entry_px, 0, 0.5)
        mfe_up_pct_16h = np.clip((hi_16h - entry_px) / entry_px, 0, 0.7)
        mfe_dn_pct_16h = np.clip((entry_px - lo_16h) / entry_px, 0, 0.7)

        mae_up_pct_8h = (entry_px - lo_8h) / entry_px
        mae_dn_pct_8h = (hi_8h - entry_px) / entry_px
        mae_up_pct_16h = (entry_px - lo_16h) / entry_px
        mae_dn_pct_16h = (hi_16h - entry_px) / entry_px

        up_high = int(mfe_up_atr_h >= REACH_HIGH_ATR)
        dn_high = int(mfe_dn_atr_h >= REACH_HIGH_ATR)

        up_impulse_8h_2atr = int(mfe_up_atr_8h >= IMPULSE_ATR_MAIN)
        dn_impulse_8h_2atr = int(mfe_dn_atr_8h >= IMPULSE_ATR_MAIN)


        first_up_high_ts = pd.NaT
        first_dn_high_ts = pd.NaT


        for bar_ts, bar in m1_slice_16h.iterrows():
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])

            up_move_atr = (bar_high - entry_px) / atr
            dn_move_atr = (entry_px - bar_low) / atr

            if pd.isna(first_up_high_ts) and up_move_atr >= REACH_HIGH_ATR:
                first_up_high_ts = pd.Timestamp(bar_ts)
            if pd.isna(first_dn_high_ts) and dn_move_atr >= REACH_HIGH_ATR:
                first_dn_high_ts = pd.Timestamp(bar_ts)

        row = {
            k: v for k, v in r.items()
            if (not str(k).endswith("_feat"))
            and (k not in ("symbol_id_feat", "side_num_feat"))
        }
        row["signal_ts"] = signal_ts
        row["entry_bar_open_ts"] = entry_bar_open_ts
        row["entry_ts_exec"] = pd.Timestamp(real_entry_ts)
        row["entry_px_exec"] = entry_px
        # === REF PRICES (строго фаза T, без leakage) ===
        row["ref_close"] = float(h4.iloc[i]["close"])

        if ref_btc_h4 is not None:
            btc_row = ref_btc_h4.loc[ref_btc_h4["ts"] == signal_ts]
            row["ref_btc_close"] = float(btc_row["close"].iloc[0]) if len(btc_row) else np.nan
        else:
            row["ref_btc_close"] = np.nan

        if ref_eth_h4 is not None:
            eth_row = ref_eth_h4.loc[ref_eth_h4["ts"] == signal_ts]
            row["ref_eth_close"] = float(eth_row["close"].iloc[0]) if len(eth_row) else np.nan
        else:
            row["ref_eth_close"] = np.nan
        row["atr14_at_signal"] = atr
        row["ttl_hours"] = int(H_HOURS)
        row["impulse_hours"] = int(IMPULSE_HOURS)

        row["atr4h"] = float(h4.iloc[i]["atr4h"]) if np.isfinite(h4.iloc[i]["atr4h"]) else np.nan
        row["ret_l1"] = float(h4.iloc[i]["ret_l1"]) if np.isfinite(h4.iloc[i]["ret_l1"]) else np.nan
        row["ret_l2"] = float(h4.iloc[i]["ret_l2"]) if np.isfinite(h4.iloc[i]["ret_l2"]) else np.nan
        row["hammer_like"] = float(h4.iloc[i]["hammer_like"]) if np.isfinite(h4.iloc[i]["hammer_like"]) else np.nan
        row["vol_regime"] = float(h4.iloc[i]["vol_regime"]) if np.isfinite(h4.iloc[i]["vol_regime"]) else np.nan

        row["mfe_up_atr_8h"] = float(mfe_up_atr_8h)
        row["mfe_dn_atr_8h"] = float(mfe_dn_atr_8h)
        row["mfe_up_atr_h"] = float(mfe_up_atr_h)
        row["mfe_dn_atr_h"] = float(mfe_dn_atr_h)

        row["mfe_up_pct_8h"] = float(mfe_up_pct_8h)
        row["mfe_dn_pct_8h"] = float(mfe_dn_pct_8h)
        row["mfe_up_pct_16h"] = float(mfe_up_pct_16h)
        row["mfe_dn_pct_16h"] = float(mfe_dn_pct_16h)

        row["mae_up_pct_8h"] = float(mae_up_pct_8h)
        row["mae_dn_pct_8h"] = float(mae_dn_pct_8h)
        row["mae_up_pct_16h"] = float(mae_up_pct_16h)
        row["mae_dn_pct_16h"] = float(mae_dn_pct_16h)

        row["gate2_up_reach_high"] = up_high
        row["gate2_dn_reach_high"] = dn_high

        row["gate2_up_impulse_8h_2atr"] = up_impulse_8h_2atr
        row["gate2_dn_impulse_8h_2atr"] = dn_impulse_8h_2atr

        row["first_up_high_ts"] = first_up_high_ts
        row["first_dn_high_ts"] = first_dn_high_ts


        row["high_first_side"] = (
            "UP"
            if pd.notna(first_up_high_ts) and (pd.isna(first_dn_high_ts) or first_up_high_ts < first_dn_high_ts)
            else "DN"
            if pd.notna(first_dn_high_ts) and (pd.isna(first_up_high_ts) or first_dn_high_ts < first_up_high_ts)
            else "TIE_OR_NONE"
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values("signal_ts").reset_index(drop=True)

# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUT_REACH_DIR)
    ensure_dir(OUT_STRENGTH_DIR)

    src_files = sorted(Path(SRC_GATE2_DIR).glob("*.parquet"))

    all_reach = []
    all_strength = []
    audit_rows = []

    all_feature_candidates = set()
    symbol_feature_map = {}

    for path in src_files:
        symbol = path.stem

        audit = {
            "symbol": symbol,
            "src_rows": 0,
            "rows_after_dedupe": 0,
            "dup_rows_removed": 0,
            "rows_final": 0,
            "valid_rows": 0,
            "status": "init",
        }

        try:
            df_src = pd.read_parquet(path)
            audit["src_rows"] = int(len(df_src))

            df_src, dup_rows_removed = dedupe_source_gate2(df_src)
            audit["rows_after_dedupe"] = int(len(df_src))
            audit["dup_rows_removed"] = int(dup_rows_removed)

            if len(df_src) == 0:
                audit["status"] = "empty_after_dedupe"
                audit_rows.append(audit)
                continue

            h4 = load_h4(symbol)
            if h4 is None:
                audit["status"] = "missing_h4"
                audit_rows.append(audit)
                continue

            m1 = load_m1(symbol)
            if m1 is None:
                audit["status"] = "missing_m1"
                audit_rows.append(audit)
                continue

            safe_source_cols = select_safe_source_cols(df_src)
            df_base = df_src[safe_source_cols].copy()

            labeled = compute_directional_labels_for_symbol(df_base, h4, m1)

            # === MAIN DIRECTION TARGET ===
            if len(labeled) == 0:
                audit["status"] = "empty_after_labeling"
                audit_rows.append(audit)
                continue

            meta_path = os.path.join(OLD_GATE2_MODELS_DIR, symbol, "gate2", "meta.json")
            valid_start_ts = infer_valid_start_ts_from_meta(
                meta=load_json_safe(meta_path),
                ts_series=labeled["signal_ts"],
                fallback_tail_share=VALID_TAIL_SHARE_FALLBACK,
            )

            labeled["upstream_valid_start_ts"] = valid_start_ts
            if valid_start_ts is not None:
                labeled["upstream_split"] = np.where(labeled["signal_ts"] >= valid_start_ts, "valid", "train")
            else:
                labeled["upstream_split"] = ""

            current_features = []

            for c in labeled.columns:
                if c in META_COLS_COMMON:
                    continue
                if c in REACH_TARGET_COLS:
                    continue
                if c in STRENGTH_TARGET_COLS:
                    continue

                if str(c).endswith("_feat"):
                    continue
                if str(c).startswith("sym_"):
                    continue
                if c in ("symbol_id", "symbol_id_feat", "side_num", "side_num_feat"):
                    continue

                if c in {
                    "entry_px",

                    # === FUTURE / LABEL ===
                    "mfe_up_atr_8h",
                    "mfe_dn_atr_8h",
                    "mfe_up_atr_h",
                    "mfe_dn_atr_h",

                    "impulse_010_first_side_8h",
                    "impulse_015_first_side_8h",

                    "mfe_up_pct_8h",
                    "mfe_dn_pct_8h",
                    "mfe_up_pct_16h",
                    "mfe_dn_pct_16h",

                    "mae_up_pct_8h",
                    "mae_dn_pct_8h",
                    "mae_up_pct_16h",
                    "mae_dn_pct_16h",

                    # === FIRST HIT TIMES ===
                    "first_up_high_ts",
                    "first_dn_high_ts",

                    # === FIRST SIDE ===
                    "high_first_side",

                    # === TARGETS (ЧТО МОДЕЛЬ ПРЕДСКАЗЫВАЕТ) ===

                    # === SERVICE FUTURE ===
                    "impulse_hours",
                }:
                    continue

                current_features.append(c)

            all_feature_candidates.update(current_features)
            symbol_feature_map[symbol] = current_features

            print(f"[DEBUG] {symbol} raw_feature_count:", len(current_features))

            reach_cols = [
                             c for c in META_COLS_COMMON if c in labeled.columns
                         ] + [
                             c for c in current_features if c in labeled.columns
                         ]

            strength_cols = [
                                c for c in META_COLS_COMMON if c in labeled.columns
                            ] + [
                                c for c in current_features if c in labeled.columns
                            ]

            df_reach = labeled[
                [c for c in META_COLS_COMMON if c in labeled.columns] +
                REACH_TARGET_COLS +
                [c for c in current_features if c in labeled.columns]
                ].copy()
            df_strength = labeled[
                [c for c in META_COLS_COMMON if c in labeled.columns] +
                STRENGTH_TARGET_COLS +
                [c for c in current_features if c in labeled.columns]
                ].copy()
            out_reach_path = os.path.join(OUT_REACH_DIR, f"{symbol}.parquet")
            out_strength_path = os.path.join(OUT_STRENGTH_DIR, f"{symbol}.parquet")

            df_reach.to_parquet(out_reach_path, index=False)
            df_strength.to_parquet(out_strength_path, index=False)

            all_reach.append(df_reach)
            all_strength.append(df_strength)

            audit["rows_final"] = int(len(labeled))
            audit["valid_rows"] = int((labeled["upstream_split"] == "valid").sum())
            audit["reach_cols"] = int(len(df_reach.columns))
            audit["strength_cols"] = int(len(df_strength.columns))
            audit["status"] = "ok"
            audit_rows.append(audit)

            print("WROTE", out_reach_path)
            print("WROTE", out_strength_path)

        except Exception as e:
            audit["status"] = f"error:{type(e).__name__}"
            audit["error_text"] = str(e)
            audit_rows.append(audit)
            print("ERROR", symbol, repr(e))
    GLOBAL_FEATURE_COLS = sorted(all_feature_candidates)
    for sym, feats in symbol_feature_map.items():
        missing = set(GLOBAL_FEATURE_COLS) - set(feats)
        extra = set(feats) - set(GLOBAL_FEATURE_COLS)

        if missing or extra:
            print(f"[SCHEMA DEBUG] {sym} missing={len(missing)} extra={len(extra)}")
    print("GLOBAL FEATURE COLS:", len(GLOBAL_FEATURE_COLS))

    reach_all = pd.concat(all_reach, ignore_index=True) if all_reach else pd.DataFrame()
    strength_all = pd.concat(all_strength, ignore_index=True) if all_strength else pd.DataFrame()

    # === GLOBAL ALIGN (ВАЖНО ДЛЯ GATE2) ===


    def force_global_schema(dfs, global_cols):
        out = []
        for df in dfs:
            for c in global_cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[global_cols + [c for c in df.columns if c not in global_cols]]
            out.append(df)
        return out

    all_reach = force_global_schema(all_reach, GLOBAL_FEATURE_COLS)
    all_strength = force_global_schema(all_strength, GLOBAL_FEATURE_COLS)

    reach_all = pd.concat(all_reach, ignore_index=True) if all_reach else pd.DataFrame()
    strength_all = pd.concat(all_strength, ignore_index=True) if all_strength else pd.DataFrame()

    audit_df = pd.DataFrame(audit_rows).sort_values(["status", "symbol"]).reset_index(drop=True)

    reach_all.to_parquet(OUT_REACH_ALL, index=False)
    strength_all.to_parquet(OUT_STRENGTH_ALL, index=False)
    audit_df.to_csv(OUT_AUDIT_CSV, index=False)

    report = {
        "source_gate2_dir": SRC_GATE2_DIR,
        "builder_variant": "5features",
        "rows_reach_all": int(len(reach_all)),
        "rows_strength_all": int(len(strength_all)),
        "symbols_total": int(len(src_files)),
        "symbols_ok": int((audit_df["status"] == "ok").sum()) if len(audit_df) else 0,
        "h_hours": int(H_HOURS),
        "entry_delay_seconds": int(ENTRY_DELAY_SECONDS),
        "reach_high_atr": float(REACH_HIGH_ATR),
                "target_definitions": {
            "gate2_up_reach_high": f"1 if mfe_up_atr_h >= {REACH_HIGH_ATR}",
            "gate2_dn_reach_high": f"1 if mfe_dn_atr_h >= {REACH_HIGH_ATR}",

            "gate2_up_impulse_8h_2atr": f"1 if mfe_up_atr_8h >= {IMPULSE_ATR_MAIN}",
            "gate2_dn_impulse_8h_2atr": f"1 if mfe_dn_atr_8h >= {IMPULSE_ATR_MAIN}",

            "mfe_up_atr_8h": "max favorable excursion upward from real executable entry during first 8h, in ATR units",
            "mfe_dn_atr_8h": "max favorable excursion downward from real executable entry during first 8h, in ATR units",
            "mfe_up_atr_h": "max favorable excursion upward from real executable entry during full horizon, in ATR units",
            "mfe_dn_atr_h": "max favorable excursion downward from real executable entry during full horizon, in ATR units",

            "mfe_up_pct_8h": "max favorable excursion upward from real executable entry during first 8h, in percent of entry price",
            "mfe_dn_pct_8h": "max favorable excursion downward from real executable entry during first 8h, in percent of entry price",
            "mfe_up_pct_16h": "max favorable excursion upward from real executable entry during full horizon, in percent of entry price",
            "mfe_dn_pct_16h": "max favorable excursion downward from real executable entry during full horizon, in percent of entry price",

            "mae_up_pct_8h": "maximum adverse excursion against long from real executable entry during first 8h, in percent of entry price",
            "mae_dn_pct_8h": "maximum adverse excursion against short from real executable entry during first 8h, in percent of entry price",
            "mae_up_pct_16h": "maximum adverse excursion against long from real executable entry during full horizon, in percent of entry price",
            "mae_dn_pct_16h": "maximum adverse excursion against short from real executable entry during full horizon, in percent of entry price",
        },
        "files": {
            "reach_by_symbol_dir": OUT_REACH_DIR,
            "strength_by_symbol_dir": OUT_STRENGTH_DIR,
            "reach_all": OUT_REACH_ALL,
            "strength_all": OUT_STRENGTH_ALL,
            "audit_csv": OUT_AUDIT_CSV,
        },
    }

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("WROTE", OUT_REACH_ALL)
    print("WROTE", OUT_STRENGTH_ALL)
    print("WROTE", OUT_AUDIT_CSV)
    print("WROTE", OUT_REPORT_JSON)
    print()

    if len(reach_all):
        print("REACH ALL SHAPE:", reach_all.shape)
        print("REACH TARGETS")
        for c in [
            "gate2_up_reach_high",
            "gate2_dn_reach_high",
            "gate2_up_impulse_8h_2atr",
            "gate2_dn_impulse_8h_2atr",
        ]:
            if c in reach_all.columns:
                print(c)
                print(reach_all[c].value_counts(dropna=False).to_string())
                print()

    if len(strength_all):
        print("STRENGTH ALL SHAPE:", strength_all.shape)

        print("MFE ATR SUMMARY")
        print(
            strength_all[["mfe_up_atr_8h", "mfe_dn_atr_8h", "mfe_up_atr_h", "mfe_dn_atr_h"]]
            .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .to_string()
        )
        print()

        print("MFE PCT SUMMARY")
        print(
            strength_all[["mfe_up_pct_8h", "mfe_dn_pct_8h", "mfe_up_pct_16h", "mfe_dn_pct_16h"]]
            .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .to_string()
        )
        print()

        print("MAE PCT SUMMARY")
        print(
            strength_all[["mae_up_pct_8h", "mae_dn_pct_8h", "mae_up_pct_16h", "mae_dn_pct_16h"]]
            .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .to_string()
        )
        print()

    if len(audit_df):
        print("AUDIT STATUS")
        print(audit_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()