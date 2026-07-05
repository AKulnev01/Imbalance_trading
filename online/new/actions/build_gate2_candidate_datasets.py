from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

SRC_GATE1_DIR = ROOT / "production" / "dataset" / "gate1"
H4_DIR = ROOT / "data" / "h4_3"
M1_DIR = ROOT / "data" / "m1_4"

OUT_BASE_DIR = ROOT / "production" / "dataset" / "gate2_candidates"
BACKUP_ROOT = ROOT / "online" / "new" / "actions" / "_artifact_backups"

H_HOURS = 16
IMPULSE_HOURS = 8
ENTRY_DELAY_SECONDS = 90

REACH_HIGH_ATR = 1.00
IMPULSE_ATR_MAIN = 2.0

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
    "entry_px",
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
    "upstream_train_end_ts",
    "upstream_valid_start_ts",
    "upstream_valid_end_ts",
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def backup_existing_artifact(path: Path, artifact_group: str, symbol: str, report: Optional[Dict[str, Any]] = None) -> str:
    p = Path(path)

    if not p.exists():
        return ""

    symbol_safe = str(symbol).upper().replace("/", "_").replace("\\", "_")
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")

    backup_dir = BACKUP_ROOT / str(artifact_group) / symbol_safe / run_id
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_path = backup_dir / p.name

    suffix_i = 1
    while backup_path.exists():
        backup_path = backup_dir / "{}.dup{}{}".format(p.stem, suffix_i, p.suffix)
        suffix_i += 1

    shutil.copy2(str(p), str(backup_path))

    if report is not None:
        report.setdefault("backup_paths", []).append(str(backup_path))

    print("    BACKUP_ARTIFACT:", p, "->", backup_path, flush=True)
    return str(backup_path)


def parse_required_ts(value: str, name: str) -> pd.Timestamp:
    ts = pd.to_datetime(str(value), utc=True, errors="coerce")
    if pd.isna(ts):
        raise RuntimeError("bad {}: {}".format(name, value))
    return pd.Timestamp(ts).tz_convert(None)


def parse_optional_ts(value: str, name: str) -> Optional[pd.Timestamp]:
    raw = str(value or "").strip()
    if not raw:
        return None

    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        raise RuntimeError("bad {}: {}".format(name, value))

    return pd.Timestamp(ts).tz_convert(None)


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
        raise RuntimeError("--valid-start and --valid-end must be provided together")

    if train_end > valid_start:
        raise RuntimeError(
            "--train-end must be <= --valid-start, got train_end={} valid_start={}".format(
                train_end,
                valid_start,
            )
        )

    if valid_start >= valid_end:
        raise RuntimeError(
            "--valid-start must be < --valid-end, got valid_start={} valid_end={}".format(
                valid_start,
                valid_end,
            )
        )

    return valid_start, valid_end


def apply_upstream_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    out = df.copy()
    out["signal_ts"] = pd.to_datetime(out["signal_ts"], errors="coerce")
    out["upstream_train_end_ts"] = train_end
    out["upstream_valid_start_ts"] = valid_start
    out["upstream_valid_end_ts"] = valid_end

    train_mask = out["signal_ts"] < train_end
    valid_mask = out["signal_ts"] >= valid_start

    if valid_end is not None:
        valid_mask = valid_mask & (out["signal_ts"] < valid_end)

    out["upstream_split"] = np.select(
        [train_mask, valid_mask],
        ["train", "valid"],
        default="gap",
    )

    return out


def parse_symbols(raw_values: List[str]) -> List[str]:
    out: List[str] = []

    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            sym = part.strip().upper()
            if sym:
                out.append(sym)

    if not out:
        raise RuntimeError("no symbols provided")

    if len(out) == 1 and out[0] in {"ALL", "*"}:
        files = sorted(p for p in SRC_GATE1_DIR.glob("*.parquet") if not p.name.startswith("_"))
        return [p.stem.upper() for p in files]

    seen = set()
    unique: List[str] = []

    for sym in out:
        if sym not in seen:
            unique.append(sym)
            seen.add(sym)

    return unique


def to_naive_utc_auto(x: Any) -> pd.Series:
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


def find_first_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def load_h4(symbol: str) -> Optional[pd.DataFrame]:
    path = H4_DIR / "{}.parquet".format(symbol)
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    tcol = find_first_col(list(df.columns), H4_TIME_COL_CANDIDATES)
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
    path_plain = M1_DIR / "{}.parquet".format(symbol)
    path_m1 = M1_DIR / "{}_m1.parquet".format(symbol)

    if path_plain.exists():
        path = path_plain
    elif path_m1.exists():
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
        tcol = find_first_col(list(df.columns), M1_TIME_COL_CANDIDATES)
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


def true_range(df: pd.DataFrame) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev).abs(), (l - prev).abs()))
    return pd.Series(tr, index=df.index, dtype=float)


def atr14(df: pd.DataFrame) -> pd.Series:
    return true_range(df).rolling(14).mean()


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
        np.where(atr14_to_price <= q66, 1.0, 2.0),
    )
    out.loc[q33.isna() | q66.isna(), "vol_regime"] = np.nan

    return out


def first_m1_bar_at_or_after(m1: pd.DataFrame, ts_exec: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, pd.Series]]:
    idx = m1.index.searchsorted(ts_exec, side="left")
    if idx >= len(m1):
        return None
    return pd.Timestamp(m1.index[idx]), m1.iloc[idx]


def dedupe_source_gate2(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    dup_rows_removed = 0

    tcol = find_first_col(list(df.columns), TIME_COL_CANDIDATES)
    if tcol is None:
        raise RuntimeError("missing time column in gate2 source")

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


def select_safe_source_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []

    for c in df.columns:
        if c in DROP_COLS_FROM_SOURCE:
            continue

        if any(str(c).startswith(p) for p in DROP_PREFIXES_FROM_SOURCE):
            continue

        if str(c).endswith("_feat"):
            continue

        if str(c) in ("symbol_id_feat", "side_num_feat"):
            continue

        cols.append(c)

    if "entry_ts" not in cols:
        cols.append("entry_ts")

    return cols


def compute_directional_labels_for_symbol(base_df: pd.DataFrame, h4: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(compute_directional_labels_for_symbol, "_ref_cache"):
        compute_directional_labels_for_symbol._ref_cache = {}

    ref_cache = compute_directional_labels_for_symbol._ref_cache

    def get_ref_h4(sym: str) -> Optional[pd.DataFrame]:
        if sym not in ref_cache:
            ref_cache[sym] = load_h4(sym)
        return ref_cache[sym]

    ref_btc_h4 = get_ref_h4("BTCUSDT")
    ref_eth_h4 = get_ref_h4("ETHUSDT")

    h4 = h4.copy()
    h4["atr14"] = atr14(h4)
    h4 = add_gate2_5features(h4)

    ts_to_pos = {pd.Timestamp(ts): i for i, ts in enumerate(h4["ts"])}
    rows: List[Dict[str, Any]] = []

    for _, r in base_df.iterrows():
        signal_ts = pd.Timestamp(r["entry_ts"])

        if signal_ts not in ts_to_pos:
            continue

        i = ts_to_pos[signal_ts]

        if i + 1 >= len(h4):
            continue

        atr = float(h4.iloc[i]["atr14"])
        if not np.isfinite(atr) or atr <= 0.0:
            continue

        entry_bar_open_ts = pd.Timestamp(h4.iloc[i + 1]["ts"])
        entry_ts_exec = entry_bar_open_ts + pd.Timedelta(seconds=ENTRY_DELAY_SECONDS)

        entry_bar = first_m1_bar_at_or_after(m1, entry_ts_exec)
        if entry_bar is None:
            continue

        real_entry_ts, entry_bar_row = entry_bar
        entry_px = float(entry_bar_row["open"])

        if not np.isfinite(entry_px) or entry_px <= 0.0:
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

        for col in ["atr4h", "ret_l1", "ret_l2", "hammer_like", "vol_regime"]:
            val = h4.iloc[i][col]
            row[col] = float(val) if np.isfinite(val) else np.nan

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


def build_current_features(labeled: pd.DataFrame) -> List[str]:
    current_features: List[str] = []

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
            "first_up_high_ts",
            "first_dn_high_ts",
            "high_first_side",
            "impulse_hours",
        }:
            continue

        current_features.append(c)

    return current_features


def force_global_schema(dfs: List[pd.DataFrame], global_cols: List[str]) -> List[pd.DataFrame]:
    out: List[pd.DataFrame] = []

    for df in dfs:
        x = df.copy()

        for c in global_cols:
            if c not in x.columns:
                x[c] = np.nan

        x = x[global_cols + [c for c in x.columns if c not in global_cols]]
        out.append(x)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Gate2 candidate datasets. Writes to "
            "production/dataset/gate2_candidates/<dataset_tag> and never overwrites prod Gate2 datasets."
        )
    )
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--symbols", nargs="+", default=["ALL"])
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--valid-start", default="")
    parser.add_argument("--valid-end", default="")
    parser.add_argument("--src-gate1-dir", default="")
    parser.add_argument("--m1-dir", default="")
    parser.add_argument("--h4-dir", default="")
    parser.add_argument("--out-base-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started_at = time.time()

    global SRC_GATE1_DIR
    global M1_DIR
    global H4_DIR
    global OUT_BASE_DIR

    if str(args.src_gate1_dir).strip():
        src_gate1_dir_arg = Path(str(args.src_gate1_dir).strip())
        SRC_GATE1_DIR = src_gate1_dir_arg if src_gate1_dir_arg.is_absolute() else ROOT / src_gate1_dir_arg

    if str(args.m1_dir).strip():
        m1_dir_arg = Path(str(args.m1_dir).strip())
        M1_DIR = m1_dir_arg if m1_dir_arg.is_absolute() else ROOT / m1_dir_arg

    if str(args.h4_dir).strip():
        h4_dir_arg = Path(str(args.h4_dir).strip())
        H4_DIR = h4_dir_arg if h4_dir_arg.is_absolute() else ROOT / h4_dir_arg

    if str(args.out_base_dir).strip():
        out_base_dir_arg = Path(str(args.out_base_dir).strip())
        OUT_BASE_DIR = out_base_dir_arg if out_base_dir_arg.is_absolute() else ROOT / out_base_dir_arg

    dataset_tag = str(args.dataset_tag).strip()
    if not dataset_tag:
        raise RuntimeError("--dataset-tag is empty")

    train_end = parse_train_end(args.train_end)
    valid_start_arg = parse_optional_ts(args.valid_start, "--valid-start")
    valid_end_arg = parse_optional_ts(args.valid_end, "--valid-end")
    valid_start, valid_end = validate_split_window(
        train_end=train_end,
        valid_start=valid_start_arg,
        valid_end=valid_end_arg,
    )
    symbols = parse_symbols(args.symbols)

    out_root = OUT_BASE_DIR / dataset_tag
    out_reach_dir = out_root / "reach_by_symbol"
    out_strength_dir = out_root / "strength_by_symbol"

    out_reach_all = out_root / "final_gate2_2_directional_reach_5features_all.parquet"
    out_strength_all = out_root / "final_gate2_3_directional_strength_5features_all.parquet"
    out_audit_csv = out_root / "final_gate2_directional_5features__AUDIT.csv"
    out_report_json = out_root / "final_gate2_directional_5features__REPORT.json"

    if out_root.exists() and any(out_root.rglob("*")) and not bool(args.overwrite):
        raise RuntimeError("candidate dataset already exists; pass --overwrite: {}".format(out_root))

    ensure_dir(out_reach_dir)
    ensure_dir(out_strength_dir)

    print("Build Gate2 Candidate Datasets")
    print("ROOT:", ROOT)
    print("SRC_GATE1_DIR:", SRC_GATE1_DIR)
    print("H4_DIR:", H4_DIR)
    print("M1_DIR:", M1_DIR)
    print("OUT_BASE_DIR:", OUT_BASE_DIR)
    print("DATASET_TAG:", dataset_tag)
    print("OUT_ROOT:", out_root)
    print("SYMBOLS:", len(symbols))
    print("TRAIN_END:", train_end)
    print("VALID_START:", valid_start)
    print("VALID_END:", valid_end)
    print("OVERWRITE:", bool(args.overwrite))
    print("=" * 120)

    all_reach: List[pd.DataFrame] = []
    all_strength: List[pd.DataFrame] = []
    audit_rows: List[Dict[str, Any]] = []

    all_feature_candidates = set()
    symbol_feature_map: Dict[str, List[str]] = {}

    for idx, symbol in enumerate(symbols, start=1):
        path = SRC_GATE1_DIR / "{}.parquet".format(symbol)

        audit: Dict[str, Any] = {
            "symbol": symbol,
            "src_path": str(path),
            "src_rows": 0,
            "rows_after_dedupe": 0,
            "dup_rows_removed": 0,
            "rows_final": 0,
            "valid_rows": 0,
            "status": "init",
            "error_text": "",
        }

        print("[{}/{}] {}".format(idx, len(symbols), symbol), flush=True)

        try:
            if not path.exists():
                audit["status"] = "missing_gate1_dataset"
                audit_rows.append(audit)
                print("    SKIP missing_gate1_dataset")
                continue

            df_src = pd.read_parquet(path)
            audit["src_rows"] = int(len(df_src))

            df_src, dup_rows_removed = dedupe_source_gate2(df_src)
            audit["rows_after_dedupe"] = int(len(df_src))
            audit["dup_rows_removed"] = int(dup_rows_removed)

            if len(df_src) == 0:
                audit["status"] = "empty_after_dedupe"
                audit_rows.append(audit)
                print("    SKIP empty_after_dedupe")
                continue

            h4 = load_h4(symbol)
            if h4 is None:
                audit["status"] = "missing_h4"
                audit_rows.append(audit)
                print("    SKIP missing_h4")
                continue

            m1 = load_m1(symbol)
            if m1 is None:
                audit["status"] = "missing_m1"
                audit_rows.append(audit)
                print("    SKIP missing_m1")
                continue

            safe_source_cols = select_safe_source_cols(df_src)
            df_base = df_src[safe_source_cols].copy()

            labeled = compute_directional_labels_for_symbol(df_base, h4, m1)

            if len(labeled) == 0:
                audit["status"] = "empty_after_labeling"
                audit_rows.append(audit)
                print("    SKIP empty_after_labeling")
                continue

            labeled = apply_upstream_split(
                df=labeled,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )

            current_features = build_current_features(labeled)

            all_feature_candidates.update(current_features)
            symbol_feature_map[symbol] = current_features

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

            out_reach_path = out_reach_dir / "{}.parquet".format(symbol)
            out_strength_path = out_strength_dir / "{}.parquet".format(symbol)

            if bool(args.overwrite):
                backup_existing_artifact(out_reach_path, "gate2_candidate_reach_dataset", symbol, audit)
                backup_existing_artifact(out_strength_path, "gate2_candidate_strength_dataset", symbol, audit)

            df_reach.to_parquet(out_reach_path, index=False)
            df_strength.to_parquet(out_strength_path, index=False)

            all_reach.append(df_reach)
            all_strength.append(df_strength)

            audit["rows_final"] = int(len(labeled))
            audit["train_end"] = str(train_end)
            audit["valid_start"] = str(valid_start)
            audit["valid_end"] = "" if valid_end is None else str(valid_end)
            audit["train_rows"] = int((labeled["upstream_split"] == "train").sum())
            audit["valid_rows"] = int((labeled["upstream_split"] == "valid").sum())
            audit["gap_rows"] = int((labeled["upstream_split"] == "gap").sum())
            audit["reach_cols"] = int(len(df_reach.columns))
            audit["strength_cols"] = int(len(df_strength.columns))
            audit["status"] = "ok"

            audit_rows.append(audit)

            print("    OK rows={} valid={} reach_cols={} strength_cols={}".format(
                audit["rows_final"],
                audit["valid_rows"],
                audit["reach_cols"],
                audit["strength_cols"],
            ))

        except Exception as exc:
            audit["status"] = "error:{}".format(type(exc).__name__)
            audit["error_text"] = str(exc)
            audit_rows.append(audit)
            print("    ERR:", type(exc).__name__, exc)

    global_feature_cols = sorted(all_feature_candidates)

    all_reach = force_global_schema(all_reach, global_feature_cols)
    all_strength = force_global_schema(all_strength, global_feature_cols)

    reach_all = pd.concat(all_reach, ignore_index=True) if all_reach else pd.DataFrame()
    strength_all = pd.concat(all_strength, ignore_index=True) if all_strength else pd.DataFrame()

    audit_df = pd.DataFrame(audit_rows)
    if len(audit_df):
        audit_df = audit_df.sort_values(["status", "symbol"]).reset_index(drop=True)

    if bool(args.overwrite):
        backup_existing_artifact(out_reach_all, "gate2_candidate_reach_all", dataset_tag)
        backup_existing_artifact(out_strength_all, "gate2_candidate_strength_all", dataset_tag)
        backup_existing_artifact(out_audit_csv, "gate2_candidate_audit", dataset_tag)
        backup_existing_artifact(out_report_json, "gate2_candidate_report", dataset_tag)

    reach_all.to_parquet(out_reach_all, index=False)
    strength_all.to_parquet(out_strength_all, index=False)
    audit_df.to_csv(out_audit_csv, index=False)

    report = {
        "created_at_utc": str(pd.Timestamp.now(tz="UTC")),
        "dataset_tag": dataset_tag,
        "source_gate1_dir": str(SRC_GATE1_DIR),
        "m1_dir": str(M1_DIR),
        "h4_dir": str(H4_DIR),
        "out_base_dir": str(OUT_BASE_DIR),
        "out_root": str(out_root),
        "train_end": str(train_end),
        "valid_start": str(valid_start),
        "valid_end": "" if valid_end is None else str(valid_end),
        "split": {
            "type": "fixed_time",
            "train_condition": "signal_ts < train_end",
            "valid_condition": "valid_start <= signal_ts < valid_end" if valid_end is not None else "signal_ts >= valid_start",
            "gap_condition": "otherwise",
        },
        "builder_variant": "5features_candidate",
        "rows_reach_all": int(len(reach_all)),
        "rows_strength_all": int(len(strength_all)),
        "symbols_total": int(len(symbols)),
        "symbols_ok": int((audit_df["status"] == "ok").sum()) if len(audit_df) else 0,
        "h_hours": int(H_HOURS),
        "impulse_hours": int(IMPULSE_HOURS),
        "entry_delay_seconds": int(ENTRY_DELAY_SECONDS),
        "reach_high_atr": float(REACH_HIGH_ATR),
        "impulse_atr_main": float(IMPULSE_ATR_MAIN),
        "global_feature_cols_count": int(len(global_feature_cols)),
        "global_feature_cols": global_feature_cols,
        "files": {
            "reach_by_symbol_dir": str(out_reach_dir),
            "strength_by_symbol_dir": str(out_strength_dir),
            "reach_all": str(out_reach_all),
            "strength_all": str(out_strength_all),
            "audit_csv": str(out_audit_csv),
            "report_json": str(out_report_json),
        },
        "elapsed_sec": round(time.time() - started_at, 3),
    }

    out_report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("=" * 120)
    print("DONE")
    print("OUT_ROOT:", out_root)
    print("REACH_ALL_SHAPE:", reach_all.shape)
    print("STRENGTH_ALL_SHAPE:", strength_all.shape)
    print("AUDIT_STATUS:")
    if len(audit_df):
        print(audit_df["status"].value_counts(dropna=False).to_string())
    print("WROTE:", out_reach_all)
    print("WROTE:", out_strength_all)
    print("WROTE:", out_audit_csv)
    print("WROTE:", out_report_json)

    if not len(reach_all):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
