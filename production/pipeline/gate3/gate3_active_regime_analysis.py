import os
import glob
import argparse
import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================

#python production/pipeline/gate3_recearch/gate3_active_regime_analysis.py

DATA_DIR = "production/dataset/pa_gate3_v3_long_short_by_symbol"
H4_DIR = "data/h4_3"

OUT_DIR = "production/models/gate3_research"

OUT_EDGE = f"{OUT_DIR}/_ACTIVE_EDGE_full_timeline.csv"
OUT_OVERLAP = f"{OUT_DIR}/_ACTIVE_OVERLAP_stats.csv"
OUT_REGIME = f"{OUT_DIR}/_ACTIVE_EDGE_regime_segmentation.csv"

MAX_FWD = 30
MIN_PATTERN_N = 10
TARGET_HIT_ATR = 0.8



REGIME_PATTERN_SPECS = [
    {
        "pattern_col": "active_pa_atr_squeeze_break_up",
        "pattern_name": "squeeze_up",
        "side": "long",
        "hit_col": "p_hit_up",
    },
    {
        "pattern_col": "active_pa_bos_up_24",
        "pattern_name": "bos_up",
        "side": "long",
        "hit_col": "p_hit_up",
    },
    {
        "pattern_col": "active_pa_atr_squeeze_break_dn",
        "pattern_name": "squeeze_dn",
        "side": "short",
        "hit_col": "p_hit_dn",
    },
    {
        "pattern_col": "active_pa_bos_dn_24",
        "pattern_name": "bos_dn",
        "side": "short",
        "hit_col": "p_hit_dn",
    },
]



# =========================
# RUNTIME / SPLIT CONFIG
# =========================

def parse_optional_ts(raw, name):
    value = str(raw or "").strip()

    if not value:
        return None

    ts = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(ts):
        raise SystemExit("bad {} value: {}".format(name, raw))

    out = pd.Timestamp(ts)

    if out.tzinfo is not None:
        out = out.tz_convert("UTC").tz_localize(None)

    return pd.Timestamp(out).floor("min")


def parse_runtime_args():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--h4-dir", default=H4_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--max-fwd", type=int, default=MAX_FWD)
    parser.add_argument("--min-pattern-n", type=int, default=MIN_PATTERN_N)
    parser.add_argument("--target-hit-atr", type=float, default=TARGET_HIT_ATR)

    parser.add_argument("--train-end", default=os.environ.get("IMB_OFFLINE_TRAIN_END", ""))
    parser.add_argument("--valid-start", default=os.environ.get("IMB_OFFLINE_VALID_START", ""))
    parser.add_argument("--valid-end", default=os.environ.get("IMB_OFFLINE_VALID_END", ""))

    args, _ = parser.parse_known_args()
    return args


def apply_runtime_args(args):
    global DATA_DIR
    global H4_DIR
    global OUT_DIR
    global MAX_FWD
    global MIN_PATTERN_N
    global TARGET_HIT_ATR

    DATA_DIR = str(args.data_dir)
    H4_DIR = str(args.h4_dir)
    OUT_DIR = str(args.out_dir)
    MAX_FWD = int(args.max_fwd)
    MIN_PATTERN_N = int(args.min_pattern_n)
    TARGET_HIT_ATR = float(args.target_hit_atr)


def refresh_output_paths():
    global OUT_EDGE
    global OUT_OVERLAP
    global OUT_REGIME

    OUT_EDGE = f"{OUT_DIR}/_ACTIVE_EDGE_full_timeline.csv"
    OUT_OVERLAP = f"{OUT_DIR}/_ACTIVE_OVERLAP_stats.csv"
    OUT_REGIME = f"{OUT_DIR}/_ACTIVE_EDGE_regime_segmentation.csv"


def build_split_config(args):
    train_end = parse_optional_ts(args.train_end, "--train-end")
    valid_start = parse_optional_ts(args.valid_start, "--valid-start")
    valid_end = parse_optional_ts(args.valid_end, "--valid-end")

    provided = [train_end is not None, valid_start is not None, valid_end is not None]

    if any(provided) and not all(provided):
        raise SystemExit(
            "split args must be provided together: --train-end --valid-start --valid-end"
        )

    if train_end is None:
        return {
            "mode": "legacy_full_timeline",
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

    train_safe_cutoff = train_end - pd.Timedelta(hours=4 * int(MAX_FWD))

    return {
        "mode": "fixed_train_safe",
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "train_safe_cutoff": train_safe_cutoff,
    }


def split_config_for_json(split_config):
    return {
        "mode": str(split_config.get("mode")),
        "train_end": None if split_config.get("train_end") is None else str(split_config.get("train_end")),
        "valid_start": None if split_config.get("valid_start") is None else str(split_config.get("valid_start")),
        "valid_end": None if split_config.get("valid_end") is None else str(split_config.get("valid_end")),
        "train_safe_cutoff": None if split_config.get("train_safe_cutoff") is None else str(split_config.get("train_safe_cutoff")),
        "safe_rule": "entry_ts < train_end - MAX_FWD * 4h" if split_config.get("mode") == "fixed_train_safe" else "full timeline",
    }


def build_train_safe_mask(ts_series, split_config, max_fwd):
    ts = pd.to_datetime(ts_series, errors="coerce")

    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    if split_config.get("mode") != "fixed_train_safe":
        return np.ones(len(ts), dtype=bool)

    cutoff = pd.Timestamp(split_config["train_safe_cutoff"])

    return (ts < cutoff).fillna(False).to_numpy(dtype=bool)



# =========================
# UTILS
# =========================
def to_naive_ts(s):
    x = pd.to_datetime(s, utc=True, errors="coerce")
    return pd.Series(x).dt.tz_localize(None)

def ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def atr14(high, low, close):
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    return atr


def sliding_windows(arr, w):
    try:
        return np.lib.stride_tricks.sliding_window_view(arr, w)
    except Exception:
        return None


# =========================
# METRICS
# =========================

def compute_metrics(h4, mask):
    h4 = h4.sort_values("ts").reset_index(drop=True)

    o = h4["open"].to_numpy(dtype=float)
    h = h4["high"].to_numpy(dtype=float)
    l = h4["low"].to_numpy(dtype=float)
    c = h4["close"].to_numpy(dtype=float)

    atr = atr14(h, l, c)

    n = len(h4)
    if n < MAX_FWD + 20:
        return None

    hs = h[1:]
    ls = l[1:]

    hwin = sliding_windows(hs, MAX_FWD)
    lwin = sliding_windows(ls, MAX_FWD)

    if hwin is None or lwin is None:
        return None

    max_len = len(hwin)

    ep = np.where(mask)[0]
    ep = ep[(ep >= 0) & (ep < max_len)]

    if len(ep) < MIN_PATTERN_N:
        return None

    entry_px_ep = o[ep + 1]
    atr_ep = atr[ep]

    ok = (atr_ep > 0) & np.isfinite(atr_ep) & np.isfinite(entry_px_ep) & (entry_px_ep > 0)
    ep = ep[ok]
    entry_px_ep = entry_px_ep[ok]
    atr_ep = atr_ep[ok]

    if len(ep) < MIN_PATTERN_N:
        return None

    hw = hwin[ep]
    lw = lwin[ep]

    mfe_up = (np.max(hw, axis=1) - entry_px_ep) / atr_ep
    mfe_dn = (entry_px_ep - np.min(lw, axis=1)) / atr_ep

    up_hit = hw >= (entry_px_ep + TARGET_HIT_ATR * atr_ep)[:, None]
    dn_hit = lw <= (entry_px_ep - TARGET_HIT_ATR * atr_ep)[:, None]

    up_any = up_hit.any(axis=1)
    dn_any = dn_hit.any(axis=1)

    up_first = np.where(up_any, up_hit.argmax(axis=1) + 1, -1)
    dn_first = np.where(dn_any, dn_hit.argmax(axis=1) + 1, -1)

    return {
        "n": int(len(ep)),
        "mfe_up_med": float(np.nanmedian(mfe_up)),
        "mfe_dn_med": float(np.nanmedian(mfe_dn)),
        "p_hit_up": float(np.mean(up_first > 0)),
        "p_hit_dn": float(np.mean(dn_first > 0)),
        "tt_hit_up": float(np.nanmedian(up_first[up_first > 0])) if np.any(up_first > 0) else np.nan,
        "tt_hit_dn": float(np.nanmedian(dn_first[dn_first > 0])) if np.any(dn_first > 0) else np.nan,
    }


# =========================
# MAIN
# =========================

def main() -> None:
    args = parse_runtime_args()
    apply_runtime_args(args)
    refresh_output_paths()
    split_config = build_split_config(args)

    print("Gate3 Active Regime Analysis")
    print("DATA_DIR:", DATA_DIR)
    print("H4_DIR:", H4_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("MAX_FWD:", MAX_FWD)
    print("MIN_PATTERN_N:", MIN_PATTERN_N)
    print("TARGET_HIT_ATR:", TARGET_HIT_ATR)
    print("SPLIT_CONFIG:", split_config_for_json(split_config))
    print("=" * 120)

    files = sorted(glob.glob(f"{DATA_DIR}/*.parquet"))

    edge_rows = []
    regime_rows = []

    overlap_rows = []

    for fp in files:
        sym = os.path.basename(fp).replace(".parquet", "")

        h4_fp = f"{H4_DIR}/{sym}.parquet"
        if not os.path.exists(h4_fp):
            continue

        df = pd.read_parquet(fp)
        h4 = pd.read_parquet(h4_fp)
        if 'pattern_cols_global' not in globals():
            pattern_cols_global = sorted([
                c for c in df.columns
                if c.startswith("active_")
            ])
        pattern_cols = pattern_cols_global


        if "entry_ts" not in df.columns:
            print(f"SKIP {sym}: no entry_ts")
            continue

        if "ts" not in h4.columns:
            print(f"SKIP {sym}: no ts in h4")
            continue

        df["entry_ts"] = to_naive_ts(df["entry_ts"])
        h4["ts"] = to_naive_ts(h4["ts"])

        df = (
            df.dropna(subset=["entry_ts"])
            .sort_values("entry_ts")
            .drop_duplicates("entry_ts", keep="last")
            .reset_index(drop=True)
        )

        h4 = (
            h4.dropna(subset=["ts"])
            .sort_values("ts")
            .drop_duplicates("ts", keep="last")
            .reset_index(drop=True)
        )

        need_h4_cols = ["open", "high", "low", "close"]
        if any(c not in h4.columns for c in need_h4_cols):
            print(f"SKIP {sym}: bad h4 columns")
            continue

        ts_to_pos = {ts: i for i, ts in enumerate(h4["ts"])}


        split_allowed_mask = build_train_safe_mask(

            h4["ts"],

            split_config=split_config,

            max_fwd=int(MAX_FWD),

        )
        ep = df["entry_ts"].map(ts_to_pos).fillna(-1).astype(int).to_numpy()
        pattern_cols = [
            spec["pattern_col"]
            for spec in REGIME_PATTERN_SPECS
            if spec["pattern_col"] in df.columns
        ]

        active_masks = {}

        for col in pattern_cols:
            mask = np.zeros(len(h4), dtype=bool)

            if col in df.columns:
                m = (pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy() == 1) & (ep >= 0)
                mask[ep[m]] = True

            active_masks[col] = mask & split_allowed_mask
        overlap_rows.append({
            "symbol": sym,
            "active_up_sq_n": int(active_masks.get("active_pa_atr_squeeze_break_up", np.zeros(len(h4))).sum()),
            "active_up_bos_n": int(active_masks.get("active_pa_bos_up_24", np.zeros(len(h4))).sum()),
            "active_dn_sq_n": int(active_masks.get("active_pa_atr_squeeze_break_dn", np.zeros(len(h4))).sum()),
            "active_dn_bos_n": int(active_masks.get("active_pa_bos_dn_24", np.zeros(len(h4))).sum()),
            "overlap_up_percent": np.nan,
            "overlap_dn_percent": np.nan,
            "split_mode": str(split_config.get("mode")),
            "train_end": None if split_config.get("train_end") is None else str(split_config.get("train_end")),
            "train_safe_cutoff": None if split_config.get("train_safe_cutoff") is None else str(split_config.get("train_safe_cutoff")),
        })

        base_mask = split_allowed_mask.copy()
        base_metrics = compute_metrics(h4, base_mask)
        if base_metrics is None:
            continue

        for spec in REGIME_PATTERN_SPECS:
            pattern_col = spec["pattern_col"]

            if pattern_col not in active_masks:
                continue

            pattern_name = spec["pattern_name"]
            side = spec["side"]
            hit_col = spec["hit_col"]
            time_col = "tt_hit_up" if side == "long" else "tt_hit_dn"

            gate1_mask = np.zeros(len(h4), dtype=bool)

            if "gate1_pass" in df.columns:
                g1 = (pd.to_numeric(df["gate1_pass"], errors="coerce").fillna(0).to_numpy() == 1) & (ep >= 0)
                gate1_mask[ep[g1]] = True
            else:
                gate1_mask = np.ones(len(h4), dtype=bool)

            mask = active_masks[pattern_col] & gate1_mask

            metrics = compute_metrics(h4, mask)
            if metrics is None:
                continue

            if side == "long":
                lift_mfe = metrics["mfe_up_med"] - base_metrics["mfe_up_med"]
            else:
                lift_mfe = metrics["mfe_dn_med"] - base_metrics["mfe_dn_med"]

            base_time = base_metrics[time_col]
            pat_time = metrics[time_col]
            if pd.isna(base_time) or pd.isna(pat_time):
                lift_time = np.nan
            else:
                lift_time = float(base_time - pat_time)

            edge_rows.append({
                "symbol": sym,
                "split_mode": str(split_config.get("mode")),
                "train_end": None if split_config.get("train_end") is None else str(split_config.get("train_end")),
                "valid_start": None if split_config.get("valid_start") is None else str(split_config.get("valid_start")),
                "valid_end": None if split_config.get("valid_end") is None else str(split_config.get("valid_end")),
                "train_safe_cutoff": None if split_config.get("train_safe_cutoff") is None else str(split_config.get("train_safe_cutoff")),
                "pattern": pattern_col,
                "side": side,
                "lift_mfe": float(lift_mfe),
                "lift_p_hit": float(metrics[hit_col] - base_metrics[hit_col]),
                "k": int(round(metrics[hit_col] * metrics["n"])),
                "lift_time": lift_time,
                "n": int(metrics["n"]),
                "pattern_mfe_med": float(metrics["mfe_up_med"] if side == "long" else metrics["mfe_dn_med"]),
                "base_mfe_med": float(base_metrics["mfe_up_med"] if side == "long" else base_metrics["mfe_dn_med"]),
                "pattern_p_hit": float(metrics[hit_col]),
                "base_p_hit": float(base_metrics[hit_col]),
                "pattern_tt_hit": float(metrics[time_col]) if pd.notna(metrics[time_col]) else np.nan,
                "base_tt_hit": float(base_metrics[time_col]) if pd.notna(base_metrics[time_col]) else np.nan,
            })

        close = h4["close"].to_numpy(dtype=float)
        high = h4["high"].to_numpy(dtype=float)
        low = h4["low"].to_numpy(dtype=float)

        ema20 = ema(close, 20)
        ema50 = ema(close, 50)

        trend_up = ema20 > ema50

        atr = atr14(high, low, close)
        atr_thr = np.nanquantile(atr, 0.7)
        high_vol = atr >= atr_thr

        regimes = {
            "trend_up": trend_up,
            "trend_down": ~trend_up,
            "high_vol": high_vol,
            "low_vol": ~high_vol,
        }

        for spec in REGIME_PATTERN_SPECS:
            pattern_col = spec["pattern_col"]

            if pattern_col not in active_masks:
                continue

            side = spec["side"]
            hit_col = spec["hit_col"]
            pattern_name = spec["pattern_name"]

            for rname, rmask in regimes.items():
                gate1_mask = np.zeros(len(h4), dtype=bool)

                if "gate1_pass" in df.columns:
                    g1 = (pd.to_numeric(df["gate1_pass"], errors="coerce").fillna(0).to_numpy() == 1) & (ep >= 0)
                    gate1_mask[ep[g1]] = True
                else:
                    gate1_mask = np.ones(len(h4), dtype=bool)

                mask = active_masks[pattern_col] & gate1_mask & rmask
                m = compute_metrics(h4, mask)
                if m is None:
                    continue

                regime_rows.append({
                    "symbol": sym,
                    "split_mode": str(split_config.get("mode")),
                    "train_end": None if split_config.get("train_end") is None else str(split_config.get("train_end")),
                    "valid_start": None if split_config.get("valid_start") is None else str(split_config.get("valid_start")),
                    "valid_end": None if split_config.get("valid_end") is None else str(split_config.get("valid_end")),
                    "train_safe_cutoff": None if split_config.get("train_safe_cutoff") is None else str(split_config.get("train_safe_cutoff")),
                    "pattern": pattern_name,
                    "side": side,
                    "regime": rname,
                    "mfe": float(m["mfe_up_med"] if side == "long" else m["mfe_dn_med"]),
                    "p_hit": float(m[hit_col]),
                    "n": int(m["n"]),
                })


    # =========================
    # SAVE
    # =========================

    os.makedirs(OUT_DIR, exist_ok=True)

    edge_df = pd.DataFrame(edge_rows)
    regime_df = pd.DataFrame(regime_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    if len(edge_df) == 0:
        raise SystemExit("edge_rows is empty: no active pattern metrics were produced")

    edge_df.to_csv(OUT_EDGE, index=False)
    regime_df.to_csv(OUT_REGIME, index=False)
    overlap_df.to_csv(OUT_OVERLAP, index=False)

    print("WROTE", OUT_EDGE)
    print("WROTE", OUT_OVERLAP)
    print("WROTE", OUT_REGIME)
    print()
    print("EDGE SHAPE:", edge_df.shape)
    print(edge_df.groupby(["side", "pattern"]).size().to_string())


if __name__ == "__main__":
    main()
