python - <<'PY'
from pathlib import Path
import os
import json
import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/Users/tema/PycharmProjects/ImbalanceSearcher")
DB_DSN = os.environ.get("IMB_DB_DSN", "dbname=imb_traid host=localhost port=5432")

M1_DATA_DIR = ROOT / "data" / "m1_4"

OUT_CSV = ROOT / "online" / "combo_compare_target3_ttl.csv"
OUT_DIR = ROOT / "online" / "_combo_compare_top12_20260101_20260501"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_TS = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
END_TS = pd.Timestamp("2026-05-01 23:59:59", tz="UTC")

GATE5_1_PROD_PAIR_NAME = "top12_gate5_1"
GATE5_23_PROD_PAIR_NAME = "top12_gate5_1_20260101_20260501"

GATE5_1_TABLE = "online_gate5_1_scores_top12_20260101_20260501"
GATE5_2_TABLE = "online_gate5_2_ranker_top12_20260101_20260501"
GATE5_3_TABLE = "online_gate5_3_decisions_top12_20260101_20260501"

EXCLUDED_GRIDS = {
    "tp100_sl075",
}

TARGET_PAIR_MODEL_NAMES = [
    "tp240_sl060__vs__tp150_sl060",
    "tp240_sl060__vs__tp200_sl050",
    "tp200_sl050__vs__tp240_sl060",
]

TTL_HOURS_LIST = [
    12,
    16,
    20,
    24,
]

GRID_LIST = sorted(
    {
        grid_name
        for pair_model_name in TARGET_PAIR_MODEL_NAMES
        for grid_name in pair_model_name.split("__vs__")
    }
)

GATE2_THRS = [0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]
GATE4_THRS = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60]
GATE5_1_THRS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55]
GATE5_3_THRS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.625, 0.63, 0.64, 0.65, 0.70]

START_CAPITAL = 100.0
ENTRY_DELAY_SECONDS = 90
DEFAULT_TTL_HOURS = 16

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.004

ATR_PERIOD = 14
H4_RULE = "4h"

MIN_TRADES_TAKEN = 20

EXCLUDE_VARIANTS = {
    "no_blacklist": set(),
    "excl_BTC_only": {"BTCUSDT"},
    "excl_risk5": {"HUSDT", "BTCUSDT", "MUSDT", "1000BONKUSDT", "HAEDALUSDT"},
    "excl_risk5_watch2": {"HUSDT", "BTCUSDT", "MUSDT", "1000BONKUSDT", "HAEDALUSDT", "FLRUSDT", "POLUSDT"},
}

SLOT_VARIANTS = {
    "slot1_100pct": {
        "max_slots": 1,
        "slot_fraction": 1.0,
    },
    "slot2_50pct": {
        "max_slots": 2,
        "slot_fraction": 0.5,
    },
}


def norm_ts(s):
    return pd.to_datetime(s, errors="coerce", utc=True)


def key_cols(df):
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["signal_ts"] = norm_ts(df["signal_ts"])
    df["key"] = (
        df["symbol"].astype(str)
        + "|"
        + df["signal_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )
    return df


def read_sql(conn, sql, params=None):
    return pd.read_sql_query(sql, conn, params=params)


def parse_grid(grid_name):
    left, right = str(grid_name).split("_")
    tp_atr = float(left.replace("tp", "")) / 100.0
    sl_atr = float(right.replace("sl", "")) / 100.0
    return tp_atr, sl_atr


def load_gate2(conn):
    sql = """
        SELECT
            symbol,
            entry_ts AS signal_ts,
            up_reach_high_proba,
            dn_reach_high_proba,
            gate2_side
        FROM online_gate2_predictions
        WHERE entry_ts >= %s
          AND entry_ts <= %s
    """
    df = read_sql(conn, sql, (START_TS.to_pydatetime(), END_TS.to_pydatetime()))
    df["signal_ts"] = norm_ts(df["signal_ts"])
    df["symbol"] = df["symbol"].astype(str)
    df["gate2_side"] = df["gate2_side"].astype(str).str.upper()

    df["gate2_long_proba"] = pd.to_numeric(df["up_reach_high_proba"], errors="coerce")
    df["gate2_short_proba"] = pd.to_numeric(df["dn_reach_high_proba"], errors="coerce")

    df = key_cols(df)

    return df


def load_gate4(conn):
    sql = """
        SELECT
            signal_key,
            symbol,
            signal_ts,
            proba_short,
            proba_long,
            gate4_confidence,
            gate4_pred_side,
            gate4_pred_side_ratio,
            gate4_pred_side_gap
        FROM online_gate4_predictions_no_raw_refs
        WHERE signal_ts >= %s
          AND signal_ts <= %s
    """
    df = read_sql(conn, sql, (START_TS.to_pydatetime(), END_TS.to_pydatetime()))
    df["signal_ts"] = norm_ts(df["signal_ts"])
    df["symbol"] = df["symbol"].astype(str)
    df["gate4_pred_side"] = df["gate4_pred_side"].astype(str).str.upper()
    df["gate4_confidence"] = pd.to_numeric(df["gate4_confidence"], errors="coerce")
    df = key_cols(df)
    return df


def load_gate5_1(conn):
    sql = """
        SELECT
            signal_key,
            symbol,
            signal_ts,
            side,
            prod_pair_name,
            grid_name,
            tp_atr,
            sl_atr,
            rr,
            gate4_confidence AS gate5_1_gate4_confidence,
            pred_side_confidence,
            pred_side_ratio,
            gate5_1_proba,
            missing_feature_count
        FROM online_gate5_1_scores_top12_20260101_20260501
        WHERE signal_ts >= %s
          AND signal_ts <= %s
          AND prod_pair_name = %s
          AND grid_name = ANY(%s)
    """
    df = read_sql(conn, sql, (START_TS.to_pydatetime(), END_TS.to_pydatetime(), GATE5_1_PROD_PAIR_NAME, GRID_LIST))
    df["signal_ts"] = norm_ts(df["signal_ts"])
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()
    df["grid_name"] = df["grid_name"].astype(str)
    df["gate5_1_proba"] = pd.to_numeric(df["gate5_1_proba"], errors="coerce")
    df["tp_atr"] = pd.to_numeric(df["tp_atr"], errors="coerce")
    df["sl_atr"] = pd.to_numeric(df["sl_atr"], errors="coerce")
    df["rr"] = pd.to_numeric(df["rr"], errors="coerce")
    df = key_cols(df)
    return df


def load_gate5_2(conn):
    sql = """
        SELECT
            signal_key,
            symbol,
            signal_ts,
            side,
            prod_pair_name,
            grid_name,
            grid_proba,
            sig_top1_proba,
            sig_top2_proba,
            sig_top1_minus_top2_proba,
            sig_mean_proba,
            sig_std_proba,
            grid_proba_to_top1_ratio,
            gate4_confidence AS gate5_2_gate4_confidence,
            pred_side_confidence AS gate5_2_pred_side_confidence,
            pred_side_ratio AS gate5_2_pred_side_ratio
        FROM online_gate5_2_ranker_top12_20260101_20260501
        WHERE signal_ts >= %s
          AND signal_ts <= %s
          AND prod_pair_name = %s
          AND grid_name = ANY(%s)
    """
    df = read_sql(conn, sql, (START_TS.to_pydatetime(), END_TS.to_pydatetime(), GATE5_23_PROD_PAIR_NAME, GRID_LIST))
    df["signal_ts"] = norm_ts(df["signal_ts"])
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()
    df["grid_name"] = df["grid_name"].astype(str)
    df["grid_proba"] = pd.to_numeric(df["grid_proba"], errors="coerce")
    df = key_cols(df)
    return df


def load_gate5_3(conn):
    sql = """
        SELECT
            signal_key,
            symbol,
            signal_ts,
            side,
            prod_pair_name,
            pair_model_name,
            safe_grid_name,
            agg_grid_name,
            chosen_grid_name,
            chosen_tp_atr,
            chosen_sl_atr,
            chosen_rr,
            pred_label,
            pred_proba,
            safe_grid_proba,
            agg_grid_proba,
            proba_diff,
            proba_ratio,
            missing_feature_count
        FROM online_gate5_3_decisions_top12_20260101_20260501
        WHERE signal_ts >= %s
          AND signal_ts <= %s
          AND prod_pair_name = %s
          AND missing_feature_count = 0
          AND pair_model_name = ANY(%s)
          AND agg_grid_name = ANY(%s)
    """
    df = read_sql(
        conn,
        sql,
        (
            START_TS.to_pydatetime(),
            END_TS.to_pydatetime(),
            GATE5_23_PROD_PAIR_NAME,
            TARGET_PAIR_MODEL_NAMES,
            GRID_LIST,
        ),
    )
    df["signal_ts"] = norm_ts(df["signal_ts"])
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()
    df["pair_model_name"] = df["pair_model_name"].astype(str)
    df["safe_grid_name"] = df["safe_grid_name"].astype(str)
    df["agg_grid_name"] = df["agg_grid_name"].astype(str)
    df["chosen_grid_name"] = df["chosen_grid_name"].astype(str)
    df["pred_proba"] = pd.to_numeric(df["pred_proba"], errors="coerce")
    df = key_cols(df)

    df = df[
        (~df["safe_grid_name"].isin(EXCLUDED_GRIDS))
        & (~df["agg_grid_name"].isin(EXCLUDED_GRIDS))
        & (~df["chosen_grid_name"].isin(EXCLUDED_GRIDS))
        ].copy()

    df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce")
    df["chosen_tp_atr"] = pd.to_numeric(df["chosen_tp_atr"], errors="coerce")
    df["chosen_sl_atr"] = pd.to_numeric(df["chosen_sl_atr"], errors="coerce")
    df["chosen_rr"] = pd.to_numeric(df["chosen_rr"], errors="coerce")

    df = df.dropna(
        subset=[
            "pred_label",
            "pred_proba",
            "chosen_grid_name",
            "chosen_tp_atr",
            "chosen_sl_atr",
            "chosen_rr",
        ]
    ).copy()

    # ВАЖНО:
    # Для настоящего Gate5_3-бэктеста торгуем именно выбранную сетку,
    # а не всегда agg_grid_name.
    df["grid_name"] = df["chosen_grid_name"]

    df["tp_atr"] = df["chosen_tp_atr"]
    df["sl_atr"] = df["chosen_sl_atr"]
    df["rr"] = df["chosen_rr"]

    # confidence выбранной стороны:
    # pred_label=1 => выбрана agg-сетка, confidence = pred_proba
    # pred_label=0 => выбрана safe-сетка, confidence = 1 - pred_proba
    df["gate5_3_choice_confidence"] = np.where(
        pd.to_numeric(df["pred_label"], errors="coerce") == 1,
        pd.to_numeric(df["pred_proba"], errors="coerce"),
        1.0 - pd.to_numeric(df["pred_proba"], errors="coerce"),
    )

    return df


def join_stack(gate2, gate4, gate5_1, gate5_2, gate5_3):
    base = gate5_3.copy()

    base["prod_pair_name"] = base["prod_pair_name"].astype(str)
    base["side"] = base["side"].astype(str).str.upper()
    base["agg_grid_name"] = base["agg_grid_name"].astype(str)
    base["safe_grid_name"] = base["safe_grid_name"].astype(str)
    base["chosen_grid_name"] = base["chosen_grid_name"].astype(str)

    # ВАЖНО ДЛЯ TOP12:
    # Gate5_3 pair_model вида safe__vs__agg.
    # Для фильтра Gate5_1/Gate5_2 мы проверяем именно agg_grid_name,
    # как в старом одиночном бэктесте agg_grid_name = GRID_NAME.
    base["grid_name"] = base["chosen_grid_name"]
    base = base.merge(
        gate5_2.drop(columns=["signal_key"], errors="ignore"),
        on=["key", "symbol", "signal_ts", "side", "prod_pair_name", "grid_name"],
        how="inner",
        suffixes=("", "_g52"),
    )

    # ВАЖНО:
    # Gate5_1 был записан с другим prod_pair_name:
    #   top12_gate5_1
    # а Gate5_2/Gate5_3:
    #   top12_gate5_1_20260101_20260501
    # Поэтому gate5_1 нельзя мержить по prod_pair_name.
    gate5_1_join = gate5_1.drop(columns=["signal_key", "prod_pair_name"], errors="ignore").copy()

    base = base.merge(
        gate5_1_join,
        on=["key", "symbol", "signal_ts", "side", "grid_name"],
        how="inner",
        suffixes=("", "_g51"),
    )

    gate4_small = gate4[
        [
            "key",
            "symbol",
            "signal_ts",
            "proba_short",
            "proba_long",
            "gate4_confidence",
            "gate4_pred_side",
            "gate4_pred_side_ratio",
            "gate4_pred_side_gap",
        ]
    ].copy()

    base = base.merge(
        gate4_small,
        on=["key", "symbol", "signal_ts"],
        how="inner",
    )

    gate2_small = gate2[
        [
            "key",
            "symbol",
            "signal_ts",
            "gate2_long_proba",
            "gate2_short_proba",
            "gate2_side",
            "up_reach_high_proba",
            "dn_reach_high_proba",
        ]
    ].copy()

    base = base.merge(
        gate2_small,
        on=["key", "symbol", "signal_ts"],
        how="inner",
    )

    base["gate2_for_gate4_side_proba"] = np.where(
        base["side"].astype(str).str.upper() == "LONG",
        pd.to_numeric(base["gate2_long_proba"], errors="coerce"),
        pd.to_numeric(base["gate2_short_proba"], errors="coerce"),
    )

    base["gate2_side_matches_gate4"] = (
        base["gate2_side"].astype(str).str.upper()
        == base["side"].astype(str).str.upper()
    )

    return base

def apply_cfg(df, pair_model_name, gate2_thr, gate4_thr, gate5_1_thr, gate5_3_thr, exclude_symbols):
    out = df.copy()

    mask = (
        (out["pair_model_name"].astype(str) == str(pair_model_name))
        & (pd.to_numeric(out["gate2_for_gate4_side_proba"], errors="coerce") >= gate2_thr)
        & (pd.to_numeric(out["gate4_confidence"], errors="coerce") >= gate4_thr)
        & (pd.to_numeric(out["gate5_1_proba"], errors="coerce") >= gate5_1_thr)
        & (pd.to_numeric(out["gate5_3_choice_confidence"], errors="coerce") >= gate5_3_thr)
        & (out["side"].isin(["LONG", "SHORT"]))
    )

    out = out.loc[mask].copy()

    if exclude_symbols:
        out = out[~out["symbol"].astype(str).isin(exclude_symbols)].copy()

    out = out.sort_values(["signal_ts", "symbol", "side", "pair_model_name"]).reset_index(drop=True)
    return out

def add_signal_strength(df, gate2_thr, gate4_thr, gate5_1_thr, gate5_3_thr):
    out = df.copy()

    gate2 = pd.to_numeric(out["gate2_for_gate4_side_proba"], errors="coerce")
    gate4 = pd.to_numeric(out["gate4_confidence"], errors="coerce")
    gate5_1 = pd.to_numeric(out["gate5_1_proba"], errors="coerce")
    gate5_3 = pd.to_numeric(out["gate5_3_choice_confidence"], errors="coerce")

    out["strength_gate2"] = ((gate2 - gate2_thr) / max(1.0 - gate2_thr, 1e-9)).clip(lower=0.0, upper=1.0)
    out["strength_gate4"] = ((gate4 - gate4_thr) / max(1.0 - gate4_thr, 1e-9)).clip(lower=0.0, upper=1.0)
    out["strength_gate5_1"] = ((gate5_1 - gate5_1_thr) / max(1.0 - gate5_1_thr, 1e-9)).clip(lower=0.0, upper=1.0)
    out["strength_gate5_3"] = ((gate5_3 - gate5_3_thr) / max(1.0 - gate5_3_thr, 1e-9)).clip(lower=0.0, upper=1.0)

    out["signal_strength"] = (
        0.30 * out["strength_gate2"]
        + 0.25 * out["strength_gate4"]
        + 0.20 * out["strength_gate5_1"]
        + 0.25 * out["strength_gate5_3"]
    )

    return out


def keep_best_signal_per_candle(df):
    if df.empty:
        return df

    out = df.copy()

    out = out.sort_values(
        [
            "signal_ts",
            "signal_strength",
            "gate5_3_choice_confidence",
            "gate5_1_proba",
            "gate4_confidence",
            "gate2_for_gate4_side_proba",
            "symbol",
            "side",
            "pair_model_name",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ],
    ).copy()

    out = out.drop_duplicates(["signal_ts"], keep="first").reset_index(drop=True)
    return out


def find_m1_path(symbol):
    candidates = [
        M1_DATA_DIR / (symbol + ".parquet"),
        M1_DATA_DIR / (symbol + "_m1.parquet"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_ts_col(df):
    for col in ["ts", "timestamp", "open_time", "time", "datetime", "dt"]:
        if col in df.columns:
            return col
    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"
    raise RuntimeError("timestamp column not found")


def read_m1(symbol, cache):
    if symbol in cache:
        return cache[symbol]

    path = find_m1_path(symbol)
    if path is None:
        cache[symbol] = None
        return None

    df = pd.read_parquet(path)
    ts_col = find_ts_col(df)

    if ts_col == "__index__":
        df = df.reset_index().rename(columns={df.index.name or "index": "ts"})
        ts_col = "ts"

    df["ts"] = norm_ts(df[ts_col])

    need = ["ts", "open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError("m1 missing columns for %s: %s" % (symbol, missing))

    df = df[need].dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cache[symbol] = df
    return df


def build_h4_from_m1(m1):
    df = m1.copy()
    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).copy()
    df = df.set_index("ts").sort_index()

    h4 = pd.DataFrame()
    h4["open"] = df["open"].resample(H4_RULE, label="right", closed="right").first()
    h4["high"] = df["high"].resample(H4_RULE, label="right", closed="right").max()
    h4["low"] = df["low"].resample(H4_RULE, label="right", closed="right").min()
    h4["close"] = df["close"].resample(H4_RULE, label="right", closed="right").last()

    h4 = h4.dropna(subset=["open", "high", "low", "close"]).reset_index()
    h4 = h4.rename(columns={"ts": "signal_ts"})

    prev_close = h4["close"].shift(1)
    tr1 = h4["high"] - h4["low"]
    tr2 = (h4["high"] - prev_close).abs()
    tr3 = (h4["low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    h4["atr14"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    h4["signal_ts"] = norm_ts(h4["signal_ts"])
    return h4


def add_online_price_refs_from_m1(candidates):
    out = candidates.copy()
    m1_cache = {}
    h4_cache = {}

    rows = []
    missing_m1 = []
    missing_h4_ref = []

    for symbol, part in out.groupby("symbol", dropna=False):
        symbol = str(symbol)

        m1 = read_m1(symbol, m1_cache)
        if m1 is None or len(m1) == 0:
            missing_m1.append(symbol)
            continue

        if symbol not in h4_cache:
            h4_cache[symbol] = build_h4_from_m1(m1)

        h4 = h4_cache[symbol]
        if len(h4) == 0:
            missing_h4_ref.append(symbol)
            continue

        refs = h4[["signal_ts", "close", "atr14"]].copy()
        refs["symbol"] = symbol
        refs = refs.rename(columns={"close": "entry_px_ref", "atr14": "atr14_ref"})

        rows.append(refs)

    if missing_m1:
        pd.DataFrame({"symbol": sorted(set(missing_m1))}).to_csv(OUT_DIR / "_debug_missing_m1_files.csv", index=False)

    if missing_h4_ref:
        pd.DataFrame({"symbol": sorted(set(missing_h4_ref))}).to_csv(OUT_DIR / "_debug_missing_h4_refs_from_m1.csv", index=False)

    if rows:
        refs_all = pd.concat(rows, ignore_index=True)
    else:
        refs_all = pd.DataFrame(columns=["signal_ts", "symbol", "entry_px_ref", "atr14_ref"])

    refs_all["signal_ts"] = norm_ts(refs_all["signal_ts"])
    refs_all["symbol"] = refs_all["symbol"].astype(str)
    refs_all = refs_all.drop_duplicates(["symbol", "signal_ts"], keep="last").reset_index(drop=True)

    out = out.merge(
        refs_all,
        on=["symbol", "signal_ts"],
        how="left",
    )

    out["entry_px"] = pd.to_numeric(out["entry_px_ref"], errors="coerce")
    out["atr14"] = pd.to_numeric(out["atr14_ref"], errors="coerce")

    out = out.drop(columns=["entry_px_ref", "atr14_ref"], errors="ignore")
    return out


def simulate_one(row, m1_cache, ttl_hours):
    symbol = str(row["symbol"])
    side = str(row["side"]).upper()

    signal_ts = pd.Timestamp(row["signal_ts"])
    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.tz_localize("UTC")
    else:
        signal_ts = signal_ts.tz_convert("UTC")

    entry_ts = signal_ts + pd.Timedelta(seconds=ENTRY_DELAY_SECONDS)
    end_ts = entry_ts + pd.Timedelta(hours=int(ttl_hours))

    atr14 = pd.to_numeric(row.get("atr14", np.nan), errors="coerce")
    tp_atr = pd.to_numeric(row.get("tp_atr", np.nan), errors="coerce")
    sl_atr = pd.to_numeric(row.get("sl_atr", np.nan), errors="coerce")

    m1 = read_m1(symbol, m1_cache)
    if m1 is None or len(m1) == 0:
        return None

    window = m1[(m1["ts"] >= entry_ts) & (m1["ts"] <= end_ts)].copy()
    if len(window) == 0:
        return None

    first = window.iloc[0]
    entry_px = float(first["open"])

    if not np.isfinite(atr14) or atr14 <= 0:
        return None

    if not np.isfinite(entry_px) or entry_px <= 0:
        return None

    if not np.isfinite(tp_atr) or tp_atr <= 0:
        return None

    if not np.isfinite(sl_atr) or sl_atr <= 0:
        return None

    if side == "LONG":
        tp_px = entry_px + tp_atr * atr14
        sl_px = entry_px - sl_atr * atr14
    elif side == "SHORT":
        tp_px = entry_px - tp_atr * atr14
        sl_px = entry_px + sl_atr * atr14
    else:
        return None

    exit_reason = "TTL"
    exit_px = float(window.iloc[-1]["close"])
    exit_ts = pd.Timestamp(window.iloc[-1]["ts"])

    for _, bar in window.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        ts = pd.Timestamp(bar["ts"])

        if side == "LONG":
            tp_hit = high >= tp_px
            sl_hit = low <= sl_px
        else:
            tp_hit = low <= tp_px
            sl_hit = high >= sl_px

        if tp_hit and sl_hit:
            exit_reason = "SL"
            exit_px = sl_px
            exit_ts = ts
            break

        if tp_hit:
            exit_reason = "TP"
            exit_px = tp_px
            exit_ts = ts
            break

        if sl_hit:
            exit_reason = "SL"
            exit_px = sl_px
            exit_ts = ts
            break

    if side == "LONG":
        gross_ret = (exit_px / entry_px) - 1.0
    else:
        gross_ret = (entry_px / exit_px) - 1.0

    net_ret = gross_ret - (2.0 * FEE_PER_SIDE) - (2.0 * SLIPPAGE_PER_SIDE)

    out = row.to_dict()
    out.update(
        {
            "ttl_hours": int(ttl_hours),
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "entry_px": float(entry_px),
            "exit_px": float(exit_px),
            "atr14": float(atr14),
            "tp_px": float(tp_px),
            "sl_px": float(sl_px),
            "exit_reason": exit_reason,
            "gross_ret": float(gross_ret),
            "net_ret": float(net_ret),
        }
    )
    return out


def simulate_candidates(candidates, ttl_hours):
    candidates = add_online_price_refs_from_m1(candidates)

    m1_cache = {}
    rows = []

    for _, row in candidates.iterrows():
        sim = simulate_one(row, m1_cache, ttl_hours=ttl_hours)
        if sim is not None:
            rows.append(sim)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def run_slot_backtest(sim_df, max_slots, slot_fraction):
    if len(sim_df) == 0:
        return pd.DataFrame(), {
            "trades_taken": 0,
            "final_capital": START_CAPITAL,
            "total_return_pct": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
            "tp_count": 0,
            "sl_count": 0,
            "ttl_count": 0,
            "first_entry_ts": None,
            "last_exit_ts": None,
            "simulated_rows": 0,
            "skipped_by_slot": 0,
            "max_slots": int(max_slots),
            "slot_fraction": float(slot_fraction),
        }

    df = sim_df.copy()
    df["entry_ts"] = norm_ts(df["entry_ts"])
    df["exit_ts"] = norm_ts(df["exit_ts"])
    sort_cols = ["entry_ts"]
    ascending = [True]

    if "signal_strength" in df.columns:
        sort_cols.append("signal_strength")
        ascending.append(False)

    sort_cols += ["symbol", "side", "pair_model_name"]
    ascending += [True, True, True]

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    cash = START_CAPITAL
    peak_equity = START_CAPITAL
    max_dd = 0.0
    open_positions = []
    closed_trades = []
    skipped_by_slot = 0

    def current_equity():
        open_alloc = 0.0
        for pos in open_positions:
            open_alloc += float(pos["allocation"])
        return float(cash + open_alloc)

    def close_due_positions(now_ts):
        nonlocal cash, peak_equity, max_dd, open_positions, closed_trades

        still_open = []

        for pos in open_positions:
            if pd.Timestamp(pos["exit_ts"]) <= now_ts:
                row = pos["row"]
                allocation = float(pos["allocation"])
                net_ret = float(row["net_ret"])
                pnl = allocation * net_ret

                cash += allocation + pnl

                r = row.to_dict()
                r["allocation_usd"] = allocation
                r["pnl_usd"] = pnl
                r["capital_after"] = current_equity()
                r["open_slots_after_exit"] = len(open_positions) - 1
                closed_trades.append(r)

                equity = current_equity()
                peak_equity = max(peak_equity, equity)
                dd = (equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0
                max_dd = min(max_dd, dd)
            else:
                still_open.append(pos)

        open_positions = still_open

    for _, row in df.iterrows():
        entry_ts = pd.Timestamp(row["entry_ts"])
        exit_ts = pd.Timestamp(row["exit_ts"])

        close_due_positions(entry_ts)

        if len(open_positions) >= max_slots:
            skipped_by_slot += 1
            continue

        equity_before_entry = current_equity()
        target_allocation = equity_before_entry * float(slot_fraction)
        allocation = min(cash, target_allocation)

        if allocation <= 0:
            skipped_by_slot += 1
            continue

        cash -= allocation

        open_positions.append(
            {
                "exit_ts": exit_ts,
                "allocation": allocation,
                "row": row,
            }
        )

    close_due_positions(pd.Timestamp("2262-04-11 00:00:00", tz="UTC"))

    trades_df = pd.DataFrame(closed_trades)

    if len(trades_df) == 0:
        return trades_df, {
            "trades_taken": 0,
            "final_capital": START_CAPITAL,
            "total_return_pct": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
            "tp_count": 0,
            "sl_count": 0,
            "ttl_count": 0,
            "first_entry_ts": None,
            "last_exit_ts": None,
            "simulated_rows": int(len(sim_df)),
            "skipped_by_slot": int(skipped_by_slot),
            "max_slots": int(max_slots),
            "slot_fraction": float(slot_fraction),
        }

    trades_df["entry_ts"] = norm_ts(trades_df["entry_ts"])
    trades_df["exit_ts"] = norm_ts(trades_df["exit_ts"])
    trades_df = trades_df.sort_values(["entry_ts", "symbol", "side", "pair_model_name"]).reset_index(drop=True)

    final_capital = float(cash)
    wins = trades_df[trades_df["net_ret"] > 0]
    losses = trades_df[trades_df["net_ret"] <= 0]

    gross_profit = float(wins["pnl_usd"].sum()) if len(wins) else 0.0
    gross_loss_abs = float(abs(losses["pnl_usd"].sum())) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else None

    summary = {
        "trades_taken": int(len(trades_df)),
        "final_capital": final_capital,
        "total_return_pct": float((final_capital / START_CAPITAL) - 1.0),
        "win_rate": float((trades_df["net_ret"] > 0).mean()),
        "mean_net_ret_pct": float(trades_df["net_ret"].mean()),
        "median_net_ret_pct": float(trades_df["net_ret"].median()),
        "min_net_ret_pct": float(trades_df["net_ret"].min()),
        "max_net_ret_pct": float(trades_df["net_ret"].max()),
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss_abs,
        "profit_factor": profit_factor,
        "max_drawdown_pct": float(max_dd),
        "tp_count": int((trades_df["exit_reason"] == "TP").sum()),
        "sl_count": int((trades_df["exit_reason"] == "SL").sum()),
        "ttl_count": int((trades_df["exit_reason"] == "TTL").sum()),
        "first_entry_ts": str(trades_df["entry_ts"].min()),
        "last_exit_ts": str(trades_df["exit_ts"].max()),
        "simulated_rows": int(len(sim_df)),
        "skipped_by_slot": int(skipped_by_slot),
        "max_slots": int(max_slots),
        "slot_fraction": float(slot_fraction),
    }

    return trades_df, summary


def main():
    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("OUT_CSV:", OUT_CSV)
    print("OUT_DIR:", OUT_DIR)
    print("PERIOD:", START_TS, "->", END_TS)
    print("MODE: TOP12 PAIR_MODEL SWEEP + ONLINE DB GATES + M1_4 MARKET REPLAY")
    print("GATE5_1_PROD_PAIR_NAME:", GATE5_1_PROD_PAIR_NAME)
    print("GATE5_23_PROD_PAIR_NAME:", GATE5_23_PROD_PAIR_NAME)
    print("GATE5_1_TABLE:", GATE5_1_TABLE)
    print("GATE5_2_TABLE:", GATE5_2_TABLE)
    print("GATE5_3_TABLE:", GATE5_3_TABLE)
    print("EXCLUDED_GRIDS:", sorted(EXCLUDED_GRIDS))
    print("GRID_LIST:", GRID_LIST)
    print("GATE2_THRS:", GATE2_THRS)
    print("GATE4_THRS:", GATE4_THRS)
    print("GATE5_1_THRS:", GATE5_1_THRS)
    print("GATE5_3_THRS:", GATE5_3_THRS)
    print("EXCLUDE_VARIANTS:", {k: sorted(v) for k, v in EXCLUDE_VARIANTS.items()})
    print("SLOT_VARIANTS:", SLOT_VARIANTS)
    print()

    conn = psycopg2.connect(DB_DSN)

    gate2 = load_gate2(conn)
    gate4 = load_gate4(conn)
    gate5_1 = load_gate5_1(conn)
    gate5_2 = load_gate5_2(conn)
    gate5_3 = load_gate5_3(conn)

    conn.close()

    print("=" * 120)
    print("LOADED FROM ONLINE DB")
    print("gate2 :", len(gate2), "symbols:", gate2["symbol"].nunique())
    print("gate4 :", len(gate4), "symbols:", gate4["symbol"].nunique())
    print("gate5_1:", len(gate5_1), "symbols:", gate5_1["symbol"].nunique(), "grids:", gate5_1["grid_name"].nunique())
    print("gate5_2:", len(gate5_2), "symbols:", gate5_2["symbol"].nunique(), "grids:", gate5_2["grid_name"].nunique())
    print("gate5_3:", len(gate5_3), "symbols:", gate5_3["symbol"].nunique(), "pairs:", gate5_3["pair_model_name"].nunique())
    print()

    joined = join_stack(gate2, gate4, gate5_1, gate5_2, gate5_3)
    joined = joined[
        (~joined["safe_grid_name"].isin(EXCLUDED_GRIDS))
        & (~joined["agg_grid_name"].isin(EXCLUDED_GRIDS))
        & (~joined["chosen_grid_name"].isin(EXCLUDED_GRIDS))
        & (~joined["grid_name"].isin(EXCLUDED_GRIDS))
        ].copy()

    joined_path = OUT_DIR / "joined_online_gate_stack_top12.csv"
    joined.to_csv(joined_path, index=False)

    print("=" * 120)
    print("JOINED")
    print("joined rows:", len(joined))
    print("joined signals:", joined["key"].nunique())
    print("joined pair models:", joined["pair_model_name"].nunique())
    print("joined agg grids:")
    print(joined["agg_grid_name"].value_counts(dropna=False).to_string())
    print("WROTE:", joined_path)
    print()

    available_pair_names = set(joined["pair_model_name"].dropna().astype(str).unique().tolist())

    missing_target_pairs = [
        pair_model_name
        for pair_model_name in TARGET_PAIR_MODEL_NAMES
        if pair_model_name not in available_pair_names
    ]

    if missing_target_pairs:
        print("WARNING missing target pairs:", missing_target_pairs)

    pair_names = [
        pair_model_name
        for pair_model_name in TARGET_PAIR_MODEL_NAMES
        if pair_model_name in available_pair_names
    ]

    all_summary_rows = []
    best_trades = None
    best_final_capital = -1.0
    best_cfg_name = None

    for pair_i, pair_model_name in enumerate(pair_names, start=1):
        pair_df = joined[joined["pair_model_name"].astype(str) == pair_model_name].copy()

        if pair_df.empty:
            continue

        agg_grid_name = str(pair_df["agg_grid_name"].iloc[0])
        safe_grid_name = str(pair_df["safe_grid_name"].iloc[0])
        tp_atr = None
        sl_atr = None

        print("=" * 120)
        print("PAIR", pair_i, "/", len(pair_names), pair_model_name)
        print("SAFE:", safe_grid_name, "AGG:", agg_grid_name, "TP_ATR:", tp_atr, "SL_ATR:", sl_atr)
        print("pair rows:", len(pair_df), "signals:", pair_df["key"].nunique(), "symbols:", pair_df["symbol"].nunique())

        pair_best_capital = -1.0
        pair_best_row = None

        for blacklist_name, exclude_symbols in EXCLUDE_VARIANTS.items():
            for g53_thr in GATE5_3_THRS:
                loose_candidates = apply_cfg(
                    df=pair_df,
                    pair_model_name=pair_model_name,
                    gate2_thr=min(GATE2_THRS),
                    gate4_thr=min(GATE4_THRS),
                    gate5_1_thr=min(GATE5_1_THRS),
                    gate5_3_thr=g53_thr,
                    exclude_symbols=exclude_symbols,
                )

                if loose_candidates.empty:
                    continue

                for ttl_hours in TTL_HOURS_LIST:
                    sim = simulate_candidates(loose_candidates, ttl_hours=ttl_hours)

                    if sim.empty:
                        continue

                    for g2_thr in GATE2_THRS:
                        for g4_thr in GATE4_THRS:
                            for g51_thr in GATE5_1_THRS:
                                cfg_sim_before_dedup = sim[
                                    (pd.to_numeric(sim["gate2_for_gate4_side_proba"], errors="coerce") >= g2_thr)
                                    & (pd.to_numeric(sim["gate4_confidence"], errors="coerce") >= g4_thr)
                                    & (pd.to_numeric(sim["gate5_1_proba"], errors="coerce") >= g51_thr)
                                    ].copy()

                                if cfg_sim_before_dedup.empty:
                                    continue

                                cfg_sim_before_dedup = add_signal_strength(
                                    cfg_sim_before_dedup,
                                    gate2_thr=g2_thr,
                                    gate4_thr=g4_thr,
                                    gate5_1_thr=g51_thr,
                                    gate5_3_thr=g53_thr,
                                )

                                cfg_sim = keep_best_signal_per_candle(cfg_sim_before_dedup)

                                if cfg_sim.empty:
                                    continue

                            if cfg_sim.empty:
                                continue

                            for slot_name, slot_cfg in SLOT_VARIANTS.items():
                                max_slots = int(slot_cfg["max_slots"])
                                slot_fraction = float(slot_cfg["slot_fraction"])

                                trades, summary = run_slot_backtest(
                                    sim_df=cfg_sim,
                                    max_slots=max_slots,
                                    slot_fraction=slot_fraction,
                                )

                                if int(summary["trades_taken"]) < MIN_TRADES_TAKEN:
                                    continue

                                cfg_name = (
                                    "{}__{}__{}__ttl_{:02d}h__g2_{:03d}__g4_{:03d}__g51_{:03d}__g53_{:03d}".format(
                                        pair_model_name,
                                        blacklist_name,
                                        slot_name,
                                        int(ttl_hours),
                                        int(round(g2_thr * 1000)),
                                        int(round(g4_thr * 1000)),
                                        int(round(g51_thr * 1000)),
                                        int(round(g53_thr * 1000)),
                                    )
                                )

                                row = {
                                    "cfg_name": cfg_name,
                                    "pair_model_name": pair_model_name,
                                    "safe_grid_name": safe_grid_name,
                                    "agg_grid_name": agg_grid_name,
                                    "trade_grid_name": "chosen_grid_name",
                                    "tp_atr": None,
                                    "sl_atr": None,
                                    "blacklist_name": blacklist_name,
                                    "exclude_symbols": ",".join(sorted(exclude_symbols)),
                                    "slot_name": slot_name,
                                    "ttl_hours": int(ttl_hours),
                                    "gate2_thr": float(g2_thr),
                                    "gate4_thr": float(g4_thr),
                                    "gate5_1_thr": float(g51_thr),
                                    "gate5_3_thr": float(g53_thr),
                                    "candidate_rows_before_dedup": int(len(cfg_sim_before_dedup)),
                                    "candidate_rows": int(len(cfg_sim)),
                                    "candidate_symbols": int(cfg_sim["symbol"].nunique()) if len(cfg_sim) else 0,
                                    "chosen_grid_distribution": json.dumps(
                                        {
                                            str(k): int(v)
                                            for k, v in
                                            cfg_sim["chosen_grid_name"].value_counts(dropna=False).to_dict().items()
                                        },
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    ),
                                }
                                row.update(summary)

                                all_summary_rows.append(row)

                                final_capital = float(summary["final_capital"])

                                if final_capital > pair_best_capital:
                                    pair_best_capital = final_capital
                                    pair_best_row = row

                                if final_capital > best_final_capital:
                                    best_final_capital = final_capital
                                    best_cfg_name = cfg_name
                                    best_trades = trades.copy()

        if all_summary_rows:
            summary_tmp = pd.DataFrame(all_summary_rows).sort_values(
                ["final_capital", "total_return_pct", "trades_taken"],
                ascending=[False, False, False],
            ).reset_index(drop=True)

            summary_tmp.to_csv(OUT_CSV, index=False)

        if pair_best_row is not None:
            print("PAIR BEST:")
            print(
                pd.DataFrame([pair_best_row])[
                    [
                        "pair_model_name",
                        "blacklist_name",
                        "slot_name",
                        "gate2_thr",
                        "gate4_thr",
                        "gate5_1_thr",
                        "gate5_3_thr",
                        "trades_taken",
                        "final_capital",
                        "total_return_pct",
                        "win_rate",
                        "profit_factor",
                        "max_drawdown_pct",
                        "tp_count",
                        "sl_count",
                        "ttl_count",
                    ]
                ].to_string(index=False)
            )
            print("CURRENT GLOBAL BEST:", best_cfg_name, "capital:", round(best_final_capital, 6))
            print("WROTE:", OUT_CSV)
            print()

    if not all_summary_rows:
        raise SystemExit("No configs passed MIN_TRADES_TAKEN")

    summary_all = pd.DataFrame(all_summary_rows).sort_values(
        ["final_capital", "total_return_pct", "trades_taken"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_all.to_csv(OUT_CSV, index=False)

    if best_trades is not None and len(best_trades):
        best_trades_path = OUT_DIR / "best_trades.csv"
        best_trades.to_csv(best_trades_path, index=False)
    else:
        best_trades_path = None

    report = {
        "mode": "top12_pair_model_sweep_online_db_gates_m1_4_replay",
        "root": str(ROOT),
        "db_dsn": DB_DSN,
        "out_csv": str(OUT_CSV),
        "out_dir": str(OUT_DIR),
        "start_ts": str(START_TS),
        "end_ts": str(END_TS),
        "gate5_1_prod_pair_name": GATE5_1_PROD_PAIR_NAME,
        "gate5_23_prod_pair_name": GATE5_23_PROD_PAIR_NAME,
        "gate5_1_table": GATE5_1_TABLE,
        "gate5_2_table": GATE5_2_TABLE,
        "gate5_3_table": GATE5_3_TABLE,
        "excluded_grids": sorted(EXCLUDED_GRIDS),
        "grid_list": GRID_LIST,
        "gate2_thrs": GATE2_THRS,
        "gate4_thrs": GATE4_THRS,
        "gate5_1_thrs": GATE5_1_THRS,
        "gate5_3_thrs": GATE5_3_THRS,
        "entry_delay_seconds": ENTRY_DELAY_SECONDS,
        "ttl_hours_list": TTL_HOURS_LIST,
        "fee_per_side": FEE_PER_SIDE,
        "slippage_per_side": SLIPPAGE_PER_SIDE,
        "joined_path": str(joined_path),
        "best_cfg_name": best_cfg_name,
        "best_final_capital": best_final_capital,
        "best_trades_path": None if best_trades_path is None else str(best_trades_path),
        "summary_rows": int(len(summary_all)),
    }

    report_path = OUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 120)
    print("FINAL TOP 50")
    show_cols = [
        "pair_model_name",
        "safe_grid_name",
        "agg_grid_name",
        "blacklist_name",
        "slot_name",
        "ttl_hours",
        "gate2_thr",
        "gate4_thr",
        "gate5_1_thr",
        "gate5_3_thr",
        "candidate_rows_before_dedup",
        "candidate_rows",
        "trades_taken",
        "skipped_by_slot",
        "final_capital",
        "total_return_pct",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "tp_count",
        "sl_count",
        "ttl_count",
        "first_entry_ts",
        "last_exit_ts",
    ]
    print(summary_all[show_cols].head(50).to_string(index=False))
    print()
    print("BEST_CFG:", best_cfg_name)
    print("BEST_FINAL_CAPITAL:", best_final_capital)
    print("WROTE:", OUT_CSV)
    print("WROTE:", report_path)
    if best_trades_path is not None:
        print("WROTE:", best_trades_path)


if __name__ == "__main__":
    main()
PY