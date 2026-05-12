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

OUT_DIR = ROOT / "online" / "_audit_single_cfg_ONLINE_ONLY_g2_600_g4_560_g51_500_g53_625_20260101_20260501_blacklist_slots_variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_TS = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
END_TS = pd.Timestamp("2026-05-01 23:59:59", tz="UTC")

PROD_PAIR_NAME = "tp225_sl075__vs__tp100_sl075"
GRID_NAME = "tp100_sl075"

GATE2_THR = 0.60
GATE4_THR = 0.56
GATE5_1_THR = 0.50
GATE5_3_THR = 0.625

TP_ATR = 1.0
SL_ATR = 0.75

START_CAPITAL = 100.0
ENTRY_DELAY_SECONDS = 90
TTL_HOURS = 16

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.004

ATR_PERIOD = 14
H4_RULE = "4h"

PRINT_TOP_N_SYMBOLS = 50
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
        FROM online_gate5_1_scores
        WHERE signal_ts >= %s
          AND signal_ts <= %s
          AND prod_pair_name = %s
          AND grid_name = %s
    """
    df = read_sql(conn, sql, (START_TS.to_pydatetime(), END_TS.to_pydatetime(), PROD_PAIR_NAME, GRID_NAME))
    df["signal_ts"] = norm_ts(df["signal_ts"])
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()
    df["gate5_1_proba"] = pd.to_numeric(df["gate5_1_proba"], errors="coerce")
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
        FROM online_gate5_2_ranker
        WHERE signal_ts >= %s
          AND signal_ts <= %s
          AND prod_pair_name = %s
          AND grid_name = %s
    """
    df = read_sql(conn, sql, (START_TS.to_pydatetime(), END_TS.to_pydatetime(), PROD_PAIR_NAME, GRID_NAME))
    df["signal_ts"] = norm_ts(df["signal_ts"])
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()
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
        FROM online_gate5_3_decisions
        WHERE signal_ts >= %s
          AND signal_ts <= %s
          AND prod_pair_name = %s
          AND agg_grid_name = %s
    """
    df = read_sql(conn, sql, (START_TS.to_pydatetime(), END_TS.to_pydatetime(), PROD_PAIR_NAME, GRID_NAME))
    df["signal_ts"] = norm_ts(df["signal_ts"])
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()
    df["chosen_grid_name"] = df["chosen_grid_name"].astype(str)
    df["pred_proba"] = pd.to_numeric(df["pred_proba"], errors="coerce")
    df = key_cols(df)
    return df


def join_stack(gate2, gate4, gate5_1, gate5_2, gate5_3):
    base = gate5_3.copy()

    base["grid_name"] = GRID_NAME
    base["prod_pair_name"] = base["prod_pair_name"].astype(str)
    base["side"] = base["side"].astype(str).str.upper()

    base = base.merge(
        gate5_2.drop(columns=["signal_key"], errors="ignore"),
        on=["key", "symbol", "signal_ts", "side", "prod_pair_name", "grid_name"],
        how="inner",
        suffixes=("", "_g52"),
    )

    base = base.merge(
        gate5_1.drop(columns=["signal_key"], errors="ignore"),
        on=["key", "symbol", "signal_ts", "side", "prod_pair_name", "grid_name"],
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


def apply_cfg(df, exclude_symbols):
    out = df.copy()

    mask = (
        (pd.to_numeric(out["gate2_for_gate4_side_proba"], errors="coerce") >= GATE2_THR)
        & (pd.to_numeric(out["gate4_confidence"], errors="coerce") >= GATE4_THR)
        & (pd.to_numeric(out["gate5_1_proba"], errors="coerce") >= GATE5_1_THR)
        & (pd.to_numeric(out["pred_proba"], errors="coerce") >= GATE5_3_THR)
        & (out["side"].isin(["LONG", "SHORT"]))
    )

    out = out.loc[mask].copy()

    if exclude_symbols:
        out = out[~out["symbol"].astype(str).isin(exclude_symbols)].copy()

    out = out.sort_values(["signal_ts", "symbol", "side"]).reset_index(drop=True)
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
        path = OUT_DIR / "_debug_missing_m1_files.csv"
        pd.DataFrame({"symbol": sorted(set(missing_m1))}).to_csv(path, index=False)
        print("WARNING: missing m1 files:", len(set(missing_m1)))
        print("WROTE:", path)

    if missing_h4_ref:
        path = OUT_DIR / "_debug_missing_h4_refs_from_m1.csv"
        pd.DataFrame({"symbol": sorted(set(missing_h4_ref))}).to_csv(path, index=False)
        print("WARNING: missing h4 refs from m1:", len(set(missing_h4_ref)))
        print("WROTE:", path)

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

    missing = out[
        out["entry_px"].isna()
        | out["atr14"].isna()
        | (out["entry_px"] <= 0)
        | (out["atr14"] <= 0)
    ].copy()

    if len(missing):
        path = OUT_DIR / "_debug_missing_entry_px_or_atr14_from_online_m1.csv"
        missing.to_csv(path, index=False)
        print("WARNING: missing entry_px/atr14 from online m1 rebuild:", len(missing))
        print("WROTE:", path)

    out = out.drop(columns=["entry_px_ref", "atr14_ref"], errors="ignore")
    return out


def simulate_one(row, m1_cache):
    symbol = str(row["symbol"])
    side = str(row["side"]).upper()

    signal_ts = pd.Timestamp(row["signal_ts"])
    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.tz_localize("UTC")
    else:
        signal_ts = signal_ts.tz_convert("UTC")

    entry_ts = signal_ts + pd.Timedelta(seconds=ENTRY_DELAY_SECONDS)
    end_ts = entry_ts + pd.Timedelta(hours=TTL_HOURS)

    atr14 = pd.to_numeric(row.get("atr14", np.nan), errors="coerce")

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

    if side == "LONG":
        tp_px = entry_px + TP_ATR * atr14
        sl_px = entry_px - SL_ATR * atr14
    elif side == "SHORT":
        tp_px = entry_px - TP_ATR * atr14
        sl_px = entry_px + SL_ATR * atr14
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


def simulate_candidates(candidates):
    candidates = add_online_price_refs_from_m1(candidates)

    m1_cache = {}
    rows = []

    for i, row in candidates.iterrows():
        sim = simulate_one(row, m1_cache)
        if sim is not None:
            rows.append(sim)

        if (i + 1) % 250 == 0:
            print("simulated progress:", i + 1, "/", len(candidates))

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
    df = df.sort_values(["entry_ts", "symbol", "side"]).reset_index(drop=True)

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
    trades_df = trades_df.sort_values(["entry_ts", "symbol", "side"]).reset_index(drop=True)

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
        "avg_win_net_ret_pct": float(wins["net_ret"].mean()) if len(wins) else None,
        "avg_loss_net_ret_pct": float(losses["net_ret"].mean()) if len(losses) else None,
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


def summarize_by_symbol(trades):
    if len(trades) == 0:
        return pd.DataFrame()

    g = trades.groupby("symbol", dropna=False)
    out = g.agg(
        trades=("symbol", "size"),
        net_ret_mean=("net_ret", "mean"),
        net_ret_median=("net_ret", "median"),
        pnl_usd=("pnl_usd", "sum"),
        last_capital=("capital_after", "last"),
        tp_count=("exit_reason", lambda s: int((s == "TP").sum())),
        sl_count=("exit_reason", lambda s: int((s == "SL").sum())),
        ttl_count=("exit_reason", lambda s: int((s == "TTL").sum())),
    ).reset_index()

    out["tp_share"] = out["tp_count"] / out["trades"]
    out["sl_share"] = out["sl_count"] / out["trades"]
    out["ttl_share"] = out["ttl_count"] / out["trades"]

    out = out.sort_values(["pnl_usd", "net_ret_mean"], ascending=[False, False]).reset_index(drop=True)
    return out


def summarize_by_side(trades):
    if len(trades) == 0:
        return pd.DataFrame()

    g = trades.groupby("side", dropna=False)
    out = g.agg(
        trades=("side", "size"),
        net_ret_mean=("net_ret", "mean"),
        net_ret_median=("net_ret", "median"),
        pnl_usd=("pnl_usd", "sum"),
        tp_count=("exit_reason", lambda s: int((s == "TP").sum())),
        sl_count=("exit_reason", lambda s: int((s == "SL").sum())),
        ttl_count=("exit_reason", lambda s: int((s == "TTL").sum())),
    ).reset_index()

    out["tp_share"] = out["tp_count"] / out["trades"]
    out["sl_share"] = out["sl_count"] / out["trades"]
    out["ttl_share"] = out["ttl_count"] / out["trades"]

    return out.sort_values("pnl_usd", ascending=False).reset_index(drop=True)


def main():
    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("OUT_DIR:", OUT_DIR)
    print("PERIOD:", START_TS, "->", END_TS)
    print("MODE: ONLINE DB GATES ONLY + M1 MARKET REPLAY + BLACKLIST/SLOT VARIANTS")
    print("CFG:", "g2_600__g4_560__g51_500__g53_625")
    print("GRID:", GRID_NAME, "TP_ATR:", TP_ATR, "SL_ATR:", SL_ATR)
    print("EXCLUDE_VARIANTS:", {k: sorted(v) for k, v in EXCLUDE_VARIANTS.items()})
    print("SLOT_VARIANTS:", SLOT_VARIANTS)

    conn = psycopg2.connect(DB_DSN)

    gate2 = load_gate2(conn)
    gate4 = load_gate4(conn)
    gate5_1 = load_gate5_1(conn)
    gate5_2 = load_gate5_2(conn)
    gate5_3 = load_gate5_3(conn)

    conn.close()

    print()
    print("=" * 120)
    print("LOADED FROM ONLINE DB")
    print("gate2 :", len(gate2), "symbols:", gate2["symbol"].nunique())
    print("gate4 :", len(gate4), "symbols:", gate4["symbol"].nunique())
    print("gate5_1:", len(gate5_1), "symbols:", gate5_1["symbol"].nunique())
    print("gate5_2:", len(gate5_2), "symbols:", gate5_2["symbol"].nunique())
    print("gate5_3:", len(gate5_3), "symbols:", gate5_3["symbol"].nunique())

    joined = join_stack(gate2, gate4, gate5_1, gate5_2, gate5_3)
    joined_path = OUT_DIR / "joined_online_gate_stack.csv"
    joined.to_csv(joined_path, index=False)

    all_summary_rows = []
    all_symbol_rows = []
    all_side_rows = []

    for blacklist_name, exclude_symbols in EXCLUDE_VARIANTS.items():
        variant_dir = OUT_DIR / blacklist_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        candidates = apply_cfg(joined, exclude_symbols)
        candidates_path = variant_dir / "candidates.csv"
        candidates.to_csv(candidates_path, index=False)

        print()
        print("=" * 120)
        print("BLACKLIST VARIANT:", blacklist_name)
        print("EXCLUDE_SYMBOLS:", sorted(exclude_symbols))
        print("candidate rows :", len(candidates), "symbols:", candidates["symbol"].nunique())

        if len(candidates):
            print("candidate ts   :", candidates["signal_ts"].min(), "->", candidates["signal_ts"].max())
            print("side:")
            print(candidates["side"].value_counts(dropna=False).to_string())

        sim = simulate_candidates(candidates)
        sim_path = variant_dir / "simulated_candidates.csv"
        sim.to_csv(sim_path, index=False)

        print()
        print("SIMULATED:", len(sim), "from candidates:", len(candidates))

        for slot_name, slot_cfg in SLOT_VARIANTS.items():
            max_slots = int(slot_cfg["max_slots"])
            slot_fraction = float(slot_cfg["slot_fraction"])

            slot_dir = variant_dir / slot_name
            slot_dir.mkdir(parents=True, exist_ok=True)

            trades, summary = run_slot_backtest(
                sim_df=sim,
                max_slots=max_slots,
                slot_fraction=slot_fraction,
            )

            trades_path = slot_dir / "trades.csv"
            summary_path = slot_dir / "summary.json"
            by_symbol_path = slot_dir / "trades_by_symbol.csv"
            by_side_path = slot_dir / "trades_by_side.csv"

            trades.to_csv(trades_path, index=False)

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            by_symbol = summarize_by_symbol(trades)
            by_side = summarize_by_side(trades)

            by_symbol.to_csv(by_symbol_path, index=False)
            by_side.to_csv(by_side_path, index=False)

            row = {
                "blacklist_name": blacklist_name,
                "exclude_symbols": ",".join(sorted(exclude_symbols)),
                "slot_name": slot_name,
                "max_slots": max_slots,
                "slot_fraction": slot_fraction,
                "candidate_rows": int(len(candidates)),
                "candidate_symbols": int(candidates["symbol"].nunique()) if len(candidates) else 0,
                "simulated_rows": int(len(sim)),
                "simulated_symbols": int(sim["symbol"].nunique()) if len(sim) else 0,
            }
            row.update(summary)
            all_summary_rows.append(row)

            if len(by_symbol):
                tmp = by_symbol.copy()
                tmp.insert(0, "slot_name", slot_name)
                tmp.insert(0, "blacklist_name", blacklist_name)
                all_symbol_rows.append(tmp)

            if len(by_side):
                tmp = by_side.copy()
                tmp.insert(0, "slot_name", slot_name)
                tmp.insert(0, "blacklist_name", blacklist_name)
                all_side_rows.append(tmp)

            print()
            print("-" * 120)
            print("RESULT:", blacklist_name, slot_name)
            print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_all = pd.DataFrame(all_summary_rows)
    summary_all_path = OUT_DIR / "variant_summary_all.csv"
    summary_all.to_csv(summary_all_path, index=False)

    if all_symbol_rows:
        symbol_all = pd.concat(all_symbol_rows, ignore_index=True)
    else:
        symbol_all = pd.DataFrame()

    if all_side_rows:
        side_all = pd.concat(all_side_rows, ignore_index=True)
    else:
        side_all = pd.DataFrame()

    symbol_all_path = OUT_DIR / "variant_trades_by_symbol_all.csv"
    side_all_path = OUT_DIR / "variant_trades_by_side_all.csv"

    symbol_all.to_csv(symbol_all_path, index=False)
    side_all.to_csv(side_all_path, index=False)

    report = {
        "mode": "online_db_gates_only_plus_m1_market_replay_blacklist_slot_variants",
        "root": str(ROOT),
        "db_dsn": DB_DSN,
        "out_dir": str(OUT_DIR),
        "start_ts": str(START_TS),
        "end_ts": str(END_TS),
        "prod_pair_name": PROD_PAIR_NAME,
        "grid_name": GRID_NAME,
        "exclude_variants": {k: sorted(v) for k, v in EXCLUDE_VARIANTS.items()},
        "slot_variants": SLOT_VARIANTS,
        "gate2_thr": GATE2_THR,
        "gate4_thr": GATE4_THR,
        "gate5_1_thr": GATE5_1_THR,
        "gate5_3_thr": GATE5_3_THR,
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "entry_delay_seconds": ENTRY_DELAY_SECONDS,
        "ttl_hours": TTL_HOURS,
        "fee_per_side": FEE_PER_SIDE,
        "slippage_per_side": SLIPPAGE_PER_SIDE,
        "joined_rows": int(len(joined)),
        "summary_path": str(summary_all_path),
        "symbol_all_path": str(symbol_all_path),
        "side_all_path": str(side_all_path),
    }

    report_path = OUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 120)
    print("FINAL VARIANT SUMMARY")
    if len(summary_all):
        show_cols = [
            "blacklist_name",
            "slot_name",
            "candidate_rows",
            "simulated_rows",
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
        ]
        print(summary_all[show_cols].sort_values("final_capital", ascending=False).to_string(index=False))
    else:
        print("EMPTY")

        print()
        print("=" * 120)
        print("WROTE:")
        print(joined_path)
        print(summary_all_path)
        print(symbol_all_path)
        print(side_all_path)
        print(report_path)

if __name__ == "__main__":
    main()
PY