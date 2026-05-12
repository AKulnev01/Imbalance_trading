from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".")

MODEL_DIR = ROOT / "production/models/gate5/gate5_3"
GRID_DATA_DIR = ROOT / "production/dataset/gate5/gate5_pair_datasets"
M1_DATA_DIR = ROOT / "data/m1_4"

OUT_DIR = ROOT / "production/models/gate5/gate5_3_backtests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY_CSV = OUT_DIR / "summary.csv"
OUT_SUMMARY_JSON = OUT_DIR / "summary.json"

PAIR_LIST = [
    "tp225_sl075__vs__tp100_sl075",
    "tp240_sl060__vs__tp120_sl060",
    "tp240_sl060__vs__tp150_sl060",
]

START_CAPITAL = 100.0
ENTRY_DELAY_SECONDS = 90
TTL_HOURS = 16

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.004

EXCLUDED_SYMBOLS = {
    "AGTUSDT",
    "GORKUSDT",
    "DMCUSDT",
    "MILKUSDT",
    "EPTUSDT",
    "A2ZUSDT",
    "OBTUSDT",
    "AINUSDT",
}
LIQUID_SYMBOLS = {
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "TONUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "UNIUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "SUIUSDT",
    "SEIUSDT",
    "INJUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "FILUSDT",
    "AAVEUSDT",
    "XMRUSDT",
    "TAOUSDT",
    "1000BONKUSDT",
    "1000PEPEUSDT",
}

USE_LIQUID_SYMBOLS_ONLY = True



def parse_pair_name(pair_name: str):
    return pair_name.split("__vs__")


def parse_grid(grid_name: str):
    tp_txt, sl_txt = grid_name.split("_")
    return int(tp_txt.replace("tp", "")) / 100.0, int(sl_txt.replace("sl", "")) / 100.0


def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns: {missing}")


def to_ts(x):
    s = pd.to_datetime(x, errors="coerce", utc=True)
    if isinstance(s, pd.Series):
        return s.dt.tz_convert(None)
    return s.tz_convert(None) if pd.notna(s) else pd.NaT


def find_ts_col(df):
    for c in ["ts", "timestamp", "open_time", "time", "datetime", "dt"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"timestamp column not found; cols={list(df.columns)[:30]}")


def load_m1_symbol(symbol):
    p = M1_DATA_DIR / f"{symbol}.parquet"
    if not p.exists():
        p = M1_DATA_DIR / f"{symbol}_m1.parquet"
    if not p.exists():
        raise FileNotFoundError(f"M1 not found for {symbol}: {p}")

    df = pd.read_parquet(p).copy()
    ts_col = find_ts_col(df)
    if ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})

    require_cols(df, ["ts", "open", "high", "low", "close"], f"m1[{symbol}]")

    df = df[["ts", "open", "high", "low", "close"]].copy()
    df["ts"] = to_ts(df["ts"])

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return (
        df.dropna(subset=["ts", "open", "high", "low", "close"])
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )


def load_pair_predictions(pair_name):
    p = MODEL_DIR / pair_name / "valid_predictions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")

    df = pd.read_parquet(p).copy()

    require_cols(
        df,
        [
            "signal_id",
            "ts",
            "symbol",
            "side",
            "safe_grid_name",
            "agg_grid_name",
            "pred_label",
            "pred_proba",
        ],
        f"valid_predictions[{pair_name}]",
    )

    df["ts"] = to_ts(df["ts"])
    df["side"] = df["side"].astype(str).str.upper()
    df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce").astype(int)
    df["pred_proba"] = pd.to_numeric(df["pred_proba"], errors="coerce")

    df["chosen_grid"] = np.where(
        df["pred_label"].eq(1),
        df["agg_grid_name"],
        df["safe_grid_name"],
    )

    return df.dropna(subset=["ts", "symbol", "side"]).sort_values(["ts", "signal_id"]).reset_index(drop=True)


def load_grid_meta(grid_name):
    p = GRID_DATA_DIR / f"gate5_dataset_{grid_name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")

    df = pd.read_parquet(p).copy()

    side_col = "pred_side" if "pred_side" in df.columns else "side"

    require_cols(df, ["ts", "symbol", side_col, "atr14", "close"], f"grid_dataset[{grid_name}]")

    keep = ["ts", "symbol", side_col, "atr14", "close"]

    if "entry_ts_exec" in df.columns:
        keep.append("entry_ts_exec")
    if "entry_px_ref" in df.columns:
        keep.append("entry_px_ref")

    out = df[keep].copy()
    out = out.rename(columns={side_col: "side"})

    out["ts"] = to_ts(out["ts"])
    out["side"] = out["side"].astype(str).str.upper()
    out["atr14"] = pd.to_numeric(out["atr14"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    if "entry_ts_exec" in out.columns:
        out["entry_ts_exec"] = to_ts(out["entry_ts_exec"])
    else:
        out["entry_ts_exec"] = out["ts"] + pd.to_timedelta(ENTRY_DELAY_SECONDS, unit="s")

    if "entry_px_ref" in out.columns:
        out["entry_px_ref"] = pd.to_numeric(out["entry_px_ref"], errors="coerce")
    else:
        out["entry_px_ref"] = out["close"]

    out = out.dropna(subset=["ts", "symbol", "side", "atr14", "close", "entry_ts_exec"]).copy()
    out = out[out["side"].isin(["LONG", "SHORT"])].copy()

    out = out.dropna(subset=["ts", "symbol", "side", "atr14", "close", "entry_ts_exec"]).copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out = out[out["side"].isin(["LONG", "SHORT"])].copy()
    out = out[~out["symbol"].isin(EXCLUDED_SYMBOLS)].copy()

    if USE_LIQUID_SYMBOLS_ONLY:
        out = out[out["symbol"].isin(LIQUID_SYMBOLS)].copy()
    return (
        out.sort_values(["ts", "symbol", "side"])
        .drop_duplicates(["ts", "symbol", "side"], keep="last")
        .reset_index(drop=True)
    )


def simulate_trade_on_m1(row, m1):
    signal_ts = pd.Timestamp(row["ts"])
    entry_ts_exec = pd.Timestamp(row["entry_ts_exec"])
    side = str(row["side"])
    grid = str(row["chosen_grid"])

    tp_atr, sl_atr = parse_grid(grid)

    atr = float(row["atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    end_ts = entry_ts_exec + pd.Timedelta(hours=TTL_HOURS)
    path = m1[(m1["ts"] >= entry_ts_exec) & (m1["ts"] <= end_ts)].copy()
    if path.empty:
        return None

    entry_bar = path.iloc[0]
    entry_ts_real = pd.Timestamp(entry_bar["ts"])
    entry_px_raw = float(entry_bar["open"])

    if side == "LONG":
        entry_px_exec = entry_px_raw * (1.0 + SLIPPAGE_PER_SIDE)
        tp_px = entry_px_exec + tp_atr * atr
        sl_px = entry_px_exec - sl_atr * atr
    elif side == "SHORT":
        entry_px_exec = entry_px_raw * (1.0 - SLIPPAGE_PER_SIDE)
        tp_px = entry_px_exec - tp_atr * atr
        sl_px = entry_px_exec + sl_atr * atr
    else:
        return None

    exit_ts = None
    exit_px_raw = None
    exit_reason = None

    for r in path.itertuples(index=False):
        high = float(r.high)
        low = float(r.low)

        if side == "LONG":
            hit_tp = high >= tp_px
            hit_sl = low <= sl_px
        else:
            hit_tp = low <= tp_px
            hit_sl = high >= sl_px

        if hit_tp and hit_sl:
            exit_ts = pd.Timestamp(r.ts)
            exit_px_raw = sl_px
            exit_reason = "SL_AMBIGUOUS_SAME_MINUTE"
            break

        if hit_sl:
            exit_ts = pd.Timestamp(r.ts)
            exit_px_raw = sl_px
            exit_reason = "SL"
            break

        if hit_tp:
            exit_ts = pd.Timestamp(r.ts)
            exit_px_raw = tp_px
            exit_reason = "TP"
            break

    if exit_ts is None:
        last = path.iloc[-1]
        exit_ts = pd.Timestamp(last["ts"])
        exit_px_raw = float(last["close"])
        exit_reason = "TTL"

    if side == "LONG":
        exit_px_exec = float(exit_px_raw) * (1.0 - SLIPPAGE_PER_SIDE)
        price_mult_exec_no_fee = exit_px_exec / entry_px_exec
    else:
        exit_px_exec = float(exit_px_raw) * (1.0 + SLIPPAGE_PER_SIDE)
        price_mult_exec_no_fee = entry_px_exec / exit_px_exec

    gross_ret_exec_no_fee = price_mult_exec_no_fee - 1.0

    return {
        "signal_id": int(row["signal_id"]),
        "ts": signal_ts,
        "entry_ts_exec": entry_ts_exec,
        "entry_ts_real": entry_ts_real,
        "exit_ts": exit_ts,
        "symbol": row["symbol"],
        "side": side,
        "chosen_grid": grid,
        "pred_proba": float(row["pred_proba"]),
        "atr14": atr,
        "entry_px_raw": entry_px_raw,
        "entry_px_exec": float(entry_px_exec),
        "exit_px_raw": float(exit_px_raw),
        "exit_px_exec": float(exit_px_exec),
        "tp_px_level": float(tp_px),
        "sl_px_level": float(sl_px),
        "tp_pct_from_entry_exec": float(abs(tp_px / entry_px_exec - 1.0)),
        "sl_pct_from_entry_exec": float(abs(sl_px / entry_px_exec - 1.0)),
        "exit_reason": exit_reason,
        "price_mult_exec_no_fee": float(price_mult_exec_no_fee),
        "gross_ret_exec_no_fee": float(gross_ret_exec_no_fee),
    }


def build_candidate_trades(pair_name, m1_cache):
    pred = load_pair_predictions(pair_name)
    pred["symbol"] = pred["symbol"].astype(str).str.upper()

    excluded_pred_rows = int(pred["symbol"].isin(EXCLUDED_SYMBOLS).sum())
    if excluded_pred_rows:
        print(f"{pair_name}: excluded stale-symbol prediction rows = {excluded_pred_rows}")

    pred = pred[~pred["symbol"].isin(EXCLUDED_SYMBOLS)].copy()

    if USE_LIQUID_SYMBOLS_ONLY:
        non_liquid_rows = int(~pred["symbol"].isin(LIQUID_SYMBOLS).sum()) if False else int(
            (~pred["symbol"].isin(LIQUID_SYMBOLS)).sum())
        if non_liquid_rows:
            print(f"{pair_name}: excluded non-liquid prediction rows = {non_liquid_rows}")
        pred = pred[pred["symbol"].isin(LIQUID_SYMBOLS)].copy()

    parts = []

    for grid_name in sorted(pred["chosen_grid"].dropna().unique()):
        block = pred[pred["chosen_grid"] == grid_name].copy()
        meta = load_grid_meta(grid_name)

        merged = block.merge(
            meta,
            on=["ts", "symbol", "side"],
            how="left",
            suffixes=("", "_grid"),
        )

        miss = int(merged["atr14"].isna().sum())
        if miss:
            raise RuntimeError(f"{pair_name} | {grid_name}: missing merged grid meta rows={miss}")

        parts.append(merged)

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["entry_ts_exec", "signal_id"]).reset_index(drop=True)

    rows = []

    for r in df.to_dict("records"):
        symbol = str(r["symbol"])

        if symbol not in m1_cache:
            m1_cache[symbol] = load_m1_symbol(symbol)

        trade = simulate_trade_on_m1(r, m1_cache[symbol])
        if trade is not None:
            rows.append(trade)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["entry_ts_real", "signal_id"]).reset_index(drop=True)


def run_one_slot_backtest(candidates, start_capital):
    capital = float(start_capital)
    position_free_at = pd.Timestamp.min
    rows = []

    for row in candidates.itertuples(index=False):
        entry_ts = pd.Timestamp(row.entry_ts_real)
        exit_ts = pd.Timestamp(row.exit_ts)

        if entry_ts < position_free_at:
            continue

        capital_before = capital

        entry_fee_usd = capital_before * FEE_PER_SIDE
        capital_after_entry_fee = capital_before - entry_fee_usd

        position_value_before_exit_fee = capital_after_entry_fee * float(row.price_mult_exec_no_fee)

        exit_fee_usd = position_value_before_exit_fee * FEE_PER_SIDE
        capital_after = position_value_before_exit_fee - exit_fee_usd

        net_ret_pct = capital_after / capital_before - 1.0

        rec = row._asdict()
        rec["capital_before"] = float(capital_before)
        rec["entry_fee_usd"] = float(entry_fee_usd)
        rec["capital_after_entry_fee"] = float(capital_after_entry_fee)
        rec["position_value_before_exit_fee"] = float(position_value_before_exit_fee)
        rec["exit_fee_usd"] = float(exit_fee_usd)
        rec["total_fee_usd"] = float(entry_fee_usd + exit_fee_usd)
        rec["capital_after"] = float(capital_after)
        rec["net_ret_pct"] = float(net_ret_pct)

        rows.append(rec)

        capital = capital_after
        position_free_at = exit_ts

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out, {
            "trades_taken": 0,
            "final_capital": float(start_capital),
            "total_return_pct": 0.0,
            "win_rate": None,
            "mean_net_ret_pct": None,
            "median_net_ret_pct": None,
            "max_drawdown_pct": None,
            "tp_count": 0,
            "sl_count": 0,
            "ttl_count": 0,
        }

    equity = out["capital_after"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    wins = out[out["net_ret_pct"] > 0].copy()
    losses = out[out["net_ret_pct"] <= 0].copy()

    gross_profit = float((wins["capital_after"] - wins["capital_before"]).sum()) if len(wins) else 0.0
    gross_loss = float((losses["capital_before"] - losses["capital_after"]).sum()) if len(losses) else 0.0

    profit_factor = None
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss

    return out, {
        "trades_taken": int(len(out)),
        "final_capital": float(out["capital_after"].iloc[-1]),
        "total_return_pct": float(out["capital_after"].iloc[-1] / start_capital - 1.0),
        "win_rate": float((out["net_ret_pct"] > 0).mean()),
        "mean_net_ret_pct": float(out["net_ret_pct"].mean()),
        "median_net_ret_pct": float(out["net_ret_pct"].median()),
        "min_net_ret_pct": float(out["net_ret_pct"].min()),
        "max_net_ret_pct": float(out["net_ret_pct"].max()),
        "avg_win_net_ret_pct": None if len(wins) == 0 else float(wins["net_ret_pct"].mean()),
        "avg_loss_net_ret_pct": None if len(losses) == 0 else float(losses["net_ret_pct"].mean()),
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": float(drawdown.min()),
        "max_capital_before": float(out["capital_before"].max()),
        "max_single_trade_profit_usd": float((out["capital_after"] - out["capital_before"]).max()),
        "max_single_trade_loss_usd": float((out["capital_after"] - out["capital_before"]).min()),
        "tp_count": int((out["exit_reason"] == "TP").sum()),
        "sl_count": int(out["exit_reason"].astype(str).str.startswith("SL").sum()),
        "ttl_count": int((out["exit_reason"] == "TTL").sum()),
        "first_entry_ts": str(out["entry_ts_real"].min()),
        "last_exit_ts": str(out["exit_ts"].max()),
    }

def main():
    reports = []
    m1_cache = {}

    for pair_name in PAIR_LIST:
        print()
        print("=" * 120)
        print("BACKTEST:", pair_name)

        pair_dir = OUT_DIR / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        candidates = build_candidate_trades(pair_name, m1_cache)

        candidates_path = pair_dir / "candidates.parquet"
        candidates.to_parquet(candidates_path, index=False)

        bt_df, metrics = run_one_slot_backtest(candidates, START_CAPITAL)

        trades_path = pair_dir / "trades.parquet"
        report_path = pair_dir / "report.json"

        bt_df.to_parquet(trades_path, index=False)

        safe_grid, agg_grid = parse_pair_name(pair_name)

        report = {
            "pair_name": pair_name,
            "safe_grid": safe_grid,
            "agg_grid": agg_grid,
            "start_capital": START_CAPITAL,
            "entry_delay_seconds": ENTRY_DELAY_SECONDS,
            "ttl_hours_from_entry_ts_exec": TTL_HOURS,
            "fee_per_side": FEE_PER_SIDE,
            "slippage_per_side": SLIPPAGE_PER_SIDE,
            "candidate_rows": int(len(candidates)),
            **metrics,
            "candidates_path": str(candidates_path),
            "trades_path": str(trades_path),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        reports.append(report)

        print("CANDIDATES   :", report["candidate_rows"])
        print("TRADES TAKEN :", report["trades_taken"])
        print("FINAL CAPITAL:", round(report["final_capital"], 6))
        print("TOTAL RETURN :", round(report["total_return_pct"], 6))
        print("WIN RATE     :", None if report["win_rate"] is None else round(report["win_rate"], 6))
        print("MAX DRAWDOWN :", None if report["max_drawdown_pct"] is None else round(report["max_drawdown_pct"], 6))
        print("TP / SL / TTL:", report["tp_count"], report["sl_count"], report["ttl_count"])
        print("WROTE:", trades_path)

    summary = pd.DataFrame(reports).sort_values(
        ["final_capital", "total_return_pct", "trades_taken"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 120)
    print("SUMMARY")
    print(summary.to_string(index=False))
    print()
    print("WROTE:", OUT_SUMMARY_CSV)
    print("WROTE:", OUT_SUMMARY_JSON)


if __name__ == "__main__":
    main()