python - <<'PY'
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/Users/tema/PycharmProjects/ImbalanceSearcher")

SRC_PATH = ROOT / "online" / "result" / "tp100_sl075_reopt_thresholds_h4open" / "simulated_all_dedup_before_thresholds.csv"

OUT_DIR = ROOT / "online" / "result" / "tp100_sl075_APPROVED_g2_063_g4_058_g51_010_g53_055_slip02_OWN_OOF_BLACKLIST"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFG_NAME = "tp100_sl075__OWN_OOF_BLACKLIST__slot1_CHULAN__g2_063__g4_058__g51_010__g53_055__slip02"

PAIR_NAME = "tp225_sl075__vs__tp100_sl075"
GRID_NAME = "tp100_sl075"

GATE2_THR = 0.63
GATE4_THR = 0.58
GATE5_1_THR = 0.10
GATE5_3_THR = 0.55

START_CAPITAL = 100.0
BASE_WORK_CAPITAL = 100.0

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.002

EXPECTED_ENTRY_MINUS_SIGNAL_SECONDS = 14490.0

FOCUS_START = pd.Timestamp("2026-04-14 00:00:00", tz="UTC")
FOCUS_END = pd.Timestamp("2026-05-04 20:01:30", tz="UTC")

# ВАЖНО:
# Здесь нет старого risk5/watch2. Blacklist строится только из статистики этого конфига.
BASE_EXCLUDE_SYMBOLS = set()

# Минимум прошлых сигналов по символу, чтобы вообще иметь право добавить его в blacklist.
MIN_PRIOR_SYMBOL_TRADES_GRID = [1, 2, 3, 4]

# Условия плохого символа.
# Символ попадает в blacklist месяца, если по прошлым месяцам:
# 1) сделок >= min_prior
# 2) суммарный net_ret < 0
# 3) winrate <= bad_winrate
BAD_WINRATE_GRID = [0.0, 25.0, 33.4, 40.0, 50.0]

# Дополнительный вариант: выкидывать худшие N символов по prior pnl.
WORST_N_GRID = [0, 3, 5, 7, 10, 15]

ROUND_DIGITS = 4


def norm_ts(s):
    return pd.to_datetime(s, errors="coerce", utc=True)


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(
            "Не найдена ни одна колонка из списка: %s\nДоступные колонки:\n%s"
            % (candidates, list(df.columns))
        )
    return None


def round_df(df):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(1)
    return out


def max_losing_streak_from_returns(ret_series):
    streak = 0
    best = 0

    for x in ret_series:
        if float(x) <= 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0

    return int(best)


def attach_core_columns(df):
    df = df.copy()

    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()

    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        df[c] = norm_ts(df[c])

    gate2_col = pick_col(df, [
        "gate2_for_side_proba",
        "gate2_for_gate4_side_proba",
        "gate2_proba",
        "gate2_side_proba",
        "gate2_selected_side_proba",
    ])

    gate4_col = pick_col(df, [
        "gate4_confidence",
        "gate4_conf",
    ])

    gate51_col = pick_col(df, [
        "gate5_1_proba",
        "g51_proba",
    ])

    gate53_col = pick_col(df, [
        "gate5_3_proba",
        "gate5_3_choice_confidence",
        "pred_proba",
        "gate5_3_confidence",
    ])

    df["gate2_p"] = pd.to_numeric(df[gate2_col], errors="coerce")
    df["gate4_p"] = pd.to_numeric(df[gate4_col], errors="coerce")
    df["gate51_p"] = pd.to_numeric(df[gate51_col], errors="coerce")
    df["gate53_p"] = pd.to_numeric(df[gate53_col], errors="coerce")

    if "gross_ret" not in df.columns:
        raise RuntimeError("В SRC_PATH нет gross_ret. Нужен файл simulated_all_dedup_before_thresholds.csv.")

    df["gross_ret"] = pd.to_numeric(df["gross_ret"], errors="coerce")

    # Пересчёт честной доходности под fee 0.1% + slip 0.2% на сторону.
    df["net_ret"] = df["gross_ret"] - (2.0 * FEE_PER_SIDE) - (2.0 * SLIPPAGE_PER_SIDE)

    if "exit_reason" not in df.columns:
        df["exit_reason"] = "UNKNOWN"

    df["entry_minus_signal_seconds"] = (df["entry_ts"] - df["signal_ts"]).dt.total_seconds()

    df["signal_strength"] = (
        df["gate2_p"].fillna(0.0)
        + df["gate4_p"].fillna(0.0)
        + df["gate51_p"].fillna(0.0)
        + df["gate53_p"].fillna(0.0)
    )

    df["month"] = df["signal_ts"].dt.strftime("%Y-%m")

    return df


def apply_thresholds(df):
    mask = (
        (df["gate2_p"] >= GATE2_THR)
        & (df["gate4_p"] >= GATE4_THR)
        & (df["gate51_p"] >= GATE5_1_THR)
        & (df["gate53_p"] >= GATE5_3_THR)
        & (~df["symbol"].isin(BASE_EXCLUDE_SYMBOLS))
        & (df["side"].isin(["LONG", "SHORT"]))
        & (df["net_ret"].notna())
        & (df["entry_ts"].notna())
        & (df["exit_ts"].notna())
    )

    return df.loc[mask].copy()


def dedup_same_h4(df):
    if len(df) == 0:
        return df.copy()

    out = df.copy()
    out = out.sort_values(
        [
            "signal_ts",
            "signal_strength",
            "gate2_p",
            "gate4_p",
            "gate51_p",
            "gate53_p",
            "symbol",
            "side",
        ],
        ascending=[True, False, False, False, False, False, True, True],
    )

    out = out.drop_duplicates(["signal_ts"], keep="first").reset_index(drop=True)
    return out


def make_prior_bad_symbols(prior_df, min_prior_trades, bad_winrate_pct, worst_n):
    if len(prior_df) == 0:
        return []

    g = prior_df.groupby("symbol", dropna=False).agg(
        prior_candidates=("symbol", "size"),
        prior_net_ret_sum=("net_ret", "sum"),
        prior_net_ret_mean=("net_ret", "mean"),
        prior_win_rate_pct=("net_ret", lambda s: float((s > 0).mean() * 100.0)),
        prior_tp_count=("exit_reason", lambda s: int((s == "TP").sum())),
        prior_sl_count=("exit_reason", lambda s: int((s == "SL").sum())),
        prior_ttl_count=("exit_reason", lambda s: int((s == "TTL").sum())),
    ).reset_index()

    bad_by_rules = g[
        (g["prior_candidates"] >= int(min_prior_trades))
        & (g["prior_net_ret_sum"] < 0)
        & (g["prior_win_rate_pct"] <= float(bad_winrate_pct))
    ].copy()

    bad_symbols = set(bad_by_rules["symbol"].astype(str).tolist())

    if int(worst_n) > 0:
        worst = g[
            g["prior_candidates"] >= int(min_prior_trades)
        ].sort_values(
            ["prior_net_ret_sum", "prior_win_rate_pct", "prior_candidates"],
            ascending=[True, True, False],
        ).head(int(worst_n))

        bad_symbols.update(worst["symbol"].astype(str).tolist())

    return sorted(bad_symbols)


def build_oof_candidates(base_candidates, min_prior_trades, bad_winrate_pct, worst_n):
    months = sorted(base_candidates["month"].dropna().unique().tolist())

    month_rows = []
    blacklist_rows = []

    for month in months:
        current = base_candidates[base_candidates["month"] == month].copy()
        prior = base_candidates[base_candidates["month"] < month].copy()

        bad_symbols = make_prior_bad_symbols(
            prior_df=prior,
            min_prior_trades=min_prior_trades,
            bad_winrate_pct=bad_winrate_pct,
            worst_n=worst_n,
        )

        filtered = current[~current["symbol"].isin(bad_symbols)].copy()
        filtered = dedup_same_h4(filtered)

        month_rows.append(filtered)

        blacklist_rows.append({
            "month": month,
            "min_prior_trades": int(min_prior_trades),
            "bad_winrate_pct": float(bad_winrate_pct),
            "worst_n": int(worst_n),
            "blacklist_symbols_count": int(len(bad_symbols)),
            "blacklist_symbols": ",".join(bad_symbols),
            "month_raw_candidates": int(len(current)),
            "month_after_blacklist_before_dedup": int(len(current[~current["symbol"].isin(bad_symbols)])),
            "month_after_dedup": int(len(filtered)),
        })

    if month_rows:
        out = pd.concat(month_rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=base_candidates.columns)

    blacklist_df = pd.DataFrame(blacklist_rows)

    out = out.sort_values(["entry_ts", "signal_strength", "symbol", "side"], ascending=[True, False, True, True]).reset_index(drop=True)

    return out, blacklist_df


def run_chulan_backtest(candidates):
    if len(candidates) == 0:
        return pd.DataFrame(), {
            "trades_taken": 0,
            "skipped_by_slot": 0,
            "base_work_capital": BASE_WORK_CAPITAL,
            "work_capital_final": BASE_WORK_CAPITAL,
            "storage_profit_final": 0.0,
            "final_total_capital": BASE_WORK_CAPITAL,
            "total_return_pct": 0.0,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_pct": np.nan,
            "max_losing_streak": 0,
            "tp_count": 0,
            "sl_count": 0,
            "ttl_count": 0,
            "focus_trades": 0,
            "focus_win_rate_pct": np.nan,
            "focus_pnl_usd": 0.0,
            "focus_storage_added": 0.0,
            "top1_symbol": None,
            "top1_symbol_pnl_share_pct": np.nan,
            "top5_symbol_pnl_share_pct": np.nan,
            "max_one_trade_profit_share_pct": np.nan,
            "symbols_traded": 0,
            "trades_per_week": np.nan,
        }

    df = candidates.copy()
    df = df.sort_values(["entry_ts", "signal_strength", "symbol", "side"], ascending=[True, False, True, True]).reset_index(drop=True)

    work_capital = START_CAPITAL
    storage_profit = 0.0

    peak_total = START_CAPITAL
    max_dd = 0.0

    open_position = None
    closed = []
    skipped_by_slot = 0

    def total_capital():
        return float(work_capital + storage_profit)

    def close_position_if_due(now_ts):
        nonlocal work_capital, storage_profit, peak_total, max_dd, open_position, closed

        if open_position is None:
            return

        if pd.Timestamp(open_position["exit_ts"]) > now_ts:
            return

        row = open_position["row"]
        allocation = float(open_position["allocation"])
        net_ret = float(row["net_ret"])
        pnl = allocation * net_ret

        work_capital = allocation + pnl

        moved_to_storage = 0.0
        if work_capital > BASE_WORK_CAPITAL:
            moved_to_storage = work_capital - BASE_WORK_CAPITAL
            storage_profit += moved_to_storage
            work_capital = BASE_WORK_CAPITAL

        r = row.to_dict()
        r["allocation_usd"] = allocation
        r["pnl_usd"] = pnl
        r["moved_to_storage"] = moved_to_storage
        r["work_capital_after"] = work_capital
        r["storage_profit_after"] = storage_profit
        r["total_capital_after"] = total_capital()
        closed.append(r)

        equity = total_capital()
        peak_total = max(peak_total, equity)
        dd = (equity / peak_total) - 1.0 if peak_total > 0 else 0.0
        max_dd = min(max_dd, dd)

        open_position = None

    for _, row in df.iterrows():
        entry_ts = pd.Timestamp(row["entry_ts"])
        exit_ts = pd.Timestamp(row["exit_ts"])

        close_position_if_due(entry_ts)

        if open_position is not None:
            skipped_by_slot += 1
            continue

        allocation = min(work_capital, BASE_WORK_CAPITAL)

        if allocation <= 0:
            skipped_by_slot += 1
            continue

        work_capital -= allocation

        open_position = {
            "exit_ts": exit_ts,
            "allocation": allocation,
            "row": row,
        }

    close_position_if_due(pd.Timestamp("2262-04-11 00:00:00", tz="UTC"))

    trades = pd.DataFrame(closed)

    if len(trades) == 0:
        return trades, {
            "trades_taken": 0,
            "skipped_by_slot": int(skipped_by_slot),
            "base_work_capital": BASE_WORK_CAPITAL,
            "work_capital_final": work_capital,
            "storage_profit_final": storage_profit,
            "final_total_capital": total_capital(),
            "total_return_pct": float((total_capital() / START_CAPITAL - 1.0) * 100.0),
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_pct": float(max_dd * 100.0),
            "max_losing_streak": 0,
            "tp_count": 0,
            "sl_count": 0,
            "ttl_count": 0,
            "focus_trades": 0,
            "focus_win_rate_pct": np.nan,
            "focus_pnl_usd": 0.0,
            "focus_storage_added": 0.0,
            "top1_symbol": None,
            "top1_symbol_pnl_share_pct": np.nan,
            "top5_symbol_pnl_share_pct": np.nan,
            "max_one_trade_profit_share_pct": np.nan,
            "symbols_traded": 0,
            "trades_per_week": np.nan,
        }

    trades["entry_ts"] = norm_ts(trades["entry_ts"])
    trades["exit_ts"] = norm_ts(trades["exit_ts"])
    trades = trades.sort_values(["entry_ts", "symbol", "side"]).reset_index(drop=True)

    wins = trades[trades["net_ret"] > 0]
    losses = trades[trades["net_ret"] <= 0]

    gross_profit = float(wins["pnl_usd"].sum()) if len(wins) else 0.0
    gross_loss_abs = float(abs(losses["pnl_usd"].sum())) if len(losses) else 0.0
    pf = gross_profit / gross_loss_abs if gross_loss_abs > 0 else np.nan

    focus = trades[(trades["entry_ts"] >= FOCUS_START) & (trades["entry_ts"] <= FOCUS_END)].copy()

    if len(focus):
        focus_win_rate_pct = float((focus["net_ret"] > 0).mean() * 100.0)
        focus_pnl_usd = float(focus["pnl_usd"].sum())
        focus_storage_added = float(focus["moved_to_storage"].sum())
    else:
        focus_win_rate_pct = np.nan
        focus_pnl_usd = 0.0
        focus_storage_added = 0.0

    sym = trades.groupby("symbol", dropna=False).agg(
        trades=("symbol", "size"),
        pnl_usd=("pnl_usd", "sum"),
    ).reset_index()

    positive_total = float(sym.loc[sym["pnl_usd"] > 0, "pnl_usd"].sum())
    sym_sorted = sym.sort_values("pnl_usd", ascending=False).reset_index(drop=True)

    if len(sym_sorted) and positive_total > 0:
        top1_symbol = str(sym_sorted.iloc[0]["symbol"])
        top1_share = float(sym_sorted.iloc[0]["pnl_usd"] / positive_total * 100.0)
        top5_share = float(sym_sorted.head(5)["pnl_usd"].clip(lower=0).sum() / positive_total * 100.0)
    else:
        top1_symbol = None
        top1_share = np.nan
        top5_share = np.nan

    pos_pnl = trades.loc[trades["pnl_usd"] > 0, "pnl_usd"]
    if len(pos_pnl) and float(pos_pnl.sum()) > 0:
        max_one_trade_profit_share_pct = float(pos_pnl.max() / pos_pnl.sum() * 100.0)
    else:
        max_one_trade_profit_share_pct = np.nan

    total_days = (trades["entry_ts"].max() - trades["entry_ts"].min()).total_seconds() / 86400.0
    trades_per_week = float(len(trades) / max(total_days / 7.0, 1e-9)) if total_days > 0 else np.nan

    summary = {
        "trades_taken": int(len(trades)),
        "skipped_by_slot": int(skipped_by_slot),
        "base_work_capital": BASE_WORK_CAPITAL,
        "work_capital_final": float(work_capital),
        "storage_profit_final": float(storage_profit),
        "final_total_capital": float(total_capital()),
        "total_return_pct": float((total_capital() / START_CAPITAL - 1.0) * 100.0),
        "win_rate_pct": float((trades["net_ret"] > 0).mean() * 100.0),
        "profit_factor": pf,
        "max_drawdown_pct": float(max_dd * 100.0),
        "max_losing_streak": max_losing_streak_from_returns(trades["net_ret"]),
        "tp_count": int((trades["exit_reason"] == "TP").sum()),
        "sl_count": int((trades["exit_reason"] == "SL").sum()),
        "ttl_count": int((trades["exit_reason"] == "TTL").sum()),
        "focus_trades": int(len(focus)),
        "focus_win_rate_pct": focus_win_rate_pct,
        "focus_pnl_usd": focus_pnl_usd,
        "focus_storage_added": focus_storage_added,
        "top1_symbol": top1_symbol,
        "top1_symbol_pnl_share_pct": top1_share,
        "top5_symbol_pnl_share_pct": top5_share,
        "max_one_trade_profit_share_pct": max_one_trade_profit_share_pct,
        "symbols_traded": int(trades["symbol"].nunique()),
        "trades_per_week": trades_per_week,
        "first_entry_ts": str(trades["entry_ts"].min()),
        "last_exit_ts": str(trades["exit_ts"].max()),
    }

    return trades, summary


def make_symbols_report(trades):
    if len(trades) == 0:
        return pd.DataFrame()

    out = trades.groupby("symbol", dropna=False).agg(
        trades=("symbol", "size"),
        pnl_usd=("pnl_usd", "sum"),
        storage_added=("moved_to_storage", "sum"),
        net_ret_mean=("net_ret", "mean"),
        net_ret_median=("net_ret", "median"),
        tp_count=("exit_reason", lambda s: int((s == "TP").sum())),
        sl_count=("exit_reason", lambda s: int((s == "SL").sum())),
        ttl_count=("exit_reason", lambda s: int((s == "TTL").sum())),
    ).reset_index()

    wr = trades.groupby("symbol")["net_ret"].apply(lambda s: float((s > 0).mean() * 100.0)).reset_index()
    wr = wr.rename(columns={"net_ret": "win_rate_pct"})

    out = out.merge(wr, on="symbol", how="left")
    out = out.sort_values("pnl_usd", ascending=False).reset_index(drop=True)

    return out


def make_monthly_report(trades):
    if len(trades) == 0:
        return pd.DataFrame()

    x = trades.copy()
    x["month"] = x["entry_ts"].dt.strftime("%Y-%m")

    rows = []

    for month, part in x.groupby("month", sort=True):
        rows.append({
            "month": month,
            "trades": int(len(part)),
            "pnl_usd": float(part["pnl_usd"].sum()),
            "storage_added": float(part["moved_to_storage"].sum()),
            "win_rate_pct": float((part["net_ret"] > 0).mean() * 100.0),
            "tp_count": int((part["exit_reason"] == "TP").sum()),
            "sl_count": int((part["exit_reason"] == "SL").sum()),
            "ttl_count": int((part["exit_reason"] == "TTL").sum()),
            "max_losing_streak": max_losing_streak_from_returns(part["net_ret"]),
        })

    return pd.DataFrame(rows)


def score_summary(row):
    trades = float(row.get("trades_taken", 0))
    ret = float(row.get("total_return_pct", 0))
    wr = float(row.get("win_rate_pct", 0))
    pf = float(row.get("profit_factor", 0)) if pd.notna(row.get("profit_factor", np.nan)) else 0
    dd = float(row.get("max_drawdown_pct", -100))
    focus_trades = float(row.get("focus_trades", 0))
    focus_pnl = float(row.get("focus_pnl_usd", 0))
    top1 = float(row.get("top1_symbol_pnl_share_pct", 999)) if pd.notna(row.get("top1_symbol_pnl_share_pct", np.nan)) else 999
    one_trade = float(row.get("max_one_trade_profit_share_pct", 999)) if pd.notna(row.get("max_one_trade_profit_share_pct", np.nan)) else 999
    losing_streak = float(row.get("max_losing_streak", 99))

    score = 0.0

    score += ret * 2.0
    score += wr * 0.6
    score += min(pf, 3.0) * 20.0
    score += dd * 1.5
    score += focus_pnl * 2.0

    if trades < 35:
        score -= 120.0
    elif trades < 45:
        score -= 50.0
    elif trades >= 50:
        score += 30.0

    if focus_trades < 4:
        score -= 70.0
    elif focus_trades >= 7:
        score += 25.0

    if losing_streak > 6:
        score -= 80.0
    elif losing_streak > 4:
        score -= 30.0

    if top1 > 50:
        score -= 50.0
    elif top1 > 35:
        score -= 20.0

    if one_trade > 30:
        score -= 50.0
    elif one_trade > 20:
        score -= 20.0

    return float(score)


def main():
    print("ROOT:", ROOT)
    print("SRC_PATH:", SRC_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("CFG_NAME:", CFG_NAME)
    print("PAIR_NAME:", PAIR_NAME)
    print("GRID_NAME:", GRID_NAME)
    print("THRS: gate2=%.2f gate4=%.2f gate5_1=%.2f gate5_3=%.2f" % (
        GATE2_THR,
        GATE4_THR,
        GATE5_1_THR,
        GATE5_3_THR,
    ))
    print("COSTS: fee_side=%.4f slip_side=%.4f total_cost_pct=%.2f" % (
        FEE_PER_SIDE,
        SLIPPAGE_PER_SIDE,
        (2.0 * FEE_PER_SIDE + 2.0 * SLIPPAGE_PER_SIDE) * 100.0,
    ))
    print("MODE: build OWN rolling OOF blacklist for this exact config")
    print("=" * 120)

    if not SRC_PATH.exists():
        raise RuntimeError("SRC_PATH not found: %s" % SRC_PATH)

    raw = pd.read_csv(SRC_PATH)
    raw = attach_core_columns(raw)

    print("RAW rows:", len(raw), "symbols:", raw["symbol"].nunique())

    bad_time = raw[
        raw["entry_minus_signal_seconds"].notna()
        & (raw["entry_minus_signal_seconds"].round(0) != EXPECTED_ENTRY_MINUS_SIGNAL_SECONDS)
    ].copy()

    if len(bad_time):
        path = OUT_DIR / "_debug_bad_entry_minus_signal_seconds.csv"
        bad_time.to_csv(path, index=False)
        print("WARNING bad entry_minus_signal_seconds:", len(bad_time), "path:", path)

    print("ENTRY_MINUS_SIGNAL_SECONDS:")
    print(raw["entry_minus_signal_seconds"].value_counts(dropna=False).sort_index().to_string())

    base_candidates = apply_thresholds(raw)

    base_candidates_path = OUT_DIR / "base_candidates_after_thresholds_before_own_blacklist.csv"
    base_candidates.to_csv(base_candidates_path, index=False)

    print("=" * 120)
    print("BASE CANDIDATES:", len(base_candidates), "symbols:", base_candidates["symbol"].nunique())
    print("BASE SYMBOLS TOP:")
    print(base_candidates["symbol"].value_counts().head(40).to_string())

    all_summary_rows = []
    all_blacklist_rows = []
    best_row = None
    best_trades = None
    best_blacklist = None
    best_candidates = None

    for min_prior in MIN_PRIOR_SYMBOL_TRADES_GRID:
        for bad_wr in BAD_WINRATE_GRID:
            for worst_n in WORST_N_GRID:
                oof_candidates, blacklist_df = build_oof_candidates(
                    base_candidates=base_candidates,
                    min_prior_trades=min_prior,
                    bad_winrate_pct=bad_wr,
                    worst_n=worst_n,
                )

                trades, summary = run_chulan_backtest(oof_candidates)

                row = {
                    "cfg_name": CFG_NAME,
                    "blacklist_mode": "rolling_month_oof",
                    "min_prior_symbol_trades": int(min_prior),
                    "bad_winrate_pct": float(bad_wr),
                    "worst_n": int(worst_n),
                    "base_candidate_rows": int(len(base_candidates)),
                    "base_candidate_symbols": int(base_candidates["symbol"].nunique()),
                    "oof_candidate_rows": int(len(oof_candidates)),
                    "oof_candidate_symbols": int(oof_candidates["symbol"].nunique()) if len(oof_candidates) else 0,
                }
                row.update(summary)
                row["selection_score"] = score_summary(row)

                all_summary_rows.append(row)

                tmp_bl = blacklist_df.copy()

                if "min_prior_symbol_trades" not in tmp_bl.columns:
                    tmp_bl.insert(0, "min_prior_symbol_trades", int(min_prior))

                if "bad_winrate_pct" not in tmp_bl.columns:
                    tmp_bl.insert(1, "bad_winrate_pct", float(bad_wr))

                if "worst_n" not in tmp_bl.columns:
                    tmp_bl.insert(2, "worst_n", int(worst_n))

                all_blacklist_rows.append(tmp_bl)
                if best_row is None or row["selection_score"] > best_row["selection_score"]:
                    best_row = dict(row)
                    best_trades = trades.copy()
                    best_blacklist = blacklist_df.copy()
                    best_candidates = oof_candidates.copy()

    summary_all = pd.DataFrame(all_summary_rows)
    summary_all = summary_all.sort_values(
        ["selection_score", "final_total_capital", "trades_taken"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    blacklist_all = pd.concat(all_blacklist_rows, ignore_index=True) if all_blacklist_rows else pd.DataFrame()

    best_summary = pd.DataFrame([best_row])
    best_symbols = make_symbols_report(best_trades)
    best_monthly = make_monthly_report(best_trades)

    summary_path = OUT_DIR / "own_oof_blacklist_summary_all.csv"
    summary_round_path = OUT_DIR / "own_oof_blacklist_summary_all_rounded.csv"
    top_path = OUT_DIR / "own_oof_blacklist_top50.csv"
    top_round_path = OUT_DIR / "own_oof_blacklist_top50_rounded.csv"
    blacklist_all_path = OUT_DIR / "own_oof_blacklist_by_month_all.csv"

    best_dir = OUT_DIR / "best_config"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_summary_path = best_dir / "best_summary.csv"
    best_summary_round_path = best_dir / "best_summary_rounded.csv"
    best_trades_path = best_dir / "best_trades.csv"
    best_trades_round_path = best_dir / "best_trades_rounded.csv"
    best_candidates_path = best_dir / "best_candidates_after_own_blacklist.csv"
    best_blacklist_path = best_dir / "best_blacklist_by_month.csv"
    best_monthly_path = best_dir / "best_monthly.csv"
    best_symbols_path = best_dir / "best_symbols.csv"
    best_tail10_path = best_dir / "best_tail10_trades.csv"
    best_focus_path = best_dir / "best_focus_trades.csv"

    summary_all.to_csv(summary_path, index=False)
    round_df(summary_all).to_csv(summary_round_path, index=False)
    summary_all.head(50).to_csv(top_path, index=False)
    round_df(summary_all.head(50)).to_csv(top_round_path, index=False)

    blacklist_all.to_csv(blacklist_all_path, index=False)

    best_summary.to_csv(best_summary_path, index=False)
    round_df(best_summary).to_csv(best_summary_round_path, index=False)

    best_trades.to_csv(best_trades_path, index=False)
    round_df(best_trades).to_csv(best_trades_round_path, index=False)

    best_candidates.to_csv(best_candidates_path, index=False)
    best_blacklist.to_csv(best_blacklist_path, index=False)

    best_monthly.to_csv(best_monthly_path, index=False)
    best_symbols.to_csv(best_symbols_path, index=False)

    round_df(best_trades.tail(10)).to_csv(best_tail10_path, index=False)

    focus = best_trades[
        (best_trades["entry_ts"] >= FOCUS_START)
        & (best_trades["entry_ts"] <= FOCUS_END)
    ].copy()
    round_df(focus).to_csv(best_focus_path, index=False)

    print("=" * 120)
    print("TOP 30 OWN BLACKLIST CONFIGS")
    show_cols = [
        "blacklist_mode",
        "min_prior_symbol_trades",
        "bad_winrate_pct",
        "worst_n",
        "oof_candidate_rows",
        "oof_candidate_symbols",
        "trades_taken",
        "skipped_by_slot",
        "work_capital_final",
        "storage_profit_final",
        "final_total_capital",
        "total_return_pct",
        "win_rate_pct",
        "profit_factor",
        "max_drawdown_pct",
        "max_losing_streak",
        "tp_count",
        "sl_count",
        "ttl_count",
        "focus_trades",
        "focus_win_rate_pct",
        "focus_pnl_usd",
        "focus_storage_added",
        "top1_symbol",
        "top1_symbol_pnl_share_pct",
        "top5_symbol_pnl_share_pct",
        "max_one_trade_profit_share_pct",
        "symbols_traded",
        "trades_per_week",
        "selection_score",
    ]
    print(round_df(summary_all[show_cols].head(30)).to_string(index=False))

    print("=" * 120)
    print("BEST SUMMARY")
    print(round_df(best_summary[show_cols]).to_string(index=False))

    print("=" * 120)
    print("BEST BLACKLIST BY MONTH")
    print(best_blacklist.to_string(index=False))

    print("=" * 120)
    print("BEST MONTHLY")
    if len(best_monthly):
        print(round_df(best_monthly).to_string(index=False))
    else:
        print("EMPTY")

    print("=" * 120)
    print("BEST SYMBOLS TOP 40")
    if len(best_symbols):
        print(round_df(best_symbols.head(40)).to_string(index=False))
    else:
        print("EMPTY")

    print("=" * 120)
    print("BEST FOCUS TRADES")
    if len(focus):
        focus_show_cols = [
            "symbol",
            "side",
            "signal_ts",
            "entry_ts",
            "exit_ts",
            "entry_px",
            "exit_px",
            "tp_px",
            "sl_px",
            "exit_reason",
            "gross_ret",
            "net_ret",
            "allocation_usd",
            "pnl_usd",
            "moved_to_storage",
            "work_capital_after",
            "storage_profit_after",
            "total_capital_after",
            "gate2_p",
            "gate4_p",
            "gate51_p",
            "gate53_p",
            "signal_strength",
        ]
        existing = [c for c in focus_show_cols if c in focus.columns]
        print(round_df(focus[existing]).to_string(index=False))
    else:
        print("EMPTY")

    report = {
        "src_path": str(SRC_PATH),
        "out_dir": str(OUT_DIR),
        "cfg_name": CFG_NAME,
        "pair_name": PAIR_NAME,
        "grid_name": GRID_NAME,
        "thresholds": {
            "gate2": GATE2_THR,
            "gate4": GATE4_THR,
            "gate5_1": GATE5_1_THR,
            "gate5_3": GATE5_3_THR,
        },
        "costs": {
            "fee_per_side": FEE_PER_SIDE,
            "slippage_per_side": SLIPPAGE_PER_SIDE,
            "total_cost_pct": float((2.0 * FEE_PER_SIDE + 2.0 * SLIPPAGE_PER_SIDE) * 100.0),
        },
        "blacklist_logic": "rolling monthly OOF blacklist built only from prior months for this exact config",
        "min_prior_symbol_trades_grid": MIN_PRIOR_SYMBOL_TRADES_GRID,
        "bad_winrate_grid": BAD_WINRATE_GRID,
        "worst_n_grid": WORST_N_GRID,
        "best_summary_path": str(best_summary_path),
        "best_trades_path": str(best_trades_path),
        "best_blacklist_path": str(best_blacklist_path),
        "summary_path": str(summary_path),
    }

    report_path = OUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 120)
    print("WROTE:")
    print(base_candidates_path)
    print(summary_path)
    print(summary_round_path)
    print(top_path)
    print(top_round_path)
    print(blacklist_all_path)
    print(best_summary_path)
    print(best_summary_round_path)
    print(best_trades_path)
    print(best_trades_round_path)
    print(best_candidates_path)
    print(best_blacklist_path)
    print(best_monthly_path)
    print(best_symbols_path)
    print(best_tail10_path)
    print(best_focus_path)
    print(report_path)


if __name__ == "__main__":
    main()
PY