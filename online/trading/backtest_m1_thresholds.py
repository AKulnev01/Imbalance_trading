from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from online.trading import config
from online.trading.db import read_sql

from online.trading.dynamic_blacklist import (
    is_symbol_allowed,
    record_symbol_outcome,
    reset_backtest_outcomes,
)


warnings.filterwarnings("ignore", category=UserWarning)


ROOT = config.ROOT
M1_DB_TABLE = "public.candles_m1"
H4_SECONDS = 4 * 60 * 60


def ask_value(prompt: str, default: Optional[str] = None) -> str:
    if default is None:
        raw = input(prompt + ": ").strip()
    else:
        raw = input(prompt + " [" + str(default) + "]: ").strip()

    if raw:
        return raw

    if default is not None:
        return str(default)

    raise RuntimeError("empty required value: " + prompt)


def ask_float(prompt: str, default: float) -> float:
    return float(ask_value(prompt, str(default)))


def ask_int(prompt: str, default: int) -> int:
    return int(ask_value(prompt, str(default)))

def ask_bool_01(prompt: str, default: int = 0) -> bool:
    value = int(ask_value(prompt, str(default)))
    if value not in [0, 1]:
        raise RuntimeError("bad value for '{}': {}. Use 0 or 1".format(prompt, value))
    return bool(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--start", default="")
    p.add_argument("--end", default="")

    p.add_argument("--gate2", type=float, default=None)
    p.add_argument("--gate4", type=float, default=None)
    p.add_argument("--gate5-1", dest="gate5_1", type=float, default=None)
    p.add_argument("--gate5-3", dest="gate5_3", type=float, default=None)

    p.add_argument("--pair-model-name", default=str(config.PAIR_MODEL_NAME))
    p.add_argument("--grid-name", default=str(config.GRID_NAME))

    p.add_argument("--tp-atr", type=float, default=float(config.TP_ATR))
    p.add_argument("--sl-atr", type=float, default=float(config.SL_ATR))
    p.add_argument("--ttl-hours", type=int, default=int(config.TTL_HOURS))

    p.add_argument("--entry-delay-seconds", type=int, default=int(getattr(config, "BACKTEST_ENTRY_DELAY_SECONDS", 90)))

    p.add_argument("--capital", type=float, default=100.0)
    p.add_argument("--fee-side", type=float, default=float(config.BACKTEST_FEE_SIDE))
    p.add_argument("--slippage-side", type=float, default=float(config.BACKTEST_SLIPPAGE_SIDE))

    p.add_argument("--exclude-symbols", default="")
    p.add_argument("--ignore-db-blacklist", action="store_true")
    p.add_argument("--out-dir", default="")
    p.add_argument("--skip-m1-sync", action="store_true")
    p.add_argument("--m1-sync-timeout-seconds", type=int, default=7200)
    p.add_argument("--write-dynamic-blacklist", type=int, choices=[0, 1], default=None)
    p.add_argument("--reset-backtest-blacklist", type=int, choices=[0, 1], default=None)
    p.add_argument("--blacklist-source", default="")
    p.add_argument("--chulan", type=int, choices=[0, 1], default=None)

    args = p.parse_args()

    if not args.start:
        args.start = ask_value("Введите начало backtest UTC, например 2026-05-01 12:00:00+00:00")

    if not args.end:
        args.end = ask_value("Введите конец backtest UTC, например 2026-05-05 16:00:00+00:00")

    if args.gate2 is None:
        args.gate2 = ask_float("Введите порог Gate2", float(config.GATE2_THR))

    if args.gate4 is None:
        args.gate4 = ask_float("Введите порог Gate4", float(config.GATE4_THR))

    if args.gate5_1 is None:
        args.gate5_1 = ask_float("Введите порог Gate5.1", float(config.GATE5_1_THR))
    if args.gate5_3 is None:
        args.gate5_3 = ask_float("Введите порог Gate5.3", float(config.GATE5_3_THR))

    if args.chulan is None:
        args.chulan = ask_int("Чулан включить? 0=без чулана, 1=с чуланом", 0)

    if int(args.chulan) not in [0, 1]:
        raise RuntimeError("bad chulan value: {}. Use 0 or 1".format(args.chulan))
    if args.write_dynamic_blacklist is None:
        args.write_dynamic_blacklist = int(
            ask_bool_01("Записывать dynamic blacklist? 0=нет, 1=да", 0)
        )

    if args.reset_backtest_blacklist is None:
        args.reset_backtest_blacklist = int(
            ask_bool_01("Сбросить blacklist для этих порогов перед запуском? 0=нет, 1=да", 0)
        )

    args.write_dynamic_blacklist = bool(int(args.write_dynamic_blacklist))
    args.reset_backtest_blacklist = bool(int(args.reset_backtest_blacklist))

    if not str(args.blacklist_source or "").strip():
        args.blacklist_source = default_blacklist_source(args)

    if not args.out_dir:
        start_safe = pd.Timestamp(norm_ts(args.start)).strftime("%Y%m%d_%H%M")
        end_safe = pd.Timestamp(norm_ts(args.end)).strftime("%Y%m%d_%H%M")

        cfg_safe = backtest_config_tag(args)

        args.out_dir = str(
            ROOT
            / "online"
            / "result"
            / "backtest_m1_thresholds"
            / (start_safe + "__" + end_safe + "__" + cfg_safe)
        )

    return args


def norm_ts(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def parse_symbol_list(raw: str) -> List[str]:
    return [x.strip().upper() for x in str(raw or "").split(",") if x.strip()]
def threshold_tag(args: argparse.Namespace) -> str:
    return (
        "g2_%03d_g4_%03d_g51_%03d_g53_%03d"
        % (
            int(round(float(args.gate2) * 1000)),
            int(round(float(args.gate4) * 1000)),
            int(round(float(args.gate5_1) * 1000)),
            int(round(float(args.gate5_3) * 1000)),
        )
    )


def chulan_tag(args: argparse.Namespace) -> str:
    return "chulan{}".format(int(args.chulan))


def backtest_config_tag(args: argparse.Namespace) -> str:
    return threshold_tag(args) + "__" + chulan_tag(args)


def default_blacklist_source(args: argparse.Namespace) -> str:
    return "backtest_approved__" + threshold_tag(args)


def get_chulan_symbols() -> set:
    raw = os.environ.get(
        "BACKTEST_CHULAN_SYMBOLS",
        str(getattr(config, "BACKTEST_CHULAN_SYMBOLS", "")),
    )
    return set(parse_symbol_list(raw))

def split_table_name(table_name: str) -> Tuple[str, str]:
    parts = str(table_name).split(".")
    if len(parts) != 2:
        raise RuntimeError("table name must be schema.table, got: " + str(table_name))
    return parts[0], parts[1]


def get_table_columns(table_name: str) -> List[str]:
    schema, table = split_table_name(table_name)

    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """

    df = read_sql(sql, [schema, table])
    if df.empty:
        return []

    return [str(x) for x in df["column_name"].tolist()]


def pick_existing_column(columns: List[str], candidates: List[str], table_name: str) -> str:
    colset = set(str(c) for c in columns)

    for c in candidates:
        if c in colset:
            return c

    raise RuntimeError(
        table_name
        + ": none of columns found. candidates="
        + str(candidates)
        + " existing="
        + str(columns[:80])
    )


def load_active_db_blacklist() -> Tuple[set, str]:
    candidate_tables = [
        "public.online_symbol_blacklist",
        "public.dynamic_symbol_blacklist",
        "public.symbol_blacklist",
        "public.trading_symbol_blacklist",
        "public.online_dynamic_symbol_blacklist",
    ]

    existing = read_sql(
        """
        SELECT table_schema || '.' || table_name AS table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        """,
        [],
    )

    if existing.empty:
        return set(), "not_found"

    existing_tables = set(existing["table_name"].astype(str).tolist())

    for table_name in candidate_tables:
        if table_name not in existing_tables:
            continue

        cols = get_table_columns(table_name)
        colset = set(cols)

        if "symbol" not in colset:
            continue

        where_parts = []

        if "is_active" in colset:
            where_parts.append("COALESCE(is_active, false) = true")

        if "active" in colset:
            where_parts.append("COALESCE(active, false) = true")

        if "blacklisted" in colset:
            where_parts.append("COALESCE(blacklisted, false) = true")

        if "blocked_until" in colset:
            where_parts.append("(blocked_until IS NULL OR blocked_until >= now())")

        if "cooldown_until" in colset:
            where_parts.append("(cooldown_until IS NULL OR cooldown_until >= now())")

        where_sql = ""
        if where_parts:
            where_sql = "WHERE " + " AND ".join(where_parts)

        df = read_sql(
            """
            SELECT DISTINCT UPPER(symbol) AS symbol
            FROM {table_name}
            {where_sql}
            ORDER BY UPPER(symbol)
            """.format(table_name=table_name, where_sql=where_sql),
            [],
        )

        if df.empty:
            return set(), table_name

        return set(df["symbol"].astype(str).str.upper().tolist()), table_name

    return set(), "not_found"


def read_m1_from_db(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    cache: Dict[str, Optional[pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    symbol = str(symbol).upper()

    cache_key = (
        symbol
        + "|"
        + pd.Timestamp(start_ts).strftime("%Y-%m-%dT%H:%M:%S%z")
        + "|"
        + pd.Timestamp(end_ts).strftime("%Y-%m-%dT%H:%M:%S%z")
    )

    if cache_key in cache:
        return cache[cache_key]

    cols = get_table_columns(M1_DB_TABLE)
    if not cols:
        raise RuntimeError("m1 DB table not found or empty columns: " + M1_DB_TABLE)

    ts_col = pick_existing_column(
        columns=cols,
        candidates=["ts", "entry_ts", "open_time", "timestamp", "time", "datetime", "dt"],
        table_name=M1_DB_TABLE,
    )

    required = ["symbol", ts_col, "open", "high", "low", "close"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(M1_DB_TABLE + ": missing required columns: " + str(missing))

    sql = """
        SELECT
            symbol,
            {ts_col} AS ts,
            open,
            high,
            low,
            close
        FROM {table_name}
        WHERE symbol = %s
          AND {ts_col} >= %s
          AND {ts_col} <= %s
        ORDER BY {ts_col} ASC
    """.format(
        table_name=M1_DB_TABLE,
        ts_col=ts_col,
    )

    df = read_sql(
        sql,
        [
            symbol,
            pd.Timestamp(start_ts).to_pydatetime(),
            pd.Timestamp(end_ts).to_pydatetime(),
        ],
    )

    if df.empty:
        cache[cache_key] = None
        return None

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["ts", "open", "high", "low", "close"])
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )

    cache[cache_key] = df
    return df


# ============================================================
# M1 AUTO SYNC THROUGH online/sync_candles_m1.py
# ============================================================

def sync_m1_for_backtest_all(args: argparse.Namespace) -> Dict[str, object]:
    sync_script = ROOT / "online" / "sync_candles_m1.py"
    report_path = ROOT / "online" / "_sync_candles_m1_report.json"

    report: Dict[str, object] = {
        "enabled": not bool(getattr(args, "skip_m1_sync", False)),
        "source": str(sync_script),
        "report_path": str(report_path),
        "mode": "full_all_symbols",
        "return_code": None,
        "status": "unknown",
        "errors": [],
    }

    if bool(getattr(args, "skip_m1_sync", False)):
        report["status"] = "disabled_by_arg"
        print("=" * 120)
        print("M1_AUTO_SYNC_BEFORE_BACKTEST")
        print("status: disabled_by_arg")
        print("source:", sync_script)
        print("M1_AUTO_SYNC_REPORT:", json.dumps(report, ensure_ascii=False, default=str))
        return report

    if not sync_script.exists():
        raise RuntimeError("sync_candles_m1.py not found: {}".format(sync_script))

    cmd = [
        sys.executable,
        str(sync_script),
    ]

    env = os.environ.copy()
    env["IMB_PROJECT_ROOT"] = str(ROOT)
    env["IMB_DB_DSN"] = os.environ.get(
        "IMB_DB_DSN",
        config.DB_DSN,
    )

    timeout_seconds = int(getattr(args, "m1_sync_timeout_seconds", 7200))

    print("=" * 120)
    print("M1_AUTO_SYNC_BEFORE_BACKTEST")
    print("source:", sync_script)
    print("mode: full_all_symbols")
    print("timeout_seconds:", timeout_seconds)
    print("command:", " ".join(cmd))

    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        report["status"] = "timeout"
        report["errors"].append("sync_candles_m1.py timed out after {} seconds".format(timeout_seconds))

        print("=" * 120)
        print("M1_SYNC_TIMEOUT")
        print("timeout_seconds:", timeout_seconds)
        if exc.stdout:
            print("=" * 120)
            print("M1_SYNC_STDOUT_TAIL_BEFORE_TIMEOUT")
            print("\n".join(str(exc.stdout).splitlines()[-160:]))

        print("M1_AUTO_SYNC_REPORT:", json.dumps(report, ensure_ascii=False, default=str))
        raise RuntimeError(
            "M1 sync timed out before backtest. Увеличь --m1-sync-timeout-seconds или сначала запусти online/sync_candles_m1.py отдельно."
        )

    report["return_code"] = int(p.returncode)

    print("=" * 120)
    print("M1_SYNC_STDOUT_TAIL")
    sync_lines = (p.stdout or "").splitlines()
    print("\n".join(sync_lines[-220:]))

    if p.returncode != 0:
        report["status"] = "sync_failed"
        report["errors"].append("sync_candles_m1.py return_code={}".format(p.returncode))
        print("M1_AUTO_SYNC_REPORT:", json.dumps(report, ensure_ascii=False, default=str))
        raise RuntimeError("M1 sync failed before backtest. See log above.")

    sync_payload = None
    if report_path.exists():
        try:
            sync_payload = json.loads(report_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            report["errors"].append("failed_to_read_sync_report: {}".format(repr(exc)))

    if isinstance(sync_payload, dict):
        report["sync_report"] = {
            "created_at_utc": sync_payload.get("created_at_utc"),
            "symbols_count": sync_payload.get("symbols_count"),
            "fetch_end_ts": sync_payload.get("fetch_end_ts"),
            "total_inserted": sync_payload.get("total_inserted"),
            "status_counts": sync_payload.get("status_counts"),
        }

    report["status"] = "ok"
    print("M1_AUTO_SYNC_REPORT:", json.dumps(report, ensure_ascii=False, default=str))
    return report

def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    start_ts = norm_ts(args.start)
    end_ts = norm_ts(args.end)

    sql = """
        WITH g51 AS (
            SELECT DISTINCT ON (signal_key, grid_name)
                signal_key,
                symbol,
                signal_ts,
                side,
                prod_pair_name,
                grid_name,
                gate4_confidence,
                gate5_1_proba,
                updated_at
            FROM public.online_gate5_1_scores
            WHERE prod_pair_name = %s
              AND grid_name = %s
              AND signal_ts >= %s
              AND signal_ts <= %s
            ORDER BY signal_key, grid_name, updated_at DESC
        ),
        g53 AS (
            SELECT DISTINCT ON (signal_key)
                signal_key,
                chosen_grid_name,
                pred_proba AS gate5_3_proba,
                updated_at
            FROM public.online_gate5_3_decisions
            WHERE chosen_grid_name = %s
            ORDER BY signal_key, updated_at DESC
        )
        SELECT
            g51.signal_key,
            g51.symbol,
            g51.signal_ts,
            g51.side,

            f.close AS h4_close,
            f.atr14 AS atr14,

            CASE
                WHEN UPPER(g51.side) = 'LONG' THEN g2.up_reach_high_proba
                WHEN UPPER(g51.side) = 'SHORT' THEN g2.dn_reach_high_proba
                ELSE NULL
            END AS gate2_for_side_proba,

            g2.up_reach_high_proba AS gate2_up,
            g2.dn_reach_high_proba AS gate2_dn,

            g51.gate4_confidence,
            g51.gate5_1_proba,
            g53.gate5_3_proba,

            (
                COALESCE(
                    CASE
                        WHEN UPPER(g51.side) = 'LONG' THEN g2.up_reach_high_proba
                        WHEN UPPER(g51.side) = 'SHORT' THEN g2.dn_reach_high_proba
                        ELSE NULL
                    END,
                    0.0
                )
                + COALESCE(g51.gate4_confidence, 0.0)
                + COALESCE(g51.gate5_1_proba, 0.0)
                + COALESCE(g53.gate5_3_proba, 0.0)
            ) AS signal_strength

        FROM g51

        INNER JOIN g53
            ON g53.signal_key = g51.signal_key

        LEFT JOIN public.online_gate4_features f
            ON f.symbol = g51.symbol
           AND f.entry_ts = g51.signal_ts

        LEFT JOIN public.online_gate2_predictions g2
            ON g2.symbol = g51.symbol
           AND g2.entry_ts = g51.signal_ts

        WHERE f.close IS NOT NULL
          AND f.atr14 IS NOT NULL
          AND f.close > 0
          AND f.atr14 > 0

        ORDER BY g51.signal_ts ASC, signal_strength DESC, g51.symbol ASC
    """

    df = read_sql(
        sql,
        [
            str(args.pair_model_name),
            str(args.grid_name),
            start_ts.to_pydatetime(),
            end_ts.to_pydatetime(),
            str(args.grid_name),
        ],
    )

    if df.empty:
        return df

    df["signal_key"] = df["signal_key"].astype(str)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["side"] = df["side"].astype(str).str.upper()
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True, errors="coerce")

    for c in [
        "h4_close",
        "atr14",
        "gate2_for_side_proba",
        "gate2_up",
        "gate2_dn",
        "gate4_confidence",
        "gate5_1_proba",
        "gate5_3_proba",
        "signal_strength",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "signal_key",
            "symbol",
            "side",
            "signal_ts",
            "h4_close",
            "atr14",
            "gate2_for_side_proba",
            "gate4_confidence",
            "gate5_1_proba",
            "gate5_3_proba",
            "signal_strength",
        ]
    ).copy()

    df = df[df["side"].isin(["LONG", "SHORT"])].copy()
    df = df[df["h4_close"] > 0].copy()
    df = df[df["atr14"] > 0].copy()

    return df.reset_index(drop=True)


def apply_thresholds(df: pd.DataFrame, args: argparse.Namespace, db_blacklist: set) -> pd.DataFrame:
    if df.empty:
        return df

    out = df[
        (df["gate2_for_side_proba"] >= float(args.gate2))
        & (df["gate4_confidence"] >= float(args.gate4))
        & (df["gate5_1_proba"] >= float(args.gate5_1))
        & (df["gate5_3_proba"] >= float(args.gate5_3))
    ].copy()

    excluded = set(parse_symbol_list(args.exclude_symbols))

    if not bool(args.ignore_db_blacklist):
        excluded |= set(str(x).upper() for x in db_blacklist)

    if excluded:
        out = out[~out["symbol"].astype(str).str.upper().isin(excluded)].copy()

    return out.reset_index(drop=True)
def keep_best_signal_per_h4(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.sort_values(
        [
            "signal_ts",
            "signal_strength",
            "gate4_confidence",
            "gate2_for_side_proba",
            "gate5_1_proba",
            "gate5_3_proba",
            "symbol",
        ],
        ascending=[True, False, False, False, False, False, True],
    ).copy()

    out = out.drop_duplicates(["signal_ts"], keep="first")
    return out.reset_index(drop=True)


def calc_tp_sl(side: str, entry_px: float, atr14: float, tp_atr: float, sl_atr: float) -> Tuple[float, float]:
    side_u = str(side).upper()

    if side_u == "LONG":
        return entry_px + tp_atr * atr14, entry_px - sl_atr * atr14

    if side_u == "SHORT":
        return entry_px - tp_atr * atr14, entry_px + sl_atr * atr14

    raise RuntimeError("bad side: {}".format(side))


def simulate_one(
    row: pd.Series,
    args: argparse.Namespace,
    m1_cache: Dict[str, Optional[pd.DataFrame]],
    m1_load_start_ts: pd.Timestamp,
    m1_load_end_ts: pd.Timestamp,
) -> Optional[Dict[str, object]]:
    symbol = str(row["symbol"]).upper()
    side = str(row["side"]).upper()

    signal_ts = pd.to_datetime(row["signal_ts"], utc=True)

    entry_ts = signal_ts + pd.Timedelta(seconds=H4_SECONDS + int(args.entry_delay_seconds))
    ttl_end_ts = entry_ts + pd.Timedelta(hours=int(args.ttl_hours))

    m1 = read_m1_from_db(
        symbol=symbol,
        start_ts=m1_load_start_ts,
        end_ts=m1_load_end_ts,
        cache=m1_cache,
    )

    if m1 is None or m1.empty:
        return None

    window = m1[(m1["ts"] >= entry_ts) & (m1["ts"] <= ttl_end_ts)].copy()
    if window.empty:
        return None

    entry_px = float(window.iloc[0]["open"])
    atr14 = float(row["atr14"])

    if not np.isfinite(entry_px) or entry_px <= 0:
        return None

    if not np.isfinite(atr14) or atr14 <= 0:
        return None

    tp_px, sl_px = calc_tp_sl(
        side=side,
        entry_px=entry_px,
        atr14=atr14,
        tp_atr=float(args.tp_atr),
        sl_atr=float(args.sl_atr),
    )

    exit_reason = "TTL"
    exit_ts = pd.to_datetime(window.iloc[-1]["ts"], utc=True)
    exit_px = float(window.iloc[-1]["close"])

    for _, bar in window.iterrows():
        ts = pd.to_datetime(bar["ts"], utc=True)
        high = float(bar["high"])
        low = float(bar["low"])

        if side == "LONG":
            tp_hit = high >= tp_px
            sl_hit = low <= sl_px
        else:
            tp_hit = low <= tp_px
            sl_hit = high >= sl_px

        if tp_hit and sl_hit:
            exit_reason = "SL_SAME_M1"
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

    net_ret = gross_ret - 2.0 * float(args.fee_side) - 2.0 * float(args.slippage_side)

    out = row.to_dict()
    out.update(
        {
            "entry_ts": entry_ts,
            "ttl_end_ts": ttl_end_ts,
            "exit_ts": exit_ts,
            "entry_px": float(entry_px),
            "tp_px": float(tp_px),
            "sl_px": float(sl_px),
            "exit_px": float(exit_px),
            "exit_reason": exit_reason,
            "gross_ret": float(gross_ret),
            "net_ret": float(net_ret),
        }
    )

    return out


def simulate_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    m1_cache: Dict[str, Optional[pd.DataFrame]] = {}

    if df.empty:
        return pd.DataFrame()

    signal_min = pd.to_datetime(df["signal_ts"], utc=True, errors="coerce").min()
    signal_max = pd.to_datetime(df["signal_ts"], utc=True, errors="coerce").max()

    m1_load_start_ts = signal_min + pd.Timedelta(seconds=H4_SECONDS + int(args.entry_delay_seconds))
    m1_load_end_ts = (
        signal_max
        + pd.Timedelta(seconds=H4_SECONDS + int(args.entry_delay_seconds))
        + pd.Timedelta(hours=int(args.ttl_hours))
    )

    for _, row in df.iterrows():
        sim = simulate_one(
            row=row,
            args=args,
            m1_cache=m1_cache,
            m1_load_start_ts=m1_load_start_ts,
            m1_load_end_ts=m1_load_end_ts,
        )

        if sim is not None:
            rows.append(sim)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce")
    out["exit_ts"] = pd.to_datetime(out["exit_ts"], utc=True, errors="coerce")
    out["signal_ts"] = pd.to_datetime(out["signal_ts"], utc=True, errors="coerce")

    return out.sort_values(["entry_ts", "signal_strength"], ascending=[True, False]).reset_index(drop=True)

def run_slot1_full_compound(sim: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if sim.empty:
        return sim, {
            "trades_taken": 0,
            "final_capital": float(args.capital),
            "total_return_pct": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
            "skipped_overlap": 0,
            "skipped_dynamic_blacklist": 0,
            "chulan": int(args.chulan),
            "blacklist_source": str(args.blacklist_source),
        }

    capital = float(args.capital)
    peak = capital
    max_dd = 0.0
    position_free_ts: Optional[pd.Timestamp] = None
    trades = []
    skipped_overlap = 0
    skipped_dynamic_blacklist = 0

    write_dynamic_blacklist = bool(getattr(args, "write_dynamic_blacklist", False))
    blacklist_source = str(getattr(args, "blacklist_source", "backtest_approved"))

    for _, row in sim.sort_values(["entry_ts", "signal_strength"], ascending=[True, False]).iterrows():
        entry_ts = pd.to_datetime(row["entry_ts"], utc=True)
        exit_ts = pd.to_datetime(row["exit_ts"], utc=True)

        if position_free_ts is not None and entry_ts < position_free_ts:
            skipped_overlap += 1
            continue

        allowed, dynamic_reason, dynamic_stats = is_symbol_allowed(
            symbol=str(row["symbol"]),
            source=blacklist_source,
            now_ts=entry_ts,
        )

        if not allowed:
            skipped_dynamic_blacklist += 1
            continue

        capital_before = capital
        net_ret = float(row["net_ret"])
        pnl_usd = capital_before * net_ret
        capital = capital_before + pnl_usd

        peak = max(peak, capital)
        dd = (capital / peak) - 1.0 if peak > 0 else 0.0
        max_dd = min(max_dd, dd)

        r = row.to_dict()
        r["capital_before"] = float(capital_before)
        r["pnl_usd"] = float(pnl_usd)
        r["capital_after"] = float(capital)
        r["drawdown_pct"] = float(dd)
        r["dynamic_symbol_allowed"] = bool(allowed)
        r["dynamic_symbol_reason"] = str(dynamic_reason)
        r["dynamic_symbol_stats"] = json.dumps(dynamic_stats, ensure_ascii=False, default=str)

        trades.append(r)
        position_free_ts = exit_ts

        if write_dynamic_blacklist:
            record_symbol_outcome(
                source=blacklist_source,
                source_run_id=str(Path(args.out_dir).name),
                signal_key=str(row["signal_key"]),
                symbol=str(row["symbol"]),
                side=str(row["side"]),
                signal_ts=pd.to_datetime(row["signal_ts"], utc=True, errors="coerce"),
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                net_ret=float(net_ret),
                exit_reason=str(row.get("exit_reason")),
                pair_model_name=str(args.pair_model_name),
                grid_name=str(args.grid_name),
            )

            is_symbol_allowed(
                symbol=str(row["symbol"]),
                now_ts=exit_ts + pd.Timedelta(seconds=1),
                source=blacklist_source,
            )

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df, {
            "trades_taken": 0,
            "final_capital": float(args.capital),
            "total_return_pct": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
            "skipped_overlap": int(skipped_overlap),
            "skipped_dynamic_blacklist": int(skipped_dynamic_blacklist),
            "dynamic_blacklist_written": bool(write_dynamic_blacklist),
            "blacklist_source": blacklist_source,
            "chulan": int(args.chulan),
        }

    wins = trades_df[trades_df["net_ret"] > 0]
    losses = trades_df[trades_df["net_ret"] <= 0]

    gross_profit = float(wins["pnl_usd"].sum()) if len(wins) else 0.0
    gross_loss_abs = float(abs(losses["pnl_usd"].sum())) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else None

    summary = {
        "trades_taken": int(len(trades_df)),
        "final_capital": float(capital),
        "total_return_pct": float((capital / float(args.capital)) - 1.0),
        "win_rate": float((trades_df["net_ret"] > 0).mean()),
        "profit_factor": profit_factor,
        "max_drawdown_pct": float(max_dd),
        "mean_net_ret": float(trades_df["net_ret"].mean()),
        "median_net_ret": float(trades_df["net_ret"].median()),
        "tp_count": int((trades_df["exit_reason"] == "TP").sum()),
        "sl_count": int(trades_df["exit_reason"].astype(str).str.startswith("SL").sum()),
        "ttl_count": int((trades_df["exit_reason"] == "TTL").sum()),
        "skipped_overlap": int(skipped_overlap),
        "skipped_dynamic_blacklist": int(skipped_dynamic_blacklist),
        "dynamic_blacklist_written": bool(write_dynamic_blacklist),
        "blacklist_source": blacklist_source,
        "chulan": int(args.chulan),
        "first_entry_ts": str(trades_df["entry_ts"].min()),
        "last_exit_ts": str(trades_df["exit_ts"].max()),
    }

    return trades_df, summary

def save_outputs(
    raw: pd.DataFrame,
    passed: pd.DataFrame,
    selected: pd.DataFrame,
    sim: pd.DataFrame,
    trades: pd.DataFrame,
    summary: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(out_dir / "raw_candidates.csv", index=False)
    passed.to_csv(out_dir / "passed_thresholds.csv", index=False)
    selected.to_csv(out_dir / "selected_one_per_h4.csv", index=False)
    sim.to_csv(out_dir / "simulated_candidates_m1.csv", index=False)

    trades_path = out_dir / "trades.csv"
    trades.to_csv(trades_path, index=False)

    payload = {
        "args": vars(args),
        "summary": summary,
        "rows": {
            "raw_candidates": int(len(raw)),
            "passed_thresholds": int(len(passed)),
            "selected_one_per_h4": int(len(selected)),
            "simulated_candidates_m1": int(len(sim)),
            "trades": int(len(trades)),
        },
    }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print("WROTE:", trades_path)

def print_report(
    raw: pd.DataFrame,
    passed: pd.DataFrame,
    selected: pd.DataFrame,
    sim: pd.DataFrame,
    trades: pd.DataFrame,
    summary: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    print("=" * 120)
    print("M1 BACKTEST THRESHOLDS")
    print("=" * 120)
    print("period:", args.start, "->", args.end)
    print("pair_model_name:", args.pair_model_name)
    print("grid_name:", args.grid_name)
    print("gate2:", args.gate2)
    print("gate4:", args.gate4)
    print("gate5_1:", args.gate5_1)
    print("gate5_3:", args.gate5_3)
    print("chulan:", int(args.chulan))
    print("blacklist_source:", str(args.blacklist_source))
    print("tp_atr:", args.tp_atr)
    print("sl_atr:", args.sl_atr)
    print("ttl_hours:", args.ttl_hours)
    print("entry_delay_seconds:", args.entry_delay_seconds)
    print("fee_side:", args.fee_side)
    print("slippage_side:", args.slippage_side)
    print("m1_source:", M1_DB_TABLE)
    print("ignore_db_blacklist:", bool(args.ignore_db_blacklist))
    print("-" * 120)
    print("raw_candidates:", len(raw))
    print("passed_thresholds:", len(passed))
    print("selected_one_per_h4:", len(selected))
    print("simulated_candidates_m1:", len(sim))
    print("-" * 120)

    if not summary:
        print("NO_SUMMARY")
        return

    for k in [
        "trades_taken",
        "final_capital",
        "total_return_pct",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "mean_net_ret",
        "median_net_ret",
        "tp_count",
        "sl_count",
        "ttl_count",
        "skipped_overlap",
        "skipped_dynamic_blacklist",
        "dynamic_blacklist_written",
        "blacklist_source",
        "chulan",
        "first_entry_ts",
        "last_exit_ts",
    ]:
        if k in summary:
            v = summary[k]
            if isinstance(v, float):
                print(k + ":", round(v, 6))
            else:
                print(k + ":", v)

    if trades.empty:
        print("-" * 120)
        print("NO_TRADES")
        return

    show_cols = [
        "signal_ts",
        "entry_ts",
        "exit_ts",
        "symbol",
        "side",
        "entry_px",
        "tp_px",
        "sl_px",
        "exit_px",
        "exit_reason",
        "net_ret",
        "capital_after",
        "gate2_for_side_proba",
        "gate4_confidence",
        "gate5_1_proba",
        "gate5_3_proba",
    ]

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 100)

    print("-" * 120)
    print("LAST 30 TRADES")
    print(trades[show_cols].tail(30).to_string(index=False))


def main() -> None:
    args = parse_args()
    if bool(getattr(args, "reset_backtest_blacklist", False)):
        reset_backtest_outcomes(source=str(getattr(args, "blacklist_source", "backtest_approved")))

    db_blacklist, db_blacklist_source = load_active_db_blacklist()

    m1_sync_report = sync_m1_for_backtest_all(args)

    raw = load_candidates(args)
    passed = apply_thresholds(raw, args, db_blacklist=db_blacklist)
    selected = keep_best_signal_per_h4(passed)

    sim = simulate_candidates(selected, args)
    trades, summary = run_slot1_full_compound(sim, args)

    save_outputs(raw, passed, selected, sim, trades, summary, args)
    print_report(raw, passed, selected, sim, trades, summary, args)

    print("-" * 120)
    print("DB_ONLY_MODE:", True)
    print("M1_DB_TABLE:", M1_DB_TABLE)
    print("M1_AUTO_SYNC:", json.dumps(m1_sync_report, ensure_ascii=False, default=str))
    print("LEGACY_DB_BLACKLIST_SOURCE:", db_blacklist_source)
    print("LEGACY_DB_BLACKLIST_SYMBOLS_COUNT:", len(db_blacklist))
    print("LEGACY_DB_BLACKLIST_SYMBOLS:", ",".join(sorted(db_blacklist)) if db_blacklist else "")
    print("DYNAMIC_BLACKLIST_SOURCE:", str(args.blacklist_source))
    print("OUT_DIR:", args.out_dir)

if __name__ == "__main__":
    main()
