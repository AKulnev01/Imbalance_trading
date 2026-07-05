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
from online.trading.excel.backtest_xlsx_export import export_backtest_xlsx

from online.trading.dynamic_blacklist import (
    is_symbol_allowed,
    record_symbol_outcome,
    reset_backtest_outcomes,
)


warnings.filterwarnings("ignore", category=UserWarning)


ROOT = config.ROOT
M1_DB_TABLE = "public.candles_m1"
H4_SECONDS = 4 * 60 * 60

# Backtest execution rule:
# if H4 candle closes at 16:00, entry is M1 open at 16:01.
BACKTEST_SECOND_MINUTE_OPEN_DELAY_SECONDS = 60

SIDE_AWARE_WHITELIST: Dict[str, List[str]] = config.SIDE_AWARE_WHITELIST
CONDITIONAL_SIDE_AWARE_WHITELIST: Dict[str, Dict[str, Dict[str, float]]] = getattr(
    config,
    "CONDITIONAL_SIDE_AWARE_WHITELIST",
    {},
)


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
    p.add_argument(
        "--symbols",
        default="",
        help=(
            "Comma-separated symbols with optional side filter. "
            "Examples: BTCUSDT or ADAUSDT L,BTCUSDT S,ETHUSDT. "
            "Side aliases: L/LONG/BUY and S/SHORT/SELL."
        ),
    )

    p.add_argument("--gate2", type=float, default=None)
    p.add_argument("--gate2-side-margin-min", dest="gate2_side_margin_min", type=float, default=0.0)
    p.add_argument("--gate4", type=float, default=None)
    p.add_argument("--gate5-1", dest="gate5_1", type=float, default=None)
    p.add_argument("--gate5-3", dest="gate5_3", type=float, default=None)

    p.add_argument("--pair-model-name", default=str(config.PAIR_MODEL_NAME))
    p.add_argument("--grid-name", default=str(config.GRID_NAME))

    p.add_argument("--tp-atr", type=float, default=float(config.TP_ATR))
    p.add_argument("--sl-atr", type=float, default=float(config.SL_ATR))
    p.add_argument("--ttl-hours", type=int, default=int(config.TTL_HOURS))

    p.add_argument(
        "--entry-delay-seconds",
        type=int,
        default=BACKTEST_SECOND_MINUTE_OPEN_DELAY_SECONDS,
    )

    p.add_argument("--capital", type=float, default=100.0)
    p.add_argument("--fee-side", type=float, default=float(config.BACKTEST_FEE_SIDE))
    p.add_argument("--slippage-side", type=float, default=float(config.BACKTEST_SLIPPAGE_SIDE))
    p.add_argument(
        "--max-full-sl-capital-risk",
        type=float,
        default=0.07,
        help="Max capital loss for full MAIN_SL. Example: 0.07 = 7%%. Use 0 to disable.",
    )

    p.add_argument("--exclude-symbols", default="")
    p.add_argument("--ignore-db-blacklist", action="store_true")
    p.add_argument("--out-dir", default="")
    p.add_argument("--skip-m1-sync", action="store_true")
    p.add_argument("--m1-sync-timeout-seconds", type=int, default=7200)
    p.add_argument("--write-dynamic-blacklist", type=int, choices=[0, 1], default=None)
    p.add_argument("--reset-backtest-blacklist", type=int, choices=[0, 1], default=None)
    p.add_argument("--blacklist-source", default="")
    p.add_argument("--chulan", type=int, choices=[0, 1], default=None)
    p.add_argument("--side-aware-whitelist", type=int, choices=[0, 1], default=None)
    p.add_argument(
        "--conditional-side-aware-whitelist",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use conditional side-aware whitelist with Gate2 side margin. Default: 1.",
    )
    p.add_argument("--slots", type=int, default=None)

    args = p.parse_args()

    symbol_research_requested = bool(str(args.symbols or "").strip())

    if symbol_research_requested:
        if args.chulan is None:
            args.chulan = 0
        if args.side_aware_whitelist is None:
            args.side_aware_whitelist = 0
        args.conditional_side_aware_whitelist = 0
        if args.slots is None:
            args.slots = 1
        if args.write_dynamic_blacklist is None:
            args.write_dynamic_blacklist = 0
        if args.reset_backtest_blacklist is None:
            args.reset_backtest_blacklist = 0
        args.ignore_db_blacklist = True

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

    if args.side_aware_whitelist is None:
        args.side_aware_whitelist = ask_int(
            "Side-aware whitelist включить? 0=нет, 1=да",
            0,
        )

    if int(args.side_aware_whitelist) not in [0, 1]:
        raise RuntimeError(
            "bad side_aware_whitelist value: {}. Use 0 or 1".format(
                args.side_aware_whitelist
            )
        )

    if args.slots is None:
        args.slots = ask_int(
            "Количество слотов",
            1,
        )

    if int(args.slots) != 1:
        raise RuntimeError(
            "Сейчас в этом backtest реализован только slot1. Получено slots={}".format(
                args.slots
            )
        )

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
    args.side_aware_whitelist = bool(int(args.side_aware_whitelist))
    args.conditional_side_aware_whitelist = bool(int(args.conditional_side_aware_whitelist))
    args.slots = int(args.slots)

    args = apply_symbol_research_mode(args)

    if float(args.max_full_sl_capital_risk) < 0:
        raise RuntimeError(
            "bad max_full_sl_capital_risk: {}. Use >= 0.".format(
                args.max_full_sl_capital_risk
            )
        )

    if float(getattr(args, "gate2_side_margin_min", 0.0) or 0.0) < 0:
        raise RuntimeError(
            "bad gate2_side_margin_min: {}. Use >= 0.".format(
                getattr(args, "gate2_side_margin_min", None)
            )
        )

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


def normalize_symbol_filter_token(raw: str) -> str:
    symbol = str(raw or "").strip().upper()
    symbol = symbol.replace("/", "")
    symbol = symbol.replace("-", "")
    symbol = symbol.replace("_", "")

    if not symbol:
        raise RuntimeError("empty symbol in --symbols")

    return symbol


def normalize_side_filter_token(raw: str) -> str:
    side = str(raw or "").strip().upper()

    if side in ["L", "LONG", "BUY"]:
        return "LONG"

    if side in ["S", "SHORT", "SELL"]:
        return "SHORT"

    raise RuntimeError(
        "bad side in --symbols: {}. Use L/S or LONG/SHORT.".format(raw)
    )


def parse_symbol_side_filters(raw: str) -> Dict[str, object]:
    text = str(raw or "").strip()

    result: Dict[str, object] = {
        "enabled": False,
        "symbols": [],
        "by_symbol": {},
    }

    if not text:
        return result

    chunks = [
        x.strip()
        for x in text.replace(";", ",").split(",")
        if x.strip()
    ]

    symbols: List[str] = []
    by_symbol: Dict[str, Optional[List[str]]] = {}

    for chunk in chunks:
        normalized_chunk = (
            chunk
            .replace(":", " ")
            .replace("|", " ")
            .replace("=", " ")
        )

        parts = [x.strip() for x in normalized_chunk.split() if x.strip()]

        if not parts:
            continue

        if len(parts) > 2:
            raise RuntimeError(
                "bad --symbols item: {}. Use SYMBOL or SYMBOL SIDE.".format(chunk)
            )

        symbol = normalize_symbol_filter_token(parts[0])
        side: Optional[str] = None

        if len(parts) == 2:
            side = normalize_side_filter_token(parts[1])

        if symbol not in symbols:
            symbols.append(symbol)

        if symbol not in by_symbol:
            by_symbol[symbol] = None if side is None else [side]
            continue

        current = by_symbol[symbol]

        if current is None:
            continue

        if side is None:
            by_symbol[symbol] = None
            continue

        if side not in current:
            current.append(side)

    result["enabled"] = bool(symbols)
    result["symbols"] = symbols
    result["by_symbol"] = by_symbol

    return result


def format_symbol_side_filters(raw: str) -> str:
    filters = parse_symbol_side_filters(raw)

    if not bool(filters["enabled"]):
        return ""

    by_symbol = filters["by_symbol"]
    symbols = filters["symbols"]

    items = []

    for symbol in symbols:
        sides = by_symbol.get(symbol)

        if sides is None:
            items.append(symbol)
            continue

        items.append("{} {}".format(symbol, "/".join(sides)))

    return ",".join(items)


def apply_symbol_side_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filters = parse_symbol_side_filters(str(getattr(args, "symbols", "") or ""))

    if not bool(filters["enabled"]):
        return df

    if df.empty:
        return df

    by_symbol = filters["by_symbol"]

    keep_mask = []

    for _, row in df.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        side = normalize_side_for_whitelist(str(row.get("side") or ""))

        if symbol not in by_symbol:
            keep_mask.append(False)
            continue

        allowed_sides = by_symbol[symbol]

        if allowed_sides is None:
            keep_mask.append(True)
            continue

        keep_mask.append(side in allowed_sides)

    out = df[pd.Series(keep_mask, index=df.index)].copy()
    return out.reset_index(drop=True)
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


def side_aware_whitelist_tag(args: argparse.Namespace) -> str:
    return "sidewl{}".format(int(bool(args.side_aware_whitelist)))


def slots_tag(args: argparse.Namespace) -> str:
    return "slot{}".format(int(args.slots))


def conditional_side_aware_whitelist_tag(args: argparse.Namespace) -> str:
    return "condwl{}".format(int(bool(getattr(args, "conditional_side_aware_whitelist", True))))


def risk_sizing_tag(args: argparse.Namespace) -> str:
    max_risk = float(getattr(args, "max_full_sl_capital_risk", 0.0) or 0.0)

    if max_risk <= 0:
        return "riskcap_off"

    return "riskcap_%04d" % int(round(max_risk * 10000))


def backtest_config_tag(args: argparse.Namespace) -> str:
    return "__".join(
        [
            threshold_tag(args),
            chulan_tag(args),
            side_aware_whitelist_tag(args),
            conditional_side_aware_whitelist_tag(args),
            slots_tag(args),
            risk_sizing_tag(args),
        ]
    )


def default_blacklist_source(args: argparse.Namespace) -> str:
    return "backtest_approved__" + threshold_tag(args)


def is_symbol_research_mode(args: argparse.Namespace) -> bool:
    return bool(str(getattr(args, "symbols", "") or "").strip())


def apply_symbol_research_mode(args: argparse.Namespace) -> argparse.Namespace:
    if not is_symbol_research_mode(args):
        args.backtest_mode = "PORTFOLIO"
        return args

    args.backtest_mode = "SYMBOL_RESEARCH"

    # In symbol research mode we test selected symbols directly.
    # Portfolio admission lists and dynamic blacklist must not affect the result.
    args.ignore_db_blacklist = True
    args.side_aware_whitelist = False
    args.conditional_side_aware_whitelist = False
    args.write_dynamic_blacklist = False
    args.reset_backtest_blacklist = False

    if args.chulan is None:
        args.chulan = 0

    if args.slots is None:
        args.slots = 1

    return args


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

            CASE
                WHEN UPPER(g51.side) = 'LONG' THEN g2.up_reach_high_proba - g2.dn_reach_high_proba
                WHEN UPPER(g51.side) = 'SHORT' THEN g2.dn_reach_high_proba - g2.up_reach_high_proba
                ELSE NULL
            END AS gate2_side_margin,

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
        "gate2_side_margin",
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

    df = apply_symbol_side_filters(df, args)

    return df.reset_index(drop=True)


def normalize_side_for_whitelist(side: str) -> str:
    side_u = str(side or "").strip().upper()

    if side_u == "BUY":
        return "LONG"

    if side_u == "SELL":
        return "SHORT"

    return side_u


def get_conditional_side_rule(symbol: str, side: str) -> Optional[Dict[str, float]]:
    rules = CONDITIONAL_SIDE_AWARE_WHITELIST or {}
    symbol_u = str(symbol or "").strip().upper()
    side_u = normalize_side_for_whitelist(side)

    symbol_rules = rules.get(symbol_u)

    if not isinstance(symbol_rules, dict):
        return None

    side_rule = symbol_rules.get(side_u)

    if not isinstance(side_rule, dict):
        return None

    return side_rule


def get_side_admission(
    row: pd.Series,
    use_regular_whitelist: bool,
    use_conditional_whitelist: bool,
) -> Tuple[bool, str, str]:
    symbol = str(row.get("symbol") or "").strip().upper()
    side = normalize_side_for_whitelist(str(row.get("side") or ""))

    if use_regular_whitelist:
        allowed_sides = [
            normalize_side_for_whitelist(x)
            for x in SIDE_AWARE_WHITELIST.get(symbol, [])
        ]

        if side in allowed_sides:
            return True, "CURRENT_WHITELIST", ""

    if use_conditional_whitelist:
        rule = get_conditional_side_rule(symbol=symbol, side=side)

        if rule is not None:
            margin = row.get("gate2_side_margin")
            min_margin = float(rule.get("min_gate2_side_margin", 0.0) or 0.0)

            if pd.isna(margin):
                return False, "", "MISSING_GATE2_SIDE_MARGIN"

            if float(margin) < min_margin:
                return False, "", "BELOW_CONDITIONAL_GATE2_MARGIN"

            return True, "CONDITIONAL_WHITELIST", ""

    return False, "", "NO_WHITELIST"


def apply_thresholds(df: pd.DataFrame, args: argparse.Namespace, db_blacklist: set) -> pd.DataFrame:
    if df.empty:
        return df

    out = df[
        (df["gate2_for_side_proba"] >= float(args.gate2))
        & (df["gate4_confidence"] >= float(args.gate4))
        & (df["gate5_1_proba"] >= float(args.gate5_1))
        & (df["gate5_3_proba"] >= float(args.gate5_3))
    ].copy()

    gate2_side_margin_min = float(getattr(args, "gate2_side_margin_min", 0.0) or 0.0)

    if gate2_side_margin_min > 0.0:
        columns = set(out.columns)

        if "gate2_margin_abs" in columns:
            gate2_margin_series = pd.to_numeric(
                out["gate2_margin_abs"],
                errors="coerce",
            ).abs()

        elif "gate2_side_margin" in columns:
            gate2_margin_series = pd.to_numeric(
                out["gate2_side_margin"],
                errors="coerce",
            ).abs()

        elif {"gate2_up", "gate2_dn"}.issubset(columns):
            gate2_margin_series = (
                pd.to_numeric(out["gate2_up"], errors="coerce")
                - pd.to_numeric(out["gate2_dn"], errors="coerce")
            ).abs()

        elif {"gate2_up_proba", "gate2_dn_proba"}.issubset(columns):
            gate2_margin_series = (
                pd.to_numeric(out["gate2_up_proba"], errors="coerce")
                - pd.to_numeric(out["gate2_dn_proba"], errors="coerce")
            ).abs()

        elif {"up_reach_high_proba", "dn_reach_high_proba"}.issubset(columns):
            gate2_margin_series = (
                pd.to_numeric(out["up_reach_high_proba"], errors="coerce")
                - pd.to_numeric(out["dn_reach_high_proba"], errors="coerce")
            ).abs()

        else:
            raise RuntimeError(
                "--gate2-side-margin-min requires one of columns: "
                "gate2_margin_abs, gate2_side_margin, gate2_up/gate2_dn, "
                "gate2_up_proba/gate2_dn_proba, "
                "up_reach_high_proba/dn_reach_high_proba. "
                "available_columns={}".format(sorted(list(out.columns)))
            )

        out = out[
            gate2_margin_series.fillna(0.0) >= gate2_side_margin_min
        ].copy()

    excluded = set(parse_symbol_list(args.exclude_symbols))

    if not bool(args.ignore_db_blacklist):
        excluded |= set(str(x).upper() for x in db_blacklist)

    if excluded:
        out = out[~out["symbol"].astype(str).str.upper().isin(excluded)].copy()

    use_regular_whitelist = bool(args.side_aware_whitelist)
    use_conditional_whitelist = bool(getattr(args, "conditional_side_aware_whitelist", True))

    out["side_admission_source"] = ""
    out["side_admission_reject_reason"] = ""

    if use_regular_whitelist or use_conditional_whitelist:
        keep_mask = []

        for idx, row in out.iterrows():
            allowed, source, reject_reason = get_side_admission(
                row=row,
                use_regular_whitelist=use_regular_whitelist,
                use_conditional_whitelist=use_conditional_whitelist,
            )

            keep_mask.append(bool(allowed))

            if allowed:
                out.at[idx, "side_admission_source"] = source
            else:
                out.at[idx, "side_admission_reject_reason"] = reject_reason

        out = out[pd.Series(keep_mask, index=out.index)].copy()

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


def directional_return(
    side: str,
    entry_px: float,
    exit_px: float,
) -> float:
    side_u = str(side).upper()
    entry = float(entry_px)
    exit_value = float(exit_px)

    if entry <= 0 or exit_value <= 0:
        raise RuntimeError(
            "bad directional return prices: entry={} exit={}".format(
                entry_px,
                exit_px,
            )
        )

    if side_u == "LONG":
        return float((exit_value / entry) - 1.0)

    if side_u == "SHORT":
        return float((entry / exit_value) - 1.0)

    raise RuntimeError("bad side: {}".format(side))


def calc_position_sizing_from_full_sl(
    side: str,
    entry_px: float,
    main_sl_px: float,
    round_trip_cost: float,
    max_full_sl_capital_risk: float,
) -> Dict[str, float]:
    max_risk = float(max_full_sl_capital_risk or 0.0)

    full_main_sl_gross_ret = directional_return(
        side=side,
        entry_px=entry_px,
        exit_px=main_sl_px,
    )

    full_main_sl_net_ret = float(full_main_sl_gross_ret - float(round_trip_cost))
    full_main_sl_capital_risk_abs = float(abs(min(full_main_sl_net_ret, 0.0)))

    if max_risk <= 0 or full_main_sl_capital_risk_abs <= 0:
        position_fraction = 1.0
    elif full_main_sl_capital_risk_abs <= max_risk:
        position_fraction = 1.0
    else:
        position_fraction = float(max_risk / full_main_sl_capital_risk_abs)

    position_fraction = min(1.0, max(0.0, float(position_fraction)))

    return {
        "position_fraction": float(position_fraction),
        "max_full_sl_capital_risk": float(max_risk),
        "full_main_sl_gross_ret": float(full_main_sl_gross_ret),
        "full_main_sl_net_ret": float(full_main_sl_net_ret),
        "full_main_sl_capital_risk_abs": float(full_main_sl_capital_risk_abs),
        "position_fraction_cap_applied": float(position_fraction < 0.999999),
    }


def price_level_hit(
    side: str,
    level_kind: str,
    level_px: float,
    high: float,
    low: float,
) -> bool:
    side_u = str(side).upper()
    kind_u = str(level_kind).upper()
    level = float(level_px)

    if side_u == "LONG" and kind_u == "PROFIT":
        return float(high) >= level

    if side_u == "LONG" and kind_u == "LOSS":
        return float(low) <= level

    if side_u == "SHORT" and kind_u == "PROFIT":
        return float(low) <= level

    if side_u == "SHORT" and kind_u == "LOSS":
        return float(high) >= level

    raise RuntimeError(
        "bad side/level_kind: {} {}".format(
            side,
            level_kind,
        )
    )


def simulate_one(
    row: pd.Series,
    args: argparse.Namespace,
    m1_cache: Dict[str, Optional[pd.DataFrame]],
    m1_load_start_ts: pd.Timestamp,
    m1_load_end_ts: pd.Timestamp,
) -> Optional[Dict[str, object]]:
    symbol = str(row["symbol"]).upper()
    side = str(row["side"]).upper()

    signal_ts = pd.to_datetime(
        row["signal_ts"],
        utc=True,
    )

    entry_ts = (
        signal_ts
        + pd.Timedelta(seconds=H4_SECONDS)
        + pd.Timedelta(
            seconds=int(args.entry_delay_seconds)
        )
    )

    ttl_end_ts = entry_ts + pd.Timedelta(
        hours=float(args.ttl_hours)
    )

    m1 = read_m1_from_db(
        symbol=symbol,
        start_ts=m1_load_start_ts,
        end_ts=m1_load_end_ts,
        cache=m1_cache,
    )

    if m1 is None or m1.empty:
        return None

    window = m1[
        (m1["ts"] >= entry_ts)
        & (m1["ts"] <= ttl_end_ts)
    ].copy()

    if window.empty:
        return None

    entry_px = float(window.iloc[0]["open"])
    atr14 = float(row["atr14"])

    if not np.isfinite(entry_px) or entry_px <= 0:
        return None

    if not np.isfinite(atr14) or atr14 <= 0:
        return None

    use_partial = bool(
        getattr(config, "PARTIAL_TP_ENABLED", False)
    )

    use_early_stop = bool(
        getattr(config, "EARLY_STOP_ENABLED", False)
    )

    use_main_stop = bool(
        getattr(
            config,
            "MAIN_STOP_AFTER_EARLY_WINDOW_ENABLED",
            True,
        )
    )

    use_rest_stop = bool(
        getattr(
            config,
            "REST_STOP_AFTER_PARTIAL_ENABLED",
            True,
        )
    )

    partial_level_fraction = float(
        getattr(
            config,
            "PARTIAL_TP_LEVEL_FRACTION",
            0.75,
        )
    )

    partial_qty_fraction = float(
        getattr(
            config,
            "PARTIAL_TP_QTY_FRACTION",
            0.5,
        )
    )

    early_stop_sl_fraction = float(
        getattr(
            config,
            "EARLY_STOP_SL_FRACTION",
            0.5,
        )
    )

    early_stop_window_minutes = int(
        getattr(
            config,
            "EARLY_STOP_WINDOW_MINUTES",
            60,
        )
    )

    rest_stop_atr_mult = float(
        getattr(
            config,
            "REST_STOP_AFTER_PARTIAL_ATR_MULT",
            float(args.tp_atr) * 0.125,
        )
    )

    partial_qty_fraction = min(
        1.0,
        max(0.0, partial_qty_fraction),
    )

    final_qty_fraction = (
        1.0 - partial_qty_fraction
        if use_partial
        else 1.0
    )

    if (
        not use_partial
        or partial_qty_fraction <= 0
        or final_qty_fraction <= 0
    ):
        use_partial = False
        partial_qty_fraction = 0.0
        final_qty_fraction = 1.0

    partial_tp_atr = (
        float(args.tp_atr)
        * partial_level_fraction
    )

    final_tp_atr = float(args.tp_atr)

    early_stop_atr = (
        float(args.sl_atr)
        * early_stop_sl_fraction
    )

    main_sl_atr = float(args.sl_atr)

    partial_tp_px, _ = calc_tp_sl(
        side=side,
        entry_px=entry_px,
        atr14=atr14,
        tp_atr=partial_tp_atr,
        sl_atr=main_sl_atr,
    )

    final_tp_px, main_sl_px = calc_tp_sl(
        side=side,
        entry_px=entry_px,
        atr14=atr14,
        tp_atr=final_tp_atr,
        sl_atr=main_sl_atr,
    )

    _, early_stop_px = calc_tp_sl(
        side=side,
        entry_px=entry_px,
        atr14=atr14,
        tp_atr=final_tp_atr,
        sl_atr=early_stop_atr,
    )

    round_trip_cost_full_position = (
        2.0 * float(args.fee_side)
        + 2.0 * float(args.slippage_side)
    )

    position_sizing = calc_position_sizing_from_full_sl(
        side=side,
        entry_px=entry_px,
        main_sl_px=main_sl_px,
        round_trip_cost=round_trip_cost_full_position,
        max_full_sl_capital_risk=float(
            getattr(args, "max_full_sl_capital_risk", 0.0) or 0.0
        ),
    )

    position_fraction = float(position_sizing["position_fraction"])

    if side == "LONG":
        rest_stop_px = (
            entry_px
            + atr14 * rest_stop_atr_mult
        )
    else:
        rest_stop_px = (
            entry_px
            - atr14 * rest_stop_atr_mult
        )

    early_stop_expires_at = entry_ts + pd.Timedelta(
        minutes=early_stop_window_minutes
    )

    partial_tp_hit = False
    partial_tp_ts = pd.NaT
    partial_tp_exit_px = np.nan

    final_tp_hit = False
    final_tp_ts = pd.NaT
    final_tp_exit_px = np.nan

    early_stop_hit = False
    early_stop_ts = pd.NaT

    main_sl_hit = False
    main_sl_ts = pd.NaT

    rest_stop_hit = False
    rest_stop_ts = pd.NaT

    rest_stop_active_from = pd.NaT

    exit_reason = "TTL"
    exit_ts = pd.to_datetime(
        window.iloc[-1]["ts"],
        utc=True,
    )

    exit_legs: List[Dict[str, object]] = []

    remaining_fraction = 1.0
    terminal = False

    for _, bar in window.iterrows():
        ts = pd.to_datetime(
            bar["ts"],
            utc=True,
        )

        high = float(bar["high"])
        low = float(bar["low"])

        if not partial_tp_hit:
            if (
                use_early_stop
                and ts < early_stop_expires_at
            ):
                active_stop_px = early_stop_px
                active_stop_reason = "EARLY_STOP"
            elif use_main_stop or not use_early_stop:
                active_stop_px = main_sl_px
                active_stop_reason = "MAIN_SL"
            else:
                active_stop_px = None
                active_stop_reason = None

            stop_hit = False

            if active_stop_px is not None:
                stop_hit = price_level_hit(
                    side=side,
                    level_kind="LOSS",
                    level_px=float(active_stop_px),
                    high=high,
                    low=low,
                )

            partial_hit_now = (
                use_partial
                and price_level_hit(
                    side=side,
                    level_kind="PROFIT",
                    level_px=partial_tp_px,
                    high=high,
                    low=low,
                )
            )

            final_hit_now = price_level_hit(
                side=side,
                level_kind="PROFIT",
                level_px=final_tp_px,
                high=high,
                low=low,
            )

            if stop_hit:
                exit_legs.append(
                    {
                        "role": active_stop_reason,
                        "fraction": float(remaining_fraction),
                        "px": float(active_stop_px),
                        "ts": ts,
                    }
                )

                exit_reason = str(active_stop_reason)
                exit_ts = ts

                if active_stop_reason == "EARLY_STOP":
                    early_stop_hit = True
                    early_stop_ts = ts
                else:
                    main_sl_hit = True
                    main_sl_ts = ts

                remaining_fraction = 0.0
                terminal = True
                break

            if final_hit_now:
                if use_partial:
                    exit_legs.append(
                        {
                            "role": "PARTIAL_TP",
                            "fraction": float(
                                partial_qty_fraction
                            ),
                            "px": float(partial_tp_px),
                            "ts": ts,
                        }
                    )

                    exit_legs.append(
                        {
                            "role": "FINAL_TP",
                            "fraction": float(
                                final_qty_fraction
                            ),
                            "px": float(final_tp_px),
                            "ts": ts,
                        }
                    )

                    partial_tp_hit = True
                    partial_tp_ts = ts
                    partial_tp_exit_px = float(
                        partial_tp_px
                    )

                    final_tp_hit = True
                    final_tp_ts = ts
                    final_tp_exit_px = float(
                        final_tp_px
                    )

                    exit_reason = (
                        "PARTIAL_TP_THEN_FINAL_TP"
                    )
                else:
                    exit_legs.append(
                        {
                            "role": "FINAL_TP",
                            "fraction": 1.0,
                            "px": float(final_tp_px),
                            "ts": ts,
                        }
                    )

                    final_tp_hit = True
                    final_tp_ts = ts
                    final_tp_exit_px = float(
                        final_tp_px
                    )

                    exit_reason = "FINAL_TP"

                exit_ts = ts
                remaining_fraction = 0.0
                terminal = True
                break

            if partial_hit_now:
                exit_legs.append(
                    {
                        "role": "PARTIAL_TP",
                        "fraction": float(
                            partial_qty_fraction
                        ),
                        "px": float(partial_tp_px),
                        "ts": ts,
                    }
                )

                partial_tp_hit = True
                partial_tp_ts = ts
                partial_tp_exit_px = float(
                    partial_tp_px
                )

                remaining_fraction = float(
                    final_qty_fraction
                )

                if use_rest_stop:
                    rest_stop_active_from = (
                        ts + pd.Timedelta(minutes=1)
                    )

                continue

        else:
            final_hit_now = price_level_hit(
                side=side,
                level_kind="PROFIT",
                level_px=final_tp_px,
                high=high,
                low=low,
            )

            rest_is_active = (
                use_rest_stop
                and pd.notna(rest_stop_active_from)
                and ts >= rest_stop_active_from
            )

            rest_hit_now = (
                rest_is_active
                and price_level_hit(
                    side=side,
                    level_kind="LOSS",
                    level_px=rest_stop_px,
                    high=high,
                    low=low,
                )
            )

            if rest_hit_now:
                exit_legs.append(
                    {
                        "role": (
                            "REST_STOP_AFTER_PARTIAL"
                        ),
                        "fraction": float(
                            remaining_fraction
                        ),
                        "px": float(rest_stop_px),
                        "ts": ts,
                    }
                )

                rest_stop_hit = True
                rest_stop_ts = ts
                exit_reason = (
                    "PARTIAL_TP_THEN_REST_STOP"
                )
                exit_ts = ts
                remaining_fraction = 0.0
                terminal = True
                break

            if final_hit_now:
                exit_legs.append(
                    {
                        "role": "FINAL_TP",
                        "fraction": float(
                            remaining_fraction
                        ),
                        "px": float(final_tp_px),
                        "ts": ts,
                    }
                )

                final_tp_hit = True
                final_tp_ts = ts
                final_tp_exit_px = float(
                    final_tp_px
                )

                exit_reason = (
                    "PARTIAL_TP_THEN_FINAL_TP"
                )
                exit_ts = ts
                remaining_fraction = 0.0
                terminal = True
                break

    if not terminal and remaining_fraction > 0:
        ttl_exit_px = float(
            window.iloc[-1]["close"]
        )

        ttl_exit_ts = pd.to_datetime(
            window.iloc[-1]["ts"],
            utc=True,
        )

        exit_legs.append(
            {
                "role": "TTL",
                "fraction": float(
                    remaining_fraction
                ),
                "px": ttl_exit_px,
                "ts": ttl_exit_ts,
            }
        )

        exit_ts = ttl_exit_ts

        if partial_tp_hit:
            exit_reason = "PARTIAL_TP_THEN_TTL"
        else:
            exit_reason = "TTL"

        remaining_fraction = 0.0

    gross_ret = 0.0
    weighted_exit_px = 0.0
    exit_fraction_sum = 0.0

    for leg in exit_legs:
        fraction = float(leg["fraction"])
        leg_px = float(leg["px"])

        gross_ret += (
            fraction
            * directional_return(
                side=side,
                entry_px=entry_px,
                exit_px=leg_px,
            )
        )

        weighted_exit_px += fraction * leg_px
        exit_fraction_sum += fraction

    if exit_fraction_sum <= 0:
        return None

    exit_px = weighted_exit_px / exit_fraction_sum

    round_trip_cost = float(round_trip_cost_full_position)

    raw_gross_ret = float(gross_ret)
    raw_round_trip_cost = float(round_trip_cost)
    raw_net_ret = float(raw_gross_ret - raw_round_trip_cost)

    gross_ret = float(raw_gross_ret * position_fraction)
    round_trip_cost = float(raw_round_trip_cost * position_fraction)
    net_ret = float(raw_net_ret * position_fraction)

    if exit_reason in {
        "FINAL_TP",
        "PARTIAL_TP_THEN_FINAL_TP",
        "PARTIAL_TP_THEN_REST_STOP",
    }:
        outcome_bucket = "TP"
    elif exit_reason in {
        "EARLY_STOP",
        "MAIN_SL",
    }:
        outcome_bucket = "SL"
    else:
        outcome_bucket = "TTL"

    out = row.to_dict()

    out.update(
        {
            "entry_ts": entry_ts,
            "ttl_end_ts": ttl_end_ts,
            "exit_ts": exit_ts,
            "entry_px": float(entry_px),
            "tp_px": float(final_tp_px),
            "sl_px": float(main_sl_px),
            "partial_tp_px": float(partial_tp_px),
            "final_tp_px": float(final_tp_px),
            "early_stop_px": float(early_stop_px),
            "main_sl_px": float(main_sl_px),
            "rest_stop_after_partial_px": float(
                rest_stop_px
            ),
            "early_stop_expires_at": (
                early_stop_expires_at
            ),
            "partial_tp_qty_fraction": float(
                partial_qty_fraction
            ),
            "final_tp_qty_fraction": float(
                final_qty_fraction
            ),
            "partial_tp_hit": bool(partial_tp_hit),
            "partial_tp_ts": partial_tp_ts,
            "partial_tp_exit_px": (
                partial_tp_exit_px
            ),
            "final_tp_hit": bool(final_tp_hit),
            "final_tp_ts": final_tp_ts,
            "final_tp_exit_px": final_tp_exit_px,
            "early_stop_hit": bool(early_stop_hit),
            "early_stop_ts": early_stop_ts,
            "main_sl_hit": bool(main_sl_hit),
            "main_sl_ts": main_sl_ts,
            "rest_stop_hit": bool(rest_stop_hit),
            "rest_stop_ts": rest_stop_ts,
            "rest_stop_active_from": (
                rest_stop_active_from
            ),
            "exit_px": float(exit_px),
            "exit_reason": str(exit_reason),
            "outcome_bucket": outcome_bucket,
            "exit_legs_json": json.dumps(
                exit_legs,
                ensure_ascii=False,
                default=str,
            ),
            "raw_gross_ret": float(raw_gross_ret),
            "raw_round_trip_cost": float(raw_round_trip_cost),
            "raw_net_ret": float(raw_net_ret),
            "position_fraction": float(position_fraction),
            "max_full_sl_capital_risk": float(position_sizing["max_full_sl_capital_risk"]),
            "full_main_sl_gross_ret": float(position_sizing["full_main_sl_gross_ret"]),
            "full_main_sl_net_ret": float(position_sizing["full_main_sl_net_ret"]),
            "full_main_sl_capital_risk_abs": float(position_sizing["full_main_sl_capital_risk_abs"]),
            "position_fraction_cap_applied": bool(position_sizing["position_fraction_cap_applied"]),
            "gross_ret": float(gross_ret),
            "round_trip_cost": float(
                round_trip_cost
            ),
            "net_ret": float(net_ret),
            "trade_management_mode": (
                "partial75_early_stop"
                if use_partial or use_early_stop
                else "legacy_full_tp_sl"
            ),
            "same_m1_policy": (
                "adverse_stop_first;"
                "rest_stop_active_next_m1"
            ),
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
            "side_aware_whitelist": bool(args.side_aware_whitelist),
            "conditional_side_aware_whitelist": bool(getattr(args, "conditional_side_aware_whitelist", True)),
            "slots": int(args.slots),
            "blacklist_source": str(args.blacklist_source),
            "max_full_sl_capital_risk": float(getattr(args, "max_full_sl_capital_risk", 0.0) or 0.0),
        }

    capital = float(args.capital)
    peak = capital
    max_dd = 0.0
    position_free_ts: Optional[pd.Timestamp] = None
    trades = []
    skipped_overlap = 0
    skipped_dynamic_blacklist = 0

    symbol_research_mode = is_symbol_research_mode(args)
    write_dynamic_blacklist = bool(getattr(args, "write_dynamic_blacklist", False)) and not symbol_research_mode
    blacklist_source = str(getattr(args, "blacklist_source", "backtest_approved"))

    for _, row in sim.sort_values(["entry_ts", "signal_strength"], ascending=[True, False]).iterrows():
        entry_ts = pd.to_datetime(row["entry_ts"], utc=True)
        exit_ts = pd.to_datetime(row["exit_ts"], utc=True)

        if position_free_ts is not None and entry_ts < position_free_ts:
            skipped_overlap += 1
            continue

        if symbol_research_mode:
            allowed = True
            dynamic_reason = "symbol_research_mode_disabled"
            dynamic_stats = {}
        else:
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
            "side_aware_whitelist": bool(args.side_aware_whitelist),
            "conditional_side_aware_whitelist": bool(getattr(args, "conditional_side_aware_whitelist", True)),
            "slots": int(args.slots),
            "max_full_sl_capital_risk": float(getattr(args, "max_full_sl_capital_risk", 0.0) or 0.0),
        }

    wins = trades_df[trades_df["net_ret"] > 0]
    losses = trades_df[trades_df["net_ret"] <= 0]

    gross_profit = float(wins["pnl_usd"].sum()) if len(wins) else 0.0
    gross_loss_abs = float(abs(losses["pnl_usd"].sum())) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else None

    capped_position_trades = int(trades_df["position_fraction_cap_applied"].fillna(False).astype(bool).sum()) if "position_fraction_cap_applied" in trades_df.columns else 0
    min_position_fraction = float(trades_df["position_fraction"].min()) if "position_fraction" in trades_df.columns and len(trades_df) else 1.0
    mean_position_fraction = float(trades_df["position_fraction"].mean()) if "position_fraction" in trades_df.columns and len(trades_df) else 1.0

    summary = {
        "trades_taken": int(len(trades_df)),
        "final_capital": float(capital),
        "total_return_pct": float((capital / float(args.capital)) - 1.0),
        "win_rate": float((trades_df["net_ret"] > 0).mean()),
        "profit_factor": profit_factor,
        "max_drawdown_pct": float(max_dd),
        "mean_net_ret": float(trades_df["net_ret"].mean()),
        "median_net_ret": float(trades_df["net_ret"].median()),
        "max_full_sl_capital_risk": float(getattr(args, "max_full_sl_capital_risk", 0.0) or 0.0),
        "capped_position_trades": int(capped_position_trades),
        "min_position_fraction": float(min_position_fraction),
        "mean_position_fraction": float(mean_position_fraction),
        "tp_count": int((trades_df["outcome_bucket"] == "TP").sum()),
        "sl_count": int((trades_df["outcome_bucket"] == "SL").sum()),
        "ttl_count": int((trades_df["outcome_bucket"] == "TTL").sum()),
        "partial_tp_trades": int(trades_df["partial_tp_hit"].fillna(False).astype(bool).sum()),
        "partial_tp_then_final_tp": int((trades_df["exit_reason"] == "PARTIAL_TP_THEN_FINAL_TP").sum()),
        "partial_tp_then_rest_stop": int((trades_df["exit_reason"] == "PARTIAL_TP_THEN_REST_STOP").sum()),
        "partial_tp_then_ttl": int((trades_df["exit_reason"] == "PARTIAL_TP_THEN_TTL").sum()),
        "early_stop_count": int((trades_df["exit_reason"] == "EARLY_STOP").sum()),
        "main_sl_count": int((trades_df["exit_reason"] == "MAIN_SL").sum()),
        "skipped_overlap": int(skipped_overlap),
        "skipped_dynamic_blacklist": int(skipped_dynamic_blacklist),
        "dynamic_blacklist_written": bool(write_dynamic_blacklist),
        "blacklist_source": blacklist_source,
        "chulan": int(args.chulan),
        "side_aware_whitelist": bool(args.side_aware_whitelist),
        "conditional_side_aware_whitelist": bool(getattr(args, "conditional_side_aware_whitelist", True)),
        "slots": int(args.slots),
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

    try:
        xlsx_path = export_backtest_xlsx(
            raw=raw,
            passed=passed,
            selected=selected,
            sim=sim,
            trades=trades,
            summary=summary,
            args=args,
        )

        print("WROTE_XLSX:", xlsx_path)

    except Exception as exc:
        print("BACKTEST_XLSX_EXPORT_ERROR:", repr(exc))

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
    print("backtest_mode:", str(getattr(args, "backtest_mode", "PORTFOLIO")))
    print("period:", args.start, "->", args.end)
    print("symbols_filter:", format_symbol_side_filters(str(getattr(args, "symbols", "") or "")))
    print("pair_model_name:", args.pair_model_name)
    print("grid_name:", args.grid_name)
    print("gate2:", args.gate2)
    print("gate2_side_margin_min:", float(getattr(args, "gate2_side_margin_min", 0.0) or 0.0))
    print("gate4:", args.gate4)
    print("gate5_1:", args.gate5_1)
    print("gate5_3:", args.gate5_3)
    print("chulan:", int(args.chulan))
    print("side_aware_whitelist:", bool(args.side_aware_whitelist))
    print("side_aware_whitelist_symbols:", len(SIDE_AWARE_WHITELIST))
    print("conditional_side_aware_whitelist:", bool(getattr(args, "conditional_side_aware_whitelist", True)))
    print("conditional_side_aware_whitelist_symbols:", len(CONDITIONAL_SIDE_AWARE_WHITELIST))
    print("max_full_sl_capital_risk:", float(getattr(args, "max_full_sl_capital_risk", 0.0) or 0.0))
    print("slots:", int(args.slots))
    print("blacklist_source:", str(args.blacklist_source))
    print("tp_atr:", args.tp_atr)
    print("sl_atr:", args.sl_atr)
    print("ttl_hours:", args.ttl_hours)
    print("entry_delay_seconds:", args.entry_delay_seconds)
    print("entry_rule:", "H4 close + 60s => second M1 open")
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
        "max_full_sl_capital_risk",
        "capped_position_trades",
        "min_position_fraction",
        "mean_position_fraction",
        "tp_count",
        "sl_count",
        "ttl_count",
        "partial_tp_trades",
        "partial_tp_then_final_tp",
        "partial_tp_then_rest_stop",
        "partial_tp_then_ttl",
        "early_stop_count",
        "main_sl_count",
        "skipped_overlap",
        "skipped_dynamic_blacklist",
        "dynamic_blacklist_written",
        "blacklist_source",
        "chulan",
        "side_aware_whitelist",
        "slots",
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
        "partial_tp_px",
        "final_tp_px",
        "early_stop_px",
        "main_sl_px",
        "rest_stop_after_partial_px",
        "exit_px",
        "exit_reason",
        "outcome_bucket",
        "partial_tp_hit",
        "final_tp_hit",
        "early_stop_hit",
        "main_sl_hit",
        "rest_stop_hit",
        "raw_net_ret",
        "position_fraction",
        "full_main_sl_capital_risk_abs",
        "gross_ret",
        "round_trip_cost",
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
    print("BACKTEST_MODE:", str(getattr(args, "backtest_mode", "PORTFOLIO")))
    print("DB_ONLY_MODE:", True)
    print("M1_DB_TABLE:", M1_DB_TABLE)
    print("M1_AUTO_SYNC:", json.dumps(m1_sync_report, ensure_ascii=False, default=str))
    print("LEGACY_DB_BLACKLIST_SOURCE:", db_blacklist_source)
    print("LEGACY_DB_BLACKLIST_SYMBOLS_COUNT:", len(db_blacklist))
    print("LEGACY_DB_BLACKLIST_SYMBOLS:", ",".join(sorted(db_blacklist)) if db_blacklist else "")
    print("DYNAMIC_BLACKLIST_SOURCE:", str(args.blacklist_source))
    print("SIDE_AWARE_WHITELIST_ENABLED:", bool(args.side_aware_whitelist))
    print("CONDITIONAL_SIDE_AWARE_WHITELIST_ENABLED:", bool(getattr(args, "conditional_side_aware_whitelist", True)))
    print(
        "SIDE_AWARE_WHITELIST:",
        json.dumps(SIDE_AWARE_WHITELIST, ensure_ascii=False, sort_keys=True),
    )
    print(
        "CONDITIONAL_SIDE_AWARE_WHITELIST:",
        json.dumps(CONDITIONAL_SIDE_AWARE_WHITELIST, ensure_ascii=False, sort_keys=True),
    )
    print("MAX_FULL_SL_CAPITAL_RISK:", float(getattr(args, "max_full_sl_capital_risk", 0.0) or 0.0))
    print("SLOTS:", int(args.slots))
    print("OUT_DIR:", args.out_dir)

if __name__ == "__main__":
    main()
