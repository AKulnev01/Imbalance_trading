from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_DSN_ENV = "IMB_DB_DSN"

DEFAULT_M1_DIR = "data/m1_4"
DEFAULT_H4_DIR = "data/h4_3"

DEFAULT_SCHEMA = "public"
DEFAULT_M1_TABLE = "candles_m1"
DEFAULT_H4_TABLE = "candles_h4"
DEFAULT_MARKET_CATEGORY = "linear"

DEFAULT_PAIR_MODEL_NAME = "tp225_sl075__vs__tp100_sl075"
DEFAULT_GRID_NAME = "tp100_sl075"

DEFAULT_GATE2_THR = 0.63
DEFAULT_GATE4_THR = 0.58
DEFAULT_GATE5_1_THR = 0.10
DEFAULT_GATE5_3_THR = 0.55

DEFAULT_FEE_SIDE_BPS = 10.0
DEFAULT_SLIPPAGE_SIDE_BPS = 20.0
DEFAULT_ENTRY_DELAY_SECONDS = 90

DEFAULT_WEEKEND_BAN = True
DEFAULT_WEEKEND_BAN_START = "FRI 16:00 UTC"
DEFAULT_WEEKEND_BAN_END = "MON 04:00 UTC"

DEFAULT_BACKTEST_SCRIPT = "online/trading/backtest_m1_thresholds.py"


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    text = text.replace("/", "")
    text = text.replace("-", "")
    text = text.replace("_", "")

    if not text:
        raise ValueError("empty symbol")

    return text


def normalize_symbols(raw_symbols: str) -> List[str]:
    text = str(raw_symbols or "").strip()
    if not text:
        raise ValueError("symbols is empty")

    parts = text.replace(";", ",").replace(" ", ",").split(",")
    out: List[str] = []

    for part in parts:
        item = part.strip()
        if not item:
            continue

        symbol = normalize_symbol(item)
        if symbol not in out:
            out.append(symbol)

    if not out:
        raise ValueError("symbols is empty after normalization")

    return out


def parse_ts(value: str, field_name: str) -> pd.Timestamp:
    ts = pd.to_datetime(str(value), utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError("invalid {}: {}".format(field_name, value))
    return pd.Timestamp(ts)


def validate_threshold(name: str, value: float) -> Optional[str]:
    try:
        numeric = float(value)
    except Exception:
        return "{} is not numeric: {}".format(name, value)

    if numeric < 0.0 or numeric > 1.0:
        return "{} must be in [0, 1], got {}".format(name, value)

    return None


def validate_non_negative(name: str, value: float) -> Optional[str]:
    try:
        numeric = float(value)
    except Exception:
        return "{} is not numeric: {}".format(name, value)

    if numeric < 0.0:
        return "{} must be >= 0, got {}".format(name, value)

    return None


def qident(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("empty SQL identifier")

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if any(ch not in allowed for ch in text):
        raise ValueError("unsafe SQL identifier: {}".format(name))

    return '"' + text.replace('"', '""') + '"'


def fq_table(schema_name: str, table_name: str) -> str:
    return "{}.{}".format(qident(schema_name), qident(table_name))


def get_psycopg2():
    try:
        import psycopg2
        import psycopg2.extras
        return psycopg2
    except Exception as exc:
        raise RuntimeError("psycopg2 is required for DB checks. Error: {}".format(repr(exc)))


def parquet_exists_report(symbol: str, m1_dir: Path, h4_dir: Path) -> Dict[str, Any]:
    m1_path = m1_dir / "{}.parquet".format(symbol)
    h4_path = h4_dir / "{}.parquet".format(symbol)

    return {
        "symbol": symbol,
        "m1_path": str(m1_path).replace("\\", "/"),
        "h4_path": str(h4_path).replace("\\", "/"),
        "m1_exists": bool(m1_path.exists()),
        "h4_exists": bool(h4_path.exists()),
    }


def db_candle_stats_for_symbol(
    conn: Any,
    schema_name: str,
    table_name: str,
    symbol: str,
    market_category: str,
    from_ts: pd.Timestamp,
    to_ts: pd.Timestamp,
) -> Dict[str, Any]:
    table = fq_table(schema_name, table_name)

    sql = """
        SELECT
            COUNT(*) AS rows,
            MIN(entry_ts) AS min_entry_ts,
            MAX(entry_ts) AS max_entry_ts
        FROM {table}
        WHERE symbol = %s
          AND market_category = %s
          AND entry_ts >= %s
          AND entry_ts < %s
    """.format(table=table)

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                symbol,
                market_category,
                from_ts.to_pydatetime(),
                to_ts.to_pydatetime(),
            ),
        )
        row = cur.fetchone()

    rows = int(row[0]) if row and row[0] is not None else 0
    min_ts = row[1] if row and row[1] is not None else None
    max_ts = row[2] if row and row[2] is not None else None

    return {
        "rows": rows,
        "min_entry_ts": str(min_ts) if min_ts is not None else None,
        "max_entry_ts": str(max_ts) if max_ts is not None else None,
    }


def expected_min_rows(
    timeframe: str,
    from_ts: pd.Timestamp,
    to_ts: pd.Timestamp,
) -> int:
    seconds = float((to_ts - from_ts).total_seconds())

    if seconds <= 0:
        return 0

    if timeframe == "m1":
        return int(seconds // 60)

    if timeframe == "h4":
        return int(seconds // (4 * 60 * 60))

    return 0


def build_backtest_step(
    backtest_script: str,
    symbols: List[str],
    from_ts: pd.Timestamp,
    to_ts: pd.Timestamp,
    gate2_thr: float,
    gate4_thr: float,
    gate5_1_thr: float,
    gate5_3_thr: float,
    fee_side_bps: float,
    slippage_side_bps: float,
    entry_delay_seconds: int,
    pair_model_name: str,
    grid_name: str,
    market_category: str,
    run_tag: str,
) -> Dict[str, Any]:
    fee_side = float(fee_side_bps) / 10000.0
    slippage_side = float(slippage_side_bps) / 10000.0

    args = [
        "--start", from_ts.isoformat(),
        "--end", to_ts.isoformat(),
        "--symbols", ",".join(symbols),
        "--gate2", str(float(gate2_thr)),
        "--gate4", str(float(gate4_thr)),
        "--gate5-1", str(float(gate5_1_thr)),
        "--gate5-3", str(float(gate5_3_thr)),
        "--pair-model-name", str(pair_model_name),
        "--grid-name", str(grid_name),
        "--entry-delay-seconds", str(int(entry_delay_seconds)),
        "--fee-side", str(fee_side),
        "--slippage-side", str(slippage_side),
        "--skip-m1-sync",
        "--out-dir", "reports/backtests/{}".format(str(run_tag)),
    ]

    script = str(backtest_script or "").strip()

    return {
        "id": "db_backtest",
        "title": "Run DB backtest for requested symbols and thresholds",
        "enabled": bool(script),
        "script": script,
        "args": args,
        "command": "python {} {}".format(script, " ".join(args)) if script else "",
        "reads": [
            "public.candles_m1",
            "public.candles_h4",
            "online_gate1_features/predictions",
            "online_gate2_features/predictions",
            "online_gate3_features/predictions",
            "online_gate4_features/predictions",
            "online_gate5_1_scores",
            "online_gate5_2_ranker",
            "online_gate5_3_decisions",
        ],
        "writes": [
            "backtest report artifact",
        ],
        "note": (
            "Backtest script is intentionally configurable. "
            "Current plan validates request and DB candle coverage; actual runner will be attached after the working DB backtester is provided."
        ),
    }


def decide_status(errors: List[str], warnings: List[str], db_check_enabled: bool) -> str:
    if errors:
        if any(x.startswith("MISSING_PARQUET") for x in errors):
            return "REJECT_MISSING_SYMBOL_DATA"

        if any(x.startswith("BAD_DATE_RANGE") for x in errors):
            return "REJECT_BAD_DATE_RANGE"

        if any(x.startswith("BAD_THRESHOLDS") for x in errors):
            return "REJECT_BAD_THRESHOLDS"

        if any(x.startswith("BAD_COSTS") for x in errors):
            return "REJECT_BAD_COSTS"

        if any(x.startswith("NO_DB_CANDLES") for x in errors):
            return "REJECT_NO_DB_CANDLES"

        if any(x.startswith("DB_CHECK_FAILED") for x in errors):
            return "REJECT_DB_CHECK_FAILED"

        return "REJECT_INVALID_REQUEST"

    if not db_check_enabled:
        return "READY_FOR_BACKTEST_WITHOUT_DB_CHECK"

    return "READY_FOR_BACKTEST"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate user backtest request and build a DB-backtest plan."
    )

    parser.add_argument("--symbols", required=True)
    parser.add_argument("--from-ts", default="")
    parser.add_argument("--to-ts", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")

    parser.add_argument("--m1-dir", default=DEFAULT_M1_DIR)
    parser.add_argument("--h4-dir", default=DEFAULT_H4_DIR)

    parser.add_argument("--gate2-thr", type=float, default=DEFAULT_GATE2_THR)
    parser.add_argument("--gate4-thr", type=float, default=DEFAULT_GATE4_THR)
    parser.add_argument("--gate5-1-thr", type=float, default=DEFAULT_GATE5_1_THR)
    parser.add_argument("--gate5-3-thr", type=float, default=DEFAULT_GATE5_3_THR)

    parser.add_argument("--fee-side-bps", type=float, default=DEFAULT_FEE_SIDE_BPS)
    parser.add_argument("--slippage-side-bps", type=float, default=DEFAULT_SLIPPAGE_SIDE_BPS)
    parser.add_argument("--entry-delay-seconds", type=int, default=DEFAULT_ENTRY_DELAY_SECONDS)

    parser.add_argument("--pair-model-name", default=DEFAULT_PAIR_MODEL_NAME)
    parser.add_argument("--grid-name", default=DEFAULT_GRID_NAME)

    parser.add_argument("--market-category", default=DEFAULT_MARKET_CATEGORY)

    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--m1-table", default=DEFAULT_M1_TABLE)
    parser.add_argument("--h4-table", default=DEFAULT_H4_TABLE)

    parser.add_argument("--dsn", default="")
    parser.add_argument("--dsn-env", default=DEFAULT_DSN_ENV)
    parser.add_argument("--skip-db-check", action="store_true")

    parser.add_argument("--backtest-script", default=DEFAULT_BACKTEST_SCRIPT)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--json-out", default="")

    args = parser.parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    symbols = normalize_symbols(str(args.symbols))

    from_raw = str(args.from_ts or "").strip() or str(args.start or "").strip()
    to_raw = str(args.to_ts or "").strip() or str(args.end or "").strip()

    if not from_raw:
        errors.append("MISSING_DATE: pass --from-ts or --start")
        from_raw = "1970-01-01 00:00:00"

    if not to_raw:
        errors.append("MISSING_DATE: pass --to-ts or --end")
        to_raw = "1970-01-01 00:00:00"

    from_ts = parse_ts(from_raw, "from-ts/start")
    to_ts = parse_ts(to_raw, "to-ts/end")

    run_tag = str(args.run_tag).strip()
    if not run_tag:
        run_tag = "backtest_{}".format(utc_now_tag())

    if from_ts >= to_ts:
        errors.append("BAD_DATE_RANGE: from-ts must be earlier than to-ts")

    threshold_checks = [
        ("gate2_thr", args.gate2_thr),
        ("gate4_thr", args.gate4_thr),
        ("gate5_1_thr", args.gate5_1_thr),
        ("gate5_3_thr", args.gate5_3_thr),
    ]

    for name, value in threshold_checks:
        err = validate_threshold(name, float(value))
        if err:
            errors.append("BAD_THRESHOLDS: {}".format(err))

    cost_checks = [
        ("fee_side_bps", args.fee_side_bps),
        ("slippage_side_bps", args.slippage_side_bps),
        ("entry_delay_seconds", args.entry_delay_seconds),
    ]

    for name, value in cost_checks:
        err = validate_non_negative(name, float(value))
        if err:
            errors.append("BAD_COSTS: {}".format(err))

    m1_dir = Path(str(args.m1_dir))
    h4_dir = Path(str(args.h4_dir))

    parquet_reports: List[Dict[str, Any]] = []
    for symbol in symbols:
        report = parquet_exists_report(symbol=symbol, m1_dir=m1_dir, h4_dir=h4_dir)
        parquet_reports.append(report)

        if not report["m1_exists"]:
            errors.append("MISSING_PARQUET_M1: {}".format(report["m1_path"]))

        if not report["h4_exists"]:
            errors.append("MISSING_PARQUET_H4: {}".format(report["h4_path"]))

    db_check_enabled = not bool(args.skip_db_check)
    dsn = str(args.dsn).strip() or str(os.environ.get(str(args.dsn_env), "")).strip()

    db_reports: Dict[str, Any] = {
        "enabled": bool(db_check_enabled),
        "dsn_available": bool(dsn),
        "schema": str(args.schema),
        "m1_table": str(args.m1_table),
        "h4_table": str(args.h4_table),
        "market_category": str(args.market_category),
        "symbols": {},
    }

    if db_check_enabled:
        if not dsn:
            warnings.append(
                "DB_CHECK_SKIPPED: DSN is empty. Pass --dsn, set {}, or use --skip-db-check intentionally.".format(
                    args.dsn_env
                )
            )
            db_check_enabled = False
            db_reports["enabled"] = False
        elif from_ts < to_ts:
            try:
                psycopg2 = get_psycopg2()
                with psycopg2.connect(dsn) as conn:
                    for symbol in symbols:
                        m1_stats = db_candle_stats_for_symbol(
                            conn=conn,
                            schema_name=str(args.schema),
                            table_name=str(args.m1_table),
                            symbol=symbol,
                            market_category=str(args.market_category),
                            from_ts=from_ts,
                            to_ts=to_ts,
                        )
                        h4_stats = db_candle_stats_for_symbol(
                            conn=conn,
                            schema_name=str(args.schema),
                            table_name=str(args.h4_table),
                            symbol=symbol,
                            market_category=str(args.market_category),
                            from_ts=from_ts,
                            to_ts=to_ts,
                        )

                        exp_m1 = expected_min_rows("m1", from_ts, to_ts)
                        exp_h4 = expected_min_rows("h4", from_ts, to_ts)

                        symbol_db_report = {
                            "m1": m1_stats,
                            "h4": h4_stats,
                            "expected_m1_rows_if_full": exp_m1,
                            "expected_h4_rows_if_full": exp_h4,
                        }

                        db_reports["symbols"][symbol] = symbol_db_report

                        if int(m1_stats["rows"]) <= 0:
                            errors.append(
                                "NO_DB_CANDLES_M1: {} has no rows in {}.{} for requested period".format(
                                    symbol,
                                    args.schema,
                                    args.m1_table,
                                )
                            )

                        if int(h4_stats["rows"]) <= 0:
                            errors.append(
                                "NO_DB_CANDLES_H4: {} has no rows in {}.{} for requested period".format(
                                    symbol,
                                    args.schema,
                                    args.h4_table,
                                )
                            )

                        if exp_m1 > 0 and int(m1_stats["rows"]) < exp_m1:
                            warnings.append(
                                "PARTIAL_DB_M1_COVERAGE: {} rows={} expected_if_full={}".format(
                                    symbol,
                                    int(m1_stats["rows"]),
                                    exp_m1,
                                )
                            )

                        if exp_h4 > 0 and int(h4_stats["rows"]) < exp_h4:
                            warnings.append(
                                "PARTIAL_DB_H4_COVERAGE: {} rows={} expected_if_full={}".format(
                                    symbol,
                                    int(h4_stats["rows"]),
                                    exp_h4,
                                )
                            )

            except Exception as exc:
                errors.append("DB_CHECK_FAILED: {}".format(repr(exc)))

    status = decide_status(
        errors=errors,
        warnings=warnings,
        db_check_enabled=db_check_enabled,
    )

    ready = status in {
        "READY_FOR_BACKTEST",
        "READY_FOR_BACKTEST_WITHOUT_DB_CHECK",
    }

    result: Dict[str, Any] = {
        "status": status,
        "ready": bool(ready),
        "created_at_utc": utc_now_iso(),
        "run_tag": run_tag,
        "symbols": symbols,
        "from_ts": from_ts.isoformat(),
        "to_ts": to_ts.isoformat(),
        "thresholds": {
            "gate2_thr": float(args.gate2_thr),
            "gate4_thr": float(args.gate4_thr),
            "gate5_1_thr": float(args.gate5_1_thr),
            "gate5_3_thr": float(args.gate5_3_thr),
        },
        "costs": {
            "fee_side_bps": float(args.fee_side_bps),
            "slippage_side_bps": float(args.slippage_side_bps),
            "fee_side": float(args.fee_side_bps) / 10000.0,
            "slippage_side": float(args.slippage_side_bps) / 10000.0,
        },
        "execution_rules": {
            "entry_delay_seconds": int(args.entry_delay_seconds),
            "weekend_ban": bool(DEFAULT_WEEKEND_BAN),
            "weekend_ban_start": DEFAULT_WEEKEND_BAN_START,
            "weekend_ban_end": DEFAULT_WEEKEND_BAN_END,
            "one_slot": True,
            "one_position": True,
            "capital_mode": "slot1_100pct",
        },
        "model_selection": {
            "pair_model_name": str(args.pair_model_name),
            "grid_name": str(args.grid_name),
        },
        "parquet_check": {
            "m1_dir": str(m1_dir).replace("\\", "/"),
            "h4_dir": str(h4_dir).replace("\\", "/"),
            "symbols": parquet_reports,
        },
        "db_check": db_reports,
        "errors": errors,
        "warnings": warnings,
        "next_action": "RUN_BACKTEST" if ready else "FIX_REQUEST",
        "steps": [
            build_backtest_step(
                backtest_script=str(args.backtest_script),
                symbols=symbols,
                from_ts=from_ts,
                to_ts=to_ts,
                gate2_thr=float(args.gate2_thr),
                gate4_thr=float(args.gate4_thr),
                gate5_1_thr=float(args.gate5_1_thr),
                gate5_3_thr=float(args.gate5_3_thr),
                fee_side_bps=float(args.fee_side_bps),
                slippage_side_bps=float(args.slippage_side_bps),
                entry_delay_seconds=int(args.entry_delay_seconds),
                pair_model_name=str(args.pair_model_name),
                grid_name=str(args.grid_name),
                market_category=str(args.market_category),
                run_tag=run_tag,
            )
        ],
    }

    text = json.dumps(result, ensure_ascii=True, indent=2)
    print(text)

    if str(args.json_out).strip():
        out_path = Path(str(args.json_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
