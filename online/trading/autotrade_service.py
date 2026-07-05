from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import psycopg2

import pandas as pd

from online.trading import config
from online.trading import audit_log
from online.trading import notify
from online.trading.db_schema import ensure_trading_schema
from online.trading.db import db_cursor, read_sql


SERVICE_NAME = "gate_autotrade_service"

ROOT = config.ROOT
LOCK_PATH = ROOT / "online" / "_state_gate_autotrade_service.lock"

STARTUP_CATCHUP_PIPELINE = os.environ.get("IMB_STARTUP_CATCHUP_PIPELINE", "1").strip() != "0"
RUN_EXECUTION_ON_CATCHUP = os.environ.get("IMB_RUN_EXECUTION_ON_CATCHUP", "0").strip() == "1"

H4_AFTER_CLOSE_DELAY_SECONDS = int(os.environ.get("IMB_H4_AFTER_CLOSE_DELAY_SECONDS", "3"))
H4_LOOP_IDLE_SECONDS = int(os.environ.get("IMB_H4_LOOP_IDLE_SECONDS", "1"))
SAFETY_LOOP_ENABLED = os.environ.get("IMB_SAFETY_LOOP_ENABLED", "1").strip() != "0"
SAFETY_LOOP_INTERVAL_SECONDS = int(os.environ.get("IMB_SAFETY_LOOP_INTERVAL_SECONDS", "120"))

TRADE_MANAGEMENT_POLL_ENABLED = os.environ.get("IMB_TRADE_MANAGEMENT_POLL_ENABLED", "1").strip() != "0"
TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS = int(os.environ.get("IMB_TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS", "30"))
TRADE_MANAGEMENT_POLL_LOG_INTERVAL_SECONDS = int(os.environ.get("IMB_TRADE_MANAGEMENT_POLL_LOG_INTERVAL_SECONDS", "1800"))

EARLY_STOP_TIMER_ENABLED = os.environ.get("IMB_EARLY_STOP_TIMER_ENABLED", "1").strip() != "0"
EARLY_STOP_TIMER_IDLE_SECONDS = int(os.environ.get("IMB_EARLY_STOP_TIMER_IDLE_SECONDS", "60"))
EARLY_STOP_TIMER_MIN_SLEEP_SECONDS = int(os.environ.get("IMB_EARLY_STOP_TIMER_MIN_SLEEP_SECONDS", "1"))
EARLY_STOP_TIMER_MAX_SLEEP_SECONDS = int(os.environ.get("IMB_EARLY_STOP_TIMER_MAX_SLEEP_SECONDS", "3600"))

WS_LIFECYCLE_ENABLED = os.environ.get("IMB_WS_LIFECYCLE_ENABLED", "1").strip() != "0"
WS_LISTENER_ENABLED = os.environ.get("IMB_WS_LISTENER_ENABLED", "1").strip() != "0"
WS_LISTENER_MODULE = os.environ.get("IMB_WS_LISTENER_MODULE", "online.trading.WSListener").strip()
WS_LISTENER_LOG_PATH = ROOT / "logs" / "ws_listener_latest.log"
WS_LISTENER_WATCHDOG_INTERVAL_SECONDS = int(os.environ.get("IMB_WS_LISTENER_WATCHDOG_INTERVAL_SECONDS", "30"))

COUNTDOWN_LOG_INTERVAL_SECONDS = int(os.environ.get("IMB_COUNTDOWN_LOG_INTERVAL_SECONDS", "1200"))
PRE_CLOSE_NOTICE_1_SECONDS = int(os.environ.get("IMB_PRE_CLOSE_NOTICE_1_SECONDS", "60"))
PRE_CLOSE_NOTICE_2_SECONDS = int(os.environ.get("IMB_PRE_CLOSE_NOTICE_2_SECONDS", "10"))

STOP_EVENT: Optional[asyncio.Event] = None
WS_LISTENER_PROCESS: Optional[subprocess.Popen] = None

DB_LOCK_CONN = None
DB_LOCK_KEY_1 = 918273645
DB_LOCK_KEY_2 = 20260510


def ensure_trade_management_columns() -> None:
    sql = """
        ALTER TABLE public.trading_positions
            ADD COLUMN IF NOT EXISTS partial_tp_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS final_tp_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS early_stop_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS main_sl_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS rest_stop_after_partial_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS partial_tp_qty_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS final_tp_qty_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS early_stop_expires_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS trade_management_mode TEXT;
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)

    value = str(raw).strip().lower()
    if value in {"0", "false", "no", "off", "n"}:
        return False
    if value in {"1", "true", "yes", "on", "y"}:
        return True

    return bool(default)


def get_dry_run() -> bool:
    return env_bool("IMB_TRADING_DRY_RUN", False)
def get_db_dsn() -> str:
    return os.environ.get(
        "IMB_DB_DSN",
        config.DB_DSN,
    )


def acquire_db_advisory_lock() -> None:
    global DB_LOCK_CONN

    dsn = get_db_dsn()

    conn = psycopg2.connect(dsn)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(%s, %s)",
            (int(DB_LOCK_KEY_1), int(DB_LOCK_KEY_2)),
        )
        locked = bool(cur.fetchone()[0])

    if not locked:
        try:
            conn.close()
        except Exception:
            pass

        raise RuntimeError(
            "autotrade service already running: PostgreSQL advisory lock is already held "
            "for key=({}, {}) db={}".format(DB_LOCK_KEY_1, DB_LOCK_KEY_2, dsn)
        )

    DB_LOCK_CONN = conn

    print("=" * 120, flush=True)
    print("DB_ADVISORY_LOCK_ACQUIRED", flush=True)
    print("db:", dsn, flush=True)
    print("lock_key_1:", DB_LOCK_KEY_1, flush=True)
    print("lock_key_2:", DB_LOCK_KEY_2, flush=True)


def release_db_advisory_lock() -> None:
    global DB_LOCK_CONN

    conn = DB_LOCK_CONN
    DB_LOCK_CONN = None

    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock(%s, %s)",
                (int(DB_LOCK_KEY_1), int(DB_LOCK_KEY_2)),
            )
            unlocked = bool(cur.fetchone()[0])

        print("=" * 120, flush=True)
        print("DB_ADVISORY_LOCK_RELEASED", flush=True)
        print("unlocked:", unlocked, flush=True)

    except Exception as e:
        print("WARNING: failed to release DB advisory lock:", e, flush=True)

    try:
        conn.close()
    except Exception:
        pass


def next_h4_close_utc(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    ts = pd.to_datetime(now if now is not None else utc_now(), utc=True)
    h = (int(ts.hour) // 4) * 4
    current_h4_open = ts.replace(hour=h, minute=0, second=0, microsecond=0)
    close_ts = current_h4_open + pd.Timedelta(hours=4)
    if close_ts <= ts:
        close_ts = close_ts + pd.Timedelta(hours=4)
    return close_ts


def latest_closed_h4_open_utc(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    ts = pd.to_datetime(now if now is not None else utc_now(), utc=True)
    h = (int(ts.hour) // 4) * 4
    current_h4_open = ts.replace(hour=h, minute=0, second=0, microsecond=0)
    return current_h4_open - pd.Timedelta(hours=4)


def seconds_until(ts: pd.Timestamp) -> float:
    return max(0.0, (pd.to_datetime(ts, utc=True) - utc_now()).total_seconds())


def fmt_left(seconds: float) -> str:
    seconds_i = int(max(0, seconds))
    h = seconds_i // 3600
    m = (seconds_i % 3600) // 60
    s = seconds_i % 60
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)


def build_child_env() -> dict:
    env = os.environ.copy()

    root_txt = str(ROOT)
    old_pythonpath = env.get("PYTHONPATH", "").strip()
    if old_pythonpath:
        env["PYTHONPATH"] = root_txt + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = root_txt

    env.setdefault("PYTHONUNBUFFERED", "1")

    old_warnings = env.get("PYTHONWARNINGS", "").strip()
    extra_warning = "ignore:pandas only supports SQLAlchemy connectable:UserWarning"
    if old_warnings:
        env["PYTHONWARNINGS"] = old_warnings + "," + extra_warning
    else:
        env["PYTHONWARNINGS"] = extra_warning

    return env


def run_module(module_name: str, allow_fail: bool = False) -> int:
    started = time.time()

    print("=" * 120, flush=True)
    print("RUN_MODULE:", module_name, flush=True)
    print("STARTED_AT_UTC:", utc_now(), flush=True)

    cmd = [sys.executable, "-m", module_name]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=build_child_env(),
    )

    elapsed = time.time() - started

    print("FINISHED_MODULE:", module_name, flush=True)
    print("ELAPSED_SEC:", round(elapsed, 3), flush=True)
    print("RETURN_CODE:", proc.returncode, flush=True)

    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError("module failed: {} return_code={}".format(module_name, proc.returncode))

    return int(proc.returncode)


def run_module_capture(module_name: str, allow_fail: bool = False) -> Tuple[int, str]:
    started = time.time()

    cmd = [sys.executable, "-m", module_name]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=build_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    elapsed = time.time() - started
    output = str(proc.stdout or "")

    if proc.returncode != 0 and not allow_fail:
        print("=" * 120, flush=True)
        print("RUN_MODULE_FAILED:", module_name, flush=True)
        print("STARTED_AT_UTC:", utc_now(), flush=True)
        print("ELAPSED_SEC:", round(elapsed, 3), flush=True)
        print("RETURN_CODE:", proc.returncode, flush=True)
        print(output.rstrip(), flush=True)
        raise RuntimeError("module failed: {} return_code={}".format(module_name, proc.returncode))

    return int(proc.returncode), output

def start_ws_listener_process() -> Optional[subprocess.Popen]:
    if not WS_LISTENER_ENABLED:
        return None

    try:
        WS_LISTENER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        log_fh = open(str(WS_LISTENER_LOG_PATH), "a", encoding="utf-8", buffering=1)

        env = build_child_env()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", WS_LISTENER_MODULE],
            cwd=str(ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print("=" * 120, flush=True)
        print("WS_LISTENER_PROCESS_STARTED", flush=True)
        print("module:", WS_LISTENER_MODULE, flush=True)
        print("pid:", proc.pid, flush=True)
        print("log:", WS_LISTENER_LOG_PATH, flush=True)

        return proc

    except Exception as e:
        print("=" * 120, flush=True)
        print("WS_LISTENER_PROCESS_START_FAILED", flush=True)
        print("module:", WS_LISTENER_MODULE, flush=True)
        print("error:", e, flush=True)
        return None


def stop_ws_listener_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return

    try:
        if proc.poll() is not None:
            print("WS_LISTENER_PROCESS_ALREADY_EXITED", flush=True)
            print("pid:", proc.pid, flush=True)
            print("return_code:", proc.returncode, flush=True)
            return

        print("=" * 120, flush=True)
        print("WS_LISTENER_PROCESS_STOPPING", flush=True)
        print("pid:", proc.pid, flush=True)

        proc.terminate()

        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("WS_LISTENER_PROCESS_KILLING", flush=True)
            print("pid:", proc.pid, flush=True)
            proc.kill()
            proc.wait(timeout=10)

        print("WS_LISTENER_PROCESS_STOPPED", flush=True)
        print("pid:", proc.pid, flush=True)
        print("return_code:", proc.returncode, flush=True)

    except Exception as e:
        print("WS_LISTENER_PROCESS_STOP_ERROR:", e, flush=True)


async def ws_listener_watchdog_loop() -> None:
    global WS_LISTENER_PROCESS

    if not WS_LIFECYCLE_ENABLED:
        print("WS_LISTENER_WATCHDOG_DISABLED", flush=True)
        return

    print("WS_LISTENER_WATCHDOG_STARTED", flush=True)
    print("WS_LISTENER_MODULE:", WS_LISTENER_MODULE, flush=True)
    print("WS_LISTENER_WATCHDOG_INTERVAL_SECONDS:", WS_LISTENER_WATCHDOG_INTERVAL_SECONDS, flush=True)

    if WS_LISTENER_PROCESS is None:
        WS_LISTENER_PROCESS = start_ws_listener_process()

    while STOP_EVENT is None or not STOP_EVENT.is_set():
        try:
            if WS_LISTENER_PROCESS is None:
                WS_LISTENER_PROCESS = start_ws_listener_process()

            elif WS_LISTENER_PROCESS.poll() is not None:
                print("=" * 120, flush=True)
                print("WS_LISTENER_PROCESS_EXITED_RESTARTING", flush=True)
                print("old_pid:", WS_LISTENER_PROCESS.pid, flush=True)
                print("return_code:", WS_LISTENER_PROCESS.returncode, flush=True)

                WS_LISTENER_PROCESS = start_ws_listener_process()

        except Exception as e:
            print("WS_LISTENER_WATCHDOG_ERROR:", e, flush=True)

            write_audit_event(
                event_type="WS_LISTENER_WATCHDOG_ERROR",
                status="ERROR",
                message=str(e),
                payload={
                    "module": WS_LISTENER_MODULE,
                    "dry_run": get_dry_run(),
                },
            )

            notify_safe(
                event_type="WS_LISTENER_WATCHDOG_ERROR",
                status="ERROR",
                payload={
                    "error": str(e),
                    "module": WS_LISTENER_MODULE,
                    "dry_run": get_dry_run(),
                },
                force=True,
            )

        await sleep_interruptible(WS_LISTENER_WATCHDOG_INTERVAL_SECONDS)


def parse_last_json_line(text: str) -> Optional[dict]:
    for raw in reversed(str(text or "").splitlines()):
        line = raw.strip()
        if not line:
            continue

        if not line.startswith("{"):
            continue

        try:
            obj = json.loads(line)
        except Exception:
            continue

        if isinstance(obj, dict):
            return obj

    return None


def should_log_protective_orders_result(return_code: int, output: str) -> bool:
    if int(return_code) != 0:
        return True

    obj = parse_last_json_line(output)

    if obj is None:
        return True

    status = str(obj.get("status") or "").strip().upper()

    cancelled = int(obj.get("cancelled") or 0)
    failed = int(obj.get("failed") or 0)
    stale_orders = int(obj.get("stale_orders") or 0)
    checked_orders = int(obj.get("checked_orders") or 0)
    errors = obj.get("errors") or []

    if failed > 0:
        return True

    if cancelled > 0:
        return True

    if stale_orders > 0:
        return True

    if errors:
        return True

    if status not in {"NO_ACTIVE_PROTECTIVE_ORDERS", "DONE"}:
        return True

    if status == "DONE" and checked_orders > 0:
        return True

    return False

def run_step(script_path: Path, allow_fail: bool = False) -> int:
    if not script_path.exists():
        raise FileNotFoundError("pipeline step not found: {}".format(script_path))

    started = time.time()

    print("=" * 120, flush=True)
    print("RUN_STEP:", script_path, flush=True)
    print("STARTED_AT_UTC:", utc_now(), flush=True)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        env=build_child_env(),
    )

    elapsed = time.time() - started

    print("FINISHED_STEP:", script_path, flush=True)
    print("ELAPSED_SEC:", round(elapsed, 3), flush=True)
    print("RETURN_CODE:", proc.returncode, flush=True)

    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError("pipeline step failed: {} return_code={}".format(script_path, proc.returncode))

    return int(proc.returncode)


def read_lock_pid() -> Optional[int]:
    if not LOCK_PATH.exists():
        return None

    raw = LOCK_PATH.read_text(encoding="utf-8").strip()

    try:
        import ast
        payload = ast.literal_eval(raw)
        pid = payload.get("pid")
        if pid is None:
            return None
        return int(pid)
    except Exception:
        return None


def is_process_alive(pid: int) -> bool:
    try:
        pid_i = int(pid)
    except Exception:
        return False

    if pid_i <= 0:
        return False

    if os.name == "nt":
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "if (Get-Process -Id {} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}".format(pid_i),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return int(proc.returncode) == 0
        except Exception:
            return False

    try:
        os.kill(pid_i, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def acquire_process_lock() -> None:
    if LOCK_PATH.exists():
        old = LOCK_PATH.read_text(encoding="utf-8").strip()
        old_pid = read_lock_pid()

        if old_pid is not None and is_process_alive(old_pid):
            raise RuntimeError(
                "autotrade service already running: pid={} | lock={}".format(
                    old_pid,
                    LOCK_PATH,
                )
            )

        print("STALE_LOCK_REMOVED:", LOCK_PATH, flush=True)
        print("STALE_LOCK_CONTENT:", old, flush=True)
        LOCK_PATH.unlink()

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pid": os.getpid(),
        "created_at_utc": str(utc_now()),
        "service": SERVICE_NAME,
        "root": str(ROOT),
        "dry_run": get_dry_run(),
    }

    LOCK_PATH.write_text(str(payload), encoding="utf-8")


def release_process_lock() -> None:
    try:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    except Exception as e:
        print("WARNING: failed to remove lock {}: {}".format(LOCK_PATH, e), flush=True)


def install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    def _stop() -> None:
        if STOP_EVENT is not None:
            STOP_EVENT.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass


def write_audit_event(event_type: str, status: str, message: str, payload: Optional[dict] = None) -> None:
    try:
        audit_log.ensure_audit_tables()
        audit_log.log_audit_event(
            event_type=event_type,
            status=status,
            message=message,
            payload=payload or {},
        )
    except Exception as e:
        print("AUDIT_LOG_ERROR:", event_type, e, flush=True)


def notify_safe(event_type: str, status: str, payload: Optional[dict] = None, force: bool = False) -> None:
    try:
        notify.notify_event(
            event_type=event_type,
            status=status,
            payload=payload or {},
            force=force,
        )
    except Exception as e:
        print("NOTIFY_ERROR:", event_type, e, flush=True)


def safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def fmt_float(value, digits: int = 6) -> str:
    v = safe_float(value)
    if v is None:
        return "None"
    return ("{:.%df}" % digits).format(v)


def print_h4_decision_report(close_ts: pd.Timestamp) -> None:
    signal_ts = pd.to_datetime(close_ts, utc=True) - pd.Timedelta(hours=4)

    sql = """
        SELECT
            signal_key,
            symbol,
            side,
            signal_ts,
            entry_ts_plan,
            gate2_proba,
            gate4_confidence,
            gate5_1_proba,
            gate5_3_proba,
            signal_strength,
            candidate_rank,
            selected,
            rejected,
            reject_reason,
            updated_at
        FROM public.trading_signals
        WHERE signal_ts = %s
        ORDER BY
            selected DESC,
            signal_strength DESC NULLS LAST,
            gate4_confidence DESC NULLS LAST,
            gate2_proba DESC NULLS LAST,
            gate5_1_proba DESC NULLS LAST,
            gate5_3_proba DESC NULLS LAST,
            symbol ASC
    """

    try:
        df = read_sql(sql, [signal_ts.to_pydatetime()])
    except Exception as e:
        print("=" * 120, flush=True)
        print("H4_DECISION_REPORT_ERROR", flush=True)
        print("close_ts_utc:", close_ts, flush=True)
        print("signal_ts:", signal_ts, flush=True)
        print("error:", e, flush=True)
        return

    print("=" * 120, flush=True)
    print("H4_DECISION_REPORT", flush=True)
    print("close_ts_utc:", close_ts, flush=True)
    print("signal_ts:", signal_ts, flush=True)

    if df.empty:
        print("status: NO_ENTRY", flush=True)
        print("candidates_checked: 0", flush=True)
        print("message: no candidates were written to trading_signals for this H4", flush=True)
        return

    selected = df[df["selected"] == True].copy()

    if not selected.empty:
        row = selected.iloc[0]
        print("status: ENTRY_SELECTED", flush=True)
        print("candidates_checked:", len(df), flush=True)
        print("selected_symbol:", str(row.get("symbol") or "").upper(), flush=True)
        print("selected_side:", str(row.get("side") or "").upper(), flush=True)
        print("signal_key:", row.get("signal_key"), flush=True)
        print("entry_ts_plan:", row.get("entry_ts_plan"), flush=True)
        print("gate2:", fmt_float(row.get("gate2_proba")), "thr={}".format(config.GATE2_THR), flush=True)
        print("gate4:", fmt_float(row.get("gate4_confidence")), "thr={}".format(config.GATE4_THR), flush=True)
        print("gate5_1:", fmt_float(row.get("gate5_1_proba")), "thr={}".format(config.GATE5_1_THR), flush=True)
        print("gate5_3:", fmt_float(row.get("gate5_3_proba")), "thr={}".format(config.GATE5_3_THR), flush=True)
        print("signal_strength:", fmt_float(row.get("signal_strength")), flush=True)
        return

    row = df.iloc[0]

    print("status: NO_ENTRY", flush=True)
    print("candidates_checked:", len(df), flush=True)
    print("top_rejected_symbol:", str(row.get("symbol") or "").upper(), flush=True)
    print("top_rejected_side:", str(row.get("side") or "").upper(), flush=True)
    print("signal_key:", row.get("signal_key"), flush=True)
    print("entry_ts_plan:", row.get("entry_ts_plan"), flush=True)
    print("gate2:", fmt_float(row.get("gate2_proba")), "thr={}".format(config.GATE2_THR), flush=True)
    print("gate4:", fmt_float(row.get("gate4_confidence")), "thr={}".format(config.GATE4_THR), flush=True)
    print("gate5_1:", fmt_float(row.get("gate5_1_proba")), "thr={}".format(config.GATE5_1_THR), flush=True)
    print("gate5_3:", fmt_float(row.get("gate5_3_proba")), "thr={}".format(config.GATE5_3_THR), flush=True)
    print("signal_strength:", fmt_float(row.get("signal_strength")), flush=True)
    print("reject_reason:", row.get("reject_reason"), flush=True)

    print("-" * 120, flush=True)
    print("H4_CANDIDATES", flush=True)

    for _, cand in df.iterrows():
        cand_symbol = str(cand.get("symbol") or "").upper()
        cand_side = str(cand.get("side") or "").upper()
        cand_selected = bool(cand.get("selected"))
        cand_rejected = bool(cand.get("rejected"))

        cand_reason = str(cand.get("reject_reason") or "")
        if not cand_reason or cand_reason.lower() == "nan":
            cand_reason = "OK" if cand_selected and not cand_rejected else "NO_REASON"

        marker = "TOP" if str(cand.get("signal_key") or "") == str(row.get("signal_key") or "") else "   "

        print(
            "{} | rank={} | {} {} | g2={} / {} | g4={} / {} | g5_1={} / {} | g5_3={} / {} | strength={} | selected={} | rejected={} | reason={}".format(
                marker,
                cand.get("candidate_rank") if "candidate_rank" in cand.index else "-",
                cand_symbol,
                cand_side,
                fmt_float(cand.get("gate2_proba")),
                config.GATE2_THR,
                fmt_float(cand.get("gate4_confidence")),
                config.GATE4_THR,
                fmt_float(cand.get("gate5_1_proba")),
                config.GATE5_1_THR,
                fmt_float(cand.get("gate5_3_proba")),
                config.GATE5_3_THR,
                fmt_float(cand.get("signal_strength")),
                cand_selected,
                cand_rejected,
                cand_reason,
            ),
            flush=True,
        )


def first_existing_table_column(table_name: str, candidates) -> Optional[str]:
    try:
        df = read_sql(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
            """,
            [str(table_name)],
        )
    except Exception as e:
        print("FIRST_EXISTING_TABLE_COLUMN_ERROR:", table_name, e, flush=True)
        return None

    if df.empty:
        return None

    cols = set(str(x) for x in df["column_name"].tolist())

    for candidate in candidates:
        if str(candidate) in cols:
            return str(candidate)

    return None


def backfill_catchup_trading_signals_from_gate5_3() -> None:
    g53_proba_col = first_existing_table_column(
        "online_gate5_3_decisions",
        [
            "gate5_3_proba",
            "pred_proba",
            "proba",
            "decision_proba",
            "selected_proba",
            "probability",
        ],
    )

    if g53_proba_col is None:
        print("CATCHUP_HISTORY_BACKFILL_SKIPPED: no Gate5.3 probability column", flush=True)
        return

    sql = """
    WITH missing_ts AS (
        SELECT DISTINCT g53.signal_ts
        FROM public.online_gate5_3_decisions g53
        LEFT JOIN public.trading_signals s
            ON s.signal_ts = g53.signal_ts
        WHERE g53.signal_ts >= NOW() - INTERVAL '14 days'
          AND g53.signal_ts < NOW()
        GROUP BY g53.signal_ts
        HAVING COUNT(s.signal_key) = 0
    ),
    base AS (
        SELECT
            g53.signal_ts,
            g53.symbol,
            UPPER(COALESCE(g53.side, '')) AS side,

            COALESCE(NULLIF(g53.pair_model_name, ''), %s::text) AS pair_model_name,
            COALESCE(NULLIF(g53.chosen_grid_name, ''), %s::text) AS grid_name,
            COALESCE(g53.chosen_tp_atr, %s::double precision) AS tp_atr,
            COALESCE(g53.chosen_sl_atr, %s::double precision) AS sl_atr,
            %s::double precision AS ttl_hours,

            gf.close::double precision AS h4_close,
            gf.atr14::double precision AS atr14,
            gf.gate2_proba::double precision AS gate2_proba,

            g52.gate4_confidence::double precision AS gate4_confidence,
            g51.gate5_1_proba::double precision AS gate5_1_proba,
            g53.{g53_proba_col}::double precision AS gate5_3_proba
        FROM public.online_gate5_3_decisions g53
        JOIN missing_ts mt
            ON mt.signal_ts = g53.signal_ts

        LEFT JOIN public.online_gate5_2_ranker g52
            ON g52.signal_ts = g53.signal_ts
           AND g52.symbol = g53.symbol
           AND UPPER(COALESCE(g52.side, '')) = UPPER(COALESCE(g53.side, ''))
           AND COALESCE(NULLIF(g52.grid_name, ''), '') = COALESCE(NULLIF(g53.chosen_grid_name, ''), '')

        LEFT JOIN public.online_gate5_1_scores g51
            ON g51.signal_ts = g53.signal_ts
           AND g51.symbol = g53.symbol
           AND UPPER(COALESCE(g51.side, '')) = UPPER(COALESCE(g53.side, ''))
           AND COALESCE(NULLIF(g51.grid_name, ''), '') = COALESCE(NULLIF(g53.chosen_grid_name, ''), '')

        LEFT JOIN public.online_gate4_features gf
            ON gf.symbol = g53.symbol
           AND gf.entry_ts = g53.signal_ts
    ),
    ranked AS (
        SELECT
            *,
            (
                COALESCE(gate2_proba, 0)
                + COALESCE(gate4_confidence, 0)
                + COALESCE(gate5_1_proba, 0)
                + COALESCE(gate5_3_proba, 0)
            ) AS signal_strength,
            ROW_NUMBER() OVER (
                PARTITION BY signal_ts
                ORDER BY
                    (
                        COALESCE(gate2_proba, 0)
                        + COALESCE(gate4_confidence, 0)
                        + COALESCE(gate5_1_proba, 0)
                        + COALESCE(gate5_3_proba, 0)
                    ) DESC,
                    symbol ASC,
                    side ASC
            ) AS candidate_rank
        FROM base
    ),
    prepared AS (
        SELECT
            'catchup_' || md5(signal_ts::text || '|' || symbol || '|' || side || '|' || pair_model_name || '|' || grid_name) AS signal_key,
            signal_ts,
            symbol,
            side,
            signal_ts + INTERVAL '4 hours' AS entry_ts_plan,
            pair_model_name,
            grid_name,
            tp_atr,
            sl_atr,
            ttl_hours,
            gate2_proba,
            gate4_confidence,
            gate5_1_proba,
            gate5_3_proba,
            signal_strength,
            FALSE AS selected,
            TRUE AS rejected,
            CASE
                WHEN COALESCE(gate2_proba, 0) < %s THEN 'BELOW_GATE2'
                WHEN COALESCE(gate4_confidence, 0) < %s THEN 'BELOW_GATE4'
                WHEN COALESCE(gate5_1_proba, 0) < %s THEN 'BELOW_GATE5_1'
                WHEN COALESCE(gate5_3_proba, 0) < %s THEN 'BELOW_GATE5_3'
                ELSE 'CATCHUP_ENTRY_WINDOW_EXPIRED'
            END AS reject_reason,
            h4_close,
            atr14,
            NULL::boolean AS dynamic_symbol_allowed,
            'CATCHUP_HISTORY_BACKFILL_NO_EXECUTION'::text AS dynamic_symbol_reason,
            candidate_rank,
            'catchup_gate5_3_history_v1'::text AS selector_version,
            NOW() AS created_at,
            NOW() AS updated_at
        FROM ranked
    )
    INSERT INTO public.trading_signals (
        signal_key,
        symbol,
        signal_ts,
        entry_ts_plan,
        side,
        pair_model_name,
        grid_name,
        tp_atr,
        sl_atr,
        ttl_hours,
        gate2_proba,
        gate4_confidence,
        gate5_1_proba,
        gate5_3_proba,
        signal_strength,
        selected,
        rejected,
        reject_reason,
        created_at,
        updated_at,
        h4_close,
        atr14,
        dynamic_symbol_allowed,
        dynamic_symbol_reason,
        candidate_rank,
        selector_version
    )
    SELECT
        signal_key,
        symbol,
        signal_ts,
        entry_ts_plan,
        side,
        pair_model_name,
        grid_name,
        tp_atr,
        sl_atr,
        ttl_hours,
        gate2_proba,
        gate4_confidence,
        gate5_1_proba,
        gate5_3_proba,
        signal_strength,
        selected,
        rejected,
        reject_reason,
        created_at,
        updated_at,
        h4_close,
        atr14,
        dynamic_symbol_allowed,
        dynamic_symbol_reason,
        candidate_rank,
        selector_version
    FROM prepared
    WHERE NOT EXISTS (
        SELECT 1
        FROM public.trading_signals s
        WHERE s.signal_ts = prepared.signal_ts
          AND s.symbol = prepared.symbol
          AND UPPER(COALESCE(s.side, '')) = UPPER(COALESCE(prepared.side, ''))
    )
    """.format(
        g53_proba_col=g53_proba_col,
    )

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(config.PAIR_MODEL_NAME),
                str(config.GRID_NAME),
                float(config.TP_ATR),
                float(config.SL_ATR),
                float(config.TTL_HOURS),
                float(config.GATE2_THR),
                float(config.GATE4_THR),
                float(config.GATE5_1_THR),
                float(config.GATE5_3_THR),
            ),
        )
        inserted = cur.rowcount

    print("=" * 120, flush=True)
    print("CATCHUP_HISTORY_BACKFILL_FROM_GATE5_3", flush=True)
    print("inserted_trading_signals:", int(inserted), flush=True)
    print("mode: no execution, history/report only", flush=True)

async def sleep_interruptible(seconds: float) -> None:
    if seconds <= 0:
        return

    if STOP_EVENT is None:
        await asyncio.sleep(seconds)
        return

    try:
        await asyncio.wait_for(STOP_EVENT.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return


async def startup_catchup_once() -> None:
    if not STARTUP_CATCHUP_PIPELINE:
        print("STARTUP_CATCHUP_DISABLED", flush=True)
        return

    print("=" * 120, flush=True)
    print("STARTUP_CATCHUP_ONCE", flush=True)
    print("DRY_RUN:", get_dry_run(), flush=True)
    print("RUN_EXECUTION_ON_CATCHUP:", RUN_EXECUTION_ON_CATCHUP, flush=True)
    print("LATEST_CLOSED_H4_OPEN_UTC:", latest_closed_h4_open_utc(), flush=True)

    write_audit_event(
        event_type="AUTOTRADE_STARTUP_CATCHUP_START",
        status="STARTED",
        message="Startup catch-up pipeline started",
        payload={
            "dry_run": get_dry_run(),
            "run_execution_on_catchup": RUN_EXECUTION_ON_CATCHUP,
            "latest_closed_h4_open_utc": str(latest_closed_h4_open_utc()),
        },
    )

    try:
        if RUN_EXECUTION_ON_CATCHUP:
            run_module("online.trading.orchestrator", allow_fail=False)
        else:
            run_module("online.sync_candles_h4", allow_fail=False)

            for step in config.ONLINE_PIPELINE_STEPS:
                step_path = Path(step)
                if step_path.name == "sync_candles_h4.py":
                    continue
                run_step(step_path, allow_fail=False)

            catchup_close_ts = latest_closed_h4_open_utc() + pd.Timedelta(hours=4)

            try:
                backfill_catchup_trading_signals_from_gate5_3()
            except Exception as e:
                print("=" * 120, flush=True)
                print("STARTUP_CATCHUP_HISTORY_BACKFILL_ERROR", flush=True)
                print("error_type:", type(e).__name__, flush=True)
                print("error:", e, flush=True)
                print("message: catchup history backfill failed, but autotrade service startup continues", flush=True)

                write_audit_event(
                    event_type="STARTUP_CATCHUP_HISTORY_BACKFILL_ERROR",
                    status="ERROR",
                    message=str(e),
                    payload={
                        "dry_run": get_dry_run(),
                    },
                )

                notify_safe(
                    event_type="STARTUP_CATCHUP_HISTORY_BACKFILL_ERROR",
                    status="ERROR",
                    payload={
                        "error": str(e),
                        "dry_run": get_dry_run(),
                    },
                    force=True,
                )

            print("=" * 120, flush=True)
            print("STARTUP_CATCHUP_HISTORY_REPORTED", flush=True)
            print("reason: catchup_without_execution_writes_history_only", flush=True)
            print("catchup_close_ts:", catchup_close_ts, flush=True)

            print_h4_decision_report(catchup_close_ts)

        write_audit_event(
            event_type="AUTOTRADE_STARTUP_CATCHUP_DONE",
            status="OK",
            message="Startup catch-up pipeline finished",
            payload={
                "dry_run": get_dry_run(),
                "run_execution_on_catchup": RUN_EXECUTION_ON_CATCHUP,
            },
        )

    except Exception as e:
        write_audit_event(
            event_type="AUTOTRADE_STARTUP_CATCHUP_ERROR",
            status="ERROR",
            message=str(e),
            payload={
                "dry_run": get_dry_run(),
            },
        )
        notify_safe(
            event_type="AUTOTRADE_STARTUP_CATCHUP_ERROR",
            status="ERROR",
            payload={"error": str(e)},
            force=True,
        )
        raise

def has_active_db_position() -> bool:
    sql = """
        SELECT 1
        FROM public.trading_positions
        WHERE status IN (
            'ENTRY_ORDER_SENT',
            'ENTRY_PARTIALLY_FILLED',
            'ENTRY_FILLED',
            'TP_SL_PLACED',
            'POSITION_OPEN',
            'TTL_CLOSE_SENT',
            'TTL_CLOSE_FAILED'
        )
        LIMIT 1
    """

    try:
        df = read_sql(sql)
    except Exception as e:
        print("SAFETY_LOOP_ACTIVE_POSITION_CHECK_ERROR:", e, flush=True)
        return False

    return not df.empty


def get_next_early_stop_due_ts() -> Optional[pd.Timestamp]:
    sql = """
        SELECT MIN(early_stop_expires_at) AS due_ts
        FROM public.trading_positions
        WHERE status IN (
            'ENTRY_ORDER_SENT',
            'ENTRY_PARTIALLY_FILLED',
            'ENTRY_FILLED',
            'TP_SL_PLACED',
            'POSITION_OPEN',
            'TTL_CLOSE_SENT',
            'TTL_CLOSE_FAILED'
        )
          AND early_stop_expires_at IS NOT NULL
          AND trade_management_mode IS NOT NULL
    """

    try:
        df = read_sql(sql)
    except Exception as e:
        print("EARLY_STOP_TIMER_DUE_TS_ERROR:", e, flush=True)
        return None

    if df.empty:
        return None

    due_ts = pd.to_datetime(df.iloc[0].get("due_ts"), utc=True, errors="coerce")

    if pd.isna(due_ts):
        return None

    return due_ts


def is_soft_bybit_network_output(output: str) -> bool:
    text = str(output or "").lower()

    markers = [
        "read timed out",
        "readtimeout",
        "handshake operation timed out",
        "socket.timeout",
        "retryable error occurred",
        "recv_window",
        "errcode: 10002",
        "10002",
        "httpsconnectionpool",
        "api.bybit.com",
        "connection aborted",
        "connection reset",
        "max retries exceeded",
    ]

    return any(marker in text for marker in markers)


def compact_soft_module_error(module_name: str, return_code: int, output: str) -> str:
    lines = []

    for raw in str(output or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        line_l = line.lower()

        if "read timed out" in line_l:
            lines.append(line)
        elif "handshake operation timed out" in line_l:
            lines.append(line)
        elif "socket.timeout" in line_l:
            lines.append(line)
        elif "retryable error occurred" in line_l:
            lines.append(line)
        elif "recv_window" in line_l:
            lines.append(line)
        elif "10002" in line_l:
            lines.append(line)
        elif "httpsconnectionpool" in line_l:
            lines.append(line)
        elif "requests.exceptions" in line_l:
            lines.append(line)

    if not lines:
        lines = ["soft network error"]

    return "{}_SOFT_NETWORK_ERROR rc={} | {}".format(
        str(module_name).upper().replace(".", "_"),
        int(return_code),
        " | ".join(lines[-3:])[:1000],
    )


def run_trade_management_sync_capture(reason: str) -> Tuple[bool, str]:
    chunks = []
    ok = True

    rc, out = run_module_capture("online.trading.reconcile", allow_fail=True)
    if int(rc) == 0:
        chunks.append("RECONCILE rc={}\n{}".format(rc, out.rstrip()))
    elif is_soft_bybit_network_output(out):
        chunks.append(compact_soft_module_error("online.trading.reconcile", int(rc), out))
    else:
        chunks.append("RECONCILE rc={}\n{}".format(rc, out.rstrip()))
        ok = False

    rc, out = run_module_capture("online.trading.monitor", allow_fail=True)
    if int(rc) == 0:
        chunks.append("MONITOR rc={}\n{}".format(rc, out.rstrip()))
    elif is_soft_bybit_network_output(out):
        chunks.append(compact_soft_module_error("online.trading.monitor", int(rc), out))
    else:
        chunks.append("MONITOR rc={}\n{}".format(rc, out.rstrip()))
        ok = False

    rc, out = run_module_capture("online.trading.cancel_stale_protective_orders", allow_fail=True)
    if int(rc) == 0:
        chunks.append("CANCEL_STALE rc={}\n{}".format(rc, out.rstrip()))
    elif is_soft_bybit_network_output(out):
        chunks.append(compact_soft_module_error("online.trading.cancel_stale_protective_orders", int(rc), out))
    else:
        chunks.append("CANCEL_STALE rc={}\n{}".format(rc, out.rstrip()))
        ok = False

    return bool(ok), "\n".join(chunks)


def output_has_important_trade_management_event(output: str) -> bool:
    text = str(output or "")
    text_u = text.upper()

    soft_markers = [
        "ONLINE_TRADING_RECONCILE_SOFT_NETWORK_ERROR",
        "ONLINE_TRADING_MONITOR_SOFT_NETWORK_ERROR",
        "ONLINE_TRADING_CANCEL_STALE_PROTECTIVE_ORDERS_SOFT_NETWORK_ERROR",
        "LOCK_BUSY: TRADING_RECONCILE",
    ]

    text_for_check = text_u

    for marker in soft_markers:
        text_for_check = text_for_check.replace(marker, "")

    hard_markers = [
        "REST_STOP_AFTER_PARTIAL_PLACED",
        "REST_STOP_AFTER_PARTIAL_FAILED",
        "EARLY_STOP_REPLACED_WITH_MAIN_SL",
        "MAIN_SL_PLACE_FAILED",
        "TTL_CLOSE_SENT",
        "TTL_CLOSE_FAILED",
        "EXCHANGE_POSITION_ZERO_DETECTED",
        "POSITION_ZERO_CANCEL_ALL_PROTECTIVE_ORDERS",
        "CANCEL_FAILED",
        "RECONCILE_CLOSED_POSITION_CLEANUP_WARNING",
    ]

    for marker in hard_markers:
        if marker in text_for_check:
            return True

    for raw in text.splitlines():
        line = raw.strip()
        line_u = line.upper()

        if not line:
            continue

        if line_u == "LOCK_BUSY: TRADING_RECONCILE":
            continue

        if "_SOFT_NETWORK_ERROR" in line_u:
            continue

        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
            except Exception:
                obj = None

            if isinstance(obj, dict):
                status = str(obj.get("status") or "").upper()
                cancelled = int(obj.get("cancelled") or 0)
                failed = int(obj.get("failed") or 0)
                stale_orders = int(obj.get("stale_orders") or 0)
                errors = obj.get("errors") or []

                if failed > 0 or cancelled > 0 or stale_orders > 0 or errors:
                    return True

                if status not in {"NO_ACTIVE_PROTECTIVE_ORDERS", "DONE"}:
                    return True

                continue

        if line_u.startswith("ERROR:"):
            return True

        if line_u.startswith("FAILED:"):
            return True

    return False


async def trade_management_poll_loop() -> None:
    if not TRADE_MANAGEMENT_POLL_ENABLED:
        print("TRADE_MANAGEMENT_POLL_DISABLED", flush=True)
        return

    print("TRADE_MANAGEMENT_POLL_STARTED", flush=True)
    print("TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS:", TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS, flush=True)
    print("TRADE_MANAGEMENT_POLL_LOG_INTERVAL_SECONDS:", TRADE_MANAGEMENT_POLL_LOG_INTERVAL_SECONDS, flush=True)

    last_log_mono = 0.0

    while STOP_EVENT is None or not STOP_EVENT.is_set():
        try:
            active_position_exists = has_active_db_position()

            if not active_position_exists:
                await sleep_interruptible(TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS)
                continue

            started_at = utc_now()
            ok, output = run_trade_management_sync_capture(reason="POLL_30_SECONDS")

            now_mono = time.time()
            log_by_interval = (now_mono - last_log_mono) >= float(TRADE_MANAGEMENT_POLL_LOG_INTERVAL_SECONDS)
            log_by_event = output_has_important_trade_management_event(output)
            log_by_error = not ok

            if log_by_interval or log_by_event or log_by_error:
                print("=" * 120, flush=True)
                print("TRADE_MANAGEMENT_POLL_SYNC", flush=True)
                print("started_at_utc:", started_at, flush=True)
                print("finished_at_utc:", utc_now(), flush=True)
                print("dry_run:", get_dry_run(), flush=True)
                print("ok:", ok, flush=True)
                print("log_reason_interval:", log_by_interval, flush=True)
                print("log_reason_event:", log_by_event, flush=True)
                print("log_reason_error:", log_by_error, flush=True)
                if output.strip():
                    print(output.rstrip(), flush=True)
                last_log_mono = now_mono

            if not ok:
                write_audit_event(
                    event_type="TRADE_MANAGEMENT_POLL_ERROR",
                    status="ERROR",
                    message="trade management poll failed",
                    payload={
                        "output_tail": str(output or "")[-3000:],
                        "dry_run": get_dry_run(),
                    },
                )

                notify_safe(
                    event_type="TRADE_MANAGEMENT_POLL_ERROR",
                    status="ERROR",
                    payload={
                        "output_tail": str(output or "")[-1000:],
                        "dry_run": get_dry_run(),
                    },
                    force=True,
                )

        except Exception as e:
            write_audit_event(
                event_type="TRADE_MANAGEMENT_POLL_FATAL_ERROR",
                status="ERROR",
                message=str(e),
                payload={"dry_run": get_dry_run()},
            )

            notify_safe(
                event_type="TRADE_MANAGEMENT_POLL_FATAL_ERROR",
                status="ERROR",
                payload={"error": str(e), "dry_run": get_dry_run()},
                force=True,
            )

            print("TRADE_MANAGEMENT_POLL_FATAL_ERROR:", e, flush=True)

        await sleep_interruptible(TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS)


async def early_stop_timer_loop() -> None:
    if not EARLY_STOP_TIMER_ENABLED:
        print("EARLY_STOP_TIMER_DISABLED", flush=True)
        return

    print("EARLY_STOP_TIMER_STARTED", flush=True)
    print("EARLY_STOP_TIMER_IDLE_SECONDS:", EARLY_STOP_TIMER_IDLE_SECONDS, flush=True)
    print("EARLY_STOP_TIMER_MIN_SLEEP_SECONDS:", EARLY_STOP_TIMER_MIN_SLEEP_SECONDS, flush=True)
    print("EARLY_STOP_TIMER_MAX_SLEEP_SECONDS:", EARLY_STOP_TIMER_MAX_SLEEP_SECONDS, flush=True)

    while STOP_EVENT is None or not STOP_EVENT.is_set():
        try:
            due_ts = get_next_early_stop_due_ts()

            if due_ts is None:
                await sleep_interruptible(EARLY_STOP_TIMER_IDLE_SECONDS)
                continue

            left_seconds = seconds_until(due_ts)

            if left_seconds > 0:
                sleep_seconds = min(
                    float(EARLY_STOP_TIMER_MAX_SLEEP_SECONDS),
                    max(float(EARLY_STOP_TIMER_MIN_SLEEP_SECONDS), float(left_seconds)),
                )

                print("=" * 120, flush=True)
                print("EARLY_STOP_TIMER_WAIT", flush=True)
                print("next_due_ts_utc:", due_ts, flush=True)
                print("left:", fmt_left(left_seconds), flush=True)
                print("sleep_seconds:", round(sleep_seconds, 3), flush=True)

                await sleep_interruptible(sleep_seconds)
                continue

            started_at = utc_now()
            ok, output = run_trade_management_sync_capture(reason="EARLY_STOP_EXPIRES_AT_DUE")

            print("=" * 120, flush=True)
            print("EARLY_STOP_TIMER_SYNC", flush=True)
            print("started_at_utc:", started_at, flush=True)
            print("finished_at_utc:", utc_now(), flush=True)
            print("dry_run:", get_dry_run(), flush=True)
            print("ok:", ok, flush=True)
            if output.strip():
                print(output.rstrip(), flush=True)

            if not ok:
                write_audit_event(
                    event_type="EARLY_STOP_TIMER_ERROR",
                    status="ERROR",
                    message="early stop timer sync failed",
                    payload={
                        "output_tail": str(output or "")[-3000:],
                        "dry_run": get_dry_run(),
                    },
                )

                notify_safe(
                    event_type="EARLY_STOP_TIMER_ERROR",
                    status="ERROR",
                    payload={
                        "output_tail": str(output or "")[-1000:],
                        "dry_run": get_dry_run(),
                    },
                    force=True,
                )

            await sleep_interruptible(EARLY_STOP_TIMER_MIN_SLEEP_SECONDS)

        except Exception as e:
            write_audit_event(
                event_type="EARLY_STOP_TIMER_FATAL_ERROR",
                status="ERROR",
                message=str(e),
                payload={"dry_run": get_dry_run()},
            )

            notify_safe(
                event_type="EARLY_STOP_TIMER_FATAL_ERROR",
                status="ERROR",
                payload={"error": str(e), "dry_run": get_dry_run()},
                force=True,
            )

            print("EARLY_STOP_TIMER_FATAL_ERROR:", e, flush=True)
            await sleep_interruptible(EARLY_STOP_TIMER_IDLE_SECONDS)


async def safety_loop() -> None:
    if not SAFETY_LOOP_ENABLED:
        print("SAFETY_LOOP_DISABLED", flush=True)
        return

    print("SAFETY_LOOP_STARTED", flush=True)
    print("SAFETY_LOOP_INTERVAL_SECONDS:", SAFETY_LOOP_INTERVAL_SECONDS, flush=True)

    while STOP_EVENT is None or not STOP_EVENT.is_set():
        try:
            started_at = utc_now()
            active_position_exists = has_active_db_position()
            if active_position_exists and WS_LIFECYCLE_ENABLED:
                await sleep_interruptible(SAFETY_LOOP_INTERVAL_SECONDS)
                continue

            if active_position_exists and TRADE_MANAGEMENT_POLL_ENABLED:
                await sleep_interruptible(SAFETY_LOOP_INTERVAL_SECONDS)
                continue

            if active_position_exists:
                print("=" * 120, flush=True)
                print("SAFETY_LOOP_ACTIVE_POSITION_SYNC_START", flush=True)
                print("started_at_utc:", started_at, flush=True)
                print("dry_run:", get_dry_run(), flush=True)

                run_module("online.trading.reconcile", allow_fail=True)
                run_module("online.trading.monitor", allow_fail=True)

                return_code, output = run_module_capture(
                    "online.trading.cancel_stale_protective_orders",
                    allow_fail=True,
                )

                if should_log_protective_orders_result(return_code, output):
                    print("=" * 120, flush=True)
                    print("SAFETY_LOOP_PROTECTIVE_ORDERS_RESULT", flush=True)
                    print("started_at_utc:", started_at, flush=True)
                    print("dry_run:", get_dry_run(), flush=True)
                    print("return_code:", return_code, flush=True)

                    if output.strip():
                        print(output.rstrip(), flush=True)

                if int(return_code) != 0:
                    write_audit_event(
                        event_type="SAFETY_LOOP_PROTECTIVE_ORDERS_ERROR",
                        status="ERROR",
                        message="cancel_stale_protective_orders failed",
                        payload={
                            "return_code": int(return_code),
                            "output": str(output or "")[-3000:],
                            "dry_run": get_dry_run(),
                        },
                    )

                    notify_safe(
                        event_type="SAFETY_LOOP_PROTECTIVE_ORDERS_ERROR",
                        status="ERROR",
                        payload={
                            "return_code": int(return_code),
                            "output_tail": str(output or "")[-1000:],
                            "dry_run": get_dry_run(),
                        },
                        force=True,
                    )

                print("SAFETY_LOOP_ACTIVE_POSITION_SYNC_DONE", flush=True)
                print("finished_at_utc:", utc_now(), flush=True)

            else:
                return_code, output = run_module_capture(
                    "online.trading.cancel_stale_protective_orders",
                    allow_fail=True,
                )

                if should_log_protective_orders_result(return_code, output):
                    print("=" * 120, flush=True)
                    print("SAFETY_LOOP_PROTECTIVE_ORDERS_RESULT", flush=True)
                    print("started_at_utc:", started_at, flush=True)
                    print("dry_run:", get_dry_run(), flush=True)
                    print("return_code:", return_code, flush=True)

                    if output.strip():
                        print(output.rstrip(), flush=True)

                if int(return_code) != 0:
                    write_audit_event(
                        event_type="SAFETY_LOOP_PROTECTIVE_ORDERS_ERROR",
                        status="ERROR",
                        message="cancel_stale_protective_orders failed",
                        payload={
                            "return_code": int(return_code),
                            "output": str(output or "")[-3000:],
                            "dry_run": get_dry_run(),
                        },
                    )

                    notify_safe(
                        event_type="SAFETY_LOOP_PROTECTIVE_ORDERS_ERROR",
                        status="ERROR",
                        payload={
                            "return_code": int(return_code),
                            "output_tail": str(output or "")[-1000:],
                            "dry_run": get_dry_run(),
                        },
                        force=True,
                    )

        except Exception as e:
            write_audit_event(
                event_type="SAFETY_LOOP_ERROR",
                status="ERROR",
                message=str(e),
                payload={
                    "dry_run": get_dry_run(),
                },
            )

            notify_safe(
                event_type="SAFETY_LOOP_ERROR",
                status="ERROR",
                payload={
                    "error": str(e),
                    "dry_run": get_dry_run(),
                },
                force=True,
            )

            print("SAFETY_LOOP_ERROR:", e, flush=True)

        await sleep_interruptible(SAFETY_LOOP_INTERVAL_SECONDS)

async def wait_until_h4_target(close_ts: pd.Timestamp, target_ts: pd.Timestamp) -> None:
    last_countdown_log_mono = 0.0
    notice_1_sent = False
    notice_2_sent = False

    while STOP_EVENT is None or not STOP_EVENT.is_set():
        left_to_close = seconds_until(close_ts)
        left_to_target = seconds_until(target_ts)
        now_mono = time.time()

        if now_mono - last_countdown_log_mono >= COUNTDOWN_LOG_INTERVAL_SECONDS:
            print("=" * 120, flush=True)
            print("COUNTDOWN_TO_NEXT_H4", flush=True)
            print("close_ts_utc:", close_ts, flush=True)
            print("target_ts_utc:", target_ts, flush=True)
            print("left_to_close:", fmt_left(left_to_close), flush=True)
            print("left_to_run:", fmt_left(left_to_target), flush=True)
            last_countdown_log_mono = now_mono

        if left_to_close <= PRE_CLOSE_NOTICE_1_SECONDS and not notice_1_sent:
            print("=" * 120, flush=True)
            print("PRE_H4_CLOSE_NOTICE", flush=True)
            print("seconds_before_close:", PRE_CLOSE_NOTICE_1_SECONDS, flush=True)
            print("close_ts_utc:", close_ts, flush=True)
            print("left_to_close:", fmt_left(left_to_close), flush=True)
            notice_1_sent = True

        if left_to_close <= PRE_CLOSE_NOTICE_2_SECONDS and not notice_2_sent:
            print("=" * 120, flush=True)
            print("PRE_H4_CLOSE_NOTICE", flush=True)
            print("seconds_before_close:", PRE_CLOSE_NOTICE_2_SECONDS, flush=True)
            print("close_ts_utc:", close_ts, flush=True)
            print("left_to_close:", fmt_left(left_to_close), flush=True)
            notice_2_sent = True

        if left_to_target <= 0:
            return

        await sleep_interruptible(min(1.0, left_to_target))


async def h4_close_loop() -> None:
    print("H4_CLOSE_LOOP_STARTED", flush=True)

    while STOP_EVENT is None or not STOP_EVENT.is_set():
        close_ts = next_h4_close_utc()
        target_ts = close_ts + pd.Timedelta(seconds=H4_AFTER_CLOSE_DELAY_SECONDS)

        await wait_until_h4_target(close_ts=close_ts, target_ts=target_ts)

        if STOP_EVENT is not None and STOP_EVENT.is_set():
            break

        started_at = utc_now()
        drift_sec = (started_at - target_ts).total_seconds()

        print("=" * 120, flush=True)
        print("H4_PIPELINE_START", flush=True)
        print("close_ts_utc:", close_ts, flush=True)
        print("target_ts_utc:", target_ts, flush=True)
        print("started_at_utc:", started_at, flush=True)
        print("drift_sec:", round(drift_sec, 3), flush=True)
        print("dry_run:", get_dry_run(), flush=True)

        write_audit_event(
            event_type="H4_PIPELINE_START",
            status="STARTED",
            message="H4 autotrade pipeline started",
            payload={
                "close_ts_utc": str(close_ts),
                "target_ts_utc": str(target_ts),
                "started_at_utc": str(started_at),
                "drift_sec": drift_sec,
                "dry_run": get_dry_run(),
            },
        )

        notify_safe(
            event_type="H4_PIPELINE_START",
            status="STARTED",
            payload={
                "close_ts_utc": str(close_ts),
                "dry_run": get_dry_run(),
            },
            force=False,
        )

        try:
            run_module("online.trading.orchestrator", allow_fail=False)

            print_h4_decision_report(close_ts)

            print("=" * 120, flush=True)
            print("H4_PIPELINE_DONE", flush=True)
            print("close_ts_utc:", close_ts, flush=True)
            print("finished_at_utc:", utc_now(), flush=True)
            print("dry_run:", get_dry_run(), flush=True)

            write_audit_event(
                event_type="H4_PIPELINE_DONE",
                status="OK",
                message="H4 autotrade pipeline finished",
                payload={
                    "close_ts_utc": str(close_ts),
                    "finished_at_utc": str(utc_now()),
                    "dry_run": get_dry_run(),
                },
            )

            notify_safe(
                event_type="H4_PIPELINE_DONE",
                status="OK",
                payload={
                    "close_ts_utc": str(close_ts),
                    "dry_run": get_dry_run(),
                },
                force=False,
            )

        except Exception as e:
            write_audit_event(
                event_type="H4_PIPELINE_ERROR",
                status="ERROR",
                message=str(e),
                payload={
                    "close_ts_utc": str(close_ts),
                    "dry_run": get_dry_run(),
                },
            )

            notify_safe(
                event_type="H4_PIPELINE_ERROR",
                status="ERROR",
                payload={
                    "close_ts_utc": str(close_ts),
                    "error": str(e),
                    "dry_run": get_dry_run(),
                },
                force=True,
            )

            print("H4_PIPELINE_ERROR:", e, flush=True)

        await sleep_interruptible(H4_LOOP_IDLE_SECONDS)

async def main_async() -> None:
    global STOP_EVENT

    loop = asyncio.get_running_loop()
    STOP_EVENT = asyncio.Event()
    install_signal_handlers(loop)

    acquire_db_advisory_lock()
    acquire_process_lock()

    try:
        ensure_trading_schema()
        ensure_trade_management_columns()
        audit_log.ensure_audit_tables()

        print("=" * 120, flush=True)
        print("AUTOTRADE_SERVICE_STARTED", flush=True)
        print("ROOT:", ROOT, flush=True)
        print("PID:", os.getpid(), flush=True)
        print("DRY_RUN:", get_dry_run(), flush=True)
        print("PAIR_MODEL_NAME:", config.PAIR_MODEL_NAME, flush=True)
        print("GRID_NAME:", config.GRID_NAME, flush=True)
        print("THRS:", config.GATE2_THR, config.GATE4_THR, config.GATE5_1_THR, config.GATE5_3_THR, flush=True)
        print("DYNAMIC_BLACKLIST_SOURCE:", str(getattr(config, "DYNAMIC_BLACKLIST_SOURCE", "prod")), flush=True)
        print("H4_AFTER_CLOSE_DELAY_SECONDS:", H4_AFTER_CLOSE_DELAY_SECONDS, flush=True)
        print("SAFETY_LOOP_ENABLED:", SAFETY_LOOP_ENABLED, flush=True)
        print("SAFETY_LOOP_INTERVAL_SECONDS:", SAFETY_LOOP_INTERVAL_SECONDS, flush=True)
        print("TRADE_MANAGEMENT_POLL_ENABLED:", TRADE_MANAGEMENT_POLL_ENABLED, flush=True)
        print("TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS:", TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS, flush=True)
        print("TRADE_MANAGEMENT_POLL_LOG_INTERVAL_SECONDS:", TRADE_MANAGEMENT_POLL_LOG_INTERVAL_SECONDS, flush=True)
        print("EARLY_STOP_TIMER_ENABLED:", EARLY_STOP_TIMER_ENABLED, flush=True)
        print("EARLY_STOP_TIMER_IDLE_SECONDS:", EARLY_STOP_TIMER_IDLE_SECONDS, flush=True)
        print("EARLY_STOP_TIMER_MIN_SLEEP_SECONDS:", EARLY_STOP_TIMER_MIN_SLEEP_SECONDS, flush=True)
        print("EARLY_STOP_TIMER_MAX_SLEEP_SECONDS:", EARLY_STOP_TIMER_MAX_SLEEP_SECONDS, flush=True)
        print("WS_LIFECYCLE_ENABLED:", WS_LIFECYCLE_ENABLED, flush=True)
        print("WS_LISTENER_MODULE:", WS_LISTENER_MODULE, flush=True)
        print("WS_LISTENER_WATCHDOG_INTERVAL_SECONDS:", WS_LISTENER_WATCHDOG_INTERVAL_SECONDS, flush=True)
        print("COUNTDOWN_LOG_INTERVAL_SECONDS:", COUNTDOWN_LOG_INTERVAL_SECONDS, flush=True)
        print("PRE_CLOSE_NOTICE_1_SECONDS:", PRE_CLOSE_NOTICE_1_SECONDS, flush=True)
        print("PRE_CLOSE_NOTICE_2_SECONDS:", PRE_CLOSE_NOTICE_2_SECONDS, flush=True)

        write_audit_event(
            event_type="AUTOTRADE_SERVICE_STARTED",
            status="STARTED",
            message="Autotrade service started",
            payload={
                "pid": os.getpid(),
                "root": str(ROOT),
                "dry_run": get_dry_run(),
                "pair_model_name": config.PAIR_MODEL_NAME,
                "grid_name": config.GRID_NAME,
                "gate2_thr": config.GATE2_THR,
                "gate4_thr": config.GATE4_THR,
                "gate5_1_thr": config.GATE5_1_THR,
                "gate5_3_thr": config.GATE5_3_THR,
                "dynamic_blacklist_source": str(getattr(config, "DYNAMIC_BLACKLIST_SOURCE", "prod")),
                "mode": "h4_with_ws_lifecycle_watchdog",
                "trade_management_poll_enabled": TRADE_MANAGEMENT_POLL_ENABLED,
                "trade_management_poll_interval_seconds": TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS,
                "trade_management_poll_log_interval_seconds": TRADE_MANAGEMENT_POLL_LOG_INTERVAL_SECONDS,
                "early_stop_timer_enabled": EARLY_STOP_TIMER_ENABLED,
                "ws_lifecycle_enabled": WS_LIFECYCLE_ENABLED,
                "ws_listener_module": WS_LISTENER_MODULE,
            },
        )

        notify_safe(
            event_type="AUTOTRADE_SERVICE_STARTED",
            status="STARTED",
            payload={
                "pid": os.getpid(),
                "dry_run": get_dry_run(),
                "grid_name": config.GRID_NAME,
                "mode": "h4_with_ws_lifecycle_watchdog",
            },
            force=True,
        )

        await startup_catchup_once()

        tasks = []

        h4_task = asyncio.create_task(h4_close_loop())
        safety_task = asyncio.create_task(safety_loop())
        ws_listener_task = asyncio.create_task(ws_listener_watchdog_loop())

        tasks.append(h4_task)
        tasks.append(safety_task)
        tasks.append(ws_listener_task)

        if TRADE_MANAGEMENT_POLL_ENABLED and not WS_LIFECYCLE_ENABLED:
            trade_management_poll_task = asyncio.create_task(trade_management_poll_loop())
            tasks.append(trade_management_poll_task)
        else:
            print("TRADE_MANAGEMENT_POLL_NOT_STARTED", flush=True)
            print("reason:", "WS_LIFECYCLE_ENABLED" if WS_LIFECYCLE_ENABLED else "TRADE_MANAGEMENT_POLL_DISABLED", flush=True)

        if EARLY_STOP_TIMER_ENABLED and not WS_LIFECYCLE_ENABLED:
            early_stop_timer_task = asyncio.create_task(early_stop_timer_loop())
            tasks.append(early_stop_timer_task)
        else:
            print("EARLY_STOP_TIMER_NOT_STARTED", flush=True)
            print("reason:", "WS_LIFECYCLE_ENABLED" if WS_LIFECYCLE_ENABLED else "EARLY_STOP_TIMER_DISABLED", flush=True)

        await STOP_EVENT.wait()

        print("STOP_SIGNAL_RECEIVED", flush=True)

        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        stop_ws_listener_process(WS_LISTENER_PROCESS)

        write_audit_event(
            event_type="AUTOTRADE_SERVICE_STOPPED",
            status="STOPPED",
            message="Autotrade service stopped",
            payload={
                "pid": os.getpid(),
                "dry_run": get_dry_run(),
            },
        )

        notify_safe(
            event_type="AUTOTRADE_SERVICE_STOPPED",
            status="STOPPED",
            payload={
                "pid": os.getpid(),
                "dry_run": get_dry_run(),
            },
            force=True,
        )


    finally:

        stop_ws_listener_process(WS_LISTENER_PROCESS)

        release_process_lock()

        release_db_advisory_lock()


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("KEYBOARD_INTERRUPT", flush=True)
    except Exception as e:
        print("=" * 120, flush=True)
        print("AUTOTRADE_SERVICE_FATAL_ERROR", flush=True)
        print("error_type:", type(e).__name__, flush=True)
        print("error:", e, flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        release_process_lock()
        release_db_advisory_lock()


if __name__ == "__main__":
    main()
