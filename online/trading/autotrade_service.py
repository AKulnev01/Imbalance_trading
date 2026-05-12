from __future__ import annotations

import asyncio
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
from online.trading.db import read_sql


SERVICE_NAME = "gate_autotrade_service"

ROOT = config.ROOT
LOCK_PATH = ROOT / "online" / "_state_gate_autotrade_service.lock"

STARTUP_CATCHUP_PIPELINE = os.environ.get("IMB_STARTUP_CATCHUP_PIPELINE", "1").strip() != "0"
RUN_EXECUTION_ON_CATCHUP = os.environ.get("IMB_RUN_EXECUTION_ON_CATCHUP", "0").strip() == "1"

H4_AFTER_CLOSE_DELAY_SECONDS = int(os.environ.get("IMB_H4_AFTER_CLOSE_DELAY_SECONDS", "3"))
H4_LOOP_IDLE_SECONDS = int(os.environ.get("IMB_H4_LOOP_IDLE_SECONDS", "1"))

COUNTDOWN_LOG_INTERVAL_SECONDS = int(os.environ.get("IMB_COUNTDOWN_LOG_INTERVAL_SECONDS", "1200"))
PRE_CLOSE_NOTICE_1_SECONDS = int(os.environ.get("IMB_PRE_CLOSE_NOTICE_1_SECONDS", "60"))
PRE_CLOSE_NOTICE_2_SECONDS = int(os.environ.get("IMB_PRE_CLOSE_NOTICE_2_SECONDS", "10"))

STOP_EVENT: Optional[asyncio.Event] = None

DB_LOCK_CONN = None
AUTOTRADE_ADVISORY_LOCK_KEY = 9102026051001


def acquire_db_advisory_lock() -> None:
    global DB_LOCK_CONN

    if DB_LOCK_CONN is not None:
        return

    try:
        import psycopg2
    except Exception as e:
        raise RuntimeError("psycopg2 import failed for autotrade advisory lock: {}".format(e))

    dsn = str(getattr(config, "DB_DSN", "") or os.environ.get("IMB_DB_DSN", "")).strip()
    if not dsn:
        raise RuntimeError("DB_DSN is empty; cannot acquire autotrade advisory lock")

    conn = psycopg2.connect(dsn)
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (AUTOTRADE_ADVISORY_LOCK_KEY,))
            locked = bool(cur.fetchone()[0])
    except Exception:
        conn.close()
        raise

    if not locked:
        conn.close()
        raise RuntimeError(
            "another autotrade_service already holds PostgreSQL advisory lock: {}".format(
                AUTOTRADE_ADVISORY_LOCK_KEY
            )
        )

    DB_LOCK_CONN = conn

    print("=" * 120, flush=True)
    print("DB_ADVISORY_LOCK_ACQUIRED", flush=True)
    print("lock_key:", AUTOTRADE_ADVISORY_LOCK_KEY, flush=True)


def release_db_advisory_lock() -> None:
    global DB_LOCK_CONN

    conn = DB_LOCK_CONN
    DB_LOCK_CONN = None

    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (AUTOTRADE_ADVISORY_LOCK_KEY,))
    except Exception as e:
        print("WARNING: failed to release db advisory lock:", e, flush=True)

    try:
        conn.close()
    except Exception:
        pass

ADVISORY_LOCK_CONN = None
ADVISORY_LOCK_KEY = "imbalance_searcher_autotrade_service_global_lock"


def get_db_dsn() -> str:
    return os.environ.get(
        "IMB_DB_DSN",
        config.DB_DSN,
    )


def acquire_global_db_lock() -> None:
    global ADVISORY_LOCK_CONN

    if ADVISORY_LOCK_CONN is not None:
        return

    dsn = get_db_dsn()
    conn = psycopg2.connect(dsn)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtext(%s))",
            (ADVISORY_LOCK_KEY,),
        )
        locked = bool(cur.fetchone()[0])

    if not locked:
        conn.close()
        raise RuntimeError(
            "autotrade service already running: PostgreSQL advisory lock is busy: {}".format(
                ADVISORY_LOCK_KEY
            )
        )

    ADVISORY_LOCK_CONN = conn

    print("=" * 120, flush=True)
    print("GLOBAL_DB_LOCK_ACQUIRED", flush=True)
    print("lock_key:", ADVISORY_LOCK_KEY, flush=True)
    print("db_dsn:", dsn, flush=True)


def release_global_db_lock() -> None:
    global ADVISORY_LOCK_CONN

    conn = ADVISORY_LOCK_CONN
    ADVISORY_LOCK_CONN = None

    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock(hashtext(%s))",
                (ADVISORY_LOCK_KEY,),
            )
    except Exception as e:
        print("WARNING: failed to release PostgreSQL advisory lock:", e, flush=True)

    try:
        conn.close()
    except Exception:
        pass

DB_LOCK_CONN = None
DB_LOCK_KEY_1 = 918273645
DB_LOCK_KEY_2 = 20260510


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

            run_module("online.trading.selector", allow_fail=False)

            catchup_close_ts = latest_closed_h4_open_utc() + pd.Timedelta(hours=4)
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
                "mode": "h4_only_no_background_monitor_reconcile",
            },
        )

        notify_safe(
            event_type="AUTOTRADE_SERVICE_STARTED",
            status="STARTED",
            payload={
                "pid": os.getpid(),
                "dry_run": get_dry_run(),
                "grid_name": config.GRID_NAME,
                "mode": "h4_only",
            },
            force=True,
        )

        await startup_catchup_once()

        task = asyncio.create_task(h4_close_loop())

        await STOP_EVENT.wait()

        print("STOP_SIGNAL_RECEIVED", flush=True)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

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
        release_db_advisory_lock()
        release_db_advisory_lock()


if __name__ == "__main__":
    main()
