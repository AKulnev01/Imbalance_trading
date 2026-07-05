from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from online.trading import config
from online.trading.cleanup_unclosed_h4_rows import cleanup_unclosed_h4_rows
from online.trading.db_schema import ensure_trading_schema
from online.trading.locks import acquire_lock, release_lock
from online.trading.selector import log_no_signal_for_latest_h4, save_selected_signal, select_best_signal
from online.trading.state import can_open_new_position, get_active_position


LOCK_NAME = "h4_autotrade_orchestrator"
QUIET_CHILD_LOGS = str(os.environ.get("IMB_VERBOSE_CHILD_LOGS", "0")).strip().lower() not in {"1", "true", "yes", "y"}

def run_step(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError("pipeline step not found: {}".format(script_path))

    started = time.time()

    env = os.environ.copy()
    env.setdefault("IMB_VERBOSE_SYMBOL_LOGS", "0")

    root_txt = str(config.ROOT)
    old_pythonpath = env.get("PYTHONPATH", "").strip()
    if old_pythonpath:
        env["PYTHONPATH"] = root_txt + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = root_txt

    step_name = str(script_path.relative_to(config.ROOT)) if str(script_path).startswith(str(config.ROOT)) else str(script_path)
    cmd = [sys.executable, str(script_path)]

    if QUIET_CHILD_LOGS:
        proc = subprocess.run(
            cmd,
            cwd=str(config.ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(config.ROOT),
            env=env,
        )

    elapsed = time.time() - started

    if proc.returncode == 0:
        print("STEP_OK | {} | elapsed_sec={}".format(step_name, round(elapsed, 3)))
        return

    print("=" * 120)
    print("STEP_FAILED:", step_name)
    print("ELAPSED_SEC:", round(elapsed, 3))
    print("RETURN_CODE:", proc.returncode)
    if QUIET_CHILD_LOGS:
        print("-" * 120)
        print((proc.stdout or "").rstrip())
    raise RuntimeError("pipeline step failed: {} return_code={}".format(script_path, proc.returncode))

def run_module(module_name: str) -> None:
    started = time.time()

    env = os.environ.copy()
    env.setdefault("IMB_VERBOSE_SYMBOL_LOGS", "0")

    root_txt = str(config.ROOT)
    old_pythonpath = env.get("PYTHONPATH", "").strip()
    if old_pythonpath:
        env["PYTHONPATH"] = root_txt + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = root_txt

    cmd = [sys.executable, "-m", module_name]

    if QUIET_CHILD_LOGS:
        proc = subprocess.run(
            cmd,
            cwd=str(config.ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(config.ROOT),
            env=env,
        )

    elapsed = time.time() - started

    if proc.returncode == 0:
        print("MODULE_OK | {} | elapsed_sec={}".format(module_name, round(elapsed, 3)))
        return

    print("=" * 120)
    print("MODULE_FAILED:", module_name)
    print("ELAPSED_SEC:", round(elapsed, 3))
    print("RETURN_CODE:", proc.returncode)
    if QUIET_CHILD_LOGS:
        print("-" * 120)
        print((proc.stdout or "").rstrip())
    raise RuntimeError("module failed: {} return_code={}".format(module_name, proc.returncode))


def run_online_pipeline() -> None:
    for step in config.ONLINE_PIPELINE_STEPS:
        run_step(step)


def main() -> None:
    orchestrator_started = time.time()
    owner = "orchestrator:{}:{}".format(config.GRID_NAME, pd.Timestamp.now(tz="UTC"))

    ensure_trading_schema()

    if not acquire_lock(LOCK_NAME, owner=owner, ttl_seconds=60 * 30):
        print("LOCK_BUSY:", LOCK_NAME)
        return

    try:
        prod_config = config.get_prod_config()

        print("=" * 120)
        print("ORCHESTRATOR_START")
        print("ROOT:", config.ROOT)
        print("PAIR_MODEL_NAME:", config.PAIR_MODEL_NAME)
        print("GRID_NAME:", config.GRID_NAME)
        print(
            "THRESHOLDS | gate2={} | gate4={} | gate5_1={} | gate5_3={}".format(
                prod_config.get("gate2_thr"),
                prod_config.get("gate4_thr"),
                prod_config.get("gate5_1_thr"),
                prod_config.get("gate5_3_thr"),
            )
        )
        print(
            "TRADE_CFG | slot_mode={} | capital_mode={} | entry_delay_seconds={}".format(
                prod_config.get("slot_mode"),
                prod_config.get("capital_mode"),
                prod_config.get("entry_delay_seconds"),
            )
        )
        print("=" * 120)

        run_online_pipeline()

        cleanup_started = time.time()

        if QUIET_CHILD_LOGS:
            cleanup_buf = io.StringIO()
            with contextlib.redirect_stdout(cleanup_buf):
                cleanup_unclosed_h4_rows()
        else:
            cleanup_unclosed_h4_rows()

        print("STEP_OK | cleanup_unclosed_h4_rows | elapsed_sec={}".format(round(time.time() - cleanup_started, 3)))

        active_before_reconcile = get_active_position()

        if active_before_reconcile is not None:
            print("=" * 120)
            print("RUN_RECONCILE_BEFORE_SELECTOR")
            run_module("online.trading.reconcile")

            print("=" * 120)
            print("RUN_MONITOR_BEFORE_SELECTOR")
            run_module("online.trading.monitor")

            print("=" * 120)
            print("RUN_CANCEL_STALE_PROTECTIVE_ORDERS_BEFORE_SELECTOR")
            run_module("online.trading.cancel_stale_protective_orders")

        entry_block_reason = None
        active_before_selector = get_active_position()

        if active_before_selector is not None:
            entry_block_reason = "ACTIVE_POSITION_EXISTS_BEFORE_SELECTOR"
        elif not can_open_new_position():
            entry_block_reason = "ACTIVE_POSITION_BLOCK_BEFORE_SELECTOR"

        signal = select_best_signal(entry_block_reason=entry_block_reason)

        if entry_block_reason is not None:
            print("SELECTOR_SNAPSHOT_WRITTEN_WITH_ENTRY_BLOCK:", entry_block_reason)
            print(active_before_selector)

        if signal is None and entry_block_reason is None:
            log_no_signal_for_latest_h4(source="orchestrator")
            print("NO_SIGNAL_SELECTED")
            print("NO_SIGNAL_FOR_LATEST_H4_WRITTEN_TO_AUDIT")

        active_after_monitor = get_active_position()

        if active_after_monitor is not None:
            print("ACTIVE_POSITION_EXISTS_AFTER_MONITOR: skip execution")
            print(active_after_monitor)
            return

        if not can_open_new_position():
            print("ACTIVE_POSITION_AFTER_RECONCILE_MONITOR: skip execution")
            print(get_active_position())
            return

        if signal is None:
            return

        save_selected_signal(signal)

        print(
            "SIGNAL_SELECTED | {} | {} | signal_ts={} | entry_ts_plan={} | gate2={} | gate4={} | gate5_1={} | gate5_3={}".format(
                signal["symbol"],
                signal["side"],
                signal["signal_ts"],
                signal["entry_ts_plan"],
                round(float(signal.get("gate2_for_side_proba") or 0.0), 6),
                round(float(signal.get("gate4_confidence") or 0.0), 6),
                round(float(signal.get("gate5_1_proba") or 0.0), 6),
                round(float(signal.get("gate5_3_proba") or 0.0), 6),
            )
        )

        print("=" * 120)
        print("RUN_EXECUTION")
        run_module("online.trading.execution")


    finally:
        release_lock(LOCK_NAME, owner=owner)
        print("ORCHESTRATOR_DONE | elapsed_sec={}".format(round(time.time() - orchestrator_started, 3)))


if __name__ == "__main__":
    main()
