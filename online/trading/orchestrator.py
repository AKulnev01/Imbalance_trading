from __future__ import annotations

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


def run_step(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError("pipeline step not found: {}".format(script_path))

    started = time.time()

    print("=" * 120)
    print("RUN_STEP:", script_path)
    print("STARTED_AT_UTC:", pd.Timestamp.now(tz="UTC"))

    env = os.environ.copy()

    root_txt = str(config.ROOT)
    old_pythonpath = env.get("PYTHONPATH", "").strip()
    if old_pythonpath:
        env["PYTHONPATH"] = root_txt + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = root_txt

    cmd = [sys.executable, str(script_path)]
    proc = subprocess.run(
        cmd,
        cwd=str(config.ROOT),
        env=env,
    )

    elapsed = time.time() - started

    print("FINISHED_STEP:", script_path)
    print("ELAPSED_SEC:", round(elapsed, 3))
    print("RETURN_CODE:", proc.returncode)

    if proc.returncode != 0:
        raise RuntimeError("pipeline step failed: {} return_code={}".format(script_path, proc.returncode))

def run_module(module_name: str) -> None:
    started = time.time()

    print("=" * 120)
    print("RUN_MODULE:", module_name)
    print("STARTED_AT_UTC:", pd.Timestamp.now(tz="UTC"))

    env = os.environ.copy()

    root_txt = str(config.ROOT)
    old_pythonpath = env.get("PYTHONPATH", "").strip()
    if old_pythonpath:
        env["PYTHONPATH"] = root_txt + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = root_txt

    cmd = [sys.executable, "-m", module_name]
    proc = subprocess.run(
        cmd,
        cwd=str(config.ROOT),
        env=env,
    )

    elapsed = time.time() - started

    print("FINISHED_MODULE:", module_name)
    print("ELAPSED_SEC:", round(elapsed, 3))
    print("RETURN_CODE:", proc.returncode)

    if proc.returncode != 0:
        raise RuntimeError("module failed: {} return_code={}".format(module_name, proc.returncode))

def run_online_pipeline() -> None:
    for step in config.ONLINE_PIPELINE_STEPS:
        run_step(step)


def main() -> None:
    owner = "orchestrator:{}:{}".format(config.GRID_NAME, pd.Timestamp.now(tz="UTC"))

    ensure_trading_schema()

    if not acquire_lock(LOCK_NAME, owner=owner, ttl_seconds=60 * 30):
        print("LOCK_BUSY:", LOCK_NAME)
        return

    try:
        print("ROOT:", config.ROOT)
        print("PAIR_MODEL_NAME:", config.PAIR_MODEL_NAME)
        print("GRID_NAME:", config.GRID_NAME)
        print("CONFIG:", config.get_prod_config())
        print("=" * 120)

        run_online_pipeline()

        print("=" * 120)
        print("RUN_CLEANUP_UNCLOSED_H4_BEFORE_SELECTOR")
        cleanup_unclosed_h4_rows()

        print("=" * 120)
        print("RUN_RECONCILE")
        run_module("online.trading.reconcile")

        print("=" * 120)
        print("RUN_MONITOR")
        run_module("online.trading.monitor")

        active = get_active_position()
        if active is not None:
            print("ACTIVE_POSITION_EXISTS_AFTER_MONITOR: skip selector/execution")
            print(active)
            return

        if not can_open_new_position():
            print("ACTIVE_POSITION_AFTER_RECONCILE_MONITOR: skip selection")
            print(get_active_position())
            return

        signal = select_best_signal()
        if signal is None:
            log_no_signal_for_latest_h4(source="orchestrator")
            print("NO_SIGNAL_SELECTED")
            print("NO_SIGNAL_FOR_LATEST_H4_WRITTEN_TO_AUDIT")
            return

        save_selected_signal(signal)

        print("=" * 120)
        print("SELECTED_SIGNAL_SAVED")
        print("signal_key:", signal["signal_key"])
        print("symbol:", signal["symbol"])
        print("side:", signal["side"])
        print("signal_ts:", signal["signal_ts"])
        print("entry_ts_plan:", signal["entry_ts_plan"])
        print("gate2:", signal.get("gate2_for_side_proba"))
        print("gate4:", signal.get("gate4_confidence"))
        print("gate5_1:", signal.get("gate5_1_proba"))
        print("gate5_3:", signal.get("gate5_3_proba"))

        print("=" * 120)
        print("RUN_EXECUTION")
        run_module("online.trading.execution")

    finally:
        release_lock(LOCK_NAME, owner=owner)


if __name__ == "__main__":
    main()
