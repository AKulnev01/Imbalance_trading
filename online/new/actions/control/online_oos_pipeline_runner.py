from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from online.trading import config
from online.trading.db_schema import ensure_trading_schema
from online.trading.locks import acquire_lock, release_lock


LOCK_NAME = "online_oos_pipeline_runner"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run online feature/prediction pipeline for OOS/backtest preparation "
            "without selector, trading_signals selection, or execution."
        )
    )

    p.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols, for example ADAUSDT,BTCUSDT.",
    )
    p.add_argument(
        "--start",
        required=True,
        help="OOS start UTC, for example 2026-01-01 00:00.",
    )
    p.add_argument(
        "--end",
        required=True,
        help="OOS end UTC, for example 2026-04-01 00:00.",
    )
    p.add_argument(
        "--run-tag",
        default="",
        help="Optional run tag for logs/manifests.",
    )
    p.add_argument(
        "--runs-root",
        default=str(config.ROOT / "production" / "artifacts" / "online_oos_runs"),
        help="Root directory for run manifests and step logs.",
    )
    p.add_argument(
        "--timeout-sec",
        type=int,
        default=7200,
        help="Timeout per online pipeline step.",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually run pipeline steps. Without this flag only dry-run manifest is written.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after failed step and record failure in manifest.",
    )
    p.add_argument(
        "--json-out",
        default="",
        help="Optional JSON summary output path.",
    )

    return p.parse_args()


def utc_text(value: Any, name: str) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(ts):
        raise RuntimeError("bad {}: {}".format(name, value))

    return pd.Timestamp(ts).floor("min").strftime("%Y-%m-%d %H:%M")


def normalize_symbol(raw: str) -> str:
    text = str(raw or "").strip().upper()
    text = text.replace("/", "")
    text = text.replace("-", "")
    text = text.replace("_", "")

    if not text:
        raise RuntimeError("empty symbol in --symbols")

    if not text.endswith("USDT"):
        text = text + "USDT"

    return text


def parse_symbols(raw: str) -> List[str]:
    out: List[str] = []

    for part in str(raw or "").replace(";", ",").split(","):
        symbol = normalize_symbol(part)

        if symbol not in out:
            out.append(symbol)

    if not out:
        raise RuntimeError("no symbols parsed from --symbols")

    return out


def build_run_tag(symbols: List[str], start: str, end: str, explicit: str) -> str:
    if str(explicit or "").strip():
        return str(explicit).strip()

    symbol_part = "_".join(symbols[:4])

    if len(symbols) > 4:
        symbol_part += "_plus{}".format(len(symbols) - 4)

    safe_start = start.replace("-", "").replace(":", "").replace(" ", "_")
    safe_end = end.replace("-", "").replace(":", "").replace(" ", "_")

    return "{}__{}__{}".format(symbol_part, safe_start, safe_end)


def resolve_step_path(step: Any) -> Path:
    path = Path(str(step))

    if path.is_absolute():
        return path

    return config.ROOT / path


def make_child_env(symbols: List[str], start: str, end: str) -> Dict[str, str]:
    env = os.environ.copy()

    root_txt = str(config.ROOT)
    old_pythonpath = env.get("PYTHONPATH", "").strip()

    if old_pythonpath:
        env["PYTHONPATH"] = root_txt + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = root_txt

    env["IMB_PROJECT_ROOT"] = root_txt
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("IMB_VERBOSE_SYMBOL_LOGS", "0")

    # Safety: this runner prepares OOS online tables only.
    # It must never trigger real execution.
    env["IMB_TRADING_DRY_RUN"] = "1"
    env["IMB_ONLINE_OOS_MODE"] = "1"
    env["IMB_ONLINE_OOS_SYMBOLS"] = ",".join(symbols)
    env["IMB_ONLINE_OOS_START"] = start
    env["IMB_ONLINE_OOS_END"] = end
    env["IMB_ONLINE_DISABLE_SELECTOR"] = "1"
    env["IMB_ONLINE_DISABLE_EXECUTION"] = "1"

    return env


def run_step(
    step_path: Path,
    env: Dict[str, str],
    log_path: Path,
    timeout_sec: int,
) -> Dict[str, Any]:
    started = time.time()

    cmd = [
        sys.executable,
        "-u",
        str(step_path),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(config.ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(timeout_sec),
    )

    elapsed = round(time.time() - started, 3)
    out = str(proc.stdout or "")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(out, encoding="utf-8")

    return {
        "step": str(step_path),
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
        "ok": int(proc.returncode) == 0,
    }


def write_json(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    symbols = parse_symbols(args.symbols)
    start = utc_text(args.start, "start")
    end = utc_text(args.end, "end")

    if pd.to_datetime(start, utc=True) >= pd.to_datetime(end, utc=True):
        raise RuntimeError("start must be earlier than end")

    run_tag = build_run_tag(symbols=symbols, start=start, end=end, explicit=args.run_tag)
    runs_root = Path(args.runs_root)
    run_dir = runs_root / run_tag
    logs_dir = run_dir / "logs"
    manifest_path = run_dir / "manifest.json"

    steps_raw = list(getattr(config, "ONLINE_PIPELINE_STEPS", []))
    steps = [resolve_step_path(step) for step in steps_raw]

    # GATE4_OOS_STEP_SUBSTITUTION:
    # Keep live online/gate4/predict_online_gate4_no_raw_refs.py untouched.
    # OOS/add-symbol runner uses isolated Gate4 predict action.
    gate4_live_step = config.ROOT / "online" / "gate4" / "predict_online_gate4_no_raw_refs.py"
    gate4_oos_step = (
        config.ROOT
        / "online"
        / "new"
        / "actions"
        / "gate4"
        / "predict_online_gate4_no_raw_refs_oos_action.py"
    )

    steps = [
        gate4_oos_step if step == gate4_live_step else step
        for step in steps
    ]

    missing_steps = [str(step) for step in steps if not step.exists()]
    if missing_steps:
        raise RuntimeError("missing online pipeline steps: {}".format(missing_steps))

    owner = "online_oos_pipeline_runner:{}:{}:{}".format(
        ",".join(symbols),
        run_tag,
        pd.Timestamp.now(tz="UTC"),
    )

    ensure_trading_schema()

    if not acquire_lock(LOCK_NAME, owner=owner, ttl_seconds=60 * 60 * 3):
        raise RuntimeError("LOCK_BUSY: {}".format(LOCK_NAME))

    manifest: Dict[str, Any] = {
        "status": "STARTED",
        "execute": bool(args.execute),
        "dry_run": not bool(args.execute),
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "symbols": symbols,
        "start": start,
        "end": end,
        "pair_model_name": getattr(config, "PAIR_MODEL_NAME", ""),
        "grid_name": getattr(config, "GRID_NAME", ""),
        "step_count": len(steps),
        "steps": [],
        "errors": [],
    }

    try:
        print("=" * 120)
        print("ONLINE_OOS_PIPELINE_RUNNER_START")
        print("ROOT:", config.ROOT)
        print("RUN_TAG:", run_tag)
        print("SYMBOLS:", ",".join(symbols))
        print("START:", start)
        print("END:", end)
        print("EXECUTE:", bool(args.execute))
        print("PAIR_MODEL_NAME:", getattr(config, "PAIR_MODEL_NAME", ""))
        print("GRID_NAME:", getattr(config, "GRID_NAME", ""))
        print("=" * 120)

        env = make_child_env(symbols=symbols, start=start, end=end)

        effective_idx = 0

        for idx, step_path in enumerate(steps, start=1):
            step_rel = str(step_path.relative_to(config.ROOT)).replace("\\", "/")

            if step_rel == "online/sync_candles_h4.py":
                skipped = {
                    "idx": idx,
                    "step": step_rel,
                    "enabled": False,
                    "skipped": True,
                    "reason": (
                        "OOS runner does not sync candles. "
                        "Validation/OOS candles must be loaded by "
                        "online/new/actions/control/oos_validation_db_loader.py "
                        "so DB contains only the requested valid/OOS window."
                    ),
                }
                manifest["steps"].append(skipped)
                print("STEP_SKIP_OOS_DB_PRELOADED | {} | {}".format(idx, step_rel))
                continue

            effective_idx += 1
            log_path = logs_dir / "{:03d}_{}.log".format(
                effective_idx,
                step_path.stem,
            )

            step_record: Dict[str, Any] = {
                "index": idx,
                "step": step_rel,
                "path": str(step_path),
                "log_path": str(log_path),
                "status": "PENDING",
            }

            manifest["steps"].append(step_record)

            if not bool(args.execute):
                step_record["status"] = "DRY_RUN"
                print("STEP_DRY_RUN | {} | {}".format(idx, step_rel))
                continue

            print("STEP_START | {} | {}".format(idx, step_rel))

            try:
                result = run_step(
                    step_path=step_path,
                    env=env,
                    log_path=log_path,
                    timeout_sec=int(args.timeout_sec),
                )

                step_record.update(result)
                step_record["status"] = "OK" if bool(result.get("ok")) else "FAILED"

                if bool(result.get("ok")):
                    print(
                        "STEP_OK | {} | {} | elapsed_sec={}".format(
                            idx,
                            step_rel,
                            result.get("elapsed_sec"),
                        )
                    )
                else:
                    msg = "STEP_FAILED | {} | {} | rc={}".format(
                        idx,
                        step_rel,
                        result.get("returncode"),
                    )
                    print(msg)
                    manifest["errors"].append(msg)

                    if not bool(args.continue_on_error):
                        break

            except subprocess.TimeoutExpired as exc:
                msg = "STEP_TIMEOUT | {} | {} | timeout_sec={}".format(
                    idx,
                    step_rel,
                    int(args.timeout_sec),
                )

                step_record["status"] = "TIMEOUT"
                step_record["error"] = str(exc)
                manifest["errors"].append(msg)
                print(msg)

                if not bool(args.continue_on_error):
                    break

            except Exception as exc:
                msg = "STEP_ERROR | {} | {} | error={}".format(idx, step_rel, exc)
                step_record["status"] = "ERROR"
                step_record["error"] = str(exc)
                manifest["errors"].append(msg)
                print(msg)

                if not bool(args.continue_on_error):
                    break

        if manifest["errors"]:
            manifest["status"] = "FAILED"
        elif bool(args.execute):
            manifest["status"] = "OK"
        else:
            manifest["status"] = "DRY_RUN"

        write_json(manifest_path, manifest)

        if args.json_out:
            write_json(Path(args.json_out), manifest)

        print("=" * 120)
        print("ONLINE_OOS_PIPELINE_RUNNER_DONE")
        print("STATUS:", manifest["status"])
        print("RUN_DIR:", run_dir)
        print("MANIFEST:", manifest_path)
        print("=" * 120)

        if manifest["errors"]:
            raise RuntimeError("online oos pipeline failed: {}".format(manifest["errors"]))

    finally:
        release_lock(LOCK_NAME, owner=owner)


if __name__ == "__main__":
    main()
