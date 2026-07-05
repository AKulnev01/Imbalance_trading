from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_RUNS_ROOT = "production/artifacts/offline_runs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_step_ids(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []

    out: List[str] = []

    for value in values:
        parts = str(value).replace(";", ",").split(",")
        for part in parts:
            item = part.strip()
            if item:
                out.append(item)

    return out


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("plan json must be object")

    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=True, indent=2, default=str)
    path.write_text(text + "\n", encoding="utf-8")


def resolve_run_tag(plan: Dict[str, Any]) -> str:
    run_tag = str(plan.get("run_tag") or "").strip()
    if run_tag:
        return run_tag

    mode = str(plan.get("mode") or "offline").strip() or "offline"
    return "{}_{}".format(mode, datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))


def resolve_run_dir(plan: Dict[str, Any], run_dir_arg: str, runs_root_arg: str) -> Path:
    if str(run_dir_arg).strip():
        return Path(str(run_dir_arg))

    runs_root = Path(str(runs_root_arg).strip() or DEFAULT_RUNS_ROOT)
    return runs_root / resolve_run_tag(plan)


def acquire_file_lock(lock_path: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "pid": os.getpid(),
            "created_at_utc": utc_now_iso(),
        }, ensure_ascii=True, indent=2))
        f.write("\n")

    return True


def release_file_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def should_run_step(
    step: Dict[str, Any],
    only_steps: List[str],
    skip_steps: List[str],
    from_step: str,
    seen_from_step: bool,
) -> bool:
    step_id = str(step.get("id") or "")

    if not bool(step.get("enabled")):
        return False

    if only_steps and step_id not in only_steps:
        return False

    if skip_steps and step_id in skip_steps:
        return False

    if from_step and not seen_from_step:
        return False

    return True


def build_step_command(
    python_exe: str,
    step: Dict[str, Any],
) -> List[str]:
    script = str(step.get("script") or "").strip()

    if not script:
        raise ValueError("step script is empty for step id={}".format(step.get("id")))

    args_raw = step.get("args") or []
    if not isinstance(args_raw, list):
        raise ValueError("step args must be list for step id={}".format(step.get("id")))

    args = [str(x) for x in args_raw]

    return [str(python_exe), "-X", "utf8", script] + args


def compact_command(cmd: List[str]) -> str:
    return " ".join(str(x) for x in cmd)


def execute_step(
    step: Dict[str, Any],
    step_index: int,
    python_exe: str,
    cwd: Path,
    logs_dir: Path,
    base_env: Dict[str, str],
    execute: bool,
    timeout_sec: Optional[int],
) -> Dict[str, Any]:
    step_id = str(step.get("id") or "step_{}".format(step_index))
    safe_step_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in step_id)
    log_path = logs_dir / "{:03d}_{}.log".format(step_index, safe_step_id)

    started_at = utc_now_iso()

    result: Dict[str, Any] = {
        "step_index": int(step_index),
        "step_id": step_id,
        "title": str(step.get("title") or ""),
        "enabled": bool(step.get("enabled")),
        "script": str(step.get("script") or ""),
        "args": step.get("args") or [],
        "reads": step.get("reads") or [],
        "writes": step.get("writes") or [],
        "note": str(step.get("note") or ""),
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "duration_sec": None,
        "dry_run": not bool(execute),
        "returncode": None,
        "status": None,
        "log_path": str(log_path).replace("\\", "/"),
    }

    cmd = build_step_command(python_exe=python_exe, step=step)
    result["command_argv"] = cmd
    result["command_text"] = compact_command(cmd)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("STEP_ID: {}\n".format(step_id))
        log.write("TITLE: {}\n".format(result["title"]))
        log.write("STARTED_AT_UTC: {}\n".format(started_at))
        log.write("DRY_RUN: {}\n".format(not bool(execute)))
        log.write("CWD: {}\n".format(str(cwd)))
        log.write("COMMAND: {}\n".format(result["command_text"]))
        log.write("=" * 120 + "\n")

        if not execute:
            log.write("DRY_RUN_ONLY: command was not executed.\n")
            result["returncode"] = 0
            result["status"] = "DRY_RUN"
        else:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=base_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )

            log.write(completed.stdout or "")
            result["returncode"] = int(completed.returncode)
            result["status"] = "OK" if completed.returncode == 0 else "FAILED"

    finished_at = utc_now_iso()
    result["finished_at_utc"] = finished_at

    try:
        t0 = datetime.fromisoformat(started_at)
        t1 = datetime.fromisoformat(finished_at)
        result["duration_sec"] = float((t1 - t0).total_seconds())
    except Exception:
        result["duration_sec"] = None

    return result


def select_steps(
    plan: Dict[str, Any],
    only_steps: List[str],
    skip_steps: List[str],
    from_step: str,
) -> List[Dict[str, Any]]:
    raw_steps = plan.get("steps") or []

    if not isinstance(raw_steps, list):
        raise ValueError("plan.steps must be list")

    selected: List[Dict[str, Any]] = []
    seen_from_step = False if from_step else True

    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            raise ValueError("each plan step must be object")

        step_id = str(raw_step.get("id") or "")

        if from_step and step_id == from_step:
            seen_from_step = True

        if should_run_step(
            step=raw_step,
            only_steps=only_steps,
            skip_steps=skip_steps,
            from_step=from_step,
            seen_from_step=seen_from_step,
        ):
            selected.append(raw_step)

    if from_step and not seen_from_step:
        raise ValueError("--from-step was not found in plan: {}".format(from_step))

    return selected


def build_manifest_base(
    plan: Dict[str, Any],
    plan_json_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
    selected_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "status": "CREATED",
        "created_at_utc": utc_now_iso(),
        "started_at_utc": None,
        "finished_at_utc": None,
        "duration_sec": None,
        "plan_json_path": str(plan_json_path).replace("\\", "/"),
        "run_dir": str(run_dir).replace("\\", "/"),
        "mode": plan.get("mode"),
        "run_tag": plan.get("run_tag"),
        "symbols": plan.get("symbols"),
        "train_end": plan.get("train_end"),
        "oos_start": plan.get("oos_start"),
        "oos_end": plan.get("oos_end"),
        "db_load_oos": plan.get("db_load_oos"),
        "execute": bool(args.execute),
        "dry_run": not bool(args.execute),
        "stop_on_error": not bool(args.continue_on_error),
        "continue_on_error": bool(args.continue_on_error),
        "timeout_sec": args.timeout_sec,
        "only_steps": normalize_step_ids(args.only_step),
        "skip_steps": normalize_step_ids(args.skip_step),
        "from_step": str(args.from_step or ""),
        "selected_step_count": len(selected_steps),
        "selected_step_ids": [str(step.get("id") or "") for step in selected_steps],
        "steps": [],
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute enabled steps from offline pipeline plan JSON with logs and manifest."
    )

    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute commands. Without this flag executor only writes dry-run manifest/logs.",
    )

    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=0)

    parser.add_argument("--only-step", action="append", default=[])
    parser.add_argument("--skip-step", action="append", default=[])
    parser.add_argument("--from-step", default="")

    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--cwd", default=".")

    args = parser.parse_args()

    plan_json_path = Path(str(args.plan_json))
    plan = load_json(plan_json_path)

    run_dir = resolve_run_dir(
        plan=plan,
        run_dir_arg=str(args.run_dir),
        runs_root_arg=str(args.runs_root),
    )

    logs_dir = run_dir / "logs"
    manifest_path = run_dir / "manifest.json"
    plan_copy_path = run_dir / "plan.copy.json"
    lock_path = run_dir / ".executor.lock"

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not acquire_file_lock(lock_path):
        raise RuntimeError("EXECUTOR_LOCK_BUSY: {}".format(lock_path))

    manifest: Dict[str, Any]

    try:
        selected_steps = select_steps(
            plan=plan,
            only_steps=normalize_step_ids(args.only_step),
            skip_steps=normalize_step_ids(args.skip_step),
            from_step=str(args.from_step or "").strip(),
        )

        write_json(plan_copy_path, plan)

        manifest = build_manifest_base(
            plan=plan,
            plan_json_path=plan_json_path,
            run_dir=run_dir,
            args=args,
            selected_steps=selected_steps,
        )

        manifest["status"] = "RUNNING"
        manifest["started_at_utc"] = utc_now_iso()
        write_json(manifest_path, manifest)

        cwd = Path(str(args.cwd))
        timeout_sec: Optional[int] = int(args.timeout_sec) if int(args.timeout_sec or 0) > 0 else None

        base_env = os.environ.copy()
        project_root = Path.cwd()
        base_env["PYTHONPATH"] = str(project_root)
        base_env["PYTHONUNBUFFERED"] = "1"
        base_env["PYTHONIOENCODING"] = "utf-8"
        base_env["PYTHONUTF8"] = "1"

        failed = False

        for idx, step in enumerate(selected_steps, start=1):
            step_result = execute_step(
                step=step,
                step_index=idx,
                python_exe=str(args.python_exe),
                cwd=cwd,
                logs_dir=logs_dir,
                base_env=base_env,
                execute=bool(args.execute),
                timeout_sec=timeout_sec,
            )

            manifest["steps"].append(step_result)
            write_json(manifest_path, manifest)

            if int(step_result.get("returncode") or 0) != 0:
                failed = True
                manifest["status"] = "FAILED"
                manifest["error"] = {
                    "step_id": step_result.get("step_id"),
                    "returncode": step_result.get("returncode"),
                    "log_path": step_result.get("log_path"),
                }
                write_json(manifest_path, manifest)

                if not bool(args.continue_on_error):
                    break

        finished_at = utc_now_iso()
        manifest["finished_at_utc"] = finished_at

        try:
            started_at = str(manifest.get("started_at_utc") or "")
            t0 = datetime.fromisoformat(started_at)
            t1 = datetime.fromisoformat(finished_at)
            manifest["duration_sec"] = float((t1 - t0).total_seconds())
        except Exception:
            manifest["duration_sec"] = None

        if manifest.get("status") != "FAILED":
            manifest["status"] = "FAILED" if failed else ("EXECUTED" if bool(args.execute) else "DRY_RUN")

        write_json(manifest_path, manifest)

        print(json.dumps({
            "status": manifest["status"],
            "run_dir": str(run_dir).replace("\\", "/"),
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "selected_step_count": len(selected_steps),
            "selected_step_ids": manifest["selected_step_ids"],
            "execute": bool(args.execute),
            "dry_run": not bool(args.execute),
            "error": manifest.get("error"),
        }, ensure_ascii=True, indent=2))

        if manifest["status"] == "FAILED":
            raise SystemExit(1)

    except Exception as exc:
        error_text = "{}: {}".format(type(exc).__name__, str(exc))
        tb = traceback.format_exc()

        manifest = {
            "status": "FAILED",
            "created_at_utc": utc_now_iso(),
            "finished_at_utc": utc_now_iso(),
            "plan_json_path": str(plan_json_path).replace("\\", "/"),
            "run_dir": str(run_dir).replace("\\", "/"),
            "error": {
                "message": error_text,
                "traceback": tb,
            },
        }

        write_json(manifest_path, manifest)

        print(json.dumps({
            "status": "FAILED",
            "run_dir": str(run_dir).replace("\\", "/"),
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "error": error_text,
        }, ensure_ascii=True, indent=2))

        raise SystemExit(1)

    finally:
        release_file_lock(lock_path)


if __name__ == "__main__":
    main()
