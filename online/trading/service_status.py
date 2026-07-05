from __future__ import annotations

import ast
import json
import os
import re
import signal
import shutil
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def bootstrap_env_before_config_import() -> None:
    root = Path(os.environ.get("IMB_PROJECT_ROOT", r"C:\Projects\ImbalanceSearcher"))
    env_file = root / ".env"

    if not env_file.exists():
        return

    for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            os.environ[key] = value


bootstrap_env_before_config_import()

from online.trading import config
from online.trading.bybit_client import BybitClient
from online.trading.db import db_cursor, read_sql


ROOT = config.ROOT
ENV_FILE = ROOT / ".env"

LOCK_PATH = ROOT / "online" / "_state_gate_autotrade_service.lock"
PID_PATH = ROOT / "gate_autotrade.pid"
LOG_LINK = ROOT / "logs" / "gate_autotrade_latest.log"

HOST_MAC = "mac"
HOST_WIN = "win"

SERVICE_MODULE_DEFAULT = "online.trading.autotrade_service"
CONTROL_MODULE_DEFAULT = "online.trading.service_status"
BACKTEST_MODULE_DEFAULT = "online.trading.backtest_m1_thresholds"


def load_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}

    if not path.exists():
        return out

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()

        if not line:
            continue
        if line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            out[key] = value

    return out


LOCAL_ENV = load_env_file(ENV_FILE)

for env_key, env_value in LOCAL_ENV.items():
    os.environ[env_key] = env_value


LOCAL_HOST = os.environ.get("IMB_LOCAL_HOST", HOST_MAC).strip().lower()

WIN_SSH_HOST = (
    os.environ.get("IMB_WIN_SSH_HOST", "")
    or os.environ.get("SSH", "")
).strip()

WIN_PROJECT_ROOT = Path(
    os.environ.get("IMB_WIN_PROJECT_ROOT", r"C:\Projects\ImbalanceSearcher")
)

WIN_PYTHON = Path(
    os.environ.get(
        "IMB_WIN_PYTHON",
        str(WIN_PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
    )
)

WINPY_CMD = os.environ.get("IMB_WINPY_CMD", "winpy").strip()

SERVICE_MODULE = os.environ.get(
    "IMB_TRADING_SERVICE_MODULE",
    SERVICE_MODULE_DEFAULT,
).strip()

CONTROL_MODULE = os.environ.get(
    "IMB_TRADING_CONTROL_MODULE",
    CONTROL_MODULE_DEFAULT,
).strip()
BACKTEST_MODULE = os.environ.get(
    "IMB_TRADING_BACKTEST_MODULE",
    BACKTEST_MODULE_DEFAULT,
).strip()

STOP_TIMEOUT_SECONDS = int(os.environ.get("IMB_TRADING_STOP_TIMEOUT_SECONDS", "20"))
START_WAIT_SECONDS = float(os.environ.get("IMB_TRADING_START_WAIT_SECONDS", "8"))


def get_service_python_executable() -> str:
    configured = os.environ.get("IMB_SERVICE_PYTHON", "").strip()
    if configured:
        return configured

    if os.name == "nt":
        return str(ROOT / ".venv" / "Scripts" / "python.exe")

    return str(ROOT / ".venv" / "bin" / "python")


def assert_service_python_exists() -> None:
    py_path = Path(get_service_python_executable())
    if not py_path.exists():
        raise RuntimeError("service python not found: {}".format(py_path))


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def fmt_money(value: Any) -> str:
    try:
        return "{:.6f}".format(float(value))
    except Exception:
        return "0.000000"


def fmt_float(value: Any) -> str:
    try:
        return "{:.8f}".format(float(value))
    except Exception:
        return "0.00000000"


def fmt_left(seconds: float) -> str:
    seconds_i = int(max(0, seconds))
    h = seconds_i // 3600
    m = (seconds_i % 3600) // 60
    s = seconds_i % 60
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)


def next_h4_close_utc(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    ts = pd.to_datetime(now if now is not None else utc_now(), utc=True)
    h = (int(ts.hour) // 4) * 4
    current_h4_open = ts.replace(hour=h, minute=0, second=0, microsecond=0)
    close_ts = current_h4_open + pd.Timedelta(hours=4)

    if close_ts <= ts:
        close_ts = close_ts + pd.Timedelta(hours=4)

    return close_ts


def latest_closed_h4_signal_ts_utc(now_ts: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC") if now_ts is None else pd.to_datetime(now_ts, utc=True)
    h = (int(now.hour) // 4) * 4
    current_h4_open = now.replace(hour=h, minute=0, second=0, microsecond=0)
    return current_h4_open - pd.Timedelta(hours=4)

def floor_to_h4_utc(ts: Any) -> pd.Timestamp:
    value = pd.to_datetime(ts, utc=True, errors="coerce")

    if pd.isna(value):
        raise RuntimeError("bad timestamp for floor_to_h4_utc: {}".format(ts))

    value = pd.Timestamp(value)
    h = (int(value.hour) // 4) * 4
    return value.replace(hour=h, minute=0, second=0, microsecond=0)


def build_expected_history_signal_ts(hours: int) -> List[pd.Timestamp]:
    hours_i = int(hours)

    if hours_i <= 0:
        raise RuntimeError("history hours must be > 0")

    now_ts = pd.Timestamp.now(tz="UTC")
    latest_signal_ts = latest_closed_h4_signal_ts_utc(now_ts)
    min_ts = floor_to_h4_utc(now_ts - pd.Timedelta(hours=hours_i))

    out: List[pd.Timestamp] = []
    cur = latest_signal_ts

    while cur >= min_ts:
        out.append(pd.Timestamp(cur))
        cur = cur - pd.Timedelta(hours=4)

    return out


def seconds_until(ts: pd.Timestamp) -> float:
    return max(0.0, (pd.to_datetime(ts, utc=True) - utc_now()).total_seconds())


def read_lock_payload() -> Dict[str, Any]:
    if not LOCK_PATH.exists():
        return {}

    try:
        return ast.literal_eval(LOCK_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return {}


def is_process_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False

    try:
        pid_i = int(pid)
    except Exception:
        return False

    if pid_i <= 0:
        return False

    if os.name == "nt":
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "if (Get-Process -Id {} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}".format(pid_i),
            ]
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return int(p.returncode) == 0
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


def get_pid_from_lock() -> Optional[int]:
    payload = read_lock_payload()
    pid = payload.get("pid")

    if pid is None:
        return None

    try:
        pid_i = int(pid)
    except Exception:
        return None

    if is_process_alive(pid_i):
        return pid_i

    return None


def get_pid_from_pid_file() -> Optional[int]:
    if not PID_PATH.exists():
        return None

    try:
        pid_i = int(PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None

    if is_process_alive(pid_i):
        return pid_i

    return None


def get_service_pid() -> Optional[int]:
    pid = get_pid_from_lock()
    if pid is not None:
        return pid

    pid = get_pid_from_pid_file()
    if pid is not None:
        return pid

    try:
        processes = find_local_autotrade_processes()
    except NameError:
        processes = []

    if len(processes) == 1:
        try:
            return int(processes[0]["pid"])
        except Exception:
            return None

    if len(processes) > 1:
        lines = []
        for p in processes:
            lines.append("pid={} cmd={}".format(p.get("pid"), p.get("cmdline")))

        raise RuntimeError(
            "multiple autotrade processes found on this host:\n{}".format(
                "\n".join(lines)
            )
        )

    return None


def cleanup_stale_runtime_files() -> None:
    lock_pid = None
    payload = read_lock_payload()

    if payload:
        try:
            lock_pid = int(payload.get("pid"))
        except Exception:
            lock_pid = None

    if LOCK_PATH.exists() and not is_process_alive(lock_pid):
        try:
            LOCK_PATH.unlink()
            print("DELETE_STALE_LOCK:", LOCK_PATH)
        except Exception as e:
            print("WARNING_DELETE_STALE_LOCK_FAILED:", e)

    pid_file_pid = None
    if PID_PATH.exists():
        try:
            pid_file_pid = int(PID_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            pid_file_pid = None

    if PID_PATH.exists() and not is_process_alive(pid_file_pid):
        try:
            PID_PATH.unlink()
            print("DELETE_STALE_PID:", PID_PATH)
        except Exception as e:
            print("WARNING_DELETE_STALE_PID_FAILED:", e)


def get_local_runtime_state() -> Dict[str, Any]:
    cleanup_stale_runtime_files()

    pid = get_service_pid()
    running = is_process_alive(pid)
    payload = read_lock_payload()

    return {
        "host": LOCAL_HOST,
        "running": bool(running),
        "pid": pid,
        "lock_payload": payload,
    }


def print_local_runtime_state(prefix: str = "LOCAL_STATUS") -> None:
    state = get_local_runtime_state()

    print("-" * 120)
    print(prefix)
    print("host:", state["host"])
    print("service_running:", state["running"])
    print("pid:", state["pid"])


def run_command_capture(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    input_text: Optional[str] = None,
) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return int(p.returncode), str(p.stdout or "")



def find_raw_local_autotrade_processes() -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    if os.name == "nt":
        code, out = run_command_capture(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { "
                    "$_.CommandLine -like '*python*' -and "
                    "$_.CommandLine -like '*-m online.trading.autotrade_service*' "
                    "} | "
                    "ForEach-Object { "
                    "Write-Output ("
                    "$_.ProcessId.ToString() + '|' + "
                    "$_.ParentProcessId.ToString() + '|' + "
                    "$_.ExecutablePath + '|' + "
                    "$_.CommandLine"
                    ") "
                    "}"
                ),
            ]
        )

        if code != 0:
            return result

        for raw in out.splitlines():
            line = raw.strip()
            parts = line.split("|", 3)

            if len(parts) != 4:
                continue

            pid_txt, ppid_txt, exe_path, cmdline = parts
            cmdline_l = cmdline.lower()

            if "powershell" in cmdline_l:
                continue
            if "get-ciminstance" in cmdline_l:
                continue
            if "service_status" in cmdline_l:
                continue
            if "-m online.trading.autotrade_service" not in cmdline_l:
                continue

            try:
                pid = int(pid_txt)
                ppid = int(ppid_txt)
            except Exception:
                continue

            if pid == os.getpid():
                continue

            result.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "exe_path": exe_path,
                    "cmdline": cmdline,
                }
            )

        return result

    code, out = run_command_capture(["ps", "aux"])

    if code != 0:
        return result

    for line in out.splitlines():
        line_l = line.lower()

        if "-m online.trading.autotrade_service" not in line_l:
            continue
        if "grep" in line_l:
            continue
        if "service_status" in line_l:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            pid = int(parts[1])
        except Exception:
            continue

        if pid == os.getpid():
            continue

        result.append(
            {
                "pid": pid,
                "ppid": None,
                "exe_path": None,
                "cmdline": line,
            }
        )

    return result


def find_local_autotrade_processes() -> List[Dict[str, Any]]:
    raw_processes = find_raw_local_autotrade_processes()

    if os.name != "nt":
        return raw_processes

    result: List[Dict[str, Any]] = []

    for proc in raw_processes:
        try:
            pid = int(proc.get("pid"))
        except Exception:
            continue

        exe_l = str(proc.get("exe_path") or "").lower()

        has_autotrade_child = False
        for child in raw_processes:
            try:
                child_ppid = int(child.get("ppid"))
            except Exception:
                continue

            if child_ppid == pid:
                has_autotrade_child = True
                break

        if "\\.venv\\scripts\\python.exe" in exe_l and has_autotrade_child:
            continue

        result.append(proc)

    return result


def assert_no_local_autotrade_processes() -> None:
    processes = find_local_autotrade_processes()

    if not processes:
        return

    lines = []
    for p in processes:
        lines.append("pid={} cmd={}".format(p.get("pid"), p.get("cmdline")))

    raise RuntimeError(
        "autotrade process already exists on this host:\n{}".format(
            "\n".join(lines)
        )
    )



def build_child_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(load_env_file(ENV_FILE))

    env["IMB_PROJECT_ROOT"] = str(ROOT)
    env["IMB_DB_DSN"] = str(getattr(config, "DB_DSN"))
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONWARNINGS"] = "ignore:pandas only supports SQLAlchemy connectable:UserWarning"
    env["IMB_LOCAL_HOST"] = LOCAL_HOST
    env["IMB_TRADING_CONTROL_MODULE"] = CONTROL_MODULE
    env["IMB_TRADING_SERVICE_MODULE"] = SERVICE_MODULE
    env["IMB_TRADING_BACKTEST_MODULE"] = BACKTEST_MODULE

    return env


def build_win_python_inline(args: List[str]) -> str:
    argv = ", ".join(repr(str(x)) for x in args)

    code = """
import os
import subprocess
from pathlib import Path

ROOT = Path(r"{root}")
PYTHON = Path(r"{python}")
ENV_FILE = ROOT / ".env"

def load_env_file(path):
    out = {{}}

    if not path.exists():
        return out

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()

        if not line:
            continue
        if line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            out[key] = value

    return out

env = os.environ.copy()
env.update(load_env_file(ENV_FILE))

env["IMB_PROJECT_ROOT"] = str(ROOT)
env["IMB_DB_DSN"] = {db_dsn!r}
env["PYTHONPATH"] = str(ROOT)
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONWARNINGS"] = "ignore:pandas only supports SQLAlchemy connectable:UserWarning"
env["IMB_LOCAL_HOST"] = "win"
env["IMB_SERVICE_PYTHON"] = str(PYTHON)
env["IMB_TRADING_CONTROL_MODULE"] = "{control_module}"
env["IMB_TRADING_SERVICE_MODULE"] = "{service_module}"
env["IMB_TRADING_BACKTEST_MODULE"] = "{backtest_module}"
env["IMB_TRADING_DRY_RUN"] = os.environ.get("IMB_TRADING_DRY_RUN", env.get("IMB_TRADING_DRY_RUN", "0"))

subprocess.check_call(
    [str(PYTHON), "-u", "-m", "{control_module}", {argv}],
    cwd=str(ROOT),
    env=env,
)
""".format(
        root=str(WIN_PROJECT_ROOT),
        python=str(WIN_PYTHON),
        db_dsn=str(getattr(config, "DB_DSN")),
        control_module=CONTROL_MODULE,
        service_module=SERVICE_MODULE,
        backtest_module=BACKTEST_MODULE,
        argv=argv,
    )

    return code


def run_win_control(args: List[str]) -> Tuple[int, str]:
    code = build_win_python_inline(args)

    if WIN_SSH_HOST:
        cmd = [
            "ssh",
            WIN_SSH_HOST,
            '"' + str(WIN_PYTHON) + '"',
            "-u",
            "-",
        ]
        return run_command_capture(cmd, input_text=code)

    if shutil.which(WINPY_CMD):
        return run_command_capture([WINPY_CMD], input_text=code)

    return (
        2,
        "Windows runner is not configured. Put SSH=USER@HOST into .env or create executable winpy command.",
    )


def print_remote_block(title: str, returncode: int, output: str) -> None:
    print("-" * 120)
    print(title)
    print("returncode:", returncode)

    if output.strip():
        print(output.rstrip())
    else:
        print("EMPTY_OUTPUT")


def is_win_running_by_status_output(output: str) -> bool:
    for line in str(output).splitlines():
        line = line.strip()

        if line.startswith("service_running:"):
            return line.split(":", 1)[1].strip().lower() == "true"

    return False


def get_global_running_state() -> Dict[str, Any]:
    local_state = get_local_runtime_state()

    win_code, win_out = run_win_control(["status-local"])
    win_running = bool(win_code == 0 and is_win_running_by_status_output(win_out))

    running_hosts = []

    if bool(local_state["running"]):
        running_hosts.append(LOCAL_HOST)

    if win_running:
        running_hosts.append(HOST_WIN)

    return {
        "local": local_state,
        "win_returncode": win_code,
        "win_output": win_out,
        "win_running": win_running,
        "running_hosts": running_hosts,
    }


def assert_no_global_service_running() -> None:
    state = get_global_running_state()
    running_hosts = list(state["running_hosts"])

    if running_hosts:
        raise RuntimeError(
            "autotrade service already running on: {}. Stop it first.".format(
                ",".join(running_hosts)
            )
        )


def wait_for_local_lock(timeout_seconds: float) -> Optional[int]:
    deadline = time.time() + float(timeout_seconds)

    while time.time() < deadline:
        pid = get_pid_from_lock()

        if pid is not None:
            return pid

        time.sleep(0.25)

    return None


def start_local_service_posix(env: Dict[str, str]) -> int:
    LOG_LINK.parent.mkdir(parents=True, exist_ok=True)

    log_f = open(str(LOG_LINK), "wb")

    p = subprocess.Popen(
        [get_service_python_executable(), "-u", "-m", SERVICE_MODULE],
        cwd=str(ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )

    log_f.close()

    PID_PATH.write_text(str(int(p.pid)), encoding="utf-8")
    return int(p.pid)


def start_local_service_windows(env: Dict[str, str]) -> int:
    LOG_LINK.parent.mkdir(parents=True, exist_ok=True)

    log_f = open(str(LOG_LINK), "ab")

    creationflags = 0

    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP

    if hasattr(subprocess, "DETACHED_PROCESS"):
        creationflags |= subprocess.DETACHED_PROCESS

    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW

    # Важно для запуска через Windows OpenSSH:
    # без CREATE_BREAKAWAY_FROM_JOB процесс может быть убит после закрытия SSH-сессии.
    creationflags |= 0x01000000

    p = subprocess.Popen(
        [get_service_python_executable(), "-u", "-m", SERVICE_MODULE],
        cwd=str(ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=False,
        creationflags=creationflags,
    )

    log_f.close()

    PID_PATH.write_text(str(int(p.pid)), encoding="utf-8")
    return int(p.pid)

def start_local_service() -> None:
    assert_service_python_exists()
    cleanup_stale_runtime_files()
    assert_no_local_autotrade_processes()

    state = get_local_runtime_state()
    if bool(state["running"]):
        raise RuntimeError("local service already running, pid={}".format(state["pid"]))

    env = build_child_env()

    if os.name == "nt":
        starter_pid = start_local_service_windows(env)
    else:
        starter_pid = start_local_service_posix(env)

    service_pid = wait_for_local_lock(START_WAIT_SECONDS)

    if service_pid is None:
        service_pid = get_service_pid()

    if service_pid is not None:
        PID_PATH.write_text(str(int(service_pid)), encoding="utf-8")

    print("STARTED_LOCAL")
    print("host:", LOCAL_HOST)
    print("starter_pid:", starter_pid)
    print("service_pid:", service_pid)
    print("module:", SERVICE_MODULE)
    print("service_python:", get_service_python_executable())
    print("log:", LOG_LINK)

    if service_pid is None:
        print("WARNING: service lock was not created during start wait")



def terminate_stale_autotrade_db_lock_if_no_processes() -> None:
    processes = find_raw_local_autotrade_processes()

    if processes:
        print("SKIP_DB_LOCK_TERMINATE_AUTOTRADE_PROCESSES_EXIST")
        for proc in processes:
            print("pid={} cmd={}".format(proc.get("pid"), proc.get("cmdline")))
        return

    try:
        from online.trading.db import db_cursor, read_sql
    except Exception as e:
        print("WARNING_DB_LOCK_TERMINATE_IMPORT_FAILED:", e)
        return

    try:
        holders = read_sql(
            """
            SELECT
                a.pid AS pg_pid,
                a.state,
                a.backend_start,
                a.query
            FROM pg_locks l
            JOIN pg_stat_activity a
                ON a.pid = l.pid
            WHERE l.locktype = 'advisory'
              AND l.classid = 918273645
              AND l.objid = 20260510
              AND l.granted = TRUE
            ORDER BY a.backend_start
            """
        )
    except Exception as e:
        print("WARNING_DB_LOCK_HOLDER_CHECK_FAILED:", e)
        return

    if holders.empty:
        print("DB_ADVISORY_LOCK_NOT_HELD")
        return

    print("STALE_DB_ADVISORY_LOCK_HOLDERS")
    print(holders.to_string(index=False))

    try:
        with db_cursor(commit=True) as (_, cur):
            cur.execute(
                """
                SELECT pg_terminate_backend(a.pid)
                FROM pg_locks l
                JOIN pg_stat_activity a
                    ON a.pid = l.pid
                WHERE l.locktype = 'advisory'
                  AND l.classid = 918273645
                  AND l.objid = 20260510
                  AND l.granted = TRUE
                  AND a.pid <> pg_backend_pid()
                """
            )
            rows = cur.fetchall()

        print("STALE_DB_ADVISORY_LOCK_TERMINATED:", rows)

    except Exception as e:
        print("WARNING_DB_LOCK_TERMINATE_FAILED:", e)


def cleanup_autotrade_runtime_files_force() -> None:
    for path in [PID_PATH, LOCK_PATH]:
        try:
            if path.exists():
                path.unlink()
                print("DELETE_RUNTIME_FILE:", path)
        except Exception as e:
            print("WARNING_DELETE_RUNTIME_FILE_FAILED:", path, e)


def stop_local_service() -> None:
    cleanup_stale_runtime_files()

    target_pids: List[int] = []

    for proc in find_raw_local_autotrade_processes():
        try:
            proc_pid = int(proc.get("pid"))
        except Exception:
            continue

        if proc_pid not in target_pids:
            target_pids.append(proc_pid)

    try:
        pid_from_state = get_service_pid()
    except RuntimeError:
        pid_from_state = None

    if pid_from_state is not None and int(pid_from_state) not in target_pids:
        target_pids.append(int(pid_from_state))

    if not target_pids:
        print("LOCAL_NOT_RUNNING")
        terminate_stale_autotrade_db_lock_if_no_processes()
        cleanup_autotrade_runtime_files_force()
        return

    print("STOP_LOCAL_REQUEST")
    print("pids:", ",".join(str(x) for x in target_pids))

    for pid in target_pids:
        if not is_process_alive(pid):
            continue

        if os.name == "nt":
            code, out = run_command_capture(
                [
                    "taskkill",
                    "/PID",
                    str(int(pid)),
                    "/T",
                    "/F",
                ]
            )

            if code != 0 and is_process_alive(pid):
                raise RuntimeError("failed to stop windows pid={}: {}".format(pid, out))
        else:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as e:
                raise RuntimeError("failed to stop local pid={}: {}".format(pid, e))

    deadline = time.time() + float(STOP_TIMEOUT_SECONDS)

    while time.time() < deadline:
        leftovers = find_raw_local_autotrade_processes()
        if not leftovers:
            break
        time.sleep(0.25)

    leftovers = find_raw_local_autotrade_processes()

    if leftovers:
        lines = []
        for proc in leftovers:
            lines.append("pid={} cmd={}".format(proc.get("pid"), proc.get("cmdline")))

        raise RuntimeError(
            "autotrade process still exists after stop:\n{}".format(
                "\n".join(lines)
            )
        )

    terminate_stale_autotrade_db_lock_if_no_processes()
    cleanup_autotrade_runtime_files_force()

    print("STOPPED_LOCAL")
    print("pids:", ",".join(str(x) for x in target_pids))


def print_status_local() -> None:
    print_status()


def print_status_global() -> None:
    local_state = get_local_runtime_state()

    win_code, win_out = run_win_control(["status-local"])
    win_running = bool(win_code == 0 and is_win_running_by_status_output(win_out))

    running_hosts = []

    if bool(local_state["running"]):
        running_hosts.append(LOCAL_HOST)

    if win_running:
        running_hosts.append(HOST_WIN)

    print("-" * 120)
    print("STATUS")

    if len(running_hosts) == 0:
        print("active_host: none")
    elif len(running_hosts) == 1:
        print("active_host:", running_hosts[0])
    else:
        print("active_host: both")
        print("WARNING: duplicate autotrade service is running")

    print("mac_running:", bool(local_state["running"]) if LOCAL_HOST == HOST_MAC else False)
    print("mac_pid:", local_state["pid"] if LOCAL_HOST == HOST_MAC else None)
    print("win_running:", win_running)

    if win_code != 0:
        print("win_status: unavailable")


def start_host(host: str) -> None:
    host = str(host).strip().lower()

    if host not in {HOST_MAC, HOST_WIN}:
        raise RuntimeError("start requires host: mac or win")

    if host == HOST_MAC:
        if LOCAL_HOST != HOST_MAC:
            raise RuntimeError("this launcher is not running on mac; LOCAL_HOST={}".format(LOCAL_HOST))

        local_state = get_local_runtime_state()
        if bool(local_state["running"]):
            raise RuntimeError("local service already running, pid={}".format(local_state["pid"]))

        start_local_service()
        return

    assert_no_global_service_running()

    win_code, win_out = run_win_control(["start-local"])
    print_remote_block("WIN_START", win_code, win_out)

    if win_code != 0:
        raise RuntimeError("win start failed")


def stop_host(host: Optional[str]) -> None:
    host_norm = str(host or "").strip().lower()

    if host_norm == "":
        local_state = get_local_runtime_state()

        win_code, win_out = run_win_control(["status-local"])
        win_running = bool(win_code == 0 and is_win_running_by_status_output(win_out))

        if bool(local_state["running"]):
            stop_local_service()

        if win_running:
            code, out = run_win_control(["stop-local"])
            print_remote_block("WIN_STOP", code, out)

            if code != 0:
                raise RuntimeError("win stop failed")

        if not bool(local_state["running"]) and not win_running:
            print("NOTHING_RUNNING")

        return

    if host_norm == HOST_MAC:
        if LOCAL_HOST != HOST_MAC:
            raise RuntimeError("this launcher is not running on mac; LOCAL_HOST={}".format(LOCAL_HOST))
        stop_local_service()
        return

    if host_norm == HOST_WIN:
        code, out = run_win_control(["stop-local"])
        print_remote_block("WIN_STOP", code, out)

        if code != 0:
            raise RuntimeError("win stop failed")

        return

    raise RuntimeError("stop host must be empty, mac or win")

def resolve_status_trade_capital_usdt(available_usdt: Any) -> float:
    try:
        available = float(available_usdt)
    except Exception:
        available = 0.0

    if available <= 0:
        return 0.0

    chulan_enabled = bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0))

    if not chulan_enabled:
        return available

    base_capital = float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0)

    if base_capital <= 0:
        return available

    return min(available, base_capital)


def resolve_status_trading_leverage() -> float:
    leverage = float(getattr(config, "TRADING_LEVERAGE", 1.0) or 1.0)

    if leverage <= 0.0:
        return 1.0

    return float(leverage)


def resolve_status_position_notional_multiplier() -> float:
    multiplier = float(getattr(config, "POSITION_NOTIONAL_MULTIPLIER", 1.0) or 1.0)

    if multiplier <= 0.0:
        return 1.0

    return float(multiplier)


def calc_status_margin_plan(trade_capital_usdt: Any) -> Dict[str, Any]:
    trade_capital = float(trade_capital_usdt or 0.0)
    leverage = resolve_status_trading_leverage()
    multiplier = resolve_status_position_notional_multiplier()

    position_notional = trade_capital * multiplier

    estimated_margin = None
    if leverage > 0.0:
        estimated_margin = position_notional / leverage

    margin_buffer = None
    margin_ok = False

    if estimated_margin is not None:
        margin_buffer = trade_capital - estimated_margin
        margin_ok = estimated_margin <= trade_capital + 1e-9

    return {
        "trading_leverage": leverage,
        "position_notional_multiplier": multiplier,
        "position_notional_usdt_plan": position_notional,
        "estimated_initial_margin_usdt": estimated_margin,
        "margin_buffer_usdt": margin_buffer,
        "margin_ok": margin_ok,
    }


def ensure_status_position_columns() -> None:
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
            ADD COLUMN IF NOT EXISTS trade_management_mode TEXT,
            ADD COLUMN IF NOT EXISTS partial_tp_handled_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS early_stop_replaced_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS protective_cleanup_done_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS closed_cleanup_done_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ws_lifecycle_updated_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ws_lifecycle_last_event_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ws_lifecycle_last_error TEXT;

        ALTER TABLE public.trading_orders
            ADD COLUMN IF NOT EXISTS lifecycle_note TEXT,
            ADD COLUMN IF NOT EXISTS ws_last_event_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ws_last_event_uid TEXT;
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)


def tcp_probe_host(host: str, port: int = 443, timeout_seconds: float = 2.0) -> Dict[str, Any]:
    started = time.time()
    result: Dict[str, Any] = {
        "host": str(host),
        "port": int(port),
        "ok": False,
        "ip": "",
        "elapsed_ms": 0.0,
        "error": "",
    }

    try:
        ip = socket.gethostbyname(str(host))
        result["ip"] = ip

        with socket.create_connection(
            (str(host), int(port)),
            timeout=float(timeout_seconds),
        ):
            pass

        result["ok"] = True

    except Exception as e:
        result["error"] = str(e)

    result["elapsed_ms"] = round((time.time() - started) * 1000.0, 1)
    return result


def find_raw_local_ws_listener_processes() -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    if os.name == "nt":
        code, out = run_command_capture(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { "
                    "$_.CommandLine -like '*python*' -and "
                    "$_.CommandLine -like '*-m online.trading.WSListener*' "
                    "} | "
                    "ForEach-Object { "
                    "Write-Output ("
                    "$_.ProcessId.ToString() + '|' + "
                    "$_.ParentProcessId.ToString() + '|' + "
                    "$_.ExecutablePath + '|' + "
                    "$_.CommandLine"
                    ") "
                    "}"
                ),
            ]
        )

        if code != 0:
            return result

        for raw in out.splitlines():
            line = raw.strip()
            parts = line.split("|", 3)

            if len(parts) != 4:
                continue

            pid_txt, ppid_txt, exe_path, cmdline = parts
            cmdline_l = cmdline.lower()

            if "powershell" in cmdline_l:
                continue
            if "get-ciminstance" in cmdline_l:
                continue
            if "service_status" in cmdline_l:
                continue
            if "-m online.trading.wslistener" not in cmdline_l:
                continue

            try:
                pid = int(pid_txt)
                ppid = int(ppid_txt)
            except Exception:
                continue

            if pid == os.getpid():
                continue

            result.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "exe_path": exe_path,
                    "cmdline": cmdline,
                }
            )

        return result

    code, out = run_command_capture(["ps", "aux"])

    if code != 0:
        return result

    for line in out.splitlines():
        line_l = line.lower()

        if "-m online.trading.wslistener" not in line_l:
            continue
        if "grep" in line_l:
            continue
        if "service_status" in line_l:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            pid = int(parts[1])
        except Exception:
            continue

        if pid == os.getpid():
            continue

        result.append(
            {
                "pid": pid,
                "ppid": None,
                "exe_path": None,
                "cmdline": line,
            }
        )

    return result


def find_local_ws_listener_processes() -> List[Dict[str, Any]]:
    raw_processes = find_raw_local_ws_listener_processes()

    if os.name != "nt":
        return raw_processes

    result: List[Dict[str, Any]] = []

    for proc in raw_processes:
        try:
            pid = int(proc.get("pid"))
        except Exception:
            continue

        exe_l = str(proc.get("exe_path") or "").lower()

        has_ws_child = False
        for child in raw_processes:
            try:
                child_ppid = int(child.get("ppid"))
            except Exception:
                continue

            if child_ppid == pid:
                has_ws_child = True
                break

        if "\\.venv\\scripts\\python.exe" in exe_l and has_ws_child:
            continue

        result.append(proc)

    return result


def get_ws_health_status() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ws_process_running": False,
        "ws_process_count": 0,
        "ws_raw_process_count": 0,
        "ws_pid": None,
        "ws_last_heartbeat_utc": "",
        "ws_last_event_utc": "",
        "ws_events_24h": 0,
        "ws_errors_tail": 0,
        "api_bybit_tcp_ok": False,
        "api_bybit_tcp_ms": 0.0,
        "api_bybit_tcp_ip": "",
        "api_bybit_tcp_error": "",
        "stream_bybit_tcp_ok": False,
        "stream_bybit_tcp_ms": 0.0,
        "stream_bybit_tcp_ip": "",
        "stream_bybit_tcp_error": "",
        "closed_position_bad_protective_orders": 0,
        "error": "",
    }

    try:
        raw_processes = find_raw_local_ws_listener_processes()
        processes = find_local_ws_listener_processes()

        result["ws_raw_process_count"] = int(len(raw_processes))
        result["ws_process_count"] = int(len(processes))
        result["ws_process_running"] = bool(len(processes) > 0)

        if processes:
            result["ws_pid"] = int(processes[0].get("pid"))

    except Exception as e:
        result["error"] = "ws_process_check_error: {}".format(e)

    ws_log = ROOT / "logs" / "ws_listener_latest.log"

    try:
        if ws_log.exists():
            lines = ws_log.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-1000:]

            for line in reversed(tail):
                if "WS_LISTENER_HEARTBEAT:" in line:
                    result["ws_last_heartbeat_utc"] = line.split("WS_LISTENER_HEARTBEAT:", 1)[1].strip()
                    break

            error_markers = [
                "WS_LISTENER_ERROR",
                "WS_LISTENER_SOCKET_CLOSED",
                "WS_LISTENER_RECONNECT_REASON",
                "ping/pong timed out",
                "Connection to remote host was lost",
                "WebSocketConnectionClosedException",
            ]
            result["ws_errors_tail"] = int(
                sum(
                    1
                    for line in tail
                    if any(marker.lower() in line.lower() for marker in error_markers)
                )
            )

    except Exception as e:
        old = str(result.get("error") or "")
        result["error"] = (old + "; " if old else "") + "ws_log_check_error: {}".format(e)

    try:
        ws_df = read_sql(
            """
            SELECT
                MAX(created_at) AS last_event_utc,
                COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours') AS events_24h
            FROM public.trading_ws_events
            """
        )

        if not ws_df.empty:
            last_event = ws_df.iloc[0].get("last_event_utc")
            if last_event is not None and not pd.isna(last_event):
                result["ws_last_event_utc"] = str(last_event)

            try:
                result["ws_events_24h"] = int(ws_df.iloc[0].get("events_24h") or 0)
            except Exception:
                result["ws_events_24h"] = 0

    except Exception as e:
        old = str(result.get("error") or "")
        result["error"] = (old + "; " if old else "") + "ws_db_event_check_error: {}".format(e)

    try:
        bad_df = read_sql(
            """
            SELECT COUNT(*) AS n
            FROM public.trading_positions p
            JOIN public.trading_orders o
                ON o.trade_id = p.trade_id
            WHERE p.status LIKE 'POSITION_CLOSED%%'
              AND UPPER(o.order_role) IN (
                  'TAKE_PROFIT',
                  'PARTIAL_TP',
                  'FINAL_TP',
                  'STOP_LOSS',
                  'EARLY_STOP',
                  'REST_STOP_AFTER_PARTIAL'
              )
              AND o.status NOT IN (
                  'CANCELLED',
                  'CANCELLED_NOT_FOUND',
                  'FILLED',
                  'TRIGGERED',
                  'FAILED',
                  'ERROR',
                  'TP_SL_FAILED',
                  'TTL_CLOSE_FAILED'
              )
            """
        )

        if not bad_df.empty:
            result["closed_position_bad_protective_orders"] = int(bad_df.iloc[0].get("n") or 0)

    except Exception as e:
        old = str(result.get("error") or "")
        result["error"] = (old + "; " if old else "") + "bad_protective_check_error: {}".format(e)

    api_probe = tcp_probe_host("api.bybit.com", 443, 2.0)
    stream_probe = tcp_probe_host("stream.bybit.com", 443, 2.0)

    result["api_bybit_tcp_ok"] = bool(api_probe.get("ok"))
    result["api_bybit_tcp_ms"] = float(api_probe.get("elapsed_ms") or 0.0)
    result["api_bybit_tcp_ip"] = str(api_probe.get("ip") or "")
    result["api_bybit_tcp_error"] = str(api_probe.get("error") or "")

    result["stream_bybit_tcp_ok"] = bool(stream_probe.get("ok"))
    result["stream_bybit_tcp_ms"] = float(stream_probe.get("elapsed_ms") or 0.0)
    result["stream_bybit_tcp_ip"] = str(stream_probe.get("ip") or "")
    result["stream_bybit_tcp_error"] = str(stream_probe.get("error") or "")

    return result


def print_ws_health_status() -> None:
    ws = get_ws_health_status()

    print("ws_process_running:", bool(ws.get("ws_process_running")))
    print("ws_process_count:", int(ws.get("ws_process_count") or 0))
    print("ws_raw_process_count:", int(ws.get("ws_raw_process_count") or 0))
    print("ws_pid:", ws.get("ws_pid"))
    print("ws_last_heartbeat_utc:", ws.get("ws_last_heartbeat_utc") or "")
    print("ws_last_event_utc:", ws.get("ws_last_event_utc") or "")
    print("ws_events_24h:", int(ws.get("ws_events_24h") or 0))
    print("ws_errors_tail:", int(ws.get("ws_errors_tail") or 0))
    print("api_bybit_tcp_ok:", bool(ws.get("api_bybit_tcp_ok")))
    print("api_bybit_tcp_ms:", fmt_float(ws.get("api_bybit_tcp_ms")))
    print("api_bybit_tcp_ip:", ws.get("api_bybit_tcp_ip") or "")
    print("api_bybit_tcp_error:", ws.get("api_bybit_tcp_error") or "")
    print("stream_bybit_tcp_ok:", bool(ws.get("stream_bybit_tcp_ok")))
    print("stream_bybit_tcp_ms:", fmt_float(ws.get("stream_bybit_tcp_ms")))
    print("stream_bybit_tcp_ip:", ws.get("stream_bybit_tcp_ip") or "")
    print("stream_bybit_tcp_error:", ws.get("stream_bybit_tcp_error") or "")
    print("closed_position_bad_protective_orders:", int(ws.get("closed_position_bad_protective_orders") or 0))

    if ws.get("error"):
        print("ws_health_error:", ws.get("error"))

def get_exchange_status() -> Dict[str, Any]:
    result = {
        "bybit_ok": False,
        "usdt_balance": None,
        "open_positions": [],
        "unrealised_pnl_sum": 0.0,
        "error": None,
    }

    try:
        client = BybitClient()
        result["usdt_balance"] = client.get_wallet_balance_usdt()
        positions = client.get_open_positions()
        result["open_positions"] = positions

        pnl_sum = 0.0
        for p in positions:
            try:
                pnl_sum += float(p.get("unrealisedPnl") or 0.0)
            except Exception:
                pass

        result["unrealised_pnl_sum"] = pnl_sum
        result["bybit_ok"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        return result



WS_LOG_PATH = ROOT / "logs" / "ws_listener_latest.log"


def find_local_ws_listener_processes() -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    if os.name == "nt":
        code, out = run_command_capture(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { "
                    "$_.CommandLine -like '*python*' -and "
                    "$_.CommandLine -like '*-m online.trading.WSListener*' "
                    "} | "
                    "ForEach-Object { "
                    "Write-Output ("
                    "$_.ProcessId.ToString() + '|' + "
                    "$_.ParentProcessId.ToString() + '|' + "
                    "$_.ExecutablePath + '|' + "
                    "$_.CommandLine"
                    ") "
                    "}"
                ),
            ]
        )

        if code != 0:
            return result

        raw_processes = []

        for raw in out.splitlines():
            line = raw.strip()
            parts = line.split("|", 3)

            if len(parts) != 4:
                continue

            pid_txt, ppid_txt, exe_path, cmdline = parts
            cmdline_l = cmdline.lower()

            if "powershell" in cmdline_l:
                continue
            if "get-ciminstance" in cmdline_l:
                continue
            if "service_status" in cmdline_l:
                continue
            if "-m online.trading.wslistener" not in cmdline_l:
                continue

            try:
                pid = int(pid_txt)
                ppid = int(ppid_txt)
            except Exception:
                continue

            raw_processes.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "exe_path": exe_path,
                    "cmdline": cmdline,
                }
            )

        for proc in raw_processes:
            try:
                pid = int(proc.get("pid"))
            except Exception:
                continue

            exe_l = str(proc.get("exe_path") or "").lower()

            has_ws_child = False
            for child in raw_processes:
                try:
                    child_ppid = int(child.get("ppid"))
                except Exception:
                    continue

                if child_ppid == pid:
                    has_ws_child = True
                    break

            if "\\.venv\\scripts\\python.exe" in exe_l and has_ws_child:
                continue

            result.append(proc)

        return result

    code, out = run_command_capture(["ps", "aux"])

    if code != 0:
        return result

    for line in out.splitlines():
        line_l = line.lower()

        if "-m online.trading.wslistener" not in line_l:
            continue
        if "grep" in line_l:
            continue
        if "service_status" in line_l:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            pid = int(parts[1])
        except Exception:
            continue

        result.append(
            {
                "pid": pid,
                "ppid": None,
                "exe_path": None,
                "cmdline": line,
            }
        )

    return result


def probe_tcp_tls(host: str, port: int = 443, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    started = time.time()

    result: Dict[str, Any] = {
        "host": str(host),
        "port": int(port),
        "ok": False,
        "ip": None,
        "elapsed_ms": None,
        "error": None,
    }

    tls_sock = None

    try:
        infos = socket.getaddrinfo(str(host), int(port), type=socket.SOCK_STREAM)

        if not infos:
            raise RuntimeError("DNS returned no addresses")

        last_error = None

        for family, socktype, proto, _, sockaddr in infos:
            raw_sock = None

            try:
                ip = str(sockaddr[0])
                raw_sock = socket.socket(family, socktype, proto)
                raw_sock.settimeout(float(timeout_seconds))
                raw_sock.connect(sockaddr)

                context = ssl.create_default_context()
                tls_sock = context.wrap_socket(raw_sock, server_hostname=str(host))

                result["ok"] = True
                result["ip"] = ip
                result["elapsed_ms"] = round((time.time() - started) * 1000.0, 2)
                return result

            except Exception as e:
                last_error = e
                try:
                    if raw_sock is not None:
                        raw_sock.close()
                except Exception:
                    pass
                continue

        raise RuntimeError(str(last_error))

    except Exception as e:
        result["error"] = str(e)
        result["elapsed_ms"] = round((time.time() - started) * 1000.0, 2)
        return result

    finally:
        try:
            if tls_sock is not None:
                tls_sock.close()
        except Exception:
            pass


def tail_lines(path: Path, max_lines: int = 2000) -> List[str]:
    if not path.exists():
        return []

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    return lines[-int(max_lines):]


def parse_last_ws_heartbeat(lines: List[str]) -> Optional[str]:
    marker = "WS_LISTENER_HEARTBEAT:"

    for line in reversed(lines):
        if marker in line:
            return line.split(marker, 1)[1].strip()

    return None


def load_ws_status_health() -> Dict[str, Any]:
    processes = find_local_ws_listener_processes()
    lines = tail_lines(WS_LOG_PATH, max_lines=2000)

    last_event_utc = None
    ws_events_24h = 0
    sql_error = None

    try:
        df = read_sql(
            """
            SELECT
                MAX(created_at) AS last_event_utc,
                COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours') AS events_24h
            FROM public.trading_ws_events
            """
        )

        if not df.empty:
            last_event_value = df.iloc[0].get("last_event_utc")

            if pd.notna(pd.to_datetime(last_event_value, utc=True, errors="coerce")):
                last_event_utc = str(pd.to_datetime(last_event_value, utc=True, errors="coerce"))

            try:
                ws_events_24h = int(df.iloc[0].get("events_24h") or 0)
            except Exception:
                ws_events_24h = 0

    except Exception as e:
        sql_error = str(e)

    return {
        "running": len(processes) > 0,
        "process_count": len(processes),
        "pid": processes[0].get("pid") if processes else None,
        "last_heartbeat_utc": parse_last_ws_heartbeat(lines),
        "last_event_utc": last_event_utc,
        "events_24h": ws_events_24h,
        "recent_ping_pong_errors": sum(1 for x in lines if "ping/pong timed out" in x.lower()),
        "recent_connection_lost_errors": sum(1 for x in lines if "connection to remote host was lost" in x.lower()),
        "recent_socket_closed_errors": sum(1 for x in lines if "websocketconnectionclosedexception" in x.lower()),
        "recent_lifecycle_errors": sum(1 for x in lines if "ws_lifecycle_timer_error" in x.lower() or "ws_lifecycle_event_error" in x.lower()),
        "recent_reconnect_reason_count": sum(1 for x in lines if "ws_listener_reconnect_reason" in x.lower()),
        "recent_ws_error_count": sum(1 for x in lines if "ws_listener_error" in x.lower()),
        "sql_error": sql_error,
        "log": str(WS_LOG_PATH),
    }


def load_cleanup_health() -> Dict[str, Any]:
    result = {
        "closed_position_bad_protective_orders": None,
        "closed_positions_without_cleanup_mark_but_no_active_orders": None,
        "error": None,
    }

    try:
        df_bad = read_sql(
            """
            SELECT COUNT(*) AS n
            FROM public.trading_positions p
            JOIN public.trading_orders o
                ON o.trade_id = p.trade_id
            WHERE p.status LIKE 'POSITION_CLOSED%%'
              AND UPPER(o.order_role) IN (
                  'TAKE_PROFIT',
                  'PARTIAL_TP',
                  'FINAL_TP',
                  'STOP_LOSS',
                  'EARLY_STOP',
                  'REST_STOP_AFTER_PARTIAL'
              )
              AND UPPER(COALESCE(o.status, '')) NOT IN (
                  'CANCELLED',
                  'CANCELLED_NOT_FOUND',
                  'FILLED',
                  'TRIGGERED',
                  'FAILED',
                  'ERROR',
                  'TP_SL_FAILED',
                  'TTL_CLOSE_FAILED'
              )
            """
        )

        result["closed_position_bad_protective_orders"] = int(df_bad.iloc[0].get("n") or 0) if not df_bad.empty else 0

        df_unmarked = read_sql(
            """
            SELECT COUNT(*) AS n
            FROM public.trading_positions p
            WHERE p.status LIKE 'POSITION_CLOSED%%'
              AND p.protective_cleanup_done_at IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM public.trading_orders o
                  WHERE o.trade_id = p.trade_id
                    AND UPPER(o.order_role) IN (
                        'TAKE_PROFIT',
                        'PARTIAL_TP',
                        'FINAL_TP',
                        'STOP_LOSS',
                        'EARLY_STOP',
                        'REST_STOP_AFTER_PARTIAL'
                    )
                    AND UPPER(COALESCE(o.status, '')) NOT IN (
                        'CANCELLED',
                        'CANCELLED_NOT_FOUND',
                        'FILLED',
                        'TRIGGERED',
                        'FAILED',
                        'ERROR',
                        'TP_SL_FAILED',
                        'TTL_CLOSE_FAILED'
                    )
              )
            """
        )

        result["closed_positions_without_cleanup_mark_but_no_active_orders"] = int(df_unmarked.iloc[0].get("n") or 0) if not df_unmarked.empty else 0

    except Exception as e:
        result["error"] = str(e)

    return result


def print_status_health_block() -> None:
    ws = load_ws_status_health()
    cleanup = load_cleanup_health()
    api_probe = probe_tcp_tls("api.bybit.com", 443, timeout_seconds=5.0)
    stream_probe = probe_tcp_tls("stream.bybit.com", 443, timeout_seconds=5.0)

    print("ws_process_running:", bool(ws.get("running")))
    print("ws_process_count:", ws.get("process_count"))
    print("ws_pid:", ws.get("pid"))
    print("ws_last_heartbeat_utc:", ws.get("last_heartbeat_utc") or "-")
    print("ws_last_event_utc:", ws.get("last_event_utc") or "-")
    print("ws_events_24h:", ws.get("events_24h"))
    print("ws_recent_ping_pong_errors:", ws.get("recent_ping_pong_errors"))
    print("ws_recent_connection_lost_errors:", ws.get("recent_connection_lost_errors"))
    print("ws_recent_socket_closed_errors:", ws.get("recent_socket_closed_errors"))
    print("ws_recent_lifecycle_errors:", ws.get("recent_lifecycle_errors"))
    print("ws_recent_reconnect_reason_count:", ws.get("recent_reconnect_reason_count"))
    print("ws_recent_ws_error_count:", ws.get("recent_ws_error_count"))
    print("ws_log:", ws.get("log"))

    if ws.get("sql_error"):
        print("ws_sql_error:", ws.get("sql_error"))

    print("api_bybit_tcp_ok:", bool(api_probe.get("ok")))
    print("api_bybit_tcp_ip:", api_probe.get("ip") or "-")
    print("api_bybit_tcp_ms:", api_probe.get("elapsed_ms"))
    if api_probe.get("error"):
        print("api_bybit_tcp_error:", api_probe.get("error"))

    print("stream_bybit_tcp_ok:", bool(stream_probe.get("ok")))
    print("stream_bybit_tcp_ip:", stream_probe.get("ip") or "-")
    print("stream_bybit_tcp_ms:", stream_probe.get("elapsed_ms"))
    if stream_probe.get("error"):
        print("stream_bybit_tcp_error:", stream_probe.get("error"))

    print("closed_position_bad_protective_orders:", cleanup.get("closed_position_bad_protective_orders"))
    print("closed_positions_without_cleanup_mark_but_no_active_orders:", cleanup.get("closed_positions_without_cleanup_mark_but_no_active_orders"))

    if cleanup.get("error"):
        print("cleanup_health_error:", cleanup.get("error"))

def print_status() -> None:
    ensure_status_position_columns()
    close_ts = next_h4_close_utc()
    left = seconds_until(close_ts)

    state = get_local_runtime_state()
    pid = state["pid"]
    running = bool(state["running"])

    exchange = get_exchange_status()
    active_positions_checked = bool(exchange["bybit_ok"])
    open_positions_count = len(exchange["open_positions"])
    current_pnl = float(exchange["unrealised_pnl_sum"] or 0.0)

    capital_usdt = float(exchange["usdt_balance"] or 0.0)
    trade_capital_usdt = resolve_status_trade_capital_usdt(capital_usdt)
    chulan_enabled = bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0))
    chulan_base_capital_usdt = float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0)
    margin_plan = calc_status_margin_plan(trade_capital_usdt)

    status = "OK" if running and active_positions_checked else "WARNING"

    print(status)
    print("host:", LOCAL_HOST)
    print("service_running:", running)
    print("pid:", pid)
    print("service_module:", SERVICE_MODULE)
    print("control_module:", CONTROL_MODULE)
    print("dry_run_env:", os.environ.get("IMB_TRADING_DRY_RUN", "not_set"))
    print("pair_model_name:", config.PAIR_MODEL_NAME)
    print("grid_name:", config.GRID_NAME)
    print("gate2_thr:", config.GATE2_THR)
    print("gate4_thr:", config.GATE4_THR)
    print("gate5_1_thr:", config.GATE5_1_THR)
    print("gate5_3_thr:", config.GATE5_3_THR)
    print("dynamic_blacklist_source:", getattr(config, "DYNAMIC_BLACKLIST_SOURCE", "prod"))
    print("dynamic_symbol_filter_enabled:", getattr(config, "DYNAMIC_SYMBOL_FILTER_ENABLED", None))
    print("chulan_enabled:", bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)))
    print("partial_tp_enabled:", getattr(config, "PARTIAL_TP_ENABLED", None))
    print("partial_tp_level_fraction:", getattr(config, "PARTIAL_TP_LEVEL_FRACTION", None))
    print("partial_tp_qty_fraction:", getattr(config, "PARTIAL_TP_QTY_FRACTION", None))
    print("final_tp_qty_fraction:", getattr(config, "FINAL_TP_QTY_FRACTION", None))
    print("early_stop_enabled:", getattr(config, "EARLY_STOP_ENABLED", None))
    print("early_stop_window_minutes:", getattr(config, "EARLY_STOP_WINDOW_MINUTES", None))
    print("early_stop_sl_fraction:", getattr(config, "EARLY_STOP_SL_FRACTION", None))
    print("main_stop_after_early_window_enabled:", getattr(config, "MAIN_STOP_AFTER_EARLY_WINDOW_ENABLED", None))
    print("rest_stop_after_partial_enabled:", getattr(config, "REST_STOP_AFTER_PARTIAL_ENABLED", None))
    print("rest_stop_after_partial_atr_mult:", getattr(config, "REST_STOP_AFTER_PARTIAL_ATR_MULT", None))
    print("weekend_entry_filter_enabled:", getattr(config, "WEEKEND_ENTRY_FILTER_ENABLED", None))
    print("trade_management_poll_enabled_env:", os.environ.get("IMB_TRADE_MANAGEMENT_POLL_ENABLED", "default_1"))
    print("trade_management_poll_interval_seconds_env:", os.environ.get("IMB_TRADE_MANAGEMENT_POLL_INTERVAL_SECONDS", "default_30"))
    print("trade_management_poll_log_interval_seconds_env:", os.environ.get("IMB_TRADE_MANAGEMENT_POLL_LOG_INTERVAL_SECONDS", "default_1800"))
    print("early_stop_timer_enabled_env:", os.environ.get("IMB_EARLY_STOP_TIMER_ENABLED", "default_1"))
    print("log:", LOG_LINK)
    print("next_h4_close_utc:", close_ts)
    print("time_to_next_h4_close:", fmt_left(left))
    print("capital_usdt:", fmt_money(capital_usdt))
    print("trade_capital_usdt:", fmt_money(trade_capital_usdt))
    print("trading_leverage:", fmt_float(margin_plan.get("trading_leverage")))
    print("position_notional_multiplier:", fmt_float(margin_plan.get("position_notional_multiplier")))
    print("position_notional_usdt_plan:", fmt_money(margin_plan.get("position_notional_usdt_plan")))
    print("estimated_initial_margin_usdt:", fmt_money(margin_plan.get("estimated_initial_margin_usdt")))
    print("margin_buffer_usdt:", fmt_money(margin_plan.get("margin_buffer_usdt")))
    print("margin_ok:", bool(margin_plan.get("margin_ok")))
    print("position_risk_cap_enabled:", getattr(config, "POSITION_RISK_CAP_ENABLED", None))
    print("max_full_sl_capital_risk:", getattr(config, "MAX_FULL_SL_CAPITAL_RISK", None))
    print("position_risk_cap_include_round_trip_cost:", getattr(config, "POSITION_RISK_CAP_INCLUDE_ROUND_TRIP_COST", None))
    print("position_risk_cap_fee_side:", getattr(config, "POSITION_RISK_CAP_FEE_SIDE", None))
    print("position_risk_cap_slippage_side:", getattr(config, "POSITION_RISK_CAP_SLIPPAGE_SIDE", None))
    print("conditional_side_aware_whitelist_enabled:", getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST_ENABLED", None))
    print("chulan_enabled:", chulan_enabled)
    print("chulan_base_capital_usdt:", fmt_money(chulan_base_capital_usdt))
    print("current_position_pnl_usdt:", fmt_money(current_pnl))
    print("open_positions_count:", open_positions_count)
    print_ws_health_status()

    if exchange["error"]:
        print("error:", exchange["error"])

    if exchange["open_positions"]:
        rows = []
        for p in exchange["open_positions"]:
            rows.append(
                {
                    "symbol": str(p.get("symbol") or "").upper(),
                    "side": p.get("side"),
                    "size": p.get("size"),
                    "avgPrice": p.get("avgPrice"),
                    "markPrice": p.get("markPrice"),
                    "unrealisedPnl": p.get("unrealisedPnl"),
                }
            )

        print(pd.DataFrame(rows).to_string(index=False))

ACTIVE_POSITION_STATUSES = {
    "CREATED",
    "ENTRY_ORDER_SENT",
    "ENTRY_PARTIALLY_FILLED",
    "ENTRY_FILLED",
    "TP_SL_PLACED",
    "POSITION_OPEN",
    "TTL_CLOSE_SENT",
}

CLOSED_POSITION_PREFIX = "POSITION_CLOSED"


def normalize_symbol(raw: Any) -> str:
    text = str(raw or "").strip().upper()

    if not text:
        raise RuntimeError("symbol is empty")

    if not text.endswith("USDT"):
        text = text + "USDT"

    return text


def normalize_optional_symbol(raw: Any) -> Optional[str]:
    text = str(raw or "").strip().upper()

    if not text:
        return None

    if text in {HOST_MAC.upper(), HOST_WIN.upper(), "LOCAL"}:
        return None

    try:
        int(text)
        return None
    except Exception:
        pass

    if not text.endswith("USDT"):
        text = text + "USDT"

    return text

def normalize_position_limit(raw: Any) -> int:
    if raw is None:
        return 1

    try:
        value = int(str(raw).strip())
    except Exception:
        raise RuntimeError("position count must be integer, example: position ENAUSDT 3")

    if value <= 0:
        raise RuntimeError("position count must be > 0")

    return min(value, 20)

def fmt_ts_utc(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(ts):
        return "-"

    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_duration_seconds(seconds: Any) -> str:
    try:
        sec = int(max(0, float(seconds)))
    except Exception:
        return "-"

    days = sec // 86400
    rest = sec % 86400
    hours = rest // 3600
    minutes = (rest % 3600) // 60
    secs = rest % 60

    if days > 0:
        return "{}d {:02d}:{:02d}:{:02d}".format(days, hours, minutes, secs)

    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, secs)


def fmt_signed_money(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"

    sign = "+" if x > 0 else ""
    return "{}{:.6f}".format(sign, x)


def fmt_signed_pct(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"

    sign = "+" if x > 0 else ""
    return "{}{:.4f}%".format(sign, x * 100.0)


def safe_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def side_to_exchange_side(side: Any) -> str:
    side_u = str(side or "").upper()

    if side_u == "LONG":
        return "Buy"

    if side_u == "SHORT":
        return "Sell"

    return side_u


def calc_entry_value_usdt(entry_avg_px: Any, qty: Any) -> Optional[float]:
    px = safe_float_or_none(entry_avg_px)
    q = safe_float_or_none(qty)

    if px is None or q is None:
        return None

    if px <= 0 or q <= 0:
        return None

    return float(px * q)


def calc_pnl_pct(pnl_usd: Any, entry_value_usdt: Any) -> Optional[float]:
    pnl = safe_float_or_none(pnl_usd)
    entry_value = safe_float_or_none(entry_value_usdt)

    if pnl is None or entry_value is None or entry_value <= 0:
        return None

    return float(pnl / entry_value)


def load_latest_position(symbol: Optional[str], closed: Optional[bool]) -> Optional[Dict[str, Any]]:
    ensure_status_position_columns()
    symbol_u = normalize_optional_symbol(symbol)

    if closed is True:
        status_filter = "AND p.status LIKE 'POSITION_CLOSED%%'"
    elif closed is False:
        status_filter = "AND p.status NOT LIKE 'POSITION_CLOSED%%' AND p.status NOT IN ('ENTRY_FAILED', 'ENTRY_REJECTED', 'CANCELLED', 'FAILED')"
    else:
        status_filter = ""

    symbol_filter = ""
    params: List[Any] = []

    if symbol_u is not None:
        symbol_filter = "AND p.symbol = %s"
        params.append(symbol_u)

    sql = """
        SELECT
            p.trade_id,
            p.signal_key,
            p.symbol,
            p.side,
            p.status,
            p.qty,
            p.entry_px_plan,
            p.entry_avg_px,
            p.entry_filled_at,
            p.tp_px_plan,
            p.sl_px_plan,
            p.partial_tp_px_plan,
            p.final_tp_px_plan,
            p.early_stop_px_plan,
            p.main_sl_px_plan,
            p.rest_stop_after_partial_px_plan,
            p.partial_tp_qty_plan,
            p.final_tp_qty_plan,
            p.early_stop_expires_at,
            p.trade_management_mode,
            p.partial_tp_handled_at,
            p.early_stop_replaced_at,
            p.protective_cleanup_done_at,
            p.closed_cleanup_done_at,
            p.ws_lifecycle_updated_at,
            p.ws_lifecycle_last_event_at,
            p.ws_lifecycle_last_error,
            p.ttl_close_ts,
            p.exit_reason,
            p.exit_order_id,
            p.exit_avg_px,
            p.exit_filled_at,
            p.pnl_usd,
            p.fee_usd,
            p.created_at,
            p.updated_at,
            s.signal_ts,
            s.entry_ts_plan,
            s.h4_close,
            s.atr14,
            s.gate2_proba,
            s.gate4_confidence,
            s.gate5_1_proba,
            s.gate5_3_proba
        FROM public.trading_positions p
        LEFT JOIN public.trading_signals s
            ON s.signal_key = p.signal_key
        WHERE 1 = 1
        {symbol_filter}
        {status_filter}
        ORDER BY
            COALESCE(p.entry_filled_at, p.created_at) DESC,
            p.trade_id DESC
        LIMIT 1
    """.format(
        symbol_filter=symbol_filter,
        status_filter=status_filter,
    )

    df = read_sql(sql, params)

    if df.empty:
        return None

    return df.iloc[0].to_dict()

def load_closed_positions(symbol: Optional[str], limit: int) -> List[Dict[str, Any]]:
    ensure_status_position_columns()
    symbol_u = normalize_optional_symbol(symbol)
    limit_i = normalize_position_limit(limit)

    symbol_filter = ""
    params: List[Any] = []

    if symbol_u is not None:
        symbol_filter = "AND p.symbol = %s"
        params.append(symbol_u)

    params.append(int(limit_i))

    sql = """
        SELECT
            p.trade_id,
            p.signal_key,
            p.symbol,
            p.side,
            p.status,
            p.qty,
            p.entry_px_plan,
            p.entry_avg_px,
            p.entry_filled_at,
            p.tp_px_plan,
            p.sl_px_plan,
            p.partial_tp_px_plan,
            p.final_tp_px_plan,
            p.early_stop_px_plan,
            p.main_sl_px_plan,
            p.rest_stop_after_partial_px_plan,
            p.partial_tp_qty_plan,
            p.final_tp_qty_plan,
            p.early_stop_expires_at,
            p.trade_management_mode,
            p.partial_tp_handled_at,
            p.early_stop_replaced_at,
            p.protective_cleanup_done_at,
            p.closed_cleanup_done_at,
            p.ws_lifecycle_updated_at,
            p.ws_lifecycle_last_event_at,
            p.ws_lifecycle_last_error,
            p.ttl_close_ts,
            p.exit_reason,
            p.exit_order_id,
            p.exit_avg_px,
            p.exit_filled_at,
            p.pnl_usd,
            p.fee_usd,
            p.created_at,
            p.updated_at,
            s.signal_ts,
            s.entry_ts_plan,
            s.h4_close,
            s.atr14,
            s.gate2_proba,
            s.gate4_confidence,
            s.gate5_1_proba,
            s.gate5_3_proba
        FROM public.trading_positions p
        LEFT JOIN public.trading_signals s
            ON s.signal_key = p.signal_key
        WHERE p.status LIKE 'POSITION_CLOSED%%'
        {symbol_filter}
        ORDER BY
            COALESCE(p.exit_filled_at, p.updated_at, p.entry_filled_at, p.created_at) DESC,
            p.trade_id DESC
        LIMIT %s
    """.format(symbol_filter=symbol_filter)

    df = read_sql(sql, params)

    if df.empty:
        return []

    return [row.to_dict() for _, row in df.iterrows()]


def load_position_fills_summary(trade_id: int) -> Dict[str, Any]:
    sql = """
        SELECT
            trade_id,
            order_role,
            exec_qty,
            exec_price,
            exec_value,
            exec_fee,
            executed_at
        FROM public.trading_fills
        WHERE trade_id = %s
        ORDER BY executed_at ASC NULLS LAST
    """

    try:
        df = read_sql(sql, [int(trade_id)])
    except Exception:
        return {}

    if df.empty:
        return {}

    df["order_role"] = df["order_role"].astype(str).str.upper()

    for col in ["exec_qty", "exec_price", "exec_value", "exec_fee"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["executed_at"] = pd.to_datetime(df["executed_at"], utc=True, errors="coerce")

    exit_roles = [
        "TAKE_PROFIT",
        "PARTIAL_TP",
        "FINAL_TP",
        "STOP_LOSS",
        "EARLY_STOP",
        "REST_STOP_AFTER_PARTIAL",
        "TTL_CLOSE",
        "EMERGENCY_CLOSE",
        "MANUAL_CLOSE",
    ]

    def sum_qty(role: str) -> Optional[float]:
        value = float(df.loc[df["order_role"] == role, "exec_qty"].sum(skipna=True) or 0.0)
        return value if value > 0 else None

    def avg_px(role: str) -> Optional[float]:
        part = df[df["order_role"] == role].copy()
        if part.empty:
            return None

        qty = float(part["exec_qty"].sum(skipna=True) or 0.0)
        value = float(part["exec_value"].sum(skipna=True) or 0.0)

        if qty <= 0:
            return None

        return float(value / qty)

    def first_ts(role: str) -> Optional[str]:
        part = df[df["order_role"] == role].copy()
        if part.empty:
            return None

        value = part["executed_at"].min()
        if pd.isna(value):
            return None

        return str(pd.Timestamp(value))

    entry_df = df[df["order_role"] == "ENTRY_MARKET"].copy()
    exit_df = df[df["order_role"].isin(exit_roles)].copy()

    entry_qty = float(entry_df["exec_qty"].sum(skipna=True) or 0.0) if not entry_df.empty else 0.0
    entry_value = float(entry_df["exec_value"].sum(skipna=True) or 0.0) if not entry_df.empty else 0.0
    exit_qty = float(exit_df["exec_qty"].sum(skipna=True) or 0.0) if not exit_df.empty else 0.0
    exit_value = float(exit_df["exec_value"].sum(skipna=True) or 0.0) if not exit_df.empty else 0.0

    first_entry_ts = entry_df["executed_at"].min() if not entry_df.empty else pd.NaT
    last_exit_ts = exit_df["executed_at"].max() if not exit_df.empty else pd.NaT

    partial_tp_qty = sum_qty("PARTIAL_TP")
    final_tp_qty = sum_qty("FINAL_TP")
    take_profit_qty = sum_qty("TAKE_PROFIT")
    stop_loss_qty = sum_qty("STOP_LOSS")
    early_stop_qty = sum_qty("EARLY_STOP")
    rest_stop_qty = sum_qty("REST_STOP_AFTER_PARTIAL")
    ttl_close_qty = sum_qty("TTL_CLOSE")

    filled_roles = []
    for role in exit_roles:
        if sum_qty(role):
            filled_roles.append(role)

    if partial_tp_qty and final_tp_qty:
        execution_kind = "PARTIAL_TP_PLUS_FINAL_TP"
    elif partial_tp_qty and rest_stop_qty:
        execution_kind = "PARTIAL_TP_PLUS_REST_STOP"
    elif partial_tp_qty and stop_loss_qty:
        execution_kind = "PARTIAL_TP_PLUS_MAIN_SL"
    elif final_tp_qty:
        execution_kind = "FINAL_TP"
    elif take_profit_qty:
        execution_kind = "TAKE_PROFIT"
    elif stop_loss_qty:
        execution_kind = "STOP_LOSS"
    elif early_stop_qty:
        execution_kind = "EARLY_STOP"
    elif rest_stop_qty:
        execution_kind = "REST_STOP_AFTER_PARTIAL"
    elif ttl_close_qty:
        execution_kind = "TTL_CLOSE"
    elif entry_qty > 0:
        execution_kind = "ENTRY_ONLY"
    else:
        execution_kind = "NO_FILLS"

    return {
        "first_entry_fill_ts": None if pd.isna(first_entry_ts) else str(pd.Timestamp(first_entry_ts)),
        "last_exit_fill_ts": None if pd.isna(last_exit_ts) else str(pd.Timestamp(last_exit_ts)),
        "entry_qty": entry_qty if entry_qty > 0 else None,
        "exit_qty": exit_qty if exit_qty > 0 else None,
        "partial_tp_qty": partial_tp_qty,
        "final_tp_qty": final_tp_qty,
        "take_profit_qty": take_profit_qty,
        "stop_loss_qty": stop_loss_qty,
        "early_stop_qty": early_stop_qty,
        "rest_stop_after_partial_qty": rest_stop_qty,
        "ttl_close_qty": ttl_close_qty,
        "stop_exit_qty": float(
            (stop_loss_qty or 0.0)
            + (early_stop_qty or 0.0)
            + (rest_stop_qty or 0.0)
        ) or None,
        "entry_value_usdt": entry_value if entry_value > 0 else None,
        "exit_value_usdt": exit_value if exit_value > 0 else None,
        "entry_avg_px_from_fills": entry_value / entry_qty if entry_qty > 0 else None,
        "exit_avg_px_from_fills": exit_value / exit_qty if exit_qty > 0 else None,
        "partial_tp_avg_px": avg_px("PARTIAL_TP"),
        "final_tp_avg_px": avg_px("FINAL_TP"),
        "take_profit_avg_px": avg_px("TAKE_PROFIT"),
        "stop_loss_avg_px": avg_px("STOP_LOSS"),
        "early_stop_avg_px": avg_px("EARLY_STOP"),
        "rest_stop_after_partial_avg_px": avg_px("REST_STOP_AFTER_PARTIAL"),
        "ttl_close_avg_px": avg_px("TTL_CLOSE"),
        "partial_tp_first_ts": first_ts("PARTIAL_TP"),
        "final_tp_first_ts": first_ts("FINAL_TP"),
        "take_profit_first_ts": first_ts("TAKE_PROFIT"),
        "stop_loss_first_ts": first_ts("STOP_LOSS"),
        "early_stop_first_ts": first_ts("EARLY_STOP"),
        "rest_stop_after_partial_first_ts": first_ts("REST_STOP_AFTER_PARTIAL"),
        "ttl_close_first_ts": first_ts("TTL_CLOSE"),
        "total_fee_usdt": float(df["exec_fee"].sum(skipna=True) or 0.0),
        "fills_count": int(len(df)),
        "execution_kind": execution_kind,
        "filled_roles": ",".join(filled_roles),
    }

def get_exchange_position(symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    symbol_u = normalize_optional_symbol(symbol)

    client = BybitClient()
    positions = client.get_open_positions()

    for p in positions:
        if symbol_u is not None and str(p.get("symbol") or "").upper() != symbol_u:
            continue

        try:
            size = abs(float(p.get("size") or 0.0))
        except Exception:
            size = 0.0

        if size > 0:
            return p

    return None

def build_position_report_item(
    title: str,
    row: Optional[Dict[str, Any]],
    exchange_position: Optional[Dict[str, Any]],
    now_ts: pd.Timestamp,
) -> Dict[str, Any]:
    if row is None:
        return {
            "title": title,
            "exists": False,
        }

    trade_id = int(row["trade_id"])
    fills_summary = load_position_fills_summary(trade_id)

    entry_filled_at = row.get("entry_filled_at")
    if pd.isna(pd.to_datetime(entry_filled_at, utc=True, errors="coerce")):
        entry_filled_at = fills_summary.get("first_entry_fill_ts")

    exit_filled_at = row.get("exit_filled_at")
    if pd.isna(pd.to_datetime(exit_filled_at, utc=True, errors="coerce")):
        exit_filled_at = fills_summary.get("last_exit_fill_ts")

    entry_ts = pd.to_datetime(entry_filled_at, utc=True, errors="coerce")
    exit_ts = pd.to_datetime(exit_filled_at, utc=True, errors="coerce")

    is_closed = str(row.get("status") or "").startswith(CLOSED_POSITION_PREFIX)

    if is_closed and pd.notna(exit_ts) and pd.notna(entry_ts):
        life_seconds = float((exit_ts - entry_ts).total_seconds())
    elif pd.notna(entry_ts):
        life_seconds = float((now_ts - entry_ts).total_seconds())
    else:
        life_seconds = None

    entry_avg_px = safe_float_or_none(row.get("entry_avg_px"))
    if entry_avg_px is None:
        entry_avg_px = safe_float_or_none(fills_summary.get("entry_avg_px_from_fills"))

    exit_avg_px = safe_float_or_none(row.get("exit_avg_px"))
    if exit_avg_px is None:
        exit_avg_px = safe_float_or_none(fills_summary.get("exit_avg_px_from_fills"))

    qty = safe_float_or_none(row.get("qty"))

    entry_value_usdt = safe_float_or_none(fills_summary.get("entry_value_usdt"))
    if entry_value_usdt is None:
        entry_value_usdt = calc_entry_value_usdt(entry_avg_px, qty)

    pnl_usd = safe_float_or_none(row.get("pnl_usd"))

    mark_price = None
    unrealised_pnl = None
    exchange_size = None
    exchange_side = None

    if exchange_position is not None:
        mark_price = safe_float_or_none(exchange_position.get("markPrice"))
        unrealised_pnl = safe_float_or_none(exchange_position.get("unrealisedPnl"))
        exchange_size = safe_float_or_none(exchange_position.get("size"))
        exchange_side = str(exchange_position.get("side") or "")

        if not is_closed and unrealised_pnl is not None:
            pnl_usd = unrealised_pnl

        position_value = safe_float_or_none(exchange_position.get("positionValue"))
        if position_value is not None and position_value > 0:
            entry_value_usdt = position_value

    pnl_pct = calc_pnl_pct(pnl_usd, entry_value_usdt)

    return {
        "title": title,
        "exists": True,
        "trade_id": trade_id,
        "signal_key": str(row.get("signal_key") or ""),
        "symbol": str(row.get("symbol") or "").upper(),
        "side": str(row.get("side") or "").upper(),
        "exchange_side": exchange_side,
        "status": str(row.get("status") or ""),
        "qty": qty,
        "exchange_size": exchange_size,
        "entry_px_plan": safe_float_or_none(row.get("entry_px_plan")),
        "entry_avg_px": entry_avg_px,
        "exit_avg_px": exit_avg_px,
        "mark_price": mark_price,
        "tp_px_plan": safe_float_or_none(row.get("tp_px_plan")),
        "sl_px_plan": safe_float_or_none(row.get("sl_px_plan")),
        "partial_tp_px_plan": safe_float_or_none(row.get("partial_tp_px_plan")),
        "final_tp_px_plan": safe_float_or_none(row.get("final_tp_px_plan")),
        "early_stop_px_plan": safe_float_or_none(row.get("early_stop_px_plan")),
        "main_sl_px_plan": safe_float_or_none(row.get("main_sl_px_plan")),
        "rest_stop_after_partial_px_plan": safe_float_or_none(row.get("rest_stop_after_partial_px_plan")),
        "partial_tp_qty_plan": safe_float_or_none(row.get("partial_tp_qty_plan")),
        "final_tp_qty_plan": safe_float_or_none(row.get("final_tp_qty_plan")),
        "partial_tp_qty_filled": safe_float_or_none(fills_summary.get("partial_tp_qty")),
        "final_tp_qty_filled": safe_float_or_none(fills_summary.get("final_tp_qty")),
        "take_profit_qty_filled": safe_float_or_none(fills_summary.get("take_profit_qty")),
        "stop_loss_qty_filled": safe_float_or_none(fills_summary.get("stop_loss_qty")),
        "early_stop_qty_filled": safe_float_or_none(fills_summary.get("early_stop_qty")),
        "rest_stop_after_partial_qty_filled": safe_float_or_none(fills_summary.get("rest_stop_after_partial_qty")),
        "ttl_close_qty_filled": safe_float_or_none(fills_summary.get("ttl_close_qty")),
        "stop_exit_qty_filled": safe_float_or_none(fills_summary.get("stop_exit_qty")),
        "execution_kind": str(fills_summary.get("execution_kind") or "NO_FILLS"),
        "filled_roles": str(fills_summary.get("filled_roles") or ""),
        "partial_tp_avg_px": safe_float_or_none(fills_summary.get("partial_tp_avg_px")),
        "final_tp_avg_px": safe_float_or_none(fills_summary.get("final_tp_avg_px")),
        "take_profit_avg_px": safe_float_or_none(fills_summary.get("take_profit_avg_px")),
        "stop_loss_avg_px": safe_float_or_none(fills_summary.get("stop_loss_avg_px")),
        "early_stop_avg_px": safe_float_or_none(fills_summary.get("early_stop_avg_px")),
        "rest_stop_after_partial_avg_px": safe_float_or_none(fills_summary.get("rest_stop_after_partial_avg_px")),
        "ttl_close_avg_px": safe_float_or_none(fills_summary.get("ttl_close_avg_px")),
        "partial_tp_first_ts": fills_summary.get("partial_tp_first_ts"),
        "final_tp_first_ts": fills_summary.get("final_tp_first_ts"),
        "take_profit_first_ts": fills_summary.get("take_profit_first_ts"),
        "stop_loss_first_ts": fills_summary.get("stop_loss_first_ts"),
        "early_stop_first_ts": fills_summary.get("early_stop_first_ts"),
        "rest_stop_after_partial_first_ts": fills_summary.get("rest_stop_after_partial_first_ts"),
        "ttl_close_first_ts": fills_summary.get("ttl_close_first_ts"),
        "early_stop_expires_at": None if pd.isna(pd.to_datetime(row.get("early_stop_expires_at"), utc=True, errors="coerce")) else str(pd.to_datetime(row.get("early_stop_expires_at"), utc=True, errors="coerce")),
        "trade_management_mode": None if row.get("trade_management_mode") is None or pd.isna(row.get("trade_management_mode")) else str(row.get("trade_management_mode")),
        "partial_tp_handled_at": None if pd.isna(pd.to_datetime(row.get("partial_tp_handled_at"), utc=True, errors="coerce")) else str(pd.to_datetime(row.get("partial_tp_handled_at"), utc=True, errors="coerce")),
        "early_stop_replaced_at": None if pd.isna(pd.to_datetime(row.get("early_stop_replaced_at"), utc=True, errors="coerce")) else str(pd.to_datetime(row.get("early_stop_replaced_at"), utc=True, errors="coerce")),
        "protective_cleanup_done_at": None if pd.isna(pd.to_datetime(row.get("protective_cleanup_done_at"), utc=True, errors="coerce")) else str(pd.to_datetime(row.get("protective_cleanup_done_at"), utc=True, errors="coerce")),
        "closed_cleanup_done_at": None if pd.isna(pd.to_datetime(row.get("closed_cleanup_done_at"), utc=True, errors="coerce")) else str(pd.to_datetime(row.get("closed_cleanup_done_at"), utc=True, errors="coerce")),
        "ws_lifecycle_updated_at": None if pd.isna(pd.to_datetime(row.get("ws_lifecycle_updated_at"), utc=True, errors="coerce")) else str(pd.to_datetime(row.get("ws_lifecycle_updated_at"), utc=True, errors="coerce")),
        "ws_lifecycle_last_event_at": None if pd.isna(pd.to_datetime(row.get("ws_lifecycle_last_event_at"), utc=True, errors="coerce")) else str(pd.to_datetime(row.get("ws_lifecycle_last_event_at"), utc=True, errors="coerce")),
        "ws_lifecycle_last_error": None if row.get("ws_lifecycle_last_error") is None or pd.isna(row.get("ws_lifecycle_last_error")) else str(row.get("ws_lifecycle_last_error")),
        "entry_filled_at": None if pd.isna(entry_ts) else str(entry_ts),
        "exit_filled_at": None if pd.isna(exit_ts) else str(exit_ts),
        "ttl_close_ts": None if pd.isna(pd.to_datetime(row.get("ttl_close_ts"), utc=True, errors="coerce")) else str(pd.to_datetime(row.get("ttl_close_ts"), utc=True, errors="coerce")),
        "life_seconds": life_seconds,
        "entry_value_usdt": entry_value_usdt,
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct,
        "fee_usd": safe_float_or_none(row.get("fee_usd")),
        "exit_reason": None if row.get("exit_reason") is None or pd.isna(row.get("exit_reason")) else str(row.get("exit_reason")),
        "fills_count": int(fills_summary.get("fills_count") or 0),
        "signal_ts": None if pd.isna(pd.to_datetime(row.get("signal_ts"), utc=True, errors="coerce")) else str(pd.to_datetime(row.get("signal_ts"), utc=True, errors="coerce")),
        "h4_close": safe_float_or_none(row.get("h4_close")),
        "gate2": safe_float_or_none(row.get("gate2_proba")),
        "gate4": safe_float_or_none(row.get("gate4_confidence")),
        "gate5_1": safe_float_or_none(row.get("gate5_1_proba")),
        "gate5_3": safe_float_or_none(row.get("gate5_3_proba")),
    }


def build_position_report(symbol: Optional[str], limit: int = 1) -> Dict[str, Any]:
    symbol_u = normalize_optional_symbol(symbol)
    limit_i = normalize_position_limit(limit)
    now_ts = pd.Timestamp.now(tz="UTC")

    current_row = load_latest_position(symbol_u, closed=False)
    closed_rows = load_closed_positions(symbol_u, limit_i)

    exchange_position = None
    exchange_error = None

    try:
        exchange_position = get_exchange_position(symbol_u)
    except Exception as e:
        exchange_error = str(e)

    current_exchange_position = None

    if current_row is not None and exchange_position is not None:
        current_symbol = str(current_row.get("symbol") or "").upper()
        exchange_symbol = str(exchange_position.get("symbol") or "").upper()

        if current_symbol == exchange_symbol:
            current_exchange_position = exchange_position

    return {
        "symbol": symbol_u or "ALL",
        "limit": limit_i,
        "now_utc": str(now_ts),
        "exchange_error": exchange_error,
        "current": build_position_report_item(
            title="current_position",
            row=current_row,
            exchange_position=current_exchange_position,
            now_ts=now_ts,
        ),
        "last_closed": [
            build_position_report_item(
                title="last_closed_position_{}".format(i),
                row=row,
                exchange_position=None,
                now_ts=now_ts,
            )
            for i, row in enumerate(closed_rows, start=1)
        ],
    }


def print_position_report_item(item: Dict[str, Any]) -> None:
    title = str(item.get("title") or "")

    print("-" * 120)
    print(title)

    if not bool(item.get("exists")):
        print("exists: False")
        return

    print("exists: True")
    print("trade_id:", item.get("trade_id"))
    print("signal_key:", item.get("signal_key"))
    print("symbol:", item.get("symbol"))
    print("side:", item.get("side"))
    print("status:", item.get("status"))
    print("qty:", fmt_money(item.get("qty")))
    print("entry_time:", fmt_ts_utc(item.get("entry_filled_at")))
    print("exit_time:", fmt_ts_utc(item.get("exit_filled_at")))
    print("life:", fmt_duration_seconds(item.get("life_seconds")))
    print("entry_plan:", fmt_float(item.get("entry_px_plan")))
    print("entry_actual:", fmt_float(item.get("entry_avg_px")))
    print("mark_price:", fmt_float(item.get("mark_price")))
    print("tp:", fmt_float(item.get("tp_px_plan")))
    print("sl:", fmt_float(item.get("sl_px_plan")))
    print("trade_management_mode:", item.get("trade_management_mode") or "-")
    print("partial_tp_px:", fmt_float(item.get("partial_tp_px_plan")))
    print("final_tp_px:", fmt_float(item.get("final_tp_px_plan")))
    print("early_stop_px:", fmt_float(item.get("early_stop_px_plan")))
    print("main_sl_px:", fmt_float(item.get("main_sl_px_plan")))
    print("rest_stop_after_partial_px:", fmt_float(item.get("rest_stop_after_partial_px_plan")))
    print("partial_tp_qty_plan:", fmt_money(item.get("partial_tp_qty_plan")))
    print("final_tp_qty_plan:", fmt_money(item.get("final_tp_qty_plan")))
    print("partial_tp_qty_filled:", fmt_money(item.get("partial_tp_qty_filled")))
    print("final_tp_qty_filled:", fmt_money(item.get("final_tp_qty_filled")))
    print("take_profit_qty_filled:", fmt_money(item.get("take_profit_qty_filled")))
    print("stop_loss_qty_filled:", fmt_money(item.get("stop_loss_qty_filled")))
    print("early_stop_qty_filled:", fmt_money(item.get("early_stop_qty_filled")))
    print("rest_stop_after_partial_qty_filled:", fmt_money(item.get("rest_stop_after_partial_qty_filled")))
    print("ttl_close_qty_filled:", fmt_money(item.get("ttl_close_qty_filled")))
    print("stop_exit_qty_filled:", fmt_money(item.get("stop_exit_qty_filled")))
    print("execution_kind:", item.get("execution_kind") or "-")
    print("filled_roles:", item.get("filled_roles") or "-")
    print("partial_tp_fill:", "px=" + fmt_float(item.get("partial_tp_avg_px")) + " time=" + fmt_ts_utc(item.get("partial_tp_first_ts")))
    print("final_tp_fill:", "px=" + fmt_float(item.get("final_tp_avg_px")) + " time=" + fmt_ts_utc(item.get("final_tp_first_ts")))
    print("take_profit_fill:", "px=" + fmt_float(item.get("take_profit_avg_px")) + " time=" + fmt_ts_utc(item.get("take_profit_first_ts")))
    print("stop_loss_fill:", "px=" + fmt_float(item.get("stop_loss_avg_px")) + " time=" + fmt_ts_utc(item.get("stop_loss_first_ts")))
    print("early_stop_fill:", "px=" + fmt_float(item.get("early_stop_avg_px")) + " time=" + fmt_ts_utc(item.get("early_stop_first_ts")))
    print("rest_stop_after_partial_fill:", "px=" + fmt_float(item.get("rest_stop_after_partial_avg_px")) + " time=" + fmt_ts_utc(item.get("rest_stop_after_partial_first_ts")))
    print("ttl_close_fill:", "px=" + fmt_float(item.get("ttl_close_avg_px")) + " time=" + fmt_ts_utc(item.get("ttl_close_first_ts")))
    print("early_stop_expires_at:", fmt_ts_utc(item.get("early_stop_expires_at")))
    print("early_stop_replaced_at:", fmt_ts_utc(item.get("early_stop_replaced_at")))
    print("partial_tp_handled_at:", fmt_ts_utc(item.get("partial_tp_handled_at")))
    print("protective_cleanup_done_at:", fmt_ts_utc(item.get("protective_cleanup_done_at")))
    print("closed_cleanup_done_at:", fmt_ts_utc(item.get("closed_cleanup_done_at")))
    print("ws_lifecycle_updated_at:", fmt_ts_utc(item.get("ws_lifecycle_updated_at")))
    print("ws_lifecycle_last_error:", item.get("ws_lifecycle_last_error") or "-")
    print("entry_value_usdt:", fmt_money(item.get("entry_value_usdt")))
    print("pnl_usdt:", fmt_signed_money(item.get("pnl_usd")))
    print("pnl_pct:", fmt_signed_pct(item.get("pnl_pct")))
    print("fee_usdt:", fmt_money(item.get("fee_usd")))
    print("exit_reason:", item.get("exit_reason") or "-")
    print("fills_count:", item.get("fills_count"))


def print_position(argv: List[str]) -> None:
    symbol: Optional[str] = None
    limit = 1
    host = HOST_WIN

    if len(argv) >= 3:
        raw_arg = str(argv[2]).strip()

        if raw_arg.lower() in {HOST_MAC, HOST_WIN, "local"}:
            host = LOCAL_HOST if raw_arg.lower() == "local" else raw_arg.lower()
        else:
            parsed_symbol = normalize_optional_symbol(raw_arg)

            if parsed_symbol is None:
                limit = normalize_position_limit(raw_arg)
            else:
                symbol = parsed_symbol

    if len(argv) >= 4:
        raw_arg = str(argv[3]).strip().lower()

        if raw_arg in {HOST_MAC, HOST_WIN, "local"}:
            host = LOCAL_HOST if raw_arg == "local" else raw_arg
        else:
            limit = normalize_position_limit(raw_arg)

    if len(argv) >= 5:
        raw_host = str(argv[4]).strip().lower()

        if raw_host in {HOST_MAC, HOST_WIN, "local"}:
            host = LOCAL_HOST if raw_host == "local" else raw_host
        else:
            raise RuntimeError("position host must be mac or win")

    if host == HOST_WIN and LOCAL_HOST != HOST_WIN:
        remote_args = ["position-local"]

        if symbol is not None:
            remote_args.append(symbol)

        remote_args.append(str(int(limit)))

        code, out = run_win_control(remote_args)
        print_remote_block("WIN_POSITION", code, out)

        if code != 0:
            raise RuntimeError("win position failed")

        return

    report = build_position_report(symbol, limit=limit)

    output_format = os.environ.get("IMB_POSITION_OUTPUT_FORMAT", "table").strip().lower()

    if output_format == "json":
        print("POSITION_JSON_BEGIN")
        print(json.dumps(report, ensure_ascii=False, default=str))
        print("POSITION_JSON_END")
        return

    print("=" * 120)
    print("POSITION_REPORT")
    print("symbol:", report.get("symbol"))
    print("limit:", report.get("limit"))
    print("now_utc:", fmt_ts_utc(report.get("now_utc")))

    if report.get("exchange_error"):
        print("exchange_error:", report.get("exchange_error"))

    print_position_report_item(report["current"])

    closed_items = report.get("last_closed") or []

    if not closed_items:
        print("-" * 120)
        print("last_closed_positions")
        print("exists: False")
        return

    for item in closed_items:
        print_position_report_item(item)

def ask_text(prompt: str, default: Optional[str] = None) -> str:
    if default is None:
        raw = input(prompt + ": ").strip()
    else:
        raw = input(prompt + " [" + str(default) + "]: ").strip()

    if raw:
        return raw

    if default is not None:
        return str(default)

    raise RuntimeError("empty required value: " + prompt)


def ask_float_value(prompt: str, default: float) -> float:
    raw = ask_text(prompt, str(default))
    return float(str(raw).replace(",", "."))


def normalize_backtest_time(raw: str) -> str:
    text = str(raw or "").strip()

    if not text:
        raise RuntimeError("empty backtest time")

    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "：": ":",
        "–": "-",
        "—": "-",
        "−": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = "".join(ch for ch in text if ch.isascii())
    text = " ".join(text.split())

    ts = pd.to_datetime(text, utc=True, errors="coerce")

    if pd.isna(ts):
        raise RuntimeError(
            "bad backtest time: {}. Use format YYYY-MM-DD HH:MM, example: 2026-05-01 12:00".format(text)
        )

    ts = pd.Timestamp(ts).floor("min")
    return ts.strftime("%Y-%m-%d %H:%M")
def collect_backtest_args_interactive() -> List[str]:
    print("-" * 120)
    print("BACKTEST_INPUT")
    print("time_format: YYYY-MM-DD HH:MM")
    print("example_start: 2026-05-01 12:00")
    print("example_end:   2026-05-05 16:00")
    print("-" * 120)

    start = normalize_backtest_time(
        ask_text("Введите начало backtest UTC, формат YYYY-MM-DD HH:MM")
    )

    end = normalize_backtest_time(
        ask_text("Введите конец backtest UTC, формат YYYY-MM-DD HH:MM")
    )

    gate2 = ask_float_value("Введите порог Gate2", float(config.GATE2_THR))
    gate4 = ask_float_value("Введите порог Gate4", float(config.GATE4_THR))
    gate5_1 = ask_float_value("Введите порог Gate5.1", float(config.GATE5_1_THR))
    gate5_3 = ask_float_value("Введите порог Gate5.3", float(config.GATE5_3_THR))

    chulan = int(ask_text("Чулан включить? 0=без чулана, 1=с чуланом", "0"))
    if chulan not in [0, 1]:
        raise RuntimeError("bad chulan value: {}. Use 0 or 1".format(chulan))

    side_aware_whitelist = int(
        ask_text("Side-aware whitelist включить? 0=нет, 1=да", "1")
    )
    if side_aware_whitelist not in [0, 1]:
        raise RuntimeError("bad side_aware_whitelist value: {}. Use 0 or 1".format(side_aware_whitelist))

    conditional_side_aware_whitelist = int(
        ask_text("Conditional whitelist включить? 0=нет, 1=да", "1")
    )
    if conditional_side_aware_whitelist not in [0, 1]:
        raise RuntimeError(
            "bad conditional_side_aware_whitelist value: {}. Use 0 or 1".format(
                conditional_side_aware_whitelist
            )
        )

    max_full_sl_risk_pct = ask_float_value(
        "Максимальный риск полного MAIN_SL, %, например 6",
        6.0,
    )
    if max_full_sl_risk_pct < 0.0 or max_full_sl_risk_pct > 100.0:
        raise RuntimeError("bad max_full_sl_risk_pct: {}".format(max_full_sl_risk_pct))

    max_full_sl_capital_risk = max_full_sl_risk_pct / 100.0

    write_dynamic_blacklist = int(
        ask_text("Записывать dynamic blacklist? 0=нет, 1=да", "0")
    )
    if write_dynamic_blacklist not in [0, 1]:
        raise RuntimeError(
            "bad write_dynamic_blacklist value: {}. Use 0 or 1".format(
                write_dynamic_blacklist
            )
        )

    reset_backtest_blacklist = int(
        ask_text("Сбросить blacklist для этих порогов перед запуском? 0=нет, 1=да", "0")
    )
    if reset_backtest_blacklist not in [0, 1]:
        raise RuntimeError(
            "bad reset_backtest_blacklist value: {}. Use 0 or 1".format(
                reset_backtest_blacklist
            )
        )

    return [
        "--start",
        start,
        "--end",
        end,
        "--gate2",
        str(gate2),
        "--gate4",
        str(gate4),
        "--gate5-1",
        str(gate5_1),
        "--gate5-3",
        str(gate5_3),
        "--chulan",
        str(chulan),
        "--side-aware-whitelist",
        str(side_aware_whitelist),
        "--conditional-side-aware-whitelist",
        str(conditional_side_aware_whitelist),
        "--max-full-sl-capital-risk",
        str(max_full_sl_capital_risk),
        "--write-dynamic-blacklist",
        str(write_dynamic_blacklist),
        "--reset-backtest-blacklist",
        str(reset_backtest_blacklist),
    ]


def run_backtest_local(backtest_args: List[str]) -> None:
    env = build_child_env()

    cmd = [
        get_service_python_executable(),
        "-u",
        "-m",
        BACKTEST_MODULE,
    ] + list(backtest_args)

    print("-" * 120)
    print("RUN_BACKTEST_LOCAL")
    print("host:", LOCAL_HOST)
    print("module:", BACKTEST_MODULE)
    print("args:", " ".join(backtest_args))
    print("-" * 120)

    code, out = run_command_capture(
        cmd=cmd,
        cwd=ROOT,
        env=env,
    )

    if out.strip():
        safe_out = out.rstrip()
        try:
            print(safe_out)
        except UnicodeEncodeError:
            encoded = safe_out.encode("utf-8", errors="replace")
            sys.stdout.buffer.write(encoded + b"\\n")
            sys.stdout.buffer.flush()

    if code != 0:
        raise RuntimeError("backtest failed with returncode={}".format(code))


def run_backtest_on_active_service_host(backtest_args: List[str]) -> None:
    state = get_global_running_state()
    running_hosts = list(state["running_hosts"])

    if len(running_hosts) == 0:
        raise RuntimeError("autotrade service is not running. Start service first, then run backtest.")

    if len(running_hosts) > 1:
        raise RuntimeError(
            "autotrade service is running on multiple hosts: {}. Stop duplicate service first.".format(
                ",".join(running_hosts)
            )
        )

    active_host = str(running_hosts[0]).strip().lower()

    if active_host == LOCAL_HOST:
        run_backtest_local(backtest_args)
        return

    if active_host == HOST_WIN:
        code, out = run_win_control(["backtest-local"] + list(backtest_args))
        print_remote_block("WIN_BACKTEST", code, out)

        if code != 0:
            raise RuntimeError("win backtest failed")

        return

    raise RuntimeError("unsupported active_host for backtest: {}".format(active_host))


def looks_like_time_token(value: Any) -> bool:
    raw = str(value or "").strip()
    return bool(re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", raw))


def normalize_add_symbol_dt_tokens(tokens: List[str], pos: int) -> Tuple[str, int]:
    if pos >= len(tokens):
        raise RuntimeError("datetime value is missing")

    first = str(tokens[pos]).strip()
    if not first:
        raise RuntimeError("datetime value is empty")

    if pos + 1 < len(tokens) and looks_like_time_token(tokens[pos + 1]):
        return first + " " + str(tokens[pos + 1]).strip(), pos + 2

    return first, pos + 1


def make_safe_run_tag(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        value = "add_symbol_" + pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")

    if not value:
        value = "add_symbol_" + pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")

    return value


def utc_now_minute_text() -> str:
    from datetime import datetime, timezone

    dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def normalize_decision_listing_start(decision: Dict[str, Any], fallback_start: str) -> str:
    import pandas as pd

    raw = str(decision.get("listing_first_kline_utc", "") or "").strip()
    if raw:
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        if not pd.isna(ts):
            return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    return str(fallback_start or "").strip()


def parse_add_symbol_local_args(raw_args: List[str]) -> Dict[str, Any]:
    positional: List[str] = []

    execute = False
    continue_on_error = False
    run_tag = ""
    timeout_sec = "7200"
    valid_months = "2"
    valid_days = 60
    min_train_days = "120"

    i = 0
    while i < len(raw_args):
        token = str(raw_args[i]).strip()

        if token == "--execute":
            execute = True
            i += 1
            continue

        if token == "--continue-on-error":
            continue_on_error = True
            i += 1
            continue

        if token in {"--run-tag", "--timeout-sec", "--valid-months", "--min-train-days"}:
            if i + 1 >= len(raw_args):
                raise RuntimeError("{} requires value".format(token))

            value = str(raw_args[i + 1]).strip()

            if token == "--run-tag":
                run_tag = value
            elif token == "--timeout-sec":
                timeout_sec = value
            elif token == "--valid-months":
                valid_months = value
            if token == "--valid-days":
                valid_days = int(raw_args[i + 1])
                i += 2
                continue

            elif token == "--min-train-days":
                min_train_days = value

            i += 2
            continue

        if token.startswith("--"):
            if token == "--valid-days":
                valid_days = int(raw_args[i + 1])
                i += 2
                continue

            raise RuntimeError("unknown add-symbol option: {}".format(token))

        positional.append(token)
        i += 1

    if len(positional) < 2:
        raise RuntimeError(
            "add-symbol-local usage: "
            "add-symbol-local SYMBOL DOWNLOAD_START [DOWNLOAD_END] "
            "[--execute] [--run-tag TAG] [--timeout-sec SEC] "
            "[--valid-months N] [--min-train-days N]"
        )

    symbol = str(positional[0]).strip().upper()
    if not symbol:
        raise RuntimeError("symbol is empty")

    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"

    pos = 1
    download_start, pos = normalize_add_symbol_dt_tokens(positional, pos)

    download_end = ""
    if pos < len(positional):
        download_end, pos = normalize_add_symbol_dt_tokens(positional, pos)

    if pos < len(positional):
        raise RuntimeError("too many positional args for add-symbol-local: {}".format(positional[pos:]))

    run_tag = make_safe_run_tag(run_tag or "add_symbol_{}".format(symbol))

    return {
        "symbol": symbol,
        "download_start": download_start,
        "download_end": download_end,
        "execute": bool(execute),
        "continue_on_error": bool(continue_on_error),
        "run_tag": run_tag,
        "timeout_sec": str(timeout_sec),
        "valid_months": str(valid_months),
        "min_train_days": str(min_train_days),
        "valid_days": valid_days,
    }


def run_control_script_json(
    script_rel: str,
    args: List[str],
    json_out: Path,
    timeout_sec: int,
) -> Dict[str, Any]:
    cmd = [
        get_service_python_executable(),
        "-u",
        script_rel,
    ] + [str(x) for x in args] + ["--json-out", str(json_out)]

    print("-" * 120)
    print("CONTROL_STEP:", script_rel)
    print("CMD:", " ".join(str(x) for x in cmd))
    print("JSON_OUT:", json_out)
    print("-" * 120)

    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=build_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(timeout_sec),
    )

    out = str(p.stdout or "")
    if out.strip():
        print(out.rstrip())

    if int(p.returncode) != 0:
        raise RuntimeError(
            "control step failed rc={} script={}\n{}".format(
                int(p.returncode),
                script_rel,
                out[-8000:],
            )
        )

    if not json_out.exists():
        raise RuntimeError("control step did not write json: {}".format(json_out))

    try:
        return json.loads(json_out.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("failed to read json {}: {}".format(json_out, exc))


def run_control_script_manifest(
    script_rel: str,
    args: List[str],
    timeout_sec: int,
) -> Dict[str, Any]:
    cmd = [
        get_service_python_executable(),
        "-u",
        script_rel,
    ] + [str(x) for x in args]

    print("-" * 120)
    print("CONTROL_STEP:", script_rel)
    print("CMD:", " ".join(str(x) for x in cmd))
    print("-" * 120)

    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=build_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(timeout_sec),
    )

    out = str(p.stdout or "")
    if out.strip():
        print(out.rstrip())

    if int(p.returncode) != 0:
        raise RuntimeError(
            "control step failed rc={} script={}\n{}".format(
                int(p.returncode),
                script_rel,
                out[-8000:],
            )
        )

    try:
        return json.loads(out)
    except Exception:
        return {
            "status": "OK",
            "stdout": out,
        }


def run_add_symbol_local(raw_args: List[str]) -> None:
    cfg = parse_add_symbol_local_args(raw_args)

    symbol = str(cfg["symbol"])
    run_tag = str(cfg["run_tag"])
    execute = bool(cfg["execute"])
    timeout_sec = int(cfg["timeout_sec"])

    run_root = ROOT / "production" / "artifacts" / "add_symbol_runs" / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("ADD_SYMBOL_LOCAL_START")
    print("ROOT:", ROOT)
    print("SYMBOL:", symbol)
    print("DOWNLOAD_START:", cfg["download_start"])
    print("DOWNLOAD_END:", cfg["download_end"] if cfg["download_end"] else "AUTO_NOW")
    print("RUN_TAG:", run_tag)
    print("EXECUTE:", execute)
    print("RUN_ROOT:", run_root)
    print("=" * 120)

    decision_json = run_root / "01_symbol_onboarding_decision.json"
    decision = run_control_script_json(
        "online/new/actions/control/symbol_onboarding_decision.py",
        [
            "--symbol",
            symbol,
            "--mode",
            "add",
            "--run-tag",
            run_tag,
        ],
        decision_json,
        timeout_sec=timeout_sec,
    )

    print("DECISION:", decision.get("decision"))
    print("ALLOWED:", decision.get("allowed"))
    print("NEXT_ACTION:", decision.get("next_action"))
    print("MESSAGE:", decision.get("message"))

    if not bool(decision.get("allowed")):
        if str(decision.get("decision")) == "SYMBOL_ALREADY_EXISTS":
            print("=" * 120)
            print("ADD_SYMBOL_LOCAL_DONE")
            print("STATUS: ALREADY_EXISTS")
            print("NEXT_ACTION: PROPOSE_BACKTEST")
            print("SYMBOL:", symbol)
            print("=" * 120)
            return

        print("=" * 120)
        print("ADD_SYMBOL_LOCAL_DONE")
        print("STATUS: REJECTED")
        print("NEXT_ACTION:", str(decision.get("next_action", "")))
        print("SYMBOL:", symbol)
        print("DECISION:", str(decision.get("decision", "")))
        print("=" * 120)
        return

    window_json = run_root / "02_onboarding_window_plan.json"
    effective_download_start = normalize_decision_listing_start(
        decision=decision,
        fallback_start=str(cfg["download_start"]),
    )
    effective_download_end = utc_now_minute_text()

    print("REQUESTED_DOWNLOAD_START:", str(cfg["download_start"]))
    print("EFFECTIVE_DOWNLOAD_START:", effective_download_start)
    print("EFFECTIVE_DOWNLOAD_END:", effective_download_end)
    print("VALID_DAYS:", str(cfg.get("valid_days", 60)))

    window_args = [
        "--symbol",
        symbol,
        "--download-start",
        effective_download_start,
        "--download-end",
        effective_download_end,
        "--valid-days",
        str(cfg.get("valid_days", 60)),
        "--min-train-days",
        str(cfg["min_train_days"]),
    ]

    window_plan = run_control_script_json(
        "online/new/actions/control/onboarding_window_plan.py",
        window_args,
        window_json,
        timeout_sec=timeout_sec,
    )

    if not bool(window_plan.get("allowed")):
        raise RuntimeError("window plan rejected: {}".format(window_plan.get("errors")))

    windows = window_plan.get("windows") or {}

    plan_json = run_root / "03_offline_pipeline_plan.json"
    offline_plan_args = [
        "--symbols",
        symbol,
        "--mode",
        "new_symbol",
        "--run-tag",
        run_tag,
        "--start",
        str(windows["download_start"]),
        "--end",
        str(windows["download_end"]),
        "--train-end",
        str(windows["train_end"]),
        "--valid-start",
        str(windows["valid_start"]),
        "--valid-end",
        str(windows["valid_end"]),
        "--oos-start",
        str(windows["oos_start"]),
        "--oos-end",
        str(windows["oos_end"]),
        "--db-load-oos",
    ]

    offline_plan = run_control_script_json(
        "online/new/actions/control/offline_pipeline_plan.py",
        offline_plan_args,
        plan_json,
        timeout_sec=timeout_sec,
    )

    enabled_steps = [
        str(step.get("id"))
        for step in offline_plan.get("steps", [])
        if bool(step.get("enabled"))
    ]

    print("OFFLINE_ENABLED_STEPS:", enabled_steps)

    executor_args = [
        "--plan-json",
        str(plan_json),
        "--runs-root",
        str(run_root / "offline_executor_runs"),
        "--timeout-sec",
        str(timeout_sec),
    ]

    if execute:
        executor_args.append("--execute")

    if bool(cfg["continue_on_error"]):
        executor_args.append("--continue-on-error")

    executor_result = run_control_script_manifest(
        "online/new/actions/control/offline_pipeline_executor.py",
        executor_args,
        timeout_sec=timeout_sec + 60,
    )

    print("OFFLINE_EXECUTOR_STATUS:", executor_result.get("status"))

    offline_executor_status = str(executor_result.get("status") or "").upper()
    offline_executor_error = executor_result.get("error")

    if offline_executor_status not in {"SUCCESS", "EXECUTED", "DRY_RUN"} or offline_executor_error is not None:
        raise RuntimeError("offline executor failed: {}".format(executor_result))

    online_oos_json = run_root / "04_online_oos_pipeline_runner.json"
    online_oos_args = [
        "--symbols",
        symbol,
        "--start",
        str(windows["oos_start"]),
        "--end",
        str(windows["oos_end"]),
        "--run-tag",
        run_tag + "_online_oos",
        "--runs-root",
        str(run_root / "online_oos_runs"),
        "--timeout-sec",
        str(timeout_sec),
    ]

    if execute:
        online_oos_args.append("--execute")

    if bool(cfg["continue_on_error"]):
        online_oos_args.append("--continue-on-error")

    online_oos_result = run_control_script_json(
        "online/new/actions/control/online_oos_pipeline_runner.py",
        online_oos_args,
        online_oos_json,
        timeout_sec=timeout_sec + 60,
    )

    print("ONLINE_OOS_STATUS:", online_oos_result.get("status"))

    online_oos_status = str(online_oos_result.get("status") or "").upper()
    online_oos_errors = online_oos_result.get("errors")

    if online_oos_status not in {"SUCCESS", "OK", "DRY_RUN"} or bool(online_oos_errors):
        raise RuntimeError("online OOS runner failed: {}".format(online_oos_result))

    summary = {
        "status": "SUCCESS" if execute else "DRY_RUN",
        "symbol": symbol,
        "run_tag": run_tag,
        "execute": execute,
        "run_root": str(run_root),
        "windows": windows,
        "decision_json": str(decision_json),
        "window_json": str(window_json),
        "plan_json": str(plan_json),
        "offline_executor_status": executor_result.get("status"),
        "online_oos_status": online_oos_result.get("status"),
    }

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 120)
    print("ADD_SYMBOL_LOCAL_DONE")
    print("STATUS:", summary["status"])
    print("SYMBOL:", symbol)
    print("TRAIN_END:", windows.get("train_end"))
    print("OOS_START:", windows.get("oos_start"))
    print("OOS_END:", windows.get("oos_end"))
    print("RUN_ROOT:", run_root)
    print("SUMMARY:", summary_path)
    print("=" * 120)

def parse_history_hours(argv: List[str]) -> int:
    if len(argv) < 3:
        return 20

    raw = str(argv[2]).strip().lower()

    if raw in {HOST_MAC, HOST_WIN, "local"}:
        return 20

    try:
        hours = int(raw)
    except Exception:
        raise RuntimeError("history argument must be hours integer, example: history 20 or history 20 win")

    if hours <= 0:
        raise RuntimeError("history hours must be > 0")

    return hours


def parse_history_host(argv: List[str]) -> str:
    if len(argv) >= 4:
        raw = str(argv[3]).strip().lower()

        if raw in {HOST_MAC, HOST_WIN, "local"}:
            return LOCAL_HOST if raw == "local" else raw

        raise RuntimeError("history host must be mac or win, example: history 20 win")

    if len(argv) >= 3:
        raw = str(argv[2]).strip().lower()

        if raw in {HOST_MAC, HOST_WIN, "local"}:
            return LOCAL_HOST if raw == "local" else raw

    return HOST_WIN


def is_history_go(row: pd.Series) -> bool:
    return (
        float(row.get("gate2_for_side_proba") or 0.0) >= float(config.GATE2_THR)
        and float(row.get("gate4_confidence") or 0.0) >= float(config.GATE4_THR)
        and float(row.get("gate5_1_proba") or 0.0) >= float(config.GATE5_1_THR)
        and float(row.get("gate5_3_proba") or 0.0) >= float(config.GATE5_3_THR)
        and str(row.get("side")).upper() in ("LONG", "SHORT")
    )


def get_history_reject_reason(row: pd.Series) -> str:
    reasons = []

    if float(row.get("gate2_for_side_proba") or 0.0) < float(config.GATE2_THR):
        reasons.append("G2")
    if float(row.get("gate4_confidence") or 0.0) < float(config.GATE4_THR):
        reasons.append("G4")
    if float(row.get("gate5_1_proba") or 0.0) < float(config.GATE5_1_THR):
        reasons.append("G5_1")
    if float(row.get("gate5_3_proba") or 0.0) < float(config.GATE5_3_THR):
        reasons.append("G5_3")

    if not reasons:
        return "OK"

    return "-".join(reasons)


def calc_history_tp_sl(row: pd.Series) -> Dict[str, float]:
    side = str(row.get("side")).upper()
    entry = float(row.get("h4_close") or 0.0)
    atr14 = float(row.get("atr14") or 0.0)

    if side == "LONG":
        tp = entry + float(config.TP_ATR) * atr14
        sl = entry - float(config.SL_ATR) * atr14
    elif side == "SHORT":
        tp = entry - float(config.TP_ATR) * atr14
        sl = entry + float(config.SL_ATR) * atr14
    else:
        tp = 0.0
        sl = 0.0

    return {
        "tp": float(tp),
        "sl": float(sl),
    }
def calc_history_tp_sl_from_trading_signals(row: pd.Series) -> Dict[str, float]:
    side = str(row.get("side")).upper()
    entry = float(row.get("h4_close") or 0.0)
    atr14 = float(row.get("atr14") or 0.0)

    tp_atr = row.get("tp_atr")
    sl_atr = row.get("sl_atr")

    if pd.isna(tp_atr) or tp_atr is None:
        tp_atr = config.TP_ATR

    if pd.isna(sl_atr) or sl_atr is None:
        sl_atr = config.SL_ATR

    tp_atr_f = float(tp_atr)
    sl_atr_f = float(sl_atr)

    if side == "LONG":
        tp = entry + tp_atr_f * atr14
        sl = entry - sl_atr_f * atr14
    elif side == "SHORT":
        tp = entry - tp_atr_f * atr14
        sl = entry + sl_atr_f * atr14
    else:
        tp = 0.0
        sl = 0.0

    return {
        "tp": float(tp),
        "sl": float(sl),
    }

def calc_history_tp_sl_from_position_or_signal(row: pd.Series) -> Dict[str, float]:
    pos_tp = row.get("position_tp_px_plan")
    pos_sl = row.get("position_sl_px_plan")

    if pos_tp is not None and pos_sl is not None:
        if pd.notna(pos_tp) and pd.notna(pos_sl):
            try:
                return {
                    "tp": float(pos_tp),
                    "sl": float(pos_sl),
                }
            except Exception:
                pass

    return calc_history_tp_sl_from_trading_signals(row)


def fmt_pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
        value_f = float(value) * 100.0
        sign = "+" if value_f > 0 else ""
        return "{}{:.4f}%".format(sign, value_f)
    except Exception:
        return ""


def calc_signal_to_actual_entry_slip_pct(side: Any, signal_price: Any, actual_entry_price: Any) -> Optional[float]:
    try:
        signal_f = float(signal_price)
        actual_f = float(actual_entry_price)
    except Exception:
        return None

    if signal_f <= 0:
        return None

    return (actual_f - signal_f) / signal_f


def parse_retrain_candidate_expanded_args(raw_args: List[str]) -> Dict[str, Any]:
    # 1) Разбираем только системный admin-вход для candidate retrain.
    #    Эта команда не используется Telegram и не должна запускаться случайно.
    positional: List[str] = []
    run_tag = ""
    timeout_sec = "7200"
    train_end = ""
    valid_start = ""
    valid_end = ""
    full_candidate_retrain = False
    execute = False
    continue_on_error = False

    i = 0
    while i < len(raw_args):
        token = str(raw_args[i]).strip()

        if token == "--execute":
            # Явное разрешение на запуск executor. Без --full-candidate-retrain ниже всё равно заблокируем.
            execute = True
            i += 1
            continue

        if token == "--continue-on-error":
            # Передаём executor-у только при реальном execute.
            continue_on_error = True
            i += 1
            continue

        if token == "--full-candidate-retrain":
            # Единственный флаг, который включает полный retrain по всем уровням в offline_pipeline_plan.py.
            full_candidate_retrain = True
            i += 1
            continue

        if token in {"--run-tag", "--timeout-sec", "--train-end", "--valid-start", "--valid-end"}:
            if i + 1 >= len(raw_args):
                raise RuntimeError("missing value for {}".format(token))

            value = str(raw_args[i + 1]).strip()

            if token == "--run-tag":
                run_tag = value
            elif token == "--timeout-sec":
                timeout_sec = value
            elif token == "--train-end":
                train_end = value
            elif token == "--valid-start":
                valid_start = value
            elif token == "--valid-end":
                valid_end = value

            i += 2
            continue

        if token.startswith("--"):
            raise RuntimeError("unknown retrain-candidate-expanded option: {}".format(token))

        positional.append(token)
        i += 1

    if len(positional) < 3:
        raise RuntimeError(
            "retrain-candidate-expanded usage: "
            "retrain-candidate-expanded SYMBOLS START END "
            "--train-end TS --valid-start TS --valid-end TS "
            "[--run-tag TAG] [--timeout-sec SEC] "
            "[--full-candidate-retrain] [--execute]"
        )

    symbols = str(positional[0]).strip().upper()
    start = str(positional[1]).strip()
    end = str(positional[2]).strip()

    if not symbols:
        raise RuntimeError("symbols are empty")

    if not start:
        raise RuntimeError("start is empty")

    if not end:
        raise RuntimeError("end is empty")

    if not train_end:
        raise RuntimeError("--train-end is required")

    if not valid_start:
        raise RuntimeError("--valid-start is required")

    if not valid_end:
        raise RuntimeError("--valid-end is required")

    if execute and not full_candidate_retrain:
        # Главная защита: executor нельзя запустить без явного полного retrain-флага.
        raise RuntimeError("--execute is forbidden without --full-candidate-retrain")

    safe_symbols_for_tag = symbols.replace(",", "_").replace(":", "_").replace(" ", "_")
    run_tag = make_safe_run_tag(run_tag or "candidate_retrain_{}".format(safe_symbols_for_tag))

    return {
        "symbols": symbols,
        "start": start,
        "end": end,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "run_tag": run_tag,
        "timeout_sec": str(timeout_sec),
        "full_candidate_retrain": bool(full_candidate_retrain),
        "execute": bool(execute),
        "continue_on_error": bool(continue_on_error),
    }


def run_retrain_candidate_expanded_local(raw_args: List[str]) -> None:
    # 1) Парсим admin-команду и сразу применяем safety-правила.
    cfg = parse_retrain_candidate_expanded_args(raw_args)

    symbols = str(cfg["symbols"])
    run_tag = str(cfg["run_tag"])
    timeout_sec = int(cfg["timeout_sec"])
    full_candidate_retrain = bool(cfg["full_candidate_retrain"])
    execute = bool(cfg["execute"])

    # 2) Все артефакты candidate retrain складываем в отдельный run-root.
    #    Это не production-path и не add_symbol_runs.
    run_root = ROOT / "production" / "artifacts" / "candidate_retrain_runs" / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("RETRAIN_CANDIDATE_EXPANDED_START")
    print("ROOT:", ROOT)
    print("SYMBOLS:", symbols)
    print("START:", cfg["start"])
    print("END:", cfg["end"])
    print("TRAIN_END:", cfg["train_end"])
    print("VALID_START:", cfg["valid_start"])
    print("VALID_END:", cfg["valid_end"])
    print("RUN_TAG:", run_tag)
    print("FULL_CANDIDATE_RETRAIN:", full_candidate_retrain)
    print("EXECUTE:", execute)
    print("RUN_ROOT:", run_root)
    print("=" * 120)

    # 3) Собираем plan через offline_pipeline_plan.py.
    #    Без --full-candidate-retrain plan обязан быть полностью безопасным: enabled_steps == [].
    plan_json = run_root / "01_offline_pipeline_plan.json"
    offline_plan_args = [
        "--symbols",
        symbols,
        "--mode",
        "candidate_retrain",
        "--run-tag",
        run_tag,
        "--start",
        str(cfg["start"]),
        "--end",
        str(cfg["end"]),
        "--train-end",
        str(cfg["train_end"]),
        "--valid-start",
        str(cfg["valid_start"]),
        "--valid-end",
        str(cfg["valid_end"]),
    ]

    if full_candidate_retrain:
        # 4) Явно включаем полный retrain: Gate1/Gate2/Gate3/Gate4/Gate5 candidate artifacts.
        offline_plan_args.append("--full-candidate-retrain")

    offline_plan = run_control_script_json(
        "online/new/actions/control/offline_pipeline_plan.py",
        offline_plan_args,
        plan_json,
        timeout_sec=timeout_sec,
    )

    enabled_steps = [
        str(step.get("id"))
        for step in offline_plan.get("steps", [])
        if bool(step.get("enabled"))
    ]

    print("OFFLINE_ENABLED_STEPS:", enabled_steps)

    if not full_candidate_retrain and enabled_steps:
        # 5) В safe-default режиме запрещаем любые включенные write/train steps.
        raise RuntimeError(
            "unsafe candidate retrain default: enabled steps without --full-candidate-retrain: {}".format(
                enabled_steps
            )
        )

    if execute and not full_candidate_retrain:
        # 6) Дублирующая защита на случай будущих изменений parse-функции.
        raise RuntimeError("execute is forbidden without full candidate retrain")

    executor_result: Dict[str, Any] = {
        "status": "NOT_RUN",
        "reason": "plan_only",
    }

    if execute:
        # 7) Executor запускается только при двух условиях:
        #    --full-candidate-retrain и --execute.
        executor_args = [
            "--plan-json",
            str(plan_json),
            "--runs-root",
            str(run_root / "offline_executor_runs"),
            "--timeout-sec",
            str(timeout_sec),
            "--execute",
        ]

        if bool(cfg["continue_on_error"]):
            executor_args.append("--continue-on-error")

        executor_result = run_control_script_manifest(
            "online/new/actions/control/offline_pipeline_executor.py",
            executor_args,
            timeout_sec=timeout_sec + 60,
        )

        print("OFFLINE_EXECUTOR_STATUS:", executor_result.get("status"))

        offline_executor_status = str(executor_result.get("status") or "").upper()
        offline_executor_error = executor_result.get("error")

        if offline_executor_status not in {"SUCCESS", "EXECUTED", "DRY_RUN"} or offline_executor_error is not None:
            raise RuntimeError("offline executor failed: {}".format(executor_result))
    else:
        # 8) По умолчанию это только построение и аудит плана, без записи файлов пайплайна.
        print("OFFLINE_EXECUTOR_STATUS:", executor_result.get("status"))
        print("OFFLINE_EXECUTOR_REASON:", executor_result.get("reason"))

    summary = {
        "status": "EXECUTED" if execute else "PLAN_ONLY",
        "symbols": symbols,
        "run_tag": run_tag,
        "execute": execute,
        "full_candidate_retrain": full_candidate_retrain,
        "run_root": str(run_root),
        "plan_json": str(plan_json),
        "enabled_steps": enabled_steps,
        "offline_executor_status": executor_result.get("status"),
    }

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 120)
    print("RETRAIN_CANDIDATE_EXPANDED_DONE")
    print("STATUS:", summary["status"])
    print("SYMBOLS:", symbols)
    print("RUN_TAG:", run_tag)
    print("SUMMARY:", summary_path)
    print("=" * 120)

def load_signal_history(hours: int) -> pd.DataFrame:
    ensure_status_position_columns()
    latest_closed_signal_ts = latest_closed_h4_signal_ts_utc()

    sql = """
        SELECT
            s.signal_key,
            s.symbol,
            s.signal_ts,
            s.entry_ts_plan,
            s.side,

            s.pair_model_name,
            s.grid_name,
            s.tp_atr,
            s.sl_atr,
            s.ttl_hours,

            s.h4_close,
            s.atr14,

            s.gate2_proba,
            s.gate4_confidence,
            s.gate5_1_proba,
            s.gate5_3_proba,
            s.signal_strength,

            s.selected,
            s.rejected,
            s.reject_reason,
            s.skipped_reason,
            s.dynamic_symbol_allowed,
            s.dynamic_symbol_reason,
            s.candidate_rank,
            s.selector_version,
            s.updated_at,

            p.trade_id,
            p.status AS position_status,
            p.qty AS position_qty,
            p.entry_px_plan AS position_entry_px_plan,
            p.entry_avg_px AS position_entry_avg_px,
            p.entry_slippage_abs AS position_entry_slippage_abs,
            p.entry_slippage_pct AS position_entry_slippage_pct,
            p.tp_px_plan AS position_tp_px_plan,
            p.sl_px_plan AS position_sl_px_plan,
            p.partial_tp_px_plan AS position_partial_tp_px_plan,
            p.final_tp_px_plan AS position_final_tp_px_plan,
            p.early_stop_px_plan AS position_early_stop_px_plan,
            p.main_sl_px_plan AS position_main_sl_px_plan,
            p.rest_stop_after_partial_px_plan AS position_rest_stop_after_partial_px_plan,
            p.trade_management_mode AS position_trade_management_mode,
            p.early_stop_expires_at AS position_early_stop_expires_at,
            p.partial_tp_handled_at AS position_partial_tp_handled_at,
            p.early_stop_replaced_at AS position_early_stop_replaced_at,
            p.protective_cleanup_done_at AS position_protective_cleanup_done_at,
            p.ws_lifecycle_updated_at AS position_ws_lifecycle_updated_at,
            p.exit_reason AS position_exit_reason,
            p.pnl_usd AS position_pnl_usd,
            p.fee_usd AS position_fee_usd,
            p.created_at AS position_created_at,
            p.updated_at AS position_updated_at,

            (
                SELECT STRING_AGG(DISTINCT f.order_role, ',' ORDER BY f.order_role)
                FROM public.trading_fills f
                WHERE f.trade_id = p.trade_id
            ) AS fill_roles,

            (
                SELECT SUM(f.exec_qty)
                FROM public.trading_fills f
                WHERE f.trade_id = p.trade_id
                  AND f.order_role = 'PARTIAL_TP'
            ) AS fill_partial_tp_qty,

            (
                SELECT SUM(f.exec_qty)
                FROM public.trading_fills f
                WHERE f.trade_id = p.trade_id
                  AND f.order_role = 'FINAL_TP'
            ) AS fill_final_tp_qty,

            (
                SELECT SUM(f.exec_qty)
                FROM public.trading_fills f
                WHERE f.trade_id = p.trade_id
                  AND f.order_role = 'TAKE_PROFIT'
            ) AS fill_take_profit_qty,

            (
                SELECT SUM(f.exec_qty)
                FROM public.trading_fills f
                WHERE f.trade_id = p.trade_id
                  AND f.order_role = 'STOP_LOSS'
            ) AS fill_stop_loss_qty,

            (
                SELECT SUM(f.exec_qty)
                FROM public.trading_fills f
                WHERE f.trade_id = p.trade_id
                  AND f.order_role = 'EARLY_STOP'
            ) AS fill_early_stop_qty,

            (
                SELECT SUM(f.exec_qty)
                FROM public.trading_fills f
                WHERE f.trade_id = p.trade_id
                  AND f.order_role = 'REST_STOP_AFTER_PARTIAL'
            ) AS fill_rest_stop_after_partial_qty,

            (
                SELECT SUM(f.exec_qty)
                FROM public.trading_fills f
                WHERE f.trade_id = p.trade_id
                  AND f.order_role = 'TTL_CLOSE'
            ) AS fill_ttl_close_qty,

            (
                SELECT MAX(f.executed_at)
                FROM public.trading_fills f
                WHERE f.trade_id = p.trade_id
                  AND f.order_role IN (
                    'PARTIAL_TP',
                    'FINAL_TP',
                    'TAKE_PROFIT',
                    'STOP_LOSS',
                    'EARLY_STOP',
                    'REST_STOP_AFTER_PARTIAL',
                    'TTL_CLOSE',
                    'EMERGENCY_CLOSE',
                    'MANUAL_CLOSE'
                  )
            ) AS fill_last_exit_ts

        FROM public.trading_signals s

        LEFT JOIN public.trading_positions p
            ON p.signal_key = s.signal_key

        WHERE s.pair_model_name = %s
          AND s.grid_name = %s
          AND s.signal_ts >= NOW() - (%s::text || ' hours')::interval
          AND s.signal_ts <= %s

        ORDER BY
            s.signal_ts DESC,
            s.selected DESC,
            s.candidate_rank ASC NULLS LAST,
            s.signal_strength DESC NULLS LAST,
            s.gate4_confidence DESC NULLS LAST,
            s.gate2_proba DESC NULLS LAST,
            s.gate5_1_proba DESC NULLS LAST,
            s.gate5_3_proba DESC NULLS LAST,
            s.symbol ASC
    """

    query_hours = int(hours) + 4

    return read_sql(
        sql,
        [
            config.PAIR_MODEL_NAME,
            config.GRID_NAME,
            str(int(query_hours)),
            latest_closed_signal_ts.to_pydatetime(),
        ],
    )


HISTORY_REAL_POSITION_STATUSES = {
    "ENTRY_ORDER_SENT",
    "ENTRY_PARTIALLY_FILLED",
    "ENTRY_FILLED",
    "TP_SL_PLACED",
    "POSITION_OPEN",
    "TTL_CLOSE_SENT",
    "TTL_CLOSE_FAILED",
}


HISTORY_NOT_REAL_POSITION_STATUSES = {
    "",
    "CREATED",
    "DRY_RUN_CREATED",
    "DRY_RUN_ENTRY_PLANNED",
    "DRY_RUN_NOT_SENT",
    "ENTRY_FAILED",
    "ENTRY_REJECTED",
    "CANCELLED",
    "FAILED",
}


def history_row_has_real_position(row: pd.Series) -> bool:
    trade_id = row.get("trade_id")

    if trade_id is None or pd.isna(trade_id):
        return False

    status = str(row.get("position_status") or "").strip().upper()

    if status in HISTORY_NOT_REAL_POSITION_STATUSES:
        return False

    if status.startswith("POSITION_CLOSED"):
        return True

    if status in HISTORY_REAL_POSITION_STATUSES:
        return True

    entry_actual = row.get("position_entry_avg_px")
    if entry_actual is not None and pd.notna(entry_actual):
        try:
            return float(entry_actual) > 0.0
        except Exception:
            return False

    return False


def get_history_reason_from_row(row: pd.Series) -> str:
    reason = str(row.get("reject_reason") or "")

    if not reason or reason.lower() == "nan":
        reason = str(row.get("skipped_reason") or "")

    if not reason or reason.lower() == "nan":
        reason = get_history_reject_reason(row)

    if not reason or reason.lower() == "nan":
        reason = "NO_SELECTED_SIGNAL"

    reason = str(reason).strip()

    dynamic_allowed = row.get("dynamic_symbol_allowed")
    dynamic_reason = str(row.get("dynamic_symbol_reason") or "").strip().upper()

    whitelist_blocked = False

    if dynamic_allowed is not None and not pd.isna(dynamic_allowed):
        try:
            whitelist_blocked = bool(dynamic_allowed) is False
        except Exception:
            whitelist_blocked = False

    if "WHITE" in dynamic_reason or "WHITELIST" in dynamic_reason or "NO_WHITELIST" in dynamic_reason:
        whitelist_blocked = True

    if whitelist_blocked:
        if reason in {"OK", "NO_SELECTED_SIGNAL"}:
            reason = "NO_WHITELIST"
        elif "NO_WHITELIST" not in reason:
            reason = reason + "-NO_WHITELIST"

    return reason


def choose_history_row_for_group(group: pd.DataFrame) -> Tuple[pd.Series, str, str]:
    g = group.copy()

    real_position_rows = g[g.apply(history_row_has_real_position, axis=1)].copy()

    if not real_position_rows.empty:
        chosen = real_position_rows.sort_values(
            ["position_updated_at", "position_created_at", "candidate_rank", "signal_strength", "symbol"],
            ascending=[False, False, True, False, True],
            na_position="last",
        ).iloc[0]

        return chosen, "GO", "OK"

    selected_rows = g[g["selected"] == True].copy()

    if not selected_rows.empty:
        chosen = selected_rows.sort_values(
            ["candidate_rank", "signal_strength", "symbol"],
            ascending=[True, False, True],
            na_position="last",
        ).iloc[0]

        return chosen, "GO", "OK"

    chosen = g.sort_values(
        ["candidate_rank", "signal_strength", "symbol"],
        ascending=[True, False, True],
        na_position="last",
    ).iloc[0]

    return chosen, "SKIP", get_history_reason_from_row(chosen)



def history_qty_positive(row: pd.Series, col: str) -> bool:
    try:
        value = row.get(col)
        if value is None or pd.isna(value):
            return False
        return float(value) > 0.0
    except Exception:
        return False


def derive_history_execution_kind(row: pd.Series) -> str:
    status = str(row.get("position_status") or "").upper()
    roles = str(row.get("fill_roles") or "").upper()

    if history_qty_positive(row, "fill_partial_tp_qty") and history_qty_positive(row, "fill_final_tp_qty"):
        return "PARTIAL+FINAL_TP"

    if history_qty_positive(row, "fill_partial_tp_qty") and history_qty_positive(row, "fill_rest_stop_after_partial_qty"):
        return "PARTIAL+REST_STOP"

    if history_qty_positive(row, "fill_partial_tp_qty") and history_qty_positive(row, "fill_stop_loss_qty"):
        return "PARTIAL+MAIN_SL"

    if history_qty_positive(row, "fill_final_tp_qty"):
        return "FINAL_TP"

    if history_qty_positive(row, "fill_take_profit_qty"):
        return "TAKE_PROFIT"

    if history_qty_positive(row, "fill_stop_loss_qty"):
        return "STOP_LOSS"

    if history_qty_positive(row, "fill_early_stop_qty"):
        return "EARLY_STOP"

    if history_qty_positive(row, "fill_rest_stop_after_partial_qty"):
        return "REST_STOP"

    if history_qty_positive(row, "fill_ttl_close_qty"):
        return "TTL_CLOSE"

    if "MANUAL_CLOSE" in roles:
        return "MANUAL_CLOSE"

    if "EMERGENCY_CLOSE" in roles:
        return "EMERGENCY_CLOSE"

    if status.startswith("POSITION_CLOSED"):
        reason = str(row.get("position_exit_reason") or "").upper()
        return reason if reason else status

    if status in HISTORY_REAL_POSITION_STATUSES:
        return "OPEN"

    return ""


def print_history(argv: List[str]) -> None:
    hours = parse_history_hours(argv)
    host = parse_history_host(argv)

    if host == HOST_WIN and LOCAL_HOST != HOST_WIN:
        code, out = run_win_control(["history-local", str(int(hours))])
        print_remote_block("WIN_HISTORY", code, out)

        if code != 0:
            raise RuntimeError("win history failed")

        return

    df = load_signal_history(hours)
    expected_signal_ts_list = build_expected_history_signal_ts(hours)

    if df.empty:
        out = pd.DataFrame()
    else:
        out = df.copy()
        out["signal_ts"] = pd.to_datetime(out["signal_ts"], utc=True, errors="coerce")

        for col in [
            "h4_close",
            "atr14",
            "gate2_proba",
            "gate4_confidence",
            "gate5_1_proba",
            "gate5_3_proba",
            "signal_strength",
            "candidate_rank",
            "position_qty",
            "position_entry_px_plan",
            "position_entry_avg_px",
            "position_entry_slippage_abs",
            "position_entry_slippage_pct",
            "position_tp_px_plan",
            "position_sl_px_plan",
            "position_partial_tp_px_plan",
            "position_final_tp_px_plan",
            "position_early_stop_px_plan",
            "position_main_sl_px_plan",
            "position_rest_stop_after_partial_px_plan",
            "position_pnl_usd",
            "position_fee_usd",
            "fill_partial_tp_qty",
            "fill_final_tp_qty",
            "fill_take_profit_qty",
            "fill_stop_loss_qty",
            "fill_early_stop_qty",
            "fill_rest_stop_after_partial_qty",
            "fill_ttl_close_qty",
        ]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

    groups_by_signal_ts: Dict[pd.Timestamp, pd.DataFrame] = {}

    if not out.empty:
        for signal_ts, group in out.groupby("signal_ts", sort=False):
            ts = pd.to_datetime(signal_ts, utc=True, errors="coerce")

            if pd.isna(ts):
                continue

            groups_by_signal_ts[pd.Timestamp(ts)] = group.copy()

    rows = []

    for expected_signal_ts in expected_signal_ts_list:
        ts_key = pd.Timestamp(expected_signal_ts)
        group = groups_by_signal_ts.get(ts_key)

        if group is None or group.empty:
            close_ts = ts_key + pd.Timedelta(hours=4)

            rows.append(
                {
                    "close": close_ts.strftime("%Y-%m-%d %H:%M"),
                    "signal": ts_key.strftime("%Y-%m-%d %H:%M"),
                    "symbol": "ALL",
                    "decision": "SKIP",
                    "side": "-",
                    "entry_signal": "",
                    "entry_plan": "",
                    "entry_actual": "",
                    "slip_pct": "",
                    "tp": "",
                    "sl": "",
                    "pos_status": "",
                    "exec": "",
                    "reason": "NO_CANDIDATES",
                    "rank": "",
                }
            )
            continue

        g = group.copy()

        chosen, decision, reason = choose_history_row_for_group(g)

        levels = calc_history_tp_sl_from_position_or_signal(chosen)

        ts = pd.to_datetime(chosen["signal_ts"], utc=True, errors="coerce")
        close_ts = ts + pd.Timedelta(hours=4) if pd.notna(ts) else pd.NaT

        entry_plan = chosen.get("position_entry_px_plan")
        entry_actual = chosen.get("position_entry_avg_px")
        entry_signal = chosen.get("h4_close")

        if entry_plan is None or pd.isna(entry_plan):
            entry_plan = entry_signal

        if entry_actual is None or pd.isna(entry_actual):
            entry_actual = ""

        slip_pct = calc_signal_to_actual_entry_slip_pct(
            side=chosen.get("side"),
            signal_price=entry_signal,
            actual_entry_price=entry_actual,
        )

        rows.append(
            {
                "close": "" if pd.isna(close_ts) else close_ts.strftime("%Y-%m-%d %H:%M"),
                "signal": "" if pd.isna(ts) else ts.strftime("%Y-%m-%d %H:%M"),
                "symbol": str(chosen.get("symbol") or "").upper(),
                "decision": decision,
                "side": str(chosen.get("side") or "").upper(),
                "entry_signal": fmt_float(entry_signal),
                "entry_plan": fmt_float(entry_plan),
                "entry_actual": "" if entry_actual == "" else fmt_float(entry_actual),
                "slip_pct": fmt_pct(slip_pct),
                "tp": fmt_float(levels["tp"]),
                "sl": fmt_float(levels["sl"]),
                "pos_status": str(chosen.get("position_status") or ""),
                "exec": derive_history_execution_kind(chosen),
                "reason": reason,
                "rank": "" if pd.isna(chosen.get("candidate_rank")) else int(chosen.get("candidate_rank")),
            }
        )

    result = pd.DataFrame(rows)

    history_output_format = os.environ.get("IMB_HISTORY_OUTPUT_FORMAT", "table").strip().lower()

    if history_output_format == "json":
        print("HISTORY_JSON_BEGIN")
        print(result.to_json(orient="records", force_ascii=False))
        print("HISTORY_JSON_END")
        return

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)

    print(result.to_string(index=False))

def normalize_aliases(argv: List[str]) -> List[str]:
    if not argv:
        return ["status"]

    cmd = str(argv[0]).strip().lower()

    alias_map = {
        "start_win": ["start", "win"],
        "start-win": ["start", "win"],
        "startwin": ["start", "win"],
        "start_mac": ["start", "mac"],
        "start-mac": ["start", "mac"],
        "startmac": ["start", "mac"],

        "stop_win": ["stop", "win"],
        "stop-win": ["stop", "win"],
        "stopwin": ["stop", "win"],
        "stop_mac": ["stop", "mac"],
        "stop-mac": ["stop", "mac"],
        "stopmac": ["stop", "mac"],

        "status_win": ["status", "win"],
        "status-win": ["status", "win"],
        "statuswin": ["status", "win"],
        "status_mac": ["status", "mac"],
        "status-mac": ["status", "mac"],
        "statusmac": ["status", "mac"],

        "history_win": ["history", "win"],
        "history-win": ["history", "win"],
        "historywin": ["history", "win"],
        "history_mac": ["history", "mac"],
        "history-mac": ["history", "mac"],
        "historymac": ["history", "mac"],
        "position_win": ["position", "win"],
        "position-win": ["position", "win"],
        "positionwin": ["position", "win"],
        "position_mac": ["position", "mac"],
        "position-mac": ["position", "mac"],
        "positionmac": ["position", "mac"],
        "pos_win": ["position", "win"],
        "pos-win": ["position", "win"],
        "poswin": ["position", "win"],
        "pos_mac": ["position", "mac"],
        "pos-mac": ["position", "mac"],
        "posmac": ["position", "mac"],
        "backtest_win": ["backtest", "win"],
        "backtest-win": ["backtest", "win"],
        "backtestwin": ["backtest", "win"],
        "backtest_mac": ["backtest", "mac"],
        "backtest-mac": ["backtest", "mac"],
        "backtestmac": ["backtest", "mac"],
        "retrain_candidate_expanded": ["retrain-candidate-expanded"],
        "retrain-candidate-expanded_win": ["retrain-candidate-expanded", "win"],
        "retrain_candidate_expanded_win": ["retrain-candidate-expanded", "win"],
    }

    if cmd in alias_map:
        return alias_map[cmd] + argv[1:]

    return argv


def main() -> None:
    args = normalize_aliases(sys.argv[1:])

    cmd = "status"
    if len(args) >= 1:
        cmd = str(args[0]).strip().lower()

    full_argv = [sys.argv[0]] + args

    if cmd == "add-symbol-local":
        run_add_symbol_local(args[1:])
        return

    if cmd == "add-symbol":
        if LOCAL_HOST == HOST_WIN:
            run_add_symbol_local(args[1:])
            return

        code, out = run_win_control(["add-symbol-local"] + list(args[1:]))
        print_remote_block("WIN_ADD_SYMBOL", code, out)

        if code != 0:
            raise RuntimeError("win add-symbol failed")

        return

    if cmd == "retrain-candidate-expanded-local":
        run_retrain_candidate_expanded_local(args[1:])
        return

    if cmd == "retrain-candidate-expanded":
        if LOCAL_HOST == HOST_WIN:
            run_retrain_candidate_expanded_local(args[1:])
            return

        code, out = run_win_control(["retrain-candidate-expanded-local"] + list(args[1:]))
        print_remote_block("WIN_RETRAIN_CANDIDATE_EXPANDED", code, out)

        if code != 0:
            raise RuntimeError("win retrain-candidate-expanded failed")

        return

    if cmd == "status-local":
        print_status_local()
        return

    if cmd == "history-local":
        print_history(full_argv)
        return

    if cmd == "position-local":
        print_position(full_argv)
        return

    if cmd == "backtest-local":
        run_backtest_local(args[1:])
        return

    if cmd == "start-local":
        start_local_service()
        return

    if cmd == "stop-local":
        stop_local_service()
        return

    if cmd == "status":
        code, out = run_win_control(["status-local"])
        print_remote_block("WIN_STATUS", code, out)

        if code != 0:
            raise RuntimeError("win status failed")

        return

    if cmd == "history":
        print_history(full_argv)
        return

    if cmd in {"position", "pos"}:
        print_position(full_argv)
        return

    if cmd == "backtest":
        if len(args) >= 2:
            host = str(args[1]).strip().lower()

            if host == HOST_MAC:
                backtest_args = collect_backtest_args_interactive()

                if LOCAL_HOST != HOST_MAC:
                    raise RuntimeError("backtest mac must be executed from mac/local launcher")

                run_backtest_local(backtest_args)
                return

            if host == HOST_WIN and len(args) > 2:
                code, out = run_win_control(["backtest-local"] + list(args[2:]))
                print_remote_block("WIN_BACKTEST", code, out)

                if code != 0:
                    raise RuntimeError("win backtest failed")

                return

            if host == HOST_WIN:
                backtest_args = collect_backtest_args_interactive()
                code, out = run_win_control(["backtest-local"] + list(backtest_args))
                print_remote_block("WIN_BACKTEST", code, out)

                if code != 0:
                    raise RuntimeError("win backtest failed")

                return

            raise RuntimeError("backtest host must be mac or win")

        backtest_args = collect_backtest_args_interactive()
        run_backtest_on_active_service_host(backtest_args)
        return

    if cmd == "start":
        start_host(HOST_WIN)
        return

    if cmd == "stop":
        stop_host(HOST_WIN)
        return

    raise RuntimeError(
        "unknown command: {}. Available: status, history, history 20, position ENAUSDT, position ENAUSDT 3, backtest, add-symbol, retrain-candidate-expanded, start, stop".format(
            cmd
        )
    )


if __name__ == "__main__":
    main()
