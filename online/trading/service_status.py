from __future__ import annotations

import ast
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from online.trading import config
from online.trading.bybit_client import BybitClient
from online.trading.db import read_sql


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
    os.environ.setdefault(env_key, env_value)


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
    env["IMB_DB_DSN"] = env["IMB_DB_DSN"]
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
env["IMB_DB_DSN"] = env["IMB_DB_DSN"]
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
        try:
            PID_PATH.unlink(missing_ok=True)
        except TypeError:
            if PID_PATH.exists():
                PID_PATH.unlink()

        try:
            LOCK_PATH.unlink(missing_ok=True)
        except TypeError:
            if LOCK_PATH.exists():
                LOCK_PATH.unlink()

        print("LOCAL_NOT_RUNNING")
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

    try:
        PID_PATH.unlink(missing_ok=True)
    except TypeError:
        if PID_PATH.exists():
            PID_PATH.unlink()

    try:
        LOCK_PATH.unlink(missing_ok=True)
    except TypeError:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()

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


def print_status() -> None:
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
    print("log:", LOG_LINK)
    print("next_h4_close_utc:", close_ts)
    print("time_to_next_h4_close:", fmt_left(left))
    print("capital_usdt:", fmt_money(capital_usdt))
    print("trade_capital_usdt:", fmt_money(trade_capital_usdt))
    print("chulan_enabled:", chulan_enabled)
    print("chulan_base_capital_usdt:", fmt_money(chulan_base_capital_usdt))
    print("current_position_pnl_usdt:", fmt_money(current_pnl))
    print("open_positions_count:", open_positions_count)

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
        return "{:.4f}%".format(float(value) * 100.0)
    except Exception:
        return ""

def load_signal_history(hours: int) -> pd.DataFrame:
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
            p.exit_reason AS position_exit_reason,
            p.pnl_usd AS position_pnl_usd,
            p.fee_usd AS position_fee_usd,
            p.created_at AS position_created_at,
            p.updated_at AS position_updated_at

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

    return read_sql(
        sql,
        [
            config.PAIR_MODEL_NAME,
            config.GRID_NAME,
            str(int(hours)),
            latest_closed_signal_ts.to_pydatetime(),
        ],
    )


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

    if df.empty:
        print("EMPTY")
        return

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
        "position_pnl_usd",
        "position_fee_usd",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    rows = []

    for _, group in out.groupby("signal_ts", sort=False):
        g = group.copy()

        selected_rows = g[g["selected"] == True].copy()

        if not selected_rows.empty:
            chosen = selected_rows.sort_values(
                ["candidate_rank", "signal_strength", "symbol"],
                ascending=[True, False, True],
                na_position="last",
            ).iloc[0]
            decision = "GO"
            reason = "OK"
        else:
            chosen = g.sort_values(
                ["candidate_rank", "signal_strength", "symbol"],
                ascending=[True, False, True],
                na_position="last",
            ).iloc[0]
            decision = "SKIP"

            reason = str(chosen.get("reject_reason") or "")
            if not reason or reason.lower() == "nan":
                reason = str(chosen.get("skipped_reason") or "")
            if not reason or reason.lower() == "nan":
                reason = "NO_SELECTED_SIGNAL"

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

        slip_pct = chosen.get("position_entry_slippage_pct")

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
        "backtest_win": ["backtest", "win"],
        "backtest-win": ["backtest", "win"],
        "backtestwin": ["backtest", "win"],
        "backtest_mac": ["backtest", "mac"],
        "backtest-mac": ["backtest", "mac"],
        "backtestmac": ["backtest", "mac"],
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

    if cmd == "status-local":
        print_status_local()
        return

    if cmd == "history-local":
        print_history(full_argv)
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
        host = HOST_WIN

        if len(args) >= 2:
            host = str(args[1]).strip().lower()

        if host == HOST_MAC:
            if LOCAL_HOST == HOST_MAC:
                print_status_local()
            else:
                raise RuntimeError("status mac must be executed from mac/local launcher")
            return

        if host == HOST_WIN:
            code, out = run_win_control(["status-local"])
            print_remote_block("WIN_STATUS", code, out)

            if code != 0:
                raise RuntimeError("win status failed")

            return

        raise RuntimeError("status host must be mac or win")

    if cmd == "history":
        print_history(full_argv)
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
        if len(args) >= 2:
            host = str(args[1]).strip().lower()
        else:
            host = HOST_WIN

        start_host(host)
        return

    if cmd == "stop":
        if len(args) >= 2:
            host = str(args[1]).strip().lower()
        else:
            host = HOST_WIN

        stop_host(host)
        return

    raise RuntimeError(
        "unknown command: {}. Available: status, status win, status mac, history, history 20, history 20 win, backtest, backtest win, backtest mac, start, start win, start mac, stop, stop win, stop mac".format(
            cmd)
    )


if __name__ == "__main__":
    main()
