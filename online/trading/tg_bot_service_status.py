from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from online.trading import config

#python -m online.trading.tg_bot_service_status start win
#python -m online.trading.tg_bot_service_status status win
#python -m online.trading.tg_bot_service_status stop win
ROOT = config.ROOT
ENV_FILE = ROOT / ".env"

LOCK_PATH = ROOT / "online" / "_state_tg_control_bot_service.lock"
PID_PATH = ROOT / "tg_control_bot.pid"
LOG_LINK = ROOT / "logs" / "tg_control_bot_latest.log"

HOST_MAC = "mac"
HOST_WIN = "win"

SERVICE_MODULE = "online.trading.tg_control_bot"
CONTROL_MODULE = "online.trading.tg_bot_service_status"


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

WIN_PROJECT_ROOT = Path(
    os.environ.get("IMB_WIN_PROJECT_ROOT", r"C:\Projects\ImbalanceSearcher")
)

WIN_PYTHON = Path(
    os.environ.get(
        "IMB_WIN_PYTHON",
        str(WIN_PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
    )
)

WIN_SSH_HOST = (
    os.environ.get("IMB_WIN_SSH_HOST", "")
    or os.environ.get("SSH", "")
).strip()

WINPY_CMD = os.environ.get("IMB_WINPY_CMD", "winpy").strip()

STOP_TIMEOUT_SECONDS = int(os.environ.get("TG_CONTROL_STOP_TIMEOUT_SECONDS", "20"))
START_WAIT_SECONDS = float(os.environ.get("TG_CONTROL_START_WAIT_SECONDS", "5"))


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


def build_child_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(load_env_file(ENV_FILE))

    env["IMB_PROJECT_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["IMB_LOCAL_HOST"] = LOCAL_HOST

    if "IMB_TRADING_DRY_RUN" not in env:
        env["IMB_TRADING_DRY_RUN"] = "0"

    old_warnings = env.get("PYTHONWARNINGS", "").strip()
    extra_warning = "ignore:pandas only supports SQLAlchemy connectable:UserWarning"
    if old_warnings:
        env["PYTHONWARNINGS"] = old_warnings + "," + extra_warning
    else:
        env["PYTHONWARNINGS"] = extra_warning

    return env


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
            code, _ = run_command_capture(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "if (Get-Process -Id {} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}".format(pid_i),
                ]
            )
            return int(code) == 0
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


def find_raw_local_tg_processes() -> List[Dict[str, Any]]:
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
                    "$_.CommandLine -like '*-m online.trading.tg_control_bot*' "
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
            if "tg_bot_service_status" in cmdline_l:
                continue
            if "-m online.trading.tg_control_bot" not in cmdline_l:
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

        if "-m online.trading.tg_control_bot" not in line_l:
            continue
        if "grep" in line_l:
            continue
        if "tg_bot_service_status" in line_l:
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


def find_local_tg_processes() -> List[Dict[str, Any]]:
    raw_processes = find_raw_local_tg_processes()

    if os.name != "nt":
        return raw_processes

    result: List[Dict[str, Any]] = []

    for proc in raw_processes:
        try:
            pid = int(proc.get("pid"))
        except Exception:
            continue

        exe_l = str(proc.get("exe_path") or "").lower()

        has_tg_child = False
        for child in raw_processes:
            try:
                child_ppid = int(child.get("ppid"))
            except Exception:
                continue

            if child_ppid == pid:
                has_tg_child = True
                break

        if "\\.venv\\scripts\\python.exe" in exe_l and has_tg_child:
            continue

        result.append(proc)

    return result


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

    processes = find_local_tg_processes()

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
            "multiple tg control bot processes found on this host:\n{}".format(
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


def write_lock(pid: int) -> None:
    payload = {
        "pid": int(pid),
        "created_at_utc": str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
        "service": "tg_control_bot",
        "root": str(ROOT),
    }

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(str(payload), encoding="utf-8")
    PID_PATH.write_text(str(int(pid)), encoding="utf-8")


def assert_no_local_tg_processes() -> None:
    processes = find_local_tg_processes()

    if not processes:
        return

    lines = []
    for p in processes:
        lines.append("pid={} cmd={}".format(p.get("pid"), p.get("cmdline")))

    raise RuntimeError(
        "tg control bot already exists on this host:\n{}".format(
            "\n".join(lines)
        )
    )


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
    return int(p.pid)


def start_local_service_posix(env: Dict[str, str]) -> int:
    LOG_LINK.parent.mkdir(parents=True, exist_ok=True)

    log_f = open(str(LOG_LINK), "ab")

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
    return int(p.pid)


def wait_for_independent_process(timeout_seconds: float) -> Optional[int]:
    deadline = time.time() + float(timeout_seconds)

    while time.time() < deadline:
        processes = find_local_tg_processes()
        if len(processes) == 1:
            try:
                return int(processes[0]["pid"])
            except Exception:
                return None

        time.sleep(0.25)

    processes = find_local_tg_processes()
    if len(processes) == 1:
        try:
            return int(processes[0]["pid"])
        except Exception:
            return None

    return None


def start_local_service() -> None:
    assert_service_python_exists()
    cleanup_stale_runtime_files()

    raw_before = find_raw_local_tg_processes()
    independent_before = find_local_tg_processes()

    if raw_before or independent_before:
        lines = []

        for proc in raw_before:
            lines.append(
                "raw pid={} ppid={} exe={} cmd={}".format(
                    proc.get("pid"),
                    proc.get("ppid"),
                    proc.get("exe_path"),
                    proc.get("cmdline"),
                )
            )

        if not lines:
            for proc in independent_before:
                lines.append(
                    "pid={} cmd={}".format(
                        proc.get("pid"),
                        proc.get("cmdline"),
                    )
                )

        raise RuntimeError(
            "tg control bot already exists on this host. Stop it first:\n{}".format(
                "\n".join(lines)
            )
        )

    env = build_child_env()

    if os.name == "nt":
        starter_pid = start_local_service_windows(env)
    else:
        starter_pid = start_local_service_posix(env)

    service_pid = wait_for_independent_process(START_WAIT_SECONDS)

    raw_after = find_raw_local_tg_processes()
    independent_after = find_local_tg_processes()

    if len(independent_after) != 1:
        lines = []
        for proc in raw_after:
            lines.append(
                "raw pid={} ppid={} exe={} cmd={}".format(
                    proc.get("pid"),
                    proc.get("ppid"),
                    proc.get("exe_path"),
                    proc.get("cmdline"),
                )
            )

        raise RuntimeError(
            "bad tg control bot start: expected exactly 1 independent process, got {}. Raw processes:\n{}".format(
                len(independent_after),
                "\n".join(lines),
            )
        )

    service_pid = int(independent_after[0]["pid"])
    write_lock(service_pid)

    print("TG_CONTROL_BOT_STARTED_LOCAL")
    print("host:", LOCAL_HOST)
    print("starter_pid:", starter_pid)
    print("service_pid:", service_pid)
    print("module:", SERVICE_MODULE)
    print("service_python:", get_service_python_executable())
    print("raw_process_count:", len(raw_after))
    print("independent_process_count:", len(independent_after))
    print("log:", LOG_LINK)


def stop_local_service() -> None:
    cleanup_stale_runtime_files()

    target_pids: List[int] = []

    for proc in find_raw_local_tg_processes():
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

        print("TG_CONTROL_BOT_NOT_RUNNING")
        return

    print("TG_CONTROL_BOT_STOP_REQUEST")
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
                os.kill(int(pid), 15)
            except ProcessLookupError:
                pass
            except Exception as e:
                raise RuntimeError("failed to stop local pid={}: {}".format(pid, e))

    deadline = time.time() + float(STOP_TIMEOUT_SECONDS)

    while time.time() < deadline:
        leftovers = find_raw_local_tg_processes()
        if not leftovers:
            break
        time.sleep(0.25)

    leftovers = find_raw_local_tg_processes()

    if leftovers:
        lines = []
        for proc in leftovers:
            lines.append("pid={} cmd={}".format(proc.get("pid"), proc.get("cmdline")))

        raise RuntimeError(
            "tg control bot process still exists after stop:\n{}".format(
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

    print("TG_CONTROL_BOT_STOPPED_LOCAL")
    print("pids:", ",".join(str(x) for x in target_pids))


def status_local_service() -> None:
    cleanup_stale_runtime_files()

    raw = find_raw_local_tg_processes()
    independent = find_local_tg_processes()

    pid = None
    running = False

    try:
        pid = get_service_pid()
        running = is_process_alive(pid)
    except RuntimeError as e:
        print("BAD")
        print("host:", LOCAL_HOST)
        print("service_running:", False)
        print("pid:", None)
        print("error:", str(e))
        print("raw_process_count:", len(raw))
        print("independent_process_count:", len(independent))
        return

    status = "OK" if running and len(independent) == 1 else "WARNING"

    print(status)
    print("host:", LOCAL_HOST)
    print("service_running:", bool(running))
    print("pid:", pid)
    print("module:", SERVICE_MODULE)
    print("raw_process_count:", len(raw))
    print("independent_process_count:", len(independent))
    print("log:", LOG_LINK)

    if raw:
        print("raw_processes:")
        for p in raw:
            print("pid={} ppid={} exe={} cmd={}".format(
                p.get("pid"),
                p.get("ppid"),
                p.get("exe_path"),
                p.get("cmdline"),
            ))


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
env["PYTHONPATH"] = str(ROOT)
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONIOENCODING"] = "utf-8"
env["IMB_LOCAL_HOST"] = "win"
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


def is_running_by_status_output(output: str) -> bool:
    for line in str(output).splitlines():
        line = line.strip()

        if line.startswith("service_running:"):
            return line.split(":", 1)[1].strip().lower() == "true"

    return False


def start_host(host: str) -> None:
    host = str(host).strip().lower()

    if host not in {HOST_MAC, HOST_WIN}:
        raise RuntimeError("start requires host: mac or win")

    if host == HOST_MAC:
        if LOCAL_HOST != HOST_MAC:
            raise RuntimeError("this launcher is not running on mac; LOCAL_HOST={}".format(LOCAL_HOST))
        start_local_service()
        return

    if LOCAL_HOST == HOST_WIN:
        start_local_service()
        return

    code, out = run_win_control(["start-local"])
    print_remote_block("WIN_TG_START", code, out)

    if code != 0:
        raise RuntimeError("win tg start failed")


def stop_host(host: str) -> None:
    host = str(host).strip().lower()

    if host not in {HOST_MAC, HOST_WIN}:
        raise RuntimeError("stop requires host: mac or win")

    if host == HOST_MAC:
        if LOCAL_HOST != HOST_MAC:
            raise RuntimeError("this launcher is not running on mac; LOCAL_HOST={}".format(LOCAL_HOST))
        stop_local_service()
        return

    if LOCAL_HOST == HOST_WIN:
        stop_local_service()
        return

    code, out = run_win_control(["stop-local"])
    print_remote_block("WIN_TG_STOP", code, out)

    if code != 0:
        raise RuntimeError("win tg stop failed")


def status_host(host: str) -> None:
    host = str(host).strip().lower()

    if host not in {HOST_MAC, HOST_WIN}:
        raise RuntimeError("status requires host: mac or win")

    if host == HOST_MAC:
        if LOCAL_HOST != HOST_MAC:
            raise RuntimeError("this launcher is not running on mac; LOCAL_HOST={}".format(LOCAL_HOST))
        status_local_service()
        return

    if LOCAL_HOST == HOST_WIN:
        status_local_service()
        return

    code, out = run_win_control(["status-local"])
    print_remote_block("WIN_TG_STATUS", code, out)

    if code != 0:
        raise RuntimeError("win tg status failed")


def normalize_aliases(argv: List[str]) -> List[str]:
    if not argv:
        return ["status", HOST_WIN]

    cmd = str(argv[0]).strip().lower()

    alias_map = {
        "run": ["start"],
        "tg_run": ["start"],
        "tg-run": ["start"],
        "tg_start": ["start"],
        "tg-start": ["start"],
        "tg_status": ["status"],
        "tg-status": ["status"],
        "tg_stop": ["stop"],
        "tg-stop": ["stop"],
    }

    if cmd in alias_map:
        return alias_map[cmd] + argv[1:]

    return argv


def main() -> None:
    args = normalize_aliases(sys.argv[1:])

    cmd = str(args[0]).strip().lower() if args else "status"
    host = str(args[1]).strip().lower() if len(args) >= 2 else HOST_WIN

    if cmd == "start-local":
        start_local_service()
        return

    if cmd == "stop-local":
        stop_local_service()
        return

    if cmd == "status-local":
        status_local_service()
        return

    if cmd == "start":
        start_host(host)
        return

    if cmd == "stop":
        stop_host(host)
        return

    if cmd == "status":
        status_host(host)
        return

    raise RuntimeError(
        "unknown command: {}. Available: start, stop, status, start win, stop win, status win, start mac, stop mac, status mac".format(cmd)
    )


if __name__ == "__main__":
    main()
