import os
import sys
import multiprocessing
from typing import Optional


def _safe_int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        x = int(v)
        return x if x > 0 else int(default)
    except Exception:
        return int(default)


def _set_default_thread_env(thread_count: int) -> None:
    env_names = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]

    for name in env_names:
        if not os.environ.get(name):
            os.environ[name] = str(thread_count)


def _boost_windows_process() -> None:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        GetCurrentProcess = kernel32.GetCurrentProcess
        GetCurrentProcess.restype = wintypes.HANDLE

        SetPriorityClass = kernel32.SetPriorityClass
        SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        SetPriorityClass.restype = wintypes.BOOL

        SetProcessAffinityMask = kernel32.SetProcessAffinityMask
        SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        SetProcessAffinityMask.restype = wintypes.BOOL

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        NORMAL_PRIORITY_CLASS = 0x00000020
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        HIGH_PRIORITY_CLASS = 0x00000080

        proc = GetCurrentProcess()

        priority_name = os.environ.get("IMB_WIN_PRIORITY", "above_normal").strip().lower()
        priority_map = {
            "below_normal": BELOW_NORMAL_PRIORITY_CLASS,
            "normal": NORMAL_PRIORITY_CLASS,
            "above_normal": ABOVE_NORMAL_PRIORITY_CLASS,
            "high": HIGH_PRIORITY_CLASS,
        }
        priority = priority_map.get(priority_name, ABOVE_NORMAL_PRIORITY_CLASS)
        SetPriorityClass(proc, priority)

        cpu_count = os.cpu_count() or 1
        use_all_cores = os.environ.get("IMB_WIN_USE_ALL_CORES", "1").strip() == "1"

        if use_all_cores and cpu_count > 0:
            affinity_mask = (1 << cpu_count) - 1
            SetProcessAffinityMask(proc, affinity_mask)

    except Exception:
        return


def configure_runtime(
    thread_count: Optional[int] = None,
    force_windows_boost: bool = True,
) -> int:
    if thread_count is None:
        thread_count = _safe_int_env("IMB_CPU_THREADS", multiprocessing.cpu_count())

    thread_count = max(1, int(thread_count))

    _set_default_thread_env(thread_count)

    if sys.platform.startswith("win") and force_windows_boost:
        _boost_windows_process()

    return thread_count