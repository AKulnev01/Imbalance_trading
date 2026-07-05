from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from online.trading import config


ROOT = config.ROOT
ENV_FILE = ROOT / ".env"
STATE_PATH = ROOT / "online" / "_state_tg_control_bot.json"

HOST_MAC = "mac"
HOST_WIN = "win"

DEFAULT_RUN_HOST = os.environ.get("TG_DEFAULT_RUN_HOST", HOST_WIN).strip().lower()
POLL_TIMEOUT_SECONDS = int(os.environ.get("TG_POLL_TIMEOUT_SECONDS", "30"))
POLL_SLEEP_SECONDS = float(os.environ.get("TG_POLL_SLEEP_SECONDS", "1.0"))
MAX_MESSAGE_LEN = int(os.environ.get("TG_MAX_MESSAGE_LEN", "3500"))
TG_GET_UPDATES_SOCKET_TIMEOUT_SECONDS = int(os.environ.get("TG_GET_UPDATES_SOCKET_TIMEOUT_SECONDS", "45"))
TG_SEND_MESSAGE_SOCKET_TIMEOUT_SECONDS = int(os.environ.get("TG_SEND_MESSAGE_SOCKET_TIMEOUT_SECONDS", "15"))
TG_COMMAND_TIMEOUT_SECONDS = int(os.environ.get("TG_COMMAND_TIMEOUT_SECONDS", "90"))
TG_BACKTEST_TIMEOUT_SECONDS = int(os.environ.get("TG_BACKTEST_TIMEOUT_SECONDS", "7200"))
TG_API_RETRIES = int(os.environ.get("TG_API_RETRIES", "2"))
TG_API_RETRY_SLEEP_SECONDS = float(os.environ.get("TG_API_RETRY_SLEEP_SECONDS", "2.0"))


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


for k, v in load_env_file(ENV_FILE).items():

    os.environ[k] = v


TELEGRAM_TOKEN = (
    os.environ.get("TELEGRAM_TOKEN", "")
    or os.environ.get("TG_TOKEN", "")
).strip()

TG_SECRET = os.environ.get("TG_SECRET", "").strip()

STATIC_CHAT_ID = (
    os.environ.get("CHAT_ID", "")
    or os.environ.get("TG_CHAT_ID", "")
).strip()

LOCAL_HOST = os.environ.get("IMB_LOCAL_HOST", HOST_MAC).strip().lower()


def log_event(message: str) -> None:
    print(str(message), flush=True)


def read_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {
            "offset": 0,
            "allowed_chat_ids": [],
        }

    try:
        raw = STATE_PATH.read_text(encoding="utf-8")
        state = json.loads(raw)
        if not isinstance(state, dict):
            return {
                "offset": 0,
                "allowed_chat_ids": [],
            }
        state.setdefault("offset", 0)
        state.setdefault("allowed_chat_ids", [])
        return state
    except Exception:
        return {
            "offset": 0,
            "allowed_chat_ids": [],
        }


def write_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def get_allowed_chat_ids() -> List[int]:
    ids: List[int] = []

    if STATIC_CHAT_ID:
        for part in STATIC_CHAT_ID.split(","):
            raw = part.strip()
            if not raw:
                continue
            try:
                ids.append(int(raw))
            except Exception:
                pass

    state = read_state()
    for x in state.get("allowed_chat_ids", []):
        try:
            ids.append(int(x))
        except Exception:
            pass

    return sorted(set(ids))



def telegram_api(
    method: str,
    payload: Dict[str, Any],
    timeout_seconds: Optional[int] = None,
    retries: Optional[int] = None,
) -> Dict[str, Any]:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is empty in .env")

    if timeout_seconds is None:
        if method == "getUpdates":
            timeout_seconds = TG_GET_UPDATES_SOCKET_TIMEOUT_SECONDS
        else:
            timeout_seconds = TG_SEND_MESSAGE_SOCKET_TIMEOUT_SECONDS

    if retries is None:
        retries = TG_API_RETRIES

    url = "https://api.telegram.org/bot{}/{}".format(TELEGRAM_TOKEN, method)
    data = urllib.parse.urlencode(payload).encode("utf-8")

    last_error: Optional[Exception] = None

    for attempt in range(1, int(retries) + 2):
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as resp:
                body = resp.read().decode("utf-8", errors="replace")

            obj = json.loads(body)

            if not obj.get("ok"):
                raise RuntimeError("Telegram API error: {}".format(obj))

            return obj

        except Exception as exc:
            last_error = exc
            log_event(
                "TG_API_ERROR method={} attempt={} timeout={} error={!r}".format(
                    method,
                    attempt,
                    timeout_seconds,
                    exc,
                )
            )

            if attempt <= int(retries):
                time.sleep(TG_API_RETRY_SLEEP_SECONDS)

    raise RuntimeError(
        "Telegram API failed method={} attempts={} last_error={!r}".format(
            method,
            int(retries) + 1,
            last_error,
        )
    )


def telegram_send_document(
    chat_id: int,
    file_path: Path,
    caption: str,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is empty in .env")

    path = Path(file_path)

    if not path.exists():
        raise RuntimeError("document file not found: {}".format(path))

    boundary = "----imbtradeboundary{}".format(int(time.time() * 1000))
    url = "https://api.telegram.org/bot{}/sendDocument".format(TELEGRAM_TOKEN)

    def part_field(name: str, value: str) -> bytes:
        return (
            "--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"{name}\"\r\n"
            "\r\n"
            "{value}\r\n"
        ).format(
            boundary=boundary,
            name=name,
            value=value,
        ).encode("utf-8")

    file_header = (
        "--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"document\"; filename=\"{filename}\"\r\n"
        "Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\r\n"
        "\r\n"
    ).format(
        boundary=boundary,
        filename=path.name,
    ).encode("utf-8")

    body = bytearray()
    body.extend(part_field("chat_id", str(int(chat_id))))
    body.extend(part_field("caption", str(caption or "")[:1000]))
    body.extend(file_header)
    body.extend(path.read_bytes())
    body.extend("\r\n".encode("utf-8"))
    body.extend(("--{}--\r\n".format(boundary)).encode("utf-8"))

    req = urllib.request.Request(
        url=url,
        data=bytes(body),
        headers={
            "Content-Type": "multipart/form-data; boundary={}".format(boundary),
            "Content-Length": str(len(body)),
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as resp:
        response_body = resp.read().decode("utf-8", errors="replace")

    obj = json.loads(response_body)

    if not obj.get("ok"):
        raise RuntimeError("Telegram sendDocument error: {}".format(obj))

    return obj


def find_backtest_xlsx_from_output(out: str) -> Optional[Path]:
    wrote_prefix = "WROTE_XLSX:"

    for raw in reversed(str(out or "").splitlines()):
        line = raw.strip()

        if not line.startswith(wrote_prefix):
            continue

        value = line[len(wrote_prefix):].strip()

        if value:
            path = Path(value)
            if path.exists():
                return path

    out_dir = find_backtest_out_dir(out)

    if out_dir is not None and out_dir.exists():
        files = sorted(
            out_dir.glob("*.xlsx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if files:
            return files[0]

    tmp_dir = ROOT / "online" / "_tmp_backtest_exports"

    if tmp_dir.exists():
        files = sorted(
            tmp_dir.glob("*.xlsx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if files:
            return files[0]

    return None


def send_backtest_excel_document_if_exists(
    chat_id: int,
    out: str,
) -> str:
    xlsx_path = find_backtest_xlsx_from_output(out)

    if xlsx_path is None:
        return "Excel файл не найден."

    caption = "Backtest Excel: {}".format(xlsx_path.name)

    try:
        log_event(
            "TG_SEND_BACKTEST_XLSX_START chat_id={} path={}".format(
                chat_id,
                xlsx_path,
            )
        )

        telegram_send_document(
            chat_id=chat_id,
            file_path=xlsx_path,
            caption=caption,
            timeout_seconds=180,
        )

        log_event(
            "TG_SEND_BACKTEST_XLSX_OK chat_id={} path={}".format(
                chat_id,
                xlsx_path,
            )
        )

        tmp_dir = ROOT / "online" / "_tmp_backtest_exports"

        if tmp_dir in xlsx_path.parents:
            try:
                xlsx_path.unlink()
                return "Excel отправлен в Telegram и удалён локально."
            except Exception as exc:
                return "Excel отправлен в Telegram, но локально не удалён: {}".format(exc)

        return "Excel отправлен в Telegram."

    except Exception as exc:
        log_event(
            "TG_SEND_BACKTEST_XLSX_ERROR chat_id={} path={} error={!r}".format(
                chat_id,
                xlsx_path,
                exc,
            )
        )

        return "Excel найден, но отправка не удалась: {} | path={}".format(
            exc,
            xlsx_path,
        )



def main_menu_keyboard() -> Dict[str, Any]:
    return {
                "keyboard": [
            ["📊 Статус", "📜 История"],
            ["📍 Позиция"],
            ["▶️ Запуск", "⏹ Стоп"],
            ["➕ Добавить символ", "🧪 Бэктест"],
            ["⚙️ Команды"],
            ["❌ Отмена"],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }

def send_message(chat_id: int, text: str) -> None:
    chunks = split_message(text, MAX_MESSAGE_LEN)
    keyboard = json.dumps(main_menu_keyboard(), ensure_ascii=False)

    for i, chunk in enumerate(chunks, start=1):
        payload = {
            "chat_id": str(chat_id),
            "text": chunk,
            "disable_web_page_preview": "true",
        }

        if i == len(chunks):
            payload["reply_markup"] = keyboard

        log_event(
            "TG_SEND_MESSAGE_START chat_id={} chunk={}/{} len={}".format(
                chat_id,
                i,
                len(chunks),
                len(chunk),
            )
        )

        telegram_api(
            "sendMessage",
            payload,
            timeout_seconds=TG_SEND_MESSAGE_SOCKET_TIMEOUT_SECONDS,
            retries=TG_API_RETRIES,
        )

        log_event(
            "TG_SEND_MESSAGE_OK chat_id={} chunk={}/{}".format(
                chat_id,
                i,
                len(chunks),
            )
        )

def split_message(text: str, max_len: int) -> List[str]:
    src = str(text or "")
    if len(src) <= max_len:
        return [src]

    chunks = []
    cur = ""

    for line in src.splitlines():
        add = line + "\n"
        if len(cur) + len(add) > max_len:
            if cur:
                chunks.append(cur.rstrip())
                cur = ""
            if len(add) > max_len:
                for i in range(0, len(add), max_len):
                    chunks.append(add[i:i + max_len])
            else:
                cur = add
        else:
            cur += add

    if cur:
        chunks.append(cur.rstrip())

    return chunks



def get_updates(offset: int) -> List[Dict[str, Any]]:
    payload = {
        "timeout": str(POLL_TIMEOUT_SECONDS),
        "offset": str(offset),
        "allowed_updates": json.dumps(["message"]),
    }

    obj = telegram_api(
        "getUpdates",
        payload,
        timeout_seconds=TG_GET_UPDATES_SOCKET_TIMEOUT_SECONDS,
        retries=TG_API_RETRIES,
    )

    result = obj.get("result", [])

    if not isinstance(result, list):
        return []

    return result

def normalize_command(text: str) -> List[str]:
    raw = str(text or "").strip()

    if not raw:
        return []

    parts = raw.split()
    if not parts:
        return []

    cmd = parts[0].strip()

    if "@" in cmd:
        cmd = cmd.split("@", 1)[0]

    parts[0] = cmd.lower()
    return parts

def normalize_menu_text(text: str) -> str:
    raw = str(text or "").strip()

    menu_map = {
        "📊 Статус": "/status",
        "📜 История": "/history 20",
        "📍 Позиция": "/position",
        "▶️ Запуск": "/run",
        "⏹ Стоп": "/stop",
        "➕ Добавить символ": "/add_symbol_wizard",
        "🧪 Бэктест": "/backtest_wizard",
        "⚙️ Команды": "/help",
        "❌ Отмена": "/cancel",
    }

    return menu_map.get(raw, raw)

def is_authorized(chat_id: int) -> bool:
    return int(chat_id) in get_allowed_chat_ids()


def authorize_chat(chat_id: int, secret: str) -> bool:
    if not TG_SECRET:
        return False

    if str(secret).strip() != TG_SECRET:
        return False

    state = read_state()
    allowed = []

    for x in state.get("allowed_chat_ids", []):
        try:
            allowed.append(int(x))
        except Exception:
            pass

    if int(chat_id) not in allowed:
        allowed.append(int(chat_id))

    state["allowed_chat_ids"] = sorted(set(allowed))
    write_state(state)
    return True


def build_base_env() -> Dict[str, str]:
    env = os.environ.copy()

    env["IMB_PROJECT_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    old_warnings = env.get("PYTHONWARNINGS", "").strip()
    extra_warning = "ignore:pandas only supports SQLAlchemy connectable:UserWarning"
    if old_warnings:
        env["PYTHONWARNINGS"] = old_warnings + "," + extra_warning
    else:
        env["PYTHONWARNINGS"] = extra_warning

    if "IMB_TRADING_DRY_RUN" not in env:
        env["IMB_TRADING_DRY_RUN"] = "0"

    return env



def run_service_status(args: List[str]) -> Tuple[int, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "online.trading.service_status",
    ] + [str(x) for x in args]

    action = str(args[0]).strip().lower() if args else ""
    is_long_action = action in {
        "backtest",
        "backtest-local",
        "add-symbol",
        "add-symbol-local",
    }

    command_timeout_seconds = (
        TG_BACKTEST_TIMEOUT_SECONDS
        if is_long_action
        else TG_COMMAND_TIMEOUT_SECONDS
    )

    log_event(
        "TG_COMMAND_SUBPROCESS_START timeout={} cmd={!r}".format(
            command_timeout_seconds,
            cmd,
        )
    )

    try:
        env = build_base_env()

        if args and str(args[0]).strip().lower() in {"history", "history-local"}:
            env["IMB_HISTORY_OUTPUT_FORMAT"] = "json"

        if args and str(args[0]).strip().lower() in {"position", "position-local", "pos"}:
            env["IMB_POSITION_OUTPUT_FORMAT"] = "json"

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=command_timeout_seconds,
        )

        out = str(proc.stdout or "")

        log_event(
            "TG_COMMAND_SUBPROCESS_DONE returncode={} out_len={}".format(
                int(proc.returncode),
                len(out),
            )
        )

        return int(proc.returncode), out

    except subprocess.TimeoutExpired as exc:
        out_any = exc.stdout or ""

        if isinstance(out_any, bytes):
            out = out_any.decode("utf-8", errors="replace")
        else:
            out = str(out_any)

        msg = (
            "COMMAND_TIMEOUT_AFTER_{}S\n"
            "cmd={!r}\n\n"
            "{}"
        ).format(
            command_timeout_seconds,
            cmd,
            out,
        )

        log_event(
            "TG_COMMAND_SUBPROCESS_TIMEOUT timeout={} cmd={!r}".format(
                command_timeout_seconds,
                cmd,
            )
        )

        return 124, msg

def parse_host(raw: Optional[str], default_host: str) -> str:
    if raw is None:
        return default_host

    host = str(raw).strip().lower()

    if host in {HOST_MAC, HOST_WIN}:
        return host

    raise RuntimeError("host must be mac or win")


def normalize_percent_input_to_ratio_text(raw: Any, name: str) -> str:
    text = str(raw or "").strip().replace(",", ".")

    if not text:
        raise RuntimeError("{} must not be empty".format(name))

    try:
        value = float(text)
    except Exception:
        raise RuntimeError("{} must be numeric, got: {}".format(name, raw))

    if value < 0.0:
        raise RuntimeError("{} must be >= 0, got: {}".format(name, raw))

    ratio = value / 100.0

    if ratio > 1.0:
        raise RuntimeError("{} is too large: {}%".format(name, value))

    out = "{:.10f}".format(ratio).rstrip("0").rstrip(".")

    if not out:
        out = "0"

    return out



def tg_status(args: List[str]) -> Tuple[int, str]:
    if len(args) >= 2:
        host = parse_host(args[1], "")

        if host == HOST_WIN and LOCAL_HOST == HOST_WIN:
            return run_service_status(["status-local"])

        if host == HOST_MAC and LOCAL_HOST == HOST_MAC:
            return run_service_status(["status-local"])

        return run_service_status(["status", host])

    if DEFAULT_RUN_HOST == LOCAL_HOST:
        return run_service_status(["status-local"])

    return run_service_status(["status", DEFAULT_RUN_HOST])

def tg_run(args: List[str]) -> Tuple[int, str]:
    host = DEFAULT_RUN_HOST

    if len(args) >= 2:
        host = parse_host(args[1], DEFAULT_RUN_HOST)

    if host == HOST_WIN and LOCAL_HOST == HOST_WIN:
        return run_service_status(["start-local"])

    if host == HOST_MAC and LOCAL_HOST == HOST_MAC:
        return run_service_status(["start-local"])

    return run_service_status(["start", host])



def tg_stop(args: List[str]) -> Tuple[int, str]:
    if len(args) >= 2:
        host = parse_host(args[1], "")

        if host == HOST_WIN and LOCAL_HOST == HOST_WIN:
            return run_service_status(["stop-local"])

        if host == HOST_MAC and LOCAL_HOST == HOST_MAC:
            return run_service_status(["stop-local"])

        return run_service_status(["stop", host])

    if DEFAULT_RUN_HOST == LOCAL_HOST:
        return run_service_status(["stop-local"])

    return run_service_status(["stop", DEFAULT_RUN_HOST])


def tg_history(args: List[str]) -> Tuple[int, str]:
    hours = "20"
    host: Optional[str] = None

    if len(args) >= 2:
        raw = str(args[1]).strip().lower()
        if raw in {HOST_MAC, HOST_WIN}:
            host = raw
        else:
            hours = raw

    if len(args) >= 3:
        host = parse_host(args[2], "")

    if host is None:
        host = DEFAULT_RUN_HOST

    if host == LOCAL_HOST:
        return run_service_status(["history-local", hours])

    return run_service_status(["history", hours, host])
def tg_position(args: List[str]) -> Tuple[int, str]:
    symbol: Optional[str] = None
    count = "1"
    host = DEFAULT_RUN_HOST

    if len(args) >= 2:
        raw = str(args[1]).strip().upper()

        if raw.lower() in {HOST_MAC, HOST_WIN}:
            host = parse_host(raw.lower(), DEFAULT_RUN_HOST)
        else:
            try:
                count_i = int(raw)
                if count_i <= 0:
                    raise RuntimeError("position count must be > 0")
                count = str(count_i)
            except ValueError:
                symbol = raw
                if symbol and not symbol.endswith("USDT"):
                    symbol = symbol + "USDT"

    if len(args) >= 3:
        raw = str(args[2]).strip().lower()

        if raw in {HOST_MAC, HOST_WIN}:
            host = parse_host(raw, DEFAULT_RUN_HOST)
        else:
            try:
                count_i = int(raw)
            except Exception:
                raise RuntimeError("position count must be integer, example: /position ENAUSDT 3")

            if count_i <= 0:
                raise RuntimeError("position count must be > 0")

            count = str(count_i)

    if len(args) >= 4:
        host = parse_host(args[3], DEFAULT_RUN_HOST)

    service_args = ["position-local" if host == LOCAL_HOST else "position"]

    if symbol is not None:
        service_args.append(symbol)

    service_args.append(count)

    if host != LOCAL_HOST:
        service_args.append(host)

    return run_service_status(service_args)

def tg_backtest(args: List[str]) -> Tuple[int, str]:
    if len(args) < 5:
        return (
            2,
            "❌ Неверный формат /backtest\n\n"
            "Формат новый:\n"
            "/backtest YYYY-MM-DD HH:MM YYYY-MM-DD HH:MM "
            "[gate2] [gate4] [gate5_1] [gate5_3] "
            "[chulan] [side_whitelist] [conditional_whitelist] [max_full_sl_risk_pct] "
            "[slots] [write_blacklist] [reset_blacklist] [sync_m1] [host]\n\n"
            "Пример:\n"
            "/backtest 2026-01-01 00:00 2026-06-14 20:00 "
            "0.70 0.57 0.10 0.54 0 1 1 6 1 0 0 1 win\n\n"
            "Старый формат после side whitelist тоже поддерживается:\n"
            "[slots] [write_blacklist] [reset_blacklist] [sync_m1] [host]\n"
            "тогда conditional=1 и risk=6%.\n\n"
            "Минимальный пример:\n"
            "/backtest 2026-01-01 00:00 2026-06-14 20:00"
        )

    start = str(args[1]).strip() + " " + str(args[2]).strip()
    end = str(args[3]).strip() + " " + str(args[4]).strip()

    gate2 = str(config.GATE2_THR)
    gate4 = str(config.GATE4_THR)
    gate5_1 = str(config.GATE5_1_THR)
    gate5_3 = str(config.GATE5_3_THR)

    chulan = "0"
    side_aware_whitelist = "1"
    conditional_side_aware_whitelist = "1"
    max_full_sl_risk_pct = "6"
    slots = "1"
    write_blacklist = "0"
    reset_blacklist = "0"
    sync_m1 = "1"
    host = DEFAULT_RUN_HOST

    if len(args) >= 6:
        gate2 = str(args[5]).strip()
    if len(args) >= 7:
        gate4 = str(args[6]).strip()
    if len(args) >= 8:
        gate5_1 = str(args[7]).strip()
    if len(args) >= 9:
        gate5_3 = str(args[8]).strip()
    if len(args) >= 10:
        chulan = str(args[9]).strip()
    if len(args) >= 11:
        side_aware_whitelist = str(args[10]).strip()

    rest = [str(x).strip() for x in args[11:]]

    if rest and rest[-1].lower() in {HOST_MAC, HOST_WIN}:
        host = parse_host(rest[-1], DEFAULT_RUN_HOST)
        rest = rest[:-1]

    if len(rest) == 4:
        slots = rest[0]
        write_blacklist = rest[1]
        reset_blacklist = rest[2]
        sync_m1 = rest[3]
    elif len(rest) > 0:
        if len(rest) >= 1:
            conditional_side_aware_whitelist = rest[0]
        if len(rest) >= 2:
            max_full_sl_risk_pct = rest[1]
        if len(rest) >= 3:
            slots = rest[2]
        if len(rest) >= 4:
            write_blacklist = rest[3]
        if len(rest) >= 5:
            reset_blacklist = rest[4]
        if len(rest) >= 6:
            sync_m1 = rest[5]
        if len(rest) > 6:
            raise RuntimeError("too many /backtest args after side whitelist: {}".format(rest))

    for name, value in [
        ("chulan", chulan),
        ("side_aware_whitelist", side_aware_whitelist),
        ("conditional_side_aware_whitelist", conditional_side_aware_whitelist),
        ("write_blacklist", write_blacklist),
        ("reset_blacklist", reset_blacklist),
        ("sync_m1", sync_m1),
    ]:
        if value not in {"0", "1"}:
            raise RuntimeError("{} must be 0 or 1, got: {}".format(name, value))

    max_full_sl_capital_risk = normalize_percent_input_to_ratio_text(
        max_full_sl_risk_pct,
        "max_full_sl_risk_pct",
    )

    try:
        slots_int = int(slots)
    except Exception:
        raise RuntimeError("slots must be integer, got: {}".format(slots))

    if slots_int != 1:
        raise RuntimeError(
            "Сейчас backtest поддерживает только slot1. Получено slots={}".format(slots_int)
        )

    service_args = [
        "backtest-local" if host == LOCAL_HOST else "backtest",
    ]

    if host != LOCAL_HOST:
        service_args.append(host)

    service_args += [
        "--start",
        start,
        "--end",
        end,
        "--gate2",
        gate2,
        "--gate4",
        gate4,
        "--gate5-1",
        gate5_1,
        "--gate5-3",
        gate5_3,
        "--chulan",
        chulan,
        "--side-aware-whitelist",
        side_aware_whitelist,
        "--conditional-side-aware-whitelist",
        conditional_side_aware_whitelist,
        "--max-full-sl-capital-risk",
        max_full_sl_capital_risk,
        "--slots",
        str(slots_int),
        "--write-dynamic-blacklist",
        write_blacklist,
        "--reset-backtest-blacklist",
        reset_blacklist,
    ]

    if sync_m1 == "0":
        service_args.append("--skip-m1-sync")

    return run_service_status(service_args)

def help_text() -> str:
    return (
        "🤖 ImbalanceSearcher\n\n"
        "Меню:\n"
        "📊 Статус — состояние автотрейда\n"
        "📜 История — история за 20 часов\n"
        "▶️ Запуск — запустить автотрейд\n"
        "⏹ Стоп — остановить автотрейд\n"
        "🧪 Бэктест — пошаговый запуск backtest\n"
        "❌ Отмена — отменить текущий мастер\n\n"
        "Ручные команды:\n"
        "/status или /status win\n"
        "/history или /history 48\n"
        "/position — последняя исполненная сделка по системе\n"
        "/position 3 — последние 3 закрытые сделки по системе\n"
        "/position ENAUSDT или /pos ENAUSDT\n"
        "/position ENAUSDT 3 — последние 3 закрытые сделки по монете\n"
        "/run\n"
        "/stop\n"
        "/cancel\n\n"
        "Backtest лучше запускать через кнопку 🧪 Бэктест."
    )

def parse_key_value_output(out: str) -> Dict[str, str]:
    data: Dict[str, str] = {}

    for raw in str(out or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key:
            data[key] = value

    return data


def format_bool_on_off(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"true", "1", "yes", "on"}:
        return "ON"
    if raw in {"false", "0", "no", "off"}:
        return "OFF"
    return str(value)




def fmt_status_float_trim(value: Any, digits: int = 8) -> str:
    if is_empty_value(value):
        return ""

    try:
        x = float(value)
    except Exception:
        return str(value)

    text = ("{:.%df}" % int(digits)).format(x)
    text = text.rstrip("0").rstrip(".")

    if text == "-0":
        text = "0"

    return text


def fmt_status_money_1(value: Any) -> str:
    if is_empty_value(value):
        return ""

    try:
        x = float(value)
    except Exception:
        return str(value)

    return "{:.1f}".format(x)


def fmt_status_ms_1(value: Any) -> str:
    if is_empty_value(value):
        return ""

    try:
        x = float(value)
    except Exception:
        return str(value)

    return "{:.1f}".format(x)


def fmt_status_h4_time(value: Any) -> str:
    if is_empty_value(value):
        return ""

    try:
        dt = pd.to_datetime(str(value), utc=True, errors="coerce")
        if pd.isna(dt):
            return str(value)
        return dt.strftime("%H:%M")
    except Exception:
        return str(value)


def fmt_status_ts_short(value: Any) -> str:
    if is_empty_value(value):
        return ""

    try:
        dt = pd.to_datetime(str(value), utc=True, errors="coerce")
        if pd.isna(dt):
            return str(value)
        return dt.strftime("%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def format_status_output(code: int, out: str, command_text: str) -> str:
    data = parse_key_value_output(out)

    if code != 0:
        body = compact_raw_output(out, max_lines=35)
        if not body:
            body = "EMPTY_OUTPUT"
        return "❌ Статус недоступен\n\n{}".format(body)

    running = str(data.get("service_running", "")).strip().lower() == "true"
    running_icon = "🟢" if running else "🔴"
    running_text = "работает" if running else "остановлен"

    lines: List[str] = [
        "📊 Статус",
        "",
        "{} Сервис: {}".format(running_icon, running_text),
    ]

    capital = data.get("capital_usdt")
    trade_capital = data.get("trade_capital_usdt")
    pnl = data.get("current_position_pnl_usdt")
    open_positions = data.get("open_positions_count")

    money_lines: List[str] = []
    if is_meaningful_value(capital):
        money_lines.append("Баланс: {} USDT".format(fmt_status_money_1(capital)))
    if is_meaningful_value(trade_capital):
        money_lines.append("Рабочий капитал: {} USDT".format(fmt_status_money_1(trade_capital)))
    if is_meaningful_value(open_positions):
        money_lines.append("Открытых позиций: {}".format(open_positions))
    if is_meaningful_value(pnl):
        money_lines.append("PnL позиции: {} USDT".format(fmt_status_money_1(pnl)))
    if money_lines:
        lines += ["", "💰 Капитал"] + money_lines

    trading_leverage = data.get("trading_leverage")
    position_notional_multiplier = data.get("position_notional_multiplier")
    position_notional_usdt_plan = data.get("position_notional_usdt_plan")
    estimated_initial_margin_usdt = data.get("estimated_initial_margin_usdt")
    margin_buffer_usdt = data.get("margin_buffer_usdt")
    margin_ok = format_bool_on_off(data.get("margin_ok", ""))
    risk_cap_enabled = format_bool_on_off(data.get("position_risk_cap_enabled", ""))
    max_full_sl_capital_risk = data.get("max_full_sl_capital_risk")
    position_risk_cap_include_round_trip_cost = format_bool_on_off(data.get("position_risk_cap_include_round_trip_cost", ""))
    position_risk_cap_fee_side = data.get("position_risk_cap_fee_side")
    position_risk_cap_slippage_side = data.get("position_risk_cap_slippage_side")

    risk_lines: List[str] = []
    leverage_parts: List[str] = []
    if is_meaningful_value(trading_leverage):
        leverage_parts.append("плечо x{}".format(fmt_status_float_trim(trading_leverage)))
    if is_meaningful_value(position_notional_multiplier):
        leverage_parts.append("множитель позиции x{}".format(fmt_status_float_trim(position_notional_multiplier)))
    if leverage_parts:
        risk_lines.append(" · ".join(leverage_parts))
    if is_meaningful_value(position_notional_usdt_plan):
        risk_lines.append("Плановый объём позиции: {} USDT".format(fmt_status_money_1(position_notional_usdt_plan)))

    margin_parts: List[str] = []
    if is_meaningful_value(estimated_initial_margin_usdt):
        margin_parts.append("маржа {} USDT".format(fmt_status_money_1(estimated_initial_margin_usdt)))
    if is_meaningful_value(margin_buffer_usdt):
        margin_parts.append("запас {} USDT".format(fmt_status_money_1(margin_buffer_usdt)))
    if is_meaningful_value(margin_ok):
        margin_parts.append("OK {}".format(margin_ok))
    if margin_parts:
        risk_lines.append(" · ".join(margin_parts))

    cap_parts: List[str] = []
    if is_meaningful_value(risk_cap_enabled):
        cap_parts.append("risk cap {}".format(risk_cap_enabled))
    if is_meaningful_value(max_full_sl_capital_risk):
        cap_parts.append("full SL {}".format(fmt_backtest_ratio_pct(max_full_sl_capital_risk)))
    if is_meaningful_value(position_risk_cap_include_round_trip_cost):
        cap_parts.append("cost {}".format(position_risk_cap_include_round_trip_cost))
    if is_meaningful_value(position_risk_cap_fee_side):
        cap_parts.append("fee {}".format(fmt_backtest_ratio_pct(position_risk_cap_fee_side)))
    if is_meaningful_value(position_risk_cap_slippage_side):
        cap_parts.append("slip {}".format(fmt_backtest_ratio_pct(position_risk_cap_slippage_side)))
    if cap_parts:
        risk_lines.append(" · ".join(cap_parts))

    if risk_lines:
        lines += ["", "🧷 Плечо, маржа и риск"] + risk_lines

    grid = data.get("grid_name")
    gate2 = data.get("gate2_thr")
    gate4 = data.get("gate4_thr")
    gate5_1 = data.get("gate5_1_thr")
    gate5_3 = data.get("gate5_3_thr")

    model_lines: List[str] = []
    if is_meaningful_value(grid):
        model_lines.append("Сетка: {}".format(grid))
    if any(is_meaningful_value(x) for x in [gate2, gate4, gate5_1, gate5_3]):
        model_lines.append(
            "Пороги: G2 {} | G4 {} | G5.1 {} | G5.3 {}".format(
                fmt_status_float_trim(gate2),
                fmt_status_float_trim(gate4),
                fmt_status_float_trim(gate5_1),
                fmt_status_float_trim(gate5_3),
            )
        )
    if model_lines:
        lines += ["", "🧠 Модель"] + model_lines

    partial_tp_enabled = format_bool_on_off(data.get("partial_tp_enabled", ""))
    early_stop_enabled = format_bool_on_off(data.get("early_stop_enabled", ""))
    rest_stop_enabled = format_bool_on_off(data.get("rest_stop_after_partial_enabled", ""))

    tm_lines: List[str] = []
    if is_meaningful_value(partial_tp_enabled):
        tm_lines.append("Partial TP: {}".format(partial_tp_enabled))
    if is_meaningful_value(early_stop_enabled):
        tm_lines.append("Early stop: {}".format(early_stop_enabled))
    if is_meaningful_value(rest_stop_enabled):
        tm_lines.append("Rest stop: {}".format(rest_stop_enabled))
    if tm_lines:
        lines += ["", "🧩 Управление позицией"] + tm_lines

    next_h4 = data.get("next_h4_close_utc")
    time_left = data.get("time_to_next_h4_close")
    if is_meaningful_value(next_h4) or is_meaningful_value(time_left):
        lines += ["", "🕓 H4"]
        if is_meaningful_value(next_h4):
            lines.append("Следующая свеча: {}".format(fmt_status_h4_time(next_h4)))
        if is_meaningful_value(time_left):
            lines.append("Осталось: {}".format(time_left))

    dyn_filter = format_bool_on_off(data.get("dynamic_symbol_filter_enabled", ""))
    if is_meaningful_value(dyn_filter):
        lines += ["", "🛡 Dynamic filter: {}".format(dyn_filter)]

    ws_running = format_bool_on_off(data.get("ws_process_running", ""))
    ws_count = data.get("ws_process_count")
    ws_raw_count = data.get("ws_raw_process_count")
    ws_pid = data.get("ws_pid")
    ws_last_heartbeat = data.get("ws_last_heartbeat_utc")
    ws_last_event = data.get("ws_last_event_utc")
    ws_events_24h = data.get("ws_events_24h")
    ws_errors_tail = data.get("ws_errors_tail")

    ws_lines: List[str] = []
    ws_head: List[str] = []
    if is_meaningful_value(ws_running):
        ws_head.append("process {}".format(ws_running))
    if is_meaningful_value(ws_count):
        ws_head.append("count {}".format(ws_count))
    if is_meaningful_value(ws_raw_count):
        ws_head.append("raw {}".format(ws_raw_count))
    if is_meaningful_value(ws_pid):
        ws_head.append("pid {}".format(ws_pid))
    if ws_head:
        ws_lines.append(" · ".join(ws_head))
    if is_meaningful_value(ws_last_heartbeat):
        ws_lines.append("heartbeat: {}".format(fmt_status_ts_short(ws_last_heartbeat)))
    if is_meaningful_value(ws_last_event):
        ws_lines.append("last event: {}".format(fmt_status_ts_short(ws_last_event)))

    ws_stat_parts: List[str] = []
    if is_meaningful_value(ws_events_24h):
        ws_stat_parts.append("events24h {}".format(ws_events_24h))

    recent_errors_text = "OK"
    if is_meaningful_value(ws_errors_tail):
        try:
            ws_errors_tail_i = int(float(str(ws_errors_tail).strip()))
        except Exception:
            ws_errors_tail_i = -1

        if ws_errors_tail_i > 0:
            recent_errors_text = str(ws_errors_tail)

    ws_stat_parts.append("recent errors: {}".format(recent_errors_text))

    if ws_stat_parts:
        ws_lines.append(" · ".join(ws_stat_parts))

    if ws_lines:
        lines += ["", "📡 WS"] + ws_lines

    api_ok = format_bool_on_off(data.get("api_bybit_tcp_ok", ""))
    api_ms = data.get("api_bybit_tcp_ms")
    api_ip = data.get("api_bybit_tcp_ip")
    stream_ok = format_bool_on_off(data.get("stream_bybit_tcp_ok", ""))
    stream_ms = data.get("stream_bybit_tcp_ms")
    stream_ip = data.get("stream_bybit_tcp_ip")

    network_lines: List[str] = []
    api_parts: List[str] = []
    if is_meaningful_value(api_ok):
        api_parts.append("api {}".format(api_ok))
    if is_meaningful_value(api_ms):
        api_parts.append("{} ms".format(fmt_status_ms_1(api_ms)))
    if is_meaningful_value(api_ip):
        api_parts.append(str(api_ip))
    if api_parts:
        network_lines.append("Bybit API: " + " · ".join(api_parts))

    stream_parts: List[str] = []
    if is_meaningful_value(stream_ok):
        stream_parts.append("stream {}".format(stream_ok))
    if is_meaningful_value(stream_ms):
        stream_parts.append("{} ms".format(fmt_status_ms_1(stream_ms)))
    if is_meaningful_value(stream_ip):
        stream_parts.append(str(stream_ip))
    if stream_parts:
        network_lines.append("Bybit WS: " + " · ".join(stream_parts))
    if network_lines:
        lines += ["", "🌐 Сеть"] + network_lines

    bad_protective = data.get("closed_position_bad_protective_orders")
    cleanup_unmarked = data.get("closed_positions_without_cleanup_mark_but_no_active_orders")

    cleanup_lines: List[str] = []
    if is_meaningful_value(bad_protective):
        icon = "✅" if str(bad_protective).strip() == "0" else "⚠️"
        cleanup_lines.append("{} bad protective orders: {}".format(icon, bad_protective))
    if is_meaningful_value(cleanup_unmarked):
        icon = "✅" if str(cleanup_unmarked).strip() == "0" else "⚠️"
        cleanup_lines.append("{} unmarked cleanup: {}".format(icon, cleanup_unmarked))
    if cleanup_lines:
        lines += ["", "🧹 Cleanup"] + cleanup_lines

    ws_health_error = data.get("ws_health_error")
    if is_meaningful_value(ws_health_error):
        lines += ["", "⚠️ WS health error", str(ws_health_error)]

    return "\n".join(lines)

def compact_raw_output(out: str, max_lines: int = 45) -> str:
    lines = []

    skip_prefixes = (
        "log:",
        "raw_processes:",
        "pid=",
        "module:",
        "service_module:",
        "control_module:",
        "dynamic_blacklist_source:",
    )

    for raw in str(out or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            continue

        low = stripped.lower()
        if any(low.startswith(x) for x in skip_prefixes):
            continue

        lines.append(line)

    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["...", "Вывод укорочен. Полный результат смотри в логах."]

    return "\n".join(lines).strip()


def is_history_header_line(line: str) -> bool:
    text = str(line or "")
    required = [
        "close",
        "signal",
        "symbol",
        "decision",
        "side",
    ]

    return all(x in text for x in required)


def find_column_spans(header: str, columns: List[str]) -> List[Tuple[str, int, int]]:
    positions: List[Tuple[str, int]] = []

    for col in columns:
        idx = str(header).find(col)
        if idx >= 0:
            positions.append((col, idx))

    positions = sorted(positions, key=lambda x: x[1])

    spans: List[Tuple[str, int, int]] = []

    for i, item in enumerate(positions):
        col, start = item

        if i + 1 < len(positions):
            end = positions[i + 1][1]
        else:
            end = len(header) + 120

        spans.append((col, start, end))

    return spans


def parse_fixed_width_history_rows(out: str) -> List[Dict[str, str]]:
    lines = str(out or "").splitlines()

    header = ""
    header_idx = -1

    for i, line in enumerate(lines):
        if is_history_header_line(line):
            header = line.lstrip()
            header_idx = i
            break

    if not header:
        return []

    columns = [
        "close",
        "signal",
        "symbol",
        "decision",
        "side",
        "entry_signal",
        "entry_plan",
        "entry_actual",
        "slip_pct",
        "tp",
        "sl",
        "pos_status",
        "reason",
        "rank",
    ]

    spans = find_column_spans(header, columns)

    rows: List[Dict[str, str]] = []

    for raw in lines[header_idx + 1:]:
        line = str(raw or "").lstrip()

        if not line.strip():
            continue

        if not re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", line):
            continue

        row: Dict[str, str] = {}

        for col, span_start, span_end in spans:
            value = line[span_start:span_end].strip() if span_start < len(line) else ""
            row[col] = value

        rows.append(row)

    return rows


def short_time(value: str) -> str:
    text = str(value or "").strip()
    parts = text.split(" ")

    if len(parts) == 2:
        return parts[1]

    return text


def short_price(value: str) -> str:
    raw = str(value or "").strip()

    if not raw:
        return ""

    try:
        x = float(raw)
    except Exception:
        return raw

    if abs(x) >= 100:
        return "{:.2f}".format(x)

    if abs(x) >= 1:
        return "{:.4f}".format(x)

    return "{:.6f}".format(x)


def short_percent(value: str) -> str:
    raw = str(value or "").strip()

    if not raw:
        return ""

    try:
        x = float(raw)
    except Exception:
        return raw

    return "{:.3f}%".format(x * 100.0)


def short_reason(value: str) -> str:
    raw = str(value or "").strip()

    reason_map = {
        "OK": "OK",
        "BELOW_GATE2": "G2",
        "BELOW_GATE4": "G4",
        "BELOW_GATE5_1": "G5.1",
        "BELOW_GATE5_3": "G5.3",
        "NO_SELECTED_SIGNAL": "NOSEL",
    }

    return reason_map.get(raw, raw[:10])


def short_position_status(value: str) -> str:
    raw = str(value or "").strip()

    status_map = {
        "ENTRY_FAILED": "FAIL",
        "ENTRY_ORDER_SENT": "SENT",
        "ENTRY_PARTIALLY_FILLED": "PART",
        "ENTRY_FILLED": "ENTRY_OK",
        "TP_SL_PLACED": "OPEN",
        "POSITION_OPEN": "OPEN",
        "POSITION_CLOSED_TAKE_PROFIT": "TP",
        "POSITION_CLOSED_PARTIAL_TP": "PARTIAL",
        "POSITION_CLOSED_FINAL_TP": "FINAL_TP",
        "POSITION_CLOSED_STOP_LOSS": "SL",
        "POSITION_CLOSED_EARLY_STOP": "EARLY",
        "POSITION_CLOSED_REST_STOP_AFTER_PARTIAL": "REST_SL",
        "POSITION_CLOSED_TTL_CLOSE": "TTL",
        "POSITION_CLOSED_EMERGENCY_CLOSE": "EMERG",
        "POSITION_CLOSED_MANUAL_CLOSE": "MANUAL",
        "POSITION_CLOSED_MANUAL": "MANUAL",
        "POSITION_CLOSED_EXTERNAL": "EXT",
        "TP_SL_FAILED": "NO_TPSL",
        "TTL_CLOSE_SENT": "TTL_SENT",
        "TTL_CLOSE_FAILED": "TTL_FAIL",
        "DRY_RUN_ENTRY_PLANNED": "DRY",
    }

    return status_map.get(raw, raw[:14])


def build_plain_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return ""

    widths: List[int] = []

    for col_idx, header in enumerate(headers):
        values = [str(header)]

        for row in rows:
            if col_idx < len(row):
                values.append(str(row[col_idx]))

        widths.append(max(len(x) for x in values))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    out = [sep]
    out.append(
        "| "
        + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
        + " |"
    )
    out.append(sep)

    for row in rows:
        out.append(
            "| "
            + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))
            + " |"
        )

    out.append(sep)

    return "\n".join(out)


def format_history_cards_from_output(out: str, max_rows: int = 10) -> str:
    parsed_rows = parse_fixed_width_history_rows(out)

    if not parsed_rows:
        return compact_raw_output(out, max_lines=50)

    blocks: List[str] = []

    for row in parsed_rows[:int(max_rows)]:
        decision = str(row.get("decision", "")).strip().upper()
        side = str(row.get("side", "")).strip().upper()
        symbol = str(row.get("symbol", "")).strip().upper()

        icon = "🟢" if decision == "GO" else "⚪️"

        close_value = str(row.get("close", "")).strip()
        close_short = close_value
        close_parts = close_value.split()
        if len(close_parts) == 2:
            close_short = "{} {}".format(close_parts[0][5:], close_parts[1])

        entry_signal = short_price(row.get("entry_signal", ""))
        entry_plan = short_price(row.get("entry_plan", ""))
        entry_actual = short_price(row.get("entry_actual", ""))
        slip = short_percent(row.get("slip_pct", ""))

        tp = short_price(row.get("tp", ""))
        sl = short_price(row.get("sl", ""))

        pos_status = short_position_status(row.get("pos_status", ""))
        reason = short_reason(row.get("reason", ""))

        line_1 = "{} {} | {} | {} {}".format(
            icon,
            close_short,
            symbol,
            decision,
            side,
        )

        line_2 = "entry: signal={} plan={}".format(
            entry_signal if entry_signal else "-",
            entry_plan if entry_plan else "-",
        )

        if entry_actual:
            line_2 += " actual={}".format(entry_actual)

        if slip:
            line_2 += " slip={}".format(slip)

        line_3 = "tp/sl: {} / {}".format(
            tp if tp else "-",
            sl if sl else "-",
        )

        tail_parts = []

        if pos_status:
            tail_parts.append("pos={}".format(pos_status))

        if reason:
            tail_parts.append("why={}".format(reason))

        if tail_parts:
            line_4 = " | ".join(tail_parts)
            block = "\n".join([line_1, line_2, line_3, line_4])
        else:
            block = "\n".join([line_1, line_2, line_3])

        blocks.append(block)

    if len(parsed_rows) > int(max_rows):
        blocks.append("... ещё строк: {}".format(len(parsed_rows) - int(max_rows)))

    return "\n\n".join(blocks)


def format_history_table_from_output(out: str, max_rows: int = 12) -> str:
    parsed_rows = parse_fixed_width_history_rows(out)

    if not parsed_rows:
        return compact_raw_output(out, max_lines=50)

    table_rows: List[List[str]] = []

    for row in parsed_rows[:int(max_rows)]:
        table_rows.append(
            [
                short_time(row.get("close", "")),
                str(row.get("symbol", "")).upper(),
                str(row.get("decision", "")),
                str(row.get("side", "")),
                short_price(row.get("entry_plan", "")),
                short_price(row.get("entry_actual", "")),
                short_percent(row.get("slip_pct", "")),
                short_position_status(row.get("pos_status", "")),
                short_reason(row.get("reason", "")),
            ]
        )

    headers = [
        "close",
        "symbol",
        "dec",
        "side",
        "plan",
        "actual",
        "slip",
        "pos",
        "why",
    ]

    return build_plain_table(headers, table_rows)


def parse_history_json_output(out: str) -> List[Dict[str, Any]]:
    text = str(out or "")

    start_marker = "HISTORY_JSON_BEGIN"
    end_marker = "HISTORY_JSON_END"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        return []

    raw_json = text[start_idx + len(start_marker):end_idx].strip()

    if not raw_json:
        return []

    try:
        data = json.loads(raw_json)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    rows: List[Dict[str, Any]] = []

    for item in data:
        if isinstance(item, dict):
            rows.append(item)

    return rows


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True

    text = str(value).strip()
    if not text:
        return True

    if text.lower() in {"nan", "none", "null", "nat"}:
        return True

    return False


def fmt_history_price(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    try:
        x = float(value)
    except Exception:
        return str(value)

    if abs(x) >= 1000:
        return "{:.2f}".format(x)

    if abs(x) >= 10:
        return "{:.4f}".format(x)

    if abs(x) >= 1:
        return "{:.5f}".format(x)

    return "{:.8f}".format(x)


def fmt_history_percent(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    try:
        x = float(value)
    except Exception:
        return str(value)

    return "{:+.3f}%".format(x * 100.0)


def short_history_reason(value: Any) -> str:
    text = str(value or "").strip()

    mapping = {
        "OK": "OK",
        "BELOW_GATE2": "G2",
        "BELOW_GATE4": "G4",
        "BELOW_GATE5_1": "G5.1",
        "BELOW_GATE5_3": "G5.3",
        "NO_SELECTED_SIGNAL": "NO_SIGNAL",
    }

    return mapping.get(text, text)


def short_history_status(value: Any) -> str:
    text = str(value or "").strip()

    mapping = {
        "ENTRY_FAILED": "ENTRY_FAIL",
        "ENTRY_ORDER_SENT": "ENTRY_SENT",
        "ENTRY_PARTIALLY_FILLED": "ENTRY_PART",
        "ENTRY_FILLED": "ENTRY_OK",
        "TP_SL_PLACED": "TP/SL_OK",
        "TP_SL_FAILED": "TP/SL_FAIL",
        "POSITION_OPEN": "OPEN",
        "POSITION_CLOSED_TAKE_PROFIT": "CLOSED_TP",
        "POSITION_CLOSED_PARTIAL_TP": "CLOSED_PARTIAL",
        "POSITION_CLOSED_FINAL_TP": "CLOSED_FINAL_TP",
        "POSITION_CLOSED_STOP_LOSS": "CLOSED_SL",
        "POSITION_CLOSED_EARLY_STOP": "CLOSED_EARLY",
        "POSITION_CLOSED_REST_STOP_AFTER_PARTIAL": "CLOSED_REST",
        "POSITION_CLOSED_TTL_CLOSE": "CLOSED_TTL",
        "POSITION_CLOSED_EMERGENCY_CLOSE": "CLOSED_EMERG",
        "POSITION_CLOSED_MANUAL_CLOSE": "CLOSED_MANUAL",
        "POSITION_CLOSED_MANUAL": "CLOSED_MANUAL",
        "POSITION_CLOSED_EXTERNAL": "CLOSED_EXT",
        "TTL_CLOSE_SENT": "TTL_SENT",
        "TTL_CLOSE_FAILED": "TTL_FAIL",
    }

    return mapping.get(text, text)


def short_history_time(value: Any) -> str:
    text = str(value or "").strip()

    parts = text.split()
    if len(parts) != 2:
        return text

    date_part = parts[0]
    time_part = parts[1]

    if len(date_part) >= 10:
        date_part = date_part[5:10]

    return "{} {}".format(date_part, time_part)



def format_history_rows_as_cards(rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
    blocks: List[str] = []

    for row in rows[:int(max_rows)]:
        close_value = short_history_time(row.get("close", ""))
        symbol = str(row.get("symbol") or "").strip().upper()
        decision = str(row.get("decision") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()

        icon = "🟢" if decision == "GO" else "⚪️"

        entry_signal = fmt_history_price(row.get("entry_signal"))
        entry_plan = fmt_history_price(row.get("entry_plan"))
        entry_actual = fmt_history_price(row.get("entry_actual"))
        slip_pct = fmt_history_percent(row.get("slip_pct"))

        tp = fmt_history_price(row.get("tp"))
        sl = fmt_history_price(row.get("sl"))

        pos_status = short_history_status(row.get("pos_status"))
        exec_kind = compact_exec_kind(row.get("exec"))
        reason = short_history_reason(row.get("reason"))

        header_parts = []
        if close_value:
            header_parts.append(close_value)
        if symbol:
            header_parts.append(symbol)
        if decision:
            header_parts.append(decision)
        if side and side != "-":
            header_parts.append(side)

        lines = [
            "{} {}".format(icon, " | ".join(header_parts))
        ]

        entry_parts = []
        if entry_signal != "-":
            entry_parts.append("signal {}".format(entry_signal))
        if entry_plan != "-":
            entry_parts.append("plan {}".format(entry_plan))
        if entry_actual != "-":
            entry_parts.append("actual {}".format(entry_actual))
        if slip_pct != "-":
            entry_parts.append("slip {}".format(slip_pct))

        if entry_parts:
            lines.append("Вход: " + " | ".join(entry_parts))

        if tp != "-" or sl != "-":
            lines.append("Цели: TP {} | SL {}".format(tp, sl))

        tail = []

        if exec_kind:
            tail.append("exec: {}".format(exec_kind))

        if pos_status:
            tail.append("status: {}".format(pos_status))

        if reason and reason != "OK":
            tail.append("reason: {}".format(reason))

        if tail:
            lines.append(" · ".join(tail))

        blocks.append("\n".join(lines))

    if len(rows) > int(max_rows):
        blocks.append("... ещё строк: {}".format(len(rows) - int(max_rows)))

    return "\n\n".join(blocks)


def format_history_output(code: int, out: str, command_text: str) -> str:
    if code != 0:
        body = compact_raw_output(out, max_lines=45)
        if not body:
            body = "EMPTY_OUTPUT"
        return "❌ История недоступна\n\n{}".format(body)

    rows = parse_history_json_output(out)

    if not rows:
        body = compact_raw_output(out, max_lines=45)
        if not body:
            body = "История пустая."
        return "📜 История\n\n{}".format(body)

    body = format_history_rows_as_cards(rows, max_rows=10)
    return "📜 История\n\n{}".format(body)

def parse_position_json_output(out: str) -> Dict[str, Any]:
    text = str(out or "")

    start_marker = "POSITION_JSON_BEGIN"
    end_marker = "POSITION_JSON_END"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        return {}

    raw_json = text[start_idx + len(start_marker):end_idx].strip()

    if not raw_json:
        return {}

    try:
        obj = json.loads(raw_json)
    except Exception:
        return {}

    if not isinstance(obj, dict):
        return {}

    return obj


def fmt_position_price(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    try:
        x = float(value)
    except Exception:
        return str(value)

    if abs(x) >= 1000:
        return "{:.2f}".format(x)

    if abs(x) >= 10:
        return "{:.4f}".format(x)

    if abs(x) >= 1:
        return "{:.5f}".format(x)

    return "{:.8f}".format(x)


def fmt_position_money(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    try:
        x = float(value)
    except Exception:
        return str(value)

    sign = "+" if x > 0 else ""
    return "{}{:.4f}".format(sign, x)


def fmt_position_pct(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    try:
        x = float(value)
    except Exception:
        return str(value)

    sign = "+" if x > 0 else ""
    return "{}{:.3f}%".format(sign, x * 100.0)


def fmt_position_qty(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    try:
        x = float(value)
    except Exception:
        return str(value)

    if abs(x) >= 100:
        return "{:.2f}".format(x)

    if abs(x) >= 1:
        return "{:.4f}".format(x)

    return "{:.8f}".format(x)


def fmt_position_time(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    ts = str(value).strip()

    try:
        dt = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(dt):
            return ts
        return dt.strftime("%m-%d %H:%M:%S")
    except Exception:
        return ts


def fmt_position_life(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    try:
        sec = int(max(0, float(value)))
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



def is_meaningful_value(value: Any) -> bool:
    if is_empty_value(value):
        return False

    text = str(value).strip()

    if text in {"-", "0.000000", "0.00000000", "+0.0000", "+0.000000", "0.0000", "0"}:
        return False

    return True


def is_positive_number(value: Any) -> bool:
    if is_empty_value(value):
        return False

    try:
        return float(value) > 0.0
    except Exception:
        return False


def append_kv(lines: List[str], label: str, value: Any) -> None:
    if is_meaningful_value(value):
        lines.append("{}: {}".format(label, value))


def append_price_line(lines: List[str], label: str, value: Any) -> None:
    formatted = fmt_position_price(value)
    if is_meaningful_value(formatted):
        lines.append("{}: {}".format(label, formatted))


def append_qty_line(lines: List[str], label: str, value: Any) -> None:
    formatted = fmt_position_qty(value)
    if is_meaningful_value(formatted):
        lines.append("{}: {}".format(label, formatted))


def append_time_line(lines: List[str], label: str, value: Any) -> None:
    formatted = fmt_position_time(value)
    if is_meaningful_value(formatted):
        lines.append("{}: {}".format(label, formatted))


def compact_exec_kind(value: Any) -> str:
    text = str(value or "").strip().upper()

    mapping = {
        "NO_FILLS": "",
        "ENTRY_ONLY": "только вход",
        "OPEN": "позиция открыта",
        "TAKE_PROFIT": "полный тейк",
        "FINAL_TP": "финальный тейк",
        "PARTIAL_TP_PLUS_FINAL_TP": "частичный тейк + финальный тейк",
        "PARTIAL+FINAL_TP": "частичный тейк + финальный тейк",
        "PARTIAL_TP_PLUS_REST_STOP": "частичный тейк + rest stop",
        "PARTIAL+REST_STOP": "частичный тейк + rest stop",
        "PARTIAL_TP_PLUS_MAIN_SL": "частичный тейк + основной стоп",
        "PARTIAL+MAIN_SL": "частичный тейк + основной стоп",
        "STOP_LOSS": "основной стоп",
        "EARLY_STOP": "ранний стоп",
        "REST_STOP_AFTER_PARTIAL": "rest stop после частичного тейка",
        "REST_STOP": "rest stop после частичного тейка",
        "TTL_CLOSE": "закрытие по TTL",
        "MANUAL_CLOSE": "ручное закрытие",
        "EMERGENCY_CLOSE": "аварийное закрытие",
    }

    return mapping.get(text, text)


def compact_position_status_ru(value: Any) -> str:
    text = str(value or "").strip().upper()

    mapping = {
        "CREATED": "создана",
        "ENTRY_ORDER_SENT": "вход отправлен",
        "ENTRY_PARTIALLY_FILLED": "вход частично исполнен",
        "ENTRY_FILLED": "вход исполнен",
        "TP_SL_PLACED": "защитные ордера стоят",
        "POSITION_OPEN": "открыта",
        "TTL_CLOSE_SENT": "TTL close отправлен",
        "TTL_CLOSE_FAILED": "ошибка TTL close",
        "TP_SL_FAILED": "ошибка TP/SL",
        "POSITION_CLOSED_TAKE_PROFIT": "закрыта по тейку",
        "POSITION_CLOSED_FINAL_TP": "закрыта по финальному тейку",
        "POSITION_CLOSED_STOP_LOSS": "закрыта по стопу",
        "POSITION_CLOSED_EARLY_STOP": "закрыта по раннему стопу",
        "POSITION_CLOSED_REST_STOP_AFTER_PARTIAL": "закрыта rest stop после частичного тейка",
        "POSITION_CLOSED_TTL_CLOSE": "закрыта по TTL",
        "POSITION_CLOSED_MANUAL_CLOSE": "закрыта вручную",
        "POSITION_CLOSED_EXTERNAL": "закрыта вне системы",
    }

    return mapping.get(text, text.lower() if text else "")


def append_fill_line(
    lines: List[str],
    title: str,
    qty: Any,
    px: Any,
    ts: Any,
) -> None:
    qty_text = fmt_position_qty(qty)
    px_text = fmt_position_price(px)
    ts_text = fmt_position_time(ts)

    parts: List[str] = []

    if is_meaningful_value(qty_text):
        parts.append("qty {}".format(qty_text))

    if is_meaningful_value(px_text):
        parts.append("@ {}".format(px_text))

    if is_meaningful_value(ts_text):
        parts.append(ts_text)

    if parts:
        lines.append("{}: {}".format(title, " | ".join(parts)))



def format_position_block(title: str, item: Dict[str, Any], current: bool) -> str:
    if not item or not bool(item.get("exists")):
        return "{}\nнет данных".format(title)

    status = compact_position_status_ru(item.get("status"))
    symbol = str(item.get("symbol") or "").upper()
    side = str(item.get("side") or "").upper()
    exec_kind = compact_exec_kind(item.get("execution_kind"))

    qty_initial = item.get("qty")
    qty_exchange = item.get("exchange_size")
    qty_current = qty_exchange if not is_empty_value(qty_exchange) else qty_initial

    lines: List[str] = [
        title,
        "{} {}{}".format(symbol, side, " | " + status if status else ""),
    ]

    trade_id = item.get("trade_id")
    if is_meaningful_value(trade_id):
        lines.append("ID сделки: {}".format(trade_id))

    entry_line_parts = []
    entry_time = fmt_position_time(item.get("entry_filled_at"))
    entry_px = fmt_position_price(item.get("entry_avg_px"))

    if is_meaningful_value(entry_time):
        entry_line_parts.append(entry_time)
    if is_meaningful_value(entry_px):
        entry_line_parts.append("@ {}".format(entry_px))

    if entry_line_parts:
        lines += ["", "🟦 Вход", " ".join(entry_line_parts)]

    qty_parts = []
    qty_now = fmt_position_qty(qty_current)
    qty_init = fmt_position_qty(qty_initial)
    entry_value = fmt_position_money(item.get("entry_value_usdt"))

    if is_meaningful_value(qty_now):
        qty_parts.append("сейчас {}".format(qty_now))
    if is_meaningful_value(qty_init) and qty_init != qty_now:
        qty_parts.append("изначально {}".format(qty_init))
    if is_meaningful_value(entry_value):
        qty_parts.append("объём {} USDT".format(entry_value))

    if qty_parts:
        lines.append("Qty: " + " | ".join(qty_parts))

    if current:
        live_lines: List[str] = []

        mark_price = fmt_position_price(item.get("mark_price"))
        pnl_money = fmt_position_money(item.get("pnl_usd"))
        pnl_pct = fmt_position_pct(item.get("pnl_pct"))
        life = fmt_position_life(item.get("life_seconds"))

        if is_meaningful_value(mark_price):
            live_lines.append("Mark: {}".format(mark_price))

        pnl_parts = []
        if is_meaningful_value(pnl_money):
            pnl_parts.append("{} USDT".format(pnl_money))
        if is_meaningful_value(pnl_pct):
            pnl_parts.append(pnl_pct)

        if pnl_parts:
            live_lines.append("PnL: {}".format(" | ".join(pnl_parts)))

        if is_meaningful_value(life):
            live_lines.append("В рынке: {}".format(life))

        if live_lines:
            lines += ["", "🟨 Сейчас"] + live_lines
    else:
        close_lines: List[str] = []

        exit_time = fmt_position_time(item.get("exit_filled_at"))
        exit_px = fmt_position_price(item.get("exit_avg_px"))
        pnl_money = fmt_position_money(item.get("pnl_usd"))
        pnl_pct = fmt_position_pct(item.get("pnl_pct"))
        life = fmt_position_life(item.get("life_seconds"))
        reason = item.get("exit_reason")

        exit_parts = []
        if is_meaningful_value(exit_time):
            exit_parts.append(exit_time)
        if is_meaningful_value(exit_px):
            exit_parts.append("@ {}".format(exit_px))
        if exit_parts:
            close_lines.append("Выход: {}".format(" ".join(exit_parts)))

        result_parts = []
        if is_meaningful_value(pnl_money):
            result_parts.append("{} USDT".format(pnl_money))
        if is_meaningful_value(pnl_pct):
            result_parts.append(pnl_pct)
        if result_parts:
            close_lines.append("Итог: {}".format(" | ".join(result_parts)))

        if exec_kind:
            close_lines.append("Исполнение: {}".format(exec_kind))

        if is_meaningful_value(life):
            close_lines.append("Время в сделке: {}".format(life))

        if is_meaningful_value(reason):
            close_lines.append("Причина: {}".format(reason))

        if close_lines:
            lines += ["", "🟥 Закрытие"] + close_lines

    plan_lines: List[str] = []

    tp_plan = fmt_position_price(item.get("tp_px_plan"))
    sl_plan = fmt_position_price(item.get("sl_px_plan"))
    partial_tp = fmt_position_price(item.get("partial_tp_px_plan"))
    final_tp = fmt_position_price(item.get("final_tp_px_plan"))
    early_stop = fmt_position_price(item.get("early_stop_px_plan"))
    main_sl = fmt_position_price(item.get("main_sl_px_plan"))
    rest_stop = fmt_position_price(item.get("rest_stop_after_partial_px_plan"))
    mode = item.get("trade_management_mode")

    if is_meaningful_value(tp_plan) or is_meaningful_value(sl_plan):
        plan_lines.append("TP/SL: {} / {}".format(tp_plan, sl_plan))

    if is_meaningful_value(partial_tp):
        plan_lines.append("Partial TP: {}".format(partial_tp))

    if is_meaningful_value(final_tp):
        plan_lines.append("Final TP: {}".format(final_tp))

    stop_parts = []
    if is_meaningful_value(early_stop):
        stop_parts.append("early {}".format(early_stop))
    if is_meaningful_value(main_sl):
        stop_parts.append("main {}".format(main_sl))
    if is_meaningful_value(rest_stop):
        stop_parts.append("rest {}".format(rest_stop))

    if stop_parts:
        plan_lines.append("Stops: " + " | ".join(stop_parts))

    if is_meaningful_value(mode):
        plan_lines.append("Mode: {}".format(mode))

    if plan_lines:
        lines += ["", "🎯 План"] + plan_lines

    exec_lines: List[str] = []

    append_fill_line(
        exec_lines,
        "Partial TP",
        item.get("partial_tp_qty_filled"),
        item.get("partial_tp_avg_px"),
        item.get("partial_tp_first_ts"),
    )
    append_fill_line(
        exec_lines,
        "Final TP",
        item.get("final_tp_qty_filled"),
        item.get("final_tp_avg_px"),
        item.get("final_tp_first_ts"),
    )
    append_fill_line(
        exec_lines,
        "Full TP",
        item.get("take_profit_qty_filled"),
        item.get("take_profit_avg_px"),
        item.get("take_profit_first_ts"),
    )
    append_fill_line(
        exec_lines,
        "Main SL",
        item.get("stop_loss_qty_filled"),
        item.get("stop_loss_avg_px"),
        item.get("stop_loss_first_ts"),
    )
    append_fill_line(
        exec_lines,
        "Early stop",
        item.get("early_stop_qty_filled"),
        item.get("early_stop_avg_px"),
        item.get("early_stop_first_ts"),
    )
    append_fill_line(
        exec_lines,
        "Rest stop",
        item.get("rest_stop_after_partial_qty_filled"),
        item.get("rest_stop_after_partial_avg_px"),
        item.get("rest_stop_after_partial_first_ts"),
    )
    append_fill_line(
        exec_lines,
        "TTL close",
        item.get("ttl_close_qty_filled"),
        item.get("ttl_close_avg_px"),
        item.get("ttl_close_first_ts"),
    )

    if exec_kind and current:
        exec_lines.insert(0, "Текущее состояние: {}".format(exec_kind))

    if exec_lines:
        lines += ["", "✅ Исполнения"] + exec_lines

    lifecycle_lines: List[str] = []

    append_time_line(lifecycle_lines, "Early stop expires", item.get("early_stop_expires_at"))
    append_time_line(lifecycle_lines, "Early stop replaced", item.get("early_stop_replaced_at"))
    append_time_line(lifecycle_lines, "Partial TP handled", item.get("partial_tp_handled_at"))
    append_time_line(lifecycle_lines, "Protective cleanup", item.get("protective_cleanup_done_at"))
    append_time_line(lifecycle_lines, "Closed cleanup", item.get("closed_cleanup_done_at"))
    append_time_line(lifecycle_lines, "WS updated", item.get("ws_lifecycle_updated_at"))

    ws_error = item.get("ws_lifecycle_last_error")
    if is_meaningful_value(ws_error):
        lifecycle_lines.append("WS error: {}".format(ws_error))

    if lifecycle_lines:
        lines += ["", "🧷 Lifecycle"] + lifecycle_lines

    service_lines: List[str] = []

    fee = fmt_position_money(item.get("fee_usd"))
    fills_count = item.get("fills_count")

    if is_meaningful_value(fee):
        service_lines.append("Fee: {} USDT".format(fee))

    if is_meaningful_value(fills_count):
        service_lines.append("Fills: {}".format(fills_count))

    if service_lines:
        lines += ["", "📎 Сервис"] + service_lines

    return "\n".join(lines)


def format_position_output(code: int, out: str, command_text: str) -> str:
    if code != 0:
        body = compact_raw_output(out, max_lines=55)
        if not body:
            body = "EMPTY_OUTPUT"
        return "❌ Позиция недоступна\n\n{}".format(body)

    report = parse_position_json_output(out)

    if not report:
        body = compact_raw_output(out, max_lines=55)
        if not body:
            body = "Нет данных."
        return "📍 Позиция\n\n{}".format(body)

    symbol = str(report.get("symbol") or "").upper()
    exchange_error = report.get("exchange_error")

    blocks: List[str] = [
        "📍 Позиция {}".format(symbol),
        "",
        format_position_block("Текущая позиция", report.get("current") or {}, current=True),
    ]

    closed_items = report.get("last_closed") or []

    if isinstance(closed_items, dict):
        closed_items = [closed_items]

    useful_closed = []
    for item in closed_items:
        if isinstance(item, dict) and bool(item.get("exists")):
            useful_closed.append(item)

    if useful_closed:
        blocks.append("")
        blocks.append("━━━━━━━━━━━━━━━━━━━━")
        blocks.append("Последние закрытые")

        for i, item in enumerate(useful_closed, start=1):
            blocks.append("")
            blocks.append(
                format_position_block(
                    "Закрытая #{}".format(i),
                    item,
                    current=False,
                )
            )

    if exchange_error:
        blocks.append("")
        blocks.append("⚠️ Bybit error: {}".format(exchange_error))

    return "\n".join(blocks)

def find_backtest_out_dir(out: str) -> Optional[Path]:
    prefix = "OUT_DIR:"

    for raw in reversed(str(out or "").splitlines()):
        line = raw.strip()

        if not line.startswith(prefix):
            continue

        value = line[len(prefix):].strip()

        if not value:
            continue

        return Path(value)

    return None


def fmt_backtest_number(
    value: Any,
    digits: int = 2,
    suffix: str = "",
    signed: bool = False,
) -> str:
    if is_empty_value(value):
        return "-"

    try:
        number = float(value)
    except Exception:
        return str(value)

    if signed:
        text = ("{:+." + str(int(digits)) + "f}").format(number)
    else:
        text = ("{:." + str(int(digits)) + "f}").format(number)

    return text + str(suffix)


def fmt_backtest_ratio_pct(value: Any, digits: int = 2) -> str:
    if is_empty_value(value):
        return "-"

    try:
        number = float(value) * 100.0
    except Exception:
        return str(value)

    return ("{:+." + str(int(digits)) + "f}%").format(number)


def fmt_backtest_time(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")

        if pd.isna(ts):
            return str(value)

        return ts.strftime("%m-%d %H:%M")
    except Exception:
        return str(value)


def fmt_backtest_price(value: Any) -> str:
    if is_empty_value(value):
        return "-"

    try:
        number = float(value)
    except Exception:
        return str(value)

    if abs(number) >= 1000:
        return "{:.2f}".format(number)

    if abs(number) >= 10:
        return "{:.4f}".format(number)

    if abs(number) >= 1:
        return "{:.5f}".format(number)

    return "{:.8f}".format(number)


def backtest_exit_reason_ru(value: Any) -> str:
    text = str(value or "").strip().upper()

    mapping = {
        "TP": "TP",
        "SL": "SL",
        "SL_SAME_M1": "SL same M1",
        "TTL": "TTL",
    }

    return mapping.get(text, text or "-")


def format_backtest_trade_card(
    row: Dict[str, Any],
    number: int,
) -> str:
    symbol = str(row.get("symbol") or "").strip().upper()
    side = str(row.get("side") or "").strip().upper()
    reason = backtest_exit_reason_ru(row.get("exit_reason"))

    net_ret = row.get("net_ret")

    try:
        positive = float(net_ret) > 0.0
    except Exception:
        positive = False

    icon = "🟢" if positive else "🔴"

    entry_ts = fmt_backtest_time(row.get("entry_ts"))
    exit_ts = fmt_backtest_time(row.get("exit_ts"))

    entry_px = fmt_backtest_price(row.get("entry_px"))
    exit_px = fmt_backtest_price(row.get("exit_px"))
    tp_px = fmt_backtest_price(row.get("tp_px"))
    sl_px = fmt_backtest_price(row.get("sl_px"))

    capital_before = fmt_backtest_number(
        row.get("capital_before"),
        digits=2,
        suffix=" USDT",
    )
    capital_after = fmt_backtest_number(
        row.get("capital_after"),
        digits=2,
        suffix=" USDT",
    )

    result_pct = fmt_backtest_ratio_pct(net_ret)

    lines = [
        "{} #{} {} {} | {}".format(
            icon,
            int(number),
            symbol,
            side,
            reason,
        ),
        "{} → {}".format(entry_ts, exit_ts),
        "Вход {} | Выход {}".format(entry_px, exit_px),
        "TP {} | SL {}".format(tp_px, sl_px),
        "Результат: {}".format(result_pct),
        "Капитал: {} → {}".format(capital_before, capital_after),
    ]

    return "\n".join(lines)


def load_backtest_result_from_out(
    out: str,
) -> Tuple[Optional[Path], Dict[str, Any], pd.DataFrame]:
    out_dir = find_backtest_out_dir(out)

    if out_dir is None:
        return None, {}, pd.DataFrame()

    report_path = out_dir / "report.json"
    trades_path = out_dir / "trades.csv"

    report: Dict[str, Any] = {}

    if report_path.exists():
        try:
            loaded = json.loads(
                report_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            if isinstance(loaded, dict):
                report = loaded
        except Exception:
            report = {}

    trades = pd.DataFrame()

    if trades_path.exists():
        try:
            trades = pd.read_csv(trades_path)
        except Exception:
            trades = pd.DataFrame()

    return out_dir, report, trades


def format_backtest_output(code: int, out: str, command_text: str) -> str:
    if code != 0:
        body = compact_raw_output(out, max_lines=45)

        if not body:
            body = "EMPTY_OUTPUT"

        return (
            "❌ Backtest завершился с ошибкой\n\n"
            "Код: {}\n\n"
            "{}"
        ).format(
            code,
            body,
        )

    out_dir, report, trades = load_backtest_result_from_out(out)

    if out_dir is None or not report:
        body = compact_raw_output(out, max_lines=45)

        if not body:
            body = "Результат backtest не найден."

        return "🧪 Backtest\n\n{}".format(body)

    args_data = report.get("args")
    if not isinstance(args_data, dict):
        args_data = {}

    summary = report.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    rows_info = report.get("rows")
    if not isinstance(rows_info, dict):
        rows_info = {}

    start = str(args_data.get("start") or "-")
    end = str(args_data.get("end") or "-")

    initial_capital = args_data.get("capital")

    if not trades.empty and "capital_before" in trades.columns:
        first_capital_values = pd.to_numeric(
            trades["capital_before"],
            errors="coerce",
        ).dropna()

        if not first_capital_values.empty:
            initial_capital = float(first_capital_values.iloc[0])

    final_capital = summary.get("final_capital")
    total_return_pct = summary.get("total_return_pct")
    win_rate = summary.get("win_rate")
    profit_factor = summary.get("profit_factor")
    max_drawdown_pct = summary.get("max_drawdown_pct")
    mean_net_ret = summary.get("mean_net_ret")
    median_net_ret = summary.get("median_net_ret")

    lines: List[str] = [
        "🧪 Backtest завершён",
        "",
        "Период:",
        "{} → {}".format(start, end),
        "",
        "📊 Результат",
        "Сделок: {}".format(summary.get("trades_taken", len(trades))),
        "Капитал: {} → {}".format(
            fmt_backtest_number(
                initial_capital,
                digits=2,
                suffix=" USDT",
            ),
            fmt_backtest_number(
                final_capital,
                digits=2,
                suffix=" USDT",
            ),
        ),
        "Доходность: {}".format(
            fmt_backtest_ratio_pct(total_return_pct)
        ),
        "Win rate: {}".format(
            fmt_backtest_ratio_pct(win_rate)
        ),
        "Profit factor: {}".format(
            fmt_backtest_number(profit_factor, digits=3)
        ),
        "Макс. просадка: {}".format(
            fmt_backtest_ratio_pct(max_drawdown_pct)
        ),
        "Средняя сделка: {}".format(
            fmt_backtest_ratio_pct(mean_net_ret)
        ),
        "Медианная сделка: {}".format(
            fmt_backtest_ratio_pct(median_net_ret)
        ),
        "",
        "📌 Исходы",
        "TP: {} | SL: {} | TTL: {}".format(
            summary.get("tp_count", 0),
            summary.get("sl_count", 0),
            summary.get("ttl_count", 0),
        ),
        "Пропущено из-за занятого слота: {}".format(
            summary.get("skipped_overlap", 0)
        ),
        "Пропущено blacklist: {}".format(
            summary.get("skipped_dynamic_blacklist", 0)
        ),
        "",
        "⚙️ Параметры",
        "G2 {} | G4 {} | G5.1 {} | G5.3 {}".format(
            args_data.get("gate2", "-"),
            args_data.get("gate4", "-"),
            args_data.get("gate5_1", "-"),
            args_data.get("gate5_3", "-"),
        ),
        "Chulan {} | whitelist {} | conditional {} | slots {}".format(
            args_data.get("chulan", "-"),
            int(bool(args_data.get("side_aware_whitelist", False))),
            int(bool(args_data.get("conditional_side_aware_whitelist", False))),
            args_data.get("slots", "-"),
        ),
        "Risk cap full SL: {}".format(
            fmt_backtest_ratio_pct(args_data.get("max_full_sl_capital_risk"))
        ),
        "Кандидаты: raw {} | passed {} | selected {}".format(
            rows_info.get("raw_candidates", 0),
            rows_info.get("passed_thresholds", 0),
            rows_info.get("selected_one_per_h4", 0),
        ),
    ]

    if trades.empty:
        lines += [
            "",
            "Сделок за выбранный интервал нет.",
        ]

        return "\n".join(lines)

    trades_work = trades.copy()

    if "entry_ts" in trades_work.columns:
        trades_work["entry_ts"] = pd.to_datetime(
            trades_work["entry_ts"],
            utc=True,
            errors="coerce",
        )
        trades_work = trades_work.sort_values(
            ["entry_ts"],
            ascending=[True],
        )

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "📋 Позиции",
    ]

    for index, row in enumerate(
        trades_work.to_dict(orient="records"),
        start=1,
    ):
        lines += [
            "",
            format_backtest_trade_card(
                row=row,
                number=index,
            ),
        ]

    return "\n".join(lines)


def format_default_output(code: int, out: str, command_text: str) -> str:
    body = compact_raw_output(out, max_lines=45)

    if not body:
        body = "EMPTY_OUTPUT"

    icon = "✅" if code == 0 else "❌"

    return (
        "{} {}\n\n"
        "{}"
    ).format(
        icon,
        command_text,
        body,
    )


def format_command_result(cmd: str, parts: List[str], code: int, out: str) -> str:
    command_text = " ".join(parts)

    if cmd in {"/status", "status"}:
        return format_status_output(code, out, command_text)

    if cmd in {"/history", "history"}:
        return format_history_output(code, out, command_text)

    if cmd in {"/position", "position", "/pos", "pos"}:
        return format_position_output(code, out, command_text)

    if cmd in {"/backtest", "backtest"}:
        return format_backtest_output(code, out, command_text)

    return format_default_output(code, out, command_text)



BACKTEST_WIZARD_KIND = "backtest"

BACKTEST_WIZARD_STEPS = [
    {
        "key": "start_date",
        "title": "Дата начала",
        "example": "2026-05-01",
        "default": "",
    },
    {
        "key": "start_time",
        "title": "Время начала",
        "example": "12:00",
        "default": "",
    },
    {
        "key": "end_date",
        "title": "Дата конца",
        "example": "2026-05-09",
        "default": "",
    },
    {
        "key": "end_time",
        "title": "Время конца",
        "example": "12:00",
        "default": "",
    },
    {
        "key": "change_defaults",
        "title": "Изменить дефолтные параметры",
        "example": "0",
        "default": "0",
    },
    {
        "key": "conditional_side_aware_whitelist",
        "title": "Использовать conditional whitelist",
        "example": "1",
        "default": "1",
    },
    {
        "key": "max_full_sl_risk_pct",
        "title": "Максимальный риск полного MAIN_SL, %",
        "example": "6",
        "default": "6",
    },
    {
        "key": "gate2",
        "title": "Gate2 threshold",
        "example": "0.70",
        "default": str(config.GATE2_THR),
    },
    {
        "key": "gate4",
        "title": "Gate4 threshold",
        "example": "0.57",
        "default": str(config.GATE4_THR),
    },
    {
        "key": "gate5_1",
        "title": "Gate5.1 threshold",
        "example": "0.10",
        "default": str(config.GATE5_1_THR),
    },
    {
        "key": "gate5_3",
        "title": "Gate5.3 threshold",
        "example": "0.54",
        "default": str(config.GATE5_3_THR),
    },
    {
        "key": "chulan",
        "title": "Использовать чулан",
        "example": "0",
        "default": "0",
    },
    {
        "key": "side_aware_whitelist",
        "title": "Использовать side-aware whitelist",
        "example": "1",
        "default": "1",
    },
    {
        "key": "slots",
        "title": "Количество слотов",
        "example": "1",
        "default": "1",
    },
    {
        "key": "write_blacklist",
        "title": "Записать dynamic blacklist",
        "example": "0",
        "default": "0",
    },
    {
        "key": "reset_blacklist",
        "title": "Сбросить backtest blacklist",
        "example": "0",
        "default": "0",
    },
    {
        "key": "sync_m1",
        "title": "Синхронизировать минутки перед backtest",
        "example": "1",
        "default": "1",
    },
    {
        "key": "host",
        "title": "Хост запуска",
        "example": "win",
        "default": DEFAULT_RUN_HOST,
    },
]


BACKTEST_MODE_WIZARD_KIND = "backtest_mode"
BACKTEST_SYMBOL_WIZARD_KIND = "backtest_symbol"

BACKTEST_MODE_SYSTEM_TEXT = "Проверить систему"
BACKTEST_MODE_SYMBOLS_TEXT = "Проверить символ(ы)"

BACKTEST_SYMBOL_DONE_TEXT = "✅ Готово"
BACKTEST_SYMBOL_RESET_TEXT = "♻️ Сбросить выбор"
BACKTEST_SYMBOL_BACK_TO_LIST_TEXT = "↩️ К списку"

BACKTEST_SYMBOL_SIDE_LONG_TEXT = "LONG"
BACKTEST_SYMBOL_SIDE_SHORT_TEXT = "SHORT"
BACKTEST_SYMBOL_SIDE_BOTH_TEXT = "BOTH"

BACKTEST_SYMBOL_DEFAULT_THRESHOLDS_TEXT = "✅ Дефолтные пороги"
BACKTEST_SYMBOL_MANUAL_THRESHOLDS_TEXT = "✏️ Ввести вручную"

BACKTEST_SYMBOL_DEFAULT_GATE2 = "0.70"
BACKTEST_SYMBOL_DEFAULT_GATE2_SIDE_MARGIN_MIN = "0.30"
BACKTEST_SYMBOL_DEFAULT_GATE4 = "0.57"
BACKTEST_SYMBOL_DEFAULT_GATE5_1 = "0.10"
BACKTEST_SYMBOL_DEFAULT_GATE5_3 = "0.54"
BACKTEST_SYMBOL_DEFAULT_CAPITAL = "100"

BACKTEST_SYMBOL_FIXED_CHULAN = "0"
BACKTEST_SYMBOL_FIXED_SIDE_AWARE_WHITELIST = "0"
BACKTEST_SYMBOL_FIXED_CONDITIONAL_WHITELIST = "1"
BACKTEST_SYMBOL_FIXED_MAX_FULL_SL_CAPITAL_RISK = "0.07"
BACKTEST_SYMBOL_FIXED_SLOTS = "1"
BACKTEST_SYMBOL_FIXED_WRITE_BLACKLIST = "0"
BACKTEST_SYMBOL_FIXED_RESET_BLACKLIST = "0"
BACKTEST_SYMBOL_FIXED_ENTRY_DELAY_SECONDS = "90"
BACKTEST_SYMBOL_FIXED_HOST = DEFAULT_RUN_HOST


def backtest_mode_keyboard() -> Dict[str, Any]:
    return {
        "keyboard": [
            [BACKTEST_MODE_SYSTEM_TEXT],
            [BACKTEST_MODE_SYMBOLS_TEXT],
            ["❌ Отмена"],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }


def get_system_backtest_symbols() -> List[str]:
    m1_dir = ROOT / "data" / "m1_4"
    h4_dir = ROOT / "data" / "h4_3"

    if not m1_dir.exists():
        return []

    out: List[str] = []

    for path in sorted(m1_dir.glob("*.parquet")):
        symbol = path.stem.strip().upper()

        if not symbol:
            continue

        h4_path = h4_dir / "{}.parquet".format(symbol)

        if not h4_path.exists():
            continue

        out.append(symbol)

    return sorted(set(out))


def get_symbol_wizard_data(session: Dict[str, Any]) -> Dict[str, Any]:
    data = session.get("data")
    if not isinstance(data, dict):
        data = {}
        session["data"] = data

    selected = data.get("selected")
    if not isinstance(selected, dict):
        selected = {}
        data["selected"] = selected

    selected_order = data.get("selected_order")
    if not isinstance(selected_order, list):
        selected_order = []
        data["selected_order"] = selected_order

    return data


def format_selected_symbol_item(symbol: str, side_code: str) -> str:
    symbol_u = str(symbol or "").strip().upper()
    side = str(side_code or "").strip().upper()

    if side == "L":
        return "{} LONG".format(symbol_u)

    if side == "S":
        return "{} SHORT".format(symbol_u)

    return "{} BOTH".format(symbol_u)


def format_selected_symbols_from_data(data: Dict[str, Any]) -> str:
    selected = data.get("selected")
    selected_order = data.get("selected_order")

    if not isinstance(selected, dict):
        selected = {}

    if not isinstance(selected_order, list):
        selected_order = sorted(selected.keys())

    items: List[str] = []

    for symbol in selected_order:
        symbol_u = str(symbol or "").strip().upper()
        if not symbol_u or symbol_u not in selected:
            continue

        side_code = str(selected.get(symbol_u) or "").strip().upper()

        if side_code == "L":
            items.append("{}:L".format(symbol_u))
        elif side_code == "S":
            items.append("{}:S".format(symbol_u))
        else:
            items.append(symbol_u)

    return ",".join(items)


def format_selected_symbols_human(data: Dict[str, Any]) -> str:
    selected = data.get("selected")
    selected_order = data.get("selected_order")

    if not isinstance(selected, dict):
        selected = {}

    if not isinstance(selected_order, list):
        selected_order = sorted(selected.keys())

    items: List[str] = []

    for symbol in selected_order:
        symbol_u = str(symbol or "").strip().upper()
        if not symbol_u or symbol_u not in selected:
            continue

        items.append(
            format_selected_symbol_item(
                symbol=symbol_u,
                side_code=str(selected.get(symbol_u) or ""),
            )
        )

    if not items:
        return "пока ничего не выбрано"

    return ", ".join(items)


def backtest_symbol_select_keyboard(data: Dict[str, Any]) -> Dict[str, Any]:
    symbols = get_system_backtest_symbols()
    rows: List[List[str]] = []

    row: List[str] = []
    for symbol in symbols:
        row.append(symbol)

        if len(row) >= 3:
            rows.append(row)
            row = []

    if row:
        rows.append(row)

    rows.append([BACKTEST_SYMBOL_DONE_TEXT, BACKTEST_SYMBOL_RESET_TEXT])
    rows.append(["❌ Отмена"])

    return {
        "keyboard": rows,
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }


def backtest_symbol_side_keyboard() -> Dict[str, Any]:
    return {
        "keyboard": [
            [
                BACKTEST_SYMBOL_SIDE_LONG_TEXT,
                BACKTEST_SYMBOL_SIDE_SHORT_TEXT,
                BACKTEST_SYMBOL_SIDE_BOTH_TEXT,
            ],
            [BACKTEST_SYMBOL_BACK_TO_LIST_TEXT],
            ["❌ Отмена"],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }


def backtest_symbol_threshold_mode_keyboard() -> Dict[str, Any]:
    return {
        "keyboard": [
            [BACKTEST_SYMBOL_DEFAULT_THRESHOLDS_TEXT],
            [BACKTEST_SYMBOL_MANUAL_THRESHOLDS_TEXT],
            ["❌ Отмена"],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }


ADD_SYMBOL_WIZARD_KIND = "add_symbol_wizard"
ADD_SYMBOL_EXISTING_SYMBOLS_TEXT = "📋 Существующие символы"


def format_existing_symbols_for_add_symbol() -> str:
    symbols = get_system_backtest_symbols()

    if not symbols:
        return "В системе пока нет локальных символов."

    lines: List[str] = []

    for i in range(0, len(symbols), 4):
        lines.append(", ".join(symbols[i:i + 4]))

    return "\n".join(lines)


def add_symbol_symbol_keyboard() -> Dict[str, Any]:
    symbols = get_system_backtest_symbols()
    rows: List[List[str]] = []

    row: List[str] = []

    for symbol in symbols:
        row.append(symbol)

        if len(row) >= 3:
            rows.append(row)
            row = []

    if row:
        rows.append(row)

    rows.append([ADD_SYMBOL_EXISTING_SYMBOLS_TEXT])
    rows.append(["❌ Отмена"])

    return {
        "keyboard": rows,
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }



def normalize_add_symbol_for_tg(raw: str) -> str:
    symbol = str(raw or "").strip().upper()
    symbol = symbol.replace(" ", "").replace("_", "").replace("/", "").replace("-", "")

    if not symbol:
        return ""

    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"

    return symbol


def start_add_symbol_wizard(chat_id: int) -> str:
    session = {
        "kind": ADD_SYMBOL_WIZARD_KIND,
        "step": "symbol",
        "data": {
            "valid_days": 60,
        },
        "created_at_utc": str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
    }

    set_chat_session(chat_id, session)

    return (
        "➕ Добавление символа\n\n"
        "Существующие символы показаны кнопками ниже.\n"
        "Новый символ введи сообщением.\n\n"
        "Формат: SYMBOLUSDT / SYMBOL / symbol\n"
        "Пример: ICNTUSDT"
    )


def is_add_symbol_wizard_active(chat_id: int) -> bool:
    session = get_chat_session(chat_id)
    if not session:
        return False

    return session.get("kind") == ADD_SYMBOL_WIZARD_KIND


def get_add_symbol_confirm_text(symbol: str, valid_days: int) -> str:
    return (
        "➕ Символ найден на Bybit futures.\n\n"
        "Символ: {symbol}\n\n"
        "Запустить добавление?"
    ).format(
        symbol=symbol,
    )


def build_add_symbol_service_args(symbol: str, valid_days: int) -> List[str]:
    run_tag = "tg_add_symbol_{}_{}".format(
        normalize_add_symbol_for_tg(symbol).lower(),
        time.strftime("%Y%m%d_%H%M%S", time.gmtime()),
    )

    return [
        "add-symbol",
        normalize_add_symbol_for_tg(symbol),
        "2000-01-01",
        "00:00",
        "2099-01-01",
        "00:00",
        "--run-tag",
        run_tag,
        "--timeout-sec",
        str(TG_BACKTEST_TIMEOUT_SECONDS),
        "--valid-days",
        str(int(valid_days)),
        "--execute",
    ]


def extract_add_symbol_field(out: str, field_name: str) -> str:
    prefix = str(field_name).strip() + ":"
    for raw_line in str(out or "").splitlines():
        line = str(raw_line or "").strip()
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def extract_int_from_text(value: str) -> Optional[int]:
    match = re.search(r"(\\d+)", str(value or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def run_add_symbol_precheck(symbol: str) -> Dict[str, Any]:
    safe_symbol = normalize_add_symbol_for_tg(symbol)
    json_out = ROOT / "online" / "_tmp_add_symbol_precheck_{}.json".format(safe_symbol.lower())

    if json_out.exists():
        try:
            json_out.unlink()
        except Exception:
            pass

    cmd = [
        sys.executable,
        "-u",
        "online/new/actions/control/symbol_onboarding_decision.py",
        "--symbol",
        safe_symbol,
        "--mode",
        "add",
        "--run-tag",
        "tg_precheck_{}".format(safe_symbol.lower()),
        "--json-out",
        str(json_out),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=build_base_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    if json_out.exists():
        try:
            data = json.loads(json_out.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                data["_returncode"] = int(proc.returncode)
                data["_stdout"] = str(proc.stdout or "")
                return data
        except Exception:
            pass

    return {
        "allowed": False,
        "decision": "PRECHECK_FAILED",
        "message": str(proc.stdout or "").strip(),
        "_returncode": int(proc.returncode),
        "_stdout": str(proc.stdout or ""),
    }


def format_add_symbol_precheck_reject(symbol: str, decision: Dict[str, Any]) -> str:
    decision_code = str(decision.get("decision") or "").strip()
    message = str(decision.get("message") or "").strip()

    if decision_code == "SYMBOL_ALREADY_EXISTS":
        return (
            "ℹ️ Символ уже есть в системе.\n\n"
            "Символ: {symbol}\n\n"
            "Его можно проверить через 🧪 Бэктест."
        ).format(symbol=symbol)

    if decision_code == "SYMBOL_NOT_AVAILABLE_ON_BYBIT":
        return (
            "❌ Символа нет на Bybit futures.\n\n"
            "Символ: {symbol}\n\n"
            "Введи другой тикер или проверь, что это USDT perpetual."
        ).format(symbol=symbol)

    if decision_code == "TOO_LITTLE_HISTORY":
        days = (
            decision.get("history_days")
            or decision.get("train_days")
            or decision.get("available_days")
            or extract_int_from_text(message)
        )

        if days is None:
            days_text = "недостаточно"
        else:
            days_text = str(days)

        return (
            "❌ Символ недавно появился на futures.\n\n"
            "Символ: {symbol}\n"
            "Доступно истории: {days} дней\n"
            "Минимум для добавления: 150 дней\n\n"
            "Выбери другой символ."
        ).format(
            symbol=symbol,
            days=days_text,
        )

    return (
        "❌ Символ не прошёл проверку.\n\n"
        "Символ: {symbol}\n"
        "Причина: {reason}"
    ).format(
        symbol=symbol,
        reason=decision_code or message or "UNKNOWN",
    )


def format_add_symbol_wizard_output(code: int, out: str, symbol: str) -> str:
    status = extract_add_symbol_field(out, "STATUS")
    decision = extract_add_symbol_field(out, "DECISION")
    next_action = extract_add_symbol_field(out, "NEXT_ACTION")
    summary = extract_add_symbol_field(out, "SUMMARY")
    run_root = extract_add_symbol_field(out, "RUN_ROOT")
    oos_start = extract_add_symbol_field(out, "OOS_START")
    oos_end = extract_add_symbol_field(out, "OOS_END")

    if int(code) != 0:
        run_tag = extract_add_symbol_field(out, "RUN_TAG")
        if not run_tag:
            run_tag = extract_add_symbol_field(out, "run_tag")

        return (
            "❌ Добавление символа не завершилось.\n\n"
            "Символ: {symbol}\n"
            "Код: {code}\n\n"
            "Технические детали записаны в логах проекта."
        ).format(
            symbol=symbol,
            code=int(code),
        )

    if status == "ALREADY_EXISTS":
        return (
            "ℹ️ Символ уже есть в системе.\n\n"
            "Символ: {symbol}\n"
            "Следующее действие: можно протестировать его в разделе Backtest."
        ).format(symbol=symbol)

    if status == "REJECTED":
        return (
            "❌ Символ не добавлен.\n\n"
            "Символ: {symbol}\n"
            "Причина: {decision}\n"
            "Следующее действие: {next_action}"
        ).format(
            symbol=symbol,
            decision=decision or "UNKNOWN",
            next_action=next_action or "REJECT",
        )

    if status in {"SUCCESS", "DRY_RUN"}:
        if status == "SUCCESS":
            return (
                "✅ Символ добавлен.\n\n"
                "Символ: {symbol}\n\n"
                "Теперь его можно проверить через 🧪 Бэктест."
            ).format(symbol=symbol)

        return (
            "✅ Проверочный dry-run завершён.\n\n"
            "Символ: {symbol}"
        ).format(symbol=symbol)

    return (
        "✅ Команда добавления символа завершилась.\n\n"
        "Символ: {symbol}\n"
        "STATUS: {status}\n\n"
        "{tail}"
    ).format(
        symbol=symbol,
        status=status or "UNKNOWN",
        tail="\n".join(str(out or "").splitlines()[-20:]),
    )



def extract_arg_value(parts: List[str], name: str, default: str = "") -> str:
    needle = str(name)
    values = [str(x) for x in parts]

    for i, value in enumerate(values):
        if value == needle and i + 1 < len(values):
            return values[i + 1]

    return str(default)


def start_add_symbol_background_job(chat_id: int, parts: List[str], symbol: str) -> str:
    run_tag = extract_arg_value(parts, "--run-tag", "")

    def worker() -> None:
        try:
            log_event(
                "TG_ADD_SYMBOL_BACKGROUND_START chat_id={} symbol={} run_tag={}".format(
                    chat_id,
                    symbol,
                    run_tag,
                )
            )

            code, out = run_service_status(parts)

            answer = format_add_symbol_wizard_output(
                code=code,
                out=out,
                symbol=symbol,
            )

            log_event(
                "TG_ADD_SYMBOL_BACKGROUND_DONE chat_id={} symbol={} run_tag={} code={}".format(
                    chat_id,
                    symbol,
                    run_tag,
                    int(code),
                )
            )

        except Exception as exc:
            answer = (
                "❌ Ошибка фонового добавления символа.\n\n"
                "Символ: {symbol}\n"
                "RUN_TAG: {run_tag}\n"
                "Ошибка: {error}"
            ).format(
                symbol=symbol,
                run_tag=run_tag or "-",
                error=exc,
            )

            log_event(
                "TG_ADD_SYMBOL_BACKGROUND_ERROR chat_id={} symbol={} run_tag={} error={!r}".format(
                    chat_id,
                    symbol,
                    run_tag,
                    exc,
                )
            )

        try:
            send_message(chat_id, answer)
        except Exception as send_exc:
            log_event(
                "TG_ADD_SYMBOL_BACKGROUND_SEND_ERROR chat_id={} symbol={} run_tag={} error={!r}".format(
                    chat_id,
                    symbol,
                    run_tag,
                    send_exc,
                )
            )

    thread = threading.Thread(
        target=worker,
        name="tg_add_symbol_{}".format(str(symbol).lower()),
        daemon=True,
    )
    thread.start()

    return (
        "✅ Добавление символа запущено в фоне.\n\n"
        "Символ: {symbol}\n\n"
        "Бот остаётся доступен. После завершения я пришлю результат отдельным сообщением."
    ).format(
        symbol=symbol,
    )
def handle_add_symbol_wizard(chat_id: int, text: str) -> Optional[str]:
    session = get_chat_session(chat_id)
    if not session or session.get("kind") != ADD_SYMBOL_WIZARD_KIND:
        return None

    raw = str(text or "").strip()

    if raw in {"/cancel", "cancel", "отмена", "❌ Отмена"}:
        clear_chat_session(chat_id)
        return "❌ Добавление символа отменено."

    data = session.get("data")
    if not isinstance(data, dict):
        data = {}

    step = str(session.get("step") or "symbol")

    if step == "symbol":
        if raw == ADD_SYMBOL_EXISTING_SYMBOLS_TEXT:
            return (
                "📋 Существующие символы\n\n"
                "{}\n\n"
                "Чтобы добавить новый символ, введи его сообщением."
            ).format(format_existing_symbols_for_add_symbol())

        symbol = normalize_add_symbol_for_tg(raw)
        if not symbol:
            return "❌ Символ пустой. Введи тикер, например ICNTUSDT."

        if symbol in set(get_system_backtest_symbols()):
            return (
                "ℹ️ Символ уже есть в системе.\n\n"
                "Символ: {}\n\n"
                "Его можно проверить через 🧪 Бэктест.\n"
                "Чтобы добавить новый символ, введи другой тикер."
            ).format(symbol)

        decision = run_add_symbol_precheck(symbol)

        if not bool(decision.get("allowed")):
            return format_add_symbol_precheck_reject(symbol=symbol, decision=decision)

        data["symbol"] = symbol
        data["valid_days"] = int(data.get("valid_days") or 60)
        session["data"] = data
        session["step"] = "confirm"
        set_chat_session(chat_id, session)

        return get_add_symbol_confirm_text(
            symbol=symbol,
            valid_days=int(data.get("valid_days") or 60),
        )

    if step == "capital":
        try:
            data["capital"] = parse_symbol_capital_value(raw)
        except Exception as exc:
            return "❌ {}\n\n{}".format(exc, get_symbol_step_text(session))

        session["step"] = "confirm"
        session["data"] = data
        set_chat_session(chat_id, session)
        return get_symbol_confirm_text(data)

    if step == "confirm":
        if raw.lower() in {
            "/go",
            "go",
            "✅ запустить",
            "запустить",
            "run",
        }:
            symbol = normalize_add_symbol_for_tg(str(data.get("symbol") or ""))
            valid_days = int(data.get("valid_days") or 60)

            clear_chat_session(chat_id)

            parts = build_add_symbol_service_args(
                symbol=symbol,
                valid_days=valid_days,
            )

            return start_add_symbol_background_job(
                chat_id=chat_id,
                parts=parts,
                symbol=symbol,
            )

        return get_add_symbol_confirm_text(
            symbol=normalize_add_symbol_for_tg(str(data.get("symbol") or "")),
            valid_days=int(data.get("valid_days") or 60),
        )

    clear_chat_session(chat_id)
    return "❌ Состояние мастера добавления символа повреждено. Запусти /add_symbol_wizard заново."
def get_active_session_keyboard(chat_id: int) -> Optional[Dict[str, Any]]:
    session = get_chat_session(chat_id)
    if not session:
        return None

    kind = str(session.get("kind") or "")

    if kind == ADD_SYMBOL_WIZARD_KIND:
        step = str(session.get("step") or "symbol")

        if step == "symbol":
            return add_symbol_symbol_keyboard()

        return wizard_control_keyboard()

    if kind == BACKTEST_MODE_WIZARD_KIND:
        return backtest_mode_keyboard()

    if kind == BACKTEST_WIZARD_KIND:
        return wizard_control_keyboard()

    if kind == BACKTEST_SYMBOL_WIZARD_KIND:
        step = str(session.get("step") or "select_symbols")
        data = get_symbol_wizard_data(session)

        if step == "select_symbols":
            return backtest_symbol_select_keyboard(data)

        if step == "choose_side":
            return backtest_symbol_side_keyboard()

        if step == "threshold_mode":
            return backtest_symbol_threshold_mode_keyboard()

        return wizard_control_keyboard()

    return None


def get_backtest_mode_text() -> str:
    return (
        "🧪 Бэктест\n\n"
        "Что проверяем?\n\n"
        "1. Проверить систему — полный системный сценарий.\n"
        "2. Проверить символ(ы) — посимвольный research-бэктест."
    )


def get_symbol_select_text(data: Dict[str, Any]) -> str:
    symbols = get_system_backtest_symbols()

    if not symbols:
        return "\n".join([
            "❌ В системе не найдено символов.",
            "",
            "Ожидались parquet-файлы в data/m1_4 и data/h4_3.",
        ])

    return "\n".join([
        "🧪 Посимвольный бэктест",
        "",
        "Выбранные символы:",
        format_selected_symbols_human(data),
    ])


def get_symbol_side_text(symbol: str) -> str:
    return "\n".join([
        "Символ: {}".format(str(symbol or "").strip().upper()),
        "",
        "Какую сторону тестируем?"
    ])


def get_symbol_step_text(session: Dict[str, Any]) -> str:
    data = get_symbol_wizard_data(session)
    step = str(session.get("step") or "select_symbols")

    if step == "select_symbols":
        return get_symbol_select_text(data)

    if step == "choose_side":
        return get_symbol_side_text(str(data.get("pending_symbol") or ""))

    if step == "start_date":
        return "\n".join([
            "🧪 Посимвольный бэктест",
            "",
            "Выбрано: {}".format(format_selected_symbols_human(data)),
            "",
            "Введите дату начала в UTC.",
            "Формат: YYYY-MM-DD",
            "Пример: 2026-05-01",
        ])

    if step == "start_time":
        return "\n".join([
            "Введите время начала в UTC.",
            "Формат: HH:MM",
            "Пример: 12:00",
        ])

    if step == "end_date":
        return "\n".join([
            "Введите дату конца в UTC.",
            "Формат: YYYY-MM-DD",
            "Пример: 2026-05-09",
        ])

    if step == "end_time":
        return "\n".join([
            "Введите время конца в UTC.",
            "Формат: HH:MM",
            "Пример: 12:00",
        ])

    if step == "threshold_mode":
        return "\n".join([
            "Пороги для посимвольного бэктеста:",
            "",
            "Дефолт:",
            "Gate2 = {}".format(BACKTEST_SYMBOL_DEFAULT_GATE2),
            "Gate2 spread = {}".format(BACKTEST_SYMBOL_DEFAULT_GATE2_SIDE_MARGIN_MIN),
            "Gate4 = {}".format(BACKTEST_SYMBOL_DEFAULT_GATE4),
            "Gate5.1 = {}".format(BACKTEST_SYMBOL_DEFAULT_GATE5_1),
            "Gate5.3 = {}".format(BACKTEST_SYMBOL_DEFAULT_GATE5_3),
            "",
            "Gate2 spread — минимальная дельта между LONG/SHORT Gate2 proba.",
            "",
            "Использовать дефолт или ввести вручную?",
        ])

    if step == "threshold_values":
        return "\n".join([
            "Введите пороги и Gate2 spread одним сообщением через пробел:",
            "",
            "gate2 gate2_spread gate4 gate5_1 gate5_3",
            "",
            "Пример:",
            "0.70 0.30 0.57 0.10 0.54",
        ])

    if step == "capital":
        return "\n".join([
            "Стартовый капитал для бэктеста, USDT.",
            "",
            "Дефолт: {}".format(BACKTEST_SYMBOL_DEFAULT_CAPITAL),
            "",
            "Введи число или '-' для дефолта.",
        ])

    if step == "confirm":
        return get_symbol_confirm_text(data)

    return get_symbol_select_text(data)


def get_symbol_confirm_text(data: Dict[str, Any]) -> str:
    start = "{} {}".format(data.get("start_date", "?"), data.get("start_time", "?"))
    end = "{} {}".format(data.get("end_date", "?"), data.get("end_time", "?"))

    return "\n".join([
        "🧪 Посимвольный бэктест готов к запуску",
        "",
        "Символы:",
        format_selected_symbols_human(data),
        "",
        "Период:",
        "{} → {}".format(start, end),
        "",
        "Пороги:",
        "Gate2: {}".format(data.get("gate2", "?")),
        "Gate2 spread: {}".format(data.get("gate2_side_margin_min", "?")),
        "Gate4: {}".format(data.get("gate4", "?")),
        "Gate5.1: {}".format(data.get("gate5_1", "?")),
        "Gate5.3: {}".format(data.get("gate5_3", "?")),
        "",
        "Стартовый капитал: ${}".format(data.get("capital", BACKTEST_SYMBOL_DEFAULT_CAPITAL)),
        "",
        "Нажми ✅ Запустить или ❌ Отмена.",
    ])


def normalize_symbol_wizard_side(raw: str) -> Optional[str]:
    text = str(raw or "").strip().upper()

    if text in {"LONG", "L", "BUY"}:
        return "L"

    if text in {"SHORT", "S", "SELL"}:
        return "S"

    if text in {"BOTH", "B", "ALL", "ОБЕ", "ВСЕ"}:
        return ""

    return None


def parse_symbol_threshold_values(raw: str) -> Tuple[str, str, str, str, str]:
    parts = [x.strip().replace(",", ".") for x in str(raw or "").split() if x.strip()]

    if len(parts) != 5:
        raise RuntimeError("Нужно ввести ровно 5 чисел: gate2 gate2_spread gate4 gate5_1 gate5_3.")

    values: List[str] = []

    for idx, part in enumerate(parts):
        try:
            value = float(part)
        except Exception:
            raise RuntimeError("Значение должно быть числом: {}".format(part))

        if value < 0.0 or value > 1.0:
            if idx == 1:
                raise RuntimeError("Gate2 spread должен быть в диапазоне 0..1: {}".format(part))
            raise RuntimeError("Порог должен быть в диапазоне 0..1: {}".format(part))

        values.append(str(value))

    return values[0], values[1], values[2], values[3], values[4]


def parse_symbol_capital_value(raw: str) -> str:
    text = str(raw or "").strip().replace(",", ".")

    if text == "-":
        return BACKTEST_SYMBOL_DEFAULT_CAPITAL

    try:
        value = float(text)
    except Exception:
        raise RuntimeError("Стартовый капитал должен быть числом, например 100.")

    if value <= 0.0:
        raise RuntimeError("Стартовый капитал должен быть больше 0.")

    return "{:.2f}".format(value).rstrip("0").rstrip(".")


def start_symbol_backtest_wizard(chat_id: int) -> str:
    session = {
        "kind": BACKTEST_SYMBOL_WIZARD_KIND,
        "step": "select_symbols",
        "data": {
            "selected": {},
            "selected_order": [],
        },
        "created_at_utc": str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
    }

    set_chat_session(chat_id, session)
    return get_symbol_step_text(session)


def build_symbol_backtest_service_args(data: Dict[str, Any]) -> List[str]:
    host = parse_host(BACKTEST_SYMBOL_FIXED_HOST, DEFAULT_RUN_HOST)
    start = "{} {}".format(data["start_date"], data["start_time"])
    end = "{} {}".format(data["end_date"], data["end_time"])
    symbols_arg = format_selected_symbols_from_data(data)

    if not symbols_arg:
        raise RuntimeError("Не выбраны символы для посимвольного бэктеста.")

    service_args = [
        "backtest-local" if host == LOCAL_HOST else "backtest",
    ]

    if host != LOCAL_HOST:
        service_args.append(host)

    service_args += [
        "--symbols",
        symbols_arg,
        "--start",
        start,
        "--end",
        end,
        "--gate2",
        str(data["gate2"]),
        "--gate2-side-margin-min",
        str(data.get("gate2_side_margin_min", BACKTEST_SYMBOL_DEFAULT_GATE2_SIDE_MARGIN_MIN)),
        "--gate4",
        str(data["gate4"]),
        "--gate5-1",
        str(data["gate5_1"]),
        "--gate5-3",
        str(data["gate5_3"]),
        "--capital",
        str(data.get("capital", BACKTEST_SYMBOL_DEFAULT_CAPITAL)),
        "--chulan",
        BACKTEST_SYMBOL_FIXED_CHULAN,
        "--max-full-sl-capital-risk",
        BACKTEST_SYMBOL_FIXED_MAX_FULL_SL_CAPITAL_RISK,
        "--slots",
        BACKTEST_SYMBOL_FIXED_SLOTS,
        "--entry-delay-seconds",
        BACKTEST_SYMBOL_FIXED_ENTRY_DELAY_SECONDS,
        "--skip-m1-sync",
    ]

    return service_args


def run_symbol_backtest_from_wizard(chat_id: int, data: Dict[str, Any]) -> str:
    service_args = build_symbol_backtest_service_args(data)
    code, out = run_service_status(service_args)
    answer = format_backtest_output(code, out, " ".join(service_args))

    if int(code) == 0:
        excel_status = send_backtest_excel_document_if_exists(
            chat_id=chat_id,
            out=out,
        )

        answer = answer + "\n\n📎 Excel: " + excel_status

    return answer


def handle_backtest_mode_wizard(chat_id: int, text: str) -> Optional[str]:
    session = get_chat_session(chat_id)

    if not session or session.get("kind") != BACKTEST_MODE_WIZARD_KIND:
        return None

    raw = str(text or "").strip()

    if raw in {"/cancel", "cancel", "отмена", "❌ Отмена"}:
        clear_chat_session(chat_id)
        return "❌ Backtest отменён."

    if raw == BACKTEST_MODE_SYSTEM_TEXT:
        return start_system_backtest_wizard(chat_id)

    if raw == BACKTEST_MODE_SYMBOLS_TEXT:
        return start_symbol_backtest_wizard(chat_id)

    return get_backtest_mode_text()


def handle_symbol_backtest_wizard(chat_id: int, text: str) -> Optional[str]:
    session = get_chat_session(chat_id)

    if not session or session.get("kind") != BACKTEST_SYMBOL_WIZARD_KIND:
        return None

    raw = str(text or "").strip()

    if raw in {"/cancel", "cancel", "отмена", "❌ Отмена"}:
        clear_chat_session(chat_id)
        return "❌ Посимвольный backtest отменён."

    data = get_symbol_wizard_data(session)
    step = str(session.get("step") or "select_symbols")

    if step == "select_symbols":
        if raw == BACKTEST_SYMBOL_RESET_TEXT:
            data["selected"] = {}
            data["selected_order"] = []
            session["data"] = data
            set_chat_session(chat_id, session)
            return get_symbol_step_text(session)

        if raw == BACKTEST_SYMBOL_DONE_TEXT or raw.lower() in {"done", "готово", "/go"}:
            selected_arg = format_selected_symbols_from_data(data)

            if not selected_arg:
                return "❌ Нужно выбрать хотя бы один символ.\n\n" + get_symbol_step_text(session)

            session["step"] = "start_date"
            session["data"] = data
            set_chat_session(chat_id, session)
            return get_symbol_step_text(session)

        symbol = raw.upper().strip()
        symbols = set(get_system_backtest_symbols())

        if symbol not in symbols:
            return "❌ Неизвестный символ: {}\n\n{}".format(raw, get_symbol_step_text(session))

        data["pending_symbol"] = symbol
        session["step"] = "choose_side"
        session["data"] = data
        set_chat_session(chat_id, session)
        return get_symbol_step_text(session)

    if step == "choose_side":
        if raw == BACKTEST_SYMBOL_BACK_TO_LIST_TEXT:
            data.pop("pending_symbol", None)
            session["step"] = "select_symbols"
            session["data"] = data
            set_chat_session(chat_id, session)
            return get_symbol_step_text(session)

        side_code = normalize_symbol_wizard_side(raw)

        if side_code is None:
            return "❌ Выбери LONG, SHORT или BOTH.\n\n" + get_symbol_step_text(session)

        symbol = str(data.get("pending_symbol") or "").strip().upper()

        if not symbol:
            session["step"] = "select_symbols"
            session["data"] = data
            set_chat_session(chat_id, session)
            return get_symbol_step_text(session)

        selected = data.get("selected")
        if not isinstance(selected, dict):
            selected = {}

        selected_order = data.get("selected_order")
        if not isinstance(selected_order, list):
            selected_order = []

        selected[symbol] = side_code

        if symbol not in selected_order:
            selected_order.append(symbol)

        data["selected"] = selected
        data["selected_order"] = selected_order
        data.pop("pending_symbol", None)

        session["step"] = "select_symbols"
        session["data"] = data
        set_chat_session(chat_id, session)

        return "✅ Добавлено: {}\n\n{}".format(
            format_selected_symbol_item(symbol, side_code),
            get_symbol_step_text(session),
        )

    if step in {"start_date", "start_time", "end_date", "end_time"}:
        error = validate_wizard_value(step, raw)

        if error:
            return "❌ {}\n\n{}".format(error, get_symbol_step_text(session))

        data[step] = raw

        next_step = {
            "start_date": "start_time",
            "start_time": "end_date",
            "end_date": "end_time",
            "end_time": "threshold_mode",
        }[step]

        session["step"] = next_step
        session["data"] = data
        set_chat_session(chat_id, session)
        return get_symbol_step_text(session)

    if step == "threshold_mode":
        if raw == BACKTEST_SYMBOL_DEFAULT_THRESHOLDS_TEXT or raw in {"-", "0", "default", "дефолт"}:
            data["gate2"] = BACKTEST_SYMBOL_DEFAULT_GATE2
            data["gate2_side_margin_min"] = BACKTEST_SYMBOL_DEFAULT_GATE2_SIDE_MARGIN_MIN
            data["gate4"] = BACKTEST_SYMBOL_DEFAULT_GATE4
            data["gate5_1"] = BACKTEST_SYMBOL_DEFAULT_GATE5_1
            data["gate5_3"] = BACKTEST_SYMBOL_DEFAULT_GATE5_3

            session["step"] = "capital"
            session["data"] = data
            set_chat_session(chat_id, session)
            return get_symbol_step_text(session)

        if raw == BACKTEST_SYMBOL_MANUAL_THRESHOLDS_TEXT or raw in {"1", "manual", "вручную"}:
            session["step"] = "threshold_values"
            session["data"] = data
            set_chat_session(chat_id, session)
            return get_symbol_step_text(session)

        return "❌ Выбери дефолтные пороги или ручной ввод.\n\n" + get_symbol_step_text(session)

    if step == "threshold_values":
        try:
            gate2, gate2_side_margin_min, gate4, gate5_1, gate5_3 = parse_symbol_threshold_values(raw)
        except Exception as exc:
            return "❌ {}\n\n{}".format(exc, get_symbol_step_text(session))

        data["gate2"] = gate2
        data["gate2_side_margin_min"] = gate2_side_margin_min
        data["gate4"] = gate4
        data["gate5_1"] = gate5_1
        data["gate5_3"] = gate5_3

        session["step"] = "capital"
        session["data"] = data
        set_chat_session(chat_id, session)
        return get_symbol_step_text(session)

    if step == "capital":
        try:
            data["capital"] = parse_symbol_capital_value(raw)
        except Exception as exc:
            return "❌ {}\n\n{}".format(exc, get_symbol_step_text(session))

        session["step"] = "confirm"
        session["data"] = data
        set_chat_session(chat_id, session)
        return get_symbol_confirm_text(data)

    if step == "confirm":
        if raw.lower() in {
            "/go",
            "go",
            "✅ запустить",
            "/run_backtest",
            "запустить",
            "run",
        }:
            clear_chat_session(chat_id)
            return run_symbol_backtest_from_wizard(chat_id=chat_id, data=data)

        return get_symbol_confirm_text(data)

    return get_symbol_step_text(session)


def get_chat_sessions(state: Dict[str, Any]) -> Dict[str, Any]:
    sessions = state.get("chat_sessions")
    if not isinstance(sessions, dict):
        sessions = {}
        state["chat_sessions"] = sessions
    return sessions


def get_chat_session(chat_id: int) -> Optional[Dict[str, Any]]:
    state = read_state()
    sessions = get_chat_sessions(state)
    raw = sessions.get(str(int(chat_id)))
    if isinstance(raw, dict):
        return raw
    return None


def set_chat_session(chat_id: int, session: Dict[str, Any]) -> None:
    state = read_state()
    sessions = get_chat_sessions(state)
    sessions[str(int(chat_id))] = session
    write_state(state)


def clear_chat_session(chat_id: int) -> None:
    state = read_state()
    sessions = get_chat_sessions(state)
    if str(int(chat_id)) in sessions:
        del sessions[str(int(chat_id))]
    write_state(state)


def wizard_control_keyboard() -> Dict[str, Any]:
    return {
        "keyboard": [
            ["✅ Запустить", "❌ Отмена"],
            ["📊 Статус", "⚙️ Команды"],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }


def send_message_with_keyboard(chat_id: int, text: str, keyboard_obj: Dict[str, Any]) -> None:
    chunks = split_message(text, MAX_MESSAGE_LEN)
    keyboard = json.dumps(keyboard_obj, ensure_ascii=False)

    for i, chunk in enumerate(chunks, start=1):
        payload = {
            "chat_id": str(chat_id),
            "text": chunk,
            "disable_web_page_preview": "true",
        }

        if i == len(chunks):
            payload["reply_markup"] = keyboard

        telegram_api(
            "sendMessage",
            payload,
            timeout_seconds=TG_SEND_MESSAGE_SOCKET_TIMEOUT_SECONDS,
            retries=TG_API_RETRIES,
        )


def is_backtest_wizard_active(chat_id: int) -> bool:
    session = get_chat_session(chat_id)
    if not session:
        return False

    return session.get("kind") in {
        BACKTEST_WIZARD_KIND,
        BACKTEST_MODE_WIZARD_KIND,
        BACKTEST_SYMBOL_WIZARD_KIND,
    }


def get_backtest_step_text(step_idx: int, data: Dict[str, str]) -> str:
    total = len(BACKTEST_WIZARD_STEPS)

    if step_idx >= total:
        return get_backtest_confirm_text(data)

    step = BACKTEST_WIZARD_STEPS[step_idx]
    default = str(step.get("default", ""))

    lines = [
        "🧪 Backtest master",
        "",
        "Шаг {}/{}".format(step_idx + 1, total),
        "{}:".format(step["title"]),
        "",
        "Пример: {}".format(step["example"]),
    ]

    if default:
        lines.append("По умолчанию: {}".format(default))
        lines.append("")
        lines.append("Отправь значение или '-' чтобы взять дефолт.")
    else:
        lines.append("")
        lines.append("Отправь значение.")

    lines.append("")
    lines.append("Для отмены: ❌ Отмена")

    return "\n".join(lines)


def get_backtest_confirm_text(data: Dict[str, str]) -> str:
    start = "{} {}".format(data.get("start_date", "?"), data.get("start_time", "?"))
    end = "{} {}".format(data.get("end_date", "?"), data.get("end_time", "?"))

    return (
        "🧪 Backtest готов к запуску\n\n"
        "Период:\n"
        "{} → {}\n\n"
        "Пороги:\n"
        "Gate2: {}\n"
        "Gate4: {}\n"
        "Gate5.1: {}\n"
        "Gate5.3: {}\n\n"
        "Параметры:\n"
        "chulan: {}\n"
        "side-aware whitelist: {}\n"
        "conditional whitelist: {}\n"
        "max full SL risk: {}%\n"
        "slots: {}\n"
        "write_blacklist: {}\n"
        "reset_blacklist: {}\n"
        "sync_m1: {}\n"
        "host: {}\n\n"
        "Для запуска отправь /go. Для отмены отправь /cancel."
    ).format(
        start,
        end,
        data.get("gate2", "?"),
        data.get("gate4", "?"),
        data.get("gate5_1", "?"),
        data.get("gate5_3", "?"),
        data.get("chulan", "?"),
        data.get("side_aware_whitelist", "?"),
        data.get("conditional_side_aware_whitelist", "?"),
        data.get("max_full_sl_risk_pct", "?"),
        data.get("slots", "?"),
        data.get("write_blacklist", "?"),
        data.get("reset_blacklist", "?"),
        data.get("sync_m1", "?"),
        data.get("host", "?"),
    )

def validate_wizard_value(key: str, value: str) -> Optional[str]:
    raw = str(value or "").strip()

    if key in {"start_date", "end_date"}:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
            return "Дата должна быть в формате YYYY-MM-DD, например 2026-05-01."

    if key in {"start_time", "end_time"}:
        if not re.match(r"^\d{2}:\d{2}$", raw):
            return "Время должно быть в формате HH:MM, например 12:00."

    if key in {"gate2", "gate4", "gate5_1", "gate5_3"}:
        try:
            x = float(raw)
        except Exception:
            return "Порог должен быть числом, например 0.70."

        if x < 0.0 or x > 1.0:
            return "Порог должен быть в диапазоне от 0 до 1."

    if key in {
        "change_defaults",
        "chulan",
        "side_aware_whitelist",
        "conditional_side_aware_whitelist",
        "write_blacklist",
        "reset_blacklist",
        "sync_m1",
    }:
        if raw not in {"0", "1"}:
            return "Значение должно быть 0 или 1."

    if key == "max_full_sl_risk_pct":
        try:
            x = float(raw.replace(",", "."))
        except Exception:
            return "Риск полного стопа должен быть числом в процентах, например 6."

        if x < 0.0 or x > 100.0:
            return "Риск полного стопа должен быть в диапазоне от 0 до 100 процентов."

    if key == "slots":
        try:
            slots = int(raw)
        except Exception:
            return "Количество слотов должно быть целым числом."

        if slots != 1:
            return "Сейчас backtest поддерживает только один слот. Введи 1."

    if key == "host":
        if raw not in {HOST_MAC, HOST_WIN}:
            return "Хост должен быть mac или win."

    return None

def fill_backtest_default_values(data: Dict[str, str]) -> Dict[str, str]:
    out = dict(data)

    required_default_keys = [
        "gate2",
        "gate4",
        "gate5_1",
        "gate5_3",
        "chulan",
        "side_aware_whitelist",
        "conditional_side_aware_whitelist",
        "max_full_sl_risk_pct",
        "slots",
        "write_blacklist",
        "reset_blacklist",
        "sync_m1",
        "host",
    ]

    steps_by_key = {
        str(step["key"]): step
        for step in BACKTEST_WIZARD_STEPS
    }

    for key in required_default_keys:
        if key in out and str(out.get(key) or "").strip():
            continue

        step = steps_by_key.get(key)

        if not isinstance(step, dict):
            raise RuntimeError(
                "Не найден шаг backtest wizard для ключа: {}".format(key)
            )

        default = str(step.get("default", "")).strip()

        if not default:
            raise RuntimeError(
                "Для backtest wizard отсутствует дефолт ключа: {}".format(key)
            )

        out[key] = default

    return out

def start_system_backtest_wizard(chat_id: int) -> str:
    session = {
        "kind": BACKTEST_WIZARD_KIND,
        "step_idx": 0,
        "data": {},
        "created_at_utc": str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
    }

    set_chat_session(chat_id, session)
    return get_backtest_step_text(0, {})


def start_backtest_wizard(chat_id: int) -> str:
    session = {
        "kind": BACKTEST_MODE_WIZARD_KIND,
        "step": "mode",
        "data": {},
        "created_at_utc": str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
    }

    set_chat_session(chat_id, session)
    return get_backtest_mode_text()


def build_backtest_parts_from_wizard(data: Dict[str, str]) -> List[str]:
    return [
        "/backtest",
        data["start_date"],
        data["start_time"],
        data["end_date"],
        data["end_time"],
        data["gate2"],
        data["gate4"],
        data["gate5_1"],
        data["gate5_3"],
        data["chulan"],
        data["side_aware_whitelist"],
        data["conditional_side_aware_whitelist"],
        data["max_full_sl_risk_pct"],
        data["slots"],
        data["write_blacklist"],
        data["reset_blacklist"],
        data["sync_m1"],
        data["host"],
    ]

def handle_backtest_wizard(chat_id: int, text: str) -> Optional[str]:
    mode_answer = handle_backtest_mode_wizard(chat_id, text)
    if mode_answer is not None:
        return mode_answer

    symbol_answer = handle_symbol_backtest_wizard(chat_id, text)
    if symbol_answer is not None:
        return symbol_answer

    session = get_chat_session(chat_id)

    if not session or session.get("kind") != BACKTEST_WIZARD_KIND:
        return None

    raw = str(text or "").strip()

    if raw in {"/cancel", "cancel", "отмена", "❌ Отмена"}:
        clear_chat_session(chat_id)
        return "❌ Backtest отменён."

    data = session.get("data")
    if not isinstance(data, dict):
        data = {}

    step_idx = int(session.get("step_idx") or 0)

    if step_idx >= len(BACKTEST_WIZARD_STEPS):
        if raw.lower() in {
            "/go",
            "go",
            "✅ запустить",
            "/run_backtest",
            "запустить",
            "run",
        }:
            data = fill_backtest_default_values(data)
            parts = build_backtest_parts_from_wizard(data)
            clear_chat_session(chat_id)

            code, out = tg_backtest(parts)
            return format_backtest_output(code, out, " ".join(parts))

        return get_backtest_confirm_text(fill_backtest_default_values(data))

    step = BACKTEST_WIZARD_STEPS[step_idx]
    key = str(step["key"])
    default = str(step.get("default", ""))
    value = raw

    if value == "-" and default:
        value = default

    if value == "-":
        return "❌ Для поля '{}' нет дефолта. Введи значение.".format(step["title"])

    error = validate_wizard_value(key, value)
    if error:
        return "❌ {}\n\n{}".format(error, get_backtest_step_text(step_idx, data))

    data[key] = value
    step_idx += 1

    # Пользователь выбрал дефолтные параметры.
    # Но всё равно спрашиваем conditional whitelist и процент risk cap.
    # После max_full_sl_risk_pct остальные параметры заполняются дефолтами.
    if key == "max_full_sl_risk_pct" and data.get("change_defaults") == "0":
        data = fill_backtest_default_values(data)
        step_idx = len(BACKTEST_WIZARD_STEPS)

    session["step_idx"] = step_idx
    session["data"] = data
    set_chat_session(chat_id, session)

    if step_idx >= len(BACKTEST_WIZARD_STEPS):
        return get_backtest_confirm_text(data)

    return get_backtest_step_text(step_idx, data)

def handle_command(chat_id: int, text: str) -> str:
    text = normalize_menu_text(text)

    if text.strip() in {"/cancel", "cancel", "отмена"}:
        if is_backtest_wizard_active(chat_id) or is_add_symbol_wizard_active(chat_id):
            clear_chat_session(chat_id)
            return "❌ Текущий мастер отменён."
        return "Нет активного мастера."

    wizard_answer = handle_add_symbol_wizard(chat_id, text)
    if wizard_answer is not None:
        return wizard_answer

    wizard_answer = handle_backtest_wizard(chat_id, text)
    if wizard_answer is not None:
        return wizard_answer

    parts = normalize_command(text)

    if not parts:
        return ""

    cmd = parts[0]

    if cmd == "/auth":
        if len(parts) < 2:
            return "❌ Нужно: /auth <TG_SECRET>"

        ok = authorize_chat(chat_id, parts[1])
        if ok:
            return "✅ Доступ разрешён для chat_id={}".format(chat_id)

        return "❌ Неверный TG_SECRET"

    if cmd in {"/start", "/help"}:
        return help_text()

    if not is_authorized(chat_id):
        return "🔒 Нет доступа. Сначала отправь: /auth <TG_SECRET>"

    if cmd in {"/add_symbol_wizard", "/add_symbol"}:
        return start_add_symbol_wizard(chat_id)

    if cmd == "/backtest_wizard":
        return start_backtest_wizard(chat_id)

    handlers = {
        "/status": tg_status,
        "/run": tg_run,
        "/start_trade": tg_run,
        "/stop": tg_stop,
        "/history": tg_history,
        "/position": tg_position,
        "/pos": tg_position,
        "/backtest": tg_backtest,

        "status": tg_status,
        "run": tg_run,
        "start": tg_run,
        "stop": tg_stop,
        "history": tg_history,
        "position": tg_position,
        "pos": tg_position,
        "backtest": tg_backtest,
    }

    if cmd not in handlers:
        return "❌ Неизвестная команда.\n\n" + help_text()

    try:
        code, out = handlers[cmd](parts)
        answer = format_command_result(cmd, parts, code, out)

        if cmd in {"/backtest", "backtest"} and int(code) == 0:
            excel_status = send_backtest_excel_document_if_exists(
                chat_id=chat_id,
                out=out,
            )

            answer = answer + "\n\n📎 Excel: " + excel_status

        return answer

    except Exception as e:
        return "❌ ERROR\ncommand: {}\nerror: {}".format(" ".join(parts), e)
def process_update(update: Dict[str, Any]) -> None:
    update_id = update.get("update_id")
    message = update.get("message")

    if not isinstance(message, dict):
        log_event("TG_UPDATE_SKIP_NO_MESSAGE update_id={}".format(update_id))
        return

    chat = message.get("chat")
    if not isinstance(chat, dict):
        log_event("TG_UPDATE_SKIP_NO_CHAT update_id={}".format(update_id))
        return

    chat_id_raw = chat.get("id")
    if chat_id_raw is None:
        log_event("TG_UPDATE_SKIP_NO_CHAT_ID update_id={}".format(update_id))
        return

    chat_id = int(chat_id_raw)
    text = str(message.get("text") or "").strip()

    if not text:
        log_event("TG_UPDATE_SKIP_EMPTY_TEXT update_id={} chat_id={}".format(update_id, chat_id))
        return

    log_event(
        "TG_UPDATE_COMMAND_START update_id={} chat_id={} text={!r}".format(
            update_id,
            chat_id,
            text,
        )
    )

    answer = handle_command(chat_id=chat_id, text=text)

    log_event(
        "TG_UPDATE_COMMAND_DONE update_id={} chat_id={} answer_len={}".format(
            update_id,
            chat_id,
            len(answer or ""),
        )
    )

    if answer:
        keyboard_obj = get_active_session_keyboard(chat_id)

        if keyboard_obj is not None:
            send_message_with_keyboard(chat_id, answer, keyboard_obj)
        else:
            send_message(chat_id, answer)


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is empty. Put TELEGRAM_TOKEN=... into .env")

    print("=" * 120, flush=True)
    print("TG_CONTROL_BOT_STARTED", flush=True)
    print("ROOT:", ROOT, flush=True)
    print("LOCAL_HOST:", LOCAL_HOST, flush=True)
    print("DEFAULT_RUN_HOST:", DEFAULT_RUN_HOST, flush=True)
    print("STATE_PATH:", STATE_PATH, flush=True)
    print("STATIC_CHAT_ID:", STATIC_CHAT_ID if STATIC_CHAT_ID else "EMPTY", flush=True)
    print("TG_SECRET_SET:", bool(TG_SECRET), flush=True)
    print("POLL_TIMEOUT_SECONDS:", POLL_TIMEOUT_SECONDS, flush=True)
    print("TG_GET_UPDATES_SOCKET_TIMEOUT_SECONDS:", TG_GET_UPDATES_SOCKET_TIMEOUT_SECONDS, flush=True)
    print("TG_SEND_MESSAGE_SOCKET_TIMEOUT_SECONDS:", TG_SEND_MESSAGE_SOCKET_TIMEOUT_SECONDS, flush=True)
    print("TG_COMMAND_TIMEOUT_SECONDS:", TG_COMMAND_TIMEOUT_SECONDS, flush=True)
    print("TG_BACKTEST_TIMEOUT_SECONDS:", TG_BACKTEST_TIMEOUT_SECONDS, flush=True)
    print("TG_API_RETRIES:", TG_API_RETRIES, flush=True)

    state = read_state()
    offset = int(state.get("offset") or 0)

    while True:
        try:
            updates = get_updates(offset)

            if updates:
                log_event(
                    "TG_UPDATES_RECEIVED count={} offset_before={}".format(
                        len(updates),
                        offset,
                    )
                )

            for upd in updates:
                update_id = int(upd.get("update_id") or 0)

                try:
                    process_update(upd)

                except Exception as exc:
                    log_event(
                        "TG_UPDATE_PROCESS_ERROR update_id={} error={!r}".format(
                            update_id,
                            exc,
                        )
                    )

                finally:
                    if update_id > 0:
                        offset = max(offset, update_id + 1)
                        state = read_state()
                        state["offset"] = offset
                        write_state(state)

                        log_event(
                            "TG_OFFSET_SAVED update_id={} offset={}".format(
                                update_id,
                                offset,
                            )
                        )

        except KeyboardInterrupt:
            print("TG_CONTROL_BOT_STOPPED_BY_KEYBOARD", flush=True)
            return

        except Exception as e:
            print("TG_CONTROL_BOT_ERROR:", repr(e), flush=True)
            time.sleep(5.0)

        time.sleep(POLL_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
