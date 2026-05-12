from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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



def main_menu_keyboard() -> Dict[str, Any]:
    return {
        "keyboard": [
            ["📊 Статус", "📜 История"],
            ["▶️ Запуск", "⏹ Стоп"],
            ["🧪 Бэктест", "⚙️ Команды"],
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
        "▶️ Запуск": "/run",
        "⏹ Стоп": "/stop",
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

    log_event(
        "TG_COMMAND_SUBPROCESS_START timeout={} cmd={!r}".format(
            TG_COMMAND_TIMEOUT_SECONDS,
            cmd,
        )
    )

    try:
        env = build_base_env()

        if args and str(args[0]).strip().lower() in {"history", "history-local"}:
            env["IMB_HISTORY_OUTPUT_FORMAT"] = "json"

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TG_COMMAND_TIMEOUT_SECONDS,
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
            TG_COMMAND_TIMEOUT_SECONDS,
            cmd,
            out,
        )

        log_event(
            "TG_COMMAND_SUBPROCESS_TIMEOUT timeout={} cmd={!r}".format(
                TG_COMMAND_TIMEOUT_SECONDS,
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

def tg_backtest(args: List[str]) -> Tuple[int, str]:
    if len(args) < 5:
        return (
            2,
            "❌ Неверный формат /backtest\n\n"
            "Формат:\n"
            "/backtest YYYY-MM-DD HH:MM YYYY-MM-DD HH:MM [gate2] [gate4] [gate5_1] [gate5_3] [chulan] [write_blacklist] [reset_blacklist] [host]\n\n"
            "Пример prod-порогов:\n"
            "/backtest 2026-05-01 12:00 2026-05-09 12:00 0.63 0.58 0.1 0.55 1 0 0 win\n\n"
            "Минимальный пример:\n"
            "/backtest 2026-05-01 12:00 2026-05-09 12:00"
        )

    start = str(args[1]).strip() + " " + str(args[2]).strip()
    end = str(args[3]).strip() + " " + str(args[4]).strip()

    gate2 = str(config.GATE2_THR)
    gate4 = str(config.GATE4_THR)
    gate5_1 = str(config.GATE5_1_THR)
    gate5_3 = str(config.GATE5_3_THR)
    chulan = "1"
    write_blacklist = "0"
    reset_blacklist = "0"
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
        write_blacklist = str(args[10]).strip()
    if len(args) >= 12:
        reset_blacklist = str(args[11]).strip()
    if len(args) >= 13:
        host = parse_host(args[12], DEFAULT_RUN_HOST)

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
        "--write-dynamic-blacklist",
        write_blacklist,
        "--reset-backtest-blacklist",
        reset_blacklist,
    ]

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


def format_status_output(code: int, out: str, command_text: str) -> str:
    data = parse_key_value_output(out)

    if code != 0:
        body = str(out or "").strip()
        if not body:
            body = "EMPTY_OUTPUT"
        return "❌ Статус недоступен\n\nКоманда: {}\nКод: {}\n\n{}".format(
            command_text,
            code,
            body[:2500],
        )

    running = str(data.get("service_running", "")).strip().lower() == "true"
    running_icon = "🟢" if running else "🔴"
    running_text = "Работает" if running else "Остановлен"

    host = data.get("host", "?")
    dry_run = data.get("dry_run_env", "?")

    capital = data.get("capital_usdt", "?")
    pnl = data.get("current_position_pnl_usdt", "?")
    open_positions = data.get("open_positions_count", "?")

    pair_model = data.get("pair_model_name", "")
    grid = data.get("grid_name", "")

    gate2 = data.get("gate2_thr", "?")
    gate4 = data.get("gate4_thr", "?")
    gate5_1 = data.get("gate5_1_thr", "?")
    gate5_3 = data.get("gate5_3_thr", "?")

    dyn_filter = format_bool_on_off(data.get("dynamic_symbol_filter_enabled", "?"))
    chulan_enabled = format_bool_on_off(data.get("chulan_enabled", "?"))
    chulan_base_capital = data.get("chulan_base_capital_usdt", "?")
    trade_capital = data.get("trade_capital_usdt", "?")

    next_h4 = data.get("next_h4_close_utc", "?")
    time_left = data.get("time_to_next_h4_close", "?")

    lines = [
        "📊 Статус автотрейда",
        "",
        "{} {}".format(running_icon, running_text),
        "Host: {} | DRY_RUN: {}".format(host, dry_run),
        "",
        "💰 Баланс: {} USDT".format(capital),
        "🏦 Рабочий капитал: {} USDT".format(trade_capital),
        "🧺 Чулан: {} | база: {} USDT".format(chulan_enabled, chulan_base_capital),
        "📍 Открытых позиций: {}".format(open_positions),
        "📈 PnL позиции: {} USDT".format(pnl),
        "",
        "🧠 Модель:",
        pair_model if pair_model else "?",
        "Сетка: {}".format(grid if grid else "?"),
        "",
        "🎯 Пороги:",
        "G2: {} | G4: {}".format(gate2, gate4),
        "G5.1: {} | G5.3: {}".format(gate5_1, gate5_3),
        "",
        "🕓 Следующая H4:",
        str(next_h4),
        "Осталось: {}".format(time_left),
        "",
        "🛡 Dynamic filter: {}".format(dyn_filter),
    ]

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

        for col, start, end in spans:
            value = line[start:end].strip() if start < len(line) else ""
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
        "TP_SL_PLACED": "OPEN",
        "POSITION_OPEN": "OPEN",
        "POSITION_CLOSED_MANUAL": "MANUAL",
        "POSITION_CLOSED_EXTERNAL": "EXT",
        "TP_SL_FAILED": "NO_TPSL",
        "DRY_RUN_ENTRY_PLANNED": "DRY",
    }

    return status_map.get(raw, raw[:8])


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

        if decision == "GO":
            icon = "🟢"
        else:
            icon = "⚪️"

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
        "TP_SL_PLACED": "TP/SL_OK",
        "TP_SL_FAILED": "TP/SL_FAIL",
        "POSITION_OPEN": "OPEN",
        "POSITION_CLOSED_MANUAL": "CLOSED_MANUAL",
        "POSITION_CLOSED_TP": "CLOSED_TP",
        "POSITION_CLOSED_SL": "CLOSED_SL",
        "POSITION_CLOSED_TTL": "CLOSED_TTL",
        "POSITION_CLOSED_EXTERNAL": "CLOSED_EXT",
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

        if decision == "GO":
            icon = "🟢"
        else:
            icon = "⚪️"

        entry_signal = fmt_history_price(row.get("entry_signal"))
        entry_plan = fmt_history_price(row.get("entry_plan"))
        entry_actual = fmt_history_price(row.get("entry_actual"))
        slip_pct = fmt_history_percent(row.get("slip_pct"))

        tp = fmt_history_price(row.get("tp"))
        sl = fmt_history_price(row.get("sl"))

        pos_status = short_history_status(row.get("pos_status"))
        reason = short_history_reason(row.get("reason"))

        lines = [
            "{} {} | {} | {} {}".format(
                icon,
                close_value,
                symbol,
                decision,
                side,
            ),
            "entry: signal={} plan={}".format(entry_signal, entry_plan),
        ]

        if entry_actual != "-":
            lines[-1] += " actual={}".format(entry_actual)

        if slip_pct != "-":
            lines[-1] += " slip={}".format(slip_pct)

        lines.append("tp/sl: {} / {}".format(tp, sl))

        tail: List[str] = []

        if pos_status:
            tail.append("pos={}".format(pos_status))

        if reason:
            tail.append("why={}".format(reason))

        if tail:
            lines.append(" | ".join(tail))

        blocks.append("\n".join(lines))

    if len(rows) > int(max_rows):
        blocks.append("... ещё строк: {}".format(len(rows) - int(max_rows)))

    return "\n\n".join(blocks)

def format_history_output(code: int, out: str, command_text: str) -> str:
    if code != 0:
        body = compact_raw_output(out, max_lines=50)

        if not body:
            body = "EMPTY_OUTPUT"

        return (
            "❌ История недоступна\n\n"
            "Команда: {}\n"
            "Код: {}\n\n"
            "{}"
        ).format(
            command_text,
            code,
            body,
        )

    rows = parse_history_json_output(out)

    if not rows:
        body = compact_raw_output(out, max_lines=50)
        if not body:
            body = "История пустая."

        return (
            "📜 История\n\n"
            "Команда: {}\n"
            "Код: {}\n\n"
            "{}"
        ).format(
            command_text,
            code,
            body,
        )

    body = format_history_rows_as_cards(rows, max_rows=10)

    return (
        "📜 История\n\n"
        "Команда: {}\n"
        "Код: {}\n\n"
        "{}"
    ).format(
        command_text,
        code,
        body,
    )

def format_backtest_output(code: int, out: str, command_text: str) -> str:
    body = compact_raw_output(out, max_lines=70)

    if not body:
        body = "EMPTY_OUTPUT"

    icon = "🧪" if code == 0 else "❌"

    return (
        "{} Backtest\n\n"
        "Команда: {}\n"
        "Код: {}\n\n"
        "{}"
    ).format(
        icon,
        command_text,
        code,
        body,
    )


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
        "key": "gate2",
        "title": "Gate2 threshold",
        "example": "0.63",
        "default": str(config.GATE2_THR),
    },
    {
        "key": "gate4",
        "title": "Gate4 threshold",
        "example": "0.58",
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
        "example": "0.55",
        "default": str(config.GATE5_3_THR),
    },
    {
        "key": "chulan",
        "title": "Использовать чулан",
        "example": "1",
        "default": str(int(getattr(config, "CHULAN_ENABLED", 0) or 0)),
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
        "key": "host",
        "title": "Хост запуска",
        "example": "win",
        "default": DEFAULT_RUN_HOST,
    },
]


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
    return session.get("kind") == BACKTEST_WIZARD_KIND


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
        "write_blacklist: {}\n"
        "reset_blacklist: {}\n"
        "host: {}\n\n"
        "Нажми ✅ Запустить или отправь /cancel."
    ).format(
        start,
        end,
        data.get("gate2", "?"),
        data.get("gate4", "?"),
        data.get("gate5_1", "?"),
        data.get("gate5_3", "?"),
        data.get("chulan", "?"),
        data.get("write_blacklist", "?"),
        data.get("reset_blacklist", "?"),
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
            return "Порог должен быть числом, например 0.63."

        if x < 0.0 or x > 1.0:
            return "Порог должен быть в диапазоне от 0 до 1."

    if key in {"chulan", "write_blacklist", "reset_blacklist"}:
        if raw not in {"0", "1"}:
            return "Значение должно быть 0 или 1."

    if key == "host":
        if raw not in {HOST_MAC, HOST_WIN}:
            return "Хост должен быть mac или win."

    return None


def start_backtest_wizard(chat_id: int) -> str:
    session = {
        "kind": BACKTEST_WIZARD_KIND,
        "step_idx": 0,
        "data": {},
        "created_at_utc": str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
    }

    set_chat_session(chat_id, session)
    return get_backtest_step_text(0, {})


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
        data["write_blacklist"],
        data["reset_blacklist"],
        data["host"],
    ]


def handle_backtest_wizard(chat_id: int, text: str) -> Optional[str]:
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
        if raw in {"✅ Запустить", "/run_backtest", "запустить", "run"}:
            parts = build_backtest_parts_from_wizard(data)
            clear_chat_session(chat_id)

            code, out = tg_backtest(parts)
            return format_backtest_output(code, out, " ".join(parts))

        return get_backtest_confirm_text(data)

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

    session["step_idx"] = step_idx
    session["data"] = data
    set_chat_session(chat_id, session)

    if step_idx >= len(BACKTEST_WIZARD_STEPS):
        return get_backtest_confirm_text(data)

    return get_backtest_step_text(step_idx, data)


def handle_command(chat_id: int, text: str) -> str:
    text = normalize_menu_text(text)

    if text.strip() in {"/cancel", "cancel", "отмена"}:
        if is_backtest_wizard_active(chat_id):
            clear_chat_session(chat_id)
            return "❌ Текущий мастер отменён."
        return "Нет активного мастера."

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

    if cmd == "/backtest_wizard":
        return start_backtest_wizard(chat_id)

    handlers = {
        "/status": tg_status,
        "/run": tg_run,
        "/start_trade": tg_run,
        "/stop": tg_stop,
        "/history": tg_history,
        "/backtest": tg_backtest,

        "status": tg_status,
        "run": tg_run,
        "start": tg_run,
        "stop": tg_stop,
        "history": tg_history,
        "backtest": tg_backtest,
    }

    if cmd not in handlers:
        return "❌ Неизвестная команда.\n\n" + help_text()

    try:
        code, out = handlers[cmd](parts)
        return format_command_result(cmd, parts, code, out)

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
        send_message(chat_id, answer)


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Telegram bot token is not configured. Put it into local .env")

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
