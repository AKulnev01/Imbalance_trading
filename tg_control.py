#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tg_control.py — легаси-бот (python-telegram-bot v13) с утилитными командами:
  /help — список команд
  /bulk [sfx] — запускает BULK_CMD (+ необязательный текст-суффикс)
  /eval [sfx] — запускает EVAL_CMD (+ суффикс)
  /bulk_eval [sfx] — подряд /bulk, затем /eval
  /checkpoint — диагностика закрытия 4h по логам автотрейда (без входов)
  /logs [bot|mom] [N] — хвост логов (по умолчанию mom, 200 строк)
  /net — краткая сетёвая сводка по логам автотрейда

Особенности:
- Отчёты ищутся в REPORT_DIR, берём последний *.xlsx (игнорируем '~$*.xlsx').
- /bulk парсит stdout на "lastbulk: /path" и шлёт именно этот файл, если найден.
- Автоподстановка --universe из ENV, если его нет в BULK_CMD.
- Логи: пишем в logs/tg_control_runtime.log и дублируем в stdout; при старте создаём
  symlink logs/tgctrl_latest.log -> текущий runtime.log, чтобы удобно было tail -f.
"""

import os
import re
import sys
import glob
import time
import shlex
import logging
import pathlib
import subprocess

from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# ========= ENV / пути =========
HOME = str(pathlib.Path.home())
LOGS_DIR            = os.getenv("LOGS_DIR", "logs")
REPORT_DIR          = os.getenv("REPORT_DIR", os.path.join(HOME, "Documents", "отчеты"))
TELEGRAM_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID             = os.getenv("CHAT_ID", "")

# команды по умолчанию (можно переопределить в ENV)
BULK_CMD            = os.getenv("BULK_CMD", "PYTHONPATH=. python scripts/listing_report.py --category ${BYBIT_CATEGORY:-linear} --interval 240")
EVAL_CMD            = os.getenv("EVAL_CMD", 'python -u evaluate_momentum.py --input "{xlsx}"')
EVAL_CMD_EXTRA      = os.getenv("EVAL_CMD_EXTRA", "")
SHELL_BIN           = os.getenv("SHELL", "/bin/bash")  # для совместимости

# где читать логи автотрейда и бота
MOMENTUM_LOG_GLOB   = os.getenv("MOMENTUM_LOG_GLOB", os.path.join(LOGS_DIR, "momentum_*.log"))
TGCTRL_LOG_GLOB     = os.getenv("TGCTRL_LOG_GLOB",   os.path.join(LOGS_DIR, "tgctrl_*.log"))

# ========= Логирование =========
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

_runtime_log = os.path.join(LOGS_DIR, "tg_control_runtime.log")
_latest_link = os.path.join(LOGS_DIR, "tgctrl_latest.log")
try:
    if os.path.islink(_latest_link) or os.path.exists(_latest_link):
        try: os.remove(_latest_link)
        except Exception: pass
    os.symlink(os.path.abspath(_runtime_log), _latest_link)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(_runtime_log, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("tg-control")

# ========= Хелперы =========
def _tail(path: str, n: int = 200) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return "".join(f.readlines()[-n:])
    except Exception as e:
        return f"❌ Не удалось прочитать лог {path}: {e}"

def _latest_file_by_glob(pattern: str) -> str:
    files = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if not files:
        return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def _latest_xlsx_in_report() -> str:
    candidates = []
    for p in glob.glob(os.path.join(REPORT_DIR, "*.xlsx")):
        base = os.path.basename(p)
        if base.startswith("~$"):
            continue
        candidates.append(p)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def _send_file(ctx: CallbackContext, chat_id, path: str, caption: str = ""):
    if not path or not os.path.exists(path):
        ctx.bot.send_message(chat_id=chat_id, text=f"❌ Файл не найден: {path or '—'}")
        return
    try:
        with open(path, "rb") as f:
            ctx.bot.send_document(chat_id=chat_id, document=f, filename=os.path.basename(path), caption=caption or os.path.basename(path))
    except Exception as e:
        ctx.bot.send_message(chat_id=chat_id, text=f"❌ Не удалось отправить файл: {e}")

def _run(cmd: str, workdir: str = ".", timeout_sec: int = 60*30) -> subprocess.CompletedProcess:
    log.info(f"[RUN] {cmd}")
    return subprocess.run(
        cmd,
        shell=True,
        cwd=workdir,
        executable=SHELL_BIN if os.path.isfile(SHELL_BIN) else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
        text=True,
        encoding="utf-8",
        errors="ignore",
        env=os.environ.copy(),
    )

def _parse_lastbulk(stdout: str) -> str:
    m = re.search(r"lastbulk:\s*(.+?\.xlsx)\s*$", stdout, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""
    cand = m.group(1).strip().strip("'\"")
    if os.path.basename(cand).startswith("~$"):
        return ""
    return cand if os.path.exists(cand) else ""

def _inject_universe_if_missing(cmd: str) -> str:
    """Если в BULK_CMD нет --universe и в ENV есть TRADE_UNIVERSE — добавим."""
    if re.search(r"(?:^|\s)--universe\b", cmd):
        return cmd
    uni = os.getenv("TRADE_UNIVERSE", "").strip()
    if not uni:
        return cmd
    return f"{cmd} --universe {shlex.quote(uni)}"

# ========= Команды =========
HELP_TEXT = (
    "🤖 Команды утилит:\n"
    "/help — список команд\n"
    "/bulk [sfx] — запустить BULK_CMD (+необяз. суффикс)\n"
    "/eval [sfx] — запустить EVAL_CMD (+необяз. суффикс)\n"
    "/bulk_eval [sfx] — подряд /bulk, потом /eval\n"
    "/checkpoint — REST/WS-диагностика закрытия 4h (по логам), без входа\n"
    "/logs [bot|mom] [N] — прислать хвост логов (по умолчанию mom, 200 строк)\n"
    "/net — краткая сетёвая статистика из логов автотрейда\n"
    "\nПримеры:\n"
    "• /bulk                     → BULK_CMD как в ENV\n"
    "• /bulk  --interval 240     → BULK_CMD + '--interval 240'\n"
    "• /eval --dry-run           → EVAL_CMD + '--dry-run'\n"
    "• /logs bot 300             → 300 строк лога бота\n"
)

def _args(text: str):
    parts = (text or "").strip().split()
    return parts[1:] if len(parts) > 1 else []

def cmd_help(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.effective_chat.id, text=HELP_TEXT)

def cmd_bulk(update: Update, context: CallbackContext):
    sfx = " ".join(_args(update.message.text))
    base = _inject_universe_if_missing(BULK_CMD)
    cmd = base + ((" " + sfx) if sfx else "")
    context.bot.send_message(chat_id=update.effective_chat.id, text=f"🚀 bulk стартует…\n{cmd}")
    try:
        res = _run(cmd)
        out = res.stdout or ""
        code = res.returncode
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"✅ bulk завершён (code={code})")

        # пробуем достать путь из stdout, иначе — последний файл из REPORT_DIR
        path = _parse_lastbulk(out) or _latest_xlsx_in_report()
        if path:
            _send_file(context, update.effective_chat.id, path, caption="📎 bulk report")
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ XLSX не найден (проверь REPORT_DIR/BULK_CMD).")

        tail = "\n".join(out.strip().splitlines()[-50:])
        if tail:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f"🧾 tail:\n{tail[:3900]}")
    except subprocess.TimeoutExpired:
        context.bot.send_message(chat_id=update.effective_chat.id, text="⏱️ bulk: timeout.")
    except Exception as e:
        log.exception("bulk error")
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ bulk error: {e}")

def cmd_eval(update: Update, context: CallbackContext):
    sfx = " ".join(_args(update.message.text))
    cmd = EVAL_CMD + (" " + EVAL_CMD_EXTRA if EVAL_CMD_EXTRA else "") + ((" " + sfx) if sfx else "")
    context.bot.send_message(chat_id=update.effective_chat.id, text=f"🧪 eval стартует…\n{cmd}")
    try:
        res = _run(cmd)
        out = res.stdout or ""
        code = res.returncode
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"✅ eval завершён (code={code})")
        path = _parse_lastbulk(out) or _latest_xlsx_in_report()
        if path:
            _send_file(context, update.effective_chat.id, path, caption="📎 eval report")
        tail = "\n".join(out.strip().splitlines()[-80:])
        if tail:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f"🧾 tail:\n{tail[:3900]}")
    except subprocess.TimeoutExpired:
        context.bot.send_message(chat_id=update.effective_chat.id, text="⏱️ eval: timeout.")
    except Exception as e:
        log.exception("eval error")
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ eval error: {e}")

def cmd_bulk_eval(update: Update, context: CallbackContext):
    cmd_bulk(update, context)
    time.sleep(1.0)
    cmd_eval(update, context)

def _read_last_momentum_log() -> str:
    return _latest_file_by_glob(MOMENTUM_LOG_GLOB)

def cmd_checkpoint(update: Update, context: CallbackContext):
    path = _read_last_momentum_log()
    if not path:
        context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Лог автотрейда не найден (MOMENTUM_LOG_GLOB).")
        return
    txt = _tail(path, 800)
    bar_closes = [ln for ln in txt.splitlines() if "[BAR_CLOSE]" in ln][-5:]
    fallbacks  = [ln for ln in txt.splitlines() if "REST итог" in ln or "[FALLBACK]" in ln][-8:]
    msg = "🔎 checkpoint (по логам автотрейда)\n"
    if bar_closes:
        msg += "• Последние BAR_CLOSE:\n" + "\n".join(bar_closes[-3:]) + "\n"
    if fallbacks:
        msg += "• REST-сводки/сообщения:\n" + "\n".join(fallbacks[-3:]) + "\n"
    if not bar_closes and not fallbacks:
        msg += "• В последних логах не найдено [BAR_CLOSE]/REST."
    context.bot.send_message(chat_id=update.effective_chat.id, text=msg[:3900])

def cmd_logs(update: Update, context: CallbackContext):
    args = _args(update.message.text)
    kind = (args[0].lower() if args else "mom")
    try:
        n = int(args[1]) if len(args) >= 2 else 200
    except Exception:
        n = 200

    if kind in ("bot", "tg", "tgctrl"):
        path = _latest_file_by_glob(TGCTRL_LOG_GLOB) or _runtime_log
    else:
        path = _read_last_momentum_log()

    if not path:
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Лог не найден (kind={kind}).")
        return
    tail = _tail(path, n)
    context.bot.send_message(chat_id=update.effective_chat.id, text=f"🗒️ {path}\n\n{tail[:3900]}")

def cmd_net(update: Update, context: CallbackContext):
    path = _read_last_momentum_log()
    if not path:
        context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Лог автотрейда не найден (MOMENTUM_LOG_GLOB).")
        return
    txt = _tail(path, 2000)
    timeouts = len(re.findall(r"\[REST_TIMEOUT\]", txt))
    conns    = len(re.findall(r"\[REST_CONNECT\]", txt))
    errs     = len(re.findall(r"\[REST_(?:RETRY|KLINE_ERR|FALLBACK_ERR)\]", txt))
    summary  = re.findall(r"timeouts=\d+,\s*conn_errs=\d+", txt)
    msg = "🌐 Сеть (по последним логам):\n"
    msg += f"• timeouts: {timeouts}\n• connect_errs: {conns}\n• other_rest_errs: {errs}\n"
    if summary:
        msg += "• " + summary[-1]
    context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

def fallback_unknown(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.effective_chat.id, text=HELP_TEXT)

# ========= main =========
def main():
    if not TELEGRAM_TOKEN or not CHAT_ID:
        log.error("TELEGRAM_TOKEN/CHAT_ID не заданы — проверь ENV.")
        print("❌ TELEGRAM_TOKEN/CHAT_ID не заданы — проверь ENV.")
        sys.exit(1)

    log.info("tg_control starting…")
    log.info(f"REPORT_DIR={REPORT_DIR}")
    log.info(f"MOMENTUM_LOG_GLOB={MOMENTUM_LOG_GLOB}")
    log.info(f"TGCTRL_LOG_GLOB={TGCTRL_LOG_GLOB}")
    log.info(f"BULK_CMD='{BULK_CMD}'")
    log.info(f"EVAL_CMD='{EVAL_CMD}' EXTRA='{EVAL_CMD_EXTRA}'")
    if os.getenv("TRADE_UNIVERSE"):
        log.info(f"TRADE_UNIVERSE (ENV) present: '{os.getenv('TRADE_UNIVERSE')}'")

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("bulk", cmd_bulk))
    dp.add_handler(CommandHandler("eval", cmd_eval))
    dp.add_handler(CommandHandler("bulk_eval", cmd_bulk_eval))
    dp.add_handler(CommandHandler("checkpoint", cmd_checkpoint))
    dp.add_handler(CommandHandler("logs", cmd_logs))
    dp.add_handler(CommandHandler("net", cmd_net))
    dp.add_handler(MessageHandler(Filters.command, fallback_unknown))

    try:
        Bot(TELEGRAM_TOKEN).send_message(chat_id=CHAT_ID, text="✅ tg_control запущен.")
    except Exception as e:
        log.warning(f"Не удалось отправить стартовый пинг: {e}")

    updater.start_polling()
    log.info("tg_control polling started.")
    updater.idle()
    log.info("tg_control stopped.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("FATAL in tg_control")
        raise