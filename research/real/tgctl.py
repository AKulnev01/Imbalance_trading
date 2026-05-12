import os
import asyncio
import subprocess
import shlex
import pathlib
import json
import time
from datetime import datetime, timezone
import pandas as pd

from telegram.ext import Updater, CommandHandler
from telegram import ParseMode

# === ENV ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")

# Где лежат отчёты (generate_signals_variant пишет в ~/Documents/отчеты)
DEFAULT_REPORT_DIR = os.path.expanduser("~/Documents/отчеты")
REPORT_DIR = pathlib.Path(os.getenv("REPORT_DIR", DEFAULT_REPORT_DIR))

LOG_DIR  = pathlib.Path(os.getenv("LOGS_DIR", "logs"))
LOG_FILE = LOG_DIR / "momentum_live.log"

# RR (как в live/eval)
def _fenv(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)

MOMENTUM_TP_PCT = _fenv("MOMENTUM_TP_PCT", 0.135)  # 13.5%
MOMENTUM_SL_PCT = _fenv("MOMENTUM_SL_PCT", 0.04)   # 4.0%

# ========= утилиты =========
def _auth(update):
    if not CHAT_ID:
        return True
    try:
        return str(update.effective_chat.id) == str(CHAT_ID)
    except Exception:
        return False

def _calc_tpsl_entry(entry: float, side: str, take_pct: float, stop_pct: float):
    if entry is None or pd.isna(entry):
        return (None, None)
    e = float(entry)
    s = str(side).strip().upper()
    if s == "SELL" or s == "SHORT":
        sl = e * (1.0 + float(stop_pct))
        tp = e - (sl - e) * (float(take_pct) / max(float(stop_pct), 1e-9))
    else:
        sl = e * (1.0 - float(stop_pct))
        tp = e + (e - sl) * (float(take_pct) / max(float(stop_pct), 1e-9))
    return (float(sl), float(tp))

def start(update, ctx):
    if not _auth(update): return
    update.message.reply_text(
        "✅ tgctl online.\n"
        "Команды:\n"
        "• /status\n"
        "• /signals [lookback_days] [interval] [mode] — сгенерировать сигналы и добавить entry/TP/SL\n"
        "• /eval_api N — быстрый API-eval для последних N сигналов (по свежему файлу *signals*)\n"
        "• /set_slip e x s — обновить ENTRY/EXIT/STOP slippage (например 0.004)\n"
        "• /tail — показать хвост live-лога"
    )

def status(update, ctx):
    if not _auth(update): return
    env = {
        "TP%": os.getenv("MOMENTUM_TP_PCT"),
        "SL%": os.getenv("MOMENTUM_SL_PCT"),
        "ENTRY_SLIP%": os.getenv("ENTRY_SLIPPAGE_PCT", os.getenv("SLIPPAGE_PCT","")),
        "EXIT_SLIP%": os.getenv("EXIT_SLIPPAGE_PCT", os.getenv("SLIPPAGE_PCT","")),
        "STOP_SLIP%": os.getenv("STOP_SLIPPAGE_PCT", os.getenv("EXIT_SLIPPAGE_PCT","")),
        "TTL_DAYS": os.getenv("DEFAULT_TTL_DAYS"),
        "MAX_CONCURRENT": os.getenv("MAX_CONCURRENT_POSITIONS"),
        "REPORT_DIR": str(REPORT_DIR),
    }
    lines = [f"{k}: {v}" for k,v in env.items()]
    update.message.reply_text("📊 STATUS\n" + "\n".join(lines))

def _find_latest_signals_path():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cands = sorted(
        list(REPORT_DIR.glob("*signals*.xlsx")) + list(REPORT_DIR.glob("*signals*.csv")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return cands[0] if cands else None

def eval_api(update, ctx):
    if not _auth(update): return
    try:
        n = int(ctx.args[0]) if ctx.args else 5
    except Exception:
        n = 5
    update.message.reply_text(f"⏱️ Запускаю quick API eval для последних {n} сигналов…")
    sig = _find_latest_signals_path()
    if not sig:
        update.message.reply_text("⚠️ Не нашёл файл сигналов в REPORT_DIR")
        return

    cmd = f'python -u eval_quick_api.py "{sig}" --last {n} --use-mainnet --category linear'
    try:
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT, cwd=str(pathlib.Path(__file__).parent))
        update.message.reply_text(out.decode("utf-8", errors="ignore")[-4000:])
    except subprocess.CalledProcessError as e:
        update.message.reply_text("❌ Ошибка eval:\n" + e.output.decode("utf-8", errors="ignore")[-4000:])
        return

    out_xlsx = pathlib.Path(sig).with_suffix("").as_posix() + "_quick_api_eval.xlsx"
    p = pathlib.Path(out_xlsx)
    if p.exists():
        update.message.reply_document(document=p.open("rb"), filename=p.name, caption="✅ Готово")
    else:
        update.message.reply_text("⚠️ Файл результата не найден")

def set_slip(update, ctx):
    if not _auth(update): return
    try:
        e = float(ctx.args[0]); x = float(ctx.args[1]); s = float(ctx.args[2])
    except Exception:
        update.message.reply_text("Использование: /set_slip <entry> <exit> <stop> (доли, например 0.004)")
        return
    env_path = pathlib.Path(".env")
    with env_path.open("a", encoding="utf-8") as f:
        f.write(f"\nENTRY_SLIPPAGE_PCT={e}\nEXIT_SLIPPAGE_PCT={x}\nSTOP_SLIPPAGE_PCT={s}\n")
    update.message.reply_text(
        f"✅ Обновил .env:\nENTRY={e}\nEXIT={x}\nSTOP={s}\n(скрипт подхватит за ENV_RELOAD_SEC)"
    )

def tail(update, ctx):
    if not _auth(update): return
    if not LOG_FILE.exists():
        update.message.reply_text("Лог пока отсутствует.")
        return
    txt = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]
    update.message.reply_text("```\n" + "\n".join(txt[-400:]) + "\n```", parse_mode=ParseMode.MARKDOWN)

# ============= /signals =============
def signals(update, ctx):
    if not _auth(update): return
    # Аргументы: [lookback_days] [interval] [mode]
    try:
        lookback = int(ctx.args[0]) if len(ctx.args) >= 1 else 60
        interval = str(ctx.args[1]).lower() if len(ctx.args) >= 2 else "4h"
        mode     = str(ctx.args[2]).lower() if len(ctx.args) >= 3 else "all"
    except Exception:
        lookback, interval, mode = 60, "4h", "all"

    # Имя файла
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename = f"signals_{interval}_{ts}.xlsx"
    out_path = REPORT_DIR / filename

    update.message.reply_text(
        f"🔄 Генерирую сигналы: lookback={lookback}d, TF={interval}, mode={mode}…"
    )

    # Запуск твоего скрипта
    cmd = f'python -u scripts/generate_signals_variant.py "{filename}" {lookback} {interval} {mode}'
    try:
        # cwd — корень проекта (рядом с scripts/)
        proj_root = pathlib.Path(__file__).parent.resolve()
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT, cwd=str(proj_root))
        tail_msg = out.decode("utf-8", errors="ignore")[-800:]
        try:
            update.message.reply_text("```" + tail_msg + "```", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            update.message.reply_text(tail_msg)
    except subprocess.CalledProcessError as e:
        update.message.reply_text("❌ Ошибка генерации сигналов:\n" + e.output.decode("utf-8", errors="ignore")[-4000:])
        return

    # Проверяем, что файл появился
    if not out_path.exists():
        # Скрипт пишет в ~/Documents/отчеты — туда мы и смотрим
        cand = _find_latest_signals_path()
        if cand:
            out_path = cand
        else:
            update.message.reply_text("⚠️ Файл сигналов не найден после генерации.")
            return

    # Загружаем и считаем entry/tp/sl
    try:
        df = pd.read_excel(out_path, sheet_name=0, engine="openpyxl")
    except Exception:
        try:
            df = pd.read_csv(out_path)
        except Exception as e:
            update.message.reply_text(f"❌ Не смог прочитать {out_path.name}: {e}")
            return

    # Нормализация колонок
    for col in ("type","side"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()

    # Выбор entry
    entry_col = None
    if "entry_px_ref" in df.columns:
        entry_col = "entry_px_ref"
    elif "close2" in df.columns:
        entry_col = "close2"
    elif "entry" in df.columns:
        entry_col = "entry"

    if entry_col is None:
        # если нет подходящей колонки — создадим пустые и отдадим просто сигналы
        df["entry"] = pd.NA
        df["sl"] = pd.NA
        df["tp"] = pd.NA
    else:
        df["entry"] = pd.to_numeric(df[entry_col], errors="coerce")

        # сторона
        if "type" in df.columns:
            side_series = df["type"].fillna("")
        elif "side" in df.columns:
            side_series = df["side"].fillna("")
        else:
            side_series = pd.Series(["BUY"] * len(df))

        sl_list, tp_list = [], []
        for e, s in zip(df["entry"], side_series):
            if pd.isna(e):
                sl_list.append(pd.NA); tp_list.append(pd.NA); continue
            sl, tp = _calc_tpsl_entry(e, s, MOMENTUM_TP_PCT, MOMENTUM_SL_PCT)
            sl_list.append(sl); tp_list.append(tp)
        df["sl"] = sl_list
        df["tp"] = tp_list

    # Ужимаем до нужного набора колонок (если есть)
    cols_pref = [c for c in ["symbol","type","strength","imb_time","entry","tp","sl"] if c in df.columns]
    cols_pref = cols_pref + [c for c in df.columns if c not in cols_pref]  # но остальные тоже сохраняем
    df = df[cols_pref]

    # Пишем новый файл рядом
    out2 = out_path.with_name(out_path.stem + "_with_tpsl.xlsx")
    with pd.ExcelWriter(out2, engine="openpyxl") as wr:
        df.to_excel(wr, index=False, sheet_name="data")
        meta = pd.DataFrame({
            "param": ["generated_utc","rows","tp_pct","sl_pct","source_file"],
            "value": [
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                len(df),
                MOMENTUM_TP_PCT,
                MOMENTUM_SL_PCT,
                out_path.name
            ]
        })
        meta.to_excel(wr, index=False, sheet_name="meta")

    # Отправляем
    try:
        with open(out2, "rb") as f:
            update.message.reply_document(document=f, filename=out2.name,
                                          caption=f"✅ Сигналы + TP/SL (lookback={lookback}d, TF={interval}, mode={mode})")
    except Exception as e:
        update.message.reply_text(f"⚠️ Не удалось отправить файл: {e}\nСмотри: {out2}")

def main():
    # Гарантируем каталог отчётов
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    up = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = up.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("status", status))
    dp.add_handler(CommandHandler("signals", signals))
    dp.add_handler(CommandHandler("eval_api", eval_api))
    dp.add_handler(CommandHandler("set_slip", set_slip))
    dp.add_handler(CommandHandler("tail", tail))
    up.start_polling(drop_pending_updates=True)
    up.idle()

if __name__ == "__main__":
    main()