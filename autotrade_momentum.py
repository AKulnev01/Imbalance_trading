# autotrade_momentum.py
# ======================
# Боевой MOMENTUM (совместим с текущими evaluate_momentum):
# • Детект FVG (как в bulk) на закрытии 4h.
# • Вход MARKET ровно на закрытии 4h.
# • TP/SL от фактического fill (evaluate_momentum: mode=entry).
# • TTL позиций — в ЧАСАХ (по умолчанию 80), затем жёсткое закрытие MARKET.
# • Разделение универса по категориям (linear/spot) через ENV.
# • WS приоритетно, можно отключить и торговать только REST.
# • Параллельный REST (конкуррентность настраивается), телеграм-алерты.
#
# ==== ШПАРГАЛКА (bash/zsh) ====
# mkdir -p logs; TS=$(date +%F_%H%M); LOG="logs/momentum_${TS}.log"; \
# PYTHONUNBUFFERED=1 nohup python -u autotrade_momentum.py > "$LOG" 2>&1 & \
# echo $! > momentum.pid; ln -sf "$LOG" logs/momentum_latest.log; \
# echo "Started PID=$(cat momentum.pid) ; log=$LOG"
#
# tail -f logs/momentum_latest.log
#
# (kill $(cat momentum.pid) 2>/dev/null || true; rm -f momentum.pid)
#
# ( kill $(cat momentum.pid) 2>/dev/null || true; sleep 1; \
#   TS=$(date +%F_%H%M); LOG="logs/momentum_${TS}.log"; \
#   PYTHONUNBUFFERED=1 nohup python -u autotrade_momentum.py > "$LOG" 2>&1 & \
#   echo $! > momentum.pid; ln -sf "$LOG" logs/momentum_latest.log; \
#   echo "Restarted PID=$(cat momentum.pid) ; log=$LOG" )
# ================================================

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import random

import pandas as pd
import aiohttp
from telegram import Bot  # python-telegram-bot v13
from telegram.ext import Updater, CommandHandler
import tempfile, io

# --- проектные утилиты ---
from utils.bybit_trade import (
    open_position_market, close_position_market,
    get_wallet_balance, get_positions
)
import utils.bybit_trade as bybit_api  # amend TP/SL, плечо/маржин-режим

from utils.symbols import fetch_top_symbols
from utils.allocator import SmartAllocator
# БЫЛО:
# from utils.detect_fvg import detect_fvg_imbalances

# СТАЛО:
from utils.detect_fvg_close import detect_fvg_imbalances_close as detect_fvg_imbalances
from evaluate_common import get_cfg

from config import (
    BYBIT_CATEGORY, USE_MAINNET_MARKET_DATA,  # дефолт, но ниже используем per-symbol
    TELEGRAM_TOKEN, CHAT_ID,
    INITIAL_CAPITAL, POSITION_FRACTION,
    ENABLE_BUY, ENABLE_SELL,
    TRADE_UNIVERSE,
    ENTRY_DETECT_TOL_SEC
)


# ---------- Логирование ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("auto-momentum")

# ---------- ENV / горячая подгрузка ----------
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

_ENV_RELOAD_SEC = int(os.getenv("ENV_RELOAD_SEC", "10"))
_LAST_ENV_READ = 0.0

# ======== Параметры: детект FVG (как в bulk/generate_signals_grid) ========
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

DEFAULT_MIN_STRENGTH = _env_float("DEFAULT_MIN_STRENGTH", _env_float("MIN_STRENGTH_PCT", 0.0))
FVG_VOL_MULT         = _env_float("FVG_VOL_MULT",         _env_float("VOLUME_MULTIPLIER", 1.0))
FVG_TOLERANCE_PCT    = _env_float("FVG_TOLERANCE_PCT",    _env_float("TOLERANCE_PCT", 0.0))

# Лимит одновременных позиций
MAX_CONCURRENT = int(get_cfg("MAX_CONCURRENT_POSITIONS", cast=int, default=0))  # 0 = без лимита
MAX_CANDLES = 200

# --- REST close scanner (батчи по юниверсу)
REST_GROUP_SIZE = int(os.getenv("REST_GROUP_SIZE", "20"))            # размер группы
REST_GROUP_PAUSE_SEC = float(os.getenv("REST_GROUP_PAUSE_SEC", "0.6"))   # пауза между группами
REST_SYMBOL_PAUSE_SEC = float(os.getenv("REST_SYMBOL_PAUSE_SEC", "0.08")) # пауза между символами
REST_SCAN_JITTER_SEC = float(os.getenv("REST_SCAN_JITTER_SEC", "0.2"))    # мелкий джиттер после close
RUN_REST_ON_CLOSE = str(os.getenv("RUN_REST_ON_CLOSE","1")).lower() in ("1","true","yes","y","on")

# ======== Риск-профиль под eval ========
MOM_TAKE_PCT   = _env_float("MOMENTUM_TP_PCT", 0.135)  # 13.5%
MOM_STOP_PCT   = _env_float("MOMENTUM_SL_PCT", 0.04)   # 4.0%
MOM_TTL_HOURS  = int(os.getenv("MOMENTUM_TTL_HOURS", "80"))

# вверху рядом со слиппеджами
AMEND_TPSL_FROM_FILL = str(os.getenv("AMEND_TPSL_FROM_FILL", "1")).lower() in ("1","true","yes","y","on")

# Слиппеджи/комиссии (для логов и контроля дрейфа)
ENTRY_SLIPPAGE_PCT = _env_float("ENTRY_SLIPPAGE_PCT", 0.004)  # 0.4%
EXIT_SLIPPAGE_PCT  = _env_float("EXIT_SLIPPAGE_PCT",  0.004)
STOP_SLIPPAGE_PCT  = _env_float("STOP_SLIPPAGE_PCT",  EXIT_SLIPPAGE_PCT)
FEE_TAKER_LIVE     = _env_float("FEE_TAKER",          _env_float("FEE_TAKER_PCT", 0.001))

# --- Плечо/маржин-режим
BYBIT_MARGIN_MODE = str(get_cfg("BYBIT_MARGIN_MODE", cast=str, default="")).strip().lower()  # isolated|cross|''
BYBIT_LEVERAGE    = float(get_cfg("BYBIT_LEVERAGE",    cast=float, default=1))

# Режим постановки TP/SL — только "entry"
MOM_TPSL_MODE = "entry"

OUT_TZ   = os.getenv("OUT_TZ", None)
MAX_DRIFT = _env_float("MAX_ACCEPT_SLIPPAGE_PCT", 0.004)  # допуcтимый дрейф fill к detect

# --- WS/REST ---
USE_WS = str(os.getenv("USE_WS", "1")).lower() in ("1","true","yes","y","on")

REST_ATTEMPTS      = int(os.getenv("REST_ATTEMPTS", "3"))
REST_TIMEOUT_SEC   = _env_float("REST_TIMEOUT_SEC", 10)
FALLBACK_CONC      = int(os.getenv("FALLBACK_CONCURRENCY", "20"))
REST_AFTER_CLOSE_DELAY_SEC = int(os.getenv("REST_AFTER_CLOSE_DELAY_SEC", "6"))
REST_CONNECT_TIMEOUT = _env_float("REST_CONNECT_TIMEOUT", 5)
REST_READ_TIMEOUT    = _env_float("REST_READ_TIMEOUT", 10)
REST_ALLOW_SPOT_FALLBACK = str(os.getenv("REST_ALLOW_SPOT_FALLBACK", "0")).lower() in ("1","true","yes","y","on")

# счётчики
_net_batch_timeouts = 0
_net_batch_errors = 0

# --- WS endpoints v5 (public) ---
WS_MAIN_SPOT   = "wss://stream.bybit.com/v5/public/spot"
WS_TEST_SPOT   = "wss://stream-testnet.bybit.com/v5/public/spot"
WS_MAIN_LINEAR = "wss://stream.bybit.com/v5/public/linear"
WS_TEST_LINEAR = "wss://stream-testnet.bybit.com/v5/public/linear"

def _ws_url(category_default: str) -> str:
    use_mainnet = bool(get_cfg("USE_MAINNET_MARKET_DATA", cast=bool, default=USE_MAINNET_MARKET_DATA))
    cat = str(get_cfg("BYBIT_CATEGORY", cast=str, default=category_default)).strip().lower()
    if cat == "spot":
        return WS_MAIN_SPOT if use_mainnet else WS_TEST_SPOT
    return WS_MAIN_LINEAR if use_mainnet else WS_TEST_LINEAR

# HTTP base
HTTP_BASE_MAIN = "https://api.bybit.com"
HTTP_BASE_TEST = "https://api-testnet.bybit.com"
def http_base() -> str:
    use_mainnet = bool(get_cfg("USE_MAINNET_MARKET_DATA", cast=bool, default=USE_MAINNET_MARKET_DATA))
    return HTTP_BASE_MAIN if use_mainnet else HTTP_BASE_TEST

# ====== глобальное состояние ======
class _DummyState:
    def all_open(self): return {}
    def upsert_open(self, *a, **kw): pass
    def pop_open(self, *a, **kw): pass

_state = _DummyState()
_allocator = SmartAllocator(INITIAL_CAPITAL)

# свечи в памяти (4h) раздельно по категориям
_candles_linear: Dict[str, List[dict]] = {}
_candles_spot:   Dict[str, List[dict]] = {}
FVG_INTERVAL = "240"  # 4h

# уже обработанные 4h-бары
_last_done: Dict[str, pd.Timestamp] = {}
_ws_last_close_bar: Dict[str, pd.Timestamp] = {}

# ---- Разделение универса (ENV) ----
def _parse_universe(env_key: str) -> List[str]:
    raw = os.getenv(env_key, "")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]

UNIVERSE_LINEAR = set(_parse_universe("TRADE_UNIVERSE_LINEAR"))
UNIVERSE_SPOT   = set(_parse_universe("TRADE_UNIVERSE_SPOT"))

# если пусто — берём TRADE_UNIVERSE целиком в дефолтную категорию BYBIT_CATEGORY
if not UNIVERSE_LINEAR and not UNIVERSE_SPOT:
    base_list = os.getenv("TRADE_UNIVERSE", ",".join(TRADE_UNIVERSE) if isinstance(TRADE_UNIVERSE, (list,tuple)) else str(TRADE_UNIVERSE))
    UNIVERSE_LINEAR = set([s.strip().upper() for s in base_list.split(",") if s.strip()])

SYMBOL_CATEGORY: Dict[str, str] = {}
for s in UNIVERSE_LINEAR:
    SYMBOL_CATEGORY[s] = "linear"
for s in UNIVERSE_SPOT:
    SYMBOL_CATEGORY[s] = "spot"

def category_of(symbol: str) -> str:
    return SYMBOL_CATEGORY.get(symbol.upper(), BYBIT_CATEGORY)

# Telegram — только отправка
_bot: Optional[Bot] = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")  # строкой; ниже сравним по str()
_STOP_EVENT: Optional[asyncio.Event] = None
_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None
async def tg(text: str):
    if not _bot or not CHAT_ID: return
    chat = int(CHAT_ID) if str(CHAT_ID).lstrip("-").isdigit() else CHAT_ID
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _bot.send_message(chat_id=chat, text=text))

async def tg_big(text: str, chunk: int = 3800):
    if not _bot or not CHAT_ID or not text: return
    for i in range(0, len(text), chunk):
        await tg(text[i:i+chunk])

# ====== хелперы ======
def _is_admin(update) -> bool:
    try:
        chat_id = str(update.effective_chat.id)
        return chat_id == str(ADMIN_CHAT_ID) or chat_id == str(CHAT_ID)
    except Exception:
        return False

def _deny_if_not_admin(update, context):
    if not _is_admin(update):
        update.message.reply_text("⛔️ Доступ запрещён.")
        return True
    return False

async def _build_signals_csv_last_hours(hours: int, symbols: list, category: str) -> str:
    """
    Собрать imbalances за последние `hours` часов по 4h-истории.
    Возвращает путь к временномy CSV с колонками: symbol,type,strength,imb_time
    """
    look_bars = max(15, int(hours/4) + 10)
    recs = []

    async with _make_session() as session:
        for sym in symbols:
            try:
                df = await _fetch_4h_df(session, sym, category, limit=min(200, look_bars))
                if df is None or df.empty:
                    continue
                imbs = detect_fvg_imbalances(
                    df,
                    volume_multiplier=FVG_VOL_MULT,
                    tolerance_pct=FVG_TOLERANCE_PCT,
                    min_strength_pct=DEFAULT_MIN_STRENGTH,
                ) or []
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
                for m in imbs:
                    t = pd.to_datetime(m.get("time"), utc=True, errors="coerce")
                    if pd.isna(t) or t < cutoff:
                        continue
                    side = str(m.get("type","")).upper()
                    if side not in ("BUY","SELL"):
                        continue
                    if (side == "BUY" and not ENABLE_BUY) or (side == "SELL" and not ENABLE_SELL):
                        continue
                    recs.append({
                        "symbol": sym,
                        "type": side,
                        "strength": float(m.get("strength", 0.0)),
                        "imb_time": t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
                    })
            except Exception as e:
                log.warning(f"[TG_EVAL] signals {sym}: {e}")

    if not recs:
        # пустой, но валидный csv
        recs = [{"symbol":"", "type":"", "strength":0.0, "imb_time":""}]

    df_out = pd.DataFrame(recs, columns=["symbol","type","strength","imb_time"])
    tmp = tempfile.NamedTemporaryFile(suffix="_signals.xlsx", delete=False)
    df_out.to_excel(tmp.name, index=False, engine="openpyxl")
    tmp.close()
    return tmp.name

def _eval_and_pack(signals_csv: str, hours: int) -> bytes:
    """
    Синхронный помощник: вызывает evaluate_momentum и возвращает байты XLSX.
    Запускается в thread executor, чтобы не блокировать loop.
    """
    from evaluate_momentum import evaluate_momentum
    # Куда положить XLSX
    out_xlsx = os.path.splitext(signals_csv)[0] + f"_momentum_eval_{hours}h.xlsx"
    # Вызов оценки (наши ENV уже выставлены)
    evaluate_momentum(
        signals_path=signals_csv,
        result_path=out_xlsx,
        lookback_days=360,
        interval="4h",
        max_days=None,
        only_filled=False,
        dedup=False,
        initial_capital=None,
        capital_aware=True,
    )
    with open(out_xlsx, "rb") as f:
        return f.read()

def _cmd_status(update, context):
    if _deny_if_not_admin(update, context): return
    try:
        bal = get_wallet_balance("USDT")
    except Exception:
        bal = float(_alloc_total_usd())
    close_ts, left = _next_4h_close_utc()
    uni = [s for s in SYMBOL_CATEGORY.keys()]
    msg = (f"📊 Status\n"
           f"• Баланс: {bal:.2f} USDT\n"
           f"• Символов (linear/spot): "
           f"{sum(1 for x in uni if SYMBOL_CATEGORY[x]=='linear')}/"
           f"{sum(1 for x in uni if SYMBOL_CATEGORY[x]=='spot')}\n"
           f"• RR: TP={MOM_TAKE_PCT*100:.2f}%, SL={MOM_STOP_PCT*100:.2f}%\n"
           f"• TTL: {MOM_TTL_HOURS}h, Max concurrent: {MAX_CONCURRENT}\n"
           f"• До закрытия 4h: {left}\n")
    update.message.reply_text(msg)

def _cmd_reload(update, context):
    if _deny_if_not_admin(update, context): return
    _reload_env_if_needed()
    update.message.reply_text("♻️ ENV перечитан (hot reload).")

def _cmd_stop(update, context):
    if _deny_if_not_admin(update, context): return
    update.message.reply_text("🛑 Останавливаю процессы…")
    try:
        if _MAIN_LOOP and _STOP_EVENT:
            _MAIN_LOOP.call_soon_threadsafe(_STOP_EVENT.set)
        else:
            log.warning("[TG] stop: main loop not ready yet")
    except Exception as e:
        log.error(f"[TG] stop signal error: {e}")


def _cmd_eval_mom(update, context):
    if _deny_if_not_admin(update, context):
        return
    try:
        hours = int(context.args[0]) if context.args else 24
        update.message.reply_text(f"🧪 Генерирую signals + eval за последние {hours}h…")

        syms_linear = [s for s, cat in SYMBOL_CATEGORY.items() if cat == "linear"]
        syms_spot   = [s for s, cat in SYMBOL_CATEGORY.items() if cat == "spot"]
        category = "linear" if syms_linear else "spot"
        symbols  = syms_linear if syms_linear else syms_spot

        async def _job():
            try:
                sig_csv = await _build_signals_csv_last_hours(hours, symbols, category)
                # тут вызов уже нашего «совместимого» eval
                xlsx_bytes = await asyncio.to_thread(_eval_and_pack, sig_csv, hours)
                bio = io.BytesIO(xlsx_bytes)
                bio.name = f"momentum_eval_{hours}h.xlsx"
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: _bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=bio,
                    caption=f"✅ MOMENTUM eval • last {hours}h"
                ))
            except Exception as e:
                try:
                    update.message.reply_text(f"❌ Ошибка eval: {e}")
                except Exception:
                    log.exception("[TG] reply failed")

        if _MAIN_LOOP is None:
            update.message.reply_text("⚠️ Главный event loop ещё не готов, попробуй чуть позже.")
            return
        asyncio.run_coroutine_threadsafe(_job(), _MAIN_LOOP)

    except Exception as e:
        update.message.reply_text(f"❌ Неверные аргументы: {e}\nПример: /eval_mom 48")

def _start_telegram_control():
    if not TELEGRAM_TOKEN:
        log.warning("[TG] TELEGRAM_TOKEN не задан — контроль по ТГ отключён")
        return None
    up = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = up.dispatcher
    dp.add_handler(CommandHandler("status", _cmd_status))
    dp.add_handler(CommandHandler("reload", _cmd_reload))
    dp.add_handler(CommandHandler("stop", _cmd_stop))
    dp.add_handler(CommandHandler("eval_mom", _cmd_eval_mom, pass_args=True))
    up.start_polling(drop_pending_updates=True)
    log.info("[TG] control polling started (/status, /eval_mom <h>, /reload, /stop)")
    return up

# ====== утилиты ======
def _parse_ts_to_utc(ts_val):
    try:
        v = int(ts_val)
        if v > 10**12: return pd.to_datetime(v, unit="ms", utc=True)
        return pd.to_datetime(v, unit="s", utc=True)
    except Exception:
        try: return pd.to_datetime(ts_val, utc=True)
        except Exception: return pd.Timestamp.utcnow()

def _fmt_ts(ts: pd.Timestamp) -> str:
    ts = pd.to_datetime(ts, utc=True)
    if OUT_TZ:
        try: return ts.tz_convert(OUT_TZ).strftime("%Y-%m-%d %H:%M:%S")
        except Exception: pass
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")

def _next_4h_close_utc(now: Optional[pd.Timestamp] = None) -> Tuple[pd.Timestamp, pd.Timedelta]:
    now = now or pd.Timestamp.utcnow()
    hour = (now.hour // 4) * 4
    this_close = now.replace(hour=hour, minute=0, second=0, microsecond=0) + pd.Timedelta(hours=4)
    if this_close <= now: this_close += pd.Timedelta(hours=4)
    return this_close, this_close - now

def _calc_tpsl_entry(entry: float, side: str, risk_pct: float, rr: float) -> Tuple[float, float]:
    k = float(risk_pct)
    if side.upper() == "SELL":
        sl = entry * (1.0 + k)
        tp = entry - (sl - entry) * rr
    else:
        sl = entry * (1.0 - k)
        tp = entry + (entry - sl) * rr
    return float(sl), float(tp)

def _amend_tpsl_safe(symbol: str, tp: float, sl: float, *, category: str) -> bool:
    # 1) наш реальный helper
    fn = getattr(bybit_api, "set_trading_stop", None)
    if callable(fn):
        try:
            fn(symbol=symbol, take_profit=str(tp), stop_loss=str(sl), category=category)
            return True
        except Exception as e:
            log.error(f"[AMEND_FAIL:set_trading_stop] {symbol}: {e}")

    # 2) про запас — если позже добавишь другие алиасы
    for name in ("amend_tpsl","update_position_tpsl","set_position_tpsl","position_set_tp_sl"):
        fn = getattr(bybit_api, name, None)
        if callable(fn):
            try:
                # пробуем обе схемы аргументов
                try:
                    fn(symbol=symbol, tp_price=str(tp), sl_price=str(sl), category=category)
                except TypeError:
                    fn(symbol=symbol, take_profit=str(tp), stop_loss=str(sl), category=category)
                return True
            except Exception as e:
                log.error(f"[AMEND_FAIL:{name}] {symbol}: {e}")
    return False

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

async def _rest_process_symbol_on_close(sym: str, expected_open: pd.Timestamp, category: str):
    """
    Простой REST: взять 4h df, срез до expected_open, детект, вход.
    """
    async with _make_session() as session:
        df_full = await _fetch_4h_df(session, sym, category, limit=min(MAX_CANDLES, 120))
    if df_full is None or df_full.empty:
        log.info(f"[REST_CLOSE] {sym}: df пусто"); return

    df = _slice_to_closed_bar(df_full, expected_open)
    if df is None or df.empty or df.index[-1] != expected_open:
        log.info(f"[REST_CLOSE] {sym}: нет закрытого бара {expected_open}"); return

    args = _pick_and_enter_args(df, sym)
    if args is None:
        log.info(f"[REST_CLOSE] {sym}: сигналов нет (min_strength={DEFAULT_MIN_STRENGTH}, vol×SMA={FVG_VOL_MULT})")
        _last_done[sym] = expected_open
        return

    bar_open, bar_close, detect_px, side, strength = args

    if not _positions_limit_ok(category):
        log.info(f"[MOM_SKIP] {sym}: лимит позиций ({MAX_CONCURRENT})")
        _last_done[sym] = bar_open
        return

    try:
        plist = (get_positions(category=category).get("result", {}) or {}).get("list", []) or []
        if any(p.get("symbol") == sym and abs(float(p.get("size") or 0.0)) > 0 for p in plist):
            log.info(f"[MOM_SKIP] {sym}: уже есть открытая позиция")
            _last_done[sym] = bar_open
            return
    except Exception:
        pass

    await _enter_momentum_market(sym, side, detect_px, bar_open, bar_close, strength=strength)

async def rest_close_scanner_loop():
    """
    Жёсткий запуск сразу после закрытия 4h:
      • ждём ровно до close + REST_AFTER_CLOSE_DELAY_SEC,
      • считаем expected_open = close - 4h,
      • бежим по юниверс батчами,
      • логируем дрейф (сколько опоздали относительно close).
    """
    if not RUN_REST_ON_CLOSE:
        log.info("[REST_CLOSE] disabled (RUN_REST_ON_CLOSE=0)")
        return

    syms_linear = [s for s, cat in SYMBOL_CATEGORY.items() if cat == "linear"]
    syms_spot   = [s for s, cat in SYMBOL_CATEGORY.items() if cat == "spot"]
    if not syms_linear and not syms_spot:
        log.warning("[REST_CLOSE] пустой юниверс (SYMBOL_CATEGORY)")
        return

    log.info(f"[REST_CLOSE] loop started | group={REST_GROUP_SIZE} | sym_pause={REST_SYMBOL_PAUSE_SEC}s | grp_pause={REST_GROUP_PAUSE_SEC}s")

    def _next_close(now=None):
        now = pd.to_datetime(now or pd.Timestamp.utcnow(), utc=True)
        hour = (now.hour // 4) * 4
        close_ts = now.replace(hour=hour, minute=0, second=0, microsecond=0) + pd.Timedelta(hours=4)
        if close_ts <= now:
            close_ts += pd.Timedelta(hours=4)
        return close_ts.tz_convert("UTC")

    async def _sleep_until(ts: pd.Timestamp):
        # ждём ровно до ts c учётом тек. времени
        while True:
            now = pd.Timestamp.utcnow().tz_localize("UTC")
            left = (ts - now).total_seconds()
            if left <= 0: break
            await asyncio.sleep(min(left, 1.0))  # шаг 1 сек для точности и устойчивости к sleep drift

    while True:
        try:
            close_ts = _next_close()
            # хотим стартовать сразу после закрытия бара
            target = close_ts + pd.Timedelta(seconds=float(REST_AFTER_CLOSE_DELAY_SEC or 0))
            await _sleep_until(target)

            # оценим дрейф запуска
            started_at = pd.Timestamp.utcnow().tz_localize("UTC")
            drift = (started_at - target).total_seconds()
            if drift > 2.0:  # больше 2 сек — предупредить
                log.warning(f"[REST_CLOSE] delayed start by {drift:.2f}s (target={target}, started={started_at})")

            expected_open = (close_ts - pd.Timedelta(hours=4)).tz_convert("UTC")
            log.info(f"[REST_CLOSE] processing close={close_ts} expected_open={expected_open}")

            async def _run_for_list(symbols: List[str], category: str):
                if not symbols:
                    return
                async with _make_session() as session:
                    for gidx, grp in enumerate(_chunks(symbols, REST_GROUP_SIZE), 1):
                        now_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                        try:
                            await tg(f"🕐 REST старт: {now_str}")
                        except Exception:
                            pass

                        lines = []
                        total_signals = []
                        window_open, window_close = None, None

                        # прогон группы
                        for sym in grp:
                            try:
                                df = await _fetch_4h_df(session, sym, category, limit=min(MAX_CANDLES, 120))
                                if df is None or df.empty:
                                    if REST_SYMBOL_PAUSE_SEC > 0:
                                        await asyncio.sleep(REST_SYMBOL_PAUSE_SEC)
                                    continue
                                dfs = _slice_to_closed_bar(df, expected_open)
                                if dfs is None or dfs.empty or dfs.index[-1] != expected_open:
                                    if REST_SYMBOL_PAUSE_SEC > 0:
                                        await asyncio.sleep(REST_SYMBOL_PAUSE_SEC)
                                    continue

                                args = _pick_and_enter_args(dfs, sym)
                                if args is None:
                                    if REST_SYMBOL_PAUSE_SEC > 0:
                                        await asyncio.sleep(REST_SYMBOL_PAUSE_SEC)
                                    continue

                                bar_open, bar_close, detect_px, side, strength = args
                                if window_open is None:
                                    window_open, window_close = bar_open.tz_convert("UTC"), (bar_open + pd.Timedelta(hours=4)).tz_convert("UTC")

                                skip_reasons = []

                                try:
                                    if not _positions_limit_ok(category):
                                        skip_reasons.append(f"лимит позиций {MAX_CONCURRENT}")
                                except Exception:
                                    pass

                                try:
                                    pp = (get_positions(category=category).get("result", {}) or {}).get("list", []) or []
                                    if any(p.get("symbol") == sym and abs(float(p.get("size") or 0.0)) > 0 for p in pp):
                                        skip_reasons.append("уже есть открытая позиция")
                                except Exception:
                                    pass

                                try:
                                    usd_wish = float(POSITION_FRACTION) * float(_alloc_total_usd())
                                    free_cash = _estimate_free_cash_usd(category)
                                    if free_cash + 1e-9 < usd_wish:
                                        skip_reasons.append(f"free_cash {free_cash:.2f} < alloc {usd_wish:.2f}")
                                except Exception:
                                    pass

                                total_signals.append((sym, side, detect_px, strength))
                                line = f"💡 {sym} | {side} | цена={detect_px:.6f} | сила={strength:.2f}"
                                if skip_reasons:
                                    line += "\n    пропущен из-за: " + "; ".join(skip_reasons)
                                lines.append(line)

                            except Exception:
                                log.exception(f"[REST_CLOSE] {sym} error")

                            if REST_SYMBOL_PAUSE_SEC > 0:
                                await asyncio.sleep(REST_SYMBOL_PAUSE_SEC)

                        if lines:
                            try:
                                await tg("\n".join(lines[:70]))
                            except Exception:
                                pass

                        try:
                            if window_open is None:
                                txt = f"📊 REST завершён: сигналов=0"
                            else:
                                txt = f"📊 REST завершён: сигналов={len(total_signals)} | окно={window_open} → {window_close}"
                            await tg(txt)
                        except Exception:
                            pass

                        log.info(f"[REST_CLOSE] group {gidx} done ({len(grp)}) [{category}]")
                        if REST_GROUP_PAUSE_SEC > 0:
                            await asyncio.sleep(REST_GROUP_PAUSE_SEC)

            # порядок не критичен
            await _run_for_list(syms_linear, "linear")
            await _run_for_list(syms_spot,   "spot")

        except asyncio.CancelledError:
            log.info("[REST_CLOSE] cancelled")
            return
        except Exception:
            log.exception("[REST_CLOSE] loop error")
            await asyncio.sleep(1.0)

async def rest_catchup_once():
    """
    Если последний закрытый 4h-бар ещё не обработан, обработаем его сразу.
    """
    try:
        close_ts = pd.Timestamp.utcnow().tz_localize("UTC")
        hour = (close_ts.hour // 4) * 4
        close_ts = close_ts.replace(hour=hour, minute=0, second=0, microsecond=0)
        expected_open = (close_ts - pd.Timedelta(hours=4)).tz_convert("UTC")

        # если прямо сейчас ровно на границе (секунды ~0) — подождём чуть-чуть, чтобы Bybit успел отдать бар
        if (pd.Timestamp.utcnow().second < 3):
            await asyncio.sleep(3)

        syms_linear = [s for s, cat in SYMBOL_CATEGORY.items() if cat == "linear"]
        syms_spot   = [s for s, cat in SYMBOL_CATEGORY.items() if cat == "spot"]

        async def _run(symbols, category):
            async with _make_session() as session:
                for sym in symbols:
                    try:
                        df = await _fetch_4h_df(session, sym, category, limit=min(MAX_CANDLES, 120))
                        if df is None or df.empty: continue
                        dfs = _slice_to_closed_bar(df, expected_open)
                        if dfs is None or dfs.empty or dfs.index[-1] != expected_open:
                            continue
                        args = _pick_and_enter_args(dfs, sym)
                        if args is None: continue
                        bar_open, bar_close, detect_px, side, strength = args
                        if not _positions_limit_ok(category): continue
                        await _enter_momentum_market(sym, side, detect_px, bar_open, bar_close, strength=strength)
                    except Exception:
                        log.exception(f"[REST_CATCHUP] {sym} error")

        await _run(syms_linear, "linear")
        await _run(syms_spot,   "spot")
        log.info("[REST_CATCHUP] done")
    except Exception:
        log.exception("[REST_CATCHUP] failed")

# --- плечо/маржа ---
def _try_call(fn_name: str, **kwargs) -> bool:
    fn = getattr(bybit_api, fn_name, None)
    if not callable(fn): return False
    try:
        fn(**kwargs); return True
    except TypeError: return False
    except Exception as e:
        log.error(f"[BYBIT_API:{fn_name}] error: {e}"); return False

def _ensure_margin_mode(symbol: str, category: str) -> bool:
    mode = (BYBIT_MARGIN_MODE or "").lower()
    if not mode or category != "linear": return True
    for name in ("set_margin_mode","position_set_margin_mode","switch_margin_mode"):
        if _try_call(name, symbol=symbol, category=category, marginMode=mode) \
           or _try_call(name, symbol=symbol, category=category, margin_mode=mode) \
           or _try_call(name, symbol=symbol, category=category, mode=mode):
            log.info(f"[MARGIN_MODE] {symbol} → {mode}"); return True
    log.warning(f"[MARGIN_MODE_FAIL] {symbol}: cannot set margin mode='{mode}'")
    return False

def _ensure_leverage(symbol: str, category: str) -> bool:
    if category != "linear": return True
    lev = max(1.0, float(BYBIT_LEVERAGE or 1.0))
    lev_s = str(int(round(lev)))
    for name in ("set_leverage","position_set_leverage","change_leverage","set_symbol_leverage"):
        if _try_call(name, symbol=symbol, category=category, buyLeverage=lev_s, sellLeverage=lev_s) \
           or _try_call(name, symbol=symbol, category=category, leverage=lev_s):
            log.info(f"[LEVERAGE] {symbol} → {lev_s}x"); return True
    log.warning(f"[LEVERAGE_FAIL] {symbol}: cannot set leverage='{lev_s}x'")
    return False

async def _ensure_margin_and_leverage(symbol: str, category: str):
    ok_m = _ensure_margin_mode(symbol, category)
    ok_l = _ensure_leverage(symbol, category)
    if not (ok_m and ok_l):
        await tg(f"⚠️ {symbol} ({category}): не выставились "
                 f"{'маржин-режим' if not ok_m else ''}{' и ' if (not ok_m and not ok_l) else ''}"
                 f"{'плечо' if not ok_l else ''}. Проверь Bybit.")

# ---- капитал ----
def _alloc_total_usd() -> float:
    try:
        for name in ("total","get_total","total_usd","available","balance","equity"):
            attr = getattr(_allocator, name, None)
            v = float(attr() if callable(attr) else attr)
            if v and v > 0: return v
    except Exception: pass
    try:
        v = float(get_wallet_balance("USDT"))
        if v > 0: return v
    except Exception: pass
    return float(INITIAL_CAPITAL or 0.0)

def _safe_equity_usd() -> float:
    """
    Безопасная оценка equity в USDT.
    1) Пытаемся взять реальный баланс кошелька (UNIFIED).
    2) Если не вышло — берём текущее значение аллокатора.
    3) Если и там пусто — откатываемся к INITIAL_CAPITAL.
    """
    try:
        bal = float(get_wallet_balance("USDT"))
        if bal > 0:
            # держим аллокатор синхронизированным, чтобы fraction работал от реального equity
            try:
                _allocator.set_total(bal)
            except Exception:
                pass
            return bal
    except Exception:
        pass

    try:
        # SmartAllocator хранит актуальное total_equity, если мы его уже синкали
        if float(getattr(_allocator, "total_equity", 0.0)) > 0:
            return float(_allocator.total_equity)
    except Exception:
        pass

    return float(INITIAL_CAPITAL or 0.0)

# --- лимит позиций ---
def _positions_limit_ok(category: str) -> bool:
    try:
        if MAX_CONCURRENT <= 0: return True
        pp = get_positions(category=category)
        plist = (pp.get("result", {}) or {}).get("list", []) or []
        active_cnt = sum(1 for p in plist if abs(float(p.get("size") or 0.0)) > 0)
        return active_cnt < MAX_CONCURRENT
    except Exception:
        return True

def _reload_env_if_needed():
    global _LAST_ENV_READ, MOM_TAKE_PCT, MOM_STOP_PCT, MOM_TTL_HOURS, MAX_CONCURRENT
    global DEFAULT_MIN_STRENGTH, FVG_VOL_MULT, FVG_TOLERANCE_PCT, USE_WS, OUT_TZ, MAX_DRIFT
    global ENTRY_SLIPPAGE_PCT, EXIT_SLIPPAGE_PCT, STOP_SLIPPAGE_PCT
    global BYBIT_MARGIN_MODE, BYBIT_LEVERAGE

    now = time.time()
    if _ENV_RELOAD_SEC <= 0 or (now - _LAST_ENV_READ) < _ENV_RELOAD_SEC: return
    if load_dotenv:
        try: load_dotenv(override=True)
        except Exception: pass

    # риск
    try: MOM_TAKE_PCT = float(os.getenv("MOMENTUM_TP_PCT",  str(MOM_TAKE_PCT)))
    except Exception: pass
    try: MOM_STOP_PCT = float(os.getenv("MOMENTUM_SL_PCT",  str(MOM_STOP_PCT)))
    except Exception: pass
    try: MOM_TTL_HOURS = int(os.getenv("MOMENTUM_TTL_HOURS", str(MOM_TTL_HOURS)))
    except Exception: pass

    # FVG
    try: DEFAULT_MIN_STRENGTH = float(os.getenv("DEFAULT_MIN_STRENGTH", os.getenv("MIN_STRENGTH_PCT", str(DEFAULT_MIN_STRENGTH))))
    except Exception: pass
    try: FVG_VOL_MULT = float(os.getenv("FVG_VOL_MULT", os.getenv("VOLUME_MULTIPLIER", str(FVG_VOL_MULT))))
    except Exception: pass
    try: FVG_TOLERANCE_PCT = float(os.getenv("FVG_TOLERANCE_PCT", os.getenv("TOLERANCE_PCT", str(FVG_TOLERANCE_PCT))))
    except Exception: pass

    # лимиты/режимы
    try: MAX_CONCURRENT = int(get_cfg("MAX_CONCURRENT_POSITIONS", cast=int, default=MAX_CONCURRENT))
    except Exception: pass

    OUT_TZ     = os.getenv("OUT_TZ", OUT_TZ)
    MAX_DRIFT  = float(os.getenv("MAX_ACCEPT_SLIPPAGE_PCT", str(MAX_DRIFT)))
    USE_WS     = str(os.getenv("USE_WS", "1")).lower() in ("1","true","yes","y","on")

    # маржа
    BYBIT_MARGIN_MODE = str(get_cfg("BYBIT_MARGIN_MODE", cast=str, default=BYBIT_MARGIN_MODE)).strip().lower()
    try: BYBIT_LEVERAGE = float(get_cfg("BYBIT_LEVERAGE", cast=float, default=BYBIT_LEVERAGE))
    except Exception: BYBIT_LEVERAGE = 1.0

    # слиппедж
    try: ENTRY_SLIPPAGE_PCT = float(os.getenv("ENTRY_SLIPPAGE_PCT", str(ENTRY_SLIPPAGE_PCT)))
    except Exception: pass
    try: EXIT_SLIPPAGE_PCT  = float(os.getenv("EXIT_SLIPPAGE_PCT",  str(EXIT_SLIPPAGE_PCT)))
    except Exception: pass
    try: STOP_SLIPPAGE_PCT  = float(os.getenv("STOP_SLIPPAGE_PCT",  str(STOP_SLIPPAGE_PCT)))
    except Exception: pass

    _LAST_ENV_READ = now
    close_ts, left = _next_4h_close_utc()
    log.info(
        f"[ENV] MOM: TP={MOM_TAKE_PCT*100:.2f}% SL={MOM_STOP_PCT*100:.2f}% TTL={MOM_TTL_HOURS}h; "
        f"FVG: min_strength={DEFAULT_MIN_STRENGTH} vol×SMA={FVG_VOL_MULT} tol={FVG_TOLERANCE_PCT}; "
        f"MAX_CONCURRENT={MAX_CONCURRENT}; USE_WS={USE_WS}; NEXT_4H_CLOSE_IN={str(left)}; OUT_TZ={OUT_TZ or 'UTC'}"
    )

# ====== детект и вход ======
def _last_closed_bar_open_utc(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """
    Возвращает OPEN последнего ЗАКРЫТОГО 4h-бара.
    Пример: сейчас 16:00:01 → last_close=16:00 → open последнего закрытого = 12:00.
    """
    now = pd.to_datetime(now or pd.Timestamp.utcnow(), utc=True)
    hour = (now.hour // 4) * 4
    last_close = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return last_close - pd.Timedelta(hours=4)

def _pick_and_enter_args(df: pd.DataFrame, symbol: str):
    """
    Детект строго на ТОЛЬКО ЧТО ЗАКРЫВШЕМСЯ 4h-баре.
    Возвращает (bar_open, bar_close, detect_px, side, strength) либо None.

    Ключевые отличия:
    • матч по imb['time_open'] ≈ bar_open (а не по imb['time'] ≈ bar_close),
    • detect_px = open[T] (а не close[T]).
    """
    df = df.copy().sort_index()
    if df.empty or len(df.index) < 3:
        return None

    # последний ЗАКРЫТЫЙ бар: open последней строки; close = open + 4ч
    bar_open = df.index[-1]
    bar_close = bar_open + pd.Timedelta(hours=4)

    # входная цена = OPEN бара, который признан имбалансом
    try:
        detect_px = float(df.loc[bar_open, "close"])
    except Exception:
        return None

    # допуск по времени для сопоставления imb.time_open ~ bar_open
    try:
        tol_sec = max(0, int(ENTRY_DETECT_TOL_SEC))
    except Exception:
        tol_sec = 60
    tol = pd.Timedelta(seconds=tol_sec)

    # считаем все FVG
    try:
        imbs = detect_fvg_imbalances(
            df,
            volume_multiplier=FVG_VOL_MULT,
            tolerance_pct=FVG_TOLERANCE_PCT,
            min_strength_pct=DEFAULT_MIN_STRENGTH,
        ) or []
    except Exception as e:
        log.error(f"[DETECT_ERR] {symbol}: {e}")
        return None

    chosen = []
    for imb in imbs:
        # КЛЮЧ: матчим по времени открытия бара-имбаланса
        t_open_imb = pd.to_datetime(imb.get("time_open"), utc=True, errors="coerce")
        if pd.isna(t_open_imb):
            continue
        if abs(t_open_imb - bar_open) > tol:
            continue

        side = str(imb.get("type", "")).upper()
        if side not in ("BUY", "SELL"):
            continue
        if side == "BUY" and not ENABLE_BUY:
            continue
        if side == "SELL" and not ENABLE_SELL:
            continue

        chosen.append({
            "side": side,
            "strength": float(imb.get("strength", 0.0)),
            "time_open": t_open_imb,
            "time_close": pd.to_datetime(imb.get("time"), utc=True, errors="coerce"),
        })

    if not chosen:
        log.debug(f"[ENTRY_PICK] {symbol}: нет имбалансов для time_open≈{bar_open} (bar_close={bar_close})")
        return None

    chosen.sort(key=lambda x: x["strength"], reverse=True)
    ch = chosen[0]

    # диагностика: подтверждаем матчи time_open и покажем связанный time (close)
    t_imb_open_dbg, t_imb_close_dbg = None, None
    try:
        t_imb_open_dbg = next(
            pd.to_datetime(i.get("time_open"), utc=True)
            for i in imbs
            if abs(pd.to_datetime(i.get("time_open"), utc=True) - bar_open) <= tol
        )
        t_imb_close_dbg = next(
            pd.to_datetime(i.get("time"), utc=True)
            for i in imbs
            if abs(pd.to_datetime(i.get("time_open"), utc=True) - bar_open) <= tol
        )
    except StopIteration:
        pass

    log.info(
        f"[ENTRY_PICK] {symbol} "
        f"| imb.time_open={t_imb_open_dbg} "
        f"| imb.time_close={t_imb_close_dbg} "
        f"| bar_open={bar_open} "
        f"| bar_close={bar_close} "
        f"| px(close[T])={detect_px:.6f} "
        f"| side={ch['side']} "
        f"| strength={ch['strength']:.2f}"
    )

    return (bar_open, bar_close, detect_px, ch["side"], ch["strength"])
def _make_session() -> aiohttp.ClientSession:
    try:
        import aiodns  # noqa
        resolver = aiohttp.AsyncResolver()
        connector = aiohttp.TCPConnector(resolver=resolver, ttl_dns_cache=300, limit=0, limit_per_host=0)
    except Exception:
        connector = aiohttp.TCPConnector(ttl_dns_cache=300, limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(
        total=REST_CONNECT_TIMEOUT + REST_READ_TIMEOUT + 2,
        connect=REST_CONNECT_TIMEOUT, sock_connect=REST_CONNECT_TIMEOUT, sock_read=REST_READ_TIMEOUT
    )

    return aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False)

async def _warmup_history(symbols: List[str], category: str, limit: int = 60):
    """Прогреваем буферы _candles_* историей 4h, чтобы сразу уметь детектить."""
    buf = _candles_linear if category == "linear" else _candles_spot
    async with _make_session() as session:
        for sym in symbols:
            try:
                df = await _fetch_4h_df(session, sym, category, limit=limit)
                if df.empty:
                    continue
                rows = []
                for ts, r in df.iterrows():
                    rows.append({
                        "t": pd.to_datetime(ts, utc=True),
                        "o": float(r["open"]), "h": float(r["high"]),
                        "l": float(r["low"]),  "c": float(r["close"]),
                        "v": float(r.get("volume", 0.0)),
                        "confirm": True  # это закрытые бары
                    })
                buf[sym] = rows[-limit:]  # хранить не больше MAX_CANDLES
            except Exception as e:
                log.warning(f"[WARMUP] {sym}({category}) fail: {e}")
    log.info(f"[WARMUP] {category}: done for {len(symbols)} symbols")

async def _enter_momentum_market(symbol: str, side_raw: str, detect_px: float,
                                 bar_open: pd.Timestamp, bar_close: pd.Timestamp,
                                 strength: float = None) -> bool:
    category = category_of(symbol)
    rr = float(MOM_TAKE_PCT) / max(float(MOM_STOP_PCT), 1e-9)

    entry_at = bar_open + pd.Timedelta(hours=4)
    assert abs((entry_at - bar_close).total_seconds()) < 0.5

    # предварительная постановка (переставим под fill)
    sl_pre, tp_pre = _calc_tpsl_entry(float(detect_px), side_raw, float(MOM_STOP_PCT), rr)
    mode_note = "entry"

    batch_time = pd.Timestamp.utcnow()
    close_deadline = batch_time + pd.Timedelta(hours=MOM_TTL_HOURS)
    usd_wish = float(POSITION_FRACTION) * float(_alloc_total_usd())
    free_cash = _estimate_free_cash_usd(category)

    if free_cash + 1e-9 < usd_wish:
        msg = (f"[MOM_SKIP] {symbol}: free_cash={free_cash:.2f} < alloc={usd_wish:.2f} "
               f"(fraction={POSITION_FRACTION}) — как в eval, отклоняем")
        log.info(msg);
        await tg("⚠️ " + msg)
        _last_done[symbol] = bar_open
        return False

    usd_value = usd_wish
    if usd_value <= 0:
        msg = f"[MOM_SKIP] {symbol}: usd_alloc=0 (нет свободной маржи?)"
        log.info(msg); await tg("⚠️ " + msg); return False

    try:
        await _ensure_margin_and_leverage(symbol, category)
        resp = open_position_market(
            symbol=symbol,
            side=("Buy" if side_raw == "BUY" else "Sell"),
            usd_value=usd_value,
            tp_price=float(tp_pre),
            sl_price=float(sl_pre),
            category=category
        )
        fill_px = None
        try:
            if isinstance(resp, dict):
                fill_px = resp.get("avgPrice") or resp.get("avgFillPrice") or resp.get("price")
                fill_px = float(fill_px) if fill_px not in (None, "") else None
        except Exception:
            fill_px = None

        amended = False
        drift_pct = None
        tp, sl = tp_pre, sl_pre

        if fill_px and AMEND_TPSL_FROM_FILL:
            drift_pct = (fill_px - detect_px) / detect_px
            if abs(drift_pct) > MAX_DRIFT > 0:
                await tg(
                    f"⚠️ {symbol}: fill дрейф {drift_pct * 100:.3f}% > {MAX_DRIFT * 100:.2f}% "
                    f"(detect={detect_px:.6f} fill={fill_px:.6f})"
                )
            # ключ: TP/SL пересчитываем ОТ ФАКТА (fill)
            sl_f, tp_f = _calc_tpsl_entry(float(fill_px), side_raw, float(MOM_STOP_PCT), rr)
            if abs(tp_f - tp_pre) > 0 or abs(sl_f - sl_pre) > 0:
                amended = _amend_tpsl_safe(symbol, tp=tp_f, sl=sl_f, category=category)
                tp, sl = (tp_f, sl_f) if amended else (tp_pre, sl_pre)

        bar_close_utc = entry_at.tz_convert('UTC')
        msg = (
                f"⚡️ MOMENTUM {symbol} {side_raw} ({category})\n"
                f"• bar_open={bar_open.tz_convert('UTC')}  bar_close={bar_close_utc} UTC\n"
                f"• entry_at(=bar_close)={bar_close_utc} UTC\n"
                f"• entry_price(close[T])={detect_px:.6f}\n"
                f"• strength={('%.2f' % strength) if strength is not None else '—'}\n"
                f"• fill≈{(fill_px if fill_px else detect_px):.6f}"
                + (f"  (drift={drift_pct * 100:.3f}%)" if drift_pct is not None else "") + "\n"
                                                                                           f"• TP={tp:.6f}  SL={sl:.6f}  mode={mode_note}"
                + ("  (amended)" if amended else "")
                + f"\n• usd_alloc={usd_value:.2f}  fees≈{FEE_TAKER_LIVE * 100:.2f}% taker"
        )

        log.info("[MOM_MKT] " + msg.replace("\n", " | "))
        await tg(msg)

        from utils.trade_logger import log_trade_open, log_trade_open_adv

        equity_before = _safe_equity_usd()
        free_before = _estimate_free_cash_usd(category)

        oid = f"{symbol}_{int(entry_at.timestamp())}"
        _state.upsert_open(oid, {
            "symbol": symbol, "side": side_raw, "category": category,
            "detect_px": float(detect_px),
            "fill_px": float(fill_px) if fill_px else None,
            "tp": float(tp), "sl": float(sl), "mode": mode_note,
            "bar_open": bar_open.isoformat(), "bar_close": entry_at.isoformat(),
            "placed_at": pd.Timestamp.utcnow().isoformat(),
            "close_deadline": close_deadline.isoformat(),
            "usd_alloc": float(usd_value),
        })

        # === базовый лог (для совместимости/отчётов eval) ===
        try:
            detect_ts_ms = int(bar_open.value // 10 ** 6)
            entry_ts_ms = int(entry_at.value // 10**6)
            entry_fill_px = float(fill_px if fill_px else detect_px)
            fees_bps = float(FEE_TAKER_LIVE) * 1e4
            slip_bps = float(ENTRY_SLIPPAGE_PCT) * 1e4

            log_trade_open(
                "./data/trades/trades.csv",
                {
                    "trade_id": oid,
                    "symbol": symbol,
                    "side": side_raw.lower(),
                    "detect_ts": detect_ts_ms,
                    "entry_ts": entry_ts_ms,
                    "entry_fill": entry_fill_px,
                    "notional_usd": float(usd_value),
                    "rr_tp": float(MOM_TAKE_PCT * 100.0),
                    "rr_sl": float(MOM_STOP_PCT * 100.0),
                    "ttl_hours": int(MOM_TTL_HOURS),
                    "fees_bps": fees_bps,
                    "slip_bps": slip_bps,
                    "spread_bps": 3.0
                }
            )

            # === расширенный диагностический лог ===
            log_trade_open_adv(
                "./data/trades/trades_diagnostics.csv",
                {
                    "trade_id": oid,
                    "symbol": symbol,
                    "side": side_raw.lower(),
                    "category": category,
                    "equity_before": float(equity_before),
                    "free_cash_before": float(free_before),
                    "usd_alloc": float(usd_value),
                    "detect_px": float(detect_px),
                    "fill_px": float(fill_px) if fill_px else float(detect_px),
                    "drift_pct": float(((fill_px - detect_px) / detect_px) if fill_px else 0.0),
                    "tp_pre": float(tp_pre), "sl_pre": float(sl_pre),
                    "tp_final": float(tp), "sl_final": float(sl),
                    "ttl_hours": int(MOM_TTL_HOURS),
                    "fees_bps": fees_bps,
                    "slip_bps": slip_bps,
                    "ts_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }
            )
        except Exception as e:
            log.error(f"[TRADE_LOGGER] {symbol} {side_raw}: {e}")

        _last_done[symbol] = bar_open
        return True
    except Exception as e:
        log.error(f"[MOM_FAIL] {symbol} {side_raw}: {e}")
        return False

async def on_candle_closed(symbol: str, closed_open: Optional[pd.Timestamp] = None):
    """WS confirm=true: детект FVG на только что закрытом баре; вход MARKET."""
    expected_open = pd.to_datetime(closed_open or _ws_last_close_bar.get(symbol), utc=True, errors="coerce")
    if pd.isna(expected_open):
        log.warning(f"[BAR_CLOSE] {symbol}: нет expected_open — пропуск")
        return
    # защита от повторной обработки того же бара
    last = _last_done.get(symbol)
    if last is not None:
        last = pd.to_datetime(last, utc=True, errors="coerce")
        if pd.notna(last) and expected_open <= last:
            log.debug(f"[BAR_CLOSE] {symbol}: уже обработан бар {expected_open} (last_done={last}) — пропуск")
            return

    category = category_of(symbol)
    cbuf = _candles_linear if category == "linear" else _candles_spot
    arr = cbuf.get(symbol, [])
    if len(arr) < 12:
        log.info(f"[BAR_CLOSE] {symbol}: bars={len(arr)} — мало данных")
        return

    df_full = pd.DataFrame([{
        "timestamp": x["t"],
        "open":  x["o"],
        "high":  x["h"],
        "low":   x["l"],
        "close": x["c"],
        "volume":x["v"],
    } for x in arr]).set_index("timestamp").sort_index()

    df = _slice_to_closed_bar(df_full, expected_open)
    if df is None or df.empty or df.index[-1] != expected_open:
        log.info(f"[BAR_CLOSE] {symbol}: REST/WS ещё не докинул бар {expected_open} — пропуск")
        return
    if REST_AFTER_CLOSE_DELAY_SEC > 0: await asyncio.sleep(REST_AFTER_CLOSE_DELAY_SEC)
    expected_open = _last_closed_bar_open_utc()
    df = _slice_to_closed_bar(df, expected_open) or df[df.index <= expected_open]
    args = _pick_and_enter_args(df, symbol)

    if args is None:
        log.info(f"[BAR_CLOSE] {symbol}: сигналов нет (min_strength={DEFAULT_MIN_STRENGTH}, vol×SMA={FVG_VOL_MULT})")
        _last_done[symbol] = expected_open
        return

    bar_open, bar_close, detect_px, side, strength = args

    if not _positions_limit_ok(category):
        log.info(f"[MOM_SKIP] {symbol}: лимит позиций ({MAX_CONCURRENT})"); _last_done[symbol] = bar_open; return
    try:
        plist = (get_positions(category=category).get("result", {}) or {}).get("list", []) or []
        if any(p.get("symbol") == symbol and abs(float(p.get("size") or 0.0)) > 0 for p in plist):
            log.info(f"[MOM_SKIP] {symbol}: уже есть открытая позиция")
            _last_done[symbol] = bar_open
            return
    except Exception:
        pass

    log.debug(
        f"[ENTRY_PICK] {symbol} bar_open={bar_open.tz_convert('UTC')}  bar_close={(bar_open+pd.Timedelta(hours=4)).tz_convert('UTC')} "
        f"entry_price(close[T])={detect_px:.6f} side={side} strength={strength:.4f}"
    )

    await _enter_momentum_market(symbol, side, detect_px, bar_open, bar_close, strength=strength)

# ====== WS по категориям ======
async def price_stream_loop(symbols: List[str], category: str):
    url = _ws_url(category)
    CHUNK = 25
    topics_all = [f"kline.{FVG_INTERVAL}.{sym}" for sym in symbols]
    log.info(f"[ws] connecting {url} | {category} | symbols={len(symbols)}")

    cbuf = _candles_linear if category == "linear" else _candles_spot

    backoff = 1.0
    while True:
        try:
            async with _make_session() as session:
                async with session.ws_connect(url, heartbeat=20, max_msg_size=2**22) as ws:
                    total = (len(topics_all) + CHUNK - 1) // CHUNK
                    for i in range(0, len(topics_all), CHUNK):
                        chunk = topics_all[i:i+CHUNK]
                        await ws.send_json({"op": "subscribe", "args": chunk})
                        log.info(f"[ws] subscribe chunk {i//CHUNK+1}/{total} size={len(chunk)}")
                        await asyncio.sleep(0.1)
                    log.info("[ws] subscribed")
                    backoff = 1.0

                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                                break
                            continue

                        data = json.loads(msg.data)
                        if data.get("op") == "subscribe" and "success" in data:
                            continue

                        topic = data.get("topic", "")
                        if not topic.startswith("kline."):
                            continue

                        bars = data.get("data", [])
                        for bar in bars:
                            sym = bar.get("symbol")
                            if not sym:
                                continue
                            item = {
                                "t": _parse_ts_to_utc(bar["start"]),
                                "o": float(bar["open"]),
                                "h": float(bar["high"]),
                                "l": float(bar["low"]),
                                "c": float(bar["close"]),
                                "v": float(bar.get("volume", 0.0)),
                                "confirm": bool(bar.get("confirm", False)),
                            }
                            arr = cbuf.setdefault(sym, [])
                            if arr and arr[-1]["t"] == item["t"]:
                                arr[-1] = item
                            else:
                                arr.append(item)
                                if len(arr) > MAX_CANDLES:
                                    arr.pop(0)

                            if item["confirm"]:
                                _ws_last_close_bar[sym] = item["t"]
                                log.info(
                                    f"[BAR_CLOSE] {sym} ({category}) close={item['c']:.6f} "
                                    f"t_open={_fmt_ts(item['t'])} t_close={_fmt_ts(item['t'] + pd.Timedelta(hours=4))}"
                                )
                                await on_candle_closed(sym, closed_open=item["t"])

        except asyncio.CancelledError:
            log.info("[ws] cancelled — выходим из price_stream_loop()")
            return
        except aiohttp.ClientConnectorDNSError as e:
            log.error(f"[DNS] resolve failed for {url}: {e}")
            await asyncio.sleep(backoff + random.uniform(0, backoff / 2))
            backoff = min(backoff * 2, 60)
        except Exception:
            log.exception("[ws] error, reconnect with backoff]")
            await asyncio.sleep(backoff + random.uniform(0, backoff / 2))
            backoff = min(backoff * 2, 60)

# ===== вспомогалка: срез до нужного закрытого бара =====
def _slice_to_closed_bar(df: pd.DataFrame, expected_open: pd.Timestamp) -> Optional[pd.DataFrame]:
    if df is None or df.empty: return None
    df = df.sort_index()
    if expected_open in df.index: return df.loc[:expected_open]
    if df.index[0] <= expected_open <= df.index[-1]: return df[df.index <= expected_open]
    return None

def _estimate_free_cash_usd(category: str) -> float:
    """
    Оценка free cash как equity (USDT) минус сумма notional по открытым позициям.
    Это повторяет идею 'free_cash' из eval-симуляции.
    """
    try:
        equity = float(get_wallet_balance("USDT"))
    except Exception:
        equity = float(INITIAL_CAPITAL or 0.0)

    try:
        pp = get_positions(category=category)
        plist = (pp.get("result", {}) or {}).get("list", []) or []
    except Exception:
        plist = []

    notional_sum = 0.0
    for p in plist:
        try:
            sz = abs(float(p.get("size") or 0.0))
            last = float(p.get("markPrice") or p.get("entryPrice") or 0.0)
            notional_sum += sz * last
        except Exception:
            pass

    free_cash = equity - notional_sum
    return max(free_cash, 0.0)

# ====== REST ======
async def _fetch_4h_df(session: aiohttp.ClientSession, symbol: str, category: str, limit: int = 50) -> pd.DataFrame:
    bases = [http_base()]
    if "api.bybit.com" in bases[0]:
        bases.append(bases[0].replace("https://api.bybit.com", "https://api.bytick.com"))

    async def _one(cat: str, base_url: str):
        url = base_url.rstrip("/") + "/v5/market/kline"
        params = {"category": cat, "symbol": symbol, "interval": FVG_INTERVAL, "limit": str(int(limit))}
        for attempt in range(1, REST_ATTEMPTS + 1):
            try:
                async with session.get(url, params=params) as r:
                    js = await r.json(content_type=None)
                code = int(js.get("retCode", -1))
                if code == 0:
                    rows = (js.get("result", {}) or {}).get("list", []) or []
                    if not rows: return pd.DataFrame()
                    rows.sort(key=lambda x: int(x[0]))
                    rec = []
                    for s, o, h, l, c, vol, turnover in rows:
                        ts = _parse_ts_to_utc(s)
                        rec.append({"timestamp": ts, "open": float(o), "high": float(h),
                                    "low": float(l), "close": float(c), "volume": float(vol)})
                    return pd.DataFrame(rec).set_index("timestamp")
                if code == 10006:
                    await asyncio.sleep(1.0 * attempt); continue
                raise RuntimeError(f"retCode={code} msg={js.get('retMsg')}")
            except asyncio.TimeoutError:
                global _net_batch_timeouts; _net_batch_timeouts += 1
                log.warning(f"[REST_TIMEOUT] {symbol} ({cat}) try={attempt}/{REST_ATTEMPTS} base={base_url}")
                if attempt < REST_ATTEMPTS: await asyncio.sleep(0.5 * attempt)
                continue
            except aiohttp.ClientConnectorError as e:
                global _net_batch_errors; _net_batch_errors += 1
                log.warning(f"[REST_CONNECT] {symbol} ({cat}) try={attempt}/{REST_ATTEMPTS} base={base_url} err={e}")
                if attempt < REST_ATTEMPTS: await asyncio.sleep(0.5 * attempt)
                continue
            except Exception as e:
                _net_batch_errors += 1
                if attempt < REST_ATTEMPTS:
                    log.warning(f"[REST_RETRY] {symbol} ({cat}) try={attempt}/{REST_ATTEMPTS} base={base_url} err={e}")
                    await asyncio.sleep(1.0 * attempt); continue
                log.error(f"[REST_KLINE_ERR] {symbol} ({cat}) base={base_url}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    df = pd.DataFrame()
    for base in bases:
        df = await _one(category, base)
        if not df.empty: break

    if df.empty and category == "linear" and REST_ALLOW_SPOT_FALLBACK:
        for base in bases:
            df = await _one("spot", base)
            if not df.empty:
                log.info(f"[REST_FALLBACK] {symbol}: linear пусто → spot (base={base})")
                break
    return df

# ====== Сервисные лупы ======
async def timeout_closer_loop():
    """Жёсткий TTL: по MOMENTUM_TTL_HOURS закрываем MARKET (страховка)."""
    while True:
        try:
            _reload_env_if_needed()
            now = pd.Timestamp.utcnow()
            open_map = _state.all_open()
            for oid, p in list(open_map.items()):
                try:
                    dl = pd.to_datetime(p.get("close_deadline"))
                    if pd.isna(dl): continue
                    if now >= dl:
                        _ = close_position_market(symbol=p["symbol"], category=p.get("category", BYBIT_CATEGORY))
                        _state.pop_open(oid)
                        log.info(f"[TIMEOUT CLOSE] {p['symbol']} oid={oid}")
                        await tg(f"⏱️ TIMEOUT CLOSE {p['symbol']} (oid={oid})")
                except Exception as e:
                    log.error(f"[timeout] close error {oid}: {e}")
        except Exception:
            log.exception("[timeout] loop err")
        await asyncio.sleep(60)

async def balance_sync_loop():
    """Периодически подтягиваем баланс и обновляем аллокатор (реинвест)."""
    _BALANCE_SYNC_MIN = int(os.getenv("BALANCE_SYNC_MIN", "10"))
    while True:
        try:
            _reload_env_if_needed()
            bal = get_wallet_balance("USDT")
            if bal > 0:
                try: _allocator.set_total(bal)
                except Exception: pass
                log.info(f"[BALANCE] sync → {bal:.2f} USDT")
        except Exception:
            log.exception("[BALANCE] sync failed")
        await asyncio.sleep(max(60, _BALANCE_SYNC_MIN*60))

async def monitor_loop():
    """Сводка каждые 10 мин."""
    last = 0.0
    while True:
        try:
            _reload_env_if_needed()
            if time.time() - last > 600:
                close_ts, left = _next_4h_close_utc()
                msg = (
                    f"📊 MOMENTUM live\n"
                    f"• TP={MOM_TAKE_PCT*100:.2f}%  SL={MOM_STOP_PCT*100:.2f}%  TTL={MOM_TTL_HOURS}h\n"
                    f"• FVG: min_strength={DEFAULT_MIN_STRENGTH}  vol×SMA={FVG_VOL_MULT}  tol={FVG_TOLERANCE_PCT}\n"
                    f"• MAX_CONCURRENT={MAX_CONCURRENT}  USE_WS={USE_WS}\n"
                    f"• До закрытия 4h: {str(left)}"
                )
                log.info(msg.replace("\n"," | "))
                await tg(msg)
                last = time.time()
        except Exception:
            log.exception("[MON] err")
        await asyncio.sleep(10)

# ====== entrypoint ======
async def main():
    global _MAIN_LOOP, _STOP_EVENT

    _MAIN_LOOP = asyncio.get_running_loop()
    _STOP_EVENT = asyncio.Event()
    # ==== Полный сброс состояния по желанию ====
    if str(os.getenv("RESET_ON_START", "0")).lower() in ("1","true","yes","y","on"):
        try:
            # 2) оперативные кэши
            _last_done.clear()
            _ws_last_close_bar.clear()
            _candles_linear.clear()
            _candles_spot.clear()
            log.info("[RESET] state and in-memory caches cleared on start")
        except Exception:
            log.exception("[RESET] failed")

    # первичный баланс
    try:
        bal = get_wallet_balance("USDT")
        if bal > 0:
            try: _allocator.set_total(bal)
            except Exception: pass
        log.info(f"[BALANCE] init={bal:.2f} USDT")
    except Exception as e:
        log.warning(f"[BALANCE] init sync failed: {e}")

    tg_updater = _start_telegram_control()

    # символы по категориям
    syms_linear = [s for s, cat in SYMBOL_CATEGORY.items() if cat == "linear"]
    syms_spot   = [s for s, cat in SYMBOL_CATEGORY.items() if cat == "spot"]

    if not syms_linear and not syms_spot:
        syms_env = os.getenv("TRADE_UNIVERSE", ",".join(TRADE_UNIVERSE) if isinstance(TRADE_UNIVERSE, (list,tuple)) else str(TRADE_UNIVERSE))
        symbols = [s.strip().upper() for s in syms_env.split(",") if s.strip()]
        if not symbols:
            try: symbols = fetch_top_symbols()[:50]
            except Exception: symbols = []
        if not symbols:
            log.error("TRADE_UNIVERSE пуст — проверь config.py/ENV"); return
        syms_linear = symbols  # всё в linear по умолчанию

    if syms_linear:
        await _warmup_history(syms_linear, "linear", limit=min(MAX_CANDLES, 102))
    if syms_spot:
        await _warmup_history(syms_spot, "spot", limit=min(MAX_CANDLES, 102))
    await rest_catchup_once()
    if _bot and CHAT_ID:
        await tg("🚀 MOMENTUM live: TP/SL от fill; TTL в часах; WS+REST; per-symbol category.")

    tasks = [
        timeout_closer_loop(),
        balance_sync_loop(),
        monitor_loop(),
        rest_close_scanner_loop(),  # <<< добавили REST-сканер по закрытию
    ]
    if USE_WS:
        if syms_linear:
            tasks.append(price_stream_loop(syms_linear, category="linear"))
        if syms_spot:
            tasks.append(price_stream_loop(syms_spot, category="spot"))

    runner = asyncio.gather(*tasks, return_exceptions=True)
    stopper = asyncio.create_task(_STOP_EVENT.wait()) if _STOP_EVENT else None

    done, pending = await asyncio.wait(
        {runner, stopper} if stopper else {runner},
        return_when=asyncio.FIRST_COMPLETED
    )

    if stopper and stopper in done:
        log.info("[MAIN] stop signal received, cancel tasks…")
        if tg_updater:
            try:
                tg_updater.stop()
                log.info("[TG] control stopped")
            except Exception:
                log.exception("[TG] stop failed")
        runner.cancel()
        try:
            await runner
        except Exception:
            pass
        return

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass