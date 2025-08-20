# autotrade_realtime.py
# ===== START COMMAND =====
# ENV:
#   ENTRY_MODE=RETEST | BREAKOUT | MOMENTUM
#   ENTRY_OFFSET_PCT=0.02         # для BREAKOUT (информативный таргет)
#   MOMENTUM_ATR_N=14
#   MOMENTUM_BODY_ATR=1.5
#   MOMENTUM_RANGE_ATR=2.0
#   MOMENTUM_VOL_SMA=2.0
#   MOMENTUM_TP_PCT=0.02          # 2% TP
#   MOMENTUM_SL_PCT=0.01          # 1% SL
#   ENTRY_TOUCH_LTF=5,1           # проверка касания entry на LTF (минуты)
#   ENTRY_BACKFILL_LOOKBACK_DAYS=7
#
# Запуск:
#   python autotrade_realtime.py > logs/auto.log 2>&1 &
#   tail -f logs/auto.log
# =========================

import os
import time
from typing import Dict, List, Optional, Tuple
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # горячая подгрузка .env отключится, если пакета нет

import json
import asyncio
import logging
import pandas as pd
import aiohttp
from telegram import Bot  # python-telegram-bot v13

from utils.detect_fvg import detect_fvg_imbalances
from utils.strategy import select_entry_price
from utils.bybit_trade import (
    open_position_market, close_position_market, get_wallet_balance,
    get_open_orders, get_positions, entry_was_touched_ltf
)
from utils.state_store import JsonState
from utils.allocator import SmartAllocator
from utils.symbols import fetch_top_symbols  # список тикеров для подписки

from config import (
    BYBIT_TESTNET, BYBIT_CATEGORY,
    TELEGRAM_TOKEN, CHAT_ID,
    INITIAL_CAPITAL, UNIVERSE_SIZE,
    DEFAULT_TTL_DAYS, DEFAULT_MIN_STRENGTH,
    ENABLE_BUY, ENABLE_SELL,
    RISK_PCT, RISK_REWARD_RATIO,
    TRADE_UNIVERSE,
    USE_MAINNET_MARKET_DATA,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("realtime")

# ---------- Runtime-config (горячая подгрузка из .env) ----------
_LIMIT_ORDER_MODE = os.getenv("LIMIT_ORDER_MODE", "GTC").upper()               # GTC | POST_ONLY
_LIMIT_POSTONLY_FALLBACK = os.getenv("LIMIT_POSTONLY_FALLBACK", "GTC").upper()
_BACKFILL_4H_BARS = int(os.getenv("BACKFILL_4H_BARS", "24"))
_BALANCE_SYNC_MIN = int(os.getenv("BALANCE_SYNC_MIN", "10"))
_POLL_ORDERS_SEC  = int(os.getenv("POLL_ORDERS_SEC", "15"))
_ENV_RELOAD_SEC   = int(os.getenv("ENV_RELOAD_SEC", "10"))

# === Переключатели стратегии входа ===
_ENTRY_MODE = os.getenv("ENTRY_MODE", "RETEST").upper()          # RETEST | BREAKOUT | MOMENTUM
_ENTRY_OFFSET_PCT = float(os.getenv("ENTRY_OFFSET_PCT", "0.02")) # 2% (для BREAKOUT; только в лог как ориентир)

# === Параметры MOMENTUM ===
_MOM_ATR_N     = int(os.getenv("MOMENTUM_ATR_N", "14"))
_MOM_BODY_ATR  = float(os.getenv("MOMENTUM_BODY_ATR", "1.5"))
_MOM_RANGE_ATR = float(os.getenv("MOMENTUM_RANGE_ATR", "2.0"))
_MOM_VOL_SMA   = float(os.getenv("MOMENTUM_VOL_SMA", "2.0"))
_MOM_TP_PCT    = float(os.getenv("MOMENTUM_TP_PCT", "0.02"))  # 2%
_MOM_SL_PCT    = float(os.getenv("MOMENTUM_SL_PCT", "0.01"))  # 1%

# === Новое: LTF проверка и датовый бэкфилл ===
_ENTRY_TOUCH_LTF = [s.strip() for s in os.getenv("ENTRY_TOUCH_LTF", "5,1").split(",") if s.strip()]
_ENTRY_BACKFILL_LOOKBACK_DAYS = int(os.getenv("ENTRY_BACKFILL_LOOKBACK_DAYS", "7"))

_LAST_ENV_READ_TS = 0.0

_BYBIT_DEBUG = os.getenv("BYBIT_DEBUG", "0").lower() in ("1", "true", "yes")

def _next_4h_close_utc(now: Optional[pd.Timestamp] = None) -> Tuple[pd.Timestamp, pd.Timedelta]:
    """Ближайшее закрытие 4h-бара (UTC) и оставшееся время."""
    now = now or pd.Timestamp.utcnow()
    # 4h бары закрываются в 00, 04, 08, 12, 16, 20 UTC
    hour = (now.hour // 4) * 4
    this_close = now.replace(hour=hour, minute=0, second=0, microsecond=0) + pd.Timedelta(hours=4)
    if this_close <= now:
        this_close += pd.Timedelta(hours=4)
    return this_close, this_close - now

def _timedelta_str(td: pd.Timedelta) -> str:
    total = int(td.total_seconds())
    if total < 0: total = 0
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _reload_env_if_needed():
    """Периодически перечитываем .env и обновляем рантайм-настройки без рестарта. Логируем ETA до закрытия 4h бара."""
    global _LAST_ENV_READ_TS, _LIMIT_ORDER_MODE, _LIMIT_POSTONLY_FALLBACK
    global _BACKFILL_4H_BARS, _BALANCE_SYNC_MIN, _POLL_ORDERS_SEC, _ENV_RELOAD_SEC
    global _ENTRY_MODE, _ENTRY_OFFSET_PCT
    global _MOM_ATR_N, _MOM_BODY_ATR, _MOM_RANGE_ATR, _MOM_VOL_SMA, _MOM_TP_PCT, _MOM_SL_PCT
    global _ENTRY_TOUCH_LTF, _ENTRY_BACKFILL_LOOKBACK_DAYS

    now = time.time()
    if _ENV_RELOAD_SEC <= 0 or (now - _LAST_ENV_READ_TS) < _ENV_RELOAD_SEC:
        return
    if load_dotenv is None:
        _LAST_ENV_READ_TS = now
        close_ts, left = _next_4h_close_utc()
        log.info(
            f"[ENV] reloaded: LIMIT_ORDER_MODE={_LIMIT_ORDER_MODE}, POSTONLY_FALLBACK={_LIMIT_POSTONLY_FALLBACK}, "
            f"BACKFILL_4H_BARS={_BACKFILL_4H_BARS}, POLL_ORDERS_SEC={_POLL_ORDERS_SEC}, "
            f"BALANCE_SYNC_MIN={_BALANCE_SYNC_MIN}, ENTRY_MODE={_ENTRY_MODE}, ENTRY_OFFSET_PCT={_ENTRY_OFFSET_PCT}, "
            f"MOM[n={_MOM_ATR_N}, body×ATR={_MOM_BODY_ATR}, range×ATR={_MOM_RANGE_ATR}, vol×SMA={_MOM_VOL_SMA}, "
            f"tp={_MOM_TP_PCT*100:.1f}%, sl={_MOM_SL_PCT*100:.1f}%], "
            f"ENTRY_TOUCH_LTF={_ENTRY_TOUCH_LTF}, BACKFILL_LOOKBACK_DAYS={_ENTRY_BACKFILL_LOOKBACK_DAYS}, "
            f"NEXT_4H_CLOSE_IN={_timedelta_str(left)}"
        )
        return
    try:
        load_dotenv(override=True)
        _LIMIT_ORDER_MODE        = os.getenv("LIMIT_ORDER_MODE", _LIMIT_ORDER_MODE).upper()
        _LIMIT_POSTONLY_FALLBACK = os.getenv("LIMIT_POSTONLY_FALLBACK", _LIMIT_POSTONLY_FALLBACK).upper()
        _BACKFILL_4H_BARS        = int(os.getenv("BACKFILL_4H_BARS", str(_BACKFILL_4H_BARS)))
        _BALANCE_SYNC_MIN        = int(os.getenv("BALANCE_SYNC_MIN", str(_BALANCE_SYNC_MIN)))
        _POLL_ORDERS_SEC         = int(os.getenv("POLL_ORDERS_SEC", str(_POLL_ORDERS_SEC)))
        _ENV_RELOAD_SEC          = int(os.getenv("ENV_RELOAD_SEC", str(_ENV_RELOAD_SEC)))
        _ENTRY_MODE              = os.getenv("ENTRY_MODE", _ENTRY_MODE).upper()
        _ENTRY_OFFSET_PCT        = float(os.getenv("ENTRY_OFFSET_PCT", str(_ENTRY_OFFSET_PCT)))
        # momentum
        _MOM_ATR_N               = int(os.getenv("MOMENTUM_ATR_N",   str(_MOM_ATR_N)))
        _MOM_BODY_ATR            = float(os.getenv("MOMENTUM_BODY_ATR", str(_MOM_BODY_ATR)))
        _MOM_RANGE_ATR           = float(os.getenv("MOMENTUM_RANGE_ATR", str(_MOM_RANGE_ATR)))
        _MOM_VOL_SMA             = float(os.getenv("MOMENTUM_VOL_SMA", str(_MOM_VOL_SMA)))
        _MOM_TP_PCT              = float(os.getenv("MOMENTUM_TP_PCT",  str(_MOM_TP_PCT)))
        _MOM_SL_PCT              = float(os.getenv("MOMENTUM_SL_PCT",  str(_MOM_SL_PCT)))
        # new
        _ENTRY_TOUCH_LTF         = [s.strip() for s in os.getenv("ENTRY_TOUCH_LTF", ",".join(_ENTRY_TOUCH_LTF)).split(",") if s.strip()]
        _ENTRY_BACKFILL_LOOKBACK_DAYS = int(os.getenv("ENTRY_BACKFILL_LOOKBACK_DAYS", str(_ENTRY_BACKFILL_LOOKBACK_DAYS)))

        _LAST_ENV_READ_TS = now
        close_ts, left = _next_4h_close_utc()
        log.info(
            f"[ENV] reloaded: LIMIT_ORDER_MODE={_LIMIT_ORDER_MODE}, POSTONLY_FALLBACK={_LIMIT_POSTONLY_FALLBACK}, "
            f"BACKFILL_4H_BARS={_BACKFILL_4H_BARS}, POLL_ORDERS_SEC={_POLL_ORDERS_SEC}, "
            f"BALANCE_SYNC_MIN={_BALANCE_SYNC_MIN}, ENTRY_MODE={_ENTRY_MODE}, ENTRY_OFFSET_PCT={_ENTRY_OFFSET_PCT}, "
            f"MOM[n={_MOM_ATR_N}, body×ATR={_MOM_BODY_ATR}, range×ATR={_MOM_RANGE_ATR}, vol×SMA={_MOM_VOL_SMA}, "
            f"tp={_MOM_TP_PCT*100:.1f}%, sl={_MOM_SL_PCT*100:.1f}%], "
            f"ENTRY_TOUCH_LTF={_ENTRY_TOUCH_LTF}, BACKFILL_LOOKBACK_DAYS={_ENTRY_BACKFILL_LOOKBACK_DAYS}, "
            f"NEXT_4H_CLOSE_IN={_timedelta_str(left)}"
        )
    except Exception as e:
        _LAST_ENV_READ_TS = now
        log.warning(f"[ENV] reload failed: {e}")

# WS endpoints v5
WS_MAIN_SPOT   = "wss://stream.bybit.com/v5/public/spot"
WS_TEST_SPOT   = "wss://stream-testnet.bybit.com/v5/public/spot"
WS_MAIN_LINEAR = "wss://stream.bybit.com/v5/public/linear"
WS_TEST_LINEAR = "wss://stream-testnet.bybit.com/v5/public/linear"

def _ws_url() -> str:
    """Публичный WS берём там же, где и маркет-данные (mainnet/testnet)."""
    use_mainnet = bool(USE_MAINNET_MARKET_DATA)
    if BYBIT_CATEGORY == "spot":
        return WS_MAIN_SPOT if use_mainnet else WS_TEST_SPOT
    return WS_MAIN_LINEAR if use_mainnet else WS_TEST_LINEAR

# HTTP endpoints (public)
HTTP_BASE_MAIN = "https://api.bybit.com"
HTTP_BASE_TEST = "https://api-testnet.bybit.com"
def http_base() -> str:
    """Публичный HTTP (свечи) берём по USE_MAINNET_MARKET_DATA, чтобы совпадало с отчётами."""
    return HTTP_BASE_MAIN if USE_MAINNET_MARKET_DATA else HTTP_BASE_TEST

# ===================== GLOBAL STATE =====================
STATE_PATH = os.getenv("AUTOTRADE_STATE_PATH", "autotrade_state.json")
PENDING_STATE_PATH = os.getenv("PENDING_STATE_PATH", "pending_orders_state.json")

_state      = JsonState(STATE_PATH)
_pending    = JsonState(PENDING_STATE_PATH)   # хранит висящие лимитки и их expiry (+ usd_alloc)

_allocator  = SmartAllocator(INITIAL_CAPITAL)

# Свечи в памяти
_candles: Dict[str, List[dict]] = {}
MAX_CANDLES = 60
FVG_INTERVAL = "240"  # 4h

# Telegram
_bot: Optional[Bot] = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
async def tg(text: str):
    if not _bot or not CHAT_ID:
        return
    chat = int(CHAT_ID) if str(CHAT_ID).lstrip("-").isdigit() else CHAT_ID
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _bot.send_message(chat_id=chat, text=text))

# ===================== Вспомогалки =====================
def _calc_sl_tp(entry: float, side: str) -> Tuple[float, float]:
    """
    SL/TP:
      SELL: SL = entry * (1 + RISK_PCT%), TP = entry - (SL-entry)*RISK_REWARD_RATIO
      BUY : SL = entry * (1 - RISK_PCT%), TP = entry + (entry-SL)*RISK_REWARD_RATIO
    """
    risk_k = float(RISK_PCT) / 100.0
    rr     = float(RISK_REWARD_RATIO)
    if side == "SELL":
        sl = entry * (1.0 + risk_k)
        tp = entry - (sl - entry) * rr
    else:
        sl = entry * (1.0 - risk_k)
        tp = entry + (entry - sl) * rr
    return float(sl), float(tp)

def _limit_kwargs():
    """Параметры TIF/пост-онли для create_order."""
    if _LIMIT_ORDER_MODE == "POST_ONLY":
        return {"timeInForce": "PostOnly"}
    return {"timeInForce": "GTC"}

def _parse_ts_to_utc(ts_val):
    """Парсим timestamp (сек/мс/iso) в pandas.Timestamp(UTC)."""
    try:
        v = int(ts_val)
        if v > 10**12:
            return pd.to_datetime(v, unit="ms", utc=True)
        return pd.to_datetime(v, unit="s", utc=True)
    except Exception:
        try:
            return pd.to_datetime(ts_val, utc=True)
        except Exception:
            return pd.Timestamp.utcnow()

def _kline_item_to_mem(item) -> dict:
    if isinstance(item, dict):
        start = _parse_ts_to_utc(item.get("start"))
        return {
            "t": start,
            "o": float(item.get("open", 0)),
            "h": float(item.get("high", 0)),
            "l": float(item.get("low", 0)),
            "c": float(item.get("close", 0)),
            "v": float(item.get("volume", 0.0)),
            "confirm": bool(item.get("confirm", True))
        }
    elif isinstance(item, (list, tuple)) and len(item) >= 6:
        start = _parse_ts_to_utc(item[0])
        return {
            "t": start,
            "o": float(item[1]),
            "h": float(item[2]),
            "l": float(item[3]),
            "c": float(item[4]),
            "v": float(item[5]),
            "confirm": True
        }
    else:
        return {"t": pd.Timestamp.utcnow(), "o":0.0,"h":0.0,"l":0.0,"c":0.0,"v":0.0,"confirm": False}

async def _place_limit_order(symbol: str, side_raw: str, entry: float, tp: float, sl: float, oid: str, backfill: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Постановка лимитки с учётом аллокатора. Резервируем кэш сразу.
    Возвращает (ok, reason_if_failed).
    """
    from utils.bybit_trade import create_order, usd_to_qty

    # 1) запросить аллокацию (резерв)
    batch_time = pd.Timestamp.utcnow()
    close_deadline = batch_time + pd.Timedelta(days=DEFAULT_TTL_DAYS)
    usd_list = _allocator.allocate_for_batch(
        batch_time=batch_time.to_pydatetime(),
        batch_entries=[{"symbol": symbol, "fill_time": batch_time.to_pydatetime(), "close_time": close_deadline.to_pydatetime()}],
        future_entries=[],
    )
    usd = float(usd_list[0]) if usd_list else 0.0
    if usd <= 0:
        return False, "alloc=0 (нет свободного кеша)"

    try:
        qty = usd_to_qty(symbol, usd, price=entry, category=BYBIT_CATEGORY)
        kwargs = _limit_kwargs()
        _ = create_order(
            symbol=symbol,
            side=("Buy" if side_raw == "BUY" else "Sell"),
            order_type="Limit",
            qty=str(qty),
            price=str(entry),
            take_profit=str(tp),
            stop_loss=str(sl),
            reduce_only=False,
            order_link_id=oid,
            category=BYBIT_CATEGORY,
            **kwargs
        )
        ttl_hours = DEFAULT_TTL_DAYS * 24 - 1
        expire_at = (pd.Timestamp.utcnow() + pd.Timedelta(hours=ttl_hours)).isoformat()
        meta = {
            "symbol": symbol, "side": side_raw, "qty": qty, "price": entry,
            "tp": tp, "sl": sl,
            "placed_at": pd.Timestamp.utcnow().isoformat(),
            "expire_at": expire_at,
            "usd_alloc": usd,  # важно для возврата при отмене
            "close_deadline": close_deadline.isoformat(),
        }
        if backfill:
            meta["backfill"] = True
        _pending.upsert(oid, meta)
        log.info(f"[LIMIT] placed {symbol} {side_raw} qty={qty} @ {entry} (tp={tp}, sl={sl}) tif={kwargs.get('timeInForce')}{' [backfill]' if backfill else ''} usd_alloc={usd:.2f}")
        return True, None
    except Exception as e:
        # если API не принял — вернём резерв обратно
        _allocator.close_one(pd.Timestamp.utcnow().to_pydatetime(), usd_alloc=usd, pnl_pct=0.0)
        log.error(f"[LIMIT_FAIL] {symbol} {side_raw} @ {entry}: {e}")
        return False, f"api_error: {e}"

# ===================== Подкачка истории (REST) =====================
async def fetch_recent_klines_for_symbol(session: aiohttp.ClientSession, symbol: str, limit: int) -> List[dict]:
    url = http_base() + "/v5/market/kline"
    params = {"category": BYBIT_CATEGORY, "symbol": symbol, "interval": FVG_INTERVAL, "limit": str(limit)}
    try:
        async with session.get(url, params=params, timeout=15) as resp:
            resp.raise_for_status()
            data = await resp.json()
            if int(data.get("retCode", -1)) != 0:
                raise RuntimeError(f"retCode={data.get('retCode')} retMsg={data.get('retMsg')}")
            rows = (data.get("result") or {}).get("list") or []
            bars = [_kline_item_to_mem(x) for x in rows]
            bars.sort(key=lambda x: x["t"])
            log.info(f"[REST_KLINE] {symbol}: fetched={len(bars)} limit={limit}")
            return bars
    except Exception as e:
        log.error(f"[REST_KLINE_FAIL] {symbol}: {e}")
        return []

async def preload_history(symbols: List[str], limit: int, concurrency: int = 8):
    if limit <= 0:
        log.info("[PRELOAD] skip: limit<=0")
        return
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def _one(sym: str):
            async with sem:
                bars = await fetch_recent_klines_for_symbol(session, sym, limit)
                if not bars:
                    log.info(f"[PRELOAD] {sym}: no bars fetched")
                    return
                mem = _candles.setdefault(sym, [])
                mem.clear()
                mem.extend(bars[-MAX_CANDLES:])
                log.info(f"[PRELOAD] {sym}: stored bars={len(mem)} (MAX={MAX_CANDLES}) last_t={mem[-1]['t'] if mem else '—'}")
        await asyncio.gather(*[_one(s) for s in symbols])

# ===================== MOMENTUM helpers =====================
def _atr(series_h, series_l, series_c, n=14):
    hl = (series_h - series_l).abs()
    hc = (series_h - series_c.shift(1)).abs()
    lc = (series_l - series_c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

async def _enter_momentum_market(symbol: str, side_raw: str, px: float, t0: pd.Timestamp):
    # аллокация
    batch_time = pd.Timestamp.utcnow()
    close_deadline = batch_time + pd.Timedelta(days=DEFAULT_TTL_DAYS)
    usd_list = _allocator.allocate_for_batch(
        batch_time=batch_time.to_pydatetime(),
        batch_entries=[{"symbol": symbol, "fill_time": batch_time.to_pydatetime(), "close_time": close_deadline.to_pydatetime()}],
        future_entries=[],
    )
    usd = float(usd_list[0]) if usd_list else 0.0
    if usd <= 0:
        log.info(f"[MOM_SKIP] {symbol} {side_raw}: alloc=0")
        return False
    entry = float(px)
    # TP/SL фикс-проценты от цены входа:
    tp = entry * (1.0 + _MOM_TP_PCT) if side_raw == "BUY" else entry * (1.0 - _MOM_TP_PCT)
    sl = entry * (1.0 - _MOM_SL_PCT) if side_raw == "BUY" else entry * (1.0 + _MOM_SL_PCT)
    try:
        open_position_market(
            symbol=symbol,
            side=("Buy" if side_raw == "BUY" else "Sell"),
            usd_value=usd,
            tp_price=tp,
            sl_price=sl,
            category=BYBIT_CATEGORY
        )
        await tg(f"⚡️ MOMENTUM {symbol} {side_raw} — MARKET ~{entry:.6f} tp={tp:.6f} sl={sl:.6f}")
        log.info(f"[MOM_MKT] {symbol} {side_raw} usd={usd:.2f} entry≈{entry:.6f} tp={tp:.6f} sl={sl:.6f} t0={t0}")
        return True
    except Exception as e:
        _allocator.close_one(pd.Timestamp.utcnow().to_pydatetime(), usd_alloc=usd, pnl_pct=0.0)
        log.error(f"[MOM_FAIL] {symbol} {side_raw}: {e}")
        return False

# ===================== BREAKOUT helper =====================
async def _enter_breakout_market(symbol: str, side_raw: str, c_close: float, imb_time: pd.Timestamp):
    """
    BREAKOUT: открываемся MARKET сразу на закрытии бара с имбом.
    SL/TP считаем от фактической «entry≈close». ENTRY_OFFSET_PCT логируем как информативный таргет (но не ждём).
    """
    # аллокация «на сейчас»
    batch_time = pd.Timestamp.utcnow()
    close_deadline = batch_time + pd.Timedelta(days=DEFAULT_TTL_DAYS)
    usd_list = _allocator.allocate_for_batch(
        batch_time=batch_time.to_pydatetime(),
        batch_entries=[{"symbol": symbol, "fill_time": batch_time.to_pydatetime(), "close_time": close_deadline.to_pydatetime()}],
        future_entries=[],
    )
    usd = float(usd_list[0]) if usd_list else 0.0
    if usd <= 0:
        log.info(f"[BRK_SKIP] {symbol} {side_raw}: alloc=0 — пропуск MARKET входа")
        return False

    # теоретический «таргет-ентри» (для лога)
    theor_entry = c_close * (1.0 + _ENTRY_OFFSET_PCT) if side_raw == "BUY" else c_close * (1.0 - _ENTRY_OFFSET_PCT)
    # SL/TP считаем от текущего close (как прокси цены входа)
    sl, tp = _calc_sl_tp(c_close, side_raw)

    try:
        # открываем MARKET (твоя утилита сама рассчитает qty по USD)
        _ = open_position_market(
            symbol=symbol,
            side=("Buy" if side_raw == "BUY" else "Sell"),
            usd_value=usd,
            tp_price=tp,
            sl_price=sl,
            category=BYBIT_CATEGORY
        )
        log.info(f"[BRK_MKT] {symbol} {side_raw} usd={usd:.2f} ~entry≈{c_close} (theor={theor_entry:.6f} @ {int(_ENTRY_OFFSET_PCT*100)}%), tp={tp:.6f}, sl={sl:.6f} t0={imb_time}")
        await tg(f"🚀 BREAKOUT {symbol} {side_raw} — MARKET open: ~entry≈{c_close:.6f} (theor±{_ENTRY_OFFSET_PCT*100:.1f}%), tp={tp:.6f}, sl={sl:.6f}")
        return True
    except Exception as e:
        # возвращаем резерв, если не получилось
        _allocator.close_one(pd.Timestamp.utcnow().to_pydatetime(), usd_alloc=usd, pnl_pct=0.0)
        log.error(f"[BRK_FAIL] {symbol} {side_raw}: {e}")
        return False

# ===================== FVG обработка (4h) =====================
async def on_candle_closed(symbol: str):
    """
    На закрытии 4h свечи:
      RETEST:
        - ищем FVG-имбалы,
        - считаем entry через strategy.select_entry_price(),
        - ставим лимитки с TTL, но только если entry ещё НЕ касался (проверяем барами строго > t0),
          + LTF-проверка на интервалах ENTRY_TOUCH_LTF.
        - backfill включён.
      BREAKOUT:
        - не ждём возврата; на свежем имбе — MARKET вход со SL/TP.
        - backfill отключён.
      MOMENTUM:
        - детект «глубокой свечи» по ATR/объёму и вход MARKET.
        - backfill не используется.
    """
    arr = _candles.get(symbol, [])
    if len(arr) < 10:
        log.info(f"[BAR_CLOSE] {symbol}: bars={len(arr)} (<10) — пропуск детекта")
        return

    df = pd.DataFrame([{"timestamp": x["t"], "open": x["o"], "high": x["h"], "low": x["l"], "close": x["c"], "volume": x["v"]} for x in arr])
    df.set_index("timestamp", inplace=True)

    try:
        imbs = detect_fvg_imbalances(
            df,
            volume_multiplier=1.5,
            max_days_to_fill=DEFAULT_TTL_DAYS,
            tolerance_pct=0,
            min_strength_pct=DEFAULT_MIN_STRENGTH
        )
        if not imbs:
            log.info(f"[FVG_CHECK] {symbol}: imbs=0 на последней свече")
            # даже если FVG нет — проверим MOMENTUM режим
            if _ENTRY_MODE == "MOMENTUM":
                await _maybe_momentum(symbol, df, arr[-1]["t"])
            return

        last = arr[-1]
        last_t = last["t"]

        # свежие имбы этой (закрытой) свечи
        fresh = [imb for imb in imbs if pd.to_datetime(imb["time"], utc=True) == last_t]
        log.info(f"[FVG_CHECK] {symbol}: свежих имб={len(fresh)} (last_t={last_t})")

        # === BREAKOUT: мгновенный вход MARKET, backfill отключаем ===
        if _ENTRY_MODE == "BREAKOUT":
            for imb in fresh:
                side_raw = str(imb["type"]).upper()
                if side_raw == "BUY" and not ENABLE_BUY:
                    log.info(f"[BRK_SKIP] {symbol}: BUY запрещён")
                    continue
                if side_raw == "SELL" and not ENABLE_SELL:
                    log.info(f"[BRK_SKIP] {symbol}: SELL запрещён")
                    continue
                await _enter_breakout_market(symbol, side_raw, last["c"], pd.to_datetime(imb["time"], utc=True))
            return  # никаких лимиток/бэкфилла

        # === MOMENTUM: deep bar + объём, MARKET ===
        if _ENTRY_MODE == "MOMENTUM":
            await _maybe_momentum(symbol, df, last_t)
            return

        # === RETEST: лимитки по entry, исключая сам бар имба из проверки касания ===
        reasons_all: List[str] = []
        placed = 0
        now_utc = pd.Timestamp.utcnow()

        for imb in fresh:
            side_raw = str(imb["type"]).upper()
            if side_raw == "BUY" and not ENABLE_BUY:
                reasons_all.append("BUY отключён")
                log.info(f"[FVG_SKIP] {symbol}: BUY запрещён ENABLE_BUY={ENABLE_BUY}")
                continue
            if side_raw == "SELL" and not ENABLE_SELL:
                reasons_all.append("SELL отключён")
                log.info(f"[FVG_SKIP] {symbol}: SELL запрещён ENABLE_SELL={ENABLE_SELL}")
                continue

            entry = select_entry_price(df, symbol, imb)
            if entry is None or entry <= 0:
                reasons_all.append("entry=None (select_entry_price)")
                log.info(f"[FVG_SKIP] {symbol}: entry=None из select_entry_price()")
                continue

            t0 = pd.to_datetime(imb["time"], utc=True)

            # 1) проверка касания на 4h барах ПОСЛЕ t0
            bars_after_imb = [b for b in arr if b["t"] > t0]   # ⛔️ исключаем сам бар
            touched_4h = any(b["l"] <= entry <= b["h"] for b in bars_after_imb)

            if touched_4h:
                reasons_all.append("entry уже касался цены (4h)")
                log.info(f"[FVG_SKIP] {symbol}: entry={entry} уже касался цены на 4h после t0={t0}")
                continue

            # 2) LTF-проверка касания между (t0, now]
            try:
                touched_ltf = entry_was_touched_ltf(
                    symbol=symbol,
                    entry=float(entry),
                    start_time=(t0 + pd.Timedelta(seconds=1)).to_pydatetime(),
                    end_time=now_utc.to_pydatetime(),
                    ltf_chain=tuple(_ENTRY_TOUCH_LTF),
                    category=BYBIT_CATEGORY
                )
            except Exception as e:
                log.warning(f"[LTF_CHECK_FAIL] {symbol} t0={t0} err={e}")
                touched_ltf = False

            if touched_ltf:
                reasons_all.append("entry уже касался цены (LTF)")
                log.info(f"[FVG_SKIP] {symbol}: entry={entry} уже касался LTF между {t0} и {now_utc}")
                continue

            sl, tp = _calc_sl_tp(float(entry), side_raw)
            oid = f"{symbol}_{int(pd.Timestamp(t0).timestamp())}"
            if _pending.get(oid):
                reasons_all.append("дубликат pending")
                log.info(f"[FVG_SKIP] {symbol}: дубликат pending oid={oid}")
                continue

            ok, reason = await _place_limit_order(symbol, side_raw, float(entry), tp, sl, oid, backfill=False)
            if ok:
                placed += 1
            else:
                reasons_all.append(reason or "unknown_fail")

        # 2) Бэкфилл только для RETEST
        if _BACKFILL_4H_BARS > 0:
            bars = [b for b in arr if b.get("confirm")]
            if bars:
                back_bars = bars[-_BACKFILL_4H_BARS:]
                log.info(f"[BACKFILL_CHECK] {symbol}: проверяем баров={len(back_bars)} (N={_BACKFILL_4H_BARS})")

                # жёсткий порог по дате (например, последняя неделя)
                cutoff_dt = pd.Timestamp.utcnow() - pd.Timedelta(days=_ENTRY_BACKFILL_LOOKBACK_DAYS)
                # минимальное время: максимум из "по числу баров" и "по дате"
                min_t = max(back_bars[0]["t"], cutoff_dt)

                # кандидаты-имбалы в окне [min_t, +∞)
                cand_imbs = [imb for imb in imbs if pd.to_datetime(imb["time"], utc=True) >= min_t]
                log.info(f"[BACKFILL_IMB] {symbol}: найдено имб={len(cand_imbs)} c min_t={min_t} (cutoff={cutoff_dt})")

                for imb in cand_imbs:
                    t0 = pd.to_datetime(imb["time"], utc=True)
                    side_raw = str(imb["type"]).upper()
                    if side_raw == "BUY" and not ENABLE_BUY:
                        reasons_all.append("BUY отключён (backfill)")
                        log.info(f"[BACKFILL_SKIP] {symbol}: BUY запрещён")
                        continue
                    if side_raw == "SELL" and not ENABLE_SELL:
                        reasons_all.append("SELL отключён (backfill)")
                        log.info(f"[BACKFILL_SKIP] {symbol}: SELL запрещён")
                        continue

                    entry = select_entry_price(df, symbol, imb)
                    if entry is None or entry <= 0:
                        reasons_all.append("entry=None (select_entry_price backfill)")
                        log.info(f"[BACKFILL_SKIP] {symbol}: entry=None из select_entry_price()")
                        continue

                    # 1) уже касалось на 4h ПОСЛЕ t0?
                    bars_after_imb = [b for b in back_bars if b["t"] > t0]
                    touched_4h = any(b["l"] <= entry <= b["h"] for b in bars_after_imb)
                    if touched_4h:
                        reasons_all.append("entry уже касался цены (4h backfill)")
                        log.info(f"[BACKFILL_SKIP] {symbol}: entry={entry} уже касался 4h после t0={t0}")
                        continue

                    # 2) LTF-проверка касания в окне (max(t0, cutoff_dt), now]
                    start_ltf = max(t0 + pd.Timedelta(seconds=1), cutoff_dt)
                    try:
                        touched_ltf = entry_was_touched_ltf(
                            symbol=symbol,
                            entry=float(entry),
                            start_time=start_ltf.to_pydatetime(),
                            end_time=pd.Timestamp.utcnow().to_pydatetime(),
                            ltf_chain=tuple(_ENTRY_TOUCH_LTF),
                            category=BYBIT_CATEGORY
                        )
                    except Exception as e:
                        log.warning(f"[LTF_CHECK_FAIL] {symbol} backfill t0={t0} err={e}")
                        touched_ltf = False

                    if touched_ltf:
                        reasons_all.append("entry уже касался цены (LTF backfill)")
                        log.info(f"[BACKFILL_SKIP] {symbol}: entry={entry} уже касался LTF между {start_ltf} и now")
                        continue

                    oid = f"{symbol}_{int(pd.Timestamp(t0).timestamp())}"
                    if _pending.get(oid):
                        reasons_all.append("дубликат pending (backfill)")
                        log.info(f"[BACKFILL_SKIP] {symbol}: дубликат pending oid={oid}")
                        continue

                    sl, tp = _calc_sl_tp(float(entry), side_raw)
                    ok, reason = await _place_limit_order(symbol, side_raw, float(entry), tp, sl, oid, backfill=True)
                    if ok:
                        placed += 1
                    else:
                        reasons_all.append(reason or "unknown_fail")
            else:
                log.info(f"[BACKFILL_CHECK] {symbol}: подтверждённых баров нет — пропуск")

        if placed:
            await tg(f"🆕 {symbol}: лимитных ордеров выставлено: {placed} (4h FVG + LTF check + backfill)")
        else:
            uniq = ", ".join(sorted(set([r for r in (reasons_all or []) if r])))
            if not uniq:
                uniq = "нет валидных сигналов / все отфильтрованы"
            log.info(f"[FVG_DONE] {symbol}: лимиток не поставлено — причины: {uniq}")

    except Exception:
        log.exception(f"[fvg] error for {symbol}")

async def _maybe_momentum(symbol: str, df: pd.DataFrame, last_t: pd.Timestamp):
    """Проверка MOMENTUM на закрытом баре и вход MARKET при выполнении условий."""
    try:
        df_p = df.copy()
        atr = _atr(df_p['high'], df_p['low'], df_p['close'], n=_MOM_ATR_N)
        vol_sma = df_p['volume'].rolling(_MOM_ATR_N, min_periods=1).mean()
        c = df_p.iloc[-1]
        rng = float(c['high'] - c['low'])
        body = float(abs(c['close'] - c['open']))
        atrv = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
        vol  = float(c['volume'])
        vavg = float(vol_sma.iloc[-1]) if pd.notna(vol_sma.iloc[-1]) else vol

        ok_range = (atrv > 0) and (rng >= _MOM_RANGE_ATR*atrv or body >= _MOM_BODY_ATR*atrv)
        ok_vol   = (vol >= _MOM_VOL_SMA*max(vavg, 1e-9))

        if ok_range and ok_vol:
            side_raw = "BUY" if float(c['close']) > float(c['open']) else "SELL"
            if side_raw == "BUY" and not ENABLE_BUY:
                log.info(f"[MOM_SKIP] {symbol}: BUY запрещён")
                return
            if side_raw == "SELL" and not ENABLE_SELL:
                log.info(f"[MOM_SKIP] {symbol}: SELL запрещён")
                return
            await _enter_momentum_market(symbol, side_raw, float(c['close']), last_t)
        else:
            log.info(f"[MOM_SKIP] {symbol}: not deep (body={body:.6f}, range={rng:.6f}, atr={atrv:.6f}, vol={vol:.2f}/{vavg:.2f})")
    except Exception:
        log.exception(f"[MOM] err for {symbol}")

# ===================== WS LOOPS =====================
async def price_stream_loop(symbols: List[str]):
    url = _ws_url()
    topics = [f"kline.{FVG_INTERVAL}.{sym}" for sym in symbols]
    payload = {"op": "subscribe", "args": topics}
    log.info(f"[ws] connecting {url} with {len(symbols)} symbols…")
    log.info(f"[ws] topics example: {topics[:5]}{' …' if len(topics)>5 else ''}")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.ws_connect(url, heartbeat=20) as ws:
                    await ws.send_json(payload)
                    log.info("[ws] subscribed")
                    if _bot and CHAT_ID:
                        await tg("🔌 WS connected & subscribed (4h klines).")

                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                                break
                            continue

                        data = json.loads(msg.data)
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
                                "confirm": bool(bar.get("confirm", False))
                            }
                            arr = _candles.setdefault(sym, [])
                            if arr and arr[-1]["t"] == item["t"]:
                                arr[-1] = item
                            else:
                                arr.append(item)
                                if len(arr) > MAX_CANDLES:
                                    arr.pop(0)

                            if item["confirm"]:
                                log.info(f"[BAR_CLOSE] {sym}: close={item['c']} t={item['t']} v={item['v']}")
                                await on_candle_closed(sym)

            except Exception:
                log.exception("[ws] error")
                if _bot and CHAT_ID:
                    await tg(f"⚠️ WS error. Reconnecting in 3s…")
                await asyncio.sleep(3)

# ===================== TTL CLOSER & BALANCE SYNC =====================
async def timeout_closer_loop():
    """Раз в минуту отменяем просроченные лимитки (TTL) и возвращаем резерв в аллокатор."""
    from utils.bybit_trade import cancel_order
    while True:
        try:
            _reload_env_if_needed()
            # отмена просроченных лимиток
            pending = _pending.all()
            if pending:
                now = pd.Timestamp.utcnow()
                for oid, row in list(pending.items()):
                    exp = pd.to_datetime(row.get("expire_at"))
                    if pd.isna(exp):
                        continue
                    if now >= exp:
                        try:
                            cancel_order(symbol=row["symbol"], order_link_id=oid, category=BYBIT_CATEGORY)
                        except Exception as e:
                            log.error(f"[TTL] cancel API error {oid}: {e}")
                        # вернуть резерв
                        usd_alloc = float(row.get("usd_alloc", 0.0) or 0.0)
                        if usd_alloc > 0:
                            _allocator.close_one(now.to_pydatetime(), usd_alloc=usd_alloc, pnl_pct=0.0)
                        _pending.pop(oid)
                        log.info(f"[TTL] cancel expired limit {row['symbol']} oid={oid} (refund {usd_alloc:.2f} USD)")
                        await tg(f"⏱️ CANCEL TTL {row['symbol']} (oid={oid})")

            # дедлайны для открытых поз (если используешь state_store для них)
            now = pd.Timestamp.utcnow()
            open_map = _state.all_open()
            for oid, p in list(open_map.items()):
                try:
                    dl = pd.to_datetime(p.get("close_deadline"))
                    if pd.isna(dl):
                        continue
                    if now >= dl:
                        _ = close_position_market(symbol=p["symbol"], category=BYBIT_CATEGORY)
                        _state.pop_open(oid)
                        log.info(f"[TIMEOUT CLOSE] {p['symbol']} oid={oid}")
                        await tg(f"⏱️ TIMEOUT CLOSE {p['symbol']} (oid={oid})")
                except Exception as e:
                    log.error(f"[timeout] close error {oid}: {e}")
        except Exception:
            log.exception("[timeout] loop err")
        await asyncio.sleep(60)

async def balance_sync_loop():
    """Периодически подтягиваем фактический баланс аккаунта и обновляем аллокатор (реинвест)."""
    while True:
        try:
            _reload_env_if_needed()
            if _BALANCE_SYNC_MIN <= 0:
                await asyncio.sleep(60)
                continue
            bal = get_wallet_balance("USDT")
            if bal > 0:
                _allocator.set_total(bal)
                log.info(f"[BALANCE] synced to {bal:.2f} USDT")
                if _bot and CHAT_ID:
                    await tg(f"💰 Balance sync → {bal:.2f} USDT")
        except Exception:
            log.exception("[BALANCE] sync failed")
        await asyncio.sleep(max(60, _BALANCE_SYNC_MIN * 60))

# ===================== МОНИТОРИНГ ЛИМИТОК/ПОЗИЦИЙ =====================
_last_counts = {"orders": -1, "positions": -1}
_last_symbols_snapshot = {"orders": set(), "positions": set()}
_last_summary_ts = 0.0

def _format_list_short(syms: List[str], limit: int = 8) -> str:
    if not syms:
        return "—"
    if len(syms) <= limit:
        return ", ".join(sorted(syms))
    head = ", ".join(sorted(syms)[:limit])
    return f"{head}, … (+{len(syms)-limit})"

async def monitor_orders_positions_loop():
    """Каждые POLL_ORDERS_SEC секунд шлём в ТГ обновления при изменениях и периодическую сводку."""
    global _last_summary_ts
    while True:
        try:
            _reload_env_if_needed()

            # открытые лимитки
            oo = get_open_orders(category=BYBIT_CATEGORY, settle_coin="USDT")
            olist = oo.get("result", {}).get("list", []) or []

            # Открытыми считаем статусы new/created/active/partiallyfilled (+ условные untriggered/triggered),
            # а также fallback по количествам: leavesQty > 0 или qty > cumExecQty.
            OPEN_OK = {"new", "created", "active", "partiallyfilled", "untriggered", "triggered"}
            CLOSED = {"filled", "cancelled", "canceled", "rejected", "partiallyfilledcancelled"}

            def _is_open(o: dict) -> bool:
                st = str(o.get("orderStatus", "") or "").replace("_", "").strip().lower()
                if st in CLOSED:
                    return False
                if st in OPEN_OK:
                    return True
                # fallback: по объёму, если статус неочевидный
                try:
                    leaves = float(o.get("leavesQty") or 0)
                    cum = float(o.get("cumExecQty") or 0)
                    qty = float(o.get("qty") or 0)
                    if leaves > 0 or qty > cum:
                        return True
                except Exception:
                    pass
                return False

            active_orders = [o for o in olist if _is_open(o)]

            # (опционально) диагностический вывод при BYBIT_DEBUG=1
            if active_orders and _BYBIT_DEBUG:
                try:
                    sample = [
                        {
                            "symbol": o.get("symbol"),
                            "status": o.get("orderStatus"),
                            "qty": o.get("qty"),
                            "cumExecQty": o.get("cumExecQty"),
                            "leavesQty": o.get("leavesQty"),
                            "tif": o.get("timeInForce"),
                            "orderLinkId": o.get("orderLinkId"),
                        } for o in active_orders[:5]
                    ]
                    log.info(f"[DBG_OPEN_ORDERS] n={len(active_orders)} sample={sample}")
                except Exception:
                    pass

            # для диагностики
            if _BYBIT_DEBUG:
                uniq_statuses = sorted({str(o.get("orderStatus", "")).lower() for o in olist})
                log.info(
                    f"[MONITOR_DEBUG] statuses={uniq_statuses} total_orders={len(olist)} active={len(active_orders)}")
            order_syms = {str(o.get("symbol")) for o in active_orders if o.get("symbol")}

            # открытые позиции (ненулевой размер)
            pp = get_positions(category=BYBIT_CATEGORY)
            plist = pp.get("result", {}).get("list", []) or []
            open_pos, pos_syms = [], set()
            for p in plist:
                size = float(p.get("size") or 0.0)
                if abs(size) > 0:
                    open_pos.append(p)
                    if p.get("symbol"):
                        pos_syms.add(str(p["symbol"]))

            cnt_o = len(active_orders)
            cnt_p = len(open_pos)

            changed = (cnt_o != _last_counts["orders"]) or (cnt_p != _last_counts["positions"]) \
                      or (order_syms != _last_symbols_snapshot["orders"]) \
                      or (pos_syms != _last_symbols_snapshot["positions"])

            summary_due = (time.time() - _last_summary_ts) > 600

            if changed or summary_due:
                _last_counts["orders"] = cnt_o
                _last_counts["positions"] = cnt_p
                _last_symbols_snapshot["orders"] = set(order_syms)
                _last_symbols_snapshot["positions"] = set(pos_syms)
                _last_summary_ts = time.time()

                close_ts, left = _next_4h_close_utc()
                msg = (
                    "📊 Статус:\n"
                    f"• Лимитки: {cnt_o}  ({_format_list_short(list(order_syms))})\n"
                    f"• Позиции: {cnt_p}  ({_format_list_short(list(pos_syms))})\n"
                    f"• TIF: {_LIMIT_ORDER_MODE}" + (f" (fallback→{_LIMIT_POSTONLY_FALLBACK})" if _LIMIT_ORDER_MODE == "POST_ONLY" else "") + "\n"
                    f"• ENTRY_MODE: {_ENTRY_MODE}" + (f" (offset={_ENTRY_OFFSET_PCT*100:.1f}%)" if _ENTRY_MODE=='BREAKOUT' else "") + "\n"
                    f"• ENTRY_TOUCH_LTF: {','.join(_ENTRY_TOUCH_LTF)} | BACKFILL_DAYS: {_ENTRY_BACKFILL_LOOKBACK_DAYS}\n"
                    f"• До закрытия 4h: {_timedelta_str(left)}"
                )
                await tg(msg)

        except Exception:
            log.exception("[MONITOR] err")

        await asyncio.sleep(max(5, _POLL_ORDERS_SEC))

# ===================== BACKFILL (только для RETEST) =====================
async def backfill_place_limits(symbols: List[str]):
    """Ставим лимитки по IMB за последние BACKFILL_4H_BARS подтверждённых свечей,
       с ДОП. ОГРАНИЧЕНИЕМ по дате (ENTRY_BACKFILL_LOOKBACK_DAYS),
       и только если entry ещё не касался (4h + LTF). История уже подкачана preload_history()."""
    if _ENTRY_MODE != "RETEST":
        log.info("[INIT_BACKFILL] пропуск: ENTRY_MODE != RETEST")
        return
    try:
        for sym in symbols:
            arr = _candles.get(sym, [])
            if not arr:
                log.info(f"[INIT_BACKFILL] {sym}: нет данных после preload — пропуск")
                continue
            bars = [b for b in arr if b.get("confirm")]
            if not bars:
                log.info(f"[INIT_BACKFILL] {sym}: нет подтверждённых баров — пропуск")
                continue
            bars = bars[-_BACKFILL_4H_BARS:] if _BACKFILL_4H_BARS > 0 else bars
            log.info(f"[INIT_BACKFILL] {sym}: проверяем баров={len(bars)} (N={_BACKFILL_4H_BARS})")

            df = pd.DataFrame([{"timestamp": x["t"], "open": x["o"], "high": x["h"], "low": x["l"], "close": x["c"], "volume": x["v"]} for x in bars])
            df.set_index("timestamp", inplace=True)

            imbs = detect_fvg_imbalances(
                df, volume_multiplier=1.5,
                max_days_to_fill=DEFAULT_TTL_DAYS,
                tolerance_pct=0,
                min_strength_pct=DEFAULT_MIN_STRENGTH
            )
            log.info(f"[INIT_BACKFILL] {sym}: найдено имб={len(imbs)} за {len(bars)} баров")

            if not imbs:
                log.info(f"[INIT_BACKFILL_DONE] {sym}: лимиток не поставлено (imbs=0)")
                continue

            placed_here = 0
            reasons: List[str] = []
            cutoff_dt = pd.Timestamp.utcnow() - pd.Timedelta(days=_ENTRY_BACKFILL_LOOKBACK_DAYS)
            min_t = max(bars[0]["t"], cutoff_dt)

            # ограничиваем бэкфилл и по числу баров, и по дате
            cand_imbs = [imb for imb in imbs if pd.to_datetime(imb["time"], utc=True) >= min_t]
            log.info(f"[INIT_BACKFILL] {sym}: кандидатов={len(cand_imbs)} с min_t={min_t} (cutoff={cutoff_dt})")

            for imb in cand_imbs:
                t0 = pd.to_datetime(imb["time"], utc=True)
                side_raw = str(imb["type"]).upper()
                if side_raw == "BUY" and not ENABLE_BUY:
                    reasons.append("BUY отключён")
                    continue
                if side_raw == "SELL" and not ENABLE_SELL:
                    reasons.append("SELL отключён")
                    continue

                entry = select_entry_price(df, sym, imb)
                if entry is None or entry <= 0:
                    reasons.append("entry=None (select_entry_price)")
                    continue

                # 1) 4h касание ПОСЛЕ t0?
                bars_after_imb = [b for b in bars if b["t"] > t0]
                touched_4h = any(b["l"] <= entry <= b["h"] for b in bars_after_imb)
                if touched_4h:
                    reasons.append("entry уже касался цены (4h)")
                    continue

                # 2) LTF касание в окне (max(t0, cutoff_dt), now]
                start_ltf = max(t0 + pd.Timedelta(seconds=1), cutoff_dt)
                try:
                    touched_ltf = entry_was_touched_ltf(
                        symbol=sym,
                        entry=float(entry),
                        start_time=start_ltf.to_pydatetime(),
                        end_time=pd.Timestamp.utcnow().to_pydatetime(),
                        ltf_chain=tuple(_ENTRY_TOUCH_LTF),
                        category=BYBIT_CATEGORY
                    )
                except Exception as e:
                    log.warning(f"[LTF_CHECK_FAIL] {sym} backfill t0={t0} err={e}")
                    touched_ltf = False

                if touched_ltf:
                    reasons.append("entry уже касался цены (LTF)")
                    continue

                oid = f"{sym}_{int(pd.Timestamp(t0).timestamp())}"
                if _pending.get(oid):
                    reasons.append("дубликат pending")
                    continue

                sl, tp = _calc_sl_tp(float(entry), side_raw)
                ok, reason = await _place_limit_order(sym, side_raw, float(entry), tp, sl, oid, backfill=True)
                if ok:
                    placed_here += 1
                else:
                    reasons.append(reason or "unknown_fail")

            if placed_here:
                await tg(f"📥 Backfill {sym}: выставлено лимиток = {placed_here}")
            else:
                uniq = ", ".join(sorted(set([r for r in reasons if r])))
                if not uniq:
                    uniq = "все сигналы отфильтрованы"
                log.info(f"[INIT_BACKFILL_DONE] {sym}: лимиток не поставлено — причины: {uniq}")

    except Exception:
        log.exception("[BACKFILL] error")

# ===================== ENTRYPOINT =====================
async def main():
    # начальный sync баланса (если получится)
    try:
        bal = get_wallet_balance("USDT")
        if bal > 0:
            _allocator.set_total(bal)
        log.info(f"[BALANCE] init set to {bal:.2f} USDT")
    except Exception as e:
        log.warning(f"[BALANCE] init sync failed: {e}")

    # первый вывод текущих параметров (с ETA закрытия 4h)
    close_ts, left = _next_4h_close_utc()
    log.info(
        f"[PARAMS] LIMIT_ORDER_MODE={_LIMIT_ORDER_MODE}, POSTONLY_FALLBACK={_LIMIT_POSTONLY_FALLBACK}, "
        f"BACKFILL_4H_BARS={_BACKFILL_4H_BARS}, POLL_ORDERS_SEC={_POLL_ORDERS_SEC}, "
        f"BALANCE_SYNC_MIN={_BALANCE_SYNC_MIN}, ENTRY_MODE={_ENTRY_MODE}, ENTRY_OFFSET_PCT={_ENTRY_OFFSET_PCT}, "
        f"MOM[n={_MOM_ATR_N}, body×ATR={_MOM_BODY_ATR}, range×ATR={_MOM_RANGE_ATR}, vol×SMA={_MOM_VOL_SMA}, "
        f"tp={_MOM_TP_PCT*100:.1f}%, sl={_MOM_SL_PCT*100:.1f}%], "
        f"ENTRY_TOUCH_LTF={_ENTRY_TOUCH_LTF}, BACKFILL_LOOKBACK_DAYS={_ENTRY_BACKFILL_LOOKBACK_DAYS}, "
        f"NEXT_4H_CLOSE_IN={_timedelta_str(left)}"
    )

    if _bot and CHAT_ID:
        await tg("🚀 Realtime автоторговля запущена (WS 4h FVG, RETEST/BREAKOUT/MOMENTUM, LTF-touch, датовый бэкфилл, аллокатор, мониторинг).")

    symbols = list(dict.fromkeys(TRADE_UNIVERSE))
    if not symbols:
        log.warning("TRADE_UNIVERSE is empty — проверь config.py.")
        symbols = []
    else:
        log.info(f"Universe size = {len(symbols)} (из TRADE_UNIVERSE)")

    # 1) Подкачиваем историю
    log.info(f"[PRELOAD] start: fetching last N={_BACKFILL_4H_BARS} x 4h bars for {len(symbols)} symbols via REST")
    await preload_history(symbols, _BACKFILL_4H_BARS)
    log.info("[PRELOAD] done")

    # 2) Бэкфилл лимиток (только для RETEST)
    await backfill_place_limits(symbols)

    # 3) Запускаем лупы
    await asyncio.gather(
        price_stream_loop(symbols),
        timeout_closer_loop(),
        balance_sync_loop(),
        monitor_orders_positions_loop(),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass