# autotrade_momentum.py
# ======================
# Чистый боевой режим MOMENTUM:
# - детект FVG (как в bulk) на закрытии 4h,
# - вход MARKET ровно на закрытии 4h (WS confirm=true — приоритет),
# - TP/SL как в evaluate_momentum (режим entry; anchored отключён),
# - Telegram-АЛЕРТЫ (только отправка), горячая подгрузка ENV, лимиты, TTL-автозакрытие,
# - REST-фолбэк через задержку (по умолчанию 6–10s) и проверка ИМЕННО только что закрытого бара.
#
# ==== ШПАРГАЛКА (bash/zsh) ====
# mkdir -p logs; TS=$(date +%F_%H%M); LOG="logs/momentum_${TS}.log"; \
# PYTHONUNBUFFERED=1 nohup python -u autotrade_momentum.py > "$LOG" 2>&1 & \
# echo $! > momentum.pid; ln -sf "$LOG" logs/momentum_latest.log; \
# echo "Started PID=$(cat momentum.pid) ; log=$LOG"
#
# tail -f logs/momentum_latest.log
#
# kill $(cat momentum.pid) 2>/dev/null || pkill -f autotrade_momentum.py; rm -f momentum.pid
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

# --- проектные утилиты ---
from utils.bybit_trade import (
    open_position_market, close_position_market,
    get_wallet_balance, get_positions
)
import utils.bybit_trade as bybit_api  # amend TP/SL, плечо/маржин-режим

from utils.symbols import fetch_top_symbols
from utils.state_store import JsonState
from utils.allocator import SmartAllocator
from utils.detect_fvg import detect_fvg_imbalances
from utils.trade_logger import log_trade_open
from evaluate_common import get_cfg

from config import (
    BYBIT_CATEGORY, USE_MAINNET_MARKET_DATA,
    TELEGRAM_TOKEN, CHAT_ID,
    INITIAL_CAPITAL, POSITION_FRACTION,
    DEFAULT_TTL_DAYS, ENABLE_BUY, ENABLE_SELL,
    TRADE_UNIVERSE,
    FEE_TAKER, SLIPPAGE_PCT
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

# FVG-детект «как в bulk»
_DEFAULT_MIN_STRENGTH = float(get_cfg("DEFAULT_MIN_STRENGTH", cast=float, default=0.0))
_FVG_VOL_MULT         = float(get_cfg("FVG_VOL_MULT",         cast=float, default=1.0))
_FVG_TOLERANCE_PCT    = float(get_cfg("FVG_TOLERANCE_PCT",    cast=float, default=0.0))

# Лимит одновременных позиций
_MAX_CONCURRENT = int(get_cfg("MAX_CONCURRENT_POSITIONS", cast=int, default=0))  # 0 = без лимита

# TP/SL
_MOM_TAKE_PCT  = float(os.getenv("MOMENTUM_TP_PCT", "0.03"))   # напр. 3%
_MOM_STOP_PCT  = float(os.getenv("MOMENTUM_SL_PCT", "0.01"))   # напр. 1%

# --- Плечо/маржин-режим (лайв, из ENV через get_cfg) ---
_BYBIT_MARGIN_MODE = str(get_cfg("BYBIT_MARGIN_MODE", cast=str, default="")).strip().lower()  # 'isolated' | 'cross' | ''
_BYBIT_LEVERAGE    = float(get_cfg("BYBIT_LEVERAGE",    cast=float, default=1))

# В лайве принудительно entry (anchored отключаем)
_MOM_TPSL_MODE = os.getenv("MOMENTUM_TP_SL_MODE", "entry").strip().lower()
if _MOM_TPSL_MODE != "entry":
    log.warning("[MOMENTUM] anchored-mode disabled in LIVE at startup; forcing entry.")
    _MOM_TPSL_MODE = "entry"

OUT_TZ = os.getenv("OUT_TZ", None)  # только для форматирования времени в логах/тг
_MAX_DRIFT = float(os.getenv("MAX_ACCEPT_SLIPPAGE_PCT", "0.003"))  # 0.3%

# --- Параметры REST-фолбэка / сеть ---
_REST_ATTEMPTS     = int(os.getenv("REST_ATTEMPTS", "3"))
_REST_TIMEOUT_SEC  = float(os.getenv("REST_TIMEOUT_SEC", "10"))
_FALLBACK_CONC     = int(os.getenv("FALLBACK_CONCURRENCY", "10"))
_REST_AFTER_CLOSE_DELAY_SEC = int(os.getenv("REST_AFTER_CLOSE_DELAY_SEC", "6"))
_REST_CONNECT_TIMEOUT = float(os.getenv("REST_CONNECT_TIMEOUT", "5"))
_REST_READ_TIMEOUT    = float(os.getenv("REST_READ_TIMEOUT", "10"))
_REST_ALLOW_SPOT_FALLBACK = str(os.getenv("REST_ALLOW_SPOT_FALLBACK", "0")).lower() in ("1","true","yes","y","on")

# сетевые счётчики (для алертов)
_net_batch_timeouts = 0
_net_batch_errors = 0

# --- WS endpoints v5 (public) ---
WS_MAIN_SPOT   = "wss://stream.bybit.com/v5/public/spot"
WS_TEST_SPOT   = "wss://stream-testnet.bybit.com/v5/public/spot"
WS_MAIN_LINEAR = "wss://stream.bybit.com/v5/public/linear"
WS_TEST_LINEAR = "wss://stream-testnet.bybit.com/v5/public/linear"

def _ws_url() -> str:
    use_mainnet = bool(get_cfg("USE_MAINNET_MARKET_DATA", cast=bool, default=USE_MAINNET_MARKET_DATA))
    category = str(get_cfg("BYBIT_CATEGORY", cast=str, default=BYBIT_CATEGORY)).strip().lower()
    if category == "spot":
        return WS_MAIN_SPOT if use_mainnet else WS_TEST_SPOT
    return WS_MAIN_LINEAR if use_mainnet else WS_TEST_LINEAR

# HTTP base
HTTP_BASE_MAIN = "https://api.bybit.com"
HTTP_BASE_TEST = "https://api-testnet.bybit.com"

def http_base() -> str:
    use_mainnet = bool(get_cfg("USE_MAINNET_MARKET_DATA", cast=bool, default=USE_MAINNET_MARKET_DATA))
    return HTTP_BASE_MAIN if use_mainnet else HTTP_BASE_TEST

# ====== глобальное состояние ======
STATE_PATH = os.getenv("AUTOTRADE_STATE_PATH", "autotrade_momentum_state.json")
_state   = JsonState(STATE_PATH)
_allocator = SmartAllocator(INITIAL_CAPITAL)

# свечи в памяти (только 4h, для WS-ветки)
_candles: Dict[str, List[dict]] = {}
MAX_CANDLES = 60
FVG_INTERVAL = "240"  # 4h

# уже обработанные 4h бары (по старту бара) — чтобы не дублировать вход WS/REST
_last_done: Dict[str, pd.Timestamp] = {}

# отметки закрытий от WS по текущему бару (значение = open закрытого бара)
_ws_last_close_bar: Dict[str, pd.Timestamp] = {}

# Telegram — только отправка сообщений (никакого polling здесь!)
_bot: Optional[Bot] = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
async def tg(text: str):
    if not _bot or not CHAT_ID:
        return
    chat = int(CHAT_ID) if str(CHAT_ID).lstrip("-").isdigit() else CHAT_ID
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _bot.send_message(chat_id=chat, text=text))

async def tg_big(text: str, chunk: int = 3800):
    if not _bot or not CHAT_ID or not text:
        return
    for i in range(0, len(text), chunk):
        await tg(text[i:i+chunk])

# ====== утилиты ======
def _parse_ts_to_utc(ts_val):
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

def _fmt_ts(ts: pd.Timestamp) -> str:
    ts = pd.to_datetime(ts, utc=True)
    if OUT_TZ:
        try:
            return ts.tz_convert(OUT_TZ).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")

def _next_4h_close_utc(now: Optional[pd.Timestamp] = None) -> Tuple[pd.Timestamp, pd.Timedelta]:
    now = now or pd.Timestamp.utcnow()
    hour = (now.hour // 4) * 4
    this_close = now.replace(hour=hour, minute=0, second=0, microsecond=0) + pd.Timedelta(hours=4)
    if this_close <= now:
        this_close += pd.Timedelta(hours=4)
    return this_close, this_close - now

def _calc_tpsl_entry(entry: float, side: str, risk_pct: float, rr: float) -> Tuple[float, float]:
    k = float(risk_pct)
    if side == "SELL":
        sl = entry * (1.0 + k)
        tp = entry - (sl - entry) * rr
    else:
        sl = entry * (1.0 - k)
        tp = entry + (entry - sl) * rr
    return float(sl), float(tp)

def _amend_tpsl_safe(symbol: str, tp: float, sl: float, *, category: str) -> bool:
    candidates = ["amend_tpsl", "update_position_tpsl", "set_position_tpsl", "position_set_tp_sl"]
    for name in candidates:
        fn = getattr(bybit_api, name, None)
        if callable(fn):
            try:
                fn(symbol=symbol, tp_price=str(tp), sl_price=str(sl), category=category)
                return True
            except Exception as e:
                log.error(f"[AMEND_FAIL:{name}] {symbol}: {e}")
    return False

# --- плечо/маржа: безопасные вызовы враппера ---
def _try_call(fn_name: str, **kwargs) -> bool:
    fn = getattr(bybit_api, name := fn_name, None)
    if not callable(fn):
        return False
    try:
        fn(**kwargs)
        return True
    except TypeError:
        return False
    except Exception as e:
        log.error(f"[BYBIT_API:{fn_name}] error: {e}")
        return False

def _ensure_margin_mode(symbol: str) -> bool:
    mode = (_BYBIT_MARGIN_MODE or "").lower()
    if not mode or BYBIT_CATEGORY != "linear":
        return True
    for name in ("set_margin_mode","position_set_margin_mode","switch_margin_mode"):
        if _try_call(name, symbol=symbol, category=BYBIT_CATEGORY, marginMode=mode) \
           or _try_call(name, symbol=symbol, category=BYBIT_CATEGORY, margin_mode=mode) \
           or _try_call(name, symbol=symbol, category=BYBIT_CATEGORY, mode=mode):
            log.info(f"[MARGIN_MODE] {symbol} → {mode}")
            return True
    log.warning(f"[MARGIN_MODE_FAIL] {symbol}: cannot set margin mode='{mode}'")
    return False

def _ensure_leverage(symbol: str) -> bool:
    if BYBIT_CATEGORY != "linear":
        return True
    lev = max(1.0, float(_BYBIT_LEVERAGE or 1.0))
    lev_s = str(int(round(lev)))
    for name in ("set_leverage","position_set_leverage","change_leverage","set_symbol_leverage"):
        if _try_call(name, symbol=symbol, category=BYBIT_CATEGORY, buyLeverage=lev_s, sellLeverage=lev_s) \
           or _try_call(name, symbol=symbol, category=BYBIT_CATEGORY, leverage=lev_s):
            log.info(f"[LEVERAGE] {symbol} → {lev_s}x")
            return True
    log.warning(f"[LEVERAGE_FAIL] {symbol}: cannot set leverage='{lev_s}x'")
    return False

async def _ensure_margin_and_leverage(symbol: str):
    ok_m = _ensure_margin_mode(symbol)
    ok_l = _ensure_leverage(symbol)
    if not (ok_m and ok_l):
        await tg(
            f"⚠️ {symbol}: не удалось выставить "
            f"{'маржин-режим' if not ok_m else ''}{' и ' if (not ok_m and not ok_l) else ''}"
            f"{'плечо' if not ok_l else ''}. Проверь вручную в Bybit."
        )

# ---- безопасный доступ к капиталу для аллокации ----
def _alloc_total_usd() -> float:
    try:
        for name in ("total","get_total","total_usd","available","balance","equity"):
            attr = getattr(_allocator, name, None)
            v = float(attr() if callable(attr) else attr)
            if v and v > 0:
                return v
    except Exception:
        pass
    try:
        v = float(get_wallet_balance("USDT"))
        if v > 0:
            return v
    except Exception:
        pass
    return float(INITIAL_CAPITAL or 0.0)

# --- лимиты позиций ---
def _positions_limit_ok() -> bool:
    try:
        if _MAX_CONCURRENT <= 0:
            return True
        pp = get_positions(category=BYBIT_CATEGORY)
        plist = (pp.get("result", {}) or {}).get("list", []) or []
        active_cnt = sum(1 for p in plist if abs(float(p.get("size") or 0.0)) > 0)
        return active_cnt < _MAX_CONCURRENT
    except Exception:
        return True  # не блокируем при сбое API

def _reload_env_if_needed():
    global _LAST_ENV_READ
    global _MOM_TAKE_PCT, _MOM_STOP_PCT, _MOM_TPSL_MODE, _MAX_DRIFT, OUT_TZ
    global _BYBIT_MARGIN_MODE, _BYBIT_LEVERAGE
    global _DEFAULT_MIN_STRENGTH, _FVG_VOL_MULT, _FVG_TOLERANCE_PCT, _MAX_CONCURRENT
    global _REST_ALLOW_SPOT_FALLBACK

    now = time.time()
    if _ENV_RELOAD_SEC <= 0 or (now - _LAST_ENV_READ) < _ENV_RELOAD_SEC:
        return
    if load_dotenv:
        try: load_dotenv(override=True)
        except Exception: pass

    _BYBIT_MARGIN_MODE = str(get_cfg("BYBIT_MARGIN_MODE", cast=str, default=_BYBIT_MARGIN_MODE)).strip().lower()
    try:
        _BYBIT_LEVERAGE = float(get_cfg("BYBIT_LEVERAGE", cast=float, default=_BYBIT_LEVERAGE))
    except Exception:
        _BYBIT_LEVERAGE = 1.0

    _MOM_TAKE_PCT  = float(os.getenv("MOMENTUM_TP_PCT",  str(_MOM_TAKE_PCT)))
    _MOM_STOP_PCT  = float(os.getenv("MOMENTUM_SL_PCT",  str(_MOM_STOP_PCT)))
    _MOM_TPSL_MODE = os.getenv("MOMENTUM_TP_SL_MODE", _MOM_TPSL_MODE).strip().lower()
    if _MOM_TPSL_MODE != "entry":
        log.warning("[MOMENTUM] anchored-mode disabled in LIVE; forcing entry.")
        _MOM_TPSL_MODE = "entry"

    _DEFAULT_MIN_STRENGTH = float(get_cfg("DEFAULT_MIN_STRENGTH", cast=float, default=_DEFAULT_MIN_STRENGTH))
    _FVG_VOL_MULT         = float(get_cfg("FVG_VOL_MULT",         cast=float, default=_FVG_VOL_MULT))
    _FVG_TOLERANCE_PCT    = float(get_cfg("FVG_TOLERANCE_PCT",    cast=float, default=_FVG_TOLERANCE_PCT))
    _MAX_CONCURRENT       = int(get_cfg("MAX_CONCURRENT_POSITIONS", cast=int, default=_MAX_CONCURRENT))

    OUT_TZ       = os.getenv("OUT_TZ", OUT_TZ)
    _MAX_DRIFT   = float(os.getenv("MAX_ACCEPT_SLIPPAGE_PCT", str(_MAX_DRIFT)))
    _REST_ALLOW_SPOT_FALLBACK = str(os.getenv("REST_ALLOW_SPOT_FALLBACK", "0")).lower() in ("1","true","yes","y","on")

    _LAST_ENV_READ = now

    close_ts, left = _next_4h_close_utc()
    log.info(
        f"[ENV] MOM: TP={_MOM_TAKE_PCT*100:.2f}%, SL={_MOM_STOP_PCT*100:.2f}% mode={_MOM_TPSL_MODE}; "
        f"FVG: min_strength={_DEFAULT_MIN_STRENGTH}, vol×SMA={_FVG_VOL_MULT}, tol={_FVG_TOLERANCE_PCT}; "
        f"MAX_CONCURRENT={_MAX_CONCURRENT}; NEXT_4H_CLOSE_IN={str(left)}; OUT_TZ={OUT_TZ or 'UTC'}"
    )

# ====== детект и вход ======
def _pick_and_enter_args(df: pd.DataFrame, symbol: str):
    df = df.copy().sort_index()
    bar_open  = df.index[-1]
    bar_close = bar_open + pd.Timedelta(hours=4)
    detect_px = float(df.iloc[-1]["close"])

    try:
        imbs = detect_fvg_imbalances(
            df,
            volume_multiplier=_FVG_VOL_MULT,
            tolerance_pct=_FVG_TOLERANCE_PCT,
            min_strength_pct=_DEFAULT_MIN_STRENGTH,
        ) or []
    except Exception as e:
        log.error(f"[DETECT_ERR] {symbol}: {e}")
        return None

    # ВАЖНО: detect_fvg.time == OPEN третьей свечи => матчим ТОЛЬКО bar_open
    tol = pd.Timedelta(seconds=60)  # небольшой допуск
    chosen = []
    for imb in imbs:
        t0 = pd.to_datetime(imb.get("time"), utc=True, errors="coerce")
        if pd.isna(t0) or abs(t0 - bar_open) > tol:
            continue
        side = str(imb.get("type","")).upper()
        if side not in ("BUY","SELL"):        continue
        if side == "BUY"  and not ENABLE_BUY:  continue
        if side == "SELL" and not ENABLE_SELL: continue
        chosen.append({"side": side, "strength": float(imb.get("strength", 0.0))})

    if not chosen:
        return None
    chosen.sort(key=lambda x: x["strength"], reverse=True)
    ch = chosen[0]
    return (bar_open, bar_close, detect_px, ch["side"], ch["strength"])

def _make_session() -> aiohttp.ClientSession:
    try:
        import aiodns  # noqa
        resolver = aiohttp.AsyncResolver()
        connector = aiohttp.TCPConnector(resolver=resolver, ttl_dns_cache=300, limit=0, limit_per_host=0)
    except Exception:
        connector = aiohttp.TCPConnector(ttl_dns_cache=300, limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(
        total=_REST_CONNECT_TIMEOUT + _REST_READ_TIMEOUT + 2,
        connect=_REST_CONNECT_TIMEOUT, sock_connect=_REST_CONNECT_TIMEOUT, sock_read=_REST_READ_TIMEOUT
    )
    return aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False)

async def _enter_momentum_market(symbol: str, side_raw: str, detect_px: float,
                                 bar_open: pd.Timestamp, bar_close: pd.Timestamp,
                                 strength: float = None) -> bool:
    rr = float(_MOM_TAKE_PCT) / max(float(_MOM_STOP_PCT), 1e-9)
    sl, tp = _calc_tpsl_entry(float(detect_px), side_raw, float(_MOM_STOP_PCT), rr)
    mode_note = "entry"

    batch_time = pd.Timestamp.utcnow()
    close_deadline = batch_time + pd.Timedelta(days=DEFAULT_TTL_DAYS)
    usd_value = float(POSITION_FRACTION) * float(_alloc_total_usd())
    if usd_value <= 0:
        msg = f"[MOM_SKIP] {symbol}: usd_alloc=0 (баланс зарезервирован/нет свободной маржи?)"
        log.info(msg); await tg("⚠️ " + msg); return False

    try:
        await _ensure_margin_and_leverage(symbol)
        resp = open_position_market(
            symbol=symbol,
            side=("Buy" if side_raw == "BUY" else "Sell"),
            usd_value=usd_value,
            tp_price=float(tp),
            sl_price=float(sl),
            category=BYBIT_CATEGORY
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
        if fill_px:
            drift_pct = (fill_px - detect_px) / detect_px
            if abs(drift_pct) > _MAX_DRIFT > 0:
                await tg(
                    f"⚠️ {symbol}: fill дрейф {drift_pct*100:.3f}% > {_MAX_DRIFT*100:.2f}% от detect_px\n"
                    f"• detect={detect_px:.6f}  fill={fill_px:.6f}"
                )
            sl_f, tp_f = _calc_tpsl_entry(float(fill_px), side_raw, float(_MOM_STOP_PCT), rr)
            if abs(tp_f - tp) > 0 or abs(sl_f - sl) > 0:
                amended = _amend_tpsl_safe(symbol, tp=tp_f, sl=sl_f, category=BYBIT_CATEGORY)
                if amended:
                    tp, sl = tp_f, sl_f
                else:
                    await tg(
                        f"‼️ MOMENTUM {symbol} {side_raw}: не удалось переставить TP/SL под fill\n"
                        f"• ПОСТАВЬ ВРУЧНУЮ: TP={tp_f:.6f}  SL={sl_f:.6f}"
                    )

        msg = (
            f"⚡️ MOMENTUM {symbol} {side_raw}\n"
            f"• bar(open)={_fmt_ts(bar_open)} → close={_fmt_ts(bar_close)}\n"
            f"• strength={('%.2f' % strength) if strength is not None else '—'}\n"
            f"• detect_px(4h close)={detect_px:.6f}\n"
            f"• fill≈{(fill_px if fill_px else detect_px):.6f}"
            + (f"  (drift={drift_pct*100:+.3f}%)" if drift_pct is not None else "") + "\n"
            f"• TP={tp:.6f}  SL={sl:.6f}  mode={mode_note}"
            + ("  (amended)" if amended else "")
            + f"\n• usd_alloc={usd_value:.2f}"
        )
        log.info("[MOM_MKT] " + msg.replace("\n", " | "))
        await tg(msg)

        oid = f"{symbol}_{int(bar_close.timestamp())}"
        _state.upsert_open(oid, {
            "symbol": symbol, "side": side_raw,
            "detect_px": float(detect_px),
            "fill_px": float(fill_px) if fill_px else None,
            "tp": float(tp), "sl": float(sl), "mode": mode_note,
            "bar_open": bar_open.isoformat(), "bar_close": bar_close.isoformat(),
            "placed_at": pd.Timestamp.utcnow().isoformat(),
            "close_deadline": close_deadline.isoformat(),
            "usd_alloc": usd_value,
        })

        # === LOG TRADE OPEN FOR COUNTERFACTUALS ===
        try:
            # detect_ts — open только что закрытого бара; entry_ts — текущее UTC (оба в ms)
            detect_ts_ms = int(bar_open.value // 10 ** 6)
            entry_ts_ms = int(pd.Timestamp.utcnow().value // 10 ** 6)
            entry_fill_px = float(fill_px if fill_px else detect_px)

            # комиссии/проскальзывание в б.п.: FEE_TAKER/SLIPPAGE_PCT в config заданы в долях
            fees_bps = float(FEE_TAKER) * 1e4
            slip_bps = float(SLIPPAGE_PCT) * 1e4

            log_trade_open(
                "./data/trades/trades.csv",
                {
                    "trade_id": oid,
                    "symbol": symbol,
                    "side": side_raw.lower(),
                    # "buy"/"sell" → ниже в evaluator приводим к "long"/"short" при необходимости
                    "detect_ts": detect_ts_ms,
                    "entry_ts": entry_ts_ms,
                    "entry_fill": entry_fill_px,
                    "notional_usd": float(usd_value),
                    "rr_tp": float(_MOM_TAKE_PCT * 100.0),  # проценты, как в примере trades.csv
                    "rr_sl": float(_MOM_STOP_PCT * 100.0),  # проценты
                    "ttl_hours": int(DEFAULT_TTL_DAYS * 24),
                    "fees_bps": fees_bps,
                    "slip_bps": slip_bps,
                    "spread_bps": 3.0  # при желании замени на динамику по спреду
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
    """На закрытии 4h (WS confirm=true): детект FVG как в bulk на ТОЛЬКО ЧТО ЗАКРЫТОМ баре; вход MARKET."""
    arr = _candles.get(symbol, [])
    if len(arr) < 12:
        log.info(f"[BAR_CLOSE] {symbol}: bars={len(arr)} — мало данных")
        return

    # полный DF из буфера WS
    df_full = pd.DataFrame([{
        "timestamp": x["t"],
        "open":  x["o"],
        "high":  x["h"],
        "low":   x["l"],
        "close": x["c"],
        "volume":x["v"],
    } for x in arr]).set_index("timestamp").sort_index()

    # open закрытого бара: либо пришёл параметром, либо из отметки WS
    expected_open = pd.to_datetime(closed_open or _ws_last_close_bar.get(symbol), utc=True, errors="coerce")
    if pd.isna(expected_open):
        log.warning(f"[BAR_CLOSE] {symbol}: нет expected_open — пропуск")
        return

    # СРЕЗ: строго до бара expected_open
    df = _slice_to_closed_bar(df_full, expected_open)
    if df is None or df.empty or df.index[-1] != expected_open:
        log.info(f"[BAR_CLOSE] {symbol}: REST/WS ещё не докинул бар {expected_open} — пропуск")
        return

    args = _pick_and_enter_args(df, symbol)  # df оканчивается закрытым баром expected_open
    if args is None:
        log.info(f"[BAR_CLOSE] {symbol}: сигналов нет (min_strength={_DEFAULT_MIN_STRENGTH}, vol×SMA={_FVG_VOL_MULT})")
        _last_done[symbol] = expected_open
        return

    bar_open, bar_close, detect_px, side, strength = args

    # анти-дубль / лимиты / занятость — как было
    if _last_done.get(symbol) == bar_open:
        return
    if not _positions_limit_ok():
        log.info(f"[MOM_SKIP] {symbol}: достигнут лимит позиций ({_MAX_CONCURRENT})")
        _last_done[symbol] = bar_open
        return
    try:
        plist = (get_positions(category=BYBIT_CATEGORY).get("result", {}) or {}).get("list", []) or []
        if any(p.get("symbol") == symbol and abs(float(p.get("size") or 0.0)) > 0 for p in plist):
            log.info(f"[MOM_SKIP] {symbol}: уже есть открытая позиция")
            _last_done[symbol] = bar_open
            return
    except Exception:
        pass

    await _enter_momentum_market(symbol, side, detect_px, bar_open, bar_close, strength=strength)

# ====== WS loop (+ бэкофф) ======
async def price_stream_loop(symbols: List[str]):
    url = _ws_url()
    CHUNK = 10
    topics_all = [f"kline.{FVG_INTERVAL}.{sym}" for sym in symbols]
    log.info(f"[ws] connecting {url} | symbols={len(symbols)}")

    backoff = 1.0
    while True:
        try:
            async with _make_session() as session:
                async with session.ws_connect(url, heartbeat=20, max_msg_size=2**22) as ws:
                    # подписываемся чанками
                    total = (len(topics_all) + CHUNK - 1) // CHUNK
                    for i in range(0, len(topics_all), CHUNK):
                        chunk = topics_all[i:i+CHUNK]
                        await ws.send_json({"op": "subscribe", "args": chunk})
                        log.info(f"[ws] subscribe chunk {i//CHUNK+1}/{total} size={len(chunk)}")
                        await asyncio.sleep(0.1)
                    log.info("[ws] subscribed")
                    backoff = 1.0  # успех → сброс бэкоффа

                    # слушаем
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                                break
                            continue

                        data = json.loads(msg.data)

                        # ACK
                        if data.get("op") == "subscribe" and "success" in data:
                            req = data.get("request") or {}
                            args = req.get("args") or []
                            log.info(f"[ws] ack subscribe success={data.get('success')} topics={len(args)}")
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
                            arr = _candles.setdefault(sym, [])
                            # нужно
                            if arr and arr[-1]["t"] == item["t"]:
                                arr[-1] = item
                            else:
                                arr.append(item)
                                if len(arr) > MAX_CANDLES:
                                    arr.pop(0)

                            # обработка закрытия ДОЛЖНА быть вне ветки, чтобы ловить confirm-апдейт той же свечи
                            if item["confirm"]:
                                _ws_last_close_bar[sym] = item["t"]  # open закрытого бара
                                log.info(
                                    f"[BAR_CLOSE] {sym} close={item['c']:.6f} "
                                    f"t_open={_fmt_ts(item['t'])} "
                                    f"t_close={_fmt_ts(item['t'] + pd.Timedelta(hours=4))}"
                                )
                                await on_candle_closed(sym, closed_open=item["t"])

        except aiohttp.ClientConnectorDNSError as e:
            log.error(f"[DNS] resolve failed for {url}: {e}")
            await asyncio.sleep(backoff + random.uniform(0, backoff / 2))
            backoff = min(backoff * 2, 60)
        except Exception:
            log.exception("[ws] error, reconnect with backoff")
            await asyncio.sleep(backoff + random.uniform(0, backoff / 2))
            backoff = min(backoff * 2, 60)

# ===== вспомогалка: срез до строго нужного закрытого бара =====
def _slice_to_closed_bar(df: pd.DataFrame, expected_open: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Вернуть df, оканчивающийся ровно баром с open == expected_open.
       Если такого бара нет — вернуть None (значит REST ещё не отдал нужную свечу)."""
    if df is None or df.empty:
        return None
    df = df.sort_index()
    if expected_open in df.index:
        return df.loc[:expected_open]
    if df.index[0] <= expected_open <= df.index[-1]:
        return df[df.index <= expected_open]
    return None

# ====== REST-фолбэк на закрытии 4h ======
async def _fetch_4h_df(session: aiohttp.ClientSession, symbol: str, limit: int = 50) -> pd.DataFrame:
    # пробуем основной домен и альтернативный
    bases = [http_base()]
    if "api.bybit.com" in bases[0]:
        bases.append(bases[0].replace("https://api.bybit.com", "https://api.bytick.com"))

    async def _one(category: str, base_url: str):
        url = base_url.rstrip("/") + "/v5/market/kline"
        params = {"category": category, "symbol": symbol, "interval": FVG_INTERVAL, "limit": str(int(limit))}
        for attempt in range(1, _REST_ATTEMPTS + 1):
            try:
                async with session.get(url, params=params) as r:
                    js = await r.json(content_type=None)
                code = int(js.get("retCode", -1))
                if code == 0:
                    rows = (js.get("result", {}) or {}).get("list", []) or []
                    if not rows:
                        return pd.DataFrame()
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
                global _net_batch_timeouts
                _net_batch_timeouts += 1
                log.warning(f"[REST_TIMEOUT] {symbol} ({category}) try={attempt}/{_REST_ATTEMPTS} base={base_url}")
                if attempt < _REST_ATTEMPTS: await asyncio.sleep(0.5 * attempt)
                continue
            except aiohttp.ClientConnectorError as e:
                global _net_batch_errors
                _net_batch_errors += 1
                log.warning(f"[REST_CONNECT] {symbol} ({category}) try={attempt}/{_REST_ATTEMPTS} base={base_url} err={e}")
                if attempt < _REST_ATTEMPTS: await asyncio.sleep(0.5 * attempt)
                continue
            except Exception as e:
                _net_batch_errors += 1
                if attempt < _REST_ATTEMPTS:
                    log.warning(f"[REST_RETRY] {symbol} ({category}) try={attempt}/{_REST_ATTEMPTS} base={base_url} err={e}")
                    await asyncio.sleep(1.0 * attempt); continue
                log.error(f"[REST_KLINE_ERR] {symbol} ({category}) base={base_url}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    df = pd.DataFrame()
    for base in bases:
        df = await _one(BYBIT_CATEGORY, base)
        if not df.empty: break

    # опциональный фолбэк на спот (по умолчанию выключен)
    if df.empty and BYBIT_CATEGORY == "linear" and _REST_ALLOW_SPOT_FALLBACK:
        for base in bases:
            df = await _one("spot", base)
            if not df.empty:
                log.info(f"[REST_FALLBACK] {symbol}: linear пусто → взяли spot (base={base})")
                break
    return df

async def fallback_close_check_loop(symbols: List[str]):
    """Если WS молчит на закрытии — дергаем REST по всем символам.
       Детект строго по ТОЛЬКО ЧТО ЗАКРЫТОМУ бару: берём бар с open == (next_close - 4h).
    """
    last_run_close: Optional[pd.Timestamp] = None
    sem = asyncio.Semaphore(max(1, _FALLBACK_CONC))

    async with _make_session() as session:
        while True:
            try:
                _reload_env_if_needed()
                next_close, left = _next_4h_close_utc()

                # триггер — момент закрытия 4h, запускаем один раз на бар
                if left <= pd.Timedelta(seconds=5) and (last_run_close is None or next_close != last_run_close):
                    delay = max(1.0, float(_REST_AFTER_CLOSE_DELAY_SEC))  # пауза чтобы REST обновился
                    await asyncio.sleep(delay)
                    last_run_close = next_close
                    target_open = next_close - pd.Timedelta(hours=4)
                    log.info(f"[FALLBACK] 4h close checkpoint at {_fmt_ts(next_close)} — запускаю REST-проверку по {len(symbols)} символам")

                    # --- что видел WS к этому закрытию
                    ws_seen = [s for s in symbols if _ws_last_close_bar.get(s) == target_open]
                    ws_missed = [s for s in symbols if s not in ws_seen]
                    await tg_big(
                        "⏱️ 4h close " + _fmt_ts(next_close) +
                        f"\n• WS закрыло: {len(ws_seen)}/{len(symbols)}" +
                        (f"\n• WS не прислал клоуз: {len(ws_missed)} — {', '.join(ws_missed[:20])}" if ws_missed else "") +
                        f"\n• Стартую REST через {int(delay)}s…"
                    )

                    # --- прогоняем REST параллельно
                    rest_signals_buy, rest_signals_sell = [], []
                    checked, no_data, errs = 0, 0, 0
                    global _net_batch_timeouts, _net_batch_errors
                    _net_batch_timeouts = 0; _net_batch_errors = 0

                    async def _process_sym(sym: str):
                        nonlocal checked, no_data, errs
                        async with sem:
                            try:
                                df = await _fetch_4h_df(session, sym, limit=50)
                                if df.empty:
                                    no_data += 1
                                    log.info(f"[FALLBACK_SKIP] {sym}: пусто из REST (см. логи выше)")
                                    return

                                if len(df) < 12:
                                    log.info(f"[FALLBACK_SKIP] {sym}: мало истории для детекта (len={len(df)})")
                                    checked += 1
                                    return

                                # вырезаем строго бар с open == target_open
                                df_closed = _slice_to_closed_bar(df, target_open)
                                if df_closed is None:
                                    # быстрый одноразовый ретрай: дать REST докинуть свечу
                                    await asyncio.sleep(2.0)
                                    df2 = await _fetch_4h_df(session, sym, limit=50)
                                    df_closed = _slice_to_closed_bar(df2, target_open)

                                if df_closed is None:
                                    log.info(f"[FALLBACK_SKIP] {sym}: REST не отдал бар {target_open} — пропуск (stale)")
                                    checked += 1
                                    return

                                bar_open_closed = target_open  # именно только что закрытый бар

                                # уже обработан (WS или прошлый фолбэк)
                                if _last_done.get(sym) == bar_open_closed:
                                    checked += 1
                                    return

                                # детект строго по закрытому бару (ровно target_open)
                                args = _pick_and_enter_args(df_closed, sym)
                                if args is None:
                                    log.info(f"[FALLBACK] {sym}: свеча закрылась — сигналов нет")
                                    _last_done[sym] = bar_open_closed
                                    checked += 1
                                    return

                                # лимит по всем позициям
                                if not _positions_limit_ok():
                                    log.info(f"[FALLBACK_SKIP] {sym}: достигнут лимит позиций ({_MAX_CONCURRENT})")
                                    _last_done[sym] = bar_open_closed
                                    checked += 1
                                    return

                                # занятость по символу
                                try:
                                    pp = get_positions(category=BYBIT_CATEGORY)
                                    plist = (pp.get("result", {}) or {}).get("list", []) or []
                                    busy = any(p.get("symbol") == sym and abs(float(p.get("size") or 0.0)) > 0 for p in plist)
                                except Exception:
                                    busy = False
                                if busy:
                                    log.info(f"[FALLBACK_SKIP] {sym}: уже есть открытая позиция")
                                    _last_done[sym] = bar_open_closed
                                    checked += 1
                                    return

                                # вход
                                bar_open, bar_close, detect_px, side, strength = args
                                ok = await _enter_momentum_market(sym, side, detect_px, bar_open, bar_close, strength=strength)
                                if ok:
                                    (rest_signals_buy if side == "BUY" else rest_signals_sell).append(sym)
                                _last_done[sym] = bar_open_closed
                                checked += 1
                            except Exception as e:
                                errs += 1
                                log.error(f"[FALLBACK_ERR] {sym}: {e}")

                    t0 = time.time()
                    await asyncio.gather(*[_process_sym(sym) for sym in symbols])
                    dt = time.time() - t0

                    msg = (
                        f"🩹 REST итог (за {dt:.1f}s)\n"
                        f"• проверено: {checked} из {len(symbols)}\n"
                        f"• сигналы BUY: {len(rest_signals_buy)} — {', '.join(rest_signals_buy[:30])}\n"
                        f"• сигналы SELL: {len(rest_signals_sell)} — {', '.join(rest_signals_sell[:30])}\n"
                        f"• пустых/нет данных: {no_data}; ошибок: {errs}\n"
                    )
                    if _net_batch_timeouts or _net_batch_errors:
                        msg += f"⚠️ сеть: timeouts={_net_batch_timeouts}, conn_errs={_net_batch_errors}\n"
                    await tg_big(msg)

                await asyncio.sleep(1.0)
            except Exception:
                log.exception("[FALLBACK_LOOP] err")
                await asyncio.sleep(2.0)

# ====== Сервисные лупы ======
async def timeout_closer_loop():
    """Жёсткий дедлайн на позиции: по DEFAULT_TTL_DAYS — закрываем MARKET (страховка)."""
    while True:
        try:
            _reload_env_if_needed()
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
    """Короткая сводка каждые 10 мин."""
    last = 0.0
    while True:
        try:
            _reload_env_if_needed()
            if time.time() - last > 600:
                close_ts, left = _next_4h_close_utc()
                msg = (
                    f"📊 MOMENTUM live\n"
                    f"• TP={_MOM_TAKE_PCT*100:.2f}%  SL={_MOM_STOP_PCT*100:.2f}%  mode={_MOM_TPSL_MODE}\n"
                    f"• FVG: min_strength={_DEFAULT_MIN_STRENGTH}  vol×SMA={_FVG_VOL_MULT}  tol={_FVG_TOLERANCE_PCT}\n"
                    f"• MAX_CONCURRENT={_MAX_CONCURRENT}\n"
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
    # первичный баланс
    try:
        bal = get_wallet_balance("USDT")
        if bal > 0:
            try: _allocator.set_total(bal)
            except Exception: pass
        log.info(f"[BALANCE] init={bal:.2f} USDT")
    except Exception as e:
        log.warning(f"[BALANCE] init sync failed: {e}")

    # символы
    syms_env = os.getenv("TRADE_UNIVERSE", ",".join(TRADE_UNIVERSE) if isinstance(TRADE_UNIVERSE, (list,tuple)) else str(TRADE_UNIVERSE))
    symbols = [s.strip().upper() for s in syms_env.split(",") if s.strip()]
    if not symbols:
        try:
            symbols = fetch_top_symbols()[:50]
        except Exception:
            symbols = []
    if not symbols:
        log.error("TRADE_UNIVERSE пуст — проверь config.py/ENV")
        return

    # привет
    if _bot and CHAT_ID:
        await tg("🚀 MOMENTUM автоторговля запущена (WS+REST; FVG как в bulk; TP/SL как в eval-entry; вход на 4h close).")

    await asyncio.gather(
        price_stream_loop(symbols),          # WS — основной триггер
        fallback_close_check_loop(symbols),  # REST — страховка (параллельный)
        timeout_closer_loop(),
        balance_sync_loop(),
        monitor_loop(),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass