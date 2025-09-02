# autotrade_momentum.py
# ======================
# Чистый боевой режим MOMENTUM:
# - детект FVG (как в bulk) на закрытии 4h,
# - вход MARKET ровно на закрытии 4h (WS-событие confirm=true — приоритет),
# - TP/SL как в evaluate_momentum (режим entry; anchored отключён),
# - Telegram-алерты, горячая подгрузка ENV, лимиты, TTL-автозакрытие,
# - REST-фолбэк через ~6с на случай проблем WS.
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
from evaluate_common import get_cfg

from config import (
    BYBIT_CATEGORY, USE_MAINNET_MARKET_DATA,
    TELEGRAM_TOKEN, CHAT_ID,
    INITIAL_CAPITAL, POSITION_FRACTION,
    DEFAULT_TTL_DAYS, ENABLE_BUY, ENABLE_SELL,
    TRADE_UNIVERSE
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

OUT_TZ = os.getenv("OUT_TZ", None)
_MAX_DRIFT = float(os.getenv("MAX_ACCEPT_SLIPPAGE_PCT", "0.003"))  # 0.3%

# --- Параметры REST-фолбэка (можно менять через ENV) ---
_REST_ATTEMPTS     = int(os.getenv("REST_ATTEMPTS", "3"))
_REST_TIMEOUT_SEC  = float(os.getenv("REST_TIMEOUT_SEC", "10"))
_FALLBACK_CONC     = int(os.getenv("FALLBACK_CONCURRENCY", "10"))

# WS endpoints v5 (public)
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

# Telegram
_bot: Optional[Bot] = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
async def tg(text: str):
    if not _bot or not CHAT_ID:
        return
    chat = int(CHAT_ID) if str(CHAT_ID).lstrip("-").isdigit() else CHAT_ID
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _bot.send_message(chat_id=chat, text=text))

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
    fn = getattr(bybit_api, fn_name, None)
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
    cands = ["set_margin_mode", "position_set_margin_mode", "switch_margin_mode"]
    for name in cands:
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
    cands = ["set_leverage", "position_set_leverage", "change_leverage", "set_symbol_leverage"]
    for name in cands:
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
        for name in ("total", "get_total", "total_usd", "available", "balance", "equity"):
            attr = getattr(_allocator, name, None)
            if callable(attr):
                v = float(attr())
            elif attr is not None:
                v = float(attr)
            else:
                continue
            if v > 0:
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

    OUT_TZ = os.getenv("OUT_TZ", OUT_TZ)
    _MAX_DRIFT = float(os.getenv("MAX_ACCEPT_SLIPPAGE_PCT", str(_MAX_DRIFT)))

    _LAST_ENV_READ = now

    close_ts, left = _next_4h_close_utc()
    log.info(
        f"[ENV] MOM: TP={_MOM_TAKE_PCT*100:.2f}%, SL={_MOM_STOP_PCT*100:.2f}% mode={_MOM_TPSL_MODE}; "
        f"FVG: min_strength={_DEFAULT_MIN_STRENGTH}, vol×SMA={_FVG_VOL_MULT}, tol={_FVG_TOLERANCE_PCT}; "
        f"MAX_CONCURRENT={_MAX_CONCURRENT}; NEXT_4H_CLOSE_IN={str(left)}; OUT_TZ={OUT_TZ or 'UTC'}"
    )

# ====== детект и вход ======
def _pick_and_enter_args(df: pd.DataFrame, symbol: str):
    """Вернёт (bar_open, bar_close, detect_px, chosen_side, strength) или None если нет сигнала.
       ВАЖНО: df должен заканчиваться ЗАКРЫТЫМ баром (как в bulk). Сигнал сопоставляется по bar_close.
    """
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

    tol = pd.Timedelta(seconds=90)  # допуск на таймстамп вокруг bar_close
    chosen = []
    for imb in imbs:
        t0 = pd.to_datetime(imb.get("time"), utc=True, errors="coerce")
        if pd.isna(t0) or abs(t0 - bar_close) > tol:
            continue
        side = str(imb.get("type","")).upper()
        if side not in ("BUY","SELL"):
            continue
        if side == "BUY" and not ENABLE_BUY:  continue
        if side == "SELL" and not ENABLE_SELL: continue
        strength = float(imb.get("strength", 0.0))
        chosen.append({"side": side, "strength": strength})
    if not chosen:
        return None
    chosen.sort(key=lambda x: x["strength"], reverse=True)
    ch = chosen[0]
    return (bar_open, bar_close, detect_px, ch["side"], ch["strength"])

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
                        f"• fill={fill_px:.6f}\n"
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
        _last_done[symbol] = bar_open
        return True
    except Exception as e:
        log.error(f"[MOM_FAIL] {symbol} {side_raw}: {e}")
        return False

async def on_candle_closed(symbol: str):
    """На закрытии 4h (по WS confirm=true): детект FVG как в bulk; входим MARKET."""
    arr = _candles.get(symbol, [])
    if len(arr) < 12:
        log.info(f"[BAR_CLOSE] {symbol}: bars={len(arr)} — мало данных")
        return

    df = pd.DataFrame([{
        "timestamp": x["t"],
        "open":  x["o"],
        "high":  x["h"],
        "low":   x["l"],
        "close": x["c"],
        "volume":x["v"],
    } for x in arr])
    df.set_index("timestamp", inplace=True)

    args = _pick_and_enter_args(df, symbol)  # df оканчивается закрытым баром
    if args is None:
        bar_open = df.index[-1]
        log.info(f"[BAR_CLOSE] {symbol}: сигналов нет (min_strength={_DEFAULT_MIN_STRENGTH}, vol×SMA={_FVG_VOL_MULT})")
        _last_done[symbol] = bar_open
        return

    bar_open, bar_close, detect_px, side, strength = args

    # анти-дубль: если уже отметили этот бар (вдруг REST успел раньше)
    if _last_done.get(symbol) == bar_open:
        return

    # лимит по всем позициям
    if not _positions_limit_ok():
        log.info(f"[MOM_SKIP] {symbol}: достигнут лимит позиций ({_MAX_CONCURRENT})")
        _last_done[symbol] = bar_open
        return

    # не входить, если по символу уже есть открытая позиция
    try:
        pp = get_positions(category=BYBIT_CATEGORY)
        plist = (pp.get("result", {}) or {}).get("list", []) or []
        for p in plist:
            if p.get("symbol") == symbol and abs(float(p.get("size") or 0.0)) > 0:
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
            async with aiohttp.ClientSession() as session:  # НЕ тянем системные прокси
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
                            if arr and arr[-1]["t"] == item["t"]:
                                arr[-1] = item
                            else:
                                arr.append(item)
                                if len(arr) > MAX_CANDLES:
                                    arr.pop(0)

                            if item["confirm"]:
                                log.info(
                                    f"[BAR_CLOSE] {sym} close={item['c']:.6f} "
                                    f"t_open={_fmt_ts(item['t'])} "
                                    f"t_close={_fmt_ts(item['t'] + pd.Timedelta(hours=4))}"
                                )
                                await on_candle_closed(sym)

        except aiohttp.ClientConnectorDNSError as e:
            log.error(f"[DNS] resolve failed for {url}: {e}")
            await asyncio.sleep(backoff + random.uniform(0, backoff / 2))
            backoff = min(backoff * 2, 60)
        except Exception:
            log.exception("[ws] error, reconnect with backoff")
            await asyncio.sleep(backoff + random.uniform(0, backoff / 2))
            backoff = min(backoff * 2, 60)

# ====== REST-фолбэк на закрытии 4h ======
async def _fetch_4h_df(session: aiohttp.ClientSession, symbol: str, limit: int = 50) -> pd.DataFrame:
    base_url = http_base() + "/v5/market/kline"
    timeout = aiohttp.ClientTimeout(
        total=_REST_TIMEOUT_SEC, connect=_REST_TIMEOUT_SEC/2,
        sock_connect=_REST_TIMEOUT_SEC/2, sock_read=_REST_TIMEOUT_SEC
    )

    async def _one(category: str):
        params = {"category": category, "symbol": symbol, "interval": FVG_INTERVAL, "limit": str(int(limit))}
        for attempt in range(_REST_ATTEMPTS):
            try:
                async with session.get(base_url, params=params, timeout=timeout) as r:
                    text = await r.text()
                try:
                    js = json.loads(text)
                except Exception:
                    log.error(f"[REST_KLINE_BADJSON] {symbol} ({category}) status={r.status} body={text[:200]!r}")
                    await asyncio.sleep(0.3 + 0.2*attempt)
                    continue
                code = int(js.get("retCode", -1))
                if code == 0:
                    rows = (js.get("result", {}) or {}).get("list", []) or []
                    if not rows:
                        return pd.DataFrame()
                    rows.sort(key=lambda x: int(x[0]))  # по возрастанию start
                    rec = []
                    for s, o, h, l, c, vol, turnover in rows:
                        ts = _parse_ts_to_utc(s)
                        rec.append({
                            "timestamp": ts, "open": float(o), "high": float(h),
                            "low": float(l), "close": float(c), "volume": float(vol)
                        })
                    return pd.DataFrame(rec).set_index("timestamp")
                if code == 10006:  # rate limit
                    wait = 1.0 + 0.8*attempt + random.uniform(0, 0.5)
                    log.warning(f"[REST_RATE] {symbol} ({category}) retCode=10006 — sleep {wait:.2f}s")
                    await asyncio.sleep(wait)
                    continue
                log.error(f"[REST_KLINE_RET] {symbol} ({category}) retCode={code} retMsg={js.get('retMsg')}")
                return pd.DataFrame()
            except asyncio.TimeoutError:
                log.warning(f"[REST_TIMEOUT] {symbol} ({category}) try={attempt+1}/{_REST_ATTEMPTS}")
                continue
            except Exception as e:
                log.error(f"[REST_KLINE_ERR] {symbol} ({category}): {e}")
                await asyncio.sleep(0.3 + 0.2*attempt)
                continue
        return pd.DataFrame()

    df = await _one(BYBIT_CATEGORY)
    # если мы на деривах и пусто — пробуем spot как в bulk
    if df.empty and BYBIT_CATEGORY == "linear":
        spot_df = await _one("spot")
        if not spot_df.empty:
            log.info(f"[REST_FALLBACK] {symbol}: linear пусто, используем spot-данные для детекта")
            return spot_df
    return df

async def fallback_close_check_loop(symbols: List[str]):
    """Если WS молчит на закрытии — дергаем REST по всем символам.
       Детект строго по закрытому бару (как bulk): df_closed = df.iloc[:-1].
    """
    last_run_close: Optional[pd.Timestamp] = None
    sem = asyncio.Semaphore(max(1, _FALLBACK_CONC))

    async with aiohttp.ClientSession() as session:  # НЕ тянем системные прокси
        while True:
            try:
                _reload_env_if_needed()
                next_close, left = _next_4h_close_utc()

                # триггер — момент закрытия 4h, запускаем один раз на бар
                if left <= pd.Timedelta(seconds=5) and (last_run_close is None or next_close != last_run_close):
                    await asyncio.sleep(6)  # даём REST обновиться
                    last_run_close = next_close
                    log.info(f"[FALLBACK] 4h close checkpoint at {_fmt_ts(next_close)} — запускаю REST-проверку по {len(symbols)} символам")

                    async def _process_sym(sym: str):
                        async with sem:
                            try:
                                df = await _fetch_4h_df(session, sym, limit=50)
                                if df.empty:
                                    log.info(f"[FALLBACK_SKIP] {sym}: пусто из REST (см. логи выше)")
                                    return

                                if len(df) < 12:
                                    log.info(f"[FALLBACK_SKIP] {sym}: мало истории для детекта (len={len(df)})")
                                    return

                                # последний элемент — уже НОВЫЙ бар → отбрасываем
                                df_closed = df.iloc[:-1].copy() if len(df) >= 2 else df.copy()
                                bar_open_closed = df_closed.index[-1]

                                # уже обработан (WS или прошлый фолбэк)
                                if _last_done.get(sym) == bar_open_closed:
                                    return

                                # детект строго по закрытому бару
                                args = _pick_and_enter_args(df_closed, sym)
                                if args is None:
                                    log.info(f"[FALLBACK] {sym}: свеча закрылась — сигналов нет")
                                    _last_done[sym] = bar_open_closed
                                    return

                                # лимит по всем позициям
                                if not _positions_limit_ok():
                                    log.info(f"[FALLBACK_SKIP] {sym}: достигнут лимит позиций ({_MAX_CONCURRENT})")
                                    _last_done[sym] = bar_open_closed
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
                                    return

                                # вход
                                bar_open, bar_close, detect_px, side, strength = args
                                await _enter_momentum_market(sym, side, detect_px, bar_open, bar_close, strength=strength)
                                _last_done[sym] = bar_open_closed
                            except Exception as e:
                                log.error(f"[FALLBACK_ERR] {sym}: {e}")

                    await asyncio.gather(*[_process_sym(sym) for sym in symbols])

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
        price_stream_loop(symbols),   # WS — основной триггер
        fallback_close_check_loop(symbols),  # REST — страховка (быстрый параллельный)
        timeout_closer_loop(),
        balance_sync_loop(),
        monitor_loop(),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass