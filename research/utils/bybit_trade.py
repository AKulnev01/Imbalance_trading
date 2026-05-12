# utils/bybit_trade.py
import time
import hmac
import hashlib
import json
import math
import requests
import os
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone

from config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    BYBIT_TESTNET,
    BYBIT_CATEGORY,
    USE_MAINNET_MARKET_DATA,  # котировки/свечи берём с real?
    EXECUTION_ENV,            # "testnet" | "mainnet" — куда ШЛЁМ ОРДЕРА
)

# === Базовые урлы (v5) ===
BASE_PRIV_MAIN = "https://api.bybit.com"
BASE_PRIV_TEST = "https://api-demo.bybit.com"   # актуальный тестнет

BASE_PUBLIC_MAIN = "https://api.bybit.com"
BASE_PUBLIC_TEST = "https://api-testnet.bybit.com"

_BYBIT_DEBUG = bool(int(os.getenv("BYBIT_DEBUG", "0")))

def _dbg(msg: str):
    if _BYBIT_DEBUG:
        print(f"[BYBIT_DEBUG] {msg}")

def _private_base() -> str:
    """Хост для приватных (торговых) запросов — по среде исполнения ордеров."""
    return BASE_PRIV_TEST if BYBIT_TESTNET else BASE_PRIV_MAIN

def _public_base(use_mainnet: bool) -> str:
    """Хост для публичных данных (тикеры/свечи)."""
    return BASE_PUBLIC_MAIN if use_mainnet else BASE_PUBLIC_TEST

# ===== helpers: время/мс, публичные клины, LTF touch =====
def _ms(dt):
    if isinstance(dt, (int, float)):
        return int(dt)
    if isinstance(dt, datetime):
        return int(dt.timestamp() * 1000)
    # ISO-строка
    return int(datetime.fromisoformat(dt.replace('Z','+00:00')).timestamp()*1000)

class BybitHTTP:
    """
    Минимальный клиент Bybit v5 (подпись для приватных GET/POST).
    Приватные вызовы всегда идут на _private_base() (testnet/mainnet по BYBIT_TESTNET).
    Публичные вызовы (get_public) идут на _public_base() с флагом USE_MAINNET_MARKET_DATA.
    """
    def __init__(self, api_key: str, api_secret: str, recv_window: int = 30000, timeout: int = 15):
        self.api_key = api_key or ""
        self.api_secret = (api_secret or "").encode()
        self.base = _private_base()
        self.recv_window = int(recv_window)
        self.sess = requests.Session()
        self.sess.headers.update({"Content-Type": "application/json"})
        self.timeout = timeout

    def _ts(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, ts: str, payload: str) -> str:
        # v5: sign = HMAC_SHA256( timestamp + api_key + recv_window + payload )
        to_sign = ts + (self.api_key or "") + str(self.recv_window) + payload
        return hmac.new(self.api_secret, to_sign.encode(), hashlib.sha256).hexdigest()

    # -------- PUBLIC (без подписи) ----------
    def get_public(self, path: str, params: Dict[str, Any], *, use_mainnet: Optional[bool] = None) -> Dict[str, Any]:
        """
        Универсальный публичный GET к /v5/... на выбранную среду маркет-данных.
        По умолчанию берёт USE_MAINNET_MARKET_DATA=True|False.
        """
        use_mainnet = USE_MAINNET_MARKET_DATA if use_mainnet is None else bool(use_mainnet)
        url = _public_base(use_mainnet) + path
        _dbg(f"PUB GET {url} params={params}")
        r = self.sess.get(url, params=params, timeout=self.timeout)
        _dbg(f"PUB RESP status={r.status_code} text={r.text[:800]}")
        r.raise_for_status()
        return r.json()

    # -------- PRIVATE (подписанные) ----------
    def get_private(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Подписанный приватный GET (/v5/...) к среде _private_base(). Порядок параметров фиксируем."""
        url = self.base + path

        # 1) Отфильтруем None, приведём к str и ОТСОРТИРУЕМ (по ключу), чтобы и подпись, и фактический запрос совпадали.
        items = [(k, str(v)) for k, v in (params or {}).items() if v is not None]
        items.sort(key=lambda kv: kv[0])  # алфавитная сортировка по ключу

        # 2) Собираем query-строку в том же порядке
        query = "&".join(f"{k}={v}" for k, v in items)

        # 3) Подпись по v5: ts + api_key + recv_window + payload(query)
        ts = self._ts()
        sign = self._sign(ts, query)

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
        }

        _dbg(f"GET {url}?{query}")
        # 4) Важно: передаём params как список кортежей в ТОЧНО таком же порядке,
        #    чтобы requests не поменял порядок при сериализации.
        r = self.sess.get(url, params=items, headers=headers, timeout=self.timeout)
        _dbg(f"GET RESP status={r.status_code} text={r.text[:800]}")
        r.raise_for_status()
        return r.json()

    def post_private(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Подписанный приватный POST (/v5/...) к среде _private_base()."""
        url = self.base + path
        payload = json.dumps(body or {}, separators=(",", ":"))
        ts = self._ts()
        sign = self._sign(ts, payload)
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
        }
        _dbg(f"POST {url} body={body}")
        r = self.sess.post(url, data=payload, headers=headers, timeout=self.timeout)
        _dbg(f"POST RESP status={r.status_code} text={r.text[:1200]}")
        r.raise_for_status()
        return r.json()


# единый клиент приватки/паблика
_client = BybitHTTP(BYBIT_API_KEY, BYBIT_API_SECRET, recv_window=30000)

# опциональный инфолог про сплит окружений (это ваш целевой сценарий)
if _BYBIT_DEBUG and BYBIT_TESTNET and USE_MAINNET_MARKET_DATA:
    _dbg("Running with TESTNET orders but MAINNET market data: this is intentional by config.")

# ===== PUBLIC kline + LTF touch =====
def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, category="linear", limit=1000):
    """
    Публичные свечи с выбранной среды маркет-данных (USE_MAINNET_MARKET_DATA).
    Возвращает [(t_ms, high, low, close), ...] по возрастанию времени.
    """
    path = "/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": str(interval),
        "start": start_ms,
        "end": min(end_ms, int(time.time() * 1000)),
        "limit": limit
    }
    res = _client.get_public(path, params)
    if res.get("retCode") != 0:
        raise RuntimeError(f"kline fetch failed: {res.get('retMsg')}")
    data = (res.get("result") or {}).get("list") or []
    rows = []
    for o in data:
        # v5 kline: [startTime, open, high, low, close, volume, turnover]
        t, _o, h, l, c, *_ = o
        rows.append((int(t), float(h), float(l), float(c)))
    rows.sort(key=lambda r: r[0])
    return rows

def entry_was_touched_ltf(symbol: str, entry: float, start_time: datetime, end_time: datetime,
                          ltf_chain=("5","1"), category="linear") -> bool:
    """True, если entry попал внутрь свечи (low <= entry <= high) на любом LTF за период."""
    start_ms = _ms(start_time)
    end_ms   = _ms(end_time)
    for tf in ltf_chain:
        try:
            kl = fetch_klines(symbol, tf, start_ms, end_ms, category=category)
        except Exception:
            continue
        for t, high, low, close in kl:
            if low <= entry <= high:
                return True
    return False

# ===== проверка/парсинг ответов =====
def _check_ok(res: Dict[str, Any], ctx: str = ""):
    if res.get("retCode") != 0:
        raise RuntimeError(f"Bybit error {ctx}: {res.get('retCode')} {res.get('retMsg')} | {res}")

def _instruments_info(symbol: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Инструмент (шаги цены/кол-ва) — берём из СРЕДЫ ИСПОЛНЕНИЯ (тот же хост, что и ордера).
    Это гарантирует соответствие qtyStep/minNotional именно там, где будет ордер.
    """
    cat = (category or BYBIT_CATEGORY)
    url = _private_base() + "/v5/market/instruments-info"
    params = {"category": cat, "symbol": symbol}
    r = requests.get(url, params=params, timeout=10)
    _dbg(f"INSTRUMENTS {url} params={params} status={r.status_code}")
    r.raise_for_status()
    res = r.json()
    _check_ok(res, "instruments-info")
    items = res.get("result", {}).get("list", [])
    if not items:
        raise RuntimeError(f"instruments-info empty for {symbol}")
    return items[0]

def _ticker_last_price_env(symbol: str, category: Optional[str] = None, use_mainnet: bool = False) -> float:
    """Публичная последняя цена из выбранной среды (mainnet/testnet)."""
    base = _public_base(use_mainnet)
    url = base + "/v5/market/tickers"
    params = {"category": category or BYBIT_CATEGORY, "symbol": symbol}
    r = requests.get(url, params=params, timeout=10)
    _dbg(f"TICKER {url} params={params} status={r.status_code}")
    r.raise_for_status()
    res = r.json()
    _check_ok(res, "market-tickers(public)")
    items = res.get("result", {}).get("list", [])
    if not items:
        raise RuntimeError(f"ticker empty for {symbol}")
    return float(items[0]["lastPrice"])

def _ticker_last_price(symbol: str, category: Optional[str] = None) -> float:
    """Цена для стратегии — по USE_MAINNET_MARKET_DATA (реал/тестнет)."""
    return _ticker_last_price_env(symbol, category=category, use_mainnet=USE_MAINNET_MARKET_DATA)

def get_dual_prices(symbol: str, category: Optional[str] = None) -> Dict[str, float]:
    """Для логов: реальная (mainnet) vs тестнет-цена."""
    return {
        "mainnet": _ticker_last_price_env(symbol, category=category, use_mainnet=True),
        "testnet": _ticker_last_price_env(symbol, category=category, use_mainnet=False),
    }

def _parse_precisions(instr: Dict[str, Any]) -> Tuple[float, float]:
    # priceFilter/lotSizeFilter могут быть dict или строка json
    pf = instr.get("priceFilter", {}) if isinstance(instr.get("priceFilter"), dict) else json.loads(instr.get("priceFilter", "{}"))
    lf = instr.get("lotSizeFilter", {}) if isinstance(instr.get("lotSizeFilter"), dict) else json.loads(instr.get("lotSizeFilter", "{}"))
    tick = float(pf.get("tickSize", "0.01"))
    qty_step = float(lf.get("qtyStep") or lf.get("basePrecision") or "0.0001")
    return (tick, qty_step)

def _round_to_step(x: float, step: float, mode: str = "floor") -> float:
    if step <= 0:
        return x
    k = x / step
    if mode == "floor":
        k = math.floor(k)
    elif mode == "ceil":
        k = math.ceil(k)
    else:
        k = round(k)
    return k * step

def usd_to_qty(symbol: str, usd: float, price: Optional[float] = None, category: Optional[str] = None) -> str:
    """
    Qty считаем по цене СРЕДЫ ИСПОЛНЕНИЯ (EXECUTION_ENV),
    чтобы удовлетворять minNotional/qtyStep именно на том рынке, куда идёт ордер.
    """
    if price is None:
        use_mainnet_for_exec = (EXECUTION_ENV == "mainnet")
        p = _ticker_last_price_env(symbol, category=category, use_mainnet=use_mainnet_for_exec)
    else:
        p = float(price)

    instr = _instruments_info(symbol, category)  # шаги — из среды клиента (исполнения)
    _, qty_step = _parse_precisions(instr)

    raw_qty = float(usd) / max(float(p), 1e-12)
    qty = _round_to_step(raw_qty, qty_step, mode="floor")
    if qty <= 0:
        raise ValueError(f"Calculated qty <= 0 for {symbol}: usd={usd}, price={p}, step={qty_step}")
    return f"{qty:.10f}".rstrip("0").rstrip(".")

# ===== ТРЕЙД =====
def create_order(
    symbol: str,
    side: str,
    order_type: str,
    qty: str,
    price: Optional[str] = None,
    take_profit: Optional[str] = None,
    stop_loss: Optional[str] = None,
    reduce_only: bool = False,
    order_link_id: Optional[str] = None,
    category: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Универсальный вызов v5 /order/create (через подписанный клиент).
    """
    cat = (category or BYBIT_CATEGORY)

    # нормализация
    side_norm = str(side).strip().capitalize()         # Buy | Sell
    type_norm = str(order_type).strip().capitalize()   # Market | Limit

    body = {
        "category": cat,
        "symbol": symbol,
        "side": side_norm,
        "orderType": type_norm,
        "qty": str(qty),
        "reduceOnly": bool(reduce_only),
        **kwargs
    }
    if price is not None:
        body["price"] = str(price)
    if order_link_id:
        body["orderLinkId"] = str(order_link_id)

    if cat == "linear":
        if take_profit is not None:
            body["takeProfit"] = str(take_profit)
        if stop_loss is not None:
            body["stopLoss"] = str(stop_loss)
        body.setdefault("tpSlMode", "Partial")

    res = _client.post_private("/v5/order/create", body)
    _dbg(f"CREATE_ORDER resp={res}")
    _check_ok(res, "order/create")
    return res

def get_positions(symbol: Optional[str] = None, category: Optional[str] = None, settle_coin: str = "USDT") -> Dict[str, Any]:
    cat = category or BYBIT_CATEGORY
    params = {"category": cat, "settleCoin": settle_coin}
    if symbol:
        params["symbol"] = symbol
    res = _client.get_private("/v5/position/list", params)
    _check_ok(res, "position/list")
    return res

def get_open_orders(symbol: Optional[str] = None, category: Optional[str] = None, settle_coin: str = "USDT") -> Dict[str, Any]:
    """
    Unified-аккаунт: просим realtime-ордеры с openOnly=1.
    На некоторых аккаунтах помогает явный accountType=UNIFIED.
    """
    cat = category or BYBIT_CATEGORY
    params = {
        "category": cat,
        "openOnly": 1,
        "settleCoin": settle_coin,
        "accountType": "UNIFIED",  # unified-аккаунт
    }
    if symbol:
        params["symbol"] = symbol
    res = _client.get_private("/v5/order/realtime", params)
    _check_ok(res, "order/realtime")
    return res

def cancel_order(symbol: str, order_id: Optional[str] = None, order_link_id: Optional[str] = None,
                 category: Optional[str] = None) -> Dict[str, Any]:
    cat = category or BYBIT_CATEGORY
    body = {"category": cat, "symbol": symbol}
    if order_id:
        body["orderId"] = order_id
    if order_link_id:
        body["orderLinkId"] = order_link_id
    res = _client.post_private("/v5/order/cancel", body)
    _check_ok(res, "order/cancel")
    return res

def set_trading_stop(symbol: str, take_profit: Optional[str] = None, stop_loss: Optional[str] = None,
                     category: Optional[str] = None) -> Dict[str, Any]:
    cat = category or BYBIT_CATEGORY
    if cat != "linear":
        raise ValueError("set_trading_stop доступен только для 'linear'")
    body = {"category": "linear", "symbol": symbol}
    if take_profit is not None:
        body["takeProfit"] = str(take_profit)
    if stop_loss is not None:
        body["stopLoss"] = str(stop_loss)
    res = _client.post_private("/v5/position/trading-stop", body)
    _check_ok(res, "position/trading-stop")
    return res

# ===== БАЛАНС =====
def get_wallet_balance(coin: str = "USDT") -> float:
    """
    Возвращает equity для UNIFIED аккаунта (coin=USDT).
    Берётся из СРЕДЫ ИСПОЛНЕНИЯ (тестнет/мэйннет) через приватный API.
    """
    params = {"accountType": "UNIFIED", "coin": coin}
    res = _client.get_private("/v5/account/wallet-balance", params)
    _check_ok(res, "wallet-balance")
    rows = res.get("result", {}).get("list", [])
    if not rows:
        return 0.0
    coins = rows[0].get("coin", [])
    for c in coins:
        if str(c.get("coin")).upper() == coin.upper():
            try:
                return float(c.get("equity", 0.0))
            except Exception:
                return 0.0
    return 0.0

# ===== УДОБНЫЕ ВЫЗОВЫ =====
def open_position_market(
    symbol: str,
    side: str,
    usd_value: Optional[float] = None,
    usd_size: Optional[float] = None,
    entry_price_hint: Optional[float] = None,
    tp_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    order_link_id: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Маркет-вход с объёмом в USD. Qty считаем по цене СРЕДЫ ИСПОЛНЕНИЯ.
    Совместимо с:
      open_position_market(..., usd_value=..., tp_price=..., sl_price=...)
      open_position_market(..., usd_size =..., take_profit=..., stop_loss=...)
    """
    usd = usd_value if usd_value is not None else usd_size
    if usd is None:
        raise ValueError("open_position_market: usd_value/usd_size is required")

    # TP/SL aliases
    tp = tp_price if tp_price is not None else take_profit
    sl = sl_price if sl_price is not None else stop_loss

    qty = usd_to_qty(symbol, float(usd), price=entry_price_hint, category=category)
    res = create_order(
        symbol=symbol,
        side=side,
        order_type="Market",
        qty=qty,
        take_profit=str(tp) if tp is not None else None,
        stop_loss=str(sl) if sl is not None else None,
        reduce_only=False,
        order_link_id=order_link_id,
        category=category
    )
    return res

def close_position_market(symbol: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Закрыть позицию по рынку (reduceOnly). Определяем сторону по size.
    """
    cat = category or BYBIT_CATEGORY
    pos = get_positions(symbol, category=cat)
    items = pos.get("result", {}).get("list", [])
    if not items:
        return {"msg": "No position"}
    p = items[0]
    size = float(p.get("size") or 0.0)
    if size == 0.0:
        return {"msg": "Position is zero"}
    side = "Sell" if size > 0 else "Buy"  # лонг закрываем Sell, шорт — Buy
    qty = str(abs(size))
    res = create_order(
        symbol=symbol,
        side=side,
        order_type="Market",
        qty=qty,
        reduce_only=True,
        category=cat
    )
    return res