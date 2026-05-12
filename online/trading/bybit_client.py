from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional


class BybitClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("BYBIT_API_KEY", "").strip()
        self.api_secret = os.environ.get("BYBIT_API_SECRET", "").strip()
        self.testnet = os.environ.get("BYBIT_TESTNET", "0").strip() == "1"
        self.category = os.environ.get("BYBIT_CATEGORY", "linear").strip()
        self.account_type = os.environ.get("BYBIT_ACCOUNT_TYPE", "UNIFIED").strip()

        self._session = None

    def _get_session(self):
        if self._session is not None:
            return self._session

        if not self.api_key or not self.api_secret:
            raise RuntimeError("BYBIT_API_KEY/BYBIT_API_SECRET are empty")

        try:
            from pybit.unified_trading import HTTP
        except ImportError as exc:
            raise RuntimeError("pybit is not installed. Install: pip install pybit") from exc

        self._session = HTTP(
            testnet=self.testnet,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        return self._session

    def get_wallet_balance_usdt(self) -> float:
        session = self._get_session()
        resp = session.get_wallet_balance(accountType=self.account_type, coin="USDT")
        self._raise_if_bad(resp)

        result = resp.get("result") or {}
        accounts = result.get("list") or []
        if not accounts:
            return 0.0

        coins = accounts[0].get("coin") or []
        for coin in coins:
            if str(coin.get("coin")).upper() == "USDT":
                value = coin.get("availableToWithdraw")
                if value in (None, ""):
                    value = coin.get("walletBalance")
                return float(value or 0.0)

        return 0.0

    def get_ticker_last_price(self, symbol: str) -> float:
        session = self._get_session()
        resp = session.get_tickers(category=self.category, symbol=symbol)
        self._raise_if_bad(resp)

        rows = (resp.get("result") or {}).get("list") or []
        if not rows:
            raise RuntimeError("ticker not found for symbol={}".format(symbol))

        return float(rows[0]["lastPrice"])

    def get_instrument_info(self, symbol: str) -> Dict[str, float]:
        session = self._get_session()
        resp = session.get_instruments_info(category=self.category, symbol=symbol)
        self._raise_if_bad(resp)

        rows = (resp.get("result") or {}).get("list") or []
        if not rows:
            raise RuntimeError("instrument not found for symbol={}".format(symbol))

        row = rows[0]
        lot = row.get("lotSizeFilter") or {}
        price_filter = row.get("priceFilter") or {}

        return {
            "qty_step": float(lot.get("qtyStep") or 0.0),
            "min_order_qty": float(lot.get("minOrderQty") or 0.0),
            "min_notional": float(lot.get("minNotionalValue") or 0.0),
            "tick_size": float(price_filter.get("tickSize") or 0.0),
        }

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_link_id: str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        session = self._get_session()

        bybit_side = "Buy" if str(side).upper() == "LONG" else "Sell"

        resp = session.place_order(
            category=self.category,
            symbol=symbol,
            side=bybit_side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=bool(reduce_only),
            orderLinkId=order_link_id,
        )
        self._raise_if_bad(resp)
        return resp
    def place_reduce_only_market_close(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_link_id: str,
    ) -> Dict[str, Any]:
        session = self._get_session()

        side_u = str(side).upper()
        if side_u == "LONG":
            close_side = "Sell"
        elif side_u == "SHORT":
            close_side = "Buy"
        else:
            raise RuntimeError("bad side for reduce-only close: {}".format(side))

        resp = session.place_order(
            category=self.category,
            symbol=str(symbol).upper(),
            side=close_side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True,
            orderLinkId=str(order_link_id),
        )
        self._raise_if_bad(resp)
        return resp

    def place_tp_sl_orders(
            self,
            symbol: str,
            side: str,
            qty: float,
            tp_px: float,
            sl_px: float,
            tp_order_link_id: str,
            sl_order_link_id: str,
    ) -> Dict[str, Any]:
        session = self._get_session()

        side_u = str(side).upper()
        if side_u == "LONG":
            close_side = "Sell"
            tp_trigger_direction = 1
            sl_trigger_direction = 2
        elif side_u == "SHORT":
            close_side = "Buy"
            tp_trigger_direction = 2
            sl_trigger_direction = 1
        else:
            raise RuntimeError("bad side for TP/SL orders: {}".format(side))

        sl_resp = session.place_order(
            category=self.category,
            symbol=str(symbol).upper(),
            side=close_side,
            orderType="Market",
            triggerPrice=str(sl_px),
            triggerDirection=sl_trigger_direction,
            qty=str(qty),
            reduceOnly=True,
            orderLinkId=str(sl_order_link_id),
        )
        self._raise_if_bad(sl_resp)

        tp_resp = session.place_order(
            category=self.category,
            symbol=str(symbol).upper(),
            side=close_side,
            orderType="Market",
            triggerPrice=str(tp_px),
            triggerDirection=tp_trigger_direction,
            qty=str(qty),
            reduceOnly=True,
            orderLinkId=str(tp_order_link_id),
        )
        self._raise_if_bad(tp_resp)

        return {
            "take_profit": tp_resp,
            "stop_loss": sl_resp,
        }

    def get_order_history(self, symbol: str, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> Dict[str, Any]:
        session = self._get_session()

        kwargs = {
            "category": self.category,
            "symbol": symbol,
        }

        if order_id:
            kwargs["orderId"] = order_id
        if order_link_id:
            kwargs["orderLinkId"] = order_link_id

        resp = session.get_order_history(**kwargs)
        self._raise_if_bad(resp)
        return resp

    def get_executions(self, symbol: str, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> Dict[str, Any]:
        session = self._get_session()

        kwargs: Dict[str, Any] = {
            "category": self.category,
            "symbol": symbol,
        }

        if order_id:
            kwargs["orderId"] = order_id
        if order_link_id:
            kwargs["orderLinkId"] = order_link_id

        resp = session.get_executions(**kwargs)
        self._raise_if_bad(resp)
        return resp
    def get_order_executions_by_link_id(
        self,
        symbol: str,
        order_link_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        resp = self.get_executions(
            symbol=str(symbol).upper(),
            order_link_id=str(order_link_id),
        )

        result = resp.get("result") or {}
        rows = result.get("list") or []

        if not isinstance(rows, list):
            return []

        return rows[:int(limit)]

    def get_avg_fill_price_by_link_id(
        self,
        symbol: str,
        order_link_id: str,
    ) -> Optional[float]:
        rows = self.get_order_executions_by_link_id(
            symbol=symbol,
            order_link_id=order_link_id,
            limit=50,
        )

        total_qty = 0.0
        total_value = 0.0

        for row in rows:
            try:
                px = float(row.get("execPrice") or 0.0)
                qty = float(row.get("execQty") or 0.0)
            except Exception:
                continue

            if px <= 0.0 or qty <= 0.0:
                continue

            total_qty += qty
            total_value += px * qty

        if total_qty <= 0.0:
            return None

        return float(total_value / total_qty)


    def get_position(self, symbol: str) -> Dict[str, Any]:
        session = self._get_session()
        resp = session.get_positions(category=self.category, symbol=symbol)
        self._raise_if_bad(resp)
        return resp

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        session = self._get_session()

        kwargs: Dict[str, Any] = {
            "category": self.category,
        }

        if symbol:
            kwargs["symbol"] = str(symbol).upper()
        else:
            kwargs["settleCoin"] = "USDT"

        resp = session.get_positions(**kwargs)
        self._raise_if_bad(resp)

        rows = (resp.get("result") or {}).get("list") or []
        out: List[Dict[str, Any]] = []

        for row in rows:
            try:
                size = abs(float(row.get("size") or 0.0))
            except Exception:
                size = 0.0

            if size <= 0.0:
                continue

            out.append(row)

        return out

    def has_open_position(self, symbol: Optional[str] = None) -> bool:
        return len(self.get_open_positions(symbol=symbol)) > 0


    def cancel_order(self, symbol: str, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> Dict[str, Any]:
        session = self._get_session()
        kwargs = {"category": self.category, "symbol": symbol}

        if order_id:
            kwargs["orderId"] = order_id
        if order_link_id:
            kwargs["orderLinkId"] = order_link_id

        resp = session.cancel_order(**kwargs)
        self._raise_if_bad(resp)
        return resp

    def _raise_if_bad(self, resp: Dict[str, Any]) -> None:
        ret_code = int(resp.get("retCode", -1))
        if ret_code != 0:
            raise RuntimeError("Bybit error retCode={} retMsg={} resp={}".format(
                resp.get("retCode"),
                resp.get("retMsg"),
                resp,
            ))
