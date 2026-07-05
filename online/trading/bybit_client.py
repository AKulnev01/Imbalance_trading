from __future__ import annotations

import os
from decimal import Decimal, ROUND_DOWN
import time
from typing import Any, Dict, List, Optional



def format_bybit_decimal(value, max_decimals=8):
    dec = Decimal(str(value))
    quant = Decimal("1").scaleb(-int(max_decimals))
    dec = dec.quantize(quant, rounding=ROUND_DOWN)

    text = format(dec, "f").rstrip("0").rstrip(".")

    if text == "" or text == "-0":
        return "0"

    return text


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

    def set_symbol_leverage(
        self,
        symbol: str,
        leverage: float,
    ) -> Dict[str, Any]:
        session = self._get_session()

        leverage_f = float(leverage)
        if leverage_f <= 0.0:
            raise RuntimeError("bad leverage: {}".format(leverage))

        leverage_text = str(int(leverage_f)) if leverage_f.is_integer() else str(leverage_f)

        try:
            resp = session.set_leverage(
                category=self.category,
                symbol=str(symbol).upper(),
                buyLeverage=leverage_text,
                sellLeverage=leverage_text,
            )
        except Exception as e:
            error_text = str(e)
            if "110043" in error_text or "not modified" in error_text.lower():
                return {
                    "ok": True,
                    "already_set": True,
                    "symbol": str(symbol).upper(),
                    "leverage": leverage_f,
                    "response": {
                        "retCode": 110043,
                        "retMsg": error_text,
                    },
                }
            raise

        ret_code = int(resp.get("retCode", -1))
        ret_msg = str(resp.get("retMsg") or "")

        if ret_code == 0:
            return {
                "ok": True,
                "already_set": False,
                "symbol": str(symbol).upper(),
                "leverage": leverage_f,
                "response": resp,
            }

        if ret_code == 110043 or "not modified" in ret_msg.lower():
            return {
                "ok": True,
                "already_set": True,
                "symbol": str(symbol).upper(),
                "leverage": leverage_f,
                "response": resp,
            }

        self._raise_if_bad(resp)

        return {
            "ok": True,
            "already_set": False,
            "symbol": str(symbol).upper(),
            "leverage": leverage_f,
            "response": resp,
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
            qty=format_bybit_decimal(qty),
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
            qty=format_bybit_decimal(qty),
            reduceOnly=True,
            orderLinkId=str(order_link_id),
        )
        self._raise_if_bad(resp)
        return resp

    def get_close_order_side_and_trigger_directions(self, side: str) -> Dict[str, Any]:
        side_u = str(side).upper().strip()

        if side_u == "LONG":
            return {
                "close_side": "Sell",
                "tp_trigger_direction": 1,
                "sl_trigger_direction": 2,
            }

        if side_u == "SHORT":
            return {
                "close_side": "Buy",
                "tp_trigger_direction": 2,
                "sl_trigger_direction": 1,
            }

        raise RuntimeError("bad side for protective order: {}".format(side))

    def place_reduce_only_trigger_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        trigger_px: float,
        trigger_kind: str,
        order_link_id: str,
    ) -> Dict[str, Any]:
        session = self._get_session()

        if float(qty) <= 0.0:
            raise RuntimeError("bad qty for trigger order: {}".format(qty))

        if float(trigger_px) <= 0.0:
            raise RuntimeError("bad trigger_px for trigger order: {}".format(trigger_px))

        trigger_kind_u = str(trigger_kind).upper().strip()
        dirs = self.get_close_order_side_and_trigger_directions(side)

        if trigger_kind_u == "TP":
            trigger_direction = int(dirs["tp_trigger_direction"])
        elif trigger_kind_u == "SL":
            trigger_direction = int(dirs["sl_trigger_direction"])
        else:
            raise RuntimeError("bad trigger_kind: {}. Use TP or SL".format(trigger_kind))

        resp = session.place_order(
            category=self.category,
            symbol=str(symbol).upper(),
            side=str(dirs["close_side"]),
            orderType="Market",
            triggerPrice=str(trigger_px),
            triggerDirection=trigger_direction,
            qty=format_bybit_decimal(qty),
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
        sl_resp = self.place_reduce_only_trigger_market_order(
            symbol=symbol,
            side=side,
            qty=qty,
            trigger_px=sl_px,
            trigger_kind="SL",
            order_link_id=sl_order_link_id,
        )

        tp_resp = self.place_reduce_only_trigger_market_order(
            symbol=symbol,
            side=side,
            qty=qty,
            trigger_px=tp_px,
            trigger_kind="TP",
            order_link_id=tp_order_link_id,
        )

        return {
            "take_profit": tp_resp,
            "stop_loss": sl_resp,
        }

    def place_partial_final_tp_and_sl_orders(
        self,
        symbol: str,
        side: str,
        total_qty: float,
        partial_tp_qty: float,
        final_tp_qty: float,
        sl_qty: float,
        partial_tp_px: float,
        final_tp_px: float,
        sl_px: float,
        partial_tp_order_link_id: str,
        final_tp_order_link_id: str,
        sl_order_link_id: str,
    ) -> Dict[str, Any]:
        if float(total_qty) <= 0.0:
            raise RuntimeError("bad total_qty: {}".format(total_qty))

        if float(partial_tp_qty) <= 0.0:
            raise RuntimeError("bad partial_tp_qty: {}".format(partial_tp_qty))

        if float(final_tp_qty) <= 0.0:
            raise RuntimeError("bad final_tp_qty: {}".format(final_tp_qty))

        if float(sl_qty) <= 0.0:
            raise RuntimeError("bad sl_qty: {}".format(sl_qty))

        sl_resp = self.place_reduce_only_trigger_market_order(
            symbol=symbol,
            side=side,
            qty=sl_qty,
            trigger_px=sl_px,
            trigger_kind="SL",
            order_link_id=sl_order_link_id,
        )

        partial_tp_resp = self.place_reduce_only_trigger_market_order(
            symbol=symbol,
            side=side,
            qty=partial_tp_qty,
            trigger_px=partial_tp_px,
            trigger_kind="TP",
            order_link_id=partial_tp_order_link_id,
        )

        final_tp_resp = self.place_reduce_only_trigger_market_order(
            symbol=symbol,
            side=side,
            qty=final_tp_qty,
            trigger_px=final_tp_px,
            trigger_kind="TP",
            order_link_id=final_tp_order_link_id,
        )

        return {
            "partial_take_profit": partial_tp_resp,
            "final_take_profit": final_tp_resp,
            "stop_loss": sl_resp,
        }

    def place_rest_stop_after_partial_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        rest_stop_px: float,
        order_link_id: str,
    ) -> Dict[str, Any]:
        return self.place_reduce_only_trigger_market_order(
            symbol=symbol,
            side=side,
            qty=qty,
            trigger_px=rest_stop_px,
            trigger_kind="SL",
            order_link_id=order_link_id,
        )

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
        kwargs = {"category": self.category, "symbol": str(symbol).upper()}

        if order_id:
            kwargs["orderId"] = order_id
        if order_link_id:
            kwargs["orderLinkId"] = order_link_id

        resp = session.cancel_order(**kwargs)
        self._raise_if_bad(resp)
        return resp

    def cancel_order_safe(self, symbol: str, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            return self.cancel_order(
                symbol=symbol,
                order_id=order_id,
                order_link_id=order_link_id,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "symbol": str(symbol).upper(),
                "order_id": order_id,
                "order_link_id": order_link_id,
            }
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        session = self._get_session()

        kwargs: Dict[str, Any] = {
            "category": self.category,
            "openOnly": 0,
        }

        if symbol:
            kwargs["symbol"] = str(symbol).upper()
        else:
            kwargs["settleCoin"] = "USDT"

        resp = session.get_open_orders(**kwargs)
        self._raise_if_bad(resp)

        rows = (resp.get("result") or {}).get("list") or []

        if not isinstance(rows, list):
            return []

        return rows

    def _raise_if_bad(self, resp: Dict[str, Any]) -> None:
        ret_code = int(resp.get("retCode", -1))
        if ret_code != 0:
            raise RuntimeError("Bybit error retCode={} retMsg={} resp={}".format(
                resp.get("retCode"),
                resp.get("retMsg"),
                resp,
            ))
