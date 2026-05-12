from __future__ import annotations

import math
from typing import Dict, Optional


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return math.floor(float(value) / float(step)) * float(step)


def calc_order_qty(
    available_usdt: float,
    entry_price: float,
    qty_step: float,
    min_order_qty: float,
    min_notional: float,
    use_balance_pct: float,
) -> Dict[str, object]:
    available_usdt = float(available_usdt)
    entry_price = float(entry_price)
    use_balance_pct = float(use_balance_pct)

    if available_usdt <= 0:
        return {
            "ok": False,
            "reason": "no_available_balance",
            "qty": 0.0,
            "notional": 0.0,
        }

    if entry_price <= 0:
        return {
            "ok": False,
            "reason": "bad_entry_price",
            "qty": 0.0,
            "notional": 0.0,
        }

    alloc_usdt = available_usdt * use_balance_pct
    raw_qty = alloc_usdt / entry_price
    qty = floor_to_step(raw_qty, qty_step)
    notional = qty * entry_price

    if qty < float(min_order_qty):
        return {
            "ok": False,
            "reason": "qty_below_min_order_qty",
            "qty": qty,
            "notional": notional,
            "alloc_usdt": alloc_usdt,
        }

    if notional < float(min_notional):
        return {
            "ok": False,
            "reason": "notional_below_min_notional",
            "qty": qty,
            "notional": notional,
            "alloc_usdt": alloc_usdt,
        }

    return {
        "ok": True,
        "reason": "ok",
        "qty": qty,
        "notional": notional,
        "alloc_usdt": alloc_usdt,
    }


def calc_tp_sl_prices(
    side: str,
    entry_price: float,
    atr14: float,
    tp_atr: float,
    sl_atr: float,
) -> Dict[str, float]:
    side = str(side).upper()
    entry_price = float(entry_price)
    atr14 = float(atr14)
    tp_atr = float(tp_atr)
    sl_atr = float(sl_atr)

    if side == "LONG":
        tp_px = entry_price + atr14 * tp_atr
        sl_px = entry_price - atr14 * sl_atr
    elif side == "SHORT":
        tp_px = entry_price - atr14 * tp_atr
        sl_px = entry_price + atr14 * sl_atr
    else:
        raise RuntimeError("bad side: {}".format(side))

    return {
        "tp_px": float(tp_px),
        "sl_px": float(sl_px),
    }
