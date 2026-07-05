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


def directional_return(
    side: str,
    entry_price: float,
    exit_price: float,
) -> float:
    side_u = str(side).upper()
    entry = float(entry_price)
    exit_value = float(exit_price)

    if entry <= 0.0:
        raise RuntimeError("bad entry_price: {}".format(entry_price))

    if exit_value <= 0.0:
        raise RuntimeError("bad exit_price: {}".format(exit_price))

    if side_u == "LONG":
        return float((exit_value / entry) - 1.0)

    if side_u == "SHORT":
        return float((entry / exit_value) - 1.0)

    raise RuntimeError("bad side: {}".format(side))


def calc_risk_capped_position_notional(
    trade_capital_usdt: float,
    base_position_notional_usdt: float,
    trading_leverage: float,
    side: str,
    entry_price: float,
    main_sl_price: float,
    max_full_sl_capital_risk: float,
    enabled: bool = True,
    include_round_trip_cost: bool = True,
    fee_side: float = 0.0,
    slippage_side: float = 0.0,
) -> Dict[str, object]:
    capital = float(trade_capital_usdt)
    raw_notional = float(base_position_notional_usdt)
    leverage = float(trading_leverage)
    max_risk = float(max_full_sl_capital_risk)

    if capital <= 0.0:
        raise RuntimeError("trade_capital_usdt must be positive for risk capped sizing.")
    if raw_notional <= 0.0:
        raise RuntimeError("base_position_notional_usdt must be positive for risk capped sizing.")
    if leverage <= 0.0:
        raise RuntimeError("trading_leverage must be positive for risk capped sizing.")
    if max_risk <= 0.0:
        raise RuntimeError("max_full_sl_capital_risk must be positive for risk capped sizing.")

    gross_ret = directional_return(side=side, entry_price=entry_price, exit_price=main_sl_price)

    round_trip_cost = 0.0
    if bool(include_round_trip_cost):
        round_trip_cost = 2.0 * float(fee_side) + 2.0 * float(slippage_side)

    net_ret = float(gross_ret - round_trip_cost)
    full_main_sl_risk_abs = abs(min(net_ret, 0.0))

    # ВАЖНО:
    # risk_cap_base_usdt — это торговая база, то есть уже увеличенный notional
    # после POSITION_NOTIONAL_MULTIPLIER / trading leverage logic.
    # Поэтому max_full_sl_capital_risk=0.06 означает:
    # полный MAIN_SL <= 6% от торговой базы, а не от чистого wallet-capital.
    risk_cap_base_usdt = raw_notional
    max_loss_usdt = risk_cap_base_usdt * max_risk

    capped_notional = raw_notional
    applied = False

    if bool(enabled) and full_main_sl_risk_abs > 0.0:
        allowed_notional = max_loss_usdt / full_main_sl_risk_abs
        if allowed_notional < raw_notional:
            capped_notional = allowed_notional
            applied = True

    position_fraction = capped_notional / raw_notional if raw_notional > 0.0 else 0.0

    raw_position_base_risk_abs = (
        raw_notional * full_main_sl_risk_abs / risk_cap_base_usdt
        if risk_cap_base_usdt > 0.0
        else 0.0
    )
    capped_position_base_risk_abs = (
        capped_notional * full_main_sl_risk_abs / risk_cap_base_usdt
        if risk_cap_base_usdt > 0.0
        else 0.0
    )

    return {
        "position_risk_cap_enabled": bool(enabled),
        "position_risk_cap_applied": bool(applied),
        "position_fraction": float(position_fraction),
        "raw_position_notional_usdt": float(raw_notional),
        "risk_capped_position_notional_usdt": float(capped_notional),
        "position_notional_usdt": float(capped_notional),
        "risk_cap_base_mode": "trade_base_notional",
        "risk_cap_base_usdt": float(risk_cap_base_usdt),
        "max_full_sl_capital_risk": float(max_risk),
        "full_main_sl_gross_ret": float(gross_ret),
        "full_main_sl_net_ret": float(net_ret),
        "full_main_sl_capital_risk_abs": float(full_main_sl_risk_abs),
        "raw_position_capital_risk_abs": float(raw_position_base_risk_abs),
        "capped_position_capital_risk_abs": float(capped_position_base_risk_abs),
        "estimated_initial_margin_usdt": float(capped_notional / leverage),
    }
