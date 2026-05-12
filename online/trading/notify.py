from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

import pandas as pd

from online.trading import config


TG_BOT_TOKEN_ENV = "IMB_TG_BOT_TOKEN"
TG_CHAT_ID_ENV = "IMB_TG_CHAT_ID"

DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_MIN_INTERVAL_SECONDS = 30


def telegram_enabled() -> bool:
    return bool(os.environ.get(TG_BOT_TOKEN_ENV, "").strip()) and bool(os.environ.get(TG_CHAT_ID_ENV, "").strip())


def compact_value(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    if isinstance(value, float):
        return "{:.6g}".format(value)
    return str(value)


def send_telegram_message(
    text: str,
    disable_notification: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    token = os.environ.get(TG_BOT_TOKEN_ENV, "").strip()
    chat_id = os.environ.get(TG_CHAT_ID_ENV, "").strip()

    if not token or not chat_id:
        return False

    url = "https://api.telegram.org/bot{}/sendMessage".format(token)

    payload = {
        "chat_id": chat_id,
        "text": text[:3900],
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
        "disable_notification": "true" if disable_notification else "false",
    }

    data = urllib.parse.urlencode(payload).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        return bool(parsed.get("ok"))
    except Exception as exc:
        print("TELEGRAM_SEND_FAILED:", type(exc).__name__, exc)
        return False


def build_event_message(
    event_type: str,
    status: str = "",
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    signal_key: Optional[str] = None,
    trade_id: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    payload = payload or {}

    lines = []
    lines.append("🤖 <b>IMB autotrade</b>")
    lines.append("<b>{}</b>{}".format(event_type, " / " + status if status else ""))

    if symbol or side:
        lines.append("symbol: <code>{}</code> {}".format(compact_value(symbol), compact_value(side)))

    if trade_id is not None:
        lines.append("trade_id: <code>{}</code>".format(trade_id))

    if signal_key:
        lines.append("signal: <code>{}</code>".format(str(signal_key)[:120]))

    for k in [
        "entry_px_plan",
        "entry_avg_px",
        "qty",
        "tp_px",
        "sl_px",
        "available_usdt",
        "gate2",
        "gate4",
        "gate5_1",
        "gate5_3",
        "delay_seconds",
        "slippage_pct",
        "net_ret",
        "pnl_usd",
    ]:
        if k in payload:
            lines.append("{}: <code>{}</code>".format(k, compact_value(payload.get(k))))

    return "\n".join(lines)


def notify_event(
    event_type: str,
    status: str = "",
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    signal_key: Optional[str] = None,
    trade_id: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,
    force: bool = False,
    disable_notification: bool = False,
) -> bool:
    if not telegram_enabled():
        return False

    if not force and not periodic_gate(event_type):
        return False

    msg = build_event_message(
        event_type=event_type,
        status=status,
        symbol=symbol,
        side=side,
        signal_key=signal_key,
        trade_id=trade_id,
        payload=payload,
    )
    return send_telegram_message(msg, disable_notification=disable_notification)


def periodic_gate(event_type: str) -> bool:
    """
    Простейший локальный антиспам.
    Критические события отправляются всегда.
    Повторяющиеся heartbeat/status — не чаще IMB_TG_MIN_INTERVAL_SECONDS.
    """
    critical_prefixes = (
        "ENTRY",
        "TP",
        "SL",
        "TTL",
        "EMERGENCY",
        "MANUAL",
        "ERROR",
        "REJECT",
        "DRY_RUN_ENTRY_PLANNED",
    )

    if event_type.startswith(critical_prefixes):
        return True

    min_interval = int(os.environ.get("IMB_TG_MIN_INTERVAL_SECONDS", str(DEFAULT_MIN_INTERVAL_SECONDS)))
    stamp_path = config.ROOT / "online" / "trading" / ".telegram_last_{}.txt".format(safe_name(event_type))

    now = time.time()

    try:
        if stamp_path.exists():
            last = float(stamp_path.read_text(encoding="utf-8").strip() or "0")
            if now - last < min_interval:
                return False
        stamp_path.write_text(str(now), encoding="utf-8")
        return True
    except Exception:
        return True


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(value))


def main() -> None:
    ok = notify_event(
        event_type="TELEGRAM_TEST",
        status="OK",
        payload={
            "message": "test",
        },
        force=True,
    )
    print("TELEGRAM_ENABLED:", telegram_enabled())
    print("TELEGRAM_SENT:", ok)


if __name__ == "__main__":
    main()
