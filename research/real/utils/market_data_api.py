import asyncio
import aiohttp
import pandas as pd
import time
from typing import List, Dict

HTTP_BASE_MAIN = "https://api.bybit.com"
HTTP_BASE_TEST = "https://api-testnet.bybit.com"

def _http_base(use_mainnet: bool) -> str:
    return HTTP_BASE_MAIN if use_mainnet else HTTP_BASE_TEST

async def _fetch(session, url, params, *, attempts=3, backoff=0.6):
    for k in range(attempts):
        try:
            async with session.get(url, params=params) as r:
                js = await r.json(content_type=None)
                if int(js.get("retCode", -1)) == 0:
                    return js
                raise RuntimeError(js.get("retMsg"))
        except Exception:
            if k + 1 == attempts:
                raise
            await asyncio.sleep(backoff * (k + 1))

async def fetch_kline(
    symbols: List[str], *,
    interval: str,
    category: str = "linear",
    lookback_minutes: int = 24 * 60,
    use_mainnet: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Возвращает dict: symbol -> df(time, open, high, low, close, volume)
    """
    base = _http_base(use_mainnet)
    url = base + "/v5/market/kline"
    out = {}
    end = int(time.time() * 1000)
    start = end - lookback_minutes * 60 * 1000
    timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for sym in symbols:
            params = {
                "category": category, "symbol": sym,
                "interval": interval, "start": start, "end": end, "limit": 1000
            }
            try:
                js = await _fetch(session, url, params)
            except Exception:
                out[sym] = pd.DataFrame()
                continue

            rows = (js.get("result", {}) or {}).get("list", []) or []
            if not rows:
                out[sym] = pd.DataFrame()
                continue
            rows.sort(key=lambda x: int(x[0]))
            rec = [
                {
                    "time": pd.to_datetime(int(s), unit="ms", utc=True),
                    "open": float(o), "high": float(h),
                    "low": float(l), "close": float(c),
                    "volume": float(vol)
                }
                for s, o, h, l, c, vol, turnover in rows
            ]
            out[sym] = pd.DataFrame(rec).set_index("time").sort_index()
    return out