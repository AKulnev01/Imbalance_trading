import os, time, json, math, argparse, requests
import pandas as pd
from datetime import datetime, timezone

API = "https://api.bybit.com"

def _ensure_dir(p): os.makedirs(p, exist_ok=True)
def _ms(dt): return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def _session(retries=5, backoff=0.5, connect=6, read=15):
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff,
                  status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    ad = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", ad); s.mount("http://", ad)
    s.request_timeout=(connect, read)
    return s

def _fetch_recent_trades(sess, category, symbol, limit=1000):
    url = f"{API}/v5/market/recent-trade"
    r = sess.get(url, params={"category": category, "symbol": symbol, "limit": str(limit)}, timeout=sess.request_timeout)
    r.raise_for_status()
    js = r.json()
    if js.get("retCode") != 0: return []
    rows = js.get("result", {}).get("list") or []
    out=[]
    # формат v5: [{"execId","symbol","price","size","side","time","isBlockTrade"}] — на некоторых рынках упрощённый массив
    for it in rows:
        if isinstance(it, dict):
            ts = int(it.get("time") or 0)
            px = float(it.get("price") or 0)
            sz = float(it.get("size") or 0)
            sd = (it.get("side") or "").lower()  # Buy/Sell
        else:
            # массив-порядок: [execId, price, size, side, time, isBlockTrade]
            ts = int(it[4]); px=float(it[1]); sz=float(it[2]); sd=str(it[3]).lower()
        out.append({"ts": ts, "price": px, "size": sz, "side": sd})
    return out

def _minute_bucket(ts_ms): return ts_ms - (ts_ms % 60000)

def _aggregate_minute(trades):
    if not trades: return pd.DataFrame(columns=["ts","trades","vwap","vol_buy","vol_sell","delta","last"])
    df = pd.DataFrame(trades)
    df["bucket"] = df["ts"].apply(_minute_bucket)
    # объёмы в базовой валюте; vwap по всем тикам
    g = df.groupby("bucket", as_index=False)
    vwap = (g.apply(lambda x: (x["price"] * x["size"]).sum() / max(x["size"].sum(), 1e-12))).reset_index()
    vwap.columns=["idx","vwap"]
    agg = g.agg(trades=("ts","count"),
                vol_buy=("size", lambda s: float(s[df.loc[s.index,"side"].eq("buy")].sum())),
                vol_sell=("size", lambda s: float(s[df.loc[s.index,"side"].eq("sell")].sum())),
                last=("price","last"))
    agg["delta"] = agg["vol_buy"] - agg["vol_sell"]
    agg = agg.merge(vwap, left_index=True, right_on="idx", how="left").drop(columns=["idx"])
    agg.rename(columns={"bucket":"ts"}, inplace=True)
    return agg[["ts","trades","vwap","vol_buy","vol_sell","delta","last"]].sort_values("ts")

def _merge_save(path, df_new):
    if df_new is None or df_new.empty: return
    if os.path.exists(path):
        try: df_old = pd.read_parquet(path)
        except Exception: df_old = pd.DataFrame(columns=df_new.columns)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    else:
        df = df_new.reset_index(drop=True)
    df.to_parquet(path, index=False)

def fetch_trades_1m(symbols, category, out_dir, retries=5, backoff=0.5, connect=6, read=15, sleep=0.3):
    s = _session(retries, backoff, connect, read)
    _ensure_dir(out_dir)
    for sym in symbols:
        try:
            rows = _fetch_recent_trades(s, category, sym, limit=1000)
            if not rows:
                print(f"[TRADES] {sym} empty"); continue
            df_min = _aggregate_minute(rows)
            _merge_save(os.path.join(out_dir, f"{sym}_trades_1m.parquet"), df_min)
            print(f"[TRADES] {sym} +{len(df_min)}")
            time.sleep(sleep)
        except Exception as e:
            print(f"[TRADES_ERR] {sym} {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--category", default=None)
    ap.add_argument("--out", default="./data/meta1m")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=0.5)
    ap.add_argument("--connect-timeout", type=float, default=6.0)
    ap.add_argument("--read-timeout", type=float, default=15.0)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    try:
        from config import TRADE_UNIVERSE, filter_universe, BYBIT_CATEGORY
    except Exception:
        TRADE_UNIVERSE, BYBIT_CATEGORY, filter_universe = [], "spot", lambda x: x
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or filter_universe(TRADE_UNIVERSE or [])
    category = (args.category or BYBIT_CATEGORY or "spot").lower()

    fetch_trades_1m(symbols, category, args.out, args.retries, args.backoff, args.connect_timeout, args.read_timeout, args.sleep)