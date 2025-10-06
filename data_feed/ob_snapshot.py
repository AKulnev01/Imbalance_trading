import os, time, json, argparse, requests
import pandas as pd

API = "https://api.bybit.com"

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _session(retries=5, backoff=0.5, connect=6, read=10):
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff,
                  status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    ad = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", ad); s.mount("http://", ad)
    s.request_timeout=(connect, read)
    return s

def _fetch_orderbook(sess, category, symbol, depth=50):
    url = f"{API}/v5/market/orderbook"
    r = sess.get(url, params={"category": category, "symbol": symbol, "limit": str(depth)}, timeout=sess.request_timeout)
    r.raise_for_status()
    js = r.json()
    if js.get("retCode") != 0: return None
    res = js.get("result", {}) or {}
    a = res.get("a") or res.get("asks") or []  # [["px","sz"],...]
    b = res.get("b") or res.get("bids") or []
    t = int(res.get("ts") or 0)
    def _agg(levels):
        vol = 0.0; notional = 0.0
        for px, sz in levels:
            px=float(px); sz=float(sz)
            vol += sz; notional += px*sz
        vwap = (notional / vol) if vol>0 else 0.0
        return vol, vwap
    bid_vol, bid_vwap = _agg(b)
    ask_vol, ask_vwap = _agg(a)
    spread_bp = 0.0
    if bid_vwap>0 and ask_vwap>0:
        mid = (bid_vwap + ask_vwap)/2.0
        spread_bp = (ask_vwap - bid_vwap)/mid * 1e4
    imb = (bid_vol - ask_vol) / max(bid_vol + ask_vol, 1e-9)
    return {"ts": t, "bid_vol": bid_vol, "ask_vol": ask_vol, "imbalance": imb, "bid_vwap": bid_vwap, "ask_vwap": ask_vwap, "spread_bp": spread_bp}

def _merge_save(path, row):
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        try: df_old = pd.read_parquet(path)
        except Exception: df_old = pd.DataFrame(columns=df_new.columns)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    else:
        df = df_new
    df.to_parquet(path, index=False)

def collect_ob_snapshot(symbols, category, out_dir, depth=50, retries=5, backoff=0.5, connect=6, read=10, sleep=0.3):
    s = _session(retries, backoff, connect, read)
    _ensure_dir(out_dir)
    for sym in symbols:
        try:
            row = _fetch_orderbook(s, category, sym, depth)
            if row:
                _merge_save(os.path.join(out_dir, f"{sym}_ob_1m.parquet"), row)
                print(f"[OB] {sym} +1")
            else:
                print(f"[OB] {sym} empty")
            time.sleep(sleep)
        except Exception as e:
            print(f"[OB_ERR] {sym} {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--category", default=None)
    ap.add_argument("--out", default="./data/meta1m")
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=0.5)
    ap.add_argument("--connect-timeout", type=float, default=6.0)
    ap.add_argument("--read-timeout", type=float, default=10.0)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    try:
        from config import TRADE_UNIVERSE, filter_universe, BYBIT_CATEGORY
    except Exception:
        TRADE_UNIVERSE, BYBIT_CATEGORY, filter_universe = [], "spot", lambda x:x
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or filter_universe(TRADE_UNIVERSE or [])
    category = (args.category or BYBIT_CATEGORY or "spot").lower()

    collect_ob_snapshot(symbols, category, args.out, args.depth, args.retries, args.backoff, args.connect_timeout, args.read_timeout, args.sleep)