import os, time, argparse, requests, pandas as pd

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

def _merge_save(path, df_new):
    if df_new is None or df_new.empty: return
    if os.path.exists(path):
        try: df_old = pd.read_parquet(path)
        except Exception: df_old = pd.DataFrame(columns=df_new.columns)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=[c for c in df.columns if c!="value"]).sort_values(df.columns[0]).reset_index(drop=True)
    else:
        df = df_new.reset_index(drop=True)
    df.to_parquet(path, index=False)

def _ticker_snapshot(sess, category, symbol):
    # содержит openInterest для linear
    url = f"{API}/v5/market/tickers"
    r = sess.get(url, params={"category": category, "symbol": symbol}, timeout=sess.request_timeout)
    r.raise_for_status()
    js = r.json()
    if js.get("retCode") != 0: return None
    lst = (js.get("result", {}) or {}).get("list") or []
    if not lst: return None
    it = lst[0]
    ts = int(it.get("ts") or 0)
    oi = float(it.get("openInterest") or 0.0)
    funding = float(it.get("fundingRate") or 0.0) if "fundingRate" in it else None
    return {"ts": ts, "open_interest": oi, "funding_rate": funding if funding is not None else 0.0}

def _funding_history(sess, category, symbol, limit=200):
    url = f"{API}/v5/market/funding/history"
    r = sess.get(url, params={"category": category, "symbol": symbol, "limit": str(limit)}, timeout=sess.request_timeout)
    r.raise_for_status()
    js = r.json()
    if js.get("retCode") != 0: return pd.DataFrame()
    rows = (js.get("result", {}) or {}).get("list") or []
    out=[]
    for it in rows:
        # поля: fundingRate, fundingRateTimestamp
        ts = int(it.get("fundingRateTimestamp") or 0)
        fr = float(it.get("fundingRate") or 0.0)
        out.append({"ts": ts, "funding_rate_hist": fr})
    return pd.DataFrame(out).sort_values("ts")

def _liquidations(sess, category, symbol, limit=200):
    url = f"{API}/v5/market/liquidation"
    r = sess.get(url, params={"category": category, "symbol": symbol, "limit": str(limit)}, timeout=sess.request_timeout)
    r.raise_for_status()
    js = r.json()
    if js.get("retCode") != 0: return pd.DataFrame()
    rows = (js.get("result", {}) or {}).get("list") or []
    out=[]
    for it in rows:
        # price, size, side, updatedTime
        ts = int(it.get("updatedTime") or 0)
        px = float(it.get("price") or 0.0)
        sz = float(it.get("size") or 0.0)
        sd = (it.get("side") or "").lower()
        out.append({"ts": ts, "liq_price": px, "liq_size": sz, "liq_side": sd})
    return pd.DataFrame(out).sort_values("ts")

def collect_meta(symbols, category, out_dir, retries=5, backoff=0.5, connect=6, read=10, sleep=0.3):
    s = _session(retries, backoff, connect, read)
    for sym in symbols:
        try:
            snap = _ticker_snapshot(s, category, sym)
            if snap:
                _merge_save(os.path.join(out_dir, f"{sym}_oi_funding_1m.parquet"), pd.DataFrame([snap]))
                print(f"[TICKER] {sym} +1")

            fr = _funding_history(s, category, sym, limit=200)
            if not fr.empty:
                _merge_save(os.path.join(out_dir, f"{sym}_funding_hist.parquet"), fr)
                print(f"[FUND] {sym} +{len(fr)}")

            liq = _liquidations(s, category, sym, limit=200)
            if not liq.empty:
                _merge_save(os.path.join(out_dir, f"{sym}_liquidations.parquet"), liq)
                print(f"[LIQ] {sym} +{len(liq)}")

            time.sleep(sleep)
        except Exception as e:
            print(f"[META_ERR] {sym} {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--category", default=None)
    ap.add_argument("--out", default="./data/meta1m")
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

    _ensure_dir(args.out)
    collect_meta(symbols, category, args.out, args.retries, args.backoff, args.connect_timeout, args.read_timeout, args.sleep)