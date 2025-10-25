# data_feed/ob_snapshot.py
import os, sys, argparse, requests, pandas as pd
from datetime import datetime, timezone
from typing import List, Tuple
from dotenv import load_dotenv

def _load_env():
    load_dotenv()
    if os.path.exists(".envrc"):
        with open(".envrc","r",encoding="utf-8") as f:
            for ln in f:
                ln=ln.strip()
                if not ln or ln.startswith("#"): continue
                if ln.startswith("export "): ln=ln[7:]
                if "=" in ln:
                    k,v = ln.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def _now_minute_ts_ms() -> int:
    dt = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc)
    return int(dt.timestamp()*1000)

def _get_symbols(arg_symbols: str) -> List[str]:
    if arg_symbols:
        return [s.strip().upper() for s in arg_symbols.split(",") if s.strip()]
    env = os.getenv("TRADE_UNIVERSE","").strip()
    if env:
        return [s.strip().upper() for s in env.split(",") if s.strip()]
    try:
        sys.path.insert(0, os.getcwd())
        from config import TRADE_UNIVERSE
        return [s.strip().upper() for s in (TRADE_UNIVERSE or [])]
    except Exception:
        return []

def _orderbook(category: str, symbol: str, depth: int, timeout: float = 10.0):
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {"category": category, "symbol": symbol, "limit": str(int(depth))}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if int(js.get("retCode",-1)) != 0:
        return [], [], None
    res = (js.get("result") or {})
    bids = res.get("b") or res.get("bids") or []
    asks = res.get("a") or res.get("asks") or []
    t    = res.get("ts")
    return bids, asks, t

def _to_float_pairs(rows) -> List[Tuple[float,float]]:
    out=[]
    for r in rows:
        try:
            # формат в v5: ["price","size","..."]; берём 0 и 1
            px = float(r[0]); sz = float(r[1])
            out.append((px,sz))
        except Exception:
            continue
    return out

def _vwap(pairs: List[Tuple[float,float]]):
    s_px_sz = sum(p*sz for p,sz in pairs)
    s_sz    = sum(sz for _,sz in pairs)
    return (s_px_sz/s_sz) if s_sz>0 else None, s_sz

def _append_row_parquet(path: str, row: dict, key_cols=("ts",)):
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df = pd.concat([df, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=list(key_cols), keep="last")
    else:
        df = df_new
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols")
    ap.add_argument("--out", default="./data/meta1m")
    ap.add_argument("--category", default=(os.getenv("BYBIT_CATEGORY") or "linear").lower(), choices=["linear","spot"])
    ap.add_argument("--depth", type=int, default=50, help="сколько уровней стакана использовать")
    args = ap.parse_args()

    syms = _get_symbols(args.symbols or "")
    if not syms:
        print("WARN: пустой список символов (TRADE_UNIVERSE не задан)."); return

    ts_min = _now_minute_ts_ms()
    for s in syms:
        try:
            bids_raw, asks_raw, ts_ob = _orderbook(args.category, s, args.depth)
            bids = _to_float_pairs(bids_raw)
            asks = _to_float_pairs(asks_raw)
            bid_vwap, bid_vol = _vwap(bids)
            ask_vwap, ask_vol = _vwap(asks)
            spread_bp = None
            if bid_vwap and ask_vwap and ask_vwap>0:
                spread_bp = (ask_vwap - bid_vwap) / ask_vwap * 1e4  # basis points
            imb = None
            if (bid_vol or 0)>0 or (ask_vol or 0)>0:
                imb = (bid_vol - ask_vol) / max(bid_vol + ask_vol, 1e-12)

            row = {
                "ts": ts_min, "bid_vol": bid_vol or 0.0, "ask_vol": ask_vol or 0.0,
                "imbalance": imb, "bid_vwap": bid_vwap, "ask_vwap": ask_vwap,
                "spread_bp": spread_bp, "symbol": s
            }
            _append_row_parquet(os.path.join(args.out, f"{s}_ob_1m.parquet"), row)
            print(f"[OB] {s} depth={args.depth} imb={imb}", flush=True)
        except Exception as e:
            print(f"[OB_ERR] {s} {e}", flush=True)

if __name__ == "__main__":
    main()