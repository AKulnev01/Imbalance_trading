# data_feed/trades_agg.py
import os, sys, argparse, requests, pandas as pd
from datetime import datetime, timezone
from typing import Optional, List, Dict
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
    # метка текущей минуты (UTC) в ms
    dt = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc)
    return int(dt.timestamp()*1000)

def _get_symbols(arg_symbols: Optional[str]) -> List[str]:
    if arg_symbols:
        return [s.strip().upper() for s in arg_symbols.split(",") if s.strip()]
    # из окружения
    env = os.getenv("TRADE_UNIVERSE","").strip()
    if env:
        return [s.strip().upper() for s in env.split(",") if s.strip()]
    # из config.py
    try:
        sys.path.insert(0, os.getcwd())
        from config import TRADE_UNIVERSE
        return [s.strip().upper() for s in (TRADE_UNIVERSE or [])]
    except Exception:
        return []

def _recent_trades(category: str, symbol: str, limit: int = 1000, timeout: float = 10.0) -> List[Dict]:
    url = "https://api.bybit.com/v5/market/recent-trade"
    params = {"category": category, "symbol": symbol, "limit": str(int(limit))}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if int(js.get("retCode", -1)) != 0:
        # не валим — просто пусто
        return []
    return (js.get("result", {}) or {}).get("list", []) or []

def _append_row_parquet(path: str, row: Dict, key_cols=("ts",)):
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df = pd.concat([df, df_new], ignore_index=True)
        # дедуп по ключам
        df = df.drop_duplicates(subset=list(key_cols), keep="last")
    else:
        df = df_new
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="список через запятую; иначе TRADE_UNIVERSE/config.py")
    ap.add_argument("--out", default="./data/meta1m")
    ap.add_argument("--category", default=(os.getenv("BYBIT_CATEGORY") or "linear").lower(), choices=["linear","spot"])
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    syms = _get_symbols(args.symbols)
    if not syms:
        print("WARN: пустой список символов (TRADE_UNIVERSE не задан)."); return

    ts_min = _now_minute_ts_ms()
    for s in syms:
        try:
            trades = _recent_trades(args.category, s, limit=args.limit)
            if not trades:
                # всё равно фиксируем пустую строку для текущей минуты (удобно при роллапе)
                row = {"ts": ts_min, "trades": 0, "vwap": None, "vol_buy": 0.0, "vol_sell": 0.0, "delta": 0.0, "last": None, "symbol": s}
                _append_row_parquet(os.path.join(args.out, f"{s}_trades_1m.parquet"), row)
                print(f"[TRADES] {s} empty", flush=True)
                continue

            tot_qty = 0.0
            sum_px_qty = 0.0
            vol_buy = 0.0
            vol_sell = 0.0
            last_px = None
            for t in trades:
                try:
                    px  = float(t.get("execPrice") or t.get("price"))
                    qty = float(t.get("execQty") or t.get("size"))
                except Exception:
                    continue
                sum_px_qty += px*qty
                tot_qty    += qty
                sd = str(t.get("side") or t.get("S") or "").lower()
                if sd == "buy":  vol_buy  += qty
                elif sd == "sell": vol_sell += qty
                if last_px is None:
                    last_px = px

            vwap = (sum_px_qty / tot_qty) if tot_qty>0 else None
            delta = vol_buy - vol_sell
            row = {
                "ts": ts_min, "trades": len(trades), "vwap": vwap,
                "vol_buy": vol_buy, "vol_sell": vol_sell, "delta": delta,
                "last": last_px, "symbol": s
            }
            _append_row_parquet(os.path.join(args.out, f"{s}_trades_1m.parquet"), row)
            print(f"[TRADES] {s} trades={row['trades']} vwap={row['vwap']}", flush=True)
        except Exception as e:
            print(f"[TRADES_ERR] {s} {e}", flush=True)

if __name__ == "__main__":
    main()