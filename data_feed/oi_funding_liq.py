# data_feed/oi_funding_liq.py
import os, sys, argparse, requests, pandas as pd
from datetime import datetime, timezone
from typing import List, Dict
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

def _tickers(category: str, symbols: List[str], timeout: float = 10.0) -> Dict[str, Dict]:
    """
    /v5/market/tickers — для linear возвращает openInterest, fundingRate (для spot — нет).
    """
    url = "https://api.bybit.com/v5/market/tickers"
    out: Dict[str, Dict] = {}
    sess = requests.Session()
    for s in symbols:
        try:
            r = sess.get(url, params={"category": category, "symbol": s}, timeout=timeout)
            r.raise_for_status()
            js = r.json()
            if int(js.get("retCode",-1)) != 0:
                out[s] = {}
                continue
            items = (js.get("result") or {}).get("list") or []
            out[s] = items[0] if items else {}
        except Exception:
            out[s] = {}
    return out

def _liquidations(category: str, symbol: str, limit: int = 200, timeout: float = 10.0) -> List[Dict]:
    """
    /v5/market/liquidation — недокументированно стабилен, но у Bybit есть.
    Если реткода !=0 — просто вернём пусто, не валим процесс.
    """
    url = "https://api.bybit.com/v5/market/liquidation"
    params = {"category": category, "symbol": symbol, "limit": str(int(limit))}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        if int(js.get("retCode",-1)) != 0:
            return []
        return (js.get("result") or {}).get("list") or []
    except Exception:
        return []

def _append_row_parquet(path: str, row: Dict, key_cols=("ts",)):
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df = pd.concat([df, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=list(key_cols), keep="last")
    else:
        df = df_new
    os.makedirs(os.path.dirname(path), exist_ok=True
    )
    df.to_parquet(path, index=False)

def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols")
    ap.add_argument("--out", default="./data/meta1m")
    ap.add_argument("--category", default=(os.getenv("BYBIT_CATEGORY") or "linear").lower(), choices=["linear","spot"])
    args = ap.parse_args()

    syms = _get_symbols(args.symbols or "")
    if not syms:
        print("WARN: пустой список символов (TRADE_UNIVERSE не задан)."); return

    ts_min = _now_minute_ts_ms()

    # --- OI & Funding (только для linear; для spot вернутся пустые поля и это ок) ---
    tick = _tickers(args.category, syms)

    for s in syms:
        try:
            t = tick.get(s, {}) or {}
            # В tickers поля строковые
            oi   = t.get("openInterest");    oi   = float(oi) if oi not in (None,"") else None
            fund = t.get("fundingRate");     fund = float(fund) if fund not in (None,"") else None
            row = {"ts": ts_min, "open_interest": oi, "funding_rate": fund, "symbol": s}
            _append_row_parquet(os.path.join(args.out, f"{s}_oi_funding_1m.parquet"), row)
            print(f"[OI/FUND] {s} oi={oi} fund={fund}", flush=True)
        except Exception as e:
            print(f"[OI/FUND_ERR] {s} {e}", flush=True)

    # --- Liquidations (если эндпоинт вернёт пусто — ничего страшного) ---
    for s in syms:
        try:
            liqs = _liquidations(args.category, s, limit=200)
            # агрегируем одноминутно (здесь минута одна — текущая)
            liq_buy_sz = 0.0
            liq_sell_sz = 0.0
            liq_px_sum = 0.0
            liq_px_cnt = 0
            for ev in liqs:
                side = str(ev.get("side") or ev.get("S") or "").lower()
                sz   = ev.get("size") or ev.get("qty") or ev.get("execQty")
                px   = ev.get("price") or ev.get("execPrice")
                try:
                    sz = float(sz) if sz is not None else 0.0
                    px = float(px) if px is not None else None
                except Exception:
                    continue
                if side == "buy":  liq_buy_sz  += sz
                elif side == "sell": liq_sell_sz += sz
                if px is not None:
                    liq_px_sum += px; liq_px_cnt += 1

            liq_avg_px = (liq_px_sum/liq_px_cnt) if liq_px_cnt>0 else None
            row = {"ts": ts_min, "liq_buy_sz": liq_buy_sz, "liq_sell_sz": liq_sell_sz,
                   "liq_count": len(liqs), "liq_avg_px": liq_avg_px, "symbol": s}
            _append_row_parquet(os.path.join(args.out, f"{s}_liquidations.parquet"), row)
            print(f"[LIQ] {s} cnt={len(liqs)}", flush=True)
        except Exception as e:
            print(f"[LIQ_ERR] {s} {e}", flush=True)

if __name__ == "__main__":
    main()