import os, math, itertools, hashlib
from pathlib import Path
from typing import List
from datetime import datetime, timezone
import pandas as pd

# локальные импорты проекта
from utils.detect_fvg import detect_fvg_imbalances
from utils.strategy import select_entry_price, get_klines_4h, _sanitize_ohlcv
from config import TRADE_UNIVERSE, DEFAULT_TTL_DAYS

# === жёстко: работаем только от локальных данных ===
os.environ.setdefault("USE_LOCAL_MINUTES", "1")
os.environ["USE_LOCAL_4H"] = "0"
os.environ.setdefault("LTF_ROOT", "./data/m1")
os.environ["DISABLE_MINUTE_FALLBACK"] = "0"  # 1m разрешены
os.environ["MINUTE_EXIT_FOR_SINGLE"]  = "1"

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = _sanitize_ohlcv(df)
    return df

def _detect_for_symbol(df4h: pd.DataFrame, symbol: str,
                       min_strength: float, vol_mult: float,
                       tol_pct: float, max_days_to_fill: int) -> List[dict]:
    imbs = detect_fvg_imbalances(
        df4h,
        volume_multiplier=float(vol_mult),
        tolerance_pct=float(tol_pct),
        min_strength_pct=float(min_strength),
        max_days_to_fill=int(max_days_to_fill),
    ) or []
    rows = []
    for imb in imbs:
        side = str(imb.get("type","")).upper()
        if side not in ("BUY","SELL"):
            continue
        entry = select_entry_price(df4h, symbol, imb)
        if entry is None or entry <= 0:
            continue
        t0 = pd.to_datetime(imb["time"], utc=True, errors="coerce")
        rows.append({
            "symbol":   str(symbol).upper(),
            "imb_time": t0,
            "type":     side,
            "entry":    float(entry),
            "stop":     pd.NA,
            "tp":       pd.NA,
            "strength": float(imb.get("strength", 0.0)),
            "touched":  bool(imb.get("touched", False)),
        })
    return rows

def _parse_floats_csv(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]

def main(
    symbols: List[str] = None,
    interval: str = "4h",
    lookback_days: int = 360,
    grid_strength: str = "3.0",                # пример: "2.0,2.5,3.0"
    grid_vol: str = "1.5",                     # пример: "1.1,1.3,1.5"
    grid_tol: str = "0.1",                     # долями: "0,0.0005,0.001"
    max_fill_days: int = 30,
    out_dir: str = "./data/signals",
    tag: str = None,                           # суффикс в имени файлов (например, "mix1")
):
    symbols = [s.strip().upper() for s in (symbols or TRADE_UNIVERSE)]
    out_root = Path(out_dir); out_root.mkdir(parents=True, exist_ok=True)

    # подгрузим 4h локально (через нашу функцию, она сама выберет источник)
    cache = {}
    for s in symbols:
        try:
            df4h = get_klines_4h(symbol=s, lookback_days=lookback_days, interval=interval)
            cache[s] = _ensure_dt_index(df4h)
        except Exception as e:
            print(f"⚠️ skip {s}: {e}")
            cache[s] = pd.DataFrame()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    strengths = _parse_floats_csv(grid_strength)
    vols      = _parse_floats_csv(grid_vol)
    tols      = _parse_floats_csv(grid_tol)

    manifest = []
    combos = list(itertools.product(strengths, vols, tols))

    print(f"🔧 combos: {len(combos)}  symbols: {len(symbols)}  lookback: {lookback_days}d  interval: {interval}")

    for (ms, vm, tol) in combos:
        # стабильное и компактное имя файла
        key = f"ms{ms}_vm{vm}_tol{tol}_max{max_fill_days}_{interval}_{lookback_days}d"
        if tag: key += f"_{tag}"
        # также короткий хэш на всякий случай
        key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:6]
        fname = f"signals_{key}_{key_hash}.xlsx"
        out_path = out_root / fname

        # сбор сигналов по сетке
        rows_all = []
        for s in symbols:
            df4h = cache.get(s)
            if df4h is None or df4h.empty:
                continue
            rows_all.extend(_detect_for_symbol(
                df4h, s, min_strength=ms, vol_mult=vm, tol_pct=tol, max_days_to_fill=max_fill_days
            ))

        df_sig = pd.DataFrame(rows_all)
        if not df_sig.empty:
            # строгое приведение времени (xlsx без tz)
            if "imb_time" in df_sig.columns:
                df_sig["imb_time"] = pd.to_datetime(df_sig["imb_time"], utc=True, errors="coerce").dt.tz_localize(None)

            with pd.ExcelWriter(out_path) as wr:
                df_sig.to_excel(wr, sheet_name="data", index=False)

            print(f"✅ saved {len(df_sig):>5} → {out_path}")
            manifest.append({
                "file": str(out_path),
                "signals": int(len(df_sig)),
                "min_strength": float(ms),
                "volume_multiplier": float(vm),
                "tolerance_pct": float(tol),
                "max_fill_days": int(max_fill_days),
                "interval": interval,
                "lookback_days": int(lookback_days),
                "tag": tag or "",
                "generated_at_utc": ts,
            })
        else:
            print(f"⛔ empty → ms={ms} vm={vm} tol={tol} (skip file)")

    # manifest
    if manifest:
        man_path = out_root / f"manifest_{ts}{('_'+tag) if tag else ''}.csv"
        pd.DataFrame(manifest).sort_values(
            ["signals","min_strength","volume_multiplier"], ascending=[False, True, True]
        ).to_csv(man_path, index=False)
        print(f"📄 manifest saved → {man_path}")
    else:
        print("⚠️ manifest is empty (no signals generated).")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Генерация РАЗЛИЧНЫХ файлов сигналов по сетке параметров (без eval)")
    p.add_argument("--interval", default="4h")
    p.add_argument("--lookback-days", type=int, default=360)
    p.add_argument("--symbols", default=None, help="через запятую; по умолчанию config.TRADE_UNIVERSE")
    p.add_argument("--grid-strength", default="3.0", help="пример: '2.0,2.5,3.0'")
    p.add_argument("--grid-vol", default="1.5", help="пример: '1.1,1.3,1.5'")
    p.add_argument("--grid-tol", default="0.1", help="пример: '0,0.0005,0.001'")
    p.add_argument("--max-fill-days", type=int, default=30)
    p.add_argument("--out-dir", default="./data/signals")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in (args.symbols.split(",") if args.symbols else TRADE_UNIVERSE) if s.strip()]
    main(
        symbols=symbols,
        interval=str(args.interval),
        lookback_days=int(args.lookback_days),
        grid_strength=str(args.grid_strength),
        grid_vol=str(args.grid_vol),
        grid_tol=str(args.grid_tol),
        max_fill_days=int(args.max_fill_days),
        out_dir=str(args.out_dir),
        tag=args.tag,
    )