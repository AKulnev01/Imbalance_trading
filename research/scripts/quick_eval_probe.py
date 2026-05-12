#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pandas as pd

# доступ к проекту
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluate_common import fetch_ltf_window, exit_on_ltf, to_utc_safe, calc_sl_tp

def main():
    import argparse
    ap = argparse.ArgumentParser("Minimal probe evaluator")
    ap.add_argument("signals", help="Путь к signals.xlsx (ваш сгенерённый файл)")
    ap.add_argument("--sheet", default="data", help="Лист с данными (по умолчанию 'data')")
    ap.add_argument("--sample", type=int, default=30, help="Сколько строк проверить")
    ap.add_argument("--ttl-days", type=int, default=int(os.getenv("DEFAULT_TTL_DAYS", "7")))
    ap.add_argument("--out", default="./data/signals/quick_eval_probe.xlsx")
    args = ap.parse_args()

    sig_path = Path(os.path.expanduser(args.signals)).resolve()
    if not sig_path.exists():
        print(f"❌ нет файла сигналов: {sig_path}")
        sys.exit(2)

    print(f"▶ читаю сигналы: {sig_path}")
    df = pd.read_excel(sig_path, sheet_name=args.sheet)
    need = ["symbol","type","imb_time","entry","stop","tp"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"❌ нет колонок: {miss}")
        sys.exit(2)

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["type"] = df["type"].astype(str).str.upper().str.strip()
    df["imb_time"] = pd.to_datetime(df["imb_time"], utc=True, errors="coerce")
    df["entry"] = pd.to_numeric(df["entry"], errors="coerce")
    df["stop"]  = pd.to_numeric(df["stop"],  errors="coerce")
    df["tp"]    = pd.to_numeric(df["tp"],    errors="coerce")

    base = df[df["imb_time"].notna() & df["entry"].notna() & df["stop"].notna() & df["tp"].notna()].head(args.sample)
    print(f"rows to probe: {len(base)}")

    rows = []
    for i, r in base.iterrows():
        sym  = r["symbol"]
        side = r["type"]
        t0   = to_utc_safe(r["imb_time"])
        entry= float(r["entry"])
        stop = float(r["stop"])
        tp   = float(r["tp"])
        t1   = t0 + pd.Timedelta(days=args.ttl_days)

        # 1) быстрая проверка окна LTF
        win = fetch_ltf_window(sym, t0, t0 + pd.Timedelta(minutes=60), candidates=["1m"])
        ltf_ok = (not win.empty)
        # 2) «грубая» оценка — считаем, что в сделку зашли в момент imbalanced bar (t0)
        winflag, exitt, price, reason = exit_on_ltf(sym, side, t0, stop, tp, t1)

        rows.append({
            "symbol": sym,
            "side": side,
            "imb_time": t0,
            "entry_assumed_at": t0,
            "ltf_1h_window_rows": len(win),
            "exit_time": exitt,
            "exit_reason": reason,
            "exit_price": price,
            "win": bool(winflag),
        })

    res = pd.DataFrame(rows)
    for col in ("imb_time", "entry_assumed_at", "exit_time"):
        if col in res.columns:
            res[col] = pd.to_datetime(res[col], utc=True, errors="coerce").dt.tz_convert(None)

    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as wr:
        res.to_excel(wr, index=False, sheet_name="probe")
    print(f"\n✅ probe saved: {out_path}  rows={len(res)}")

    print("\n=== SUMMARY (head) ===")
    print(res.head(10).to_string(index=False))

    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as wr:
        res.to_excel(wr, index=False, sheet_name="probe")
    print(f"\n✅ probe saved: {out_path}  rows={len(res)}")

if __name__ == "__main__":
    # окружение для локальных минуток
    os.environ.setdefault("USE_LOCAL_MINUTES", "1")
    os.environ.setdefault("LTF_ROOT", "./data/m1")
    main()