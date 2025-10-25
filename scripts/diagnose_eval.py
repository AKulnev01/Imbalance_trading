#!/usr/bin/env python3
import os
import sys
import time
import random
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# попытка подтянуть проектный корень
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# пробуем импортировать evaluate_common.fetch_ltf_window (не обязательно)
_fetch_ltf_window = None
_to_utc_safe = None
try:
    from evaluate_common import fetch_ltf_window as _fetch_ltf_window, to_utc_safe as _to_utc_safe
except Exception:
    pass

def to_utc_safe(ts):
    if _to_utc_safe:
        return _to_utc_safe(ts)
    if pd.isna(ts):
        return pd.NaT
    t = pd.to_datetime(ts, errors='coerce')
    if t is pd.NaT:
        return pd.NaT
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize('UTC')
    return t.tz_convert('UTC')

def normalize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """самодостаточная нормализация времени — без зависимостей проекта."""
    if df is None or df.empty:
        return df
    out = df.copy()
    candidates = ["time","timestamp","open_time","datetime","date","ts"]
    time_col = next((c for c in candidates if c in out.columns), None)

    if time_col is None:
        if isinstance(out.index, pd.DatetimeIndex):
            idx = out.index
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            out.index = idx
            return out.sort_index()
        raise RuntimeError(f"нет колонки времени среди {candidates}, имеем: {list(out.columns)}")

    t = out[time_col]
    if np.issubdtype(t.dtype, np.number):
        mx = float(t.max()) if len(t) else 0.0
        unit = "ms" if mx > 1e12 else "s"
        dt = pd.to_datetime(t, unit=unit, utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(t, utc=True, errors="coerce")

    out = out.assign(__t=dt).dropna(subset=["__t"]).set_index("__t").sort_index()
    out.index.name = "time"
    return out

def find_ltf_file(ltf_root: Path, symbol: str) -> Path:
    sym = str(symbol).upper().strip()
    p = ltf_root / f"{sym}_m1.parquet"
    if p.exists():
        return p
    p_alt = ltf_root / f"{sym}.parquet"
    if p_alt.exists():
        return p_alt
    return None

def main():
    ap = argparse.ArgumentParser(description="Diagnose evaluate/minute data pipeline")
    ap.add_argument("signals", help="Путь к файлу сигналов (xlsx)")
    ap.add_argument("--ltf-root", default=os.getenv("LTF_ROOT", "../data/m1"),
                    help="Путь к минуткам (parquet). По умолчанию $LTF_ROOT или ./data/m1")
    ap.add_argument("--sample", type=int, default=10, help="Сколько символов проверить (по сигналам)")
    ap.add_argument("--window-min", type=int, default=60, help="Размер окна минуток после imb_time")
    args = ap.parse_args()

    sig_path = Path(os.path.expanduser(args.signals)).resolve()
    ltf_root = Path(os.path.expanduser(args.ltf_root)).resolve()

    print("=== ENV / paths ===")
    print(f"USE_LOCAL_MINUTES={os.getenv('USE_LOCAL_MINUTES')}")
    print(f"USE_LOCAL_4H     ={os.getenv('USE_LOCAL_4H')}")
    print(f"LTF_ROOT (arg)   ={ltf_root}")
    print(f"Signals path     ={sig_path}")
    print(f"fetch_ltf_window ={bool(_fetch_ltf_window)} (импорт из evaluate_common {'успешен' if _fetch_ltf_window else 'НЕ доступен'})")
    print("")

    if not sig_path.exists():
        print(f"❌ файл сигналов не найден: {sig_path}")
        sys.exit(2)

    try:
        head = pd.read_excel(sig_path, sheet_name=0, nrows=0)
        sheet = "data" if "data" in pd.ExcelFile(sig_path).sheet_names else 0
        sig = pd.read_excel(sig_path, sheet_name=sheet)
    except Exception as e:
        print(f"❌ не удалось прочитать Excel: {e}")
        sys.exit(2)

    print(f"OK: сигналов загружено {len(sig)} строк, колонки: {list(sig.columns)}")
    needed = ["symbol","imb_time","entry","stop","tp"]
    miss = [c for c in needed if c not in sig.columns]
    if miss:
        print(f"❌ не хватает колонок: {miss}")
        sys.exit(2)

    # базовая чистка
    sig["symbol"] = sig["symbol"].astype(str).str.upper().str.strip()
    sig["imb_time"] = pd.to_datetime(sig["imb_time"], utc=True, errors="coerce")
    n_bad_time = sig["imb_time"].isna().sum()
    if n_bad_time:
        print(f"⚠️ imb_time не распарсен у {n_bad_time} строк")

    # сводка по LTF_ROOT
    print("\n=== Проверка LTF_ROOT ===")
    if not ltf_root.exists():
        print(f"❌ LTF_ROOT не существует: {ltf_root}")
        sys.exit(2)
    n_parquet = len(list(ltf_root.glob("*.parquet")))
    print(f"OK: найдено parquet-файлов: {n_parquet}")

    # возьмём sample символов из сигналов
    syms = sig["symbol"].dropna().unique().tolist()
    random.seed(0)
    random.shuffle(syms)
    syms = syms[: max(1, args.sample)]

    problems = []
    ok_files = 0
    print("\n=== Проверка наличия файлов минуток ===")
    for s in syms:
        p = find_ltf_file(ltf_root, s)
        if p is None:
            print(f"❌ нет минуток для {s}: {ltf_root}/{s}_m1.parquet (или {s}.parquet)")
            problems.append(("missing_parquet", s))
        else:
            print(f"OK: {s} → {p.name}")
            ok_files += 1
    if ok_files == 0:
        print("❌ ни одного файла минуток из сэмпла не найдено — дальше проверять нечего")
        sys.exit(2)

    # детальная проверка первых 3-5 символов: структура parquet, диапазон дат, окно вокруг imb_time
    check_syms = syms[: min(5, len(syms))]
    print("\n=== Детальная проверка минуток и окон вокруг imb_time ===")
    for s in check_syms:
        p = find_ltf_file(ltf_root, s)
        if p is None:
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"❌ {s}: не читается parquet: {e}")
            problems.append(("bad_parquet", s))
            continue

        cols = list(df.columns)
        print(f"\n[{s}] parquet columns: {cols[:12]}{'...' if len(cols)>12 else ''}  rows={len(df)}")
        try:
            dfn = normalize_time_index(df)
        except Exception as e:
            print(f"❌ {s}: не удалось нормализовать время: {e}")
            problems.append(("bad_time_col", s))
            continue

        if dfn.empty:
            print(f"❌ {s}: нормализованный df пустой")
            problems.append(("empty_after_normalize", s))
            continue

        tmin, tmax = dfn.index.min(), dfn.index.max()
        print(f"    time range (UTC): {tmin} → {tmax}")

        # все сигналы по этому символу:
        rows = sig[sig["symbol"] == s].copy()
        rows = rows[rows["imb_time"].notna()]
        if rows.empty:
            print(f"    ⚠️ в сигнале для {s} нет валидных imb_time")
            continue

        inside = 0
        for _, r in rows.head(3).iterrows():
            t0 = to_utc_safe(r["imb_time"])
            ok = (t0 >= tmin) and (t0 <= tmax)
            print(f"    imb_time={t0} → {'OK' if ok else 'OUT-OF-RANGE'}")
            inside += int(ok)

            # проверим окно LTF 1m [+{window_min}m]
            t1 = t0 + pd.Timedelta(minutes=int(args.window_min))
            t0s = time.time()
            if _fetch_ltf_window:
                try:
                    w = _fetch_ltf_window(s, t0, t1, candidates=["1m"])
                    took = time.time() - t0s
                    print(f"      fetch_ltf_window: len={len(w)}  took={took:.3f}s "
                          f"range={w.index.min() if not w.empty else None} → {w.index.max() if not w.empty else None}")
                except Exception as e:
                    took = time.time() - t0s
                    print(f"      ❌ fetch_ltf_window error ({took:.3f}s): {e}")
                    problems.append(("fetch_ltf_error", s))
            else:
                # fallback без evaluate_common: просто вырезаем окно из dfn
                win = dfn[(dfn.index >= t0) & (dfn.index <= t1)]
                print(f"      local cut (fallback): len={len(win)} "
                      f"range={win.index.min() if not win.empty else None} → {win.index.max() if not win.empty else None}")

        if inside == 0:
            print("    ⚠️ все проверенные imb_time вне диапазона минуток — оценщик будет видеть пустые окна")

    print("\n=== РЕЗЮМЕ ===")
    if not problems:
        print("✅ явных проблем не найдено. Если всё ещё «таймауты», включи профилинг:")
        print("   export EVAL_PROF=1  (и смотри длительность fetch_ltf_window в логе evaluate)")
    else:
        kinds = {}
        for k, _ in problems:
            kinds[k] = kinds.get(k, 0) + 1
        print("Найдены проблемы по типам:", kinds)
        print("См. сообщения выше для конкретных символов.")

if __name__ == "__main__":
    main()