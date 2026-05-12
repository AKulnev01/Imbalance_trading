# generate_bonk_grid_v14.py — версия с корректной логикой SLIPPAGE “0.4% туда и обратно”

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ===================== ПАРАМЕТРЫ =====================

SYMBOL = "1000BONKUSDT"
M1_PATH = Path("data/m1") / f"{SYMBOL}_m1.parquet"
OUT_PATH = Path("bonk_grid_v14.parquet")

# базовые оптимальные TP/SL в долях (13% и 4%)
TP_BASE = 0.13
SL_BASE = 0.04

# сетка множителей
N_TP = 20
N_SL = 10
TP_MIN_MULT = 0.5
TP_MAX_MULT = 2.0
SL_MIN_MULT = 0.6
SL_MAX_MULT = 1.5

# TTL сделки в часах
TTL_HOURS = 80

# комиссии и слиппедж (в долях от ноты)
FEE_ENTRY = 0.001   # 0.1%
FEE_EXIT  = 0.001   # 0.1%
SLIPPAGE  = 0.004   # 0.4% на входе и 0.4% на выходе (только в деньгах, НЕ в цене)

# фиксированная нота на сделку
NOTIONAL = 100.0

# ===================== ЗАГРУЗКА M1 =====================

print("Loading m1:", M1_PATH)
if not M1_PATH.exists():
    raise SystemExit(f"Нет минуток: {M1_PATH}")

m1 = pd.read_parquet(M1_PATH)

# ---- время ----
time_col = None
for c in ["open_time", "ts", "timestamp", "time"]:
    if c in m1.columns:
        time_col = c
        break
if time_col is None:
    raise SystemExit("нет time колонки")

t0 = m1[time_col].iloc[0]
if t0 > 10**12:
    m1["dt"] = pd.to_datetime(m1[time_col], unit="ms")
elif t0 > 10**9:
    m1["dt"] = pd.to_datetime(m1[time_col], unit="s")
else:
    m1["dt"] = pd.to_datetime(m1[time_col])

m1 = m1.sort_values("dt").set_index("dt")

for c in ["open", "high", "low", "close"]:
    if c not in m1.columns:
        raise SystemExit("нет OHLC данных (нужны open/high/low/close)")

m1_ts   = m1.index.values
m_high  = m1["high"].values
m_low   = m1["low"].values
m_close = m1["close"].values

print("Loaded m1 rows:", len(m1))

# ===================== АГРЕГАЦИЯ В 4H =====================

h4 = (
    m1[["open", "high", "low", "close", "volume"]]
    .resample("4h", label="right", closed="right")
    .agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    .dropna()
)
h4 = h4.reset_index().rename(columns={"dt": "bar_ts"})
print("4H bars:", len(h4))

# ===================== СЕТКА KS =====================

tp_mults = np.linspace(TP_MIN_MULT, TP_MAX_MULT, N_TP)
sl_mults = np.linspace(SL_MIN_MULT, SL_MAX_MULT, N_SL)
KS_GRID = [(float(a), float(b)) for a in tp_mults for b in sl_mults]

print("KS combos:", len(KS_GRID))

# ===================== СИМУЛЯЦИЯ TP/SL/TTL =====================

def simulate_numpy(idx_start, idx_end, ref_px, tp_pct, sl_pct, side):
    """
    На вход:
      idx_start, idx_end — диапазон индексов минуток
      ref_px             — цена ref_close (по ней входим)
      tp_pct, sl_pct     — TP/SL в долях (0.1 = 10%)
      side               — +1 (long) или -1 (short)

    Возвращает:
      exit_px, exit_reason
      exit_px — рыночная цена, по которой выходим (без комиссий и слиппеджа)
    """
    if idx_start >= idx_end or ref_px <= 0:
        return ref_px, "no_data"

    # ВАЖНО: здесь SLIPPAGE НЕ используем — только чистые ценовые уровни
    entry_px = ref_px

    if side > 0:
        # LONG: TP = +tp_pct, SL = -sl_pct
        tp_lvl = entry_px * (1.0 + tp_pct)
        sl_lvl = entry_px * (1.0 - sl_pct)

        hi_slice = m_high[idx_start:idx_end]
        lo_slice = m_low[idx_start:idx_end]

        tp_hits = np.where(hi_slice >= tp_lvl)[0]
        sl_hits = np.where(lo_slice <= sl_lvl)[0]
    else:
        # SHORT: выгодно падение цены.
        # TP = -tp_pct, SL = +sl_pct
        tp_lvl = entry_px * (1.0 - tp_pct)
        sl_lvl = entry_px * (1.0 + sl_pct)

        lo_slice = m_low[idx_start:idx_end]
        hi_slice = m_high[idx_start:idx_end]

        tp_hits = np.where(lo_slice <= tp_lvl)[0]
        sl_hits = np.where(hi_slice >= sl_lvl)[0]

    tp_i = int(tp_hits[0]) if tp_hits.size > 0 else None
    sl_i = int(sl_hits[0]) if sl_hits.size > 0 else None

    # кто сработал раньше: TP или SL
    if tp_i is not None and sl_i is not None:
        if sl_i <= tp_i:
            # сначала стоп, потом или одновременно TP
            return sl_lvl, "sl_both"
        else:
            return tp_lvl, "tp"
    if tp_i is not None:
        return tp_lvl, "tp"
    if sl_i is not None:
        return sl_lvl, "sl"

    # ни TP, ни SL — выходим по последней цене минутки в TTL
    return m_close[idx_end - 1], "ttl"

# ===================== ОБРАБОТКА ОДНОГО 4H-БАРА =====================

def process_bar(args):
    bar_ts, ref_close = args

    # окно жизни сделки
    t_start = bar_ts
    t_end   = bar_ts + pd.Timedelta(hours=TTL_HOURS)

    idx_start = np.searchsorted(m1_ts, np.datetime64(t_start), side="left")
    idx_end   = np.searchsorted(m1_ts, np.datetime64(t_end),   side="right")

    if idx_start >= idx_end:
        return []

    out = []

    for side in (+1, -1):  # сначала long, потом short
        for tp_mult, sl_mult in KS_GRID:
            tp_pct = TP_BASE * tp_mult
            sl_pct = SL_BASE * sl_mult

            exit_px, reason = simulate_numpy(
                idx_start, idx_end,
                ref_close, tp_pct, sl_pct,
                side
            )

            entry_px = ref_close
            if entry_px <= 0:
                continue

            # доходность по цене с учётом направления
            # r > 0 — прибыль, r < 0 — убыток
            r = side * (exit_px - entry_px) / entry_px

            # модель денег:
            # V0 — изначально 100$
            V0 = NOTIONAL

            # вход: минус slippage
            V1 = V0 * (1.0 - SLIPPAGE)

            # движение цены: применяем доходность r
            V2 = V1 * (1.0 + r)

            # выход: снова минус slippage
            V3 = V2 * (1.0 - SLIPPAGE)

            # комиссии считаем относительно исходной ноты (упрощение)
            fees = V0 * (FEE_ENTRY + FEE_EXIT)

            pnl_net = V3 - V0 - fees

            out.append([
                SYMBOL,
                bar_ts,
                side,
                float(tp_mult),
                float(sl_mult),
                TTL_HOURS,
                float(ref_close),
                reason,
                float(pnl_net),
            ])

    return out

# ===================== MULTIPROC =====================

if __name__ == "__main__":
    cpu_n = min(8, cpu_count())
    print(f"Using {cpu_n} workers")

    tasks = list(zip(h4["bar_ts"].values, h4["close"].values))

    all_rows = []
    with Pool(cpu_n) as pool:
        for i, chunk in enumerate(pool.imap_unordered(process_bar, tasks, chunksize=16), start=1):
            if chunk:
                all_rows.extend(chunk)
            if i % 200 == 0:
                print(f"  processed {i}/{len(tasks)} bars")

    df = pd.DataFrame(
        all_rows,
        columns=[
            "symbol",
            "entry_ts",
            "side",
            "ks_tp_mult",
            "ks_sl_mult",
            "ks_ttl_hours",
            "ref_close",
            "exit_reason",
            "pnl_net",
        ],
    )

    print("Total rows:", len(df))
    print(df["pnl_net"].describe())

    df.to_parquet(OUT_PATH)
    print("Saved ->", OUT_PATH)