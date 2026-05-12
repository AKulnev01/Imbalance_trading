# predict/ks/bonk/generate_bonk_grid_v15_from_base.py
#
# Строим НОВУЮ KS-сетку (v15) по ТЕМ ЖЕ барам, что и в dataset_ks_v14_bonk_merged.parquet,
# но с:
#   - RR-фильтром по оракулам (RR_MIN..RR_MAX),
#   - новой логикой PnL (slippage 0.4% туда/обратно + fee).
#
# На выходе:
#   bonk_grid_v15_from_base.parquet
#
# Ключи для merge:
#   ["symbol","entry_ts","side","ks_tp_mult","ks_sl_mult","ks_ttl_hours"]

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

SYMBOL = "1000BONKUSDT"

BASE_PATH = Path("reports/features/dataset_ks_v14_bonk_merged.parquet")
M1_PATH   = Path("data/m1") / f"{SYMBOL}_m1.parquet"
OUT_PATH  = Path("bonk_grid_v15_from_base.parquet")

# базовые TP/SL
TP_BASE = 0.13
SL_BASE = 0.04

# сетка множителей (та же плотность, но обрежем по RR)
N_TP = 20
N_SL = 10

TP_MIN_MULT = 0.5
TP_MAX_MULT = 2.0
SL_MIN_MULT = 0.6
SL_MAX_MULT = 1.5

# RR по 10–90% оракулов (ты уже посчитал)
RR_MIN = 2.4342
RR_MAX = 7.2222

# комиссии/слип
FEE_ENTRY = 0.001
FEE_EXIT  = 0.001
SLIPPAGE  = 0.004
NOTIONAL  = 100.0


print("[LOAD BASE]", BASE_PATH)
df_base = pd.read_parquet(BASE_PATH)
df_base = df_base[df_base["symbol"] == SYMBOL].copy()
df_base["entry_ts"] = pd.to_datetime(df_base["entry_ts"])

# TTL берём из базы (ожидаем 80)
ttl_vals = df_base["ks_ttl_hours"].dropna().unique()
if len(ttl_vals) != 1:
    raise SystemExit(f"Ожидался один TTL, а их {len(ttl_vals)}: {ttl_vals}")
TTL_HOURS = int(ttl_vals[0])
print(f"[INFO] TTL_HOURS from base: {TTL_HOURS}")

# исходные бары/стороны/реф-прайсы
bars = (
    df_base[["symbol", "entry_ts", "side", "ref_close_x"]]
    .drop_duplicates()
    .sort_values(["entry_ts", "side"])
    .reset_index(drop=True)
)
print("[INFO] bars from base:", len(bars))


# === KS GRID с RR-фильтром ===

tp_mults = np.linspace(TP_MIN_MULT, TP_MAX_MULT, N_TP)
sl_mults = np.linspace(SL_MIN_MULT, SL_MAX_MULT, N_SL)

ks_grid = []
for a in tp_mults:
    for b in sl_mults:
        rr = (TP_BASE * a) / (SL_BASE * b)
        if RR_MIN <= rr <= RR_MAX:
            ks_grid.append((float(a), float(b)))

KS_GRID = ks_grid
print("[INFO] KS combos after RR filter:", len(KS_GRID))


# === ЗАГРУЗКА M1 ===

print("[LOAD M1]", M1_PATH)
if not M1_PATH.exists():
    raise SystemExit(f"Нет минуток: {M1_PATH}")

m1 = pd.read_parquet(M1_PATH)

time_col = None
for c in ["open_time", "ts", "timestamp", "time"]:
    if c in m1.columns:
        time_col = c
        break
if time_col is None:
    raise SystemExit("нет time колонки в m1")

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
        raise SystemExit("нет OHLC в m1 (open/high/low/close)")

m1_ts   = m1.index.values
m_high  = m1["high"].values
m_low   = m1["low"].values
m_close = m1["close"].values

print("[INFO] m1 rows:", len(m1))


# === СИМУЛЯЦИЯ ===

def simulate_numpy(idx_start, idx_end, ref_px, tp_pct, sl_pct, side):
    if idx_start >= idx_end or ref_px <= 0:
        return ref_px, "no_data"

    entry_px = ref_px

    if side > 0:
        tp_lvl = entry_px * (1.0 + tp_pct)
        sl_lvl = entry_px * (1.0 - sl_pct)

        hi_slice = m_high[idx_start:idx_end]
        lo_slice = m_low[idx_start:idx_end]

        tp_hits = np.where(hi_slice >= tp_lvl)[0]
        sl_hits = np.where(lo_slice <= sl_lvl)[0]
    else:
        tp_lvl = entry_px * (1.0 - tp_pct)
        sl_lvl = entry_px * (1.0 + sl_pct)

        lo_slice = m_low[idx_start:idx_end]
        hi_slice = m_high[idx_start:idx_end]

        tp_hits = np.where(lo_slice <= tp_lvl)[0]
        sl_hits = np.where(hi_slice >= sl_lvl)[0]

    tp_i = int(tp_hits[0]) if tp_hits.size > 0 else None
    sl_i = int(sl_hits[0]) if sl_hits.size > 0 else None

    if tp_i is not None and sl_i is not None:
        if sl_i <= tp_i:
            return sl_lvl, "sl_both"
        else:
            return tp_lvl, "tp"
    if tp_i is not None:
        return tp_lvl, "tp"
    if sl_i is not None:
        return sl_lvl, "sl"

    return m_close[idx_end - 1], "ttl"


def process_bar(row):
    symbol, bar_ts, side, ref_close = row

    t_start = bar_ts
    t_end   = bar_ts + pd.Timedelta(hours=TTL_HOURS)

    idx_start = np.searchsorted(m1_ts, np.datetime64(t_start), side="left")
    idx_end   = np.searchsorted(m1_ts, np.datetime64(t_end),   side="right")

    if idx_start >= idx_end or ref_close <= 0:
        return []

    out = []

    for tp_mult, sl_mult in KS_GRID:
        tp_pct = TP_BASE * tp_mult
        sl_pct = SL_BASE * sl_mult

        exit_px, reason = simulate_numpy(
            idx_start, idx_end,
            ref_close, tp_pct, sl_pct,
            side
        )

        entry_px = ref_close
        r = side * (exit_px - entry_px) / entry_px

        V0 = NOTIONAL
        V1 = V0 * (1.0 - SLIPPAGE)
        V2 = V1 * (1.0 + r)
        V3 = V2 * (1.0 - SLIPPAGE)

        fees = V0 * (FEE_ENTRY + FEE_EXIT)
        pnl_net = V3 - V0 - fees

        out.append([
            symbol,
            bar_ts,
            int(side),
            float(tp_mult),
            float(sl_mult),
            float(TTL_HOURS),
            float(ref_close),
            reason,
            float(pnl_net),
        ])

    return out


if __name__ == "__main__":
    cpu_n = min(8, cpu_count())
    print(f"[INFO] Using {cpu_n} workers")

    tasks = list(bars[["symbol","entry_ts","side","ref_close_x"]].itertuples(index=False, name=None))

    all_rows = []
    with Pool(cpu_n) as pool:
        for i, chunk in enumerate(pool.imap_unordered(process_bar, tasks, chunksize=16), start=1):
            if chunk:
                all_rows.extend(chunk)
            if i % 200 == 0:
                print(f"  processed {i}/{len(tasks)} bars")

    df_grid = pd.DataFrame(
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

    print("Total rows:", len(df_grid))
    print(df_grid["pnl_net"].describe())

    df_grid.to_parquet(OUT_PATH)
    print("Saved ->", OUT_PATH)