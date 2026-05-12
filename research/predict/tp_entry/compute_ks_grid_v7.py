import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp
from multiprocessing import get_context
import sys


# ======================================================
# LOG
# ======================================================
def log(msg):
    print(msg)
    sys.stdout.flush()


# ======================================================
# GRID SETTINGS
# ======================================================
TP_SCALE_GRID  = [0.7, 0.85, 1.0, 1.15, 1.3]
SL_SCALE_GRID  = [0.7, 0.85, 1.0, 1.15, 1.3]
TTL_HOURS_GRID = [16, 24, 36, 48, 60, 72, 80]


# ======================================================
# SIMULATION LOGIC (V5, корректное)
# ======================================================
def simulate_trade(side, entry_px, tp_px, sl_px, ttl_hours, df_m1):
    if side == "BUY":
        def hit_tp(h, l): return h >= tp_px
        def hit_sl(h, l): return l <= sl_px
    else:
        def hit_tp(h, l): return l <= tp_px
        def hit_sl(h, l): return h >= sl_px

    max_ts = df_m1.index[0] + pd.Timedelta(hours=ttl_hours)
    chunk = df_m1[df_m1.index <= max_ts]

    for _, row in chunk.iterrows():
        h, l = row["high"], row["low"]

        if hit_tp(h, l):
            return (tp_px - entry_px)/entry_px if side == "BUY" else (entry_px - tp_px)/entry_px

        if hit_sl(h, l):
            return (sl_px - entry_px)/entry_px if side == "BUY" else (entry_px - sl_px)/entry_px

    last_px = chunk.iloc[-1]["close"]
    return (last_px - entry_px)/entry_px if side == "BUY" else (entry_px - last_px)/entry_px


# ======================================================
# WORKER
# ======================================================
def worker(args):
    part, best_ks_dict, m1_cache, chunk_id = args

    out = []
    total = len(part)
    log(f"[WORKER {chunk_id}] Start, rows={total}")

    skipped_ks = 0
    skipped_m1 = 0
    skipped_ts = 0

    for _, row in part.iterrows():
        symbol = row["symbol"]
        side   = row["side"]

        key = (symbol, side)
        if key not in best_ks_dict:
            skipped_ks += 1
            continue

        if symbol not in m1_cache:
            skipped_m1 += 1
            continue

        df_m1 = m1_cache[symbol]

        entry_ts = row["entry_ts"]
        df_m1_loc = df_m1[df_m1.index >= entry_ts]

        if len(df_m1_loc) < 3:
            skipped_ts += 1
            continue

        base_tp_abs, base_sl_abs, base_ttl = best_ks_dict[key]

        entry_px = float(row["ref_close"]) * (1 + 0.004 if side == "BUY" else 1 - 0.004)

        for tp_mul in TP_SCALE_GRID:
            for sl_mul in SL_SCALE_GRID:
                for ttl in TTL_HOURS_GRID:

                    tp_px = entry_px * (1 + base_tp_abs * tp_mul if side == "BUY"
                                        else 1 - base_tp_abs * tp_mul)
                    sl_px = entry_px * (1 - base_sl_abs * sl_mul if side == "BUY"
                                        else 1 + base_sl_abs * sl_mul)

                    ret = simulate_trade(side, entry_px, tp_px, sl_px, ttl, df_m1_loc)

                    nd = row.to_dict()
                    nd["ks_tp_scale"]  = tp_mul
                    nd["ks_sl_scale"]  = sl_mul
                    nd["ks_ttl_scale"] = ttl / base_ttl
                    nd["ks_tp_abs"]    = base_tp_abs * tp_mul
                    nd["ks_sl_abs"]    = base_sl_abs * sl_mul
                    nd["ks_ttl_hours"] = ttl
                    nd["ks_ret_adj"]   = ret
                    out.append(nd)

    log(f"[WORKER {chunk_id}] DONE: out={len(out)}, skip_ks={skipped_ks}, skip_m1={skipped_m1}, skip_ts={skipped_ts}")
    return out


# ======================================================
# MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--ks-base", required=True)
    parser.add_argument("--m1", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Load BASE
    log(f"[LOAD BASE] {args.base}")
    df = pd.read_parquet(args.base)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

    # Load KS
    log(f"[LOAD KS BASE] {args.ks_base}")
    ks = pd.read_csv(args.ks_base)
    ks["symbol"] = ks["symbol"].str.upper().str.strip()
    ks["side"]   = ks["side"].str.upper().str.strip()

    best_ks_dict = {
        (r.symbol, r.side): (float(r.k_tp_abs), float(r.k_sl_abs), int(r.ttl_hours))
        for _, r in ks.iterrows()
    }

    # Load m1 with pyarrow mmap
    log(f"[LOAD M1 CACHE]")
    m1_cache = {}
    m1_root = Path(args.m1)
    files = list(m1_root.glob("*_m1.parquet"))

    for fp in tqdm(files):
        sym = fp.name.replace("_m1.parquet", "")
        d = pq.read_table(fp, memory_map=True).to_pandas()
        d["ts"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
        d = d.set_index("ts")[["open","high","low","close","volume"]]
        m1_cache[sym] = d

    # Chunking
    total = len(df)
    chunk_size = max(50000, total // (args.workers * 5))
    log(f"[CHUNKS] total={total}, chunk_size={chunk_size}, workers={args.workers}")

    chunks = []
    cid = 0
    for i in range(0, total, chunk_size):
        chunks.append((df.iloc[i:i+chunk_size], best_ks_dict, m1_cache, cid))
        cid += 1

    # Run
    log("[START MULTIPROCESSING]")

    ctx = get_context("spawn")
    pool = ctx.Pool(args.workers)

    results = []
    for out_rows in tqdm(pool.imap(worker, chunks), total=len(chunks)):
        results.extend(out_rows)

    pool.close()
    pool.join()

    log(f"[RESULTS] rows={len(results)}")
    out_df = pd.DataFrame(results)

    log(f"[SAVE] {args.out}")
    out_df.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()