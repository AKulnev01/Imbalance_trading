#!/usr/bin/env python3
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# откуда берём сетку ks
SRC = Path("reports/features/dataset_ks_v11_symbol_split/dataset_ks_v11_full.parquet")

# куда раскладываем по символам
OUT_DIR = Path("reports/features/dataset_ks_v11_by_symbol")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source parquet not found: {SRC}")

    print("=== SPLIT KS V11 BY SYMBOL ===")
    print("SRC    :", SRC)
    print("OUTDIR :", OUT_DIR)

    pf = pq.ParquetFile(SRC)
    n_rg = pf.num_row_groups
    print("Row groups:", n_rg)

    writers: dict[str, pq.ParquetWriter] = {}
    counts: dict[str, int] = {}

    for rg_idx in range(n_rg):
        tbl = pf.read_row_group(rg_idx)
        sym_col = tbl["symbol"]

        unique_syms = sym_col.unique()
        print(f"[RG {rg_idx}/{n_rg}] unique symbols:", len(unique_syms))

        for s in unique_syms:
            sym = s.as_py()  # str
            mask = pc.equal(sym_col, s)
            sub = tbl.filter(mask)

            if sub.num_rows == 0:
                continue

            out_path = OUT_DIR / f"{sym}.parquet"

            if sym not in writers:
                writers[sym] = pq.ParquetWriter(out_path, sub.schema)
                counts[sym] = 0

            writers[sym].write_table(sub)
            counts[sym] += sub.num_rows

        print(f"[RG {rg_idx}] done")

    # закрываем все writer'ы
    for w in writers.values():
        w.close()

    # короткий отчёт
    total_rows = sum(counts.values())
    print("=== DONE SPLIT ===")
    print("Symbols:", len(counts))
    print("Total rows written:", total_rows)
    for sym, n in sorted(counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {sym}: {n} rows")


if __name__ == "__main__":
    main()