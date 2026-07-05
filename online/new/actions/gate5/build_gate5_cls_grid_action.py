from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "test" / "gate5" / "build_gate5_cls_grid.py"


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="pipeline/test/gate5/gate5_2_grid_ranker_no_raw_refs_thr010/gate5_grid_ranker_dataset.parquet")
    parser.add_argument("--pair-data-dir", default="pipeline/test/gate5/gate5_pair_datasets_no_raw_refs_thr010")
    parser.add_argument("--out-dir", default="pipeline/test/gate5/gate5_3_no_raw_refs_thr010")
    parser.add_argument("--grid-list", default="tp100_sl075,tp150_sl075,tp120_sl060,tp150_sl060,tp100_sl050,tp160_sl040,tp225_sl075,tp240_sl060,tp200_sl050,tp180_sl060,tp120_sl040,tp125_sl050")
    parser.add_argument("--safe-list", default="")
    parser.add_argument("--aggr-list", default="")
    parser.add_argument("--delta-min", type=float, default=0.25)
    parser.add_argument("--entry-delay-seconds", type=int, default=90)
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate5_3_pairwise_builder_wrapped__")

    grid_list = parse_csv_list(args.grid_list)
    safe_list = parse_csv_list(args.safe_list) if str(args.safe_list).strip() else list(grid_list)
    aggr_list = parse_csv_list(args.aggr_list) if str(args.aggr_list).strip() else list(grid_list)

    if not grid_list:
        raise RuntimeError("--grid-list is empty")
    if not safe_list:
        raise RuntimeError("--safe-list is empty")
    if not aggr_list:
        raise RuntimeError("--aggr-list is empty")

    out_dir = Path(str(args.out_dir))

    ns["DATA_PATH"] = Path(str(args.data_path))
    ns["PAIR_DATA_DIR"] = Path(str(args.pair_data_dir))
    ns["OUT_DIR"] = out_dir
    ns["OUT_REPORT"] = out_dir / "_build_report.json"
    ns["GRID_LIST"] = grid_list
    ns["SAFE_LIST"] = safe_list
    ns["AGGR_LIST"] = aggr_list
    ns["DELTA_MIN"] = float(args.delta_min)
    ns["ENTRY_DELAY_SECONDS"] = int(args.entry_delay_seconds)

    print("Gate5_3 Pairwise Dataset Builder Wrapper")
    print("SCRIPT:", SCRIPT)
    print("DATA_PATH:", ns["DATA_PATH"])
    print("PAIR_DATA_DIR:", ns["PAIR_DATA_DIR"])
    print("OUT_DIR:", ns["OUT_DIR"])
    print("GRID_LIST:", ns["GRID_LIST"])
    print("SAFE_LIST:", ns["SAFE_LIST"])
    print("AGGR_LIST:", ns["AGGR_LIST"])
    print("DELTA_MIN:", ns["DELTA_MIN"])
    print("ENTRY_DELAY_SECONDS:", ns["ENTRY_DELAY_SECONDS"])
    print("=" * 120)

    ns["main"]()


if __name__ == "__main__":
    main()
