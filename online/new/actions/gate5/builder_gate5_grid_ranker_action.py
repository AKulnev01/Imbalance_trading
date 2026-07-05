from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "test" / "gate5" / "builder_gate5_grid_ranker.py"


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="pipeline/test/gate5/gate5_1_oof_no_raw_refs_thr010")
    parser.add_argument("--grid-data-dir", default="pipeline/test/gate5/gate5_pair_datasets_no_raw_refs_thr010")
    parser.add_argument("--out-dir", default="pipeline/test/gate5/gate5_2_grid_ranker_no_raw_refs_thr010")
    parser.add_argument("--grid-list", default="tp100_sl075,tp150_sl075,tp120_sl060,tp150_sl060,tp100_sl050,tp160_sl040,tp225_sl075,tp240_sl060,tp200_sl050,tp180_sl060,tp120_sl040,tp125_sl050")
    parser.add_argument("--require-full-proba-coverage", action="store_true")
    parser.add_argument("--keep-signals-with-missing-grid-proba", action="store_true")
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate5_2_ranker_builder_wrapped__")

    grid_list = parse_csv_list(args.grid_list)
    if not grid_list:
        raise RuntimeError("--grid-list is empty")

    out_dir = Path(str(args.out_dir))

    ns["GRID_LIST"] = grid_list
    ns["PRED_DIR"] = Path(str(args.pred_dir))
    ns["GRID_DATA_DIR"] = Path(str(args.grid_data_dir))
    ns["OUT_DIR"] = out_dir
    ns["OUT_DATASET"] = out_dir / "gate5_grid_ranker_dataset.parquet"
    ns["OUT_REPORT"] = out_dir / "gate5_grid_ranker_report.json"
    ns["REQUIRE_FULL_PROBA_COVERAGE"] = bool(args.require_full_proba_coverage)
    ns["DROP_SIGNALS_WITH_ANY_MISSING_GRID_PROBA"] = not bool(args.keep_signals_with_missing_grid_proba)

    print("Gate5_2 Grid Ranker Builder Wrapper")
    print("SCRIPT:", SCRIPT)
    print("PRED_DIR:", ns["PRED_DIR"])
    print("GRID_DATA_DIR:", ns["GRID_DATA_DIR"])
    print("OUT_DIR:", ns["OUT_DIR"])
    print("GRID_LIST:", ns["GRID_LIST"])
    print("=" * 120)

    ns["main"]()


if __name__ == "__main__":
    main()
