from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "test" / "gate5" / "build_gate5_pair_datasets.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate4-dataset-parquet", default="production/dataset/gate4/gate4_1_side_builder/gate4_1_side_dataset.parquet")
    parser.add_argument("--gate4-predictions-csv", default="pipeline/test/gate4/gate4_y_side_clean_multiclass_no_raw_refs/all_predictions_raw.csv")
    parser.add_argument("--m1-data-dir", default="data/m1_4")
    parser.add_argument("--experiment-name", default="gate5_pair_datasets_no_raw_refs_thr010")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--gate4-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--side-ratio-min", type=float, default=1.25)
    parser.add_argument("--gate5-valid-pct", type=float, default=0.10)
    parser.add_argument("--train-end", default="")
    parser.add_argument("--valid-start", default="")
    parser.add_argument("--valid-end", default="")
    parser.add_argument("--use-only-valid-split", action="store_true")
    parser.add_argument("--clean-old-test-outputs", action="store_true")
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate5_pair_dataset_builder_wrapped__")

    experiment_name = str(args.experiment_name)
    out_root = Path(str(args.out_root)) if str(args.out_root).strip() else ROOT / "pipeline" / "test" / "gate5" / experiment_name

    ns["GATE4_DATASET_PARQUET"] = Path(str(args.gate4_dataset_parquet))
    ns["GATE4_PREDICTIONS_CSV"] = Path(str(args.gate4_predictions_csv))
    ns["M1_DATA_DIR"] = Path(str(args.m1_data_dir))
    ns["EXPERIMENT_NAME"] = experiment_name
    ns["OUT_ROOT"] = out_root
    ns["OUT_SUMMARY_CSV"] = out_root / "_SUMMARY.csv"
    ns["OUT_REPORT_JSON"] = out_root / "_REPORT.json"

    ns["CLEAN_OLD_TEST_OUTPUTS"] = bool(args.clean_old_test_outputs)
    ns["ALLOWED_CLEAN_ROOT_PARTS"] = (
        "pipeline",
        "test",
        "gate5",
        experiment_name,
    )

    ns["GATE4_CONFIDENCE_THRESHOLD"] = float(args.gate4_confidence_threshold)
    ns["SIDE_RATIO_MIN"] = float(args.side_ratio_min)
    ns["GATE5_VALID_PCT"] = float(args.gate5_valid_pct)
    ns["TRAIN_END"] = str(args.train_end).strip()
    ns["VALID_START"] = str(args.valid_start).strip()
    ns["VALID_END"] = str(args.valid_end).strip()
    ns["USE_ONLY_VALID_SPLIT"] = bool(args.use_only_valid_split)

    print("Gate5 Pair Dataset Builder Wrapper")
    print("SCRIPT:", SCRIPT)
    print("GATE4_DATASET_PARQUET:", ns["GATE4_DATASET_PARQUET"])
    print("GATE4_PREDICTIONS_CSV:", ns["GATE4_PREDICTIONS_CSV"])
    print("M1_DATA_DIR:", ns["M1_DATA_DIR"])
    print("OUT_ROOT:", ns["OUT_ROOT"])
    print("CLEAN_OLD_TEST_OUTPUTS:", ns["CLEAN_OLD_TEST_OUTPUTS"])
    print("TRAIN_END:", ns["TRAIN_END"])
    print("VALID_START:", ns["VALID_START"])
    print("VALID_END:", ns["VALID_END"])
    print("=" * 120)

    ns["main"]()


if __name__ == "__main__":
    main()
