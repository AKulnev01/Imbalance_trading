from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "test" / "gate4" / "build_gate4_predictions.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-parquet", default="production/dataset/gate4/gate4_1_side_builder/gate4_1_side_dataset.parquet")
    parser.add_argument("--model-dir", default="pipeline/test/gate4/gate4_y_side_clean_multiclass_no_raw_refs")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--features-csv", default="")
    parser.add_argument("--out-predictions-csv", default="")
    parser.add_argument("--out-build-report-json", default="")
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate4_predictions_wrapped__")

    model_dir = Path(str(args.model_dir))
    model_path = Path(str(args.model_path)) if str(args.model_path).strip() else model_dir / "gate4_y_side_clean_multiclass.cbm"
    features_csv = Path(str(args.features_csv)) if str(args.features_csv).strip() else model_dir / "features.csv"
    out_predictions_csv = Path(str(args.out_predictions_csv)) if str(args.out_predictions_csv).strip() else model_dir / "all_predictions_raw.csv"
    out_build_report_json = Path(str(args.out_build_report_json)) if str(args.out_build_report_json).strip() else model_dir / "all_predictions_raw_build_report.json"

    ns["DATASET_PARQUET"] = Path(str(args.dataset_parquet))
    ns["MODEL_DIR"] = model_dir
    ns["MODEL_PATH"] = model_path
    ns["FEATURES_CSV"] = features_csv
    ns["OUT_PREDICTIONS_CSV"] = out_predictions_csv
    ns["OUT_BUILD_REPORT_JSON"] = out_build_report_json

    print("Gate4 Predictions Builder Wrapper")
    print("SCRIPT:", SCRIPT)
    print("DATASET_PARQUET:", ns["DATASET_PARQUET"])
    print("MODEL_DIR:", ns["MODEL_DIR"])
    print("MODEL_PATH:", ns["MODEL_PATH"])
    print("FEATURES_CSV:", ns["FEATURES_CSV"])
    print("OUT_PREDICTIONS_CSV:", ns["OUT_PREDICTIONS_CSV"])
    print("OUT_BUILD_REPORT_JSON:", ns["OUT_BUILD_REPORT_JSON"])
    print("=" * 120)

    ns["main"]()


if __name__ == "__main__":
    main()
