from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "test" / "gate4" / "train_gate4_bin.py"


def set_if_exists(ns: dict, key: str, value) -> None:
    if key in ns:
        ns[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-parquet", default="production/dataset/gate4/gate4_1_side_builder/gate4_1_side_dataset.parquet")
    parser.add_argument("--out-root", default="pipeline/test/gate4/gate4_y_side_clean_multiclass_no_raw_refs")
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate4_trainer_wrapped__")

    out_root = str(args.out_root)

    ns["DATASET_PARQUET"] = str(args.dataset_parquet)
    ns["OUT_ROOT"] = out_root

    set_if_exists(ns, "MODEL_PATH", os.path.join(out_root, "gate4_y_side_clean_multiclass.cbm"))
    set_if_exists(ns, "FEATURES_CSV", os.path.join(out_root, "features.csv"))
    set_if_exists(ns, "REPORT_JSON", os.path.join(out_root, "report.json"))
    set_if_exists(ns, "METRICS_JSON", os.path.join(out_root, "metrics.json"))
    set_if_exists(ns, "VALID_PREDICTIONS_CSV", os.path.join(out_root, "valid_predictions.csv"))
    set_if_exists(ns, "VALID_PREDICTIONS_PARQUET", os.path.join(out_root, "valid_predictions.parquet"))

    print("Gate4 Trainer Wrapper")
    print("SCRIPT:", SCRIPT)
    print("DATASET_PARQUET:", ns["DATASET_PARQUET"])
    print("OUT_ROOT:", ns["OUT_ROOT"])
    print("=" * 120)

    ns["main"]()


if __name__ == "__main__":
    main()
