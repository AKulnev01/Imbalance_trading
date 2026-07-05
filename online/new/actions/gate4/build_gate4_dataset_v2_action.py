from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "gate4" / "build_gate4_dataset_v2.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data-dir", default="production/dataset/gate1")
    parser.add_argument("--gate3-data-dir", default="production/dataset/pa_gate3_v3_long_short_by_symbol")
    parser.add_argument("--gate1-models-dir", default="production/models/final_gate1")
    parser.add_argument("--gate2-mod-dir", default="production/models/gate2_mod_5features")
    parser.add_argument("--gate3-score-root", default="production/models/final_gate3_score_long_short")
    parser.add_argument("--policy-csv", default="production/models/ks/gate3_symbol_policy.csv.updated")
    parser.add_argument("--out-root", default="production/dataset/gate4/gate4_1_side_builder")
    parser.add_argument("--train-end", default="")
    parser.add_argument("--valid-start", default="")
    parser.add_argument("--valid-end", default="")
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate4_dataset_builder_wrapped__")

    out_root = str(args.out_root)

    ns["BASE_DATA_DIR"] = str(args.base_data_dir)
    ns["GATE3_DATA_DIR"] = str(args.gate3_data_dir)
    ns["GATE1_MODELS_DIR"] = str(args.gate1_models_dir)
    ns["GATE2_MOD_DIR"] = str(args.gate2_mod_dir)
    ns["GATE3_SCORE_ROOT"] = str(args.gate3_score_root)
    ns["POLICY_CSV"] = str(args.policy_csv)
    ns["OUT_ROOT"] = out_root
    ns["TRAIN_END"] = str(args.train_end).strip()
    ns["VALID_START"] = str(args.valid_start).strip()
    ns["VALID_END"] = str(args.valid_end).strip()
    ns["OUT_RAW_PARQUET"] = os.path.join(out_root, "gate4_1_candidates_raw.parquet")
    ns["OUT_DATASET_PARQUET"] = os.path.join(out_root, "gate4_1_side_dataset.parquet")
    ns["OUT_AUDIT_CSV"] = os.path.join(out_root, "_AUDIT.csv")
    ns["OUT_REPORT_JSON"] = os.path.join(out_root, "_REPORT.json")

    print("Gate4 Dataset Builder Wrapper")
    print("SCRIPT:", SCRIPT)
    print("BASE_DATA_DIR:", ns["BASE_DATA_DIR"])
    print("GATE3_DATA_DIR:", ns["GATE3_DATA_DIR"])
    print("GATE1_MODELS_DIR:", ns["GATE1_MODELS_DIR"])
    print("GATE2_MOD_DIR:", ns["GATE2_MOD_DIR"])
    print("GATE3_SCORE_ROOT:", ns["GATE3_SCORE_ROOT"])
    print("POLICY_CSV:", ns["POLICY_CSV"])
    print("OUT_ROOT:", ns["OUT_ROOT"])
    print("OUT_DATASET_PARQUET:", ns["OUT_DATASET_PARQUET"])
    print("TRAIN_END:", ns["TRAIN_END"])
    print("VALID_START:", ns["VALID_START"])
    print("VALID_END:", ns["VALID_END"])
    print("=" * 120)

    ns["main"]()


if __name__ == "__main__":
    main()
