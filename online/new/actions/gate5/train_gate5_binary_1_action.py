from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "test" / "gate5" / "train_gate5_binary_1.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="pipeline/test/gate5/gate5_pair_datasets_no_raw_refs_thr010")
    parser.add_argument("--out-dir", default="pipeline/test/gate5/gate5_1_oof_no_raw_refs_thr010")
    parser.add_argument("--min-kept", type=int, default=20)
    parser.add_argument("--oof-n-folds", type=int, default=5)
    parser.add_argument("--oof-min-train-rows", type=int, default=1000)
    parser.add_argument("--oof-min-fold-rows", type=int, default=200)
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate5_1_trainer_wrapped__")

    data_dir = Path(str(args.data_dir))
    out_dir = Path(str(args.out_dir))

    ns["DATA_DIR"] = data_dir
    ns["DATASETS"] = sorted(
        p.stem.replace("gate5_dataset_", "")
        for p in data_dir.glob("gate5_dataset_*.parquet")
    )

    ns["OUT_DIR"] = out_dir
    ns["OUT_REPORT"] = out_dir / "report.json"
    ns["OUT_SUMMARY_CSV"] = out_dir / "summary.csv"
    ns["OUT_SWEEP_CSV"] = out_dir / "threshold_sweep.csv"

    ns["MIN_KEPT"] = int(args.min_kept)
    ns["OOF_N_FOLDS"] = int(args.oof_n_folds)
    ns["OOF_MIN_TRAIN_ROWS"] = int(args.oof_min_train_rows)
    ns["OOF_MIN_FOLD_ROWS"] = int(args.oof_min_fold_rows)

    print("Gate5_1 Binary Trainer Wrapper")
    print("SCRIPT:", SCRIPT)
    print("DATA_DIR:", ns["DATA_DIR"])
    print("OUT_DIR:", ns["OUT_DIR"])
    print("DATASETS:", len(ns["DATASETS"]))
    print("=" * 120)

    ns["main"]()


if __name__ == "__main__":
    main()
