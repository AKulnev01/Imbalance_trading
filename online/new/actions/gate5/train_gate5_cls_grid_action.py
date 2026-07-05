from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "test" / "gate5" / "train_gate5_cls_grid.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="pipeline/test/gate5/gate5_3_no_raw_refs_thr010")
    parser.add_argument("--out-dir", default="pipeline/test/gate5/gate5_3_cls_models_no_raw_refs_thr010")
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate5_3_trainer_wrapped__")

    out_dir = Path(str(args.out_dir))

    ns["DATA_DIR"] = Path(str(args.data_dir))
    ns["OUT_DIR"] = out_dir
    ns["SUMMARY_CSV"] = out_dir / "summary.csv"
    ns["SUMMARY_JSON"] = out_dir / "summary.json"

    print("Gate5_3 Pairwise Trainer Wrapper")
    print("SCRIPT:", SCRIPT)
    print("DATA_DIR:", ns["DATA_DIR"])
    print("OUT_DIR:", ns["OUT_DIR"])
    print("=" * 120)

    ns["main"]()


if __name__ == "__main__":
    main()
