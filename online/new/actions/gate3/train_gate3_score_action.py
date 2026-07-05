from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "gate3" / "train_gate3_score.py"


def set_if_exists(ns: dict, key: str, value) -> None:
    if key in ns:
        ns[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4-dir", default="data/h4_3")
    parser.add_argument("--base-data-dir", default="production/dataset/gate1")
    parser.add_argument("--gate3-data-dir", default="production/dataset/pa_gate3_v3_long_short_by_symbol")
    parser.add_argument("--gate3-audit-csv", default="production/models/ks/gate3_active_regime_edge.csv")
    parser.add_argument("--gate1-models-dir", default="production/models/final_gate1")
    parser.add_argument("--policy-csv", default="production/models/ks/gate3_symbol_policy.csv.updated")
    parser.add_argument("--out-root", default="production/models/final_gate3_score_long_short")
    parser.add_argument("--train-end", default="")
    parser.add_argument("--valid-start", default="")
    parser.add_argument("--valid-end", default="")
    parser.add_argument("--symbols", default="")
    args = parser.parse_args()

    if str(args.train_end).strip():
        os.environ["IMB_OFFLINE_TRAIN_END"] = str(args.train_end).strip()
    else:
        os.environ.pop("IMB_OFFLINE_TRAIN_END", None)

    if str(args.valid_start).strip():
        os.environ["IMB_OFFLINE_VALID_START"] = str(args.valid_start).strip()
    else:
        os.environ.pop("IMB_OFFLINE_VALID_START", None)

    if str(args.valid_end).strip():
        os.environ["IMB_OFFLINE_VALID_END"] = str(args.valid_end).strip()
    else:
        os.environ.pop("IMB_OFFLINE_VALID_END", None)

    if str(args.symbols).strip():
        os.environ["IMB_OFFLINE_SYMBOLS"] = str(args.symbols).strip()
    else:
        os.environ.pop("IMB_OFFLINE_SYMBOLS", None)

    os.environ["IMB_GATE3_H4_DIR"] = str(args.h4_dir)
    os.environ["IMB_GATE3_BASE_DATA_DIR"] = str(args.base_data_dir)
    os.environ["IMB_GATE3_DATA_DIR"] = str(args.gate3_data_dir)
    os.environ["IMB_GATE3_AUDIT_CSV"] = str(args.gate3_audit_csv)
    os.environ["IMB_GATE3_GATE1_MODELS_DIR"] = str(args.gate1_models_dir)
    os.environ["IMB_GATE3_POLICY_CSV"] = str(args.policy_csv)
    os.environ["IMB_GATE3_OUT_ROOT"] = str(args.out_root)

    ns = runpy.run_path(str(SCRIPT), run_name="__gate3_score_train_wrapped__")

    set_if_exists(ns, "H4_DIR", str(args.h4_dir))
    set_if_exists(ns, "BASE_DATA_DIR", str(args.base_data_dir))
    set_if_exists(ns, "GATE3_DATA_DIR", str(args.gate3_data_dir))
    set_if_exists(ns, "GATE3_AUDIT_CSV", str(args.gate3_audit_csv))
    set_if_exists(ns, "GATE1_MODELS_DIR", str(args.gate1_models_dir))
    set_if_exists(ns, "POLICY_CSV", str(args.policy_csv))
    set_if_exists(ns, "OUT_ROOT", str(args.out_root))

    print("Gate3 Score Train Wrapper")
    print("SCRIPT:", SCRIPT)
    print("H4_DIR:", args.h4_dir)
    print("BASE_DATA_DIR:", args.base_data_dir)
    print("GATE3_DATA_DIR:", args.gate3_data_dir)
    print("GATE3_AUDIT_CSV:", args.gate3_audit_csv)
    print("GATE1_MODELS_DIR:", args.gate1_models_dir)
    print("POLICY_CSV:", args.policy_csv)
    print("OUT_ROOT:", args.out_root)
    print("TRAIN_END:", args.train_end)
    print("VALID_START:", args.valid_start)
    print("VALID_END:", args.valid_end)
    print("SYMBOLS:", args.symbols if str(args.symbols).strip() else "ALL_FROM_POLICY")
    print("=" * 120)

    # train_gate3_score.py is a legacy top-level script:
    # runpy.run_path(...) already executes training. Some versions do not define main().
    if "main" in ns:
        ns["main"]()
    else:
        print("NO_MAIN_FUNCTION_AFTER_RUNPY_TOP_LEVEL_SCRIPT_ALREADY_EXECUTED")


if __name__ == "__main__":
    main()
