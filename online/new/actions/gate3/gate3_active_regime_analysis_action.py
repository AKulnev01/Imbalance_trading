from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "gate3" / "gate3_active_regime_analysis.py"


def set_if_exists(ns: dict, key: str, value) -> None:
    if key in ns:
        ns[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="production/dataset/pa_gate3_v3_long_short_by_symbol")
    parser.add_argument("--h4-dir", default="data/h4_3")
    parser.add_argument("--out-dir", default="production/models/ks")
    parser.add_argument("--max-fwd", type=int, default=4)
    parser.add_argument("--min-pattern-n", type=int, default=80)
    parser.add_argument("--target-hit-atr", type=float, default=0.8)
    parser.add_argument("--train-end", default="")
    parser.add_argument("--valid-start", default="")
    parser.add_argument("--valid-end", default="")
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

    ns = runpy.run_path(str(SCRIPT), run_name="__gate3_active_regime_analysis_wrapped__")

    set_if_exists(ns, "DATA_DIR", str(args.data_dir))
    set_if_exists(ns, "H4_DIR", str(args.h4_dir))
    set_if_exists(ns, "OUT_DIR", str(args.out_dir))
    set_if_exists(ns, "MAX_FWD", int(args.max_fwd))
    set_if_exists(ns, "MIN_PATTERN_N", int(args.min_pattern_n))
    set_if_exists(ns, "TARGET_HIT_ATR", float(args.target_hit_atr))

    print("Gate3 Active Regime Analysis Wrapper")
    print("SCRIPT:", SCRIPT)
    print("DATA_DIR:", args.data_dir)
    print("H4_DIR:", args.h4_dir)
    print("OUT_DIR:", args.out_dir)
    print("MAX_FWD:", args.max_fwd)
    print("MIN_PATTERN_N:", args.min_pattern_n)
    print("TARGET_HIT_ATR:", args.target_hit_atr)
    print("TRAIN_END:", args.train_end)
    print("VALID_START:", args.valid_start)
    print("VALID_END:", args.valid_end)
    print("=" * 120)

    # The production script has legacy hardcoded globals.
    # Override them after exec() and before main(), otherwise it ignores wrapper CLI args.
    ns["DATA_DIR"] = args.data_dir
    ns["H4_DIR"] = args.h4_dir
    ns["OUT_DIR"] = args.out_dir
    ns["MAX_FWD"] = args.max_fwd
    ns["MIN_PATTERN_N"] = args.min_pattern_n
    ns["TARGET_HIT_ATR"] = args.target_hit_atr

    ns["main"]()

    # Keep backward-compatible filename expected by downstream onboarding plan.
    from pathlib import Path as _Path
    import shutil as _shutil

    out_dir = _Path(args.out_dir)
    src_edge = out_dir / "_ACTIVE_EDGE_full_timeline.csv"
    dst_edge = out_dir / "gate3_active_regime_edge.csv"

    if src_edge.exists():
        _shutil.copy2(src_edge, dst_edge)
        print("WROTE_ALIAS", dst_edge)
    else:
        raise FileNotFoundError("missing active edge output: {}".format(src_edge))


if __name__ == "__main__":
    main()
