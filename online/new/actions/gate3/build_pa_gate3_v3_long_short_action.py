from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "gate3" / "build_pa_gate3_v3_long_short.py"


def set_if_exists(ns: dict, key: str, value) -> None:
    if key in ns:
        ns[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate1-root", "--gate1_root", dest="gate1_root", default="production/dataset/gate1")
    parser.add_argument("--h4-root", "--h4_root", dest="h4_root", default="data/h4_3")
    parser.add_argument("--out-root", "--out_root", dest="out_root", default="production/dataset/pa_gate3_v3_long_short_by_symbol")
    parser.add_argument("--context-bars", "--context_bars", dest="context_bars", type=int, default=72)
    parser.add_argument("--active-max-bars", "--active_max_bars", dest="active_max_bars", type=int, default=4)
    parser.add_argument("--active-stop-atr-mult", "--active_stop_atr_mult", dest="active_stop_atr_mult", type=float, default=1.0)
    parser.add_argument("--max-symbols", "--max_symbols", dest="max_symbols", type=int, default=0)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--replace-all-outputs", "--replace_all_outputs", dest="replace_all_outputs", action="store_true")
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate3_pa_dataset_wrapped__")

    set_if_exists(ns, "GATE1_ROOT", str(args.gate1_root))
    set_if_exists(ns, "H4_ROOT", str(args.h4_root))
    set_if_exists(ns, "OUT_ROOT", str(args.out_root))

    set_if_exists(ns, "CONTEXT_BARS", int(args.context_bars))
    set_if_exists(ns, "ACTIVE_MAX_BARS", int(args.active_max_bars))
    set_if_exists(ns, "ACTIVE_STOP_ATR_MULT", float(args.active_stop_atr_mult))

    if int(args.max_symbols) > 0:
        set_if_exists(ns, "MAX_SYMBOLS", int(args.max_symbols))

    symbols = [x.strip().upper() for x in str(args.symbols).split(",") if x.strip()]
    if symbols:
        set_if_exists(ns, "SYMBOLS", symbols)

    set_if_exists(ns, "REPLACE_ALL_OUTPUTS", bool(args.replace_all_outputs))
    set_if_exists(ns, "replace_all_outputs", bool(args.replace_all_outputs))

    print("Gate3 PA Dataset Wrapper")
    print("SCRIPT:", SCRIPT)
    print("GATE1_ROOT:", args.gate1_root)
    print("H4_ROOT:", args.h4_root)
    print("OUT_ROOT:", args.out_root)
    print("SYMBOLS:", symbols if symbols else "ALL")
    print("REPLACE_ALL_OUTPUTS:", bool(args.replace_all_outputs))
    print("=" * 120)

    ns["main"]()


if __name__ == "__main__":
    main()
