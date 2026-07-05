from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "production" / "pipeline" / "gate3" / "gate3_build_symbol_policy.py"


def set_if_exists(ns: dict, key: str, value) -> None:
    if key in ns:
        ns[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-edge", default="production/models/ks/gate3_active_regime_edge.csv")
    parser.add_argument("--out-policy", default="production/models/ks/gate3_symbol_policy.csv")
    parser.add_argument("--strong-score", type=float, default=0.0)
    args = parser.parse_args()

    ns = runpy.run_path(str(SCRIPT), run_name="__gate3_symbol_policy_wrapped__")

    set_if_exists(ns, "IN_EDGE", str(args.in_edge))
    set_if_exists(ns, "OUT_POLICY", str(args.out_policy))
    set_if_exists(ns, "STRONG_SCORE", float(args.strong_score))

    print("Gate3 Symbol Policy Wrapper")
    print("SCRIPT:", SCRIPT)
    print("IN_EDGE:", args.in_edge)
    print("OUT_POLICY:", args.out_policy)
    print("STRONG_SCORE:", args.strong_score)
    print("=" * 120)

    if "apply_runtime_args" in ns:
        ns["apply_runtime_args"](args)
    else:
        ns["IN_EDGE"] = str(args.in_edge)
        ns["OUT_POLICY"] = str(args.out_policy)
        ns["STRONG_SCORE"] = float(args.strong_score)

    ns["main"]()


if __name__ == "__main__":
    main()
