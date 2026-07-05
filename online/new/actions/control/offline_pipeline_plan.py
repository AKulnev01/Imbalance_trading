from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[4]


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    text = text.replace("/", "")
    text = text.replace("-", "")
    text = text.replace("_", "")

    if not text:
        raise ValueError("empty symbol")

    return text


def normalize_symbols(symbols_raw: str) -> List[str]:
    parts = []
    for x in str(symbols_raw).replace(";", ",").split(","):
        s = normalize_symbol(x)
        if s not in parts:
            parts.append(s)
    return parts


def make_run_tag(mode: str, run_tag: str) -> str:
    if str(run_tag).strip():
        return str(run_tag).strip()
    return "{}_{}".format(str(mode).strip().lower(), utc_now_tag())


def q(value: str) -> str:
    text = str(value)
    if " " in text:
        return '"{}"'.format(text)
    return text


def build_step(
    step_id: str,
    title: str,
    script: str,
    args: List[str],
    writes: Optional[List[str]] = None,
    reads: Optional[List[str]] = None,
    enabled: bool = True,
    note: str = "",
) -> Dict[str, Any]:
    cmd = ["python", script] + args
    return {
        "id": step_id,
        "title": title,
        "enabled": bool(enabled),
        "script": script,
        "args": args,
        "command": " ".join(q(x) for x in cmd),
        "reads": reads or [],
        "writes": writes or [],
        "note": note,
    }


def paths_for_new_symbol() -> Dict[str, str]:
    return {
        "m1_dir": "data/m1_4",
        "h4_dir": "data/h4_3",

        "gate1_dataset_dir": "production/dataset/gate1",
        "gate1_models_root": "production/models/final_gate1",

        "gate2_models_root": "production/models/gate2_mod_5features",

        "gate3_dataset_dir": "production/dataset/pa_gate3_v3_long_short_by_symbol",
        "gate3_ks_dir": "production/models/ks",
        "gate3_policy_csv": "production/models/ks/gate3_symbol_policy.csv.updated",
        "gate3_models_root": "production/models/final_gate3_score_long_short",

        "gate4_dataset_root": "production/dataset/gate4/gate4_1_side_builder",
        "gate4_models_root": "pipeline/test/gate4/gate4_y_side_clean_multiclass_no_raw_refs",

        "gate5_pair_dataset_root": "pipeline/test/gate5/gate5_pair_datasets_no_raw_refs_thr010",
        "gate5_1_models_root": "pipeline/test/gate5/gate5_1_oof_no_raw_refs_thr010",
        "gate5_2_dataset_root": "pipeline/test/gate5/gate5_2_grid_ranker_no_raw_refs_thr010",
        "gate5_3_pairwise_root": "pipeline/test/gate5/gate5_3_no_raw_refs_thr010",
        "gate5_3_models_root": "pipeline/test/gate5/gate5_3_cls_models_no_raw_refs_thr010",
    }


def paths_for_candidate_retrain(run_tag: str) -> Dict[str, str]:
    return {
        "m1_dir": "data/m1_4/{}".format(run_tag),
        "h4_dir": "data/h4_3/{}".format(run_tag),

        "gate1_dataset_dir": "production/dataset/gate1_candidates/{}".format(run_tag),
        "gate1_models_root": "production/models/final_gate1_candidates/{}".format(run_tag),

        "gate2_dataset_tag": run_tag,
        "gate2_model_tag": run_tag,
        "gate2_dataset_root": "production/dataset/gate2_candidates/{}".format(run_tag),
        "gate2_models_root": "production/models/gate2_mod_5features_candidates/{}".format(run_tag),

        "gate3_dataset_dir": "production/dataset/pa_gate3_v3_long_short_candidates/{}".format(run_tag),
        "gate3_ks_dir": "production/models/ks_candidates/{}".format(run_tag),
        "gate3_edge_csv": "production/models/ks_candidates/{}/gate3_active_regime_edge.csv".format(run_tag),
        "gate3_policy_csv": "production/models/ks_candidates/{}/gate3_symbol_policy.csv.updated".format(run_tag),
        "gate3_models_root": "production/models/final_gate3_score_long_short_candidates/{}".format(run_tag),

        "gate4_dataset_root": "production/dataset/gate4_candidates/{}/gate4_1_side_builder".format(run_tag),
        "gate4_models_root": "production/models/gate4_candidates/{}/gate4_y_side_clean_multiclass_no_raw_refs".format(run_tag),

        "gate5_pair_dataset_root": "production/dataset/gate5_candidates/{}/gate5_pair_datasets".format(run_tag),
        "gate5_1_models_root": "production/models/gate5_1_candidates/{}".format(run_tag),
        "gate5_2_dataset_root": "production/dataset/gate5_candidates/{}/gate5_2_grid_ranker".format(run_tag),
        "gate5_3_pairwise_root": "production/dataset/gate5_candidates/{}/gate5_3_pairwise".format(run_tag),
        "gate5_3_models_root": "production/models/gate5_3_candidates/{}".format(run_tag),
    }


def build_offline_plan(
    symbols: List[str],
    mode: str,
    run_tag: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    start: str,
    end: str,
    parquet_cutoff: str,
    include_gate2_train: bool,
    include_gate4_train: bool,
    include_gate5_train: bool,
    oos_start: str,
    oos_end: str,
    db_load_oos: bool,
) -> Dict[str, Any]:
    mode = str(mode).strip().lower()
    run_tag = make_run_tag(mode=mode, run_tag=run_tag)

    if mode == "new_symbol":
        paths = paths_for_new_symbol()
        default_include_gate2_train = False
        default_include_gate4_train = False
        default_include_gate5_train = False
        mode_note = (
            "New symbol onboarding: Gate1 and Gate3 are symbol-dependent. "
            "Gate2 is common and is not retrained by default. "
            "Gate4/Gate5 are common downstream layers and are not retrained by default unless explicitly enabled."
        )
    elif mode == "candidate_retrain":
        paths = paths_for_candidate_retrain(run_tag=run_tag)
        default_include_gate2_train = False
        default_include_gate4_train = False
        default_include_gate5_train = False
        mode_note = (
            "Candidate retrain: outputs go to dated candidate paths only when explicit train flags are enabled. "
            "Prod models are not overwritten."
        )
    else:
        raise ValueError("unsupported mode: {}".format(mode))

    if include_gate2_train is None:
        include_gate2_train = default_include_gate2_train
    if include_gate4_train is None:
        include_gate4_train = default_include_gate4_train
    if include_gate5_train is None:
        include_gate5_train = default_include_gate5_train

    symbols_arg = ",".join(symbols)
    symbols_cli = []
    for s in symbols:
        symbols_cli.append(s)

    steps: List[Dict[str, Any]] = []

    candidate_train_requested = bool(
        mode == "candidate_retrain"
        and (
            bool(include_gate2_train)
            or bool(include_gate4_train)
            or bool(include_gate5_train)
        )
    )

    base_symbol_pipeline_enabled = bool(
        mode == "new_symbol"
        or candidate_train_requested
    )

    steps.append(build_step(
        step_id="download_m1_h4",
        title="Download M1 and build H4 parquet",
        script="online/new/actions/download_symbol_m1_h4_parquet.py",
        args=[
            "--symbols",
            *symbols_cli,
            "--start", start,
            "--end", end,
            "--m1-dir", paths["m1_dir"],
            "--h4-dir", paths["h4_dir"],
            "--parquet-cutoff", parquet_cutoff,
            "--snapshot-if-base-exists",
            "--overwrite-snapshot",
        ],
        writes=[
            paths["m1_dir"],
            paths["h4_dir"],
        ],
        enabled=base_symbol_pipeline_enabled,
    ))

    steps.append(build_step(
        step_id="gate1_dataset",
        title="Build Gate1 datasets",
        script="online/new/actions/build_gate1_dataset_for_symbols.py",
        args=[
            "--symbols",
            *symbols_cli,
            "--m1-dir", paths["m1_dir"],
            "--out-dir", paths["gate1_dataset_dir"],
        ],
        reads=[paths["m1_dir"]],
        writes=[paths["gate1_dataset_dir"]],
        enabled=base_symbol_pipeline_enabled,
    ))

    steps.append(build_step(
        step_id="gate1_train",
        title="Train Gate1 per-symbol models",
        script="online/new/actions/train_gate1_models_for_symbols.py",
        args=[
            "--symbols",
            *symbols_cli,
            "--dataset-dir", paths["gate1_dataset_dir"],
            "--out-root", paths["gate1_models_root"],
            "--train-end", train_end,
            "--valid-start", valid_start,
            "--valid-end", valid_end,
            "--overwrite",
        ],
        reads=[paths["gate1_dataset_dir"]],
        writes=[paths["gate1_models_root"]],
        enabled=base_symbol_pipeline_enabled,
    ))

    steps.append(build_step(
        step_id="gate2_candidate_datasets",
        title="Build Gate2 candidate datasets",
        script="online/new/actions/build_gate2_candidate_datasets.py",
        args=[
            "--dataset-tag", run_tag,
            "--symbols",
            *symbols_cli,
            "--train-end", train_end,
            "--valid-start", valid_start,
            "--valid-end", valid_end,
            "--src-gate1-dir", paths["gate1_dataset_dir"],
            "--m1-dir", paths["m1_dir"],
            "--h4-dir", paths["h4_dir"],
            "--out-base-dir", "production/dataset/gate2_candidates",
        ],
        reads=[
            paths["gate1_dataset_dir"],
            paths["m1_dir"],
            paths["h4_dir"],
        ],
        writes=[
            "production/dataset/gate2_candidates/{}".format(run_tag),
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate2_train)),
        note="Disabled for simple new-symbol onboarding because Gate2 is common prod model.",
    ))

    steps.append(build_step(
        step_id="gate2_candidate_train",
        title="Train Gate2 candidate common model",
        script="online/new/actions/train_gate2_candidate_from_existing_trainer.py",
        args=[
            "--dataset-tag", run_tag,
            "--model-tag", run_tag,
            "--dataset-base-dir", "production/dataset/gate2_candidates",
            "--out-base-dir", "production/models/gate2_mod_5features_candidates",
        ],
        reads=[
            "production/dataset/gate2_candidates/{}".format(run_tag),
        ],
        writes=[
            "production/models/gate2_mod_5features_candidates/{}".format(run_tag),
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate2_train)),
        note="Disabled for simple new-symbol onboarding.",
    ))

    gate2_mod_dir = (
        paths["gate2_models_root"]
        if mode == "candidate_retrain" and bool(include_gate2_train)
        else "production/models/gate2_mod_5features"
    )

    steps.append(build_step(
        step_id="gate3_pa_dataset",
        title="Build Gate3 PA dataset",
        script="online/new/actions/gate3/build_pa_gate3_v3_long_short_action.py",
        args=[
            "--gate1-root", paths["gate1_dataset_dir"],
            "--h4-root", paths["h4_dir"],
            "--out-root", paths["gate3_dataset_dir"],
            "--symbols", symbols_arg,
        ],
        reads=[
            paths["gate1_dataset_dir"],
            paths["h4_dir"],
        ],
        writes=[
            paths["gate3_dataset_dir"],
        ],
        enabled=base_symbol_pipeline_enabled,
    ))

    steps.append(build_step(
        step_id="gate3_active_regime_analysis",
        title="Gate3 active regime analysis",
        script="online/new/actions/gate3/gate3_active_regime_analysis_action.py",
        args=[
            "--data-dir", paths["gate3_dataset_dir"],
            "--h4-dir", paths["h4_dir"],
            "--out-dir", paths["gate3_ks_dir"],
            "--train-end", train_end,
            "--valid-start", valid_start,
            "--valid-end", valid_end,
        ],
        reads=[
            paths["gate3_dataset_dir"],
            paths["h4_dir"],
        ],
        writes=[
            paths["gate3_ks_dir"],
        ],
        enabled=base_symbol_pipeline_enabled,
    ))

    gate3_edge_csv = paths.get("gate3_edge_csv", "{}/gate3_active_regime_edge.csv".format(paths["gate3_ks_dir"]))

    steps.append(build_step(
        step_id="gate3_symbol_policy",
        title="Build Gate3 symbol policy",
        script="online/new/actions/gate3/gate3_build_symbol_policy_action.py",
        args=[
            "--in-edge", gate3_edge_csv,
            "--out-policy", paths["gate3_policy_csv"],
        ],
        reads=[
            gate3_edge_csv,
        ],
        writes=[
            paths["gate3_policy_csv"],
        ],
        enabled=base_symbol_pipeline_enabled,
    ))

    steps.append(build_step(
        step_id="gate3_score_train",
        title="Train Gate3 score models",
        script="online/new/actions/gate3/train_gate3_score_action.py",
        args=[
            "--h4-dir", paths["h4_dir"],
            "--base-data-dir", paths["gate1_dataset_dir"],
            "--gate3-data-dir", paths["gate3_dataset_dir"],
            "--gate3-audit-csv", gate3_edge_csv,
            "--gate1-models-dir", paths["gate1_models_root"],
            "--policy-csv", paths["gate3_policy_csv"],
            "--out-root", paths["gate3_models_root"],
            "--train-end", train_end,
            "--valid-start", valid_start,
            "--valid-end", valid_end,
            "--symbols", symbols_arg,
        ],
        reads=[
            paths["h4_dir"],
            paths["gate1_dataset_dir"],
            paths["gate3_dataset_dir"],
            gate3_edge_csv,
            paths["gate1_models_root"],
            paths["gate3_policy_csv"],
        ],
        writes=[
            paths["gate3_models_root"],
        ],
        enabled=base_symbol_pipeline_enabled,
    ))

    for _symbol_for_oos in symbols:
        steps.append(build_step(
            step_id="load_oos_candles_to_db_{}".format(_symbol_for_oos),
            title="Load OOS candles to DB for {}".format(_symbol_for_oos),
            script="online/new/actions/control/oos_validation_db_loader.py",
            args=[
                "--symbol", _symbol_for_oos,
                "--m1-parquet", "{}/{}.parquet".format(paths["m1_dir"], _symbol_for_oos),
                "--h4-parquet", "{}/{}.parquet".format(paths["h4_dir"], _symbol_for_oos),
                "--oos-start", oos_start,
                "--oos-end", oos_end,
                "--market-category", "linear",
                "--source", "bybit",
                "--on-conflict", "skip",
            ] + (["--write"] if bool(db_load_oos) else []),
            reads=[
                "{}/{}.parquet".format(paths["m1_dir"], _symbol_for_oos),
                "{}/{}.parquet".format(paths["h4_dir"], _symbol_for_oos),
            ],
            writes=[
                "public.candles_m1",
                "public.candles_h4",
            ],
            enabled=(mode == "new_symbol" and bool(oos_start)),
            note=(
                "For a new symbol this loads only validation/OOS candles into public.candles_m1 and public.candles_h4. "
                "Without --db-load-oos the generated command is dry-run. "
                "With --db-load-oos it writes using --on-conflict skip, so existing candles do not break retry runs."
            ),
        ))

    steps.append(build_step(
        step_id="gate4_dataset",
        title="Build Gate4 dataset",
        script="online/new/actions/gate4/build_gate4_dataset_v2_action.py",
        args=[
            "--base-data-dir", paths["gate1_dataset_dir"],
            "--gate3-data-dir", paths["gate3_dataset_dir"],
            "--gate1-models-dir", paths["gate1_models_root"],
            "--gate2-mod-dir", gate2_mod_dir,
            "--gate3-score-root", paths["gate3_models_root"],
            "--policy-csv", paths["gate3_policy_csv"],
            "--out-root", paths["gate4_dataset_root"],
            "--train-end", train_end,
            "--valid-start", valid_start,
            "--valid-end", valid_end,
        ],
        reads=[
            paths["gate1_dataset_dir"],
            paths["gate3_dataset_dir"],
            paths["gate1_models_root"],
            gate2_mod_dir,
            paths["gate3_models_root"],
            paths["gate3_policy_csv"],
        ],
        writes=[
            paths["gate4_dataset_root"],
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate4_train)),
        note=("Disabled for new_symbol. Gate4 common layer must be applied later by online/OOS DB branch." if mode == "new_symbol" else "Enabled only when Gate4 candidate train is explicit."),
    ))

    steps.append(build_step(
        step_id="gate4_train",
        title="Train Gate4 common side model",
        script="online/new/actions/gate4/train_gate4_bin_action.py",
        args=[
            "--dataset-parquet", "{}/gate4_1_side_dataset.parquet".format(paths["gate4_dataset_root"]),
            "--out-root", paths["gate4_models_root"],
        ],
        reads=[
            "{}/gate4_1_side_dataset.parquet".format(paths["gate4_dataset_root"]),
        ],
        writes=[
            paths["gate4_models_root"],
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate4_train)),
        note="Disabled by default for simple new-symbol onboarding because Gate4 is common.",
    ))

    steps.append(build_step(
        step_id="gate4_predictions",
        title="Build Gate4 predictions",
        script="online/new/actions/gate4/build_gate4_predictions_action.py",
        args=[
            "--dataset-parquet", "{}/gate4_1_side_dataset.parquet".format(paths["gate4_dataset_root"]),
            "--model-dir", paths["gate4_models_root"],
        ],
        reads=[
            "{}/gate4_1_side_dataset.parquet".format(paths["gate4_dataset_root"]),
            paths["gate4_models_root"],
        ],
        writes=[
            "{}/all_predictions_raw.csv".format(paths["gate4_models_root"]),
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate4_train)),
        note=("Disabled for new_symbol. Gate4 predictions must be produced by online/OOS branch, not by offline training plan." if mode == "new_symbol" else "Enabled only when Gate4 candidate train is explicit."),
    ))

    steps.append(build_step(
        step_id="gate5_pair_datasets",
        title="Build Gate5 pair datasets",
        script="online/new/actions/gate5/build_gate5_pair_datasets_action.py",
        args=[
            "--gate4-dataset-parquet", "{}/gate4_1_side_dataset.parquet".format(paths["gate4_dataset_root"]),
            "--gate4-predictions-csv", "{}/all_predictions_raw.csv".format(paths["gate4_models_root"]),
            "--m1-data-dir", paths["m1_dir"],
            "--out-root", paths["gate5_pair_dataset_root"],
            "--train-end", train_end,
            "--valid-start", valid_start,
            "--valid-end", valid_end,
        ],
        reads=[
            "{}/gate4_1_side_dataset.parquet".format(paths["gate4_dataset_root"]),
            "{}/all_predictions_raw.csv".format(paths["gate4_models_root"]),
            paths["m1_dir"],
        ],
        writes=[
            paths["gate5_pair_dataset_root"],
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate4_train) and bool(include_gate5_train)),
        note=("Disabled for new_symbol. Gate5 pair datasets are training/evaluation artifacts, not simple onboarding artifacts." if mode == "new_symbol" else "Enabled only when Gate4 and Gate5 candidate train are explicit."),
    ))

    steps.append(build_step(
        step_id="gate5_1_train",
        title="Train Gate5_1 binary models",
        script="online/new/actions/gate5/train_gate5_binary_1_action.py",
        args=[
            "--data-dir", paths["gate5_pair_dataset_root"],
            "--out-dir", paths["gate5_1_models_root"],
        ],
        reads=[
            paths["gate5_pair_dataset_root"],
        ],
        writes=[
            paths["gate5_1_models_root"],
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate4_train) and bool(include_gate5_train)),
        note="Enabled only when Gate4 and Gate5 candidate train are explicit.",
    ))

    gate5_pred_dir = (
        paths["gate5_1_models_root"]
        if mode == "candidate_retrain" and bool(include_gate5_train)
        else "pipeline/test/gate5/gate5_1_oof_no_raw_refs_thr010"
    )

    steps.append(build_step(
        step_id="gate5_2_ranker",
        title="Build Gate5_2 grid-ranker dataset",
        script="online/new/actions/gate5/builder_gate5_grid_ranker_action.py",
        args=[
            "--pred-dir", gate5_pred_dir,
            "--grid-data-dir", paths["gate5_pair_dataset_root"],
            "--out-dir", paths["gate5_2_dataset_root"],
        ],
        reads=[
            gate5_pred_dir,
            paths["gate5_pair_dataset_root"],
        ],
        writes=[
            paths["gate5_2_dataset_root"],
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate4_train) and bool(include_gate5_train)),
        note=("Disabled for new_symbol. Gate5_2 ranker dataset belongs to candidate retrain/evaluation flow." if mode == "new_symbol" else "Enabled only when Gate4 and Gate5 candidate train are explicit."),
    ))

    steps.append(build_step(
        step_id="gate5_3_pairwise_dataset",
        title="Build Gate5_3 pairwise grid dataset",
        script="online/new/actions/gate5/build_gate5_cls_grid_action.py",
        args=[
            "--data-path", "{}/gate5_grid_ranker_dataset.parquet".format(paths["gate5_2_dataset_root"]),
            "--pair-data-dir", paths["gate5_pair_dataset_root"],
            "--out-dir", paths["gate5_3_pairwise_root"],
        ],
        reads=[
            "{}/gate5_grid_ranker_dataset.parquet".format(paths["gate5_2_dataset_root"]),
            paths["gate5_pair_dataset_root"],
        ],
        writes=[
            paths["gate5_3_pairwise_root"],
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate4_train) and bool(include_gate5_train)),
        note=("Disabled for new_symbol. Gate5_3 pairwise dataset belongs to candidate retrain/evaluation flow." if mode == "new_symbol" else "Enabled only when Gate4 and Gate5 candidate train are explicit."),
    ))

    steps.append(build_step(
        step_id="gate5_3_train",
        title="Train Gate5_3 pairwise grid classifiers",
        script="online/new/actions/gate5/train_gate5_cls_grid_action.py",
        args=[
            "--data-dir", paths["gate5_3_pairwise_root"],
            "--out-dir", paths["gate5_3_models_root"],
        ],
        reads=[
            paths["gate5_3_pairwise_root"],
        ],
        writes=[
            paths["gate5_3_models_root"],
        ],
        enabled=(mode == "candidate_retrain" and bool(include_gate5_train)),
        note="Disabled by default for simple new-symbol onboarding because Gate5 is common.",
    ))

    return {
        "mode": mode,
        "run_tag": run_tag,
        "symbols": symbols,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "start": start,
        "end": end,
        "parquet_cutoff": parquet_cutoff,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "db_load_oos": bool(db_load_oos),
        "paths": paths,
        "control": {
            "include_gate2_train": bool(include_gate2_train),
            "include_gate4_train": bool(include_gate4_train),
            "include_gate5_train": bool(include_gate5_train),
        },
        "note": mode_note,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline pipeline command plan for symbol onboarding/retrain."
    )
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--mode", choices=["new_symbol", "candidate_retrain"], required=True)
    parser.add_argument("--run-tag", default="")

    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="now")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--valid-start", default="")
    parser.add_argument("--valid-end", default="")
    parser.add_argument("--parquet-cutoff", default="")

    parser.add_argument("--oos-start", default="")
    parser.add_argument("--oos-end", default="")
    parser.add_argument("--db-load-oos", action="store_true")

    parser.add_argument("--include-gate2-train", action="store_true")
    parser.add_argument("--include-gate4-train", action="store_true")
    parser.add_argument("--include-gate5-train", action="store_true")
    parser.add_argument("--full-candidate-retrain", action="store_true")

    parser.add_argument("--json-out", default="")

    args = parser.parse_args()

    symbols = normalize_symbols(args.symbols)

    valid_start = str(args.valid_start or "").strip() or str(args.oos_start or "").strip() or str(args.train_end).strip()
    valid_end = str(args.valid_end or "").strip() or str(args.oos_end or "").strip() or str(args.end).strip()

    oos_start = str(args.oos_start or "").strip() or valid_start
    oos_end = str(args.oos_end or "").strip() or valid_end

    plan = build_offline_plan(
        symbols=symbols,
        mode=str(args.mode),
        run_tag=str(args.run_tag),
        train_end=str(args.train_end),
        valid_start=valid_start,
        valid_end=valid_end,
        start=str(args.start),
        end=str(args.end),
        parquet_cutoff=str(args.parquet_cutoff),
        include_gate2_train=bool(args.include_gate2_train or args.full_candidate_retrain),
        include_gate4_train=bool(args.include_gate4_train or args.full_candidate_retrain),
        include_gate5_train=bool(args.include_gate5_train or args.full_candidate_retrain),
        oos_start=oos_start,
        oos_end=oos_end,
        db_load_oos=bool(args.db_load_oos),
    )

    text = json.dumps(plan, ensure_ascii=True, indent=2)
    print(text)

    if str(args.json_out).strip():
        out_path = Path(str(args.json_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
