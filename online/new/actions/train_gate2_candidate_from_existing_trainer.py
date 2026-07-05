from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

GATE2_PIPELINE_DIR = ROOT / "production" / "pipeline" / "gate2_mod"
DATASET_BASE_DIR = ROOT / "production" / "dataset" / "gate2_candidates"
CANDIDATE_MODEL_BASE_DIR = ROOT / "production" / "models" / "gate2_mod_5features_candidates"
PROD_GATE2_MODEL_DIR = ROOT / "production" / "models" / "gate2_mod_5features"
BACKUP_ROOT = ROOT / "online" / "new" / "actions" / "_artifact_backups"


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_existing_gate2_trainer() -> Path:
    candidates: List[Path] = []

    for path in sorted(GATE2_PIPELINE_DIR.glob("*.py")):
        if path.name == "common_5features.py":
            continue

        text = path.read_text(encoding="utf-8", errors="replace")

        required = [
            "CATBOOST_PARAMS_BASE",
            "TASKS",
            "def train_one",
            "REACH_ALL_PATH",
            "OUT_ROOT",
            "gate2_up_reach_high",
            "gate2_dn_reach_high",
        ]

        if all(x in text for x in required):
            candidates.append(path)

    if len(candidates) != 1:
        print("GATE2_TRAINER_CANDIDATES:")
        for item in candidates:
            print("  ", item)
        raise RuntimeError(
            "Expected exactly one existing Gate2 trainer in {}, found {}. "
            "Pass --trainer-path explicitly.".format(GATE2_PIPELINE_DIR, len(candidates))
        )

    return candidates[0]


def import_module_from_path(path: Path):
    module_name = "gate2_existing_trainer_for_candidate"

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load module spec: {}".format(path))

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def backup_existing_tree(path: Path, artifact_group: str, tag: str) -> List[str]:
    if not path.exists() or not any(path.rglob("*")):
        return []

    tag_safe = str(tag).replace("/", "_").replace("\\", "_")
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUP_ROOT / artifact_group / tag_safe / run_id
    backup_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []

    for src in sorted(path.rglob("*")):
        if not src.is_file():
            continue

        rel = src.relative_to(path)
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        copied.append(str(dst))

    print("BACKUP_TREE:", path, "->", backup_dir, "files=", len(copied), flush=True)
    return copied


def read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def collect_task_reports(out_root: Path, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []

    for task in tasks:
        name = str(task.get("name"))
        report_path = out_root / "cls" / name / "report.json"
        report = read_json_safe(report_path)

        reports.append(
            {
                "task": name,
                "target_col": str(task.get("target_col")),
                "report_path": str(report_path),
                "report_exists": bool(report_path.exists()),
                "rows_train": report.get("rows_train"),
                "rows_valid": report.get("rows_valid"),
                "feature_count": report.get("feature_count"),
                "class_rate_valid": report.get("class_rate_valid"),
                "best_threshold": report.get("best_threshold"),
                "metrics": report.get("metrics", {}),
                "files": report.get("files", {}),
            }
        )

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Gate2 candidate model by wrapping the existing production.pipeline.gate2_mod trainer. "
            "The wrapper only redirects REACH_ALL_PATH and OUT_ROOT to candidate paths. "
            "It never writes to production/models/gate2_mod_5features."
        )
    )
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--trainer-path", default="")
    parser.add_argument("--dataset-base-dir", default="")
    parser.add_argument("--out-base-dir", default="")
    parser.add_argument("--prod-gate2-model-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started_at = time.time()

    global DATASET_BASE_DIR
    global CANDIDATE_MODEL_BASE_DIR
    global PROD_GATE2_MODEL_DIR

    if str(args.dataset_base_dir).strip():
        dataset_base_dir_arg = Path(str(args.dataset_base_dir).strip())
        DATASET_BASE_DIR = dataset_base_dir_arg if dataset_base_dir_arg.is_absolute() else ROOT / dataset_base_dir_arg

    if str(args.out_base_dir).strip():
        out_base_dir_arg = Path(str(args.out_base_dir).strip())
        CANDIDATE_MODEL_BASE_DIR = out_base_dir_arg if out_base_dir_arg.is_absolute() else ROOT / out_base_dir_arg

    if str(args.prod_gate2_model_dir).strip():
        prod_gate2_model_dir_arg = Path(str(args.prod_gate2_model_dir).strip())
        PROD_GATE2_MODEL_DIR = prod_gate2_model_dir_arg if prod_gate2_model_dir_arg.is_absolute() else ROOT / prod_gate2_model_dir_arg

    dataset_tag = str(args.dataset_tag).strip()
    model_tag = str(args.model_tag).strip()

    if not dataset_tag:
        raise RuntimeError("--dataset-tag is empty")

    if not model_tag:
        raise RuntimeError("--model-tag is empty")

    dataset_root = DATASET_BASE_DIR / dataset_tag
    reach_all_path = dataset_root / "final_gate2_2_directional_reach_5features_all.parquet"

    if not reach_all_path.exists():
        raise FileNotFoundError(str(reach_all_path))

    out_root = CANDIDATE_MODEL_BASE_DIR / model_tag

    if out_root.resolve() == PROD_GATE2_MODEL_DIR.resolve():
        raise RuntimeError("SAFETY_STOP: candidate OUT_ROOT equals prod Gate2 model dir")

    if str(PROD_GATE2_MODEL_DIR.resolve()).lower() in str(out_root.resolve()).lower():
        raise RuntimeError(
            "SAFETY_STOP: candidate OUT_ROOT must not be inside prod Gate2 model dir: {}".format(out_root)
        )

    trainer_path = Path(args.trainer_path) if str(args.trainer_path).strip() else find_existing_gate2_trainer()
    if not trainer_path.is_absolute():
        trainer_path = ROOT / trainer_path

    if not trainer_path.exists():
        raise FileNotFoundError(str(trainer_path))

    if out_root.exists() and any(out_root.rglob("*")) and not bool(args.overwrite):
        raise RuntimeError("candidate model dir already exists; pass --overwrite: {}".format(out_root))

    backup_paths: List[str] = []
    if bool(args.overwrite):
        backup_paths = backup_existing_tree(
            path=out_root,
            artifact_group="gate2_candidate_model_tree",
            tag=model_tag,
        )

    ensure_dir(out_root)

    print("Train Gate2 Candidate From Existing Trainer")
    print("ROOT:", ROOT)
    print("TRAINER_PATH:", trainer_path)
    print("DATASET_TAG:", dataset_tag)
    print("MODEL_TAG:", model_tag)
    print("DATASET_BASE_DIR:", DATASET_BASE_DIR)
    print("CANDIDATE_MODEL_BASE_DIR:", CANDIDATE_MODEL_BASE_DIR)
    print("DATASET_ROOT:", dataset_root)
    print("REACH_ALL_PATH:", reach_all_path)
    print("CANDIDATE_OUT_ROOT:", out_root)
    print("PROD_GATE2_MODEL_DIR_NOT_TOUCHED:", PROD_GATE2_MODEL_DIR)
    print("OVERWRITE:", bool(args.overwrite))
    print("=" * 120)

    trainer = import_module_from_path(trainer_path)

    if not hasattr(trainer, "train_one"):
        raise RuntimeError("trainer has no train_one: {}".format(trainer_path))

    if not hasattr(trainer, "TASKS"):
        raise RuntimeError("trainer has no TASKS: {}".format(trainer_path))

    old_reach_all_path = getattr(trainer, "REACH_ALL_PATH", None)
    old_out_root = getattr(trainer, "OUT_ROOT", None)

    trainer.REACH_ALL_PATH = str(reach_all_path)
    trainer.OUT_ROOT = str(out_root)

    print("PATCHED_IN_MEMORY:")
    print("  OLD_REACH_ALL_PATH:", old_reach_all_path)
    print("  NEW_REACH_ALL_PATH:", trainer.REACH_ALL_PATH)
    print("  OLD_OUT_ROOT:", old_out_root)
    print("  NEW_OUT_ROOT:", trainer.OUT_ROOT)
    print("=" * 120)

    df = pd.read_parquet(str(reach_all_path))

    tasks = list(trainer.TASKS)

    for task in tasks:
        print("RUN_EXISTING_TRAIN_ONE:", task, flush=True)
        trainer.train_one(df, task)

    task_reports = collect_task_reports(out_root=out_root, tasks=tasks)

    wrapper_report = {
        "created_at_utc": str(pd.Timestamp.now(tz="UTC")),
        "name": "Train Gate2 Candidate From Existing Trainer",
        "dataset_tag": dataset_tag,
        "model_tag": model_tag,
        "trainer_path": str(trainer_path),
        "dataset_base_dir": str(DATASET_BASE_DIR),
        "candidate_model_base_dir": str(CANDIDATE_MODEL_BASE_DIR),
        "reach_all_path": str(reach_all_path),
        "candidate_out_root": str(out_root),
        "prod_gate2_model_dir_not_touched": str(PROD_GATE2_MODEL_DIR),
        "old_trainer_reach_all_path": str(old_reach_all_path),
        "old_trainer_out_root": str(old_out_root),
        "overwrite": bool(args.overwrite),
        "backup_paths": backup_paths,
        "tasks": task_reports,
        "elapsed_sec": round(time.time() - started_at, 3),
    }

    wrapper_report_path = out_root / "_WRAPPER_REPORT.json"
    wrapper_report_path.write_text(
        json.dumps(wrapper_report, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("=" * 120)
    print("DONE")
    print("WROTE_WRAPPER_REPORT:", wrapper_report_path)
    print("CANDIDATE_OUT_ROOT:", out_root)
    print("PROD_GATE2_MODEL_DIR_NOT_TOUCHED:", PROD_GATE2_MODEL_DIR)
    print("ELAPSED_SEC:", wrapper_report["elapsed_sec"])


if __name__ == "__main__":
    main()
