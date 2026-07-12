"""Run the full held-out caption+QA evaluation suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class EvalTarget:
    label: str
    checkpoint: Path
    output_dir: Path


SUITES = {
    "stage1": {
        "config": "configs/pretrain_astraq_vl.yaml",
        "checkpoint_root": "checkpoints/astraq-vl-stage1",
        "checkpoints": [
            ("stage1_ep1", "checkpoint-1300"),
            ("stage1_ep2", "checkpoint-2500"),
            ("stage1_ep3", "checkpoint-3789"),
        ],
        "package": "astraq-vl-stage1-full-heldout-eval-v1.zip",
    },
    "stage2": {
        "config": "configs/finetune_astraq_vl_stage2.yaml",
        "checkpoint_root": "checkpoints/astraq-vl-stage2",
        "checkpoints": [("stage2", "checkpoint-2526")],
        "package": "astraq-vl-stage2-full-heldout-eval-v1.zip",
    },
}


def run(cmd: List[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def resolve_stage(stage: str, config: Optional[str]) -> str:
    if stage != "auto":
        return stage
    branch = current_branch().lower()
    if "stg2" in branch or "stage2" in branch:
        return "stage2"
    if "stg1" in branch or "stage1" in branch:
        return "stage1"
    if config and "stage2" in config.lower():
        return "stage2"
    return "stage1"


def checkpoint_present(path: Path, stage: str) -> bool:
    if not (path / "connector.safetensors").exists():
        return False
    if stage == "stage2" and not (path / "lora" / "adapter_model.safetensors").exists():
        return False
    return True


def label_from_checkpoint(path: Path) -> str:
    return path.name.replace("-", "_") or "checkpoint"


def parse_checkpoint_specs(specs: Optional[List[str]], checkpoint_root: Path) -> List[tuple]:
    if not specs:
        return []
    parsed = []
    for spec in specs:
        if "=" in spec:
            label, raw_path = spec.split("=", 1)
        else:
            raw_path = spec
            label = label_from_checkpoint(Path(raw_path))
        path = Path(raw_path)
        if not path.is_absolute() and len(path.parts) == 1:
            path = checkpoint_root / path
        parsed.append((label, path))
    return parsed


def build_targets(args: argparse.Namespace, stage: str) -> List[EvalTarget]:
    suite = SUITES[stage]
    checkpoint_root = Path(args.checkpoint_root or suite["checkpoint_root"])
    checkpoint_specs = parse_checkpoint_specs(args.checkpoint, checkpoint_root)
    if not checkpoint_specs:
        checkpoint_specs = [(label, checkpoint_root / name) for label, name in suite["checkpoints"]]
    output_root = Path(args.output_root)
    return [
        EvalTarget(label=label, checkpoint=Path(path), output_dir=output_root / label)
        for label, path in checkpoint_specs
    ]


def ensure_checkpoints(
    targets: List[EvalTarget], stage: str, config: str, args: argparse.Namespace
) -> None:
    missing = [t for t in targets if not checkpoint_present(t.checkpoint, stage)]
    if not missing:
        print("All expected checkpoints are present; skipping training.")
        return

    print("Missing checkpoints:")
    for target in missing:
        print(f"  - {target.label}: {target.checkpoint}")

    if args.no_train_if_missing:
        raise SystemExit("Missing checkpoints and --no-train-if-missing was set.")

    run([args.python, "train.py", "--config", config], args.dry_run)
    if args.dry_run:
        return

    still_missing = [t for t in targets if not checkpoint_present(t.checkpoint, stage)]
    if still_missing:
        names = ", ".join(f"{t.label} ({t.checkpoint})" for t in still_missing)
        raise SystemExit(f"Training finished, but expected checkpoints are still missing: {names}")


def generate_and_score(target: EvalTarget, config: str, args: argparse.Namespace) -> Path:
    predictions = target.output_dir / "predictions_full_heldout.jsonl"
    metrics_stem = target.output_dir / "metrics_full_heldout"
    if not args.dry_run:
        target.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        cmd = [
            args.python,
            "scripts/generate_heldout_records.py",
            "--config",
            config,
            "--checkpoint",
            str(target.checkpoint),
            "--records-json",
            args.records_json,
            "--image-dir",
            args.image_dir,
            "--output",
            str(predictions),
            "--num-samples",
            str(args.num_samples),
            "--seed",
            str(args.seed),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--device",
            args.device,
        ]
        if args.resume:
            cmd.append("--resume")
        if args.overwrite:
            cmd.append("--overwrite")
        run(cmd, args.dry_run)

    if not args.skip_score:
        cmd = [
            args.python,
            "scripts/score_predictions.py",
            "--predictions",
            str(predictions),
            "--records-json",
            args.records_json,
            "--label",
            target.label,
            "--out",
            str(metrics_stem),
            "--device",
            args.score_device,
            "--nli-model",
            args.nli_model,
            "--sbert-model",
            args.sbert_model,
        ]
        if args.no_nli:
            cmd.append("--no-nli")
        if args.no_semantic:
            cmd.append("--no-semantic")
        run(cmd, args.dry_run)

    return Path(f"{metrics_stem}.json")


def compare_metrics(metric_paths: Iterable[Path], labels: Iterable[str], args: argparse.Namespace) -> None:
    if args.skip_compare:
        return
    comparison_dir = Path(args.output_root) / "comparison"
    if not args.dry_run:
        comparison_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "scripts/compare_metrics.py",
        *[str(path) for path in metric_paths],
        "--labels",
        ",".join(labels),
        "--out",
        str(comparison_dir / "full_heldout_comparison"),
        "--split-rows",
    ]
    run(cmd, args.dry_run)


def reproduce_note(stage: str, config: str, targets: List[EvalTarget], args: argparse.Namespace) -> str:
    target_lines = "\n".join(f"- {t.label}: {t.checkpoint}" for t in targets)
    return f"""# AstraQ-VL full held-out evaluation

Stage: {stage}
Config: {config}
Records: {args.records_json}
Images: {args.image_dir}
Targets:
{target_lines}

Run:
python scripts/run_full_heldout_eval.py --stage {stage} --num-samples {args.num_samples} --resume --package

Metrics are produced by scripts/score_predictions.py and include ROUGE-L, token-F1, exact match,
specificity hallucination, NLI consistency, contradiction rate, and SBERT cosine when enabled.
"""


def package_outputs(stage: str, config: str, targets: List[EvalTarget], args: argparse.Namespace) -> None:
    if not args.package:
        return
    package_path = Path(args.output_root) / SUITES[stage]["package"]
    print(f"Packaging {package_path}")
    if args.dry_run:
        return
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in (Path(config), Path(args.records_json)):
            if path.exists():
                zf.write(path, path.name)
        for target in targets:
            for file_path in sorted(target.output_dir.glob("*")):
                if file_path.is_file():
                    zf.write(file_path, f"{target.label}/{file_path.name}")
        comparison_dir = Path(args.output_root) / "comparison"
        if comparison_dir.exists():
            for file_path in sorted(comparison_dir.glob("*")):
                if file_path.is_file():
                    zf.write(file_path, f"comparison/{file_path.name}")
        zf.writestr("REPRODUCE_FULL_HELDOUT.md", reproduce_note(stage, config, targets, args))
    print(f"Wrote {package_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full held-out caption+QA evaluation.")
    parser.add_argument("--stage", choices=["auto", "stage1", "stage2"], default="auto")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument(
        "--checkpoint",
        action="append",
        help="Custom checkpoint as LABEL=PATH or checkpoint-NAME. Can be repeated.",
    )
    parser.add_argument("--records-json", default="datasets/astrollava_llava/test.json")
    parser.add_argument("--image-dir", default="datasets/astrollava_llava/images")
    parser.add_argument("--output-root", default="eval_runs/full_heldout")
    parser.add_argument("--num-samples", type=int, default=0, help="0 means all records.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-device", default="cuda")
    parser.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-train-if-missing", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = resolve_stage(args.stage, args.config)
    config = args.config or SUITES[stage]["config"]
    targets = build_targets(args, stage)

    print(f"Stage: {stage}")
    print(f"Config: {config}")
    for target in targets:
        status = "present" if checkpoint_present(target.checkpoint, stage) else "missing"
        print(f"Target {target.label}: {target.checkpoint} [{status}]")

    ensure_checkpoints(targets, stage, config, args)
    metric_paths = [generate_and_score(target, config, args) for target in targets]
    compare_metrics(metric_paths, [target.label for target in targets], args)
    package_outputs(stage, config, targets, args)


if __name__ == "__main__":
    main()
