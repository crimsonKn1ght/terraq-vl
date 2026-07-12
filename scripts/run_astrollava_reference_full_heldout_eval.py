"""Run original AstroLLaVA full held-out generation, scoring, comparison, and packaging."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List


DEFAULT_LABEL = "astrollava_reference"
DEFAULT_MODEL_ID = "UniverseTBD/AstroLLaVA"
PACKAGE_NAME = "astrollava-reference-full-heldout-eval-v1.zip"


def run(cmd: List[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def reproduce_note(args: argparse.Namespace, output_dir: Path) -> str:
    return f"""# AstroLLaVA reference full held-out baseline

Model: {args.model_id}
Backend: {args.backend}
Records: {args.records_json}
Images: {args.image_dir}
Output directory: {output_dir}

Run:
python scripts/run_astrollava_reference_full_heldout_eval.py --num-samples {args.num_samples} --resume --package

This is a domain reference baseline with possible overlap against this repository's held-out split,
because the reference model was trained on the same AstroLLaVA data lineage. Treat it as a
domain comparator, not as a clean external benchmark.

Metrics are produced by scripts/score_predictions.py and include ROUGE-L, token-F1, exact match,
specificity hallucination, NLI consistency, contradiction rate, and SBERT cosine when enabled.
"""


def package_outputs(args: argparse.Namespace, output_dir: Path, comparison_dir: Path) -> None:
    if not args.package:
        return
    package_path = Path(args.output_root) / PACKAGE_NAME
    print(f"Packaging {package_path}")
    if args.dry_run:
        return
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        records_path = Path(args.records_json)
        if records_path.exists():
            zf.write(records_path, "test.json")
        for file_path in sorted(output_dir.glob("*")):
            if file_path.is_file():
                zf.write(file_path, f"{args.label}/{file_path.name}")
        if comparison_dir.exists():
            for file_path in sorted(comparison_dir.glob("*")):
                if file_path.is_file():
                    zf.write(file_path, f"comparison/{file_path.name}")
        zf.writestr("REPRODUCE_ASTROLLAVA_REFERENCE_FULL_HELDOUT.md", reproduce_note(args, output_dir))
    print(f"Wrote {package_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run original AstroLLaVA full held-out baseline eval.")
    parser.add_argument("--records-json", default="datasets/astrollava_llava/test.json")
    parser.add_argument("--image-dir", default="datasets/astrollava_llava/images")
    parser.add_argument("--output-root", default="eval_runs/full_heldout")
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--backend", choices=["auto", "astrollava", "llava"], default="auto")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--num-samples", type=int, default=0, help="0 means all records.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--score-device", default="cuda")
    parser.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
    output_dir = Path(args.output_root) / args.label
    comparison_dir = Path(args.output_root) / "comparison"
    predictions = output_dir / "predictions_full_heldout.jsonl"
    metrics_stem = output_dir / "metrics_full_heldout"
    metrics_json = Path(f"{metrics_stem}.json")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        cmd = [
            args.python,
            "scripts/generate_astrollava_reference_heldout.py",
            "--records-json",
            args.records_json,
            "--image-dir",
            args.image_dir,
            "--output",
            str(predictions),
            "--model-id",
            args.model_id,
            "--backend",
            args.backend,
            "--conv-mode",
            args.conv_mode,
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
            "--device-map",
            args.device_map,
        ]
        if args.model_base:
            cmd.extend(["--model-base", args.model_base])
        if args.load_8bit:
            cmd.append("--load-8bit")
        if args.load_4bit:
            cmd.append("--load-4bit")
        if args.use_flash_attn:
            cmd.append("--use-flash-attn")
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
            args.label,
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

    if not args.skip_compare:
        cmd = [
            args.python,
            "scripts/compare_metrics.py",
            str(metrics_json),
            "--labels",
            args.label,
            "--out",
            str(comparison_dir / "full_heldout_comparison"),
            "--split-rows",
        ]
        run(cmd, args.dry_run)

    package_outputs(args, output_dir, comparison_dir)


if __name__ == "__main__":
    main()
