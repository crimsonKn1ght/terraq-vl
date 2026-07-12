"""Sample records for human or LLM-as-judge evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from heldout_artifacts import (
    ANSWER_KEYS,
    PREDICTION_KEYS,
    get_first,
    load_per_sample_artifacts,
    prompt_for_key,
    record_type,
    rows_by_key,
)


RUBRIC = {
    "faithfulness_1_5": "1=contradicts or fabricates major claims; 5=fully supported by the reference.",
    "hallucination": "yes/no: includes unsupported specific names, instruments, dates, measurements, or objects.",
    "visual_relevance_1_5": "1=not relevant to the image/prompt; 5=directly relevant.",
    "answer_correctness_1_5": "1=wrong; 5=correct and complete for the prompt.",
}


def parse_artifacts(specs: List[str]) -> Dict[str, Mapping[Tuple[str, ...], Mapping[str, Any]]]:
    if not specs:
        raise SystemExit("Pass at least one --artifact PATH or --artifact LABEL=PATH.")
    models = {}
    for spec in specs:
        for artifact in load_per_sample_artifacts(spec):
            if artifact.label in models:
                raise SystemExit(f"Duplicate label {artifact.label!r}; use LABEL=PATH to disambiguate.")
            models[artifact.label] = rows_by_key(artifact.rows)
    return models


def stratified_keys(
    common_keys: List[Tuple[str, ...]],
    rows: Mapping[Tuple[str, ...], Mapping[str, Any]],
    sample_size: int,
    seed: int,
) -> List[Tuple[str, ...]]:
    rng = random.Random(seed)
    by_split: Dict[str, List[Tuple[str, ...]]] = {"caption": [], "qa": []}
    for key in common_keys:
        by_split.setdefault(record_type(rows[key]), []).append(key)

    caption_target = min(len(by_split.get("caption", [])), sample_size // 2)
    qa_target = sample_size - caption_target
    if qa_target > len(by_split.get("qa", [])):
        qa_target = len(by_split.get("qa", []))
        caption_target = min(len(by_split.get("caption", [])), sample_size - qa_target)

    chosen = []
    for split, count in (("caption", caption_target), ("qa", qa_target)):
        candidates = by_split.get(split, [])
        if count:
            chosen.extend(rng.sample(candidates, count))
    rng.shuffle(chosen)
    return chosen


def build_rows(
    models: Mapping[str, Mapping[Tuple[str, ...], Mapping[str, Any]]],
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    labels = sorted(models)
    common = set(models[labels[0]])
    for label in labels[1:]:
        common &= set(models[label])
    if not common:
        raise SystemExit("No common records across supplied artifacts.")

    reference_rows = models[labels[0]]
    chosen = stratified_keys(sorted(common), reference_rows, sample_size, seed)
    output_rows = []
    for sample_idx, key in enumerate(chosen, 1):
        first = reference_rows[key]
        for model_idx, label in enumerate(labels, 1):
            row = models[label][key]
            output_rows.append(
                {
                    "sample_id": f"judge_{sample_idx:04d}",
                    "model_slot": f"model_{model_idx}",
                    "model_label": label,
                    "image": row.get("image"),
                    "record_type": record_type(row),
                    "prompt": prompt_for_key(row),
                    "reference": get_first(row, ANSWER_KEYS) or "",
                    "prediction": get_first(row, PREDICTION_KEYS) or "",
                    "faithfulness_1_5": "",
                    "hallucination": "",
                    "visual_relevance_1_5": "",
                    "answer_correctness_1_5": "",
                    "judge_notes": "",
                }
            )
    return output_rows


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_rubric(path: Path, rows: List[Mapping[str, Any]], labels: List[str]) -> None:
    sample_count = len({row["sample_id"] for row in rows})
    lines = [
        "# Judge sample rubric",
        "",
        f"Records sampled: {sample_count}",
        f"Model outputs per record: {', '.join(labels)}",
        "",
        "Score each model output independently against the reference and, when available, the image.",
        "For LLM-as-judge, keep model labels hidden or shuffled if you want a blind comparison.",
        "",
        "## Fields",
        "",
    ]
    for field, description in RUBRIC.items():
        lines.append(f"- `{field}`: {description}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a human/LLM judge sample from per-sample artifacts.")
    parser.add_argument("--artifact", action="append", required=True, help="Artifact as PATH or LABEL=PATH.")
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_runs/full_heldout/analysis/judge_sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_artifacts(args.artifact)
    rows = build_rows(models, args.sample_size, args.seed)
    out_stem = Path(args.out)
    write_csv(Path(f"{out_stem}.csv"), rows)
    write_jsonl(Path(f"{out_stem}.jsonl"), rows)
    write_rubric(Path(f"{out_stem}.rubric.md"), rows, sorted(models))
    print(Path(f"{out_stem}.rubric.md").read_text(encoding="utf-8"))
    print(f"Wrote {out_stem}.csv, {out_stem}.jsonl, and {out_stem}.rubric.md")


if __name__ == "__main__":
    main()
