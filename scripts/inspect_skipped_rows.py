"""Inspect prediction rows that were not present in scored per-sample metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heldout_artifacts import (  # noqa: E402
    ANSWER_KEYS,
    PREDICTION_KEYS,
    ArtifactRows,
    clean_prompt,
    get_first,
    load_per_sample_artifacts,
    load_prediction_artifacts,
    record_type,
    row_key,
    rows_by_key,
    write_csv,
    write_json,
)
from score_predictions import find_reference, load_reference_maps  # noqa: E402


def pick_artifact(artifacts: List[ArtifactRows], label: Optional[str], kind: str) -> ArtifactRows:
    if label:
        matches = [artifact for artifact in artifacts if artifact.label == label]
        if not matches:
            loaded = ", ".join(artifact.label for artifact in artifacts)
            raise SystemExit(f"No {kind} artifact with label {label!r}. Loaded: {loaded}")
        if len(matches) > 1:
            raise SystemExit(f"Multiple {kind} artifacts matched label {label!r}.")
        return matches[0]
    if len(artifacts) != 1:
        loaded = ", ".join(artifact.label for artifact in artifacts)
        raise SystemExit(f"Pass --label to choose one {kind} artifact. Loaded: {loaded}")
    return artifacts[0]


def load_inputs(args: argparse.Namespace) -> tuple[ArtifactRows, ArtifactRows]:
    if args.artifact:
        predictions = load_prediction_artifacts(args.artifact)
        per_sample = load_per_sample_artifacts(args.artifact)
    else:
        if not args.predictions or not args.per_sample:
            raise SystemExit("Pass --artifact, or pass both --predictions and --per-sample.")
        predictions = load_prediction_artifacts(args.predictions, args.label)
        per_sample = load_per_sample_artifacts(args.per_sample, args.label)
    return (
        pick_artifact(predictions, args.label, "prediction"),
        pick_artifact(per_sample, args.label, "per-sample"),
    )


def reference_available(row: Mapping[str, Any], refs: Dict[str, Any]) -> bool:
    if get_first(row, ANSWER_KEYS):
        return True
    if not refs:
        return False
    prompt = str(row.get("prompt") or row.get("question") or "").strip()
    return find_reference(dict(row), refs, prompt) is not None


def skipped_reason(row: Mapping[str, Any], refs: Dict[str, Any]) -> str:
    reasons: List[str] = []
    if row.get("error"):
        reasons.append("generation_error")
    if not get_first(row, PREDICTION_KEYS):
        reasons.append("missing_prediction")
    if not reference_available(row, refs):
        reasons.append("missing_reference")
    if not reasons:
        reasons.append("not_in_scored_per_sample")
    return ";".join(reasons)


def compact_row(row: Mapping[str, Any], refs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": row.get("index"),
        "id": row.get("id"),
        "image": row.get("image"),
        "record_type": record_type(row),
        "prompt": clean_prompt(row.get("prompt") or row.get("question") or ""),
        "reason": skipped_reason(row, refs),
        "error": row.get("error"),
        "has_prediction": bool(get_first(row, PREDICTION_KEYS)),
        "has_reference": reference_available(row, refs),
    }


def write_jsonl(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_md(path: Path, label: str, prediction_n: int, scored_n: int, rows: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["reason"]] = counts.get(row["reason"], 0) + 1

    lines = [
        f"# Skipped rows: {label}",
        "",
        f"- Prediction rows: {prediction_n}",
        f"- Scored rows: {scored_n}",
        f"- Skipped rows: {len(rows)}",
        "",
        "## Reason counts",
        "",
        "| reason | n |",
        "| --- | ---: |",
    ]
    for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count} |")

    lines.extend(["", "## Skipped row preview", "", "| index | id | record_type | image | reason | error |", "| --- | --- | --- | --- | --- | --- |"])
    for row in rows[:50]:
        error = str(row.get("error") or "").replace("|", "\\|")
        lines.append(
            f"| {row.get('index')} | {row.get('id')} | {row.get('record_type')} | "
            f"{row.get('image')} | {row.get('reason')} | {error} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect rows skipped by score_predictions.py.")
    parser.add_argument("--artifact", default=None, help="ZIP or directory containing predictions and metrics.")
    parser.add_argument("--predictions", default=None, help="Direct predictions JSONL.")
    parser.add_argument("--per-sample", default=None, help="Direct metrics_full_heldout.per_sample.jsonl.")
    parser.add_argument("--records-json", default=None, help="Optional test.json for reference matching.")
    parser.add_argument("--label", default=None, help="Model label when an artifact contains multiple models.")
    parser.add_argument("--out", default="eval_runs/full_heldout/analysis/skipped_rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, per_sample = load_inputs(args)
    refs = load_reference_maps(args.records_json) if args.records_json else {}
    scored_keys = set(rows_by_key(per_sample.rows))

    skipped = [
        compact_row(row, refs)
        for row in predictions.rows
        if row_key(row) not in scored_keys
    ]

    out_stem = Path(args.out)
    write_json(Path(f"{out_stem}.json"), skipped)
    write_csv(Path(f"{out_stem}.csv"), skipped)
    write_jsonl(Path(f"{out_stem}.jsonl"), skipped)
    write_summary_md(Path(f"{out_stem}.md"), predictions.label, len(predictions.rows), len(per_sample.rows), skipped)
    print(Path(f"{out_stem}.md").read_text(encoding="utf-8"))
    print(f"Wrote {out_stem}.json, {out_stem}.csv, {out_stem}.jsonl, and {out_stem}.md")


if __name__ == "__main__":
    main()
