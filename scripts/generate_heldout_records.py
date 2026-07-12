"""Generate predictions for every held-out LLaVA record.

Unlike ``batch_inference.py`` (one caption per unique image), this script walks every
record in ``test.json``: caption prompts and all QA prompts. It writes one JSONL row
per record with enough metadata for ``scripts/score_predictions.py`` to compute
overall, caption-only, and QA-only metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_TOKEN = "<image>"


def clean_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(IMAGE_TOKEN, " ")).strip()


def record_type(row: Dict[str, Any]) -> str:
    explicit = row.get("record_type") or row.get("task_type") or row.get("type")
    if explicit:
        value = str(explicit).lower()
        if "qa" in value or "question" in value or "vqa" in value:
            return "qa"
        if "caption" in value:
            return "caption"
    return "qa" if "_qa" in str(row.get("id", "")).lower() else "caption"


def extract_single_turn(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """Extract the last human/assistant pair, matching data.conversation behavior."""
    human_msg = ""
    assistant_msg = ""
    for turn in row.get("conversations") or []:
        role = str(turn.get("from", ""))
        value = str(turn.get("value", ""))
        if role == "human":
            human_msg = value
        elif role == "gpt":
            assistant_msg = value

    image = str(row.get("image", "")).strip()
    answer = assistant_msg.strip()
    if not image or not human_msg or not answer:
        return None

    return {
        "index": idx,
        "id": str(row.get("id") or f"record_{idx}"),
        "image": image,
        "record_type": record_type(row),
        "prompt": clean_prompt(human_msg),
        "reference": answer,
    }


def load_records(records_json: str) -> List[Dict[str, Any]]:
    with open(records_json, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    records = []
    for idx, row in enumerate(raw, 1):
        item = extract_single_turn(row, idx)
        if item:
            records.append(item)
    return records


def sample_records(records: List[Dict[str, Any]], num_samples: int, seed: int) -> List[Dict[str, Any]]:
    if num_samples <= 0 or num_samples >= len(records):
        return records
    chosen = set(random.Random(seed).sample(range(len(records)), num_samples))
    return [record for idx, record in enumerate(records) if idx in chosen]


def completed_ids(output_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("id")
            if row_id:
                done.add(str(row_id))
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions for all held-out caption+QA records."
    )
    parser.add_argument("--config", default=None, help="Training/inference config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory to load.")
    parser.add_argument("--records-json", required=True, help="Held-out test.json.")
    parser.add_argument("--image-dir", required=True, help="Directory containing held-out images.")
    parser.add_argument("--output", default="predictions_full_heldout.jsonl")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Deterministic record sample; 0 means all records.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Append only missing record ids.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted records and exit without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = sample_records(load_records(args.records_json), args.num_samples, args.seed)
    output_path = Path(args.output)

    if args.dry_run:
        caption_n = sum(1 for r in records if r["record_type"] == "caption")
        qa_n = sum(1 for r in records if r["record_type"] == "qa")
        print(
            f"Would generate {len(records)} records "
            f"({caption_n} caption, {qa_n} qa) from {args.records_json}"
        )
        for row in records[: min(5, len(records))]:
            print(json.dumps(row, ensure_ascii=False))
        return

    if not args.config or not args.checkpoint:
        raise SystemExit("--config and --checkpoint are required unless --dry-run is set")
    if output_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(f"{output_path} exists. Pass --resume or --overwrite.")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output_path) if args.resume else set()
    pending = [r for r in records if r["id"] not in done]
    if done:
        print(f"Resuming {output_path}: {len(done)} rows already present, {len(pending)} pending")

    from inference import load_vlm, run_inference  # noqa: WPS433 - keep dry-run lightweight

    model = load_vlm(args.config, args.checkpoint, args.device)
    with output_path.open("a" if args.resume and output_path.exists() else "w", encoding="utf-8") as f:
        for i, rec in enumerate(pending, 1):
            row = {
                **rec,
                "checkpoint": args.checkpoint,
                "config": args.config,
                "response": "",
            }
            image_path = Path(args.image_dir) / rec["image"]
            try:
                if not image_path.exists():
                    raise FileNotFoundError(str(image_path))
                row["response"] = run_inference(
                    model=model,
                    image_path=str(image_path),
                    prompt=rec["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=args.device,
                )
            except Exception as exc:  # noqa: BLE001 - keep long evals running
                row["error"] = repr(exc)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            status = "error" if row.get("error") else "ok"
            print(f"[{i}/{len(pending)}] {rec['id']} ({rec['record_type']}): {status}")

    print(f"Wrote {len(pending)} new rows to {output_path}")


if __name__ == "__main__":
    main()
