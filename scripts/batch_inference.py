"""Caption many images with a single model load, writing predictions to JSONL.

Reuses inference.load_vlm / inference.run_inference, but loads the model ONCE (the per-image
CLI reloads ~5 GB every call). Output is one JSON object per line:
    {"image": ..., "prompt": ..., "response": ..., "reference": <optional ground-truth caption>}

Pass --records-json datasets/astrollava_llava/test.json to score EXACTLY the held-out test
images (and attach their reference captions); otherwise it samples from --image-dir.

Usage (from repo root) — held-out test set, all unseen images:
    python scripts/batch_inference.py \
        --config configs/pretrain_astraq_vl.yaml \
        --checkpoint checkpoints/astraq-vl-stage1/checkpoint-3786 \
        --image-dir datasets/astrollava_llava/images \
        --records-json datasets/astrollava_llava/test.json \
        --num-samples 0 --temperature 0 \
        --output predictions_test.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Allow `import inference` when run as `python scripts/batch_inference.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import load_vlm, run_inference  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif"}


def load_references(train_json: str) -> dict:
    """Map image filename -> ground-truth caption (the non-QA caption record)."""
    if not train_json or not Path(train_json).exists():
        return {}
    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    refs = {}
    for r in data:
        if "_qa" not in r["id"] and r.get("conversations"):
            refs.setdefault(r["image"], r["conversations"][1]["value"])
    return refs


def images_from_records(records_json: str):
    """Return (ordered unique image names, {image -> reference caption}) from a records JSON."""
    with open(records_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    names, seen, refs = [], set(), {}
    for r in data:
        img = r["image"]
        if img not in seen:
            seen.add(img)
            names.append(img)
        if "_qa" not in r["id"] and r.get("conversations"):
            refs.setdefault(img, r["conversations"][1]["value"])
    return names, refs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch image captioning with a single model load.")
    p.add_argument("--config", required=True, help="Training/inference config YAML.")
    p.add_argument("--checkpoint", required=True, help="Connector checkpoint dir.")
    p.add_argument("--image-dir", required=True, help="Directory of images to caption.")
    p.add_argument("--output", default="predictions.jsonl", help="Output JSONL path.")
    p.add_argument(
        "--num-samples", type=int, default=200,
        help="Random sample size; pass 0 to caption EVERY image in the dir (slow).",
    )
    p.add_argument("--prompt", default="Describe this astronomical image.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic greedy.")
    p.add_argument("--seed", type=int, default=42, help="Seed for the image sample.")
    p.add_argument(
        "--train-json", default=None,
        help="Optional train.json to attach ground-truth captions for side-by-side comparison.",
    )
    p.add_argument(
        "--records-json", default=None,
        help="Score exactly the images listed in this JSON (e.g. test.json for held-out eval), "
        "attaching their reference captions. Overrides directory sampling.",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    image_dir = Path(args.image_dir)
    if args.records_json:
        names, refs = images_from_records(args.records_json)
        image_paths = [image_dir / n for n in names if (image_dir / n).exists()]
    else:
        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        refs = load_references(args.train_json)

    if not image_paths:
        raise SystemExit(f"No images found ({args.records_json or args.image_dir})")
    if args.num_samples and args.num_samples < len(image_paths):
        image_paths = sorted(rng.sample(image_paths, args.num_samples))
    model = load_vlm(args.config, args.checkpoint, args.device)

    out_path = Path(args.output)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, img in enumerate(image_paths, 1):
            try:
                response = run_inference(
                    model=model,
                    image_path=str(img),
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=args.device,
                )
            except Exception as exc:  # keep going; record the failure
                response = f"<error: {exc}>"
            record = {"image": img.name, "prompt": args.prompt, "response": response}
            if img.name in refs:
                record["reference"] = refs[img.name]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            print(f"[{i}/{len(image_paths)}] {img.name}: {response[:80]}")

    print(f"\nWrote {written} predictions to {out_path}")


if __name__ == "__main__":
    main()
