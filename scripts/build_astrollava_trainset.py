"""Build an astronomy training set (LLaVA format) from AstroLLaVA_convos.

``UniverseTBD/AstroLLaVA_convos`` (CC-BY-SA-4.0) is the training set behind the AstroLLaVA
paper (arXiv:2504.08583) — ~29.8k real astronomy images (NASA APOD / ESO / Hubble) with a
human-written ``caption`` and a multi-turn ``conversation``. This script materializes the
images and emits the ``train.json`` + ``images/`` layout that ``data/dataset.py`` /
``train.py`` expect.

Two record types are produced:
  * caption pairs (default, always on): human asks to describe, assistant answers with the
    human-written caption — clean image->text alignment.
  * QA pairs (``--include-qa``): each (human, assistant) turn of the conversation becomes its
    own single-turn record, because this repo's tokenizer (data/conversation.py) keeps only
    the last turn — so multi-turn convos must be flattened to single turns.

The dataset's ``conversation`` is a dict-of-lists ({"from": [...], "value": [...]}) whose
assistant role is "astrollava"; both are normalized to the repo's
{"from": "human"/"gpt", "value": ...} turns with a "<image>" token on the human side.

Run from the repo root:

    python scripts/build_astrollava_trainset.py --output-dir datasets/astrollava_llava
    python scripts/build_astrollava_trainset.py --output-dir datasets/astrollava_llava --include-qa

Use ``--max-samples 50`` first for a quick smoke test, then re-run with ``--overwrite``.
"""

import argparse
import io
import itertools
import json
import os
import random
import sys
from pathlib import Path

from datasets import Image as HFImage, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm

# Hubble/ESO frames can be hundreds of megapixels. This is a trusted, curated dataset, so
# disable PIL's decompression-bomb guard (which otherwise hard-errors mid-stream on big images).
PILImage.MAX_IMAGE_PIXELS = None

IMAGE_TOKEN = "<image>"

CAPTION_PROMPTS = [
    "Describe this astronomical image.",
    "What does this image show?",
    "Provide a detailed description of this image.",
    "Explain what is depicted in this astronomical image.",
]


def normalize_turns(conversation) -> list:
    """Return a list of (role, text) with role in {'human','gpt'}; [] if unparseable."""
    if conversation is None:
        return []

    raw = []
    if isinstance(conversation, dict) and "from" in conversation and "value" in conversation:
        raw = list(zip(conversation["from"], conversation["value"]))
    elif isinstance(conversation, list):
        for turn in conversation:
            if isinstance(turn, dict) and "from" in turn and "value" in turn:
                raw.append((turn["from"], turn["value"]))

    turns = []
    for role, text in raw:
        role = "human" if str(role).strip().lower() == "human" else "gpt"
        turns.append((role, str(text)))
    return turns


def clean_question(text: str) -> str:
    return text.replace(IMAGE_TOKEN, "").strip()


def decode_image(value):
    """Turn a datasets Image(decode=False) value (or a PIL image) into a PIL image."""
    if hasattr(value, "convert"):  # already a decoded PIL image
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return PILImage.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return PILImage.open(value["path"])
    raise ValueError("Unsupported image value from dataset row")


def qa_records_from_conversation(conversation, pair_id: str, image_name: str) -> list:
    """Flatten a multi-turn conversation into single-turn (human, gpt) records."""
    turns = normalize_turns(conversation)
    records = []
    pending_q = None
    n = 0
    for role, text in turns:
        if role == "human":
            pending_q = clean_question(text)
        elif role == "gpt" and pending_q is not None and text.strip():
            records.append(
                {
                    "id": f"{pair_id}_qa{n}",
                    "image": image_name,
                    "conversations": [
                        {"from": "human", "value": f"{IMAGE_TOKEN}\n{pending_q}"},
                        {"from": "gpt", "value": text.strip()},
                    ],
                }
            )
            n += 1
            pending_q = None
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AstroLLaVA_convos as a LLaVA-format VLM training set."
    )
    parser.add_argument("--hf-id", default="UniverseTBD/AstroLLaVA_convos", help="HF dataset id.")
    parser.add_argument("--split", default="train", help="Split to export.")
    parser.add_argument(
        "--output-dir",
        default="datasets/astrollava_llava",
        help="Directory for {split}.json and images/.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Cap rows (use for a smoke test)."
    )
    parser.add_argument(
        "--include-qa",
        action="store_true",
        help="Also emit single-turn records from the GPT-4 conversations (more samples).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Hold out this fraction of IMAGES as a disjoint test split (test.json). The split "
        "is per-image (an image's caption and QA records stay together) and seeded by --seed, so "
        "it is deterministic and reproducible. 0.0 = no test split (default).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for prompt selection / split.")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="If set, downscale each image so its long side is at most this many pixels. "
        "CLIP only uses 224x224, so e.g. 384 keeps quality while shrinking disk a lot.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download via the datasets cache instead of streaming rows.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild {split}.json if it exists."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    split_rng = random.Random(f"{args.seed}-test-split")

    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    train_json = output_dir / f"{args.split}.json"
    test_json = output_dir / "test.json"

    if train_json.exists() and not args.overwrite:
        raise SystemExit(f"{train_json} already exists. Pass --overwrite to rebuild it.")

    image_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Streaming {args.hf_id} split={args.split} "
        f"(cap={args.max_samples}, test_fraction={args.test_fraction})"
    )
    ds = load_dataset(args.hf_id, split=args.split, streaming=not args.no_streaming)
    # Decode images ourselves (decode=False) so a corrupt/oversized image is caught by the
    # per-row try/except below instead of crashing the whole dataset iterator mid-stream.
    ds = ds.cast_column("image", HFImage(decode=False))
    rows = iter(ds)
    if args.max_samples is not None:
        rows = itertools.islice(rows, args.max_samples)

    train_records = []
    test_records = []
    train_images = 0
    test_images = 0
    caption_count = 0
    qa_count = 0
    skipped = 0
    for idx, row in enumerate(tqdm(rows, total=args.max_samples, desc="Exporting")):
        try:
            pair_id = f"astrollava_{args.split}_{idx}"
            image_name = f"{pair_id}.jpg"
            image_path = image_dir / image_name
            if not image_path.exists():
                img = decode_image(row["image"]).convert("RGB")
                if args.max_image_size:
                    img.thumbnail((args.max_image_size, args.max_image_size))
                img.save(image_path, format="JPEG", quality=90)

            # Route this image (and ALL of its records) to one side, so a held-out image
            # never leaks across the train/test boundary.
            is_test = args.test_fraction > 0 and split_rng.random() < args.test_fraction
            bucket = test_records if is_test else train_records
            n_before = len(bucket)

            caption = (row.get("caption") or "").strip()
            if caption:
                bucket.append(
                    {
                        "id": pair_id,
                        "image": image_name,
                        "conversations": [
                            {"from": "human", "value": f"{IMAGE_TOKEN}\n{rng.choice(CAPTION_PROMPTS)}"},
                            {"from": "gpt", "value": caption},
                        ],
                    }
                )
                caption_count += 1

            if args.include_qa:
                qa = qa_records_from_conversation(row.get("conversation"), pair_id, image_name)
                bucket.extend(qa)
                qa_count += len(qa)

            if len(bucket) > n_before:
                if is_test:
                    test_images += 1
                else:
                    train_images += 1
        except Exception as exc:  # skip unreadable rows rather than abort the export
            skipped += 1
            print(f"Skipping row {idx}: {exc}")

    with train_json.open("w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    if args.test_fraction > 0:
        with test_json.open("w", encoding="utf-8") as f:
            json.dump(test_records, f, ensure_ascii=False, indent=2)

    print("\nExport complete")
    print(f"Caption records: {caption_count}")
    print(f"QA records:      {qa_count}")
    print(f"Train: {len(train_records)} records / {train_images} images -> {train_json}")
    if args.test_fraction > 0:
        print(f"Test:  {len(test_records)} records / {test_images} images -> {test_json}")
    print(f"Rows skipped:    {skipped}")
    print(f"Images: {image_dir}")


if __name__ == "__main__":
    main()
    # The `datasets` streaming backend (hf_xet / pyarrow worker threads) can abort during
    # interpreter shutdown with "PyGILState_Release: thread state must be current". The
    # export is fully flushed to disk by now, so exit hard to skip the buggy finalizer and
    # return a clean exit code (otherwise runpod_setup.sh sees a false failure).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
