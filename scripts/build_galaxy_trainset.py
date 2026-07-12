"""Build an astronomy training set (LLaVA format) from Galaxy10 DECaLS.

Galaxy10 DECaLS (``matthieulel/galaxy10_decals``) ships galaxy cutouts with a morphology
class label (0-9) but no free-text. This script materializes the images to disk and turns
each class label into an image-caption conversation, producing the ``train.json`` +
``images/`` layout that ``data/dataset.py`` / ``train.py`` expect:

    {
      "id": "galaxy_train_0",
      "image": "galaxy_train_0.png",
      "conversations": [
        {"from": "human", "value": "<image>\\nDescribe this galaxy."},
        {"from": "gpt",   "value": "A spiral galaxy with a central bar ..."}
      ]
    }

The class list / descriptions mirror ``corpus/galaxy_zoo.py`` (the RAG corpus adapter) so
both halves of the repo describe galaxies the same way. Run from the repo root:

    python scripts/build_galaxy_trainset.py --output-dir datasets/galaxy10_llava --split train

Use ``--max-samples 50`` first for a quick smoke test, then re-run with ``--overwrite`` for
the full split.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

IMAGE_TOKEN = "<image>"

# Galaxy10 DECaLS index -> class name (from the dataset card; mirrors corpus/galaxy_zoo.py).
GALAXY10_CLASSES = [
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge",
]

# Short morphological descriptions (the caption target).
GALAXY10_DESCRIPTIONS = {
    "Disturbed Galaxies": "An irregular galaxy with a distorted, asymmetric shape, often from gravitational interaction.",
    "Merging Galaxies": "Two or more galaxies colliding and merging, showing tidal tails and overlapping structure.",
    "Round Smooth Galaxies": "A smooth elliptical galaxy with a round, featureless light profile and no spiral arms.",
    "In-between Round Smooth Galaxies": "A smooth elliptical galaxy of intermediate, slightly flattened roundness.",
    "Cigar Shaped Smooth Galaxies": "A smooth elliptical galaxy with an elongated, cigar-like elliptical shape.",
    "Barred Spiral Galaxies": "A spiral galaxy with a central bar-shaped structure of stars from which spiral arms extend.",
    "Unbarred Tight Spiral Galaxies": "A spiral galaxy without a central bar, with tightly wound spiral arms.",
    "Unbarred Loose Spiral Galaxies": "A spiral galaxy without a central bar, with loosely wound, open spiral arms.",
    "Edge-on Galaxies without Bulge": "A disk galaxy seen edge-on as a thin line of light, lacking a prominent central bulge.",
    "Edge-on Galaxies with Bulge": "A disk galaxy seen edge-on with a prominent bright central bulge.",
}

# Natural singular phrasing of each (plural) class name, for "what type" answers.
GALAXY10_SINGULAR = {
    "Disturbed Galaxies": "a disturbed galaxy",
    "Merging Galaxies": "a pair of merging galaxies",
    "Round Smooth Galaxies": "a round, smooth elliptical galaxy",
    "In-between Round Smooth Galaxies": "an in-between round smooth galaxy",
    "Cigar Shaped Smooth Galaxies": "a cigar-shaped smooth galaxy",
    "Barred Spiral Galaxies": "a barred spiral galaxy",
    "Unbarred Tight Spiral Galaxies": "an unbarred tight spiral galaxy",
    "Unbarred Loose Spiral Galaxies": "an unbarred loose spiral galaxy",
    "Edge-on Galaxies without Bulge": "an edge-on galaxy without a bulge",
    "Edge-on Galaxies with Bulge": "an edge-on galaxy with a bulge",
}

# Instruction templates. "describe" answers with the morphology sentence; "classify"
# answers by naming the type first. Varied phrasing keeps the connector from latching
# onto a single prompt string.
DESCRIBE_PROMPTS = [
    "Describe this galaxy.",
    "Provide a short description of the galaxy in this image.",
    "What does this astronomical image show?",
    "Describe the morphology of the galaxy in the image.",
]
CLASSIFY_PROMPTS = [
    "What type of galaxy is shown in this image?",
    "Classify the morphology of this galaxy.",
    "Which galaxy morphology class does this image belong to?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Galaxy10 DECaLS as a LLaVA-format VLM training set."
    )
    parser.add_argument("--hf-id", default="matthieulel/galaxy10_decals", help="HF dataset id.")
    parser.add_argument("--split", default="train", help="Split to export (train/test).")
    parser.add_argument(
        "--output-dir",
        default="datasets/galaxy10_llava",
        help="Directory for {split}.json and images/.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Cap rows (use for a smoke test)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for prompt selection.")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download via the datasets cache instead of streaming rows.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild {split}.json if it exists."
    )
    return parser.parse_args()


def build_conversation(label: int, rng: random.Random) -> list:
    name = GALAXY10_CLASSES[label]
    desc = GALAXY10_DESCRIPTIONS[name]

    if rng.random() < 0.5:
        question = rng.choice(DESCRIBE_PROMPTS)
        answer = desc
    else:
        question = rng.choice(CLASSIFY_PROMPTS)
        answer = f"This image shows {GALAXY10_SINGULAR[name]}. {desc}"

    return [
        {"from": "human", "value": f"{IMAGE_TOKEN}\n{question}"},
        {"from": "gpt", "value": answer},
    ]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    train_json = output_dir / f"{args.split}.json"

    if train_json.exists() and not args.overwrite:
        raise SystemExit(f"{train_json} already exists. Pass --overwrite to rebuild it.")

    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {args.hf_id} split={args.split} (cap={args.max_samples})")
    ds = load_dataset(args.hf_id, split=args.split, streaming=not args.no_streaming)
    rows = iter(ds)
    if args.max_samples is not None:
        rows = itertools.islice(rows, args.max_samples)

    records = []
    skipped = 0
    for idx, row in enumerate(tqdm(rows, total=args.max_samples, desc="Exporting")):
        try:
            pair_id = f"galaxy_{args.split}_{idx}"
            image_name = f"{pair_id}.png"
            image_path = image_dir / image_name
            if not image_path.exists():
                row["image"].convert("RGB").save(image_path)

            records.append(
                {
                    "id": pair_id,
                    "image": image_name,
                    "conversations": build_conversation(int(row["label"]), rng),
                }
            )
        except Exception as exc:  # skip unreadable rows rather than abort the export
            skipped += 1
            print(f"Skipping row {idx}: {exc}")

    with train_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\nExport complete")
    print(f"Rows written: {len(records)}")
    print(f"Rows skipped: {skipped}")
    print(f"JSON:   {train_json}")
    print(f"Images: {image_dir}")


if __name__ == "__main__":
    main()
