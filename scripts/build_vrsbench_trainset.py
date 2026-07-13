"""Build a remote-sensing training set (LLaVA format) from VRSBench.

``xiang709/VRSBench`` (https://huggingface.co/datasets/xiang709/VRSBench) is a versatile
remote-sensing vision-language benchmark: ~29.6k aerial/satellite images (DOTA/DIOR sources
via GoogleEarth) each paired with a human-verified detailed caption and multiple visual
question-answer turns. The training split ships as a single LLaVA-format file
``VRSBench_train.json`` plus an ``Images_train.zip`` archive.

This script materializes the images and emits the ``train.json`` + ``images/`` layout that
``data/dataset.py`` / ``train.py`` expect — the same output shape as
``scripts/build_astrollava_trainset.py`` (the astronomy backbone builder this is modeled on).

VRSBench's training records are one-per-image with a multi-turn ``conversations`` list (a
caption turn followed by VQA turns). This repo's tokenizer (``data/conversation.py``) keeps
only the last turn, so multi-turn convos are flattened into single-turn (human, gpt) records —
one caption record and one record per VQA turn, all sharing the same image and split bucket.

Run from the repo root:

    # Full build with a 2% held-out test split (downloads VRSBench_train.json + Images_train.zip ~8.4 GB):
    python scripts/build_vrsbench_trainset.py --output-dir datasets/vrsbench_llava --test-fraction 0.02

    # Three-way split for in-training validation: hold out 4% and split it evenly into val/test
    # (~2% val.json for the trainer's validation loss, ~2% test.json kept untouched for final eval).
    # The training set is byte-identical to a --test-fraction 0.04 test-only build (val_rng only
    # re-partitions the held-out side), so a Stage-1 connector trained on it stays valid.
    python scripts/build_vrsbench_trainset.py --output-dir datasets/vrsbench_llava \
        --test-fraction 0.04 --val-fraction 0.5

    # Quick smoke test on 50 source images first (still needs the image zip, or use --images-dir):
    python scripts/build_vrsbench_trainset.py --output-dir datasets/vrsbench_llava --max-samples 50

    # Reuse an already-downloaded/extracted copy instead of pulling from the Hub:
    python scripts/build_vrsbench_trainset.py --output-dir datasets/vrsbench_llava \
        --json-path /path/VRSBench_train.json --images-dir /path/Images_train

Notes:
- VRSBench is released for research use; keep its attribution/license if you redistribute.
- Box coordinates in some VQA answers are normalized to 0-100 (per the VRSBench paper); they are
  kept verbatim as text — no coordinate handling is needed for caption/VQA text alignment.
"""

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from random import Random

from PIL import Image as PILImage
from tqdm import tqdm

# Remote-sensing frames are modest, but disable PIL's decompression-bomb guard to be safe on any
# oversized tiles (matches the astronomy builder's behavior on a trusted, curated dataset).
PILImage.MAX_IMAGE_PIXELS = None

IMAGE_TOKEN = "<image>"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

DEFAULT_HF_ID = "xiang709/VRSBench"
TRAIN_JSON_NAME = "VRSBench_train.json"
IMAGES_ZIP_NAME = "Images_train.zip"

# Keys VRSBench (and other LLaVA-format sets) may use for the image filename, tried in order.
IMAGE_KEYS = ("image", "image_id", "img", "file_name", "filename")


def get_image_name(record: dict):
    for key in IMAGE_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    return None


def normalize_turns(conversations) -> list:
    """Return a list of (role, text) with role in {'human','gpt'}; [] if unparseable.

    Handles both the list-of-dicts form ([{"from":..,"value":..}, ...]) and the dict-of-lists
    form ({"from":[...], "value":[...]}) so the builder is robust to either export style.
    """
    if conversations is None:
        return []

    raw = []
    if isinstance(conversations, dict) and "from" in conversations and "value" in conversations:
        raw = list(zip(conversations["from"], conversations["value"]))
    elif isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and "from" in turn and "value" in turn:
                raw.append((turn["from"], turn["value"]))

    turns = []
    for role, text in raw:
        role = "human" if str(role).strip().lower() in ("human", "user") else "gpt"
        turns.append((role, str(text)))
    return turns


def clean_question(text: str) -> str:
    return text.replace(IMAGE_TOKEN, "").strip()


def single_turn_records(conversations, pair_id: str, image_name: str) -> list:
    """Flatten a (possibly multi-turn) conversation into single-turn (human, gpt) records."""
    turns = normalize_turns(conversations)
    records = []
    pending_q = None
    n = 0
    for role, text in turns:
        if role == "human":
            pending_q = clean_question(text)
        elif role == "gpt" and pending_q is not None and text.strip():
            records.append(
                {
                    "id": f"{pair_id}_t{n}",
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


class ImageSource:
    """Resolve image bytes either from an extracted folder or lazily from the VRSBench zip."""

    def __init__(self, images_dir: str = None, images_zip: str = None):
        self.images_dir = Path(images_dir) if images_dir else None
        self._zip = zipfile.ZipFile(images_zip) if images_zip else None
        # basename -> full entry name, so we tolerate any internal folder prefix in the archive.
        self._zip_index = {}
        if self._zip is not None:
            for name in self._zip.namelist():
                if name.lower().endswith(IMAGE_EXTS):
                    self._zip_index[os.path.basename(name).lower()] = name

    def read(self, image_name: str) -> PILImage.Image:
        base = os.path.basename(image_name)
        if self.images_dir is not None:
            path = self.images_dir / base
            if not path.exists():  # allow a nested layout
                matches = list(self.images_dir.rglob(base))
                if not matches:
                    raise FileNotFoundError(f"{base} not found under {self.images_dir}")
                path = matches[0]
            return PILImage.open(path)
        if self._zip is not None:
            entry = self._zip_index.get(base.lower())
            if entry is None:
                raise FileNotFoundError(f"{base} not present in the image zip")
            return PILImage.open(io.BytesIO(self._zip.read(entry)))
        raise RuntimeError("No image source configured (need --images-dir, --images-zip, or Hub download).")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None


def resolve_inputs(args) -> tuple:
    """Return (train_json_path, ImageSource, downloaded_zip_path). Downloads from the Hub unless overridden.

    ``downloaded_zip_path`` is set only when this script fetched the image zip itself, so
    ``--cleanup-zip`` never deletes a ``--images-zip`` the caller supplied.
    """
    json_path = args.json_path
    images_zip = args.images_zip
    images_dir = args.images_dir
    downloaded_zip = None

    need_download = json_path is None or (images_dir is None and images_zip is None and not args.no_images)
    if need_download:
        from huggingface_hub import hf_hub_download  # lazy: only needed for the Hub path

        if json_path is None:
            print(f"Downloading {TRAIN_JSON_NAME} from {args.hf_id} ...")
            json_path = hf_hub_download(args.hf_id, TRAIN_JSON_NAME, repo_type="dataset")
        if images_dir is None and images_zip is None and not args.no_images:
            print(f"Downloading {IMAGES_ZIP_NAME} from {args.hf_id} (~8.4 GB, first run only) ...")
            images_zip = hf_hub_download(args.hf_id, IMAGES_ZIP_NAME, repo_type="dataset")
            downloaded_zip = images_zip

    source = None
    if not args.no_images:
        source = ImageSource(images_dir=images_dir, images_zip=images_zip)
    return json_path, source, downloaded_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export VRSBench as a LLaVA-format VLM training set."
    )
    parser.add_argument("--hf-id", default=DEFAULT_HF_ID, help="HF dataset id.")
    parser.add_argument("--split", default="train", help="Name for the emitted {split}.json.")
    parser.add_argument(
        "--output-dir", default="datasets/vrsbench_llava", help="Directory for {split}.json and images/."
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Cap source images (smoke test).")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Hold out this fraction of IMAGES from training (the held-out pool). Per-image and "
        "seeded by --seed, so an image's caption and all its VQA records stay together. 0.0 = none.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Of the HELD-OUT images (see --test-fraction), route this fraction to a disjoint "
        "validation split (val.json); the rest go to test.json. Carved from the held-out pool with a "
        "separate seeded RNG, so the train/held-out partition is IDENTICAL to a test-only build — "
        "adding validation only re-partitions the held-out test, it never touches the training data. "
        "e.g. --test-fraction 0.04 --val-fraction 0.5 -> ~2%% val + ~2%% test. 0.0 = no val split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the train/held-out split.")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="If set, downscale each image so its long side is at most this many pixels "
        "(CLIP only uses 224x224, so e.g. 512 shrinks disk with no quality loss for training).",
    )
    parser.add_argument("--json-path", default=None, help="Local VRSBench_train.json (skip download).")
    parser.add_argument("--images-zip", default=None, help="Local Images_train.zip (skip download).")
    parser.add_argument("--images-dir", default=None, help="Local extracted images folder (skip zip).")
    parser.add_argument("--no-images", action="store_true", help="Emit JSON only; do not materialize images.")
    parser.add_argument(
        "--cleanup-zip",
        action="store_true",
        help="After extracting images, delete the downloaded Images_train.zip to reclaim ~8 GB of disk "
        "(only affects a zip this script downloaded; a --images-zip / --images-dir you pass is left alone). "
        "Note: a later re-run will re-download the archive.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild {split}.json if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_rng = Random(f"{args.seed}-test-split")
    # Separate stream for the val/test sub-split so the train vs. held-out decision above is
    # unaffected by --val-fraction — the training set stays byte-identical to a test-only build.
    val_rng = Random(f"{args.seed}-val-split")

    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    train_json = output_dir / f"{args.split}.json"
    val_json = output_dir / "val.json"
    test_json = output_dir / "test.json"

    if train_json.exists() and not args.overwrite:
        raise SystemExit(f"{train_json} already exists. Pass --overwrite to rebuild it.")

    image_dir.mkdir(parents=True, exist_ok=True)

    json_path, source, downloaded_zip = resolve_inputs(args)
    print(f"Reading {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    train_records, val_records, test_records = [], [], []
    image_bucket = {}          # image_name -> "train" | "val" | "test" (decided once per image)
    saved_images = set()       # output basenames already written
    train_images = val_images = test_images = 0
    caption_like = qa_count = skipped = 0

    for idx, row in enumerate(tqdm(rows, desc="Exporting")):
        try:
            src_image = get_image_name(row)
            if not src_image:
                skipped += 1
                continue

            pair_id = str(row.get("id") or f"vrsbench_{args.split}_{idx}")
            out_name = f"{Path(src_image).stem}.jpg"

            # Route this image (and ALL of its records) to one side, so a held-out image never leaks.
            if src_image not in image_bucket:
                is_heldout = args.test_fraction > 0 and split_rng.random() < args.test_fraction
                if is_heldout:
                    # Carve validation out of the held-out pool (val_rng keeps train unaffected).
                    is_val = args.val_fraction > 0 and val_rng.random() < args.val_fraction
                    image_bucket[src_image] = "val" if is_val else "test"
                    if is_val:
                        val_images += 1
                    else:
                        test_images += 1
                else:
                    image_bucket[src_image] = "train"
                    train_images += 1
            bucket_name = image_bucket[src_image]
            bucket = {"val": val_records, "test": test_records}.get(bucket_name, train_records)

            # Materialize the image once (re-encoded JPEG, optionally downscaled).
            if source is not None and out_name not in saved_images:
                img = source.read(src_image).convert("RGB")
                if args.max_image_size:
                    img.thumbnail((args.max_image_size, args.max_image_size))
                img.save(image_dir / out_name, format="JPEG", quality=90)
                saved_images.add(out_name)

            recs = single_turn_records(row.get("conversations"), pair_id, out_name)
            bucket.extend(recs)
            # First turn in VRSBench is the caption ("Describe the image ..."); the rest are VQA.
            if recs:
                caption_like += 1
                qa_count += max(0, len(recs) - 1)
        except Exception as exc:  # skip unreadable rows rather than abort the export
            skipped += 1
            print(f"Skipping row {idx}: {exc}")

    with train_json.open("w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    if val_records:
        with val_json.open("w", encoding="utf-8") as f:
            json.dump(val_records, f, ensure_ascii=False, indent=2)
    if args.test_fraction > 0:
        with test_json.open("w", encoding="utf-8") as f:
            json.dump(test_records, f, ensure_ascii=False, indent=2)

    if source is not None:
        source.close()
    if args.cleanup_zip and downloaded_zip:
        # The referenced images are now extracted to images/, so the ~8 GB archive is dead weight.
        for path in {downloaded_zip, os.path.realpath(downloaded_zip)}:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                print(f"Could not delete {path}: {exc}")
        print(f"Removed the downloaded {IMAGES_ZIP_NAME} (reclaimed ~8 GB of disk).")

    print("\nExport complete")
    print(
        f"Source images:   {len(image_bucket)} "
        f"(train {train_images} / val {val_images} / test {test_images})"
    )
    print(f"Caption records: ~{caption_like}   VQA records: ~{qa_count}")
    print(f"Train: {len(train_records)} records -> {train_json}")
    if val_records:
        print(f"Val:   {len(val_records)} records -> {val_json}")
    if args.test_fraction > 0:
        print(f"Test:  {len(test_records)} records -> {test_json}")
    print(f"Rows skipped:    {skipped}")
    print(f"Images: {image_dir}")


if __name__ == "__main__":
    main()
    # Flush and exit hard: the datasets/hf_xet backend can throw a spurious error during
    # interpreter shutdown even though the export is already fully written (mirrors the
    # astronomy builder). os._exit sidesteps the buggy finalizer and returns a clean code.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
