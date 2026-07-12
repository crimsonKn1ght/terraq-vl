import argparse
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


IMAGE_TOKEN = "<image>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LLaVA-ReCap-558K and export it for this VLM repo."
    )
    parser.add_argument(
        "--dataset-name",
        default="lmms-lab/LLaVA-ReCap-558K",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--config-name",
        default="default",
        help="Dataset config/subset name.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to export.")
    parser.add_argument(
        "--output-dir",
        default="datasets/llava_recap_558k",
        help="Directory where train.json and images/ will be written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample limit. Use this first for a small smoke test.",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Image format to save locally.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download through the normal datasets cache instead of streaming rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing exported JSON file.",
    )
    parser.add_argument(
        "--update-config",
        default=None,
        help="Optional path to a YAML config to update with the exported dataset paths.",
    )
    return parser.parse_args()


def safe_filename(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or fallback


def image_extension(image_format: str) -> str:
    return "jpg" if image_format == "jpg" else "png"


def save_image(image_value: Any, output_path: Path, image_format: str) -> None:
    if output_path.exists():
        return

    image: Optional[Image.Image] = None

    if isinstance(image_value, Image.Image):
        image = image_value
    elif isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            image = Image.open(BytesIO(image_value["bytes"]))
        elif image_value.get("path"):
            image = Image.open(image_value["path"])
    elif isinstance(image_value, (str, os.PathLike)):
        image = Image.open(image_value)

    if image is None:
        raise ValueError(f"Unsupported image value type: {type(image_value)!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "jpg":
        image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=95)
    else:
        image.save(output_path, format="PNG")


def to_plain_conversations(value: Any) -> List[Dict[str, str]]:
    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, list):
        raise ValueError("conversations must be a list")

    conversations: List[Dict[str, str]] = []
    for turn in value:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip()
        text = str(turn.get("value", ""))
        if role and text:
            conversations.append({"from": role, "value": text})

    if not conversations:
        raise ValueError("no valid conversation turns found")

    has_image_token = any(IMAGE_TOKEN in turn["value"] for turn in conversations)
    if not has_image_token:
        conversations[0]["value"] = f"{IMAGE_TOKEN}\n{conversations[0]['value']}"

    return conversations


def row_to_record(row: Dict[str, Any], image_name: str) -> Dict[str, Any]:
    return {
        "id": str(row.get("id", Path(image_name).stem)),
        "image": image_name,
        "conversations": to_plain_conversations(row["conversations"]),
    }


def iter_rows(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    return load_dataset(
        args.dataset_name,
        args.config_name,
        split=args.split,
        streaming=not args.no_streaming,
    )


def update_yaml_config(config_path: Path, train_json: Path, image_dir: Path) -> None:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("data", {})
    config["data"]["train_data_path"] = train_json.as_posix()
    config["data"]["image_dir"] = image_dir.as_posix()

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    train_json = output_dir / f"{args.split}.json"
    temp_json = output_dir / f"{args.split}.json.tmp"

    if train_json.exists() and not args.overwrite:
        raise SystemExit(
            f"{train_json} already exists. Pass --overwrite to rebuild it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = iter_rows(args)
    total = args.max_samples
    written = 0
    skipped = 0
    extension = image_extension(args.image_format)

    with temp_json.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        for idx, row in enumerate(tqdm(rows, total=total, desc="Exporting")):
            if args.max_samples is not None and idx >= args.max_samples:
                break

            try:
                row_id = safe_filename(row.get("id"), f"sample_{idx:09d}")
                image_name = f"{idx:09d}_{row_id}.{extension}"
                image_path = image_dir / image_name

                save_image(row["image"], image_path, args.image_format)
                record = row_to_record(row, image_name)

                if not first:
                    f.write(",\n")
                json.dump(record, f, ensure_ascii=False)
                first = False
                written += 1
            except Exception as exc:
                skipped += 1
                print(f"Skipping row {idx}: {exc}")

        f.write("\n]\n")

    temp_json.replace(train_json)

    if args.update_config:
        update_yaml_config(Path(args.update_config), train_json, image_dir)

    print("\nExport complete")
    print(f"Dataset: {args.dataset_name}/{args.config_name}:{args.split}")
    print(f"Rows written: {written}")
    print(f"Rows skipped: {skipped}")
    print(f"JSON: {train_json}")
    print(f"Images: {image_dir}")
    if args.update_config:
        print(f"Updated config: {Path(args.update_config).resolve()}")


if __name__ == "__main__":
    main()
