"""Generate full held-out predictions with Qwen2.5-VL.

This is an external-baseline generator. It reads the same LLaVA-format
``test.json`` used by the AstraQ-VL checkpoints and writes JSONL rows compatible
with ``scripts/score_predictions.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_heldout_records import completed_ids, load_records, sample_records  # noqa: E402


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def parse_dtype(value: str) -> Any:
    if value == "auto":
        return "auto"
    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if value not in dtype_map:
        raise ValueError(f"Unsupported dtype {value!r}; use auto, bfloat16, float16, or float32")
    return dtype_map[value]


def load_qwen_vl(
    model_id: str,
    device: str,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str | None,
) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    kwargs: Dict[str, Any] = {"torch_dtype": parse_dtype(torch_dtype)}
    if device_map != "none":
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if device_map == "none":
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: --device starts with cuda, but torch.cuda.is_available() is false.")

    return model, processor


def build_messages(image_path: Path, prompt: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_one(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = build_messages(image_path, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = 0.9

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen2.5-VL predictions for all held-out caption+QA records."
    )
    parser.add_argument("--records-json", required=True, help="Held-out test.json.")
    parser.add_argument("--image-dir", required=True, help="Directory containing held-out images.")
    parser.add_argument("--output", default="predictions_qwen2_5_vl_7b_full_heldout.jsonl")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--num-samples", type=int, default=0, help="0 means all records.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto", help="Use 'none' to call model.to(device).")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional Transformers attention implementation, e.g. flash_attention_2.",
    )
    parser.add_argument("--resume", action="store_true", help="Append only missing record ids.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted records and exit without loading Qwen2.5-VL.",
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
            f"({caption_n} caption, {qa_n} qa) with {args.model_id}"
        )
        for row in records[: min(5, len(records))]:
            print(json.dumps(row, ensure_ascii=False))
        return

    if output_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(f"{output_path} exists. Pass --resume or --overwrite.")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output_path) if args.resume else set()
    pending = [r for r in records if r["id"] not in done]
    if done:
        print(f"Resuming {output_path}: {len(done)} rows already present, {len(pending)} pending")

    model, processor = load_qwen_vl(
        model_id=args.model_id,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    with output_path.open("a" if args.resume and output_path.exists() else "w", encoding="utf-8") as f:
        for i, rec in enumerate(pending, 1):
            row = {
                **rec,
                "model": args.model_id,
                "response": "",
            }
            image_path = Path(args.image_dir) / rec["image"]
            try:
                if not image_path.exists():
                    raise FileNotFoundError(str(image_path))
                row["response"] = generate_one(
                    model=model,
                    processor=processor,
                    image_path=image_path,
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
