"""Generate full held-out predictions with the original AstroLLaVA reference model.

The official AstroLLaVA README advertises ``from astrollava import AstroLLaVA``.
The repository itself is also an adapted LLaVA codebase, so this script first tries
that high-level API and falls back to the official ``llava`` loader when needed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_heldout_records import completed_ids, load_records, sample_records  # noqa: E402


DEFAULT_MODEL_ID = "UniverseTBD/AstroLLaVA"


def move_model_if_possible(model: Any, device: str) -> Any:
    if hasattr(model, "to"):
        try:
            return model.to(device)
        except Exception:  # noqa: BLE001 - third-party model wrappers vary
            return model
    return model


class AstroLLaVAApiBackend:
    """Backend for the high-level API shown in the official README."""

    def __init__(self, model_id: str, device: str):
        from astrollava import AstroLLaVA  # type: ignore

        self.model = AstroLLaVA.from_pretrained(model_id)
        self.model = move_model_if_possible(self.model, device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def generate(
        self,
        image_path: Path,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        generate_response = getattr(self.model, "generate_response")
        kwargs: Dict[str, Any] = {}
        try:
            params = inspect.signature(generate_response).parameters
            if "max_new_tokens" in params:
                kwargs["max_new_tokens"] = max_new_tokens
            elif "max_tokens" in params:
                kwargs["max_tokens"] = max_new_tokens
            if "temperature" in params:
                kwargs["temperature"] = temperature
        except (TypeError, ValueError):
            kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature}

        try:
            response = generate_response(image, prompt, **kwargs)
        except TypeError:
            response = generate_response(str(image_path), prompt, **kwargs)
        return str(response).strip()


class OfficialLLaVABackend:
    """Backend for the official AstroLLaVA repo's adapted LLaVA implementation."""

    def __init__(
        self,
        model_id: str,
        model_base: Optional[str],
        device: str,
        device_map: str,
        conv_mode: str,
        load_8bit: bool,
        load_4bit: bool,
        use_flash_attn: bool,
    ):
        import torch
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        self.torch = torch
        self.device = device
        self.conv_mode = conv_mode
        disable_torch_init()
        model_name = get_model_name_from_path(model_id)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_id,
            model_base,
            model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device_map=device_map,
            device=device,
            use_flash_attn=use_flash_attn,
        )
        self.model.eval()

    def generate(
        self,
        image_path: Path,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        import torch
        from PIL import Image
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        image = Image.open(image_path).convert("RGB")
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            image_token = DEFAULT_IMAGE_TOKEN

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], image_token + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)
        images_tensor = process_images(
            [image],
            self.image_processor,
            self.model.config,
        ).to(self.model.device, dtype=torch.float16)

        kwargs = {
            "images": images_tensor,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": 1.0,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        image_sizes = [image.size]
        with torch.inference_mode():
            try:
                output_ids = self.model.generate(
                    input_ids,
                    image_sizes=image_sizes,
                    **kwargs,
                )
            except TypeError:
                output_ids = self.model.generate(input_ids, **kwargs)

        generated = output_ids[:, input_ids.shape[1] :]
        response = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if stop_str and response.endswith(stop_str):
            response = response[: -len(stop_str)].strip()
        return response


def load_reference_model(args: argparse.Namespace) -> Any:
    if args.backend in {"auto", "astrollava"}:
        try:
            return AstroLLaVAApiBackend(args.model_id, args.device)
        except Exception as exc:  # noqa: BLE001
            if args.backend == "astrollava":
                raise RuntimeError(
                    "Failed to load the high-level `astrollava.AstroLLaVA` API. "
                    "Install/verify the official package or use --backend llava."
                ) from exc
            print(f"High-level astrollava API unavailable ({exc!r}); falling back to llava backend.")

    try:
        return OfficialLLaVABackend(
            model_id=args.model_id,
            model_base=args.model_base,
            device=args.device,
            device_map=args.device_map,
            conv_mode=args.conv_mode,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
            use_flash_attn=args.use_flash_attn,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load AstroLLaVA through the official llava backend. "
            "Install the official repo from requirements-astrollava-reference.txt "
            "and confirm the model id/base path."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate original AstroLLaVA predictions for held-out caption+QA records."
    )
    parser.add_argument("--records-json", required=True, help="Held-out test.json.")
    parser.add_argument("--image-dir", required=True, help="Directory containing held-out images.")
    parser.add_argument("--output", default="predictions_astrollava_reference_full_heldout.jsonl")
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
    parser.add_argument("--resume", action="store_true", help="Append only missing record ids.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted records and exit without loading AstroLLaVA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.load_8bit and args.load_4bit:
        raise SystemExit("--load-8bit and --load-4bit are mutually exclusive")

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
    done: Set[str] = completed_ids(output_path) if args.resume else set()
    pending = [r for r in records if r["id"] not in done]
    if done:
        print(f"Resuming {output_path}: {len(done)} rows already present, {len(pending)} pending")

    backend = load_reference_model(args)
    with output_path.open("a" if args.resume and output_path.exists() else "w", encoding="utf-8") as f:
        for i, rec in enumerate(pending, 1):
            row = {
                **rec,
                "model": args.model_id,
                "backend": args.backend,
                "response": "",
            }
            image_path = Path(args.image_dir) / rec["image"]
            try:
                if not image_path.exists():
                    raise FileNotFoundError(str(image_path))
                row["response"] = backend.generate(
                    image_path=image_path,
                    prompt=rec["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
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
