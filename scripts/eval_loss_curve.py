"""Recompute a clean loss curve from saved checkpoints — no retraining, no training logs needed.

Each checkpoint's meta.json holds only a single-micro-batch loss (very noisy). This script instead
loads every checkpoint (connector + LoRA) and computes the teacher-forced **answer-token loss over a
FIXED set of samples** — the same samples for every checkpoint — so the curve is smooth and directly
comparable across steps. Run on the held-out test.json (default intent) it is a proper *validation*
loss curve.

Base models are loaded once; only the connector + LoRA adapter are swapped per checkpoint, so this
is fast (a forward pass over N samples per checkpoint, no backward).

Usage (from repo root, on a GPU):
    python scripts/eval_loss_curve.py \
        --config configs/finetune_astraq_vl_stage2.yaml \
        --checkpoint-dir checkpoints/astraq-vl-stage2 \
        --records-json datasets/astrollava_llava/test.json \
        --image-dir datasets/astrollava_llava/images \
        --num-samples 512 --out eval_loss_curve --plot

Outputs eval_loss_curve.csv / .json (columns: step, loss, n_samples) and, with --plot, a PNG.
Works for Stage-1 checkpoints too (no lora/ subdir -> LoRA load is skipped).
"""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Allow `import ...` of repo packages + sibling script when run as `python scripts/eval_loss_curve.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vlm_model.vlm import VLMForCausalLM  # noqa: E402
from vlm_model.utils import IGNORE_INDEX  # noqa: E402
from data.dataset import LLaVAPretrainDataset  # noqa: E402
from data.collator import VLMDataCollator  # noqa: E402
from training.checkpoint import load_connector_checkpoint, load_lora_adapter  # noqa: E402
from plot_training_curve import write_outputs, summarize, plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute a loss curve by evaluating each checkpoint.")
    p.add_argument("--config", required=True, help="Stage-2 (or Stage-1) config YAML.")
    p.add_argument("--checkpoint-dir", required=True, help="Dir containing checkpoint-*/ subdirs.")
    p.add_argument("--records-json", required=True, help="LLaVA-format records to score (e.g. test.json).")
    p.add_argument("--image-dir", required=True, help="Directory of the images referenced by records.")
    p.add_argument("--num-samples", type=int, default=512, help="Fixed sample count (0 = all records).")
    p.add_argument("--batch-size", type=int, default=8, help="Eval batch size (no grad, can exceed training).")
    p.add_argument("--seed", type=int, default=42, help="Seed for the fixed sample subset.")
    p.add_argument("--out", default="eval_loss_curve", help="Output stem (.csv/.json/.png).")
    p.add_argument("--plot", action="store_true", help="Also render a PNG (needs matplotlib).")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def checkpoint_dirs(root: str) -> list:
    dirs = [d for d in Path(root).glob("checkpoint-*") if (d / "connector.safetensors").exists()]
    return sorted(dirs, key=lambda d: int(d.name.split("-")[1]))


@torch.no_grad()
def eval_loss(model, loader, device: str) -> tuple:
    """Token-weighted mean answer-token loss over the loader. Returns (loss, n_label_tokens)."""
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    total_loss, total_tokens = 0.0, 0
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
        for batch in loader:
            labels = batch["labels"].to(device)
            n = int((labels != IGNORE_INDEX).sum().item())
            if n == 0:
                continue
            out = model(
                input_ids=batch["input_ids"].to(device),
                images=batch["images"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels,
            )
            # out.loss is the mean over valid tokens in this batch; weight by token count so the
            # aggregate is a true mean (and identical sampling across checkpoints keeps it comparable).
            total_loss += float(out.loss.item()) * n
            total_tokens += n
    return (total_loss / total_tokens if total_tokens else float("nan")), total_tokens


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ckpts = checkpoint_dirs(args.checkpoint_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoint-*/connector.safetensors under {args.checkpoint_dir}")
    print(f"Found {len(ckpts)} checkpoints: {', '.join(d.name for d in ckpts)}")

    print("Building base model (CLIP + LLM) once...")
    model = VLMForCausalLM(config)
    model = model.to(args.device)
    model.eval()
    is_lora = getattr(model.language_model, "is_lora", False)

    # Fixed, seeded subset -> the SAME samples are scored for every checkpoint.
    dataset = LLaVAPretrainDataset(
        data_path=args.records_json,
        image_dir=args.image_dir,
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        image_token_id=model.image_token_id,
        max_length=config.get("data", {}).get("max_length", 512),
    )
    indices = list(range(len(dataset)))
    if args.num_samples and 0 < args.num_samples < len(indices):
        random.Random(args.seed).shuffle(indices)
        indices = sorted(indices[: args.num_samples])
    collator = VLMDataCollator(
        tokenizer=model.tokenizer,
        max_length=config.get("data", {}).get("max_length", 512),
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    print(f"Scoring {len(indices)} samples per checkpoint (batch {args.batch_size}).")

    rows = []
    for ck in ckpts:
        load_connector_checkpoint(model.connector, str(ck))
        if is_lora:
            load_lora_adapter(model.language_model.model, str(ck))
        model.eval()
        step = int(ck.name.split("-")[1])
        loss, n_tok = eval_loss(model, loader, args.device)
        rows.append({"step": step, "loss": round(loss, 6), "n_samples": len(indices)})
        print(f"  {ck.name}: loss {loss:.4f}  ({n_tok} answer tokens)")

    rows.sort(key=lambda r: r["step"])
    write_outputs(rows, args.out)
    summarize(rows)
    if args.plot:
        plot(rows, args.out)


if __name__ == "__main__":
    main()
