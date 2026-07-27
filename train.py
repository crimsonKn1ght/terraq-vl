"""Training entry point: build the VLM from a YAML config and run Stage-1 / Stage-2 training.

    python train.py --config configs/pretrain_stage1.yaml        # Stage 1 (connector only)
    python train.py --config configs/finetune_..._stage2.yaml    # Stage 2 (connector + LoRA)
"""

import argparse
import logging
import os
import random

import yaml
import torch
from torch.utils.data import Subset
from accelerate import Accelerator

from vlm_model.vlm import VLMForCausalLM
from vlm_model.utils import count_trainable_parameters, count_total_parameters
from data.dataset import LLaVAPretrainDataset
from training.trainer import VLMTrainer
from training.checkpoint import load_connector_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train VLM Stage 1 - Feature Alignment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_stage1.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint dir to resume from (e.g. checkpoints/vrsbench-stage1/checkpoint-2300): "
        "restores the connector, optimizer, LR scheduler, and step counter, then continues the run.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.resume:
        config.setdefault("training", {})["resume_from"] = args.resume
        logger.info(f"Will resume training from {args.resume}")

    train_cfg = config.get("training", {})
    use_bf16 = train_cfg.get("bf16", True)

    accelerator = Accelerator(
        mixed_precision="bf16" if use_bf16 else "no",
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 32),
    )

    if accelerator.is_main_process:
        os.makedirs(train_cfg.get("output_dir", "./checkpoints"), exist_ok=True)

    logger.info("Building model...")
    model = VLMForCausalLM(config)

    # Stage 2: warm-start the connector from a Stage-1 checkpoint, then continue training it
    # alongside the LLM's LoRA adapters.
    stage1_checkpoint = config.get("stage1_checkpoint")
    if stage1_checkpoint:
        logger.info(f"Initializing connector from Stage-1 checkpoint: {stage1_checkpoint}")
        load_connector_checkpoint(model.connector, stage1_checkpoint)

    # Gradient checkpointing on the LLM (needed to fit the Stage-2 backward pass in ~48 GB).
    if train_cfg.get("gradient_checkpointing", False):
        logger.info("Enabling gradient checkpointing on the LLM")
        model.enable_gradient_checkpointing()

    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    logger.info(f"Trainable parameters: {trainable:,} ({trainable / total:.4%} of {total:,})")

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  [TRAINABLE] {name}: {param.shape}")

    data_cfg = config.get("data", {})
    logger.info("Building dataset...")
    dataset = LLaVAPretrainDataset(
        data_path=data_cfg["train_data_path"],
        image_dir=data_cfg["image_dir"],
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        image_token_id=model.image_token_id,
        max_length=data_cfg.get("max_length", 2048),
    )
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Optional held-out validation set. When data.val_data_path is given (e.g. the disjoint test.json
    # emitted by the builders' --test-fraction), the trainer computes a held-out loss every eval_steps
    # so overfitting / under-training is visible during the run. A missing file is a warning, not a
    # crash; a long run should never abort over a stale val path.
    val_dataset = None
    val_data_path = data_cfg.get("val_data_path")
    if val_data_path and not os.path.exists(val_data_path):
        logger.warning(
            f"val_data_path is set to '{val_data_path}' but the file does not exist; continuing "
            "WITHOUT validation loss. Build a held-out split (builders support --test-fraction) or "
            "fix the path to enable it."
        )
        val_data_path = None
    if val_data_path:
        val_image_dir = data_cfg.get("val_image_dir", data_cfg["image_dir"])
        logger.info("Building validation dataset...")
        val_dataset = LLaVAPretrainDataset(
            data_path=val_data_path,
            image_dir=val_image_dir,
            tokenizer=model.tokenizer,
            image_processor=model.image_processor,
            image_token_id=model.image_token_id,
            max_length=data_cfg.get("max_length", 2048),
        )
        # Deterministically subsample large held-out sets so each eval stays fast; the seed keeps the
        # scored subset identical across evals (and across resumes), preserving comparability.
        eval_num_samples = train_cfg.get("eval_num_samples", 0)
        if eval_num_samples and 0 < eval_num_samples < len(val_dataset):
            indices = list(range(len(val_dataset)))
            random.Random(train_cfg.get("seed", 42)).shuffle(indices)
            val_dataset = Subset(val_dataset, sorted(indices[:eval_num_samples]))
        logger.info(f"Validation dataset size: {len(val_dataset)} samples")

    trainer = VLMTrainer(
        model=model,
        train_dataset=dataset,
        config=config,
        accelerator=accelerator,
        val_dataset=val_dataset,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Done.")


if __name__ == "__main__":
    main()
