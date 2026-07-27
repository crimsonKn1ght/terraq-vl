"""Training loop for the VLM: optimizer, cosine schedule, accumulation, logging, checkpointing."""

import logging
import time
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from vlm_model.vlm import VLMForCausalLM
from vlm_model.utils import count_trainable_parameters, count_total_parameters, IGNORE_INDEX
from data.collator import VLMDataCollator
from .lr_scheduler import build_cosine_warmup_scheduler
from .checkpoint import save_connector_checkpoint, load_connector_checkpoint, load_lora_adapter

logger = logging.getLogger(__name__)


class VLMTrainer:
    """Accelerate training loop (connector-only in Stage 1, connector + LoRA in Stage 2).

    Sets up the AdamW optimizer (optionally a split connector/LoRA LR), cosine-warmup schedule,
    gradient accumulation and clipping, periodic logging, and connector(+LoRA) checkpointing.

    The logged training loss is a *token-weighted* running mean, and, when a ``val_dataset`` is
    given, a held-out validation loss is computed every ``eval_steps`` (same token-weighted method
    as ``scripts/eval_loss_curve.py``) and written into each checkpoint's ``meta.json``, so overfitting
    (val rising while train falls) and under-training (val still falling) are visible during the run.
    """

    def __init__(
        self,
        model: VLMForCausalLM,
        train_dataset,
        config: Dict[str, Any],
        accelerator: Accelerator,
        val_dataset=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.accelerator = accelerator

        train_cfg = config.get("training", {})
        self.output_dir = train_cfg.get("output_dir", "./checkpoints")
        self.num_epochs = train_cfg.get("num_epochs", 1)
        self.per_device_batch_size = train_cfg.get("per_device_batch_size", 8)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 32)
        self.learning_rate = float(train_cfg.get("learning_rate", 2e-3))
        self.warmup_ratio = train_cfg.get("warmup_ratio", 0.03)
        self.weight_decay = train_cfg.get("weight_decay", 0.0)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.logging_steps = train_cfg.get("logging_steps", 10)
        self.save_steps = train_cfg.get("save_steps", 500)
        # How often to compute held-out validation loss (defaults to the checkpoint cadence, so every
        # saved checkpoint carries a matching val_loss). Only used when a val_dataset is provided.
        self.eval_steps = train_cfg.get("eval_steps", self.save_steps)
        self.latest_val_loss = None
        self.dataloader_num_workers = train_cfg.get("dataloader_num_workers", 4)
        self.seed = train_cfg.get("seed", 42)
        # Stage 1 = connector-only (default). Stage 2 = connector + LoRA on the LLM.
        self.stage = train_cfg.get("stage", 1)
        # Optional separate LR for the (pretrained) connector vs the fresh LoRA adapters.
        self.connector_lr = train_cfg.get("connector_lr", None)
        # Optional resume: checkpoint dir to restore connector + optimizer + scheduler + step from.
        self.resume_from = train_cfg.get("resume_from", None)

    def train(self):
        """Run the full training loop, checkpointing the connector (and LoRA adapter in Stage 2)."""
        trainable = count_trainable_parameters(self.model)
        total = count_total_parameters(self.model)
        logger.info(f"Trainable parameters: {trainable:,}")
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable ratio: {trainable / total:.6%}")

        # The connector is trainable in both stages; Stage 2 additionally trains the LoRA adapters.
        connector_params = list(self.model.connector.parameters())
        assert all(
            p.requires_grad for p in connector_params
        ), "Connector parameters must be trainable"

        all_trainable = [p for p in self.model.parameters() if p.requires_grad]
        connector_ids = {id(p) for p in connector_params}
        other_trainable = [p for p in all_trainable if id(p) not in connector_ids]

        adamw_kwargs = dict(weight_decay=self.weight_decay, betas=(0.9, 0.999))
        if self.connector_lr is not None and other_trainable:
            # Give the pretrained connector its own (typically lower) LR than the fresh LoRA.
            optimizer = torch.optim.AdamW(
                [
                    {"params": connector_params, "lr": float(self.connector_lr)},
                    {"params": other_trainable, "lr": self.learning_rate},
                ],
                **adamw_kwargs,
            )
        else:
            optimizer = torch.optim.AdamW(
                all_trainable, lr=self.learning_rate, **adamw_kwargs
            )

        collator = VLMDataCollator(
            tokenizer=self.model.tokenizer,
            max_length=self.config.get("data", {}).get("max_length", 2048),
        )

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
        )

        # Held-out loader: no shuffle, keep every sample (drop_last=False) so the val loss is a stable,
        # fully-comparable number across evals.
        val_dataloader = None
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.per_device_batch_size,
                shuffle=False,
                num_workers=self.dataloader_num_workers,
                pin_memory=True,
                collate_fn=collator,
                drop_last=False,
            )

        num_update_steps_per_epoch = len(dataloader) // self.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.num_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)

        scheduler = build_cosine_warmup_scheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.model, optimizer, dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, dataloader, scheduler
        )
        if val_dataloader is not None:
            val_dataloader = self.accelerator.prepare(val_dataloader)

        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")
        logger.info(
            f"Effective batch size: {self.per_device_batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes}"
        )

        global_step = 0
        running_loss = 0.0
        running_tokens = 0
        start_time = time.time()
        grad_checked = False

        self._set_train_mode()
        unwrapped = self.accelerator.unwrap_model(self.model)

        is_lora = getattr(unwrapped.language_model, "is_lora", False)
        trainable_for_step = [p for p in unwrapped.parameters() if p.requires_grad]
        # Stage-2 also checkpoints the LoRA adapter; Stage-1 passes None (connector only).
        peft_model = unwrapped.language_model.model if is_lora else None

        # Resume: restore connector + optimizer + scheduler (+ LoRA adapter) and continue from the
        # saved step. As long as the effective batch matches the original run, num_training_steps and
        # the cosine schedule line up (batch 4 x accum 32 == batch 8 x accum 16 == effective 128), so
        # training simply completes the REMAINING steps.
        if self.resume_from:
            resumed_step = load_connector_checkpoint(
                unwrapped.connector, self.resume_from, optimizer, scheduler
            )
            if is_lora:
                load_lora_adapter(unwrapped.language_model.model, self.resume_from)
            global_step = resumed_step
            logger.info(
                f"Resumed from {self.resume_from} at step {global_step}/{num_training_steps} "
                f"({max(0, num_training_steps - global_step)} steps remaining)"
            )

        for epoch in range(self.num_epochs):
            if global_step >= num_training_steps:
                break
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")

            for step, batch in enumerate(dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        images=batch["images"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                    loss = outputs.loss
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        # Sanity check once, BEFORE zero_grad clears the grads: the connector
                        # must receive gradient from the loss (i.e. the image-embedding merge
                        # path is not detached).
                        if not grad_checked:
                            assert any(
                                p.grad is not None
                                for p in unwrapped.connector.parameters()
                            ), (
                                "Connector received no gradient before the optimizer step: "
                                "the image-embedding merge path is detached from the loss."
                            )
                            if is_lora:
                                assert any(
                                    p.grad is not None for p in trainable_for_step
                                    if id(p) not in {id(c) for c in unwrapped.connector.parameters()}
                                ), (
                                    "No LoRA parameter received gradient: the LLM adapters are "
                                    "detached from the loss."
                                )
                            grad_checked = True
                        self.accelerator.clip_grad_norm_(
                            trainable_for_step,
                            self.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Token-weight the running loss: each micro-batch's loss is a mean over ITS supervised
                # tokens, so weighting by that token count (not counting every micro-batch equally)
                # makes the logged value a true per-token mean. This is what removes the high-frequency
                # jitter from the curve: a 3-token VQA answer no longer swings the average like a
                # 200-token caption. Same weighting scheme as scripts/eval_loss_curve.py.
                label_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
                running_loss += loss.detach().item() * label_tokens
                running_tokens += label_tokens

                if self.accelerator.sync_gradients:
                    global_step += 1

                    if global_step % self.logging_steps == 0:
                        avg_loss = running_loss / max(1, running_tokens)
                        elapsed = time.time() - start_time
                        samples_per_sec = (
                            global_step
                            * self.per_device_batch_size
                            * self.gradient_accumulation_steps
                            * self.accelerator.num_processes
                            / elapsed
                        )
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {global_step}/{num_training_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Samples/s: {samples_per_sec:.1f}"
                        )
                        running_loss = 0.0
                        running_tokens = 0

                    # Validate BEFORE saving so a checkpoint written on the same step records the
                    # matching val_loss. evaluate() runs on all processes (it reduces across them);
                    # only the main process logs the line.
                    if (
                        val_dataloader is not None
                        and global_step % self.eval_steps == 0
                    ):
                        self.latest_val_loss = self.evaluate(val_dataloader)
                        if self.accelerator.is_main_process:
                            logger.info(
                                f"Step {global_step}/{num_training_steps} | "
                                f"Val loss: {self.latest_val_loss:.4f}"
                            )

                    if (
                        global_step % self.save_steps == 0
                        and self.accelerator.is_main_process
                    ):
                        save_connector_checkpoint(
                            connector=unwrapped.connector,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=global_step,
                            loss=loss.item(),
                            output_dir=self.output_dir,
                            peft_model=peft_model,
                            val_loss=self.latest_val_loss,
                        )
                        logger.info(f"Saved checkpoint at step {global_step}")

                    if global_step >= num_training_steps:
                        break

        if val_dataloader is not None:
            self.latest_val_loss = self.evaluate(val_dataloader)
            if self.accelerator.is_main_process:
                logger.info(
                    f"Step {global_step}/{num_training_steps} | "
                    f"Final val loss: {self.latest_val_loss:.4f}"
                )

        if self.accelerator.is_main_process:
            save_connector_checkpoint(
                connector=unwrapped.connector,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                loss=loss.item(),
                output_dir=self.output_dir,
                peft_model=peft_model,
                val_loss=self.latest_val_loss,
            )
            logger.info(f"Training complete. Final checkpoint saved at step {global_step}")

    def _set_train_mode(self) -> None:
        """Put the model in training mode while keeping the frozen sub-modules in eval.

        The vision encoder is always frozen (keep eval). The LLM is frozen in Stage 1 (keep eval) but
        carries trainable LoRA in Stage 2, so it stays in train() there for LoRA dropout. Called at
        startup and again after every validation pass, which flips the whole model to eval().
        """
        self.model.train()
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.vision_encoder.model.eval()
        if self.stage < 2:
            unwrapped.language_model.model.eval()

    @torch.no_grad()
    def evaluate(self, val_dataloader) -> float:
        """Return the token-weighted mean answer-token loss over the held-out loader.

        Mirrors scripts/eval_loss_curve.py: each batch's loss (a mean over its supervised tokens) is
        weighted by that token count, so the aggregate is a true per-token mean: smooth and directly
        comparable across evals. Totals are reduced across processes; training mode is restored before
        returning. No backward pass, so eval batches can be as large as the training ones.
        """
        self.model.eval()
        device = self.accelerator.device
        total_loss = torch.zeros(1, device=device)
        total_tokens = torch.zeros(1, device=device)
        for batch in val_dataloader:
            n_tokens = (batch["labels"] != IGNORE_INDEX).sum()
            if n_tokens.item() == 0:
                continue
            outputs = self.model(
                input_ids=batch["input_ids"],
                images=batch["images"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.detach() * n_tokens
            total_tokens += n_tokens

        total_loss = self.accelerator.reduce(total_loss, reduction="sum")
        total_tokens = self.accelerator.reduce(total_tokens, reduction="sum")

        self._set_train_mode()
        if total_tokens.item() == 0:
            return float("nan")
        return (total_loss / total_tokens).item()
