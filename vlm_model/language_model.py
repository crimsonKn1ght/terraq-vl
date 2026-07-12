from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils import freeze_module, IMAGE_TOKEN

# Qwen2.5 attention + MLP projections — the standard LoRA target set for Stage-2 instruction tuning.
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class LanguageModel(nn.Module):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype: torch.dtype = torch.bfloat16,
        lora: Optional[dict] = None,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [IMAGE_TOKEN]}
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Always start from a fully frozen base. Stage-1 leaves it frozen; Stage-2 then re-opens a
        # small set of low-rank adapters (LoRA) while the original weights stay frozen.
        freeze_module(self.model)

        self.is_lora = lora is not None
        if self.is_lora:
            self._apply_lora(lora)

    def _apply_lora(self, lora: dict) -> None:
        # Imported lazily so Stage-1 (no LoRA) does not require `peft` to be installed.
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("lora_alpha", 32)),
            lora_dropout=float(lora.get("lora_dropout", 0.05)),
            target_modules=list(lora.get("target_modules", DEFAULT_LORA_TARGET_MODULES)),
            bias="none",
        )
        # get_peft_model freezes the base and marks only the adapter weights trainable. The resized
        # embedding table (for <image>) is not a LoRA target, so it stays frozen — correct, since
        # <image> positions are swapped for visual embeds and never embedded.
        self.model = get_peft_model(self.model, lora_config)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.LongTensor:
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
