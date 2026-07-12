from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils import IGNORE_INDEX
from .vision_encoder import VisionEncoder
from .connector import VisionLanguageConnector
from .language_model import LanguageModel


class VLMForCausalLM(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        ve_cfg = config.get("vision_encoder", {})
        lm_cfg = config.get("language_model", {})
        cn_cfg = config.get("connector", {})

        self.vision_encoder = VisionEncoder(
            model_name=ve_cfg.get("model_name", "openai/clip-vit-large-patch14"),
            select_layer=ve_cfg.get("select_layer", -2),
            select_feature=ve_cfg.get("select_feature", "patch"),
        )

        dtype_str = lm_cfg.get("torch_dtype", "bfloat16")
        torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

        self.language_model = LanguageModel(
            model_name=lm_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            torch_dtype=torch_dtype,
            lora=lm_cfg.get("lora"),
        )

        self.connector = VisionLanguageConnector(
            vision_hidden_size=cn_cfg.get(
                "vision_hidden_size", self.vision_encoder.hidden_size
            ),
            llm_hidden_size=cn_cfg.get(
                "llm_hidden_size", self.language_model.hidden_size
            ),
        )

        self.image_token_id = self.language_model.image_token_id
        self.num_patches = self.vision_encoder.num_patches

    def enable_gradient_checkpointing(self) -> None:
        """Trade compute for memory on the LLM backward pass (Stage-2 needs this to fit ~48 GB).

        The connector output requires grad, so the concatenated ``inputs_embeds`` requires grad and
        checkpointing works without ``enable_input_require_grads``. ``use_cache`` must be off because
        gradient checkpointing and the KV cache are mutually exclusive.
        """
        llm = self.language_model.model
        llm.gradient_checkpointing_enable()
        # For a PeftModel, ``.config`` proxies the base model's config; fall back to the wrapped base.
        base_config = getattr(llm, "config", None)
        if base_config is None and hasattr(llm, "get_base_model"):
            base_config = llm.get_base_model().config
        if base_config is not None:
            base_config.use_cache = False

    @property
    def tokenizer(self):
        return self.language_model.tokenizer

    @property
    def image_processor(self):
        return self.vision_encoder.image_processor

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vision_features = self.vision_encoder(images)
        return self.connector(vision_features)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor],
        images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = input_ids.shape[0]
        embed_tokens = self.language_model.get_input_embeddings()

        if images is None:
            return (
                embed_tokens(input_ids),
                attention_mask,
                labels,
            )

        image_embeds = self.encode_images(images)

        new_embeds_list = []
        new_labels_list = []
        new_mask_list = []

        for i in range(batch_size):
            ids = input_ids[i]
            cur_labels = labels[i] if labels is not None else None
            cur_mask = attention_mask[i]

            image_positions = torch.where(ids == self.image_token_id)[0]

            if len(image_positions) == 0:
                new_embeds_list.append(embed_tokens(ids))
                new_labels_list.append(cur_labels)
                new_mask_list.append(cur_mask)
                continue

            img_pos = image_positions[0].item()
            cur_image_embeds = image_embeds[i]

            before_ids = ids[:img_pos]
            after_ids = ids[img_pos + 1 :]

            before_embeds = embed_tokens(before_ids)
            after_embeds = embed_tokens(after_ids)

            combined_embeds = torch.cat(
                [before_embeds, cur_image_embeds, after_embeds], dim=0
            )
            new_embeds_list.append(combined_embeds)

            if cur_labels is not None:
                before_labels = cur_labels[:img_pos]
                after_labels = cur_labels[img_pos + 1 :]
                image_labels = torch.full(
                    (self.num_patches,),
                    IGNORE_INDEX,
                    dtype=cur_labels.dtype,
                    device=cur_labels.device,
                )
                combined_labels = torch.cat(
                    [before_labels, image_labels, after_labels], dim=0
                )
                new_labels_list.append(combined_labels)

            before_mask = cur_mask[:img_pos]
            after_mask = cur_mask[img_pos + 1 :]
            image_mask = torch.ones(
                self.num_patches,
                dtype=cur_mask.dtype,
                device=cur_mask.device,
            )
            combined_mask = torch.cat(
                [before_mask, image_mask, after_mask], dim=0
            )
            new_mask_list.append(combined_mask)

        max_len = max(e.shape[0] for e in new_embeds_list)
        embed_dim = new_embeds_list[0].shape[-1]
        device = new_embeds_list[0].device
        dtype = new_embeds_list[0].dtype

        padded_embeds = torch.zeros(
            batch_size, max_len, embed_dim, dtype=dtype, device=device
        )
        padded_mask = torch.zeros(
            batch_size, max_len, dtype=attention_mask.dtype, device=device
        )
        padded_labels = (
            torch.full(
                (batch_size, max_len),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=device,
            )
            if labels is not None
            else None
        )

        for i in range(batch_size):
            seq_len = new_embeds_list[i].shape[0]
            padded_embeds[i, :seq_len] = new_embeds_list[i]
            padded_mask[i, :seq_len] = new_mask_list[i]
            if padded_labels is not None and new_labels_list:
                padded_labels[i, :seq_len] = new_labels_list[i]

        return padded_embeds, padded_mask, padded_labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        images: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds, attention_mask, labels = self.prepare_inputs_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        images: torch.Tensor,
        attention_mask: torch.Tensor,
        **generate_kwargs,
    ) -> torch.LongTensor:
        inputs_embeds, attention_mask, _ = self.prepare_inputs_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            images=images,
        )
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
