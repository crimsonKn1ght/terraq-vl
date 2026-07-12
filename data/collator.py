from typing import List, Dict

import torch
from transformers import PreTrainedTokenizer

from vlm_model.utils import IGNORE_INDEX


class VLMDataCollator:

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        images = torch.stack([item["images"] for item in batch])

        max_len = min(
            max(ids.shape[0] for ids in input_ids_list),
            self.max_length,
        )

        padded_input_ids = torch.full(
            (len(batch), max_len), self.pad_token_id, dtype=torch.long
        )
        padded_labels = torch.full(
            (len(batch), max_len), IGNORE_INDEX, dtype=torch.long
        )
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, (ids, labels) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = min(ids.shape[0], max_len)
            padded_input_ids[i, :seq_len] = ids[:seq_len]
            padded_labels[i, :seq_len] = labels[:seq_len]
            attention_mask[i, :seq_len] = 1

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
            "images": images,
        }
