import json
import os
import logging
from typing import Dict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, CLIPImageProcessor

from .image_processing import load_and_process_image
from .conversation import tokenize_conversation

logger = logging.getLogger(__name__)


class LLaVAPretrainDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer: PreTrainedTokenizer,
        image_processor: CLIPImageProcessor,
        image_token_id: int,
        max_length: int = 2048,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_id = image_token_id
        self.max_length = max_length

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        for offset in range(10):
            actual_idx = (idx + offset) % len(self.data)
            item = self.data[actual_idx]
            try:
                image_path = os.path.join(self.image_dir, item["image"])
                pixel_values = load_and_process_image(
                    image_path, self.image_processor
                )

                input_ids, labels = tokenize_conversation(
                    conversations=item["conversations"],
                    tokenizer=self.tokenizer,
                    image_token_id=self.image_token_id,
                    max_length=self.max_length,
                )

                return {
                    "input_ids": input_ids,
                    "labels": labels,
                    "images": pixel_values,
                }
            except Exception as e:
                if offset == 0:
                    logger.warning(
                        f"Failed to load sample {actual_idx}: {e}"
                    )
                continue

        return self._get_dummy_sample()

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        dummy_ids = torch.zeros(1, dtype=torch.long)
        dummy_labels = torch.full((1,), -100, dtype=torch.long)
        dummy_image = torch.zeros(3, 224, 224)
        return {
            "input_ids": dummy_ids,
            "labels": dummy_labels,
            "images": dummy_image,
        }
