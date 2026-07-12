import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

from .utils import freeze_module


class VisionEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        select_layer: int = -2,
        select_feature: str = "patch",
    ):
        super().__init__()
        self.select_layer = select_layer
        self.select_feature = select_feature

        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        freeze_module(self.model)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def num_patches(self) -> int:
        return (self.model.config.image_size // self.model.config.patch_size) ** 2

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            pixel_values=images,
            output_hidden_states=True,
        )
        features = outputs.hidden_states[self.select_layer]

        if self.select_feature == "patch":
            features = features[:, 1:, :]  # drop CLS token → (B, 256, 1024)

        return features
