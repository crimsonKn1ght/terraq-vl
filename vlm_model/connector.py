"""Trainable vision->language projection (the only module trained in Stage 1)."""

import torch
import torch.nn as nn


class VisionLanguageConnector(nn.Module):
    """2-layer MLP (Linear -> GELU -> Linear) mapping vision features into the LLM embedding space."""

    def __init__(self, vision_hidden_size: int = 1024, llm_hidden_size: int = 1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(vision_features)
