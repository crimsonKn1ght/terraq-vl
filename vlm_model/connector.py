import torch
import torch.nn as nn


class VisionLanguageConnector(nn.Module):

    def __init__(self, vision_hidden_size: int = 1024, llm_hidden_size: int = 1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(vision_features)
