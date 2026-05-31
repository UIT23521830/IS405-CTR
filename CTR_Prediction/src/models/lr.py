from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureLinear


class LR(nn.Module):
    def __init__(self, field_dims: list[int]) -> None:
        super().__init__()
        self.linear = FeatureLinear(field_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.linear(x)
        return {"logits": logits}

