from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureEmbedding, FeatureLinear


class FactorizationMachineLayer(nn.Module):
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        square_of_sum = embeddings.sum(dim=1).pow(2)
        sum_of_square = embeddings.pow(2).sum(dim=1)
        return 0.5 * (square_of_sum - sum_of_square).sum(dim=1)


class FM(nn.Module):
    def __init__(self, field_dims: list[int], embedding_dim: int = 16) -> None:
        super().__init__()
        self.linear = FeatureLinear(field_dims)
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.fm = FactorizationMachineLayer()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.embedding(x)
        logits = self.linear(x) + self.fm(embeddings)
        return {"logits": logits}

