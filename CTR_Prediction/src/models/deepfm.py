from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureEmbedding, FeatureLinear
from src.models.fm import FactorizationMachineLayer
from src.models.mlp import MLP


class DeepFM(nn.Module):
    def __init__(
        self,
        field_dims: list[int],
        embedding_dim: int = 16,
        hidden_units: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_units = hidden_units or [128, 64, 32]
        self.num_fields = len(field_dims)
        self.linear = FeatureLinear(field_dims)
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.fm = FactorizationMachineLayer()
        self.deep = MLP(self.num_fields * embedding_dim, hidden_units, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.embedding(x)
        deep_logits = self.deep(embeddings.flatten(start_dim=1)).squeeze(-1)
        logits = self.linear(x) + self.fm(embeddings) + deep_logits
        return {"logits": logits}

