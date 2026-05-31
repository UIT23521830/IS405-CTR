from __future__ import annotations

import torch
from torch import nn


class FeatureEmbedding(nn.Module):
    """Per-field categorical embedding layer."""

    def __init__(self, field_dims: list[int], embedding_dim: int) -> None:
        super().__init__()
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embedding_dim) for dim in field_dims])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape [batch, fields], got {tuple(x.shape)}")
        if x.size(1) != len(self.embeddings):
            raise ValueError(f"Expected {len(self.embeddings)} fields, got {x.size(1)}")
        outputs = []
        for idx, embedding in enumerate(self.embeddings):
            field_x = torch.remainder(x[:, idx], embedding.num_embeddings)
            outputs.append(embedding(field_x))
        return torch.stack(outputs, dim=1)


class FeatureLinear(nn.Module):
    """Per-field linear weights for LR/FM terms."""

    def __init__(self, field_dims: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, 1) for dim in field_dims])
        self.bias = nn.Parameter(torch.zeros(1))
        for embedding in self.embeddings:
            nn.init.zeros_(embedding.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = []
        for idx, embedding in enumerate(self.embeddings):
            field_x = torch.remainder(x[:, idx], embedding.num_embeddings)
            logits.append(embedding(field_x))
        return torch.stack(logits, dim=1).sum(dim=1).squeeze(-1) + self.bias

