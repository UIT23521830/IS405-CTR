from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureEmbedding
from src.models.mlp import MLP


class AutoInt(nn.Module):
    def __init__(
        self,
        field_dims: list[int],
        embedding_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.output = MLP(len(field_dims) * embedding_dim, [64], dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.embedding(x)
        attention_weights = None
        for attn, norm in zip(self.layers, self.norms):
            residual = out
            out, attention_weights = attn(out, out, out, need_weights=True, average_attn_weights=False)
            out = norm(out + residual)
        logits = self.output(out.flatten(start_dim=1)).squeeze(-1)
        return {"logits": logits, "attention_weights": attention_weights}

