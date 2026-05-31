from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureEmbedding
from src.models.mlp import MLP


class FINBranch(nn.Module):
    """Feature interaction branch based on multi-head self-attention."""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.attn_layers = nn.ModuleList(
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
        self.activation = nn.ReLU()
        self.output = MLP(num_fields * embedding_dim, [64, 32], output_dim=1, dropout=dropout)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = embeddings
        attention_weights = None
        for attn, norm in zip(self.attn_layers, self.norms):
            residual = out
            out, attention_weights = attn(out, out, out, need_weights=True, average_attn_weights=False)
            if self.use_residual:
                out = out + residual
            out = norm(self.activation(out))
        logits = self.output(out.flatten(start_dim=1)).squeeze(-1)
        return logits, attention_weights


class FIN(nn.Module):
    def __init__(
        self,
        field_dims: list[int],
        embedding_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.branch = FINBranch(
            len(field_dims),
            embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_residual=use_residual,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, attention_weights = self.branch(self.embedding(x))
        return {"logits": logits, "attention_weights": attention_weights}

