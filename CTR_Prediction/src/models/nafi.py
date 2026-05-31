from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureEmbedding
from src.models.fin import FINBranch
from src.models.nam import NAMBranch


class NAFI(nn.Module):
    """Neural Additive Feature Interaction model: logits = NAM + FIN."""

    def __init__(
        self,
        field_dims: list[int],
        embedding_dim: int = 16,
        nam_hidden_units: list[int] | None = None,
        fin_num_heads: int = 4,
        fin_num_layers: int = 2,
        nam_dropout: float = 0.1,
        fin_dropout: float = 0.1,
        use_residual: bool = True,
        nam_activation: str = "relu",
        exu_max_value: float = 1.0,
        exu_weight_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.nam = NAMBranch(
            len(field_dims),
            embedding_dim,
            nam_hidden_units,
            nam_dropout,
            activation=nam_activation,
            exu_max_value=exu_max_value,
            exu_weight_clip=exu_weight_clip,
        )
        self.fin = FINBranch(
            len(field_dims),
            embedding_dim,
            num_heads=fin_num_heads,
            num_layers=fin_num_layers,
            dropout=fin_dropout,
            use_residual=use_residual,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.embedding(x)
        nam_logits, contributions = self.nam(embeddings)
        fin_logits, attention_weights = self.fin(embeddings)
        logits = nam_logits + fin_logits
        return {
            "logits": logits,
            "nam_logits": nam_logits,
            "fin_logits": fin_logits,
            "attention_weights": attention_weights,
            "feature_contributions": contributions,
        }
