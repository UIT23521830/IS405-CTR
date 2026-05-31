from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureEmbedding
from src.models.mlp import MLP


class NAMBranch(nn.Module):
    """Neural additive branch over already embedded fields."""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        hidden_units: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        exu_max_value: float = 1.0,
        exu_weight_clip: float = 10.0,
    ) -> None:
        super().__init__()
        hidden_units = hidden_units or [32, 16]
        self.feature_nets = nn.ModuleList(
            [
                MLP(
                    embedding_dim,
                    hidden_units,
                    output_dim=1,
                    dropout=dropout,
                    activation=activation,
                    exu_max_value=exu_max_value,
                    exu_weight_clip=exu_weight_clip,
                )
                for _ in range(num_fields)
            ]
        )

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contributions = []
        for idx, net in enumerate(self.feature_nets):
            contributions.append(net(embeddings[:, idx, :]))
        contribution_tensor = torch.cat(contributions, dim=1)
        logits = contribution_tensor.sum(dim=1)
        return logits, contribution_tensor


class NAM(nn.Module):
    def __init__(
        self,
        field_dims: list[int],
        embedding_dim: int = 16,
        hidden_units: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        exu_max_value: float = 1.0,
        exu_weight_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.branch = NAMBranch(
            len(field_dims),
            embedding_dim,
            hidden_units,
            dropout,
            activation=activation,
            exu_max_value=exu_max_value,
            exu_weight_clip=exu_weight_clip,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.embedding(x)
        logits, contributions = self.branch(embeddings)
        return {"logits": logits, "feature_contributions": contributions}
