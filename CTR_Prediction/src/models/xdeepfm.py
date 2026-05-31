from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureEmbedding, FeatureLinear
from src.models.mlp import MLP


class CIN(nn.Module):
    """Compressed Interaction Network used by xDeepFM."""

    def __init__(self, num_fields: int, layer_sizes: list[int]) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.layer_sizes = layer_sizes
        prev_fields = num_fields
        self.conv_layers = nn.ModuleList()
        for layer_size in layer_sizes:
            self.conv_layers.append(nn.Conv1d(prev_fields * num_fields, layer_size, kernel_size=1))
            prev_fields = layer_size
        self.output_dim = sum(layer_sizes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x0 = embeddings
        hidden = embeddings
        pooled_outputs = []
        for conv in self.conv_layers:
            interactions = torch.einsum("bhd,bmd->bhmd", hidden, x0)
            batch_size, prev_fields, num_fields, emb_dim = interactions.shape
            interactions = interactions.reshape(batch_size, prev_fields * num_fields, emb_dim)
            hidden = torch.relu(conv(interactions))
            pooled_outputs.append(hidden.sum(dim=2))
        return torch.cat(pooled_outputs, dim=1)


class xDeepFM(nn.Module):
    def __init__(
        self,
        field_dims: list[int],
        embedding_dim: int = 16,
        cin_layers: list[int] | None = None,
        hidden_units: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        cin_layers = cin_layers or [16, 16]
        hidden_units = hidden_units or [128, 64, 32]
        self.num_fields = len(field_dims)
        self.linear = FeatureLinear(field_dims)
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.cin = CIN(self.num_fields, cin_layers)
        self.cin_linear = nn.Linear(self.cin.output_dim, 1)
        self.deep = MLP(self.num_fields * embedding_dim, hidden_units, output_dim=1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.embedding(x)
        linear_logits = self.linear(x)
        cin_logits = self.cin_linear(self.cin(embeddings)).squeeze(-1)
        deep_logits = self.deep(embeddings.flatten(start_dim=1)).squeeze(-1)
        logits = linear_logits + cin_logits + deep_logits
        return {"logits": logits}

