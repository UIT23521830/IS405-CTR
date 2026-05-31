from __future__ import annotations

import torch
from torch import nn


def get_loss(name: str = "bce_with_logits") -> nn.Module:
    normalized = name.lower()
    if normalized not in {"bce_with_logits", "binary_cross_entropy", "logloss", "log_loss"}:
        raise ValueError(f"Unsupported loss: {name}")
    return nn.BCEWithLogitsLoss()


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.detach())
