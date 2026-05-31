from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_auc: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "best_auc": best_auc,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint["model_state_dict"]
    cleaned = {key.removeprefix("module."): value for key, value in state_dict.items()}
    unwrap_model(model).load_state_dict(cleaned)
    return checkpoint

