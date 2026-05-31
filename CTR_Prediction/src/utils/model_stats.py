from __future__ import annotations

from torch import nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    non_trainable = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def count_named_children_parameters(model: nn.Module, child_names: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for child_name in child_names:
        child = getattr(model, child_name, None)
        if isinstance(child, nn.Module):
            counts[child_name] = count_parameters(child)
    return counts


def format_parameter_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count:,} ({count / 1_000_000:.2f}M)"
    if count >= 1_000:
        return f"{count:,} ({count / 1_000:.2f}K)"
    return f"{count:,}"
