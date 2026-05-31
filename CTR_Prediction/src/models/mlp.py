from __future__ import annotations

import torch
from torch import nn


class ReLUN(nn.Module):
    """Clipped ReLU used by ExU layers in Neural Additive Models."""

    def __init__(self, max_value: float = 1.0) -> None:
        super().__init__()
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.relu(x), max=self.max_value)


class ExULayer(nn.Module):
    """Exp-centered unit layer: h(x) = f((x - b) @ exp(W))."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_value: float = 1.0,
        weight_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(input_dim))
        self.activation = ReLUN(max_value=max_value)
        self.weight_clip = weight_clip
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.bias, mean=0.0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_weight = torch.exp(torch.clamp(self.weight, min=-self.weight_clip, max=self.weight_clip))
        return self.activation((x - self.bias) @ exp_weight)


def get_activation(activation: str | type[nn.Module]) -> type[nn.Module]:
    if isinstance(activation, str):
        name = activation.lower()
        if name == "relu":
            return nn.ReLU
        if name == "gelu":
            return nn.GELU
        if name == "tanh":
            return nn.Tanh
        if name == "sigmoid":
            return nn.Sigmoid
        raise ValueError(f"Unsupported activation: {activation}")
    return activation


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: list[int],
        output_dim: int = 1,
        dropout: float = 0.0,
        activation: str | type[nn.Module] = nn.ReLU,
        exu_max_value: float = 1.0,
        exu_weight_clip: float = 10.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        use_exu = isinstance(activation, str) and activation.lower() == "exu"
        activation_cls = None if use_exu else get_activation(activation)
        for hidden_dim in hidden_units:
            if use_exu:
                layers.append(
                    ExULayer(
                        prev_dim,
                        hidden_dim,
                        max_value=exu_max_value,
                        weight_clip=exu_weight_clip,
                    )
                )
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
