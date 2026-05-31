from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.base import build_model
from src.models.kd_nafi import binary_logits_to_two_class_logits
from src.training.checkpoint import load_checkpoint


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    checkpoint: Path


class TeacherEnsemble(nn.Module):
    """Frozen teacher ensemble for binary CTR distillation."""

    def __init__(
        self,
        teachers: list[nn.Module],
        names: list[str],
        mode: str = "uniform",
        weights: list[float] | None = None,
        adaptive_hidden_units: list[int] | None = None,
    ) -> None:
        super().__init__()
        if not teachers:
            raise ValueError("TeacherEnsemble requires at least one teacher")
        self.teachers = nn.ModuleList(teachers)
        self.names = names
        self.mode = mode
        self.num_teachers = len(teachers)
        if weights is not None:
            if len(weights) != len(teachers):
                raise ValueError("Number of teacher weights must match number of teachers")
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
            weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-12)
        else:
            weight_tensor = torch.full((len(teachers),), 1.0 / len(teachers), dtype=torch.float32)
        self.register_buffer("weights", weight_tensor)
        adaptive_hidden_units = adaptive_hidden_units or [8]
        layers: list[nn.Module] = []
        prev_dim = 1
        for hidden_dim in adaptive_hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.adaptive_mapper = nn.Sequential(*layers)
        self.adaptive_mapper.requires_grad_(mode == "learned")
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = []
            for teacher in self.teachers:
                logits.append(teacher(x)["logits"])
        stacked_logits = torch.stack(logits, dim=0)
        teacher_class_probs_t = torch.softmax(
            binary_logits_to_two_class_logits(stacked_logits) / temperature,
            dim=-1,
        )
        teacher_class_probs_1 = torch.softmax(
            binary_logits_to_two_class_logits(stacked_logits),
            dim=-1,
        )
        teacher_click_probs = teacher_class_probs_t[..., 1]

        if self.mode == "uniform" or self.mode == "fixed":
            weights = self.weights.to(device=teacher_class_probs_t.device, dtype=teacher_class_probs_t.dtype)
            batch_weights = weights.view(-1, 1).expand(-1, stacked_logits.size(1))
        elif self.mode == "confidence":
            scores = torch.abs(teacher_click_probs - 0.5)
            batch_weights = torch.softmax(scores, dim=0)
        elif self.mode == "learned":
            # Implements alpha_i = softmax(f(Z_i)) where Z_i is each teacher logit.
            mapper_input = stacked_logits.detach().transpose(0, 1).unsqueeze(-1)
            scores = self.adaptive_mapper(mapper_input).squeeze(-1).transpose(0, 1)
            batch_weights = torch.softmax(scores, dim=0)
        else:
            raise ValueError(f"Unsupported teacher ensemble mode: {self.mode}")

        ensemble_class_probs = (teacher_class_probs_t * batch_weights.unsqueeze(-1)).sum(dim=0)
        ensemble_class_probs = ensemble_class_probs.clamp(1e-6, 1.0 - 1e-6)
        ensemble_class_probs = ensemble_class_probs / ensemble_class_probs.sum(dim=-1, keepdim=True)
        ensemble_probs = ensemble_class_probs[..., 1]
        hard_ensemble_class_probs = (teacher_class_probs_1 * batch_weights.unsqueeze(-1)).sum(dim=0)
        hard_ensemble_probs = hard_ensemble_class_probs[..., 1].clamp(1e-6, 1.0 - 1e-6)
        ensemble_logits = torch.logit(hard_ensemble_probs)
        return {
            "logits": ensemble_logits,
            "probs": ensemble_probs,
            "class_probs": ensemble_class_probs,
            "teacher_logits": stacked_logits,
            "weights": batch_weights,
        }


def parse_teacher_spec(value: str) -> TeacherSpec:
    if ":" not in value:
        raise ValueError("Teacher spec must have format model_name:/path/to/checkpoint.pt")
    name, checkpoint = value.split(":", 1)
    return TeacherSpec(name=name.strip().lower(), checkpoint=Path(checkpoint.strip()))


def load_teacher_ensemble(
    teacher_specs: list[TeacherSpec],
    field_dims: list[int],
    default_config: dict[str, Any],
    device: torch.device,
    mode: str = "uniform",
    weights: list[float] | None = None,
    adaptive_hidden_units: list[int] | None = None,
) -> TeacherEnsemble:
    teachers: list[nn.Module] = []
    names: list[str] = []
    for spec in teacher_specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {spec.checkpoint}")
        checkpoint = torch.load(spec.checkpoint, map_location="cpu")
        teacher_config = checkpoint.get("config", default_config)
        teacher = build_model(spec.name, field_dims, teacher_config)
        load_checkpoint(spec.checkpoint, teacher, map_location="cpu")
        teacher.to(device)
        teacher.eval()
        teachers.append(teacher)
        names.append(spec.name)
    return TeacherEnsemble(
        teachers=teachers,
        names=names,
        mode=mode,
        weights=weights,
        adaptive_hidden_units=adaptive_hidden_units,
    ).to(device)
