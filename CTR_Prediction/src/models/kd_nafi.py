from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.nafi import NAFI


class KDNAFI(NAFI):
    """Student NAFI model for knowledge distillation experiments."""


def binary_logits_to_two_class_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert one binary logit z into two-class logits [0, z]."""
    return torch.stack([torch.zeros_like(logits), logits], dim=-1)


def binary_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, hard_labels)
    teacher_probs = torch.sigmoid(teacher_logits / temperature)
    soft_loss = F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_probs)
    return alpha * hard_loss + (1.0 - alpha) * (temperature**2) * soft_loss


def binary_kl_kd_loss_from_teacher_class_probs(
    student_logits: torch.Tensor,
    teacher_class_probs: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KD objective matching the paper diagram using two-class softmax at T=t.

    The model emits one logit for binary CTR. We represent it as two-class
    logits [not_click=0, click=z], so softmax([0, z] / T) equals sigmoid(z / T)
    for the click class.
    """
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, hard_labels)
    student_log_probs_t = F.log_softmax(
        binary_logits_to_two_class_logits(student_logits) / temperature,
        dim=-1,
    )
    soft_loss = F.kl_div(
        student_log_probs_t,
        teacher_class_probs,
        reduction="batchmean",
    )
    loss = alpha * hard_loss + (1.0 - alpha) * (temperature**2) * soft_loss
    return loss, hard_loss, soft_loss


def binary_kd_loss_from_teacher_probs(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, hard_labels)
    soft_loss = F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_probs)
    loss = alpha * hard_loss + (1.0 - alpha) * (temperature**2) * soft_loss
    return loss, hard_loss, soft_loss
