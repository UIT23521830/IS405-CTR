from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.kd_nafi import binary_kl_kd_loss_from_teacher_class_probs
from src.training.distillation import TeacherEnsemble
from src.training.trainer import CTRTrainer


class KDCTRTrainer(CTRTrainer):
    """Train a student CTR model with hard labels and teacher soft labels."""

    def __init__(
        self,
        model: nn.Module,
        teacher_ensemble: TeacherEnsemble,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        config: dict[str, Any],
    ) -> None:
        super().__init__(model=model, train_loader=train_loader, valid_loader=valid_loader, config=config)
        distill_cfg = config.get("distillation", {})
        self.teacher_ensemble = teacher_ensemble.to(self.device)
        self.teacher_ensemble.eval()
        self.temperature = float(distill_cfg.get("temperature", 4.0))
        self.alpha = float(distill_cfg.get("alpha", 0.5))
        self.adaptive_loss_weight = float(distill_cfg.get("adaptive_loss_weight", 1.0))
        adaptive_params = [param for param in self.teacher_ensemble.parameters() if param.requires_grad]
        if adaptive_params:
            self.optimizer.add_param_group(
                {
                    "params": adaptive_params,
                    "lr": float(distill_cfg.get("adaptive_learning_rate", config.get("training", {}).get("learning_rate", 1e-3))),
                    "weight_decay": float(distill_cfg.get("adaptive_l2_reg", 0.0)),
                }
            )
        self.logger.info(
            "kd_enabled=true kd_loss=kl_softmax_2class teachers=%s temperature=%.4f alpha=%.4f ensemble_mode=%s adaptive_trainable_params=%d",
            ",".join(self.teacher_ensemble.names),
            self.temperature,
            self.alpha,
            self.teacher_ensemble.mode,
            sum(param.numel() for param in adaptive_params),
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        started_at = time.perf_counter()
        self.model.train()
        self.teacher_ensemble.eval()
        running_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0
        running_adaptive_loss = 0.0
        teacher_weight_sums = torch.zeros(len(self.teacher_ensemble.names), dtype=torch.float64)
        teacher_weight_rows = 0
        step_count = 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(self.train_loader, desc=f"kd train epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                teacher_output = self.teacher_ensemble(x, temperature=self.temperature)
                teacher_class_probs = teacher_output["class_probs"].detach()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(x)
                logits = output["logits"]
                if self.teacher_ensemble.mode == "learned":
                    adaptive_loss = self.criterion(teacher_output["logits"], y)
                else:
                    adaptive_loss = logits.new_zeros(())
                loss, hard_loss, soft_loss = binary_kl_kd_loss_from_teacher_class_probs(
                    student_logits=logits,
                    teacher_class_probs=teacher_class_probs,
                    hard_labels=y,
                    temperature=self.temperature,
                    alpha=self.alpha,
                )
                loss = loss + self.adaptive_loss_weight * adaptive_loss
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()
            if step % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().cpu()) * self.grad_accum
            running_hard_loss += float(hard_loss.detach().cpu())
            running_soft_loss += float(soft_loss.detach().cpu())
            running_adaptive_loss += float(adaptive_loss.detach().cpu())
            if "weights" in teacher_output:
                batch_weights = teacher_output["weights"].detach().float().cpu()
                teacher_weight_sums += batch_weights.sum(dim=1).double()
                teacher_weight_rows += batch_weights.size(1)
            step_count += 1
            progress.set_postfix(
                loss=running_loss / max(step_count, 1),
                hard=running_hard_loss / max(step_count, 1),
                soft=running_soft_loss / max(step_count, 1),
                adaptive=running_adaptive_loss / max(step_count, 1),
            )

        if step_count % self.grad_accum != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        metrics = {
            "train_loss": running_loss / max(step_count, 1),
            "train_hard_loss": running_hard_loss / max(step_count, 1),
            "train_soft_kd_loss": running_soft_loss / max(step_count, 1),
            "train_adaptive_teacher_loss": running_adaptive_loss / max(step_count, 1),
            "train_seconds": time.perf_counter() - started_at,
            "train_steps": float(step_count),
        }
        if teacher_weight_rows:
            mean_weights = teacher_weight_sums / teacher_weight_rows
            for name, value in zip(self.teacher_ensemble.names, mean_weights.tolist()):
                metrics[f"teacher_weight_{name}"] = float(value)
        return metrics
