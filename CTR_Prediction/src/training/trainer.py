from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training.callbacks import EarlyStopping
from src.training.checkpoint import save_checkpoint
from src.training.losses import get_loss
from src.training.metrics import compute_auc, compute_logloss
from src.utils.logger import get_logger
from src.utils.memory import log_gpu_usage, log_memory_usage


class CTRTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        config: dict[str, Any],
    ) -> None:
        self.config = config
        self.output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
        self.logger = get_logger("train", self.output_dir / "logs" / "train.log")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if torch.cuda.device_count() >= 2:
            self.logger.info("Using DataParallel with gpu_count=%d", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        train_cfg = config.get("training", {})
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = get_loss(train_cfg.get("loss", "bce_with_logits"))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("model", {}).get("l2_reg", 0.0) or 0.0),
        )
        self.epochs = int(train_cfg.get("epochs", 1))
        self.grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 5.0))
        self.use_amp = bool(train_cfg.get("mixed_precision", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        early_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=int(early_cfg.get("patience", 2)),
            mode=early_cfg.get("mode", "max"),
        )
        self.early_enabled = bool(early_cfg.get("enabled", True))
        self.best_auc = -float("inf")
        self.history: list[dict[str, float]] = []

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        started_at = time.perf_counter()
        self.model.train()
        running_loss = 0.0
        step_count = 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(self.train_loader, desc=f"train epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(x)
                logits = output["logits"]
                loss = self.criterion(logits, y) / self.grad_accum

            self.scaler.scale(loss).backward()
            if step % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().cpu()) * self.grad_accum
            step_count += 1
            progress.set_postfix(loss=running_loss / max(step_count, 1))

        if step_count % self.grad_accum != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "train_loss": running_loss / max(step_count, 1),
            "train_seconds": time.perf_counter() - started_at,
            "train_steps": float(step_count),
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        started_at = time.perf_counter()
        if self.valid_loader is None:
            return {"valid_auc": 0.5, "valid_logloss": float("nan"), "valid_seconds": 0.0}

        self.model.eval()
        y_true: list[float] = []
        y_pred: list[float] = []
        y_pred_nam: list[float] = []
        y_pred_fin: list[float] = []
        losses: list[float] = []
        nam_losses: list[float] = []
        fin_losses: list[float] = []

        for batch in tqdm(self.valid_loader, desc="valid", leave=False):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(x)
                logits = output["logits"]
                loss = self.criterion(logits, y)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            y_pred.extend(probs.tolist())
            y_true.extend(y.detach().float().cpu().numpy().tolist())
            losses.append(float(loss.detach().cpu()))

            if "nam_logits" in output:
                nam_logits = output["nam_logits"]
                y_pred_nam.extend(torch.sigmoid(nam_logits).detach().float().cpu().numpy().tolist())
                nam_losses.append(float(self.criterion(nam_logits, y).detach().cpu()))
            if "fin_logits" in output:
                fin_logits = output["fin_logits"]
                y_pred_fin.extend(torch.sigmoid(fin_logits).detach().float().cpu().numpy().tolist())
                fin_losses.append(float(self.criterion(fin_logits, y).detach().cpu()))

        metrics = {
            "valid_loss": float(np.mean(losses)) if losses else float("nan"),
            "valid_auc": compute_auc(y_true, y_pred),
            "valid_logloss": compute_logloss(y_true, y_pred),
            "valid_seconds": time.perf_counter() - started_at,
        }
        if y_pred_nam:
            metrics["valid_nam_loss"] = float(np.mean(nam_losses)) if nam_losses else float("nan")
            metrics["valid_nam_auc"] = compute_auc(y_true, y_pred_nam)
            metrics["valid_nam_logloss"] = compute_logloss(y_true, y_pred_nam)
            metrics["valid_auc_gain_over_nam"] = metrics["valid_auc"] - metrics["valid_nam_auc"]
            metrics["valid_logloss_gain_over_nam"] = metrics["valid_nam_logloss"] - metrics["valid_logloss"]
        if y_pred_fin:
            metrics["valid_fin_loss"] = float(np.mean(fin_losses)) if fin_losses else float("nan")
            metrics["valid_fin_auc"] = compute_auc(y_true, y_pred_fin)
            metrics["valid_fin_logloss"] = compute_logloss(y_true, y_pred_fin)
            metrics["valid_auc_gain_over_fin"] = metrics["valid_auc"] - metrics["valid_fin_auc"]
            metrics["valid_logloss_gain_over_fin"] = metrics["valid_fin_logloss"] - metrics["valid_logloss"]
        return metrics

    def fit(self) -> list[dict[str, float]]:
        ckpt_dir = self.output_dir / "checkpoints"
        metrics_dir = self.output_dir / "metrics"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            epoch_started_at = time.perf_counter()
            train_metrics = self.train_one_epoch(epoch)
            valid_metrics = self.validate()
            metrics = {
                "epoch": float(epoch),
                **train_metrics,
                **valid_metrics,
                "epoch_seconds": time.perf_counter() - epoch_started_at,
            }
            self.history.append(metrics)
            self.logger.info("epoch=%d metrics=%s", epoch, metrics)
            log_memory_usage(self.logger, prefix=f"epoch_{epoch}")
            log_gpu_usage(self.logger, prefix=f"epoch_{epoch}")

            save_checkpoint(ckpt_dir / "last_model.pt", self.model, self.optimizer, epoch, self.best_auc, self.config)
            valid_auc = float(valid_metrics.get("valid_auc", 0.5))
            if valid_auc > self.best_auc:
                self.best_auc = valid_auc
                save_checkpoint(ckpt_dir / "best_model.pt", self.model, self.optimizer, epoch, self.best_auc, self.config)

            if self.early_enabled and self.early_stopping.step(valid_auc):
                self.logger.info("early_stopping=true epoch=%d", epoch)
                break

        with (metrics_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return self.history
