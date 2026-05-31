from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.training.metrics import compute_auc, compute_logloss


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, config: dict[str, Any]) -> dict[str, float]:
    started_at = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    use_amp = bool(config.get("training", {}).get("mixed_precision", True)) and device.type == "cuda"
    y_true: list[float] = []
    y_pred: list[float] = []
    y_pred_nam: list[float] = []
    y_pred_fin: list[float] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(x)
            logits = output["logits"]
        y_pred.extend(torch.sigmoid(logits).detach().float().cpu().numpy().tolist())
        y_true.extend(y.detach().float().cpu().numpy().tolist())
        if "nam_logits" in output:
            y_pred_nam.extend(torch.sigmoid(output["nam_logits"]).detach().float().cpu().numpy().tolist())
        if "fin_logits" in output:
            y_pred_fin.extend(torch.sigmoid(output["fin_logits"]).detach().float().cpu().numpy().tolist())

    metrics = {
        "auc": compute_auc(y_true, y_pred),
        "logloss": compute_logloss(y_true, y_pred),
        "seconds": time.perf_counter() - started_at,
    }
    if y_pred_nam:
        metrics["nam_auc"] = compute_auc(y_true, y_pred_nam)
        metrics["nam_logloss"] = compute_logloss(y_true, y_pred_nam)
        metrics["auc_gain_over_nam"] = metrics["auc"] - metrics["nam_auc"]
        metrics["logloss_gain_over_nam"] = metrics["nam_logloss"] - metrics["logloss"]
    if y_pred_fin:
        metrics["fin_auc"] = compute_auc(y_true, y_pred_fin)
        metrics["fin_logloss"] = compute_logloss(y_true, y_pred_fin)
        metrics["auc_gain_over_fin"] = metrics["auc"] - metrics["fin_auc"]
        metrics["logloss_gain_over_fin"] = metrics["fin_logloss"] - metrics["logloss"]
    return metrics
