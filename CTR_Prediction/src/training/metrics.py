from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def compute_auc(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(np.unique(y_true_arr)) < 2:
        return 0.5
    return float(roc_auc_score(y_true_arr, y_pred_arr))


def compute_logloss(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.clip(np.asarray(y_pred), 1e-7, 1 - 1e-7)
    return float(log_loss(y_true_arr, y_pred_arr, labels=[0, 1]))

