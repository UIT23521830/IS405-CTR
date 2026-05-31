from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: list[dict[str, float]], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    for key, filename in [
        ("train_loss", "loss_curve.png"),
        ("valid_auc", "auc_curve.png"),
        ("valid_logloss", "logloss_curve.png"),
    ]:
        values = [item.get(key) for item in history if key in item]
        if not values:
            continue
        plt.figure()
        plt.plot(epochs[: len(values)], values)
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(output / filename)
        plt.close()


def plot_feature_importance(values: np.ndarray, feature_names: list[str], output_path: str | Path) -> None:
    scores = np.abs(values).mean(axis=0)
    order = np.argsort(scores)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar([feature_names[idx] for idx in order], scores[order])
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_attention_heatmap(attention: np.ndarray, feature_names: list[str], output_path: str | Path) -> None:
    matrix = attention.mean(axis=tuple(range(attention.ndim - 2))) if attention.ndim > 2 else attention
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.xticks(range(len(feature_names)), feature_names, rotation=60, ha="right")
    plt.yticks(range(len(feature_names)), feature_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

