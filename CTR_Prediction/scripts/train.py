from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataloader import make_dataloader
from src.data.dataset import AvazuParquetDataset
from src.data.metadata import load_metadata
from src.models.base import build_model
from src.training.trainer import CTRTrainer
from src.utils.config import ensure_dirs, load_config
from src.utils.logger import get_logger
from src.utils.model_stats import count_named_children_parameters, count_parameters, format_parameter_count
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CTR model.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--model", default=None, choices=["lr", "fm", "deepfm", "xdeepfm", "autoint", "nam", "nafi", "kd_nafi"])
    parser.add_argument("--processed-dir", default=None, help="Override processed parquet directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.processed_dir:
        config.setdefault("paths", {})["processed_dir"] = args.processed_dir
    if args.output_dir:
        config.setdefault("paths", {})["output_dir"] = args.output_dir
    ensure_dirs(config)
    seed_everything(int(config.get("project", {}).get("seed", 42)))
    model_name = args.model or config.get("model", {}).get("name", "nafi")
    metadata = load_metadata(config["paths"]["processed_dir"])

    env_cfg = config.get("environment", {})
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 2048))
    num_workers = int(env_cfg.get("num_workers", 0))
    pin_memory = bool(env_cfg.get("pin_memory", True))

    train_dataset = AvazuParquetDataset(
        metadata["split_files"]["train"],
        metadata["feature_cols"],
        metadata["target_col"],
        shuffle_files=True,
        shuffle_rows=True,
        seed=int(config.get("project", {}).get("seed", 42)),
    )
    valid_dataset = AvazuParquetDataset(
        metadata["split_files"]["valid"],
        metadata["feature_cols"],
        metadata["target_col"],
        shuffle_files=False,
        shuffle_rows=False,
        seed=int(config.get("project", {}).get("seed", 42)),
    )
    train_loader = make_dataloader(train_dataset, batch_size, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = make_dataloader(valid_dataset, batch_size, num_workers=num_workers, pin_memory=pin_memory)
    model = build_model(model_name, metadata["field_dims"], config)

    logger = get_logger("train", Path(config.get("paths", {}).get("output_dir", "outputs")) / "logs" / "train.log")
    param_counts = count_parameters(model)
    logger.info(
        "model=%s total_params=%s trainable_params=%s non_trainable_params=%s",
        model_name,
        format_parameter_count(param_counts["total"]),
        format_parameter_count(param_counts["trainable"]),
        format_parameter_count(param_counts["non_trainable"]),
    )
    branch_param_counts = count_named_children_parameters(model, ["embedding", "nam", "fin"])
    for branch_name, counts in branch_param_counts.items():
        logger.info(
            "model=%s branch=%s total_params=%s trainable_params=%s non_trainable_params=%s",
            model_name,
            branch_name,
            format_parameter_count(counts["total"]),
            format_parameter_count(counts["trainable"]),
            format_parameter_count(counts["non_trainable"]),
        )

    param_summary = {
        "model": model_name,
        "total_params": param_counts["total"],
        "trainable_params": param_counts["trainable"],
        "non_trainable_params": param_counts["non_trainable"],
    }
    if branch_param_counts:
        param_summary["branches"] = branch_param_counts
    print(
        json.dumps(param_summary, indent=2)
    )

    trainer = CTRTrainer(model, train_loader, valid_loader, config)
    history = trainer.fit()
    print(json.dumps(history[-1] if history else {}, indent=2))


if __name__ == "__main__":
    main()
