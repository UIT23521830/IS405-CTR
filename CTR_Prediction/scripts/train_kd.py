from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.dataloader import make_dataloader
from src.data.dataset import AvazuParquetDataset
from src.data.metadata import load_metadata
from src.models.base import build_model
from src.training.distillation import load_teacher_ensemble, parse_teacher_spec
from src.training.kd_trainer import KDCTRTrainer
from src.utils.config import ensure_dirs, load_config
from src.utils.logger import get_logger
from src.utils.model_stats import count_named_children_parameters, count_parameters, format_parameter_count
from src.utils.seed import seed_everything


MODEL_CHOICES = ["lr", "fm", "deepfm", "xdeepfm", "autoint", "nam", "nafi", "kd_nafi"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KD-NAFI student from teacher checkpoints.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--processed-dir", default=None, help="Override processed parquet directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--student-model", default="kd_nafi", choices=MODEL_CHOICES)
    parser.add_argument(
        "--teacher",
        action="append",
        required=True,
        help="Teacher spec in format model_name:/path/to/best_model.pt. Repeat for multiple teachers.",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        nargs="*",
        default=None,
        help="Optional fixed teacher weights, same order as --teacher.",
    )
    parser.add_argument("--ensemble-mode", default=None, choices=["uniform", "fixed", "confidence", "learned"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--adaptive-hidden-units", type=int, nargs="*", default=None)
    parser.add_argument("--adaptive-learning-rate", type=float, default=None)
    parser.add_argument("--adaptive-loss-weight", type=float, default=None)
    return parser.parse_args()


def log_model_params(model_name: str, model, output_dir: str | Path) -> None:
    logger = get_logger("train", Path(output_dir) / "logs" / "train.log")
    param_counts = count_parameters(model)
    logger.info(
        "student_model=%s total_params=%s trainable_params=%s non_trainable_params=%s",
        model_name,
        format_parameter_count(param_counts["total"]),
        format_parameter_count(param_counts["trainable"]),
        format_parameter_count(param_counts["non_trainable"]),
    )
    branch_param_counts = count_named_children_parameters(model, ["embedding", "nam", "fin"])
    for branch_name, counts in branch_param_counts.items():
        logger.info(
            "student_model=%s branch=%s total_params=%s trainable_params=%s non_trainable_params=%s",
            model_name,
            branch_name,
            format_parameter_count(counts["total"]),
            format_parameter_count(counts["trainable"]),
            format_parameter_count(counts["non_trainable"]),
        )
    summary = {
        "student_model": model_name,
        "total_params": param_counts["total"],
        "trainable_params": param_counts["trainable"],
        "non_trainable_params": param_counts["non_trainable"],
    }
    if branch_param_counts:
        summary["branches"] = branch_param_counts
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.processed_dir:
        config.setdefault("paths", {})["processed_dir"] = args.processed_dir
    if args.output_dir:
        config.setdefault("paths", {})["output_dir"] = args.output_dir

    distill_cfg = config.setdefault("distillation", {})
    distill_cfg["enabled"] = True
    if args.temperature is not None:
        distill_cfg["temperature"] = args.temperature
    if args.alpha is not None:
        distill_cfg["alpha"] = args.alpha
    if args.ensemble_mode is not None:
        distill_cfg["ensemble_mode"] = args.ensemble_mode
    if args.adaptive_hidden_units is not None:
        distill_cfg["adaptive_hidden_units"] = args.adaptive_hidden_units
    if args.adaptive_learning_rate is not None:
        distill_cfg["adaptive_learning_rate"] = args.adaptive_learning_rate
    if args.adaptive_loss_weight is not None:
        distill_cfg["adaptive_loss_weight"] = args.adaptive_loss_weight
    ensemble_mode = distill_cfg.get("ensemble_mode", "uniform")
    if args.teacher_weight is not None and len(args.teacher_weight) > 0:
        ensemble_mode = "fixed"
        distill_cfg["ensemble_mode"] = "fixed"

    ensure_dirs(config)
    seed_everything(int(config.get("project", {}).get("seed", 42)))
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

    student = build_model(args.student_model, metadata["field_dims"], config)
    log_model_params(args.student_model, student, config.get("paths", {}).get("output_dir", "outputs"))

    teacher_specs = [parse_teacher_spec(value) for value in args.teacher]
    teacher_ensemble = load_teacher_ensemble(
        teacher_specs=teacher_specs,
        field_dims=metadata["field_dims"],
        default_config=config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mode=ensemble_mode,
        weights=args.teacher_weight,
        adaptive_hidden_units=list(distill_cfg.get("adaptive_hidden_units", [8])),
    )

    trainer = KDCTRTrainer(student, teacher_ensemble, train_loader, valid_loader, config)
    history = trainer.fit()
    print(json.dumps(history[-1] if history else {}, indent=2))


if __name__ == "__main__":
    main()
