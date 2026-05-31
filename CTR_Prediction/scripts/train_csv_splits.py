from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.csv_dataset import AvazuCSVChunkDataset
from src.data.dataloader import make_dataloader
from src.evaluation.evaluate import evaluate_model
from src.evaluation.report import save_metrics_report
from src.features.feature_map import build_field_dims, get_feature_cols
from src.models.base import build_model
from src.training.checkpoint import load_checkpoint
from src.training.trainer import CTRTrainer
from src.utils.config import ensure_dirs, load_config
from src.utils.logger import get_logger
from src.utils.model_stats import count_named_children_parameters, count_parameters, format_parameter_count
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CTR model from pre-split CSV files without loading them into RAM.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--train-csv", required=True, nargs="+", help="Train CSV file path(s)")
    parser.add_argument("--valid-csv", required=True, nargs="+", help="Valid CSV file path(s)")
    parser.add_argument("--test-csv", nargs="+", default=None, help="Optional labeled test CSV file path(s)")
    parser.add_argument("--model", default=None, choices=["lr", "fm", "deepfm", "xdeepfm", "autoint", "nam", "nafi", "kd_nafi"])
    parser.add_argument("--chunksize", type=int, default=None, help="CSV rows per chunk")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 for one large CSV to avoid repeated scans")
    parser.add_argument("--encoded", action="store_true", help="Set if CSV files are already numeric-encoded feature columns")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    return parser.parse_args()


def log_model_params(model_name: str, model, output_dir: str | Path) -> None:
    logger = get_logger("train", Path(output_dir) / "logs" / "train.log")
    param_counts = count_parameters(model)
    logger.info(
        "model=%s total_params=%s trainable_params=%s non_trainable_params=%s",
        model_name,
        format_parameter_count(param_counts["total"]),
        format_parameter_count(param_counts["trainable"]),
        format_parameter_count(param_counts["non_trainable"]),
    )
    branch_counts = count_named_children_parameters(model, ["embedding", "nam", "fin"])
    for branch_name, counts in branch_counts.items():
        logger.info(
            "model=%s branch=%s total_params=%s trainable_params=%s non_trainable_params=%s",
            model_name,
            branch_name,
            format_parameter_count(counts["total"]),
            format_parameter_count(counts["trainable"]),
            format_parameter_count(counts["non_trainable"]),
        )
    summary = {
        "model": model_name,
        "total_params": param_counts["total"],
        "trainable_params": param_counts["trainable"],
        "non_trainable_params": param_counts["non_trainable"],
    }
    if branch_counts:
        summary["branches"] = branch_counts
    print(json.dumps(summary, indent=2))


def make_csv_loader(
    csv_files: list[str],
    feature_cols: list[str],
    target_col: str,
    config: dict,
    chunksize: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    encoded: bool,
    shuffle_files: bool,
    shuffle_rows: bool,
) -> object:
    dataset = AvazuCSVChunkDataset(
        csv_files=csv_files,
        feature_cols=feature_cols,
        target_col=target_col,
        config=config,
        chunksize=chunksize,
        encoded=encoded,
        shuffle_files=shuffle_files,
        shuffle_rows=shuffle_rows,
        seed=int(config.get("project", {}).get("seed", 42)),
    )
    return make_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("paths", {})["output_dir"] = args.output_dir
    ensure_dirs(config)
    seed_everything(int(config.get("project", {}).get("seed", 42)))

    data_cfg = config.get("data", {})
    env_cfg = config.get("environment", {})
    train_cfg = config.get("training", {})
    feature_cols = get_feature_cols(config)
    field_dims = build_field_dims(feature_cols, config)
    target_col = data_cfg.get("target_col", "click")
    chunksize = int(args.chunksize or data_cfg.get("chunksize", 250000))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 2048))
    pin_memory = bool(env_cfg.get("pin_memory", True))
    model_name = args.model or config.get("model", {}).get("name", "nafi")

    train_loader = make_csv_loader(
        args.train_csv,
        feature_cols,
        target_col,
        config,
        chunksize,
        batch_size,
        args.num_workers,
        pin_memory,
        args.encoded,
        shuffle_files=True,
        shuffle_rows=True,
    )
    valid_loader = make_csv_loader(
        args.valid_csv,
        feature_cols,
        target_col,
        config,
        chunksize,
        batch_size,
        args.num_workers,
        pin_memory,
        args.encoded,
        shuffle_files=False,
        shuffle_rows=False,
    )

    model = build_model(model_name, field_dims, config)
    log_model_params(model_name, model, config.get("paths", {}).get("output_dir", "outputs"))
    trainer = CTRTrainer(model, train_loader, valid_loader, config)
    history = trainer.fit()
    print(json.dumps(history[-1] if history else {}, indent=2))

    if args.test_csv:
        test_loader = make_csv_loader(
            args.test_csv,
            feature_cols,
            target_col,
            config,
            chunksize,
            batch_size,
            args.num_workers,
            pin_memory,
            args.encoded,
            shuffle_files=False,
            shuffle_rows=False,
        )
        best_ckpt = Path(config.get("paths", {}).get("output_dir", "outputs")) / "checkpoints" / "best_model.pt"
        test_model = build_model(model_name, field_dims, config)
        load_checkpoint(best_ckpt, test_model)
        metrics = evaluate_model(test_model, test_loader, config)
        report_path = Path(config.get("paths", {}).get("output_dir", "outputs")) / "metrics" / "csv_test_metrics.json"
        save_metrics_report(metrics, report_path)
        print(json.dumps({"test_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
