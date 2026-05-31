from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.csv_split_converter import convert_csv_splits_to_parquet
from src.utils.config import ensure_dirs, load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pre-split Avazu CSV files to parquet partitions.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--train-csv", required=True, nargs="+", help="Train CSV file path(s)")
    parser.add_argument("--valid-csv", required=True, nargs="+", help="Valid CSV file path(s)")
    parser.add_argument("--test-csv", required=True, nargs="+", help="Labeled test CSV file path(s)")
    parser.add_argument("--processed-dir", default=None, help="Output processed parquet directory")
    parser.add_argument("--chunksize", type=int, default=None, help="Rows per CSV chunk")
    parser.add_argument("--encoded", action="store_true", help="Set if CSV feature columns are already numeric encoded")
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet parts and metadata in output dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.processed_dir:
        config.setdefault("paths", {})["processed_dir"] = args.processed_dir
    ensure_dirs(config)
    seed_everything(int(config.get("project", {}).get("seed", 42)))

    split_paths = {
        "train": [Path(path) for path in args.train_csv],
        "valid": [Path(path) for path in args.valid_csv],
        "test": [Path(path) for path in args.test_csv],
    }
    metadata = convert_csv_splits_to_parquet(
        split_paths=split_paths,
        output_dir=config["paths"]["processed_dir"],
        config=config,
        chunksize=args.chunksize,
        encoded=args.encoded,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

