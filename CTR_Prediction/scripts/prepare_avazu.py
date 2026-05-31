from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.parquet_converter import convert_gz_to_parquet
from src.utils.config import ensure_dirs, load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Avazu gzip CSV into parquet partitions.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--raw-train-gz", default=None, help="Override path to Avazu train.gz")
    parser.add_argument("--processed-dir", default=None, help="Override output processed parquet directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.raw_train_gz:
        config.setdefault("paths", {})["raw_train_gz"] = args.raw_train_gz
    if args.processed_dir:
        config.setdefault("paths", {})["processed_dir"] = args.processed_dir
    ensure_dirs(config)
    seed_everything(int(config.get("project", {}).get("seed", 42)))
    raw_path = Path(config["paths"]["raw_train_gz"])
    processed_dir = Path(config["paths"]["processed_dir"])
    metadata = convert_gz_to_parquet(raw_path, processed_dir, config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
