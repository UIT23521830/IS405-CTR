from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.chunk_reader import GzipChunkReader
from src.data.preprocess import clean_chunk
from src.data.schema import DTYPE_MAP
from src.features.feature_map import build_field_dims, get_feature_cols
from src.features.hashing import hash_features
from src.utils.logger import get_logger
from src.utils.memory import log_memory_usage


def _split_name(global_index: pd.Series, train_ratio: float, valid_ratio: float) -> pd.Series:
    bucket = (global_index % 10000) / 10000.0
    return pd.Series(
        pd.Categorical(
            bucket.map(
                lambda value: "train"
                if value < train_ratio
                else "valid"
                if value < train_ratio + valid_ratio
                else "test"
            )
        ),
        index=global_index.index,
    )


def convert_gz_to_parquet(raw_path: str | Path, output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Convert gzip CSV to split parquet partitions without full-data concatenation."""
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("prepare_data", Path(config.get("paths", {}).get("output_dir", "outputs")) / "logs" / "prepare_data.log")

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        logger.info("metadata_exists=true path=%s; using existing conversion", metadata_path)
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    data_cfg = config.get("data", {})
    project_cfg = config.get("project", {})
    feature_cfg = config.get("features", {})
    feature_cols = get_feature_cols(config)
    field_dims = build_field_dims(feature_cols, config)
    target_col = data_cfg.get("target_col", "click")
    chunksize = int(data_cfg.get("chunksize", 250000))
    max_rows = data_cfg.get("max_rows")
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    valid_ratio = float(data_cfg.get("valid_ratio", 0.1))
    use_hashing = bool(data_cfg.get("use_hashing", True))

    split_counts = {"train": 0, "valid": 0, "test": 0}
    split_files: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    part_counter = {"train": 0, "valid": 0, "test": 0}
    total_rows = 0

    for split in split_files:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    reader = GzipChunkReader(
        raw_path,
        chunksize=chunksize,
        dtype_map=DTYPE_MAP,
        max_rows=int(max_rows) if max_rows is not None else None,
    )
    for chunk_idx, chunk in enumerate(reader):
        if max_rows is not None:
            remaining = int(max_rows) - total_rows
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()

        chunk = clean_chunk(chunk, config)
        if use_hashing:
            chunk = hash_features(
                chunk,
                feature_cols=feature_cols,
                hash_buckets=feature_cfg.get("hash_buckets", {}),
                default_bucket=int(feature_cfg.get("hash_bucket_default", 100000)),
                seed=int(project_cfg.get("seed", 42)),
            )

        missing = [col for col in feature_cols + [target_col] if col not in chunk.columns]
        if missing:
            raise ValueError(f"Missing required columns after preprocessing: {missing}")

        global_index = pd.Series(range(total_rows, total_rows + len(chunk)), index=chunk.index)
        chunk["_split"] = _split_name(global_index, train_ratio, valid_ratio)

        for split in split_files:
            split_df = chunk.loc[chunk["_split"] == split, feature_cols + [target_col]]
            if split_df.empty:
                continue
            path = output_dir / split / f"{split}_part_{part_counter[split]:05d}.parquet"
            if path.exists():
                logger.info("resume_skip_existing_part=%s", path)
            else:
                split_df.to_parquet(path, index=False)
            split_files[split].append(str(path))
            split_counts[split] += len(split_df)
            part_counter[split] += 1

        total_rows += len(chunk)
        logger.info("converted_chunk=%d total_rows=%d split_counts=%s", chunk_idx, total_rows, split_counts)
        log_memory_usage(logger, prefix=f"convert_chunk_{chunk_idx}")

    metadata = {
        "feature_cols": feature_cols,
        "field_dims": field_dims,
        "target_col": target_col,
        "split_files": split_files,
        "split_counts": split_counts,
        "num_rows": total_rows,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("conversion_done metadata=%s", metadata_path)
    return metadata
