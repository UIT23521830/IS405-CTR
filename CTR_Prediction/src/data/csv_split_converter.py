from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocess import clean_chunk
from src.data.schema import DTYPE_MAP
from src.features.feature_map import build_field_dims, get_feature_cols
from src.features.hashing import hash_features
from src.utils.logger import get_logger
from src.utils.memory import log_memory_usage


def _prepare_csv_chunk(
    chunk: pd.DataFrame,
    config: dict[str, Any],
    feature_cols: list[str],
    target_col: str,
    encoded: bool,
) -> pd.DataFrame:
    if encoded:
        df = chunk.copy()
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype("int8")
    else:
        df = clean_chunk(chunk, config)
        if config.get("data", {}).get("use_hashing", True):
            feature_cfg = config.get("features", {})
            project_cfg = config.get("project", {})
            df = hash_features(
                df,
                feature_cols=feature_cols,
                hash_buckets=feature_cfg.get("hash_buckets", {}),
                default_bucket=int(feature_cfg.get("hash_bucket_default", 100000)),
                seed=int(project_cfg.get("seed", 42)),
            )

    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after CSV preprocessing: {missing}")
    return df[feature_cols + [target_col]]


def convert_csv_splits_to_parquet(
    split_paths: dict[str, list[str | Path]],
    output_dir: str | Path,
    config: dict[str, Any],
    chunksize: int | None = None,
    encoded: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Convert already split CSV files to parquet partitions compatible with scripts/train.py."""
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(
        "prepare_csv_splits",
        Path(config.get("paths", {}).get("output_dir", "outputs")) / "logs" / "prepare_csv_splits.log",
    )

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not force:
        logger.info("metadata_exists=true path=%s; using existing conversion", metadata_path)
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    data_cfg = config.get("data", {})
    feature_cols = get_feature_cols(config)
    field_dims = build_field_dims(feature_cols, config)
    target_col = data_cfg.get("target_col", "click")
    chunksize = int(chunksize or data_cfg.get("chunksize", 250000))
    max_rows = data_cfg.get("max_rows")

    split_files: dict[str, list[str]] = {split: [] for split in split_paths}
    split_counts: dict[str, int] = {split: 0 for split in split_paths}

    for split in split_paths:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    dtype_map = None if encoded else DTYPE_MAP
    for split, paths in split_paths.items():
        part_idx = 0
        split_seen_rows = 0
        for csv_path in paths:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file for split={split} not found: {csv_path}")

            reader = pd.read_csv(csv_path, chunksize=chunksize, dtype=dtype_map)
            for chunk_idx, chunk in enumerate(reader):
                if max_rows is not None and split == "train":
                    remaining = int(max_rows) - split_seen_rows
                    if remaining <= 0:
                        break
                    chunk = chunk.iloc[:remaining].copy()

                df = _prepare_csv_chunk(chunk, config, feature_cols, target_col, encoded=encoded)
                part_path = output_dir / split / f"{split}_part_{part_idx:05d}.parquet"
                if part_path.exists() and not force:
                    logger.info("resume_skip_existing_part=%s", part_path)
                else:
                    df.to_parquet(part_path, index=False)

                split_files[split].append(str(part_path))
                split_counts[split] += len(df)
                split_seen_rows += len(df)
                logger.info(
                    "converted_split=%s source=%s chunk=%d rows=%d split_rows=%d",
                    split,
                    csv_path,
                    chunk_idx,
                    len(df),
                    split_seen_rows,
                )
                log_memory_usage(logger, prefix=f"{split}_chunk_{chunk_idx}")

                part_idx += 1

    metadata = {
        "feature_cols": feature_cols,
        "field_dims": field_dims,
        "target_col": target_col,
        "split_files": split_files,
        "split_counts": split_counts,
        "num_rows": sum(split_counts.values()),
        "source": "csv_splits",
        "encoded": encoded,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("csv_split_conversion_done metadata=%s", metadata_path)
    return metadata

