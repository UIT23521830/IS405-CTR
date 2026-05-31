from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.data.preprocess import clean_chunk
from src.data.schema import DTYPE_MAP
from src.features.hashing import hash_features


class AvazuCSVChunkDataset(IterableDataset):
    """Stream one or more CSV files by chunks, with optional Avazu preprocessing."""

    def __init__(
        self,
        csv_files: list[str | Path],
        feature_cols: list[str],
        target_col: str,
        config: dict[str, Any],
        chunksize: int = 250000,
        encoded: bool = False,
        shuffle_files: bool = False,
        shuffle_rows: bool = False,
        seed: int = 42,
    ) -> None:
        self.csv_files = [str(path) for path in csv_files]
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.config = config
        self.chunksize = chunksize
        self.encoded = encoded
        self.shuffle_files = shuffle_files
        self.shuffle_rows = shuffle_rows
        self.seed = seed

    def _worker_files(self) -> list[str]:
        worker = get_worker_info()
        files = list(self.csv_files)
        if self.shuffle_files:
            rng = random.Random(self.seed + (worker.id if worker else 0))
            rng.shuffle(files)
        if worker is None:
            return files
        return files[worker.id :: worker.num_workers]

    def _prepare_chunk(self, chunk: pd.DataFrame, chunk_idx: int) -> pd.DataFrame:
        if self.encoded:
            df = chunk.copy()
            for col in self.feature_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        else:
            df = clean_chunk(chunk, self.config)
            feature_cfg = self.config.get("features", {})
            project_cfg = self.config.get("project", {})
            df = hash_features(
                df,
                feature_cols=self.feature_cols,
                hash_buckets=feature_cfg.get("hash_buckets", {}),
                default_bucket=int(feature_cfg.get("hash_bucket_default", 100000)),
                seed=int(project_cfg.get("seed", 42)),
            )

        missing = [col for col in self.feature_cols + [self.target_col] if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV chunk {chunk_idx}: {missing}")

        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce").fillna(0).astype("float32")
        if self.shuffle_rows:
            df = df.sample(frac=1.0, random_state=self.seed + chunk_idx).reset_index(drop=True)
        return df[self.feature_cols + [self.target_col]]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        dtype_map = DTYPE_MAP if not self.encoded else None
        for path in self._worker_files():
            if not Path(path).exists():
                raise FileNotFoundError(f"CSV split file not found: {path}")
            reader = pd.read_csv(path, chunksize=self.chunksize, dtype=dtype_map)
            for chunk_idx, chunk in enumerate(reader):
                df = self._prepare_chunk(chunk, chunk_idx)
                x = torch.as_tensor(df[self.feature_cols].to_numpy(), dtype=torch.long)
                y = torch.as_tensor(df[self.target_col].to_numpy(), dtype=torch.float32)
                for row_x, row_y in zip(x, y):
                    yield row_x, row_y

