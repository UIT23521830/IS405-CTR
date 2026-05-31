from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info


class AvazuParquetDataset(IterableDataset):
    """Iterable dataset that reads one parquet partition at a time."""

    def __init__(
        self,
        parquet_files: list[str | Path],
        feature_cols: list[str],
        target_col: str = "click",
        shuffle_files: bool = False,
        shuffle_rows: bool = False,
        seed: int = 42,
    ) -> None:
        self.parquet_files = [str(path) for path in parquet_files]
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.shuffle_files = shuffle_files
        self.shuffle_rows = shuffle_rows
        self.seed = seed

    def _worker_files(self) -> list[str]:
        worker = get_worker_info()
        files = list(self.parquet_files)
        if self.shuffle_files:
            rng = random.Random(self.seed + (worker.id if worker else 0))
            rng.shuffle(files)
        if worker is None:
            return files
        return files[worker.id :: worker.num_workers]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for path in self._worker_files():
            df = pd.read_parquet(path, columns=self.feature_cols + [self.target_col])
            if self.shuffle_rows:
                df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
            x = torch.as_tensor(df[self.feature_cols].to_numpy(), dtype=torch.long)
            y = torch.as_tensor(df[self.target_col].to_numpy(), dtype=torch.float32)
            for row_x, row_y in zip(x, y):
                yield row_x, row_y

