from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from src.data.dataset import AvazuParquetDataset


def ctr_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    x, y = zip(*batch)
    return {
        "x": torch.stack(list(x)).long(),
        "y": torch.stack(list(y)).float(),
    }


def make_dataloader(
    dataset: AvazuParquetDataset | IterableDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ctr_collate_fn,
    )
