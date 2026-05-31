from pathlib import Path

import pandas as pd

from src.data.dataloader import ctr_collate_fn
from src.data.dataset import AvazuParquetDataset


def test_parquet_dataset(tmp_path: Path) -> None:
    path = tmp_path / "part.parquet"
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "click": [0, 1]}).to_parquet(path, index=False)
    dataset = AvazuParquetDataset([path], ["a", "b"], "click")
    batch = ctr_collate_fn(list(dataset))
    assert batch["x"].shape == (2, 2)
    assert batch["y"].shape == (2,)

