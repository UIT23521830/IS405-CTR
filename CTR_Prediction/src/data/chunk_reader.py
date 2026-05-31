from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from src.data.schema import DTYPE_MAP
from src.utils.logger import get_logger
from src.utils.memory import log_memory_usage


class GzipChunkReader:
    """Read a gzip CSV in chunks to avoid loading the full file into RAM."""

    def __init__(
        self,
        path: str | Path,
        chunksize: int,
        dtype_map: dict[str, str] | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.chunksize = chunksize
        self.dtype_map = dtype_map or DTYPE_MAP
        self.max_rows = max_rows
        self.logger = get_logger(self.__class__.__name__)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        if not self.path.exists():
            raise FileNotFoundError(f"Gzip CSV not found: {self.path}")

        total_rows = 0
        try:
            reader = pd.read_csv(
                self.path,
                compression="gzip",
                chunksize=self.chunksize,
                dtype=self.dtype_map,
                nrows=self.max_rows,
            )
            for chunk_idx, chunk in enumerate(reader):
                total_rows += len(chunk)
                self.logger.info(
                    "chunk=%d rows=%d total_rows=%d",
                    chunk_idx,
                    len(chunk),
                    total_rows,
                )
                log_memory_usage(self.logger, prefix=f"chunk_{chunk_idx}")
                yield chunk
        except Exception as exc:
            raise RuntimeError(f"Failed while reading {self.path}: {exc}") from exc
