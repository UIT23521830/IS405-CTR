from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


class VocabEncoder:
    """
    Frequency-based vocabulary encoder for categorical features.

    Pass 1 (fit): scan all parquet files, count frequency per field.
    Pass 2 (transform): map value -> int index (0 = UNK for rare/unseen).
    """

    UNK = 0

    def __init__(self, min_freq: int = 4) -> None:
        self.min_freq = min_freq
        self.vocab: dict[str, dict[str, int]] = {}  # col -> {value: index}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        parquet_files: list[str | Path],
        cat_cols: list[str],
        logger: Any = None,
    ) -> "VocabEncoder":
        """Scan all parquet files and build per-field vocabulary."""
        freq: dict[str, defaultdict[str, int]] = {col: defaultdict(int) for col in cat_cols}

        for i, fpath in enumerate(parquet_files):
            # Only load the columns we need
            available = pd.read_parquet(fpath, columns=cat_cols[:1]).columns.tolist()
            cols_in_file = [c for c in cat_cols if c in pd.read_parquet(fpath, columns=cat_cols[:1]).columns or True]
            try:
                df = pd.read_parquet(fpath, columns=cat_cols)
            except Exception:
                df = pd.read_parquet(fpath)
            for col in cat_cols:
                if col not in df.columns:
                    continue
                counts = df[col].astype(str).value_counts()
                for val, cnt in counts.items():
                    freq[col][val] += int(cnt)
            del df
            gc.collect()
            if logger and (i + 1) % 5 == 0:
                logger.info("vocab_fit_progress file=%d/%d", i + 1, len(parquet_files))

        # Build vocab: index starts at 1, 0 reserved for UNK
        self.vocab = {}
        for col in cat_cols:
            mapping: dict[str, int] = {}
            idx = 1
            for val, cnt in sorted(freq[col].items()):
                if cnt >= self.min_freq:
                    mapping[val] = idx
                    idx += 1
            self.vocab[col] = mapping
            if logger:
                logger.info("vocab_built col=%s unique=%d", col, len(mapping))

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform_col(self, series: pd.Series, col: str) -> pd.Series:
        mapping = self.vocab.get(col, {})
        return series.astype(str).map(mapping).fillna(self.UNK).astype("int32")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, mapping in self.vocab.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(mapping).fillna(self.UNK).astype("int32")
        return df

    # ------------------------------------------------------------------
    # Field dims (for embedding table sizes)
    # ------------------------------------------------------------------

    def field_dims(self, feature_cols: list[str]) -> list[int]:
        """
        Return embedding table size per feature column.
        For vocab-encoded cols: vocab size + 1 (for UNK=0).
        For derived integer cols (day, hour_of_day, weekday): use fixed safe sizes.
        """
        DERIVED_SIZES = {
            "day": 32,          # 0-31
            "hour_of_day": 24,  # 0-23
            "weekday": 7,       # 0-6
        }
        dims = []
        for col in feature_cols:
            if col in DERIVED_SIZES:
                dims.append(DERIVED_SIZES[col])
            elif col in self.vocab:
                dims.append(len(self.vocab[col]) + 1)  # +1 for UNK
            else:
                dims.append(1)  # fallback
        return dims

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"min_freq": self.min_freq, "vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str | Path) -> "VocabEncoder":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        enc = cls(min_freq=data.get("min_freq", 4))
        enc.vocab = data["vocab"]
        return enc
