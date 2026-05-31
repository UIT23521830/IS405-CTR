from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd


def stable_hash(value: str, num_buckets: int, seed: int = 42) -> int:
    """Hash a value reproducibly into [0, num_buckets)."""
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    payload = f"{seed}:{value}".encode("utf-8", errors="ignore")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest, 16) % num_buckets


def hash_categorical_column(series: pd.Series, num_buckets: int, seed: int = 42) -> pd.Series:
    return series.astype(str).map(lambda value: stable_hash(value, num_buckets, seed)).astype("int64")


def hash_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    hash_buckets: dict[str, int],
    default_bucket: int,
    seed: int = 42,
) -> pd.DataFrame:
    for col in feature_cols:
        if col in df.columns:
            bucket = int(hash_buckets.get(col, default_bucket))
            df[col] = hash_categorical_column(df[col], bucket, seed=seed)
    return df

