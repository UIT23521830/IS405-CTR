from __future__ import annotations

from pathlib import Path
from typing import Any


def get_feature_cols(config: dict[str, Any]) -> list[str]:
    features = config.get("features", {})
    cols = list(features.get("categorical_cols", []))
    for col in features.get("derived_cols", []):
        if col not in cols:
            cols.append(col)
    return cols


def build_field_dims(feature_cols: list[str], config: dict[str, Any]) -> list[int]:
    """Build field dims. Uses vocab if use_vocab=true in config, else hashing."""
    features = config.get("features", {})

    if config.get("data", {}).get("use_vocab", False):
        # Field dims come from the vocab file written by prepare_from_parquet.py
        # They are already baked into metadata.json, so this is a fallback.
        from src.features.vocab_encoder import VocabEncoder

        vocab_path = config.get("paths", {}).get("vocab_path")
        if vocab_path and Path(vocab_path).exists():
            encoder = VocabEncoder.load(vocab_path)
            return encoder.field_dims(feature_cols)

    # Default: hashing-based field dims
    default_bucket = int(features.get("hash_bucket_default", 100000))
    buckets = features.get("hash_buckets", {})
    return [int(buckets.get(col, default_bucket)) for col in feature_cols]

