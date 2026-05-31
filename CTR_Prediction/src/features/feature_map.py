from __future__ import annotations

from typing import Any


def get_feature_cols(config: dict[str, Any]) -> list[str]:
    features = config.get("features", {})
    cols = list(features.get("categorical_cols", []))
    for col in features.get("derived_cols", []):
        if col not in cols:
            cols.append(col)
    return cols


def build_field_dims(feature_cols: list[str], config: dict[str, Any]) -> list[int]:
    features = config.get("features", {})
    default_bucket = int(features.get("hash_bucket_default", 100000))
    buckets = features.get("hash_buckets", {})
    return [int(buckets.get(col, default_bucket)) for col in feature_cols]

