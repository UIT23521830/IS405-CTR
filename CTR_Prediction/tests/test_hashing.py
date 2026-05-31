from src.features.hashing import hash_categorical_column, stable_hash


def test_stable_hash_reproducible() -> None:
    assert stable_hash("abc", 100, seed=42) == stable_hash("abc", 100, seed=42)
    assert 0 <= stable_hash("abc", 100, seed=42) < 100


def test_hash_categorical_column() -> None:
    import pandas as pd

    values = hash_categorical_column(pd.Series(["a", "b", "a"]), 10, seed=7)
    assert values.iloc[0] == values.iloc[2]
    assert values.max() < 10

