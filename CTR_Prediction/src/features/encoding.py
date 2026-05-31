from __future__ import annotations

import pandas as pd


def rare_category_encode(series: pd.Series, min_count: int = 10, rare_token: str = "__RARE__") -> pd.Series:
    counts = series.value_counts(dropna=False)
    return series.where(series.map(counts) >= min_count, rare_token)

