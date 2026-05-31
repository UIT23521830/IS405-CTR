from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.schema import DTYPE_MAP

MISSING = "__MISSING__"


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply compact numeric dtypes where columns exist."""
    for col, dtype in DTYPE_MAP.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
    return df


def parse_hour_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive day, hour_of_day, and weekday from Avazu YYMMDDHH hour column."""
    if "hour" not in df.columns:
        return df

    hour_str = df["hour"].astype(str).str.zfill(8)
    parsed = pd.to_datetime(hour_str, format="%y%m%d%H", errors="coerce")
    df["day"] = parsed.dt.day.fillna(0).astype("int8")
    df["hour_of_day"] = parsed.dt.hour.fillna(0).astype("int8")
    df["weekday"] = parsed.dt.weekday.fillna(0).astype("int8")
    return df


def clean_chunk(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Clean one chunk without retaining global dataset state."""
    data_cfg = config.get("data", {})
    target_col = data_cfg.get("target_col", "click")
    id_col = data_cfg.get("id_col", "id")

    df = optimize_dtypes(df.copy())
    if data_cfg.get("parse_hour", True):
        df = parse_hour_column(df)

    if data_cfg.get("drop_id", True) and id_col in df.columns:
        df = df.drop(columns=[id_col])

    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(MISSING).astype(str)

    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype("int8")

    return df

