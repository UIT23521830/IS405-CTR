from __future__ import annotations

import os


def is_kaggle() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

