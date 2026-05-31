from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create standard output directories from config."""
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    for child in ["checkpoints", "logs", "metrics", "figures", "submissions"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def get_by_path(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value

