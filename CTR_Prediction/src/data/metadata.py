from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _candidate_paths(path: str, processed_dir: Path, split: str) -> list[Path]:
    original = Path(path)
    candidates = [original]
    candidates.append(processed_dir / split / original.name)
    candidates.append(processed_dir / original.name)

    parts = original.parts
    if "processed" in parts:
        processed_idx = parts.index("processed")
        suffix = Path(*parts[processed_idx + 1 :])
        candidates.append(processed_dir / suffix)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _rebase_split_files(metadata: dict[str, Any], processed_dir: Path) -> dict[str, Any]:
    split_files = metadata.get("split_files", {})
    rebased: dict[str, list[str]] = {}
    missing: list[str] = []

    for split, files in split_files.items():
        rebased[split] = []
        for path in files:
            resolved = next((candidate for candidate in _candidate_paths(path, processed_dir, split) if candidate.exists()), None)
            if resolved is None:
                fallback = processed_dir / split / Path(path).name
                rebased[split].append(str(fallback))
                missing.append(str(fallback))
            else:
                rebased[split].append(str(resolved))

    metadata["split_files"] = rebased
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            "Some parquet files listed in metadata do not exist after path rebasing. "
            f"processed_dir={processed_dir}\nMissing examples:\n{preview}"
        )
    return metadata


def load_metadata(processed_dir: str | Path) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata. Run prepare first or set --processed-dir correctly: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return _rebase_split_files(metadata, processed_dir)

