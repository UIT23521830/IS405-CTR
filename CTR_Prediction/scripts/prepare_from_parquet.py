from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd

from src.features.vocab_encoder import VocabEncoder
from src.utils.config import ensure_dirs, load_config
from src.utils.logger import get_logger
from src.utils.memory import log_memory_usage
from src.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Feature columns (must match what parquet files contain)
# ---------------------------------------------------------------------------

RAW_CAT_COLS = [
    "C1", "banner_pos",
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model",
    "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
DERIVED_COLS = ["day", "hour_of_day", "weekday"]
FEATURE_COLS = DERIVED_COLS + RAW_CAT_COLS  # 24 fields total
TARGET_COL = "click"
DROP_COLS = ["id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_hour(df: pd.DataFrame) -> pd.DataFrame:
    """YYMMDDHH -> day, hour_of_day, weekday."""
    if "hour" not in df.columns:
        return df
    hour_str = df["hour"].astype(str).str.zfill(8)
    parsed = pd.to_datetime(hour_str, format="%y%m%d%H", errors="coerce")
    df["day"] = parsed.dt.day.fillna(0).clip(0, 31).astype("int32")
    df["hour_of_day"] = parsed.dt.hour.fillna(0).clip(0, 23).astype("int32")
    df["weekday"] = parsed.dt.weekday.fillna(0).clip(0, 6).astype("int32")
    return df


def get_hour_split_thresholds(
    parquet_files: list[Path],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> tuple[int, int]:
    """Find hour values for time-based 8:1:1 split."""
    hours: set[int] = set()
    for fpath in parquet_files:
        df = pd.read_parquet(fpath, columns=["hour"])
        hours.update(df["hour"].dropna().astype(int).tolist())
        del df
    hours_sorted = sorted(hours)
    n = len(hours_sorted)
    valid_thresh = hours_sorted[int(n * train_ratio)]
    test_thresh = hours_sorted[int(n * (train_ratio + valid_ratio))]
    return valid_thresh, test_thresh


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def prepare_from_parquet(
    raw_parquet_dir: Path,
    processed_dir: Path,
    config: dict,
    logger,
) -> dict:
    """
    Two-pass pipeline:
      Pass 1: fit VocabEncoder on raw parquet files.
      Pass 2: encode + parse hour + time-based split + save encoded parquets.

    Output: metadata.json compatible with scripts/train.py
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = processed_dir / "metadata.json"

    if metadata_path.exists():
        logger.info("metadata_exists=true; skipping conversion. Delete metadata.json to re-run.")
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    data_cfg = config.get("data", {})
    project_cfg = config.get("project", {})
    min_freq = int(config.get("features", {}).get("min_freq", 4))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    valid_ratio = float(data_cfg.get("valid_ratio", 0.1))

    raw_files = sorted(raw_parquet_dir.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No parquet files found in {raw_parquet_dir}")
    logger.info("found_raw_files=%d in=%s", len(raw_files), raw_parquet_dir)

    # Create output split directories
    for split in ("train", "valid", "test"):
        (processed_dir / split).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Pass 1 – fit vocab
    # -----------------------------------------------------------------------
    vocab_path = processed_dir / "vocab.json"
    if vocab_path.exists():
        logger.info("loading_existing_vocab path=%s", vocab_path)
        encoder = VocabEncoder.load(vocab_path)
    else:
        logger.info("pass1_start: fitting vocab min_freq=%d", min_freq)
        t0 = time.time()
        encoder = VocabEncoder(min_freq=min_freq)
        encoder.fit(raw_files, RAW_CAT_COLS, logger=logger)
        encoder.save(vocab_path)
        logger.info("pass1_done elapsed=%.1fs vocab_saved=%s", time.time() - t0, vocab_path)

    field_dims = encoder.field_dims(FEATURE_COLS)
    logger.info("field_dims=%s", field_dims)

    # -----------------------------------------------------------------------
    # Determine time-based split thresholds
    # -----------------------------------------------------------------------
    logger.info("computing_time_split_thresholds train_ratio=%s valid_ratio=%s", train_ratio, valid_ratio)
    valid_thresh, test_thresh = get_hour_split_thresholds(raw_files, train_ratio, valid_ratio)
    logger.info("valid_thresh=%d test_thresh=%d", valid_thresh, test_thresh)

    # -----------------------------------------------------------------------
    # Pass 2 – encode + split
    # -----------------------------------------------------------------------
    logger.info("pass2_start: encode + split")
    t0 = time.time()
    split_counts = {"train": 0, "valid": 0, "test": 0}
    split_files: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    part_counter = {"train": 0, "valid": 0, "test": 0}

    for i, fpath in enumerate(raw_files):
        df = pd.read_parquet(fpath)

        # Drop unused columns
        for col in DROP_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Parse hour → derived features
        df = parse_hour(df)

        # Encode categorical columns via vocab
        df = encoder.transform(df)

        # Ensure target is int8
        if TARGET_COL in df.columns:
            df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype("int8")

        # Time-based split using raw hour column
        splits_masks = [
            ("test",  df["hour"] >= test_thresh),
            ("valid", (df["hour"] >= valid_thresh) & (df["hour"] < test_thresh)),
            ("train", df["hour"] < valid_thresh),
        ]
        keep_cols = FEATURE_COLS + [TARGET_COL]

        for split, mask in splits_masks:
            sub = df.loc[mask, keep_cols].copy()
            if sub.empty:
                continue
            out_path = processed_dir / split / f"{split}_part_{part_counter[split]:05d}.parquet"
            sub.to_parquet(out_path, index=False)
            split_files[split].append(str(out_path))
            split_counts[split] += len(sub)
            part_counter[split] += 1

        del df
        gc.collect()
        if (i + 1) % 5 == 0:
            logger.info("pass2_progress file=%d/%d counts=%s", i + 1, len(raw_files), split_counts)
            log_memory_usage(logger, prefix=f"pass2_file_{i}")

    logger.info("pass2_done elapsed=%.1fs counts=%s", time.time() - t0, split_counts)

    # -----------------------------------------------------------------------
    # Save metadata.json (same schema as convert_gz_to_parquet)
    # -----------------------------------------------------------------------
    metadata = {
        "feature_cols": FEATURE_COLS,
        "field_dims": field_dims,
        "target_col": TARGET_COL,
        "split_files": split_files,
        "split_counts": split_counts,
        "num_rows": sum(split_counts.values()),
        "valid_hour_thresh": valid_thresh,
        "test_hour_thresh": test_thresh,
        "vocab_path": str(vocab_path),
        "min_freq": min_freq,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("metadata_saved path=%s", metadata_path)
    return metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare already-parquet Avazu data: vocab encode + time-based split."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--raw-parquet-dir", default=None, help="Directory containing raw train_part_*.parquet files")
    parser.add_argument("--processed-dir", default=None, help="Output directory for encoded parquets + metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.raw_parquet_dir:
        config.setdefault("paths", {})["raw_parquet_dir"] = args.raw_parquet_dir
    if args.processed_dir:
        config.setdefault("paths", {})["processed_dir"] = args.processed_dir

    ensure_dirs(config)
    seed_everything(int(config.get("project", {}).get("seed", 42)))

    raw_parquet_dir = Path(config["paths"]["raw_parquet_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    output_dir = Path(config["paths"].get("output_dir", "outputs"))

    logger = get_logger("prepare_from_parquet", output_dir / "logs" / "prepare_from_parquet.log")
    metadata = prepare_from_parquet(raw_parquet_dir, processed_dir, config, logger)
    print(json.dumps({k: v for k, v in metadata.items() if k != "split_files"}, indent=2))


if __name__ == "__main__":
    main()
