from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.metadata import load_metadata
from src.features.feature_map import build_field_dims, get_feature_cols
from src.features.vocab_encoder import VocabEncoder
from src.models.base import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import ensure_dirs, load_config
from src.utils.logger import get_logger
from src.utils.memory import log_memory_usage

# Copy from prepare_from_parquet.py to ensure identical parsing
def parse_hour(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in df.columns:
        return df
    hour_str = df["hour"].astype(str).str.zfill(8)
    parsed = pd.to_datetime(hour_str, format="%y%m%d%H", errors="coerce")
    df["day"] = parsed.dt.day.fillna(0).clip(0, 31).astype("int32")
    df["hour_of_day"] = parsed.dt.hour.fillna(0).clip(0, 23).astype("int32")
    df["weekday"] = parsed.dt.weekday.fillna(0).clip(0, 6).astype("int32")
    return df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Avazu competition test.gz and write submission.csv.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to trained best_model.pt")
    parser.add_argument("--output", required=True, help="Output submission CSV path")
    parser.add_argument("--test-gz", default=None, help="Override path to competition test.gz")
    parser.add_argument("--model", default=None, choices=["lr", "fm", "deepfm", "xdeepfm", "autoint", "nam", "nafi", "kd_nafi"])
    parser.add_argument("--chunksize", type=int, default=None, help="Rows per gzip chunk")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size")
    parser.add_argument("--processed-dir", default=None, help="Override processed parquet directory")
    return parser.parse_args()


def predict_chunk(
    model: torch.nn.Module,
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> list[float]:
    probs: list[float] = []
    values = df[feature_cols].to_numpy()
    for start in range(0, len(values), batch_size):
        batch = torch.as_tensor(values[start : start + batch_size], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch)["logits"]
        probs.extend(torch.sigmoid(logits).detach().float().cpu().numpy().tolist())
    return probs


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.processed_dir:
        config.setdefault("paths", {})["processed_dir"] = args.processed_dir
    ensure_dirs(config)

    logger = get_logger(
        "predict_competition",
        Path(config.get("paths", {}).get("output_dir", "outputs")) / "logs" / "predict_competition.log",
    )
    metadata = load_metadata(config["paths"]["processed_dir"])

    test_path = Path(args.test_gz or config.get("paths", {}).get("raw_test_gz", "test.gz"))
    if not test_path.exists():
        raise FileNotFoundError(f"Competition test.gz not found: {test_path}")

    data_cfg = config.get("data", {})
    target_col = data_cfg.get("target_col", "click")
    id_col = data_cfg.get("id_col", "id")
    chunksize = int(args.chunksize or data_cfg.get("chunksize", 250000))
    batch_size = int(args.batch_size or config.get("training", {}).get("batch_size", 2048))
    feature_cols = metadata["feature_cols"]

    # Load vocab if enabled
    use_vocab = data_cfg.get("use_vocab", False)
    encoder = None
    if use_vocab:
        vocab_path = config.get("paths", {}).get("vocab_path")
        if not vocab_path or not Path(vocab_path).exists():
            raise FileNotFoundError(f"Vocab mode enabled but vocab_path not found: {vocab_path}")
        logger.info("Loading VocabEncoder from %s", vocab_path)
        encoder = VocabEncoder.load(vocab_path)

    model_name = args.model or config.get("model", {}).get("name", "nafi")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, metadata["field_dims"], config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    use_amp = bool(config.get("training", {}).get("mixed_precision", True)) and device.type == "cuda"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    total_rows = 0

    reader = pd.read_csv(
        test_path,
        compression="gzip",
        chunksize=chunksize,
        dtype=str,  # Read everything as string first to match prepare config
    )

    for chunk_idx, chunk in enumerate(reader):
        if id_col not in chunk.columns:
            raise ValueError(f"Missing required id column in competition test file: {id_col}")
        ids = chunk[id_col].copy()

        # Parse hour
        chunk = parse_hour(chunk)

        if use_vocab and encoder:
            # Transform categorical columns based on vocab
            chunk = encoder.transform(chunk)
        else:
            # Fallback to older logic if not vocab
            from src.features.hashing import hash_features
            feature_cfg = config.get("features", {})
            project_cfg = config.get("project", {})
            chunk = hash_features(
                chunk,
                feature_cols=feature_cols,
                hash_buckets=feature_cfg.get("hash_buckets", {}),
                default_bucket=int(feature_cfg.get("hash_bucket_default", 100000)),
                seed=int(project_cfg.get("seed", 42)),
            )

        missing = [col for col in feature_cols if col not in chunk.columns]
        if missing:
            raise ValueError(f"Missing feature columns after preprocessing test chunk: {missing}")

        probs = predict_chunk(model, chunk, feature_cols, batch_size, device, use_amp)
        pd.DataFrame({id_col: ids.astype(str).to_numpy(), target_col: probs}).to_csv(
            output_path,
            mode="w" if not wrote_header else "a",
            index=False,
            header=not wrote_header,
        )
        wrote_header = True
        total_rows += len(chunk)
        logger.info("predicted_chunk=%d rows=%d total_rows=%d", chunk_idx, len(chunk), total_rows)
        log_memory_usage(logger, prefix=f"predict_chunk_{chunk_idx}")

    logger.info("submission_done path=%s rows=%d", output_path, total_rows)
    print(f"wrote {output_path} rows={total_rows}")

if __name__ == "__main__":
    main()
