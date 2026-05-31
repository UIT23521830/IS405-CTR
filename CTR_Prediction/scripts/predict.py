from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.data.metadata import load_metadata
from src.models.base import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import ensure_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CTR probabilities for processed test parquet files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--processed-dir", default=None, help="Override processed parquet directory")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.processed_dir:
        config.setdefault("paths", {})["processed_dir"] = args.processed_dir
    ensure_dirs(config)
    metadata = load_metadata(config["paths"]["processed_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.get("model", {}).get("name", "nafi"), metadata["field_dims"], config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    use_amp = bool(config.get("training", {}).get("mixed_precision", True)) and device.type == "cuda"

    row_ids: list[int] = []
    probs: list[float] = []
    offset = 0
    for path in metadata["split_files"][args.split]:
        df = pd.read_parquet(path, columns=metadata["feature_cols"])
        x = torch.as_tensor(df[metadata["feature_cols"]].to_numpy(), dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)["logits"]
        part_probs = torch.sigmoid(logits).detach().float().cpu().numpy().tolist()
        probs.extend(part_probs)
        row_ids.extend(range(offset, offset + len(part_probs)))
        offset += len(part_probs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": row_ids, "click": probs}).to_csv(output_path, index=False)
    print(f"wrote {output_path} rows={len(probs)}")


if __name__ == "__main__":
    main()
