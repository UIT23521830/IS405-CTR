import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd

from src.features.vocab_encoder import VocabEncoder

RAW_CAT_COLS = [
    "C1", "banner_pos",
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model",
    "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
DERIVED_COLS = ["day", "hour_of_day", "weekday"]
FEATURE_COLS = DERIVED_COLS + RAW_CAT_COLS
TARGET_COL = "click"
DROP_COLS = ["id"]

def parse_hour(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in df.columns:
        return df
    hour_str = df["hour"].astype(str).str.zfill(8)
    parsed = pd.to_datetime(hour_str, format="%y%m%d%H", errors="coerce")
    df["day"] = parsed.dt.day.fillna(0).clip(0, 31).astype("int32")
    df["hour_of_day"] = parsed.dt.hour.fillna(0).clip(0, 23).astype("int32")
    df["weekday"] = parsed.dt.weekday.fillna(0).clip(0, 6).astype("int32")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Path to input dir containing train.csv, val.csv, test.csv")
    parser.add_argument("--out", required=True, help="Output dir (e.g. data/avazu_parquet)")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--chunksize", type=int, default=2000000)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    
    # Map input filenames to output split names
    files_to_process = {
        "train": raw_dir / "train.csv",
        "valid": raw_dir / "val.csv",
        "test": raw_dir / "test.csv"
    }

    # Verify input files exist
    for split, fpath in files_to_process.items():
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required file: {fpath}")
            
    for split in files_to_process.keys():
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    
    print(f"Pass 1: Building Vocab from all files with min_freq={args.min_freq}")
    encoder = VocabEncoder(min_freq=args.min_freq)
    
    # 1. First Pass: Build Vocabulary
    for split, fpath in files_to_process.items():
        print(f"  [Vocab] Scanning {fpath.name}")
        reader = pd.read_csv(fpath, chunksize=args.chunksize, iterator=True)
        for i, df in enumerate(reader):
            for col in RAW_CAT_COLS:
                if col in df.columns:
                    counts = df[col].astype(str).value_counts()
                    if col not in encoder.vocab:
                        encoder.vocab[col] = {}
                    for val, cnt in counts.items():
                        encoder.vocab[col][val] = encoder.vocab[col].get(val, 0) + int(cnt)
            print(f"    - Processed chunk {i+1}")
            del df
            gc.collect()
            
    # Re-map vocab indexes based on min_freq
    final_vocab = {}
    for col in RAW_CAT_COLS:
        mapping = {}
        idx = 1
        for val, cnt in sorted(encoder.vocab.get(col, {}).items()):
            if cnt >= args.min_freq:
                mapping[val] = idx
                idx += 1
        final_vocab[col] = mapping
    encoder.vocab = final_vocab
    print("Vocab built successfully.")
    
    # 2. Second Pass: Encode and Save
    print(f"Pass 2: Encoding and saving parquets to {out_dir}")
    
    split_counts = {"train": 0, "valid": 0, "test": 0}
    split_files = {"train": [], "valid": [], "test": []}
    
    for split, fpath in files_to_process.items():
        print(f"  [Encode] Processing {fpath.name} into {split}/")
        reader = pd.read_csv(fpath, chunksize=args.chunksize, iterator=True)
        
        part_counter = 0
        for i, df in enumerate(reader):
            for col in DROP_COLS:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
                    
            df = parse_hour(df)
            df = encoder.transform(df)
            if TARGET_COL in df.columns:
                df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype("int8")
                
            keep_cols = FEATURE_COLS + [TARGET_COL]
            # Handle case where test set might not have a target column
            avail_cols = [c for c in keep_cols if c in df.columns]
            
            sub = df[avail_cols]
            out_path = out_dir / split / f"{split}_part_{part_counter:05d}.parquet"
            sub.to_parquet(out_path, index=False)
            split_files[split].append(f"{split}/{out_path.name}")
            split_counts[split] += len(sub)
            part_counter += 1
                
            print(f"    - Saved chunk {i+1} to {out_path.name}")
            del df
            gc.collect()
        
    print("Saving metadata.json")
    field_dims = encoder.field_dims(FEATURE_COLS)
    metadata = {
        "feature_cols": FEATURE_COLS,
        "field_dims": field_dims,
        "target_col": TARGET_COL,
        "split_files": split_files,
        "split_counts": split_counts,
        "num_rows": sum(split_counts.values()),
        "min_freq": args.min_freq
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("Done! You can now create a Kaggle dataset from the output directory.")

if __name__ == "__main__":
    main()
