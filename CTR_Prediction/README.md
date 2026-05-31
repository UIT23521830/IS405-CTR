# CTR Avazu NAFI

Modular PyTorch project for Click-Through Rate prediction on the Avazu Kaggle dataset. The project is designed for Kaggle notebooks with constrained RAM and T4 x2 GPUs.

## Dataset

Use the Kaggle Avazu Click-Through Rate Prediction dataset. The main input is:

```text
/kaggle/input/avazu-ctr-prediction/train.gz
```

The pipeline never loads the full gzip file into memory. It reads chunks, preprocesses each chunk, hashes high-cardinality categorical fields, and writes parquet partitions.

## Structure

```text
config/       YAML configs for debug, default, and Kaggle T4 x2
src/data/     chunk reading, preprocessing, parquet conversion, dataset
src/features/ hashing and feature metadata helpers
src/models/   LR, FM, DeepFM, AutoInt, NAM, FIN, NAFI, KD-NAFI helpers
src/training/ trainer, metrics, losses, checkpoints
src/evaluation/ evaluation, plots, reports
src/utils/    config, logging, seed, memory, Kaggle helpers
scripts/      CLI entrypoints
tests/        focused smoke tests
```

## Kaggle Workflow

Start with debug mode:

```bash
python scripts/prepare_avazu.py --config config/debug.yaml
python scripts/train.py --config config/debug.yaml --model nafi
python scripts/evaluate.py --config config/debug.yaml --checkpoint outputs/checkpoints/best_model.pt
```

Then scale to `config/default.yaml` or `config/kaggle_t4x2.yaml`.

## Prepare Data

```bash
python scripts/prepare_avazu.py --config config/kaggle_t4x2.yaml
```

This creates parquet partitions under `processed_dir` and writes `metadata.json` with feature columns, field dimensions, split files, and row counts.

## Train Models

```bash
python scripts/train.py --config config/kaggle_t4x2.yaml --model lr
python scripts/train.py --config config/kaggle_t4x2.yaml --model fm
python scripts/train.py --config config/kaggle_t4x2.yaml --model deepfm
python scripts/train.py --config config/kaggle_t4x2.yaml --model autoint
python scripts/train.py --config config/kaggle_t4x2.yaml --model nam
python scripts/train.py --config config/kaggle_t4x2.yaml --model nafi
```

Training uses `BCEWithLogitsLoss`, optional mixed precision, gradient clipping, early stopping, and checkpointing.

## Evaluate

```bash
python scripts/evaluate.py --config config/kaggle_t4x2.yaml --checkpoint outputs/checkpoints/best_model.pt
```

Outputs AUC and logloss.

## Predict Submission

```bash
python scripts/predict.py \
  --config config/kaggle_t4x2.yaml \
  --checkpoint outputs/checkpoints/best_model.pt \
  --output outputs/submissions/submission.csv
```

## Expected Results

- Debug: 200,000 rows, 1 epoch, AUC/logloss emitted, no RAM crash.
- Small: about 3,000,000 rows, 2 epochs, checkpoint and metrics saved.
- Full: full Avazu gzip converted to parquet partitions and trained with batch size around 2048.

## Common Kaggle Issues

- RAM crash while reading gzip: lower `data.chunksize`, keep hashing enabled, and avoid full dataframe operations.
- CUDA OOM: lower `training.batch_size`, `model.embedding_dim`, `fin.num_heads`, or enable gradient accumulation.
- Slow parquet conversion: use larger chunks only after debug mode is stable.
- Missing parquet engine: install `pyarrow` from `requirements.txt`.

