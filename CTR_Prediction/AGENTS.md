# AGENTS.md

## Project Mission

Build a modular Python/PyTorch project for Avazu Click-Through Rate prediction on Kaggle. The primary constraint is memory safety: never load the full `train.gz` into RAM. All data preparation must stream gzip chunks, optimize dtypes, hash categorical values deterministically, and write parquet partitions for training.

## Operating Principles

- Prefer simple, Kaggle-safe implementations over leaderboard-specific tricks.
- Do not use `pd.read_csv("train.gz")` without `chunksize`.
- Do not concatenate the full Avazu dataset in memory.
- Keep modules small and explicit. Scripts should orchestrate, not contain model or data logic.
- Use stable hashing from `src/features/hashing.py`; do not use Python `hash()`.
- Use logits during training with `BCEWithLogitsLoss`.
- Save checkpoints and metrics under `outputs/`.

## Key Commands

```bash
python scripts/prepare_avazu.py --config config/debug.yaml
python scripts/train.py --config config/debug.yaml --model nafi
python scripts/evaluate.py --config config/debug.yaml --checkpoint outputs/checkpoints/best_model.pt
python scripts/predict.py --config config/debug.yaml --checkpoint outputs/checkpoints/best_model.pt --output outputs/submissions/submission.csv
```

## Acceptance Focus

1. Debug mode runs end-to-end without RAM crashes.
2. Parquet conversion resumes safely when existing partitions are present.
3. LR/FM/DeepFM/AutoInt/NAM/NAFI can be instantiated and trained.
4. AUC and logloss are reported.
5. Best and last checkpoints are saved.
6. Kaggle T4 x2 uses mixed precision and `DataParallel` when available.

