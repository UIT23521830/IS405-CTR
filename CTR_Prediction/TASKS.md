# TASKS.md

## Completed

- [x] Create project structure for config, scripts, source modules, tests, notebooks, and outputs.
- [x] Add Kaggle/debug/default YAML configs.
- [x] Implement gzip chunk reader with dtype map and memory logging.
- [x] Implement chunk preprocessing with dtype optimization, missing value handling, hour parsing, and optional `id` drop.
- [x] Implement deterministic categorical hashing using MD5.
- [x] Implement gzip-to-parquet partition conversion with train/valid/test split metadata.
- [x] Implement iterable parquet dataset and CTR collate function.
- [x] Implement LR, FM, DeepFM, AutoInt, NAM, FIN, NAFI, and KD loss helpers.
- [x] Implement training loop with AMP, gradient clipping, early stopping, metrics, checkpoints, and optional DataParallel.
- [x] Add CLI scripts for prepare, train, evaluate, predict.
- [x] Add focused tests for hashing, chunk reading, dataset, and model forward passes.
- [x] Add README with Kaggle workflow.

## Recommended Next Steps

- [ ] Run debug mode on Kaggle with the real Avazu `train.gz`.
- [ ] Tune `batch_size`, `embedding_dim`, and `chunksize` after observing RAM/VRAM.
- [ ] Add richer visualization from stored NAFI feature contributions and FIN attention weights.
- [ ] Add true multi-teacher checkpoint loading for KD-NAFI experiments.
- [ ] Add optional DDP support if training time becomes a bottleneck.

