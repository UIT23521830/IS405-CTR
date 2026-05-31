#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/kaggle_t4x2.yaml}"
MODEL="${2:-nafi}"
OUTPUT_DIR="$(python -c "import yaml; print((yaml.safe_load(open('$CONFIG')) or {}).get('paths', {}).get('output_dir', 'outputs'))")"

python scripts/prepare_avazu.py --config "$CONFIG"
python scripts/train.py --config "$CONFIG" --model "$MODEL"
python scripts/evaluate.py --config "$CONFIG" --checkpoint "$OUTPUT_DIR/checkpoints/best_model.pt"
