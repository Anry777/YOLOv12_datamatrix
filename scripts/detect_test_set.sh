#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

MODEL_PATH="${1:-runs/datamatrix_yolo12n_640/weights/best.pt}"
OUTPUT_DIR="${2:-outputs/test_set_custom}"

python scripts/detect_datamatrix.py \
  test_set \
  --model "$MODEL_PATH" \
  --conf 0.25 \
  --imgsz 1280 \
  --device 0 \
  --save-crops \
  --output-dir "$OUTPUT_DIR"
