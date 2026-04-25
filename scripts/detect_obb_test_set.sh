#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

MODEL_PATH="${1:-runs/datamatrix_obb_960/weights/best.pt}"
OUTPUT_DIR="${2:-outputs/test_set_obb}"

python scripts/detect_obb_datamatrix.py \
  test_set \
  --model "$MODEL_PATH" \
  --conf 0.25 \
  --imgsz 960 \
  --device 0 \
  --save-crops \
  --output-dir "$OUTPUT_DIR"
