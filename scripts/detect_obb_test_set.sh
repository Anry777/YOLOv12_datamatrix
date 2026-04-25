#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

MODEL_PATH="${1:-runs/datamatrix_obb_960/weights/best.pt}"
OUTPUT_DIR="${2:-outputs/test_set_obb}"
CONF="${CONF:-0.05}"
CROP_PADDING_RATIO="${CROP_PADDING_RATIO:-0.25}"
CROP_PADDING_PX="${CROP_PADDING_PX:-32}"

python scripts/detect_obb_datamatrix.py \
  test_set \
  --model "$MODEL_PATH" \
  --conf "$CONF" \
  --imgsz 960 \
  --device 0 \
  --save-crops \
  --crop-padding-ratio "$CROP_PADDING_RATIO" \
  --crop-padding-px "$CROP_PADDING_PX" \
  --output-dir "$OUTPUT_DIR"
