#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python scripts/train_yolo_obb.py \
  --data dataset_oriented_prepared/data.yaml \
  --model yolo11n-obb.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 10 \
  --device 0 \
  --workers 4 \
  --patience 30 \
  --name datamatrix_obb_960
