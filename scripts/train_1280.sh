#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python scripts/train_yolo_datamatrix.py \
  --data dataset/data.yaml \
  --model yolo12n.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 4 \
  --device 0 \
  --workers 4 \
  --patience 25 \
  --name datamatrix_yolo12n_1280
