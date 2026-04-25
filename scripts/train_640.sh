#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python scripts/train_yolo_datamatrix.py \
  --data dataset/data.yaml \
  --model yolo12n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 4 \
  --patience 20 \
  --name datamatrix_yolo12n_640
