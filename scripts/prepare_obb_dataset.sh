#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python scripts/prepare_obb_dataset.py \
  --source dataset_oriented \
  --output dataset_oriented_prepared \
  --val-ratio 0.15 \
  --seed 20260425
