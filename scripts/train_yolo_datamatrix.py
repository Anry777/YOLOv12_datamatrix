from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector for DataMatrix.")
    parser.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    parser.add_argument("--model", default=str(ROOT / "yolo12n.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default=str(ROOT / "runs"))
    parser.add_argument("--name", default="datamatrix_yolo12n_640")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_dataset_yaml(data_path: Path) -> None:
    data_path = data_path.resolve()
    if not data_path.exists():
        raise SystemExit(f"dataset yaml не найден: {data_path}")

    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    missing = {"train", "val", "names"} - set(data)
    if missing:
        raise SystemExit(f"В dataset yaml не хватает: {', '.join(sorted(missing))}")

    base_path = Path(data.get("path", data_path.parent))
    if not base_path.is_absolute():
        base_path = (data_path.parent / base_path).resolve()

    for split in ("train", "val"):
        split_path = Path(data[split])
        if not split_path.is_absolute():
            split_path = (base_path / split_path).resolve()
        if not split_path.exists():
            raise SystemExit(f"Путь {split} не найден: {split_path}")


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    validate_dataset_yaml(data_path)

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
    }

    print("Параметры обучения:")
    print(f"  model: {args.model}")
    for key, value in train_kwargs.items():
        print(f"  {key}: {value}")

    if args.dry_run:
        return 0

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
