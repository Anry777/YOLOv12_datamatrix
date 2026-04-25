from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO OBB detector.")
    parser.add_argument("--data", default=str(ROOT / "dataset_oriented_prepared" / "data.yaml"))
    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", default=str(ROOT / "runs"))
    parser.add_argument("--name", default="datamatrix_obb_960")
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
    validate_dataset_yaml(Path(args.data))

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "task": "obb",
    }

    print("Параметры OBB-обучения:")
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
