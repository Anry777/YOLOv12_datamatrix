from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Roboflow polygon/OBB export for Ultralytics YOLO OBB.",
    )
    parser.add_argument("--source", default="dataset_oriented")
    parser.add_argument("--output", default="dataset_oriented_prepared")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260425)
    return parser.parse_args()


def source_group_key(image_path: Path) -> str:
    name = image_path.stem
    marker = ".rf."
    if marker in name:
        return name.split(marker, 1)[0]
    return name


def convert_label_line(line: str) -> str:
    parts = line.split()
    if len(parts) not in {9, 11}:
        raise ValueError(f"Ожидалось 9 или 11 значений, получено {len(parts)}: {line}")

    coords = parts[1:]
    if len(coords) == 10:
        coords = coords[:8]

    values = [float(value) for value in coords]
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Координаты OBB должны быть нормализованы 0..1: {line}")

    return "0 " + " ".join(f"{value:.10f}".rstrip("0").rstrip(".") for value in values)


def copy_group(
    image_paths: list[Path],
    source_labels_dir: Path,
    output_dir: Path,
    split: str,
) -> int:
    image_output_dir = output_dir / split / "images"
    label_output_dir = output_dir / split / "labels"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in image_paths:
        label_path = source_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Label не найден: {label_path}")

        shutil.copy2(image_path, image_output_dir / image_path.name)
        converted_lines = [
            convert_label_line(line.strip())
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        (label_output_dir / label_path.name).write_text(
            "\n".join(converted_lines) + "\n",
            encoding="utf-8",
        )
        copied += 1
    return copied


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()
    images_dir = source_dir / "train" / "images"
    labels_dir = source_dir / "train" / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"Не найден raw train/images или train/labels в {source_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    groups: dict[str, list[Path]] = defaultdict(list)
    for image_path in sorted(images_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            groups[source_group_key(image_path)].append(image_path)

    if not groups:
        raise SystemExit(f"В {images_dir} нет изображений")

    group_keys = sorted(groups)
    random.Random(args.seed).shuffle(group_keys)
    val_count = max(1, round(len(group_keys) * args.val_ratio))
    val_keys = set(group_keys[:val_count])

    train_images: list[Path] = []
    val_images: list[Path] = []
    for key in sorted(groups):
        if key in val_keys:
            val_images.extend(groups[key])
        else:
            train_images.extend(groups[key])

    copied_train = copy_group(train_images, labels_dir, output_dir, "train")
    copied_val = copy_group(val_images, labels_dir, output_dir, "valid")

    data = {
        "path": str(output_dir).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "names": {0: "datamatrix"},
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Groups: {len(group_keys)}")
    print(f"Train images: {copied_train}")
    print(f"Valid images: {copied_val}")
    print(f"Dataset: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
