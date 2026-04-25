from __future__ import annotations

import argparse
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_split_path(data_path: Path, data: dict, split: str) -> Path:
    base_path = Path(data.get("path", data_path.parent))
    if not base_path.is_absolute():
        base_path = (data_path.parent / base_path).resolve()

    split_path = Path(data[split])
    if not split_path.is_absolute():
        split_path = (base_path / split_path).resolve()
    return split_path


def count_files(path: Path, extensions: set[str] | None = None) -> int:
    if not path.exists():
        return 0
    files = [item for item in path.iterdir() if item.is_file()]
    if extensions is None:
        return len(files)
    return sum(1 for item in files if item.suffix.lower() in extensions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check YOLO dataset layout.")
    parser.add_argument("--data", default="dataset/data.yaml")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"data.yaml не найден: {data_path}")

    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    for key in ("train", "val", "names"):
        if key not in data:
            raise SystemExit(f"В data.yaml нет обязательного поля: {key}")

    print(f"data.yaml: {data_path}")
    print(f"names: {data['names']}")

    for split in ("train", "val"):
        images_dir = resolve_split_path(data_path, data, split)
        labels_dir = images_dir.parent / "labels"
        image_count = count_files(images_dir, IMAGE_EXTENSIONS)
        label_count = count_files(labels_dir, {".txt"})
        print(f"{split}: images={image_count}, labels={label_count}")

        if not images_dir.exists():
            raise SystemExit(f"Папка изображений не найдена: {images_dir}")
        if not labels_dir.exists():
            raise SystemExit(f"Папка labels не найдена: {labels_dir}")
        if image_count == 0:
            raise SystemExit(f"Нет изображений в {images_dir}")
        if image_count != label_count:
            raise SystemExit(
                f"Количество images и labels не совпадает для {split}: "
                f"{image_count} != {label_count}"
            )

    print("OK: датасет выглядит готовым к обучению.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
