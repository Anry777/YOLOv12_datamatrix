from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float]


@dataclass
class ImageReport:
    image_path: str
    annotated_path: str
    width: int
    height: int
    elapsed_ms: int
    detections: list[Detection]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect DataMatrix with trained YOLO.")
    parser.add_argument("input", nargs="?", default=str(ROOT / "test_set"))
    parser.add_argument(
        "--model",
        default=str(ROOT / "runs" / "datamatrix_yolo12n_640" / "weights" / "best.pt"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "test_set_custom"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-crops", action="store_true")
    return parser.parse_args()


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise SystemExit(f"Входной путь не найден: {input_path}")
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def get_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = get_font()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.xyxy
        label = f"DM {index} {detection.confidence:.2f}"
        color = "#00A86B"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label_bbox = draw.textbbox((x1, y1), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        label_top = max(0, y1 - label_height - 8)
        draw.rectangle(
            (x1, label_top, x1 + label_width + 10, label_top + label_height + 8),
            fill=color,
        )
        draw.text((x1 + 5, label_top + 4), label, fill="white", font=font)
    return annotated


def result_to_detections(result) -> list[Detection]:
    detections: list[Detection] = []
    names = result.names or {}
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        detections.append(
            Detection(
                class_id=class_id,
                class_name=str(names.get(class_id, class_id)),
                confidence=float(box.conf[0].item()),
                xyxy=[float(value) for value in box.xyxy[0].tolist()],
            )
        )
    return detections


def save_crops(
    image: Image.Image,
    image_path: Path,
    detections: list[Detection],
    output_dir: Path,
) -> None:
    crops_dir = output_dir / "crops" / image_path.stem
    crops_dir.mkdir(parents=True, exist_ok=True)
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(round(value)) for value in detection.xyxy]
        image.crop((x1, y1, x2, y2)).save(crops_dir / f"{image_path.stem}_{index:03d}.png")


def process_image(
    model: YOLO,
    image_path: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    save_crop_files: bool,
) -> ImageReport:
    image = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
    started = time.perf_counter()
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    detections = result_to_detections(results[0]) if results else []

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
    draw_detections(image, detections).save(annotated_path, quality=92)
    if save_crop_files and detections:
        save_crops(image, image_path, detections, output_dir)

    report = ImageReport(
        image_path=str(image_path),
        annotated_path=str(annotated_path),
        width=image.width,
        height=image.height,
        elapsed_ms=elapsed_ms,
        detections=detections,
    )
    (output_dir / f"{image_path.stem}_detections.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Модель не найдена: {model_path}")

    model = YOLO(str(model_path))
    reports = [
        process_image(
            model=model,
            image_path=image_path,
            output_dir=Path(args.output_dir),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save_crop_files=args.save_crops,
        )
        for image_path in iter_images(Path(args.input))
    ]
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(report) for report in reports], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for report in reports:
        print(
            f"{Path(report.image_path).name}: найдено {len(report.detections)}; "
            f"{report.elapsed_ms} мс; {report.annotated_path}"
        )
    print(f"Сводка: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
