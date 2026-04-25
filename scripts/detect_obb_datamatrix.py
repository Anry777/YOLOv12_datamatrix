from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class OBBDetection:
    class_id: int
    class_name: str
    confidence: float
    points: list[list[float]]


@dataclass
class OBBReport:
    image_path: str
    annotated_path: str
    width: int
    height: int
    elapsed_ms: int
    detections: list[OBBDetection]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect DataMatrix with YOLO OBB.")
    parser.add_argument("input", nargs="?", default=str(ROOT / "test_set"))
    parser.add_argument(
        "--model",
        default=str(ROOT / "runs" / "datamatrix_obb_960" / "weights" / "best.pt"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "test_set_obb"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--crop-padding-ratio", type=float, default=0.12)
    parser.add_argument("--crop-padding-px", type=int, default=12)
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


def order_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start_index = np.argmin(ordered.sum(axis=1))
    return np.roll(ordered, -start_index, axis=0).astype("float32")


def pad_points(points: np.ndarray, padding_px: int, padding_ratio: float) -> np.ndarray:
    center = points.mean(axis=0)
    padded = points.copy().astype("float32")
    for index, point in enumerate(points):
        vector = point - center
        length = np.linalg.norm(vector)
        if length <= 0:
            continue
        padded[index] = point + vector / length * padding_px + vector * padding_ratio
    return padded


def perspective_crop(
    image: Image.Image,
    points: list[list[float]],
    padding_px: int,
    padding_ratio: float,
) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    source = pad_points(np.array(points, dtype="float32"), padding_px, padding_ratio)
    source = order_points(source)

    width_top = np.linalg.norm(source[1] - source[0])
    width_bottom = np.linalg.norm(source[2] - source[3])
    height_left = np.linalg.norm(source[3] - source[0])
    height_right = np.linalg.norm(source[2] - source[1])
    output_width = max(1, int(round(max(width_top, width_bottom))))
    output_height = max(1, int(round(max(height_left, height_right))))

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    crop = cv2.warpPerspective(
        rgb,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(crop)


def draw_detections(image: Image.Image, detections: list[OBBDetection]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = get_font()
    for index, detection in enumerate(detections, start=1):
        polygon = [tuple(point) for point in detection.points]
        draw.line(polygon + [polygon[0]], fill="#00A86B", width=4)
        x, y = polygon[0]
        label = f"OBB {index} {detection.confidence:.2f}"
        bbox = draw.textbbox((x, y), label, font=font)
        draw.rectangle((x, max(0, y - 24), x + (bbox[2] - bbox[0]) + 10, y), fill="#00A86B")
        draw.text((x + 5, max(0, y - 20)), label, fill="white", font=font)
    return annotated


def result_to_detections(result) -> list[OBBDetection]:
    if result.obb is None:
        return []

    names = result.names or {}
    points = result.obb.xyxyxyxy.cpu().numpy()
    confidences = result.obb.conf.cpu().numpy()
    classes = result.obb.cls.cpu().numpy()
    detections: list[OBBDetection] = []
    for class_id_raw, confidence, point_set in zip(classes, confidences, points):
        class_id = int(class_id_raw)
        detections.append(
            OBBDetection(
                class_id=class_id,
                class_name=str(names.get(class_id, class_id)),
                confidence=float(confidence),
                points=[[float(x), float(y)] for x, y in point_set],
            )
        )
    return detections


def save_crops(
    image: Image.Image,
    image_path: Path,
    detections: list[OBBDetection],
    output_dir: Path,
    padding_px: int,
    padding_ratio: float,
) -> None:
    crops_dir = output_dir / "crops" / image_path.stem
    crops_dir.mkdir(parents=True, exist_ok=True)
    for index, detection in enumerate(detections, start=1):
        crop = perspective_crop(
            image=image,
            points=detection.points,
            padding_px=padding_px,
            padding_ratio=padding_ratio,
        )
        crop.save(crops_dir / f"{image_path.stem}_obb_{index:03d}.png")


def process_image(
    model: YOLO,
    image_path: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    save_crop_files: bool,
    crop_padding_px: int,
    crop_padding_ratio: float,
) -> OBBReport:
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
        save_crops(
            image=image,
            image_path=image_path,
            detections=detections,
            output_dir=output_dir,
            padding_px=crop_padding_px,
            padding_ratio=crop_padding_ratio,
        )

    report = OBBReport(
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
            crop_padding_px=args.crop_padding_px,
            crop_padding_ratio=args.crop_padding_ratio,
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
