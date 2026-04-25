# YOLOv12 DataMatrix Training

Минимальный репозиторий для обучения детектора DataMatrix на VPS с NVIDIA GPU.

## Что внутри

- `dataset/` - YOLO detect датасет Roboflow, класс `datamatrix`.
- `test_set/` - реальные тестовые изображения для проверки после обучения.
- `yolo12n.pt` - стартовые веса.
- `scripts/train_yolo_datamatrix.py` - wrapper для обучения Ultralytics YOLO.
- `scripts/detect_datamatrix.py` - проверка обученных весов на `test_set`.
- `scripts/*.sh` - готовые команды для VPS.

Текущий датасет:

- train: 936 images / 936 labels
- val: 35 images / 35 labels
- classes: `datamatrix`

## Подготовка VPS

На VPS лучше остановить процессы, которые занимают GPU-память, например
`llama-server`, если он сейчас висит в `nvidia-smi`.

Дальше:

```bash
cd YOLOv12_datamatrix
bash scripts/setup_vps.sh
```

Скрипт создаст `.venv`, поставит PyTorch CUDA 12.1 wheels, Ultralytics и
проверит датасет. Для твоего драйвера NVIDIA 535 / CUDA 12.2 это нормальный
стартовый вариант.

Проверить GPU вручную:

```bash
source .venv/bin/activate
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Smoke-test

Сначала короткий прогон, чтобы убедиться, что обучение стартует:

```bash
bash scripts/train_smoke.sh
```

Ожидаемый результат:

```text
runs/datamatrix_smoke/weights/best.pt
```

## Основное обучение

Стартовый режим для P104-100 8 GB:

```bash
bash scripts/train_640.sh
```

Если качество на `test_set` недостаточное, второй прогон:

```bash
bash scripts/train_960.sh
```

Самый тяжелый вариант для 8 GB VRAM:

```bash
bash scripts/train_1280.sh
```

Если будет `CUDA out of memory`, уменьши `batch` в соответствующем скрипте:
`16 -> 8 -> 4 -> 2 -> 1`.

Итоговые веса:

```text
runs/<run-name>/weights/best.pt
runs/<run-name>/weights/last.pt
```

## Проверка на test_set

Для результата `train_640.sh`:

```bash
bash scripts/detect_test_set.sh
```

Для другого run:

```bash
bash scripts/detect_test_set.sh runs/datamatrix_yolo12n_960/weights/best.pt outputs/test_set_yolo12n_960
```

Результаты:

```text
outputs/test_set_custom/summary.json
outputs/test_set_custom/*_annotated.jpg
outputs/test_set_custom/crops/
```

## Что забрать обратно

После обучения скопируй с VPS:

```text
runs/datamatrix_yolo12n_640/weights/best.pt
runs/datamatrix_yolo12n_960/weights/best.pt
outputs/
```

На локальной машине эти `best.pt` можно подставлять в текущий стенд
`tools/gs1_detection_bench/detect_datamatrix.py`.

## Текущий обученный baseline

Папка `results/` содержит первый обученный рабочий результат:

```text
results/weights/best.pt
results/results.csv
results/results.png
results/test_set_yolo12n_960-2/
```

Run на VPS: `datamatrix_yolo12n_960-2`.

Основные параметры:

```text
model: yolo12n.pt
imgsz: 960
batch: 10
patience: 25
```

Лучший результат был на 59-й эпохе:

```text
P=1.000
R=0.984
mAP50=0.995
mAP50-95=0.863
```

На `test_set` этот вес дал 9/9 детекций класса `datamatrix`.

## Oriented Dataset

`dataset_oriented/` - сырой Roboflow export с повернутой разметкой DataMatrix.

Важно:

- сейчас в нем есть только `train`;
- labels содержат 4 угла плюс повтор первой точки в конце;
- перед обучением YOLO OBB нужно подготовить отдельный clean split
  `train/valid` и убрать повторную последнюю точку.

Этот датасет нужен для следующего этапа: обучить OBB-модель, чтобы получать
повернутые боксы по марке, а не горизонтальные bbox.

Подготовленный вариант лежит в `dataset_oriented_prepared/`:

```text
train: 255 images / 255 labels
valid: 45 images / 45 labels
label format: class + 4 normalized corner points
```

Датасет пересобирается командой:

```bash
bash scripts/prepare_obb_dataset.sh
```

Smoke-check параметров OBB-обучения:

```bash
source .venv/bin/activate
python scripts/train_yolo_obb.py \
  --data dataset_oriented_prepared/data.yaml \
  --model yolo11n-obb.pt \
  --epochs 3 \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --dry-run
```

Основной запуск OBB на VPS:

```bash
bash scripts/train_obb_960.sh
```

Проверка OBB-весов на `test_set`:

```bash
bash scripts/detect_obb_test_set.sh \
  runs/datamatrix_obb_960/weights/best.pt \
  outputs/test_set_obb
```

OBB detector сохраняет:

- `*_annotated.jpg` с повернутыми рамками;
- `*_detections.json` с 4 точками OBB;
- `crops/` с перспективно выровненными crops по четырем точкам.
