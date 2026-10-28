
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "screw_dataset_v5_vertical"
SAVE_DIR = BASE_DIR / "screw_dataset_v6_roi"

YOLO_PATH = (
    BASE_DIR / "runs" / "detect"
    / "bolt_detection_v2" / "weights" / "best.pt"
)

CLASSES = [
    "M3x8", "M3x12",
    "M4x8", "M4x12",
    "M5x8", "M5x12"
]

CONF = 0.25
PADDING = 0.15

model = YOLO(str(YOLO_PATH))

total = 0
saved = 0
failed = 0

for class_name in CLASSES:
    source_folder = SOURCE_DIR / class_name
    save_folder = SAVE_DIR / class_name
    save_folder.mkdir(parents=True, exist_ok=True)

    if not source_folder.exists():
        print(f"폴더 없음: {source_folder}")
        continue

    image_paths = sorted(
        p for p in source_folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    print(f"\n[{class_name}] 사진 {len(image_paths)}장 처리 시작")

    for image_path in image_paths:
        total += 1

        image = cv2.imread(str(image_path))
        if image is None:
            failed += 1
            print(f"읽기 실패: {image_path.name}")
            continue

        results = model.predict(
            image,
            conf=CONF,
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            failed += 1
            print(f"YOLO 미검출: {image_path.name}")
            continue

        # 신뢰도가 가장 높은 검출 박스 사용
        best_idx = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = (
            boxes.xyxy[best_idx].cpu().numpy().astype(int)
        )

        h, w = image.shape[:2]

        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * PADDING)
        pad_y = int(bh * PADDING)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            failed += 1
            continue

        save_path = save_folder / image_path.name

        if cv2.imwrite(str(save_path), roi):
            saved += 1
        else:
            failed += 1

    print(f"[{class_name}] 처리 완료")

print("\n========== 결과 ==========")
print(f"전체 사진: {total}")
print(f"ROI 저장: {saved}")
print(f"미검출/실패: {failed}")
print(f"저장 위치: {SAVE_DIR}")
