from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------
BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "runs" / "classify" / "bolt_v2" / "weights" / "best.pt"
TEST_DIR = BASE / "screw_cls_v2" / "test"

OUTPUT_DIR = BASE / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise SystemExit(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

if not TEST_DIR.exists():
    raise SystemExit(f"테스트 폴더를 찾을 수 없습니다: {TEST_DIR}")

# --------------------------------------------------
# 2. 모델 불러오기
# --------------------------------------------------
model = YOLO(str(MODEL_PATH))

class_names = [
    "M3x8", "M3x12",
    "M4x8", "M4x12",
    "M5x8", "M5x12",
]

print("모델 클래스:", model.names)

# --------------------------------------------------
# 3. 테스트 이미지 예측
# --------------------------------------------------
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

y_true = []
y_pred = []

for true_class in class_names:
    class_dir = TEST_DIR / true_class

    if not class_dir.exists():
        raise SystemExit(f"테스트 클래스 폴더가 없습니다: {class_dir}")

    image_paths = sorted(
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    for image_path in image_paths:
        result = model.predict(
            source=str(image_path),
            verbose=False
        )[0]

        predicted_id = int(result.probs.top1)
        predicted_class = result.names[predicted_id]

        y_true.append(true_class)
        y_pred.append(predicted_class)

        if predicted_class != true_class:
            print(
                f"오분류: 실제 {true_class} "
                f"-> 예측 {predicted_class} | {image_path.name}"
            )

if not y_true:
    raise SystemExit("테스트 이미지가 없습니다.")

# --------------------------------------------------
# 4. 성능지표 계산
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)

macro_precision = precision_score(
    y_true, y_pred,
    labels=class_names,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true, y_pred,
    labels=class_names,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true, y_pred,
    labels=class_names,
    average="macro",
    zero_division=0
)

print("\n========== 최종 성능평가 ==========")
print(f"테스트 이미지 수: {len(y_true)}")
print(f"Accuracy:       {accuracy:.4f} ({accuracy:.1%})")
print(f"Macro Precision:{macro_precision:.4f}")
print(f"Macro Recall:   {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

print("\n========== 규격별 성능 ==========")
print(classification_report(
    y_true,
    y_pred,
    labels=class_names,
    target_names=class_names,
    digits=4,
    zero_division=0
))

# --------------------------------------------------
# 5. 혼동행렬 생성 및 저장
# --------------------------------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=class_names
)

# 원본 혼동행렬
fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45,
    colorbar=False
)

ax.set_title("Bolt Classification - Confusion Matrix")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

# 정규화 혼동행렬: 실제 규격별 정답 비율
row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0
)

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=class_names
).plot(
    ax=ax,
    cmap="Blues",
    values_format=".2f",
    xticks_rotation=45,
    colorbar=True
)

ax.set_title("Bolt Classification - Normalized Confusion Matrix")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "confusion_matrix_normalized.png",
    dpi=200
)
plt.close(fig)

print(f"\n혼동행렬 저장 위치: {OUTPUT_DIR}")
print("평가 완료!")