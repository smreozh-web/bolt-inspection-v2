from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = BASE / "screw_cls_v2"

# 데이터셋이 만들어졌는지 확인
if not (DATA / "train").exists():
    raise SystemExit("screw_cls_v2가 없어요. prepare_dataset.py부터 실행하세요.")

# 사전학습된 분류 모델
model = YOLO("yolo11n-cls.pt")

# 새 데이터셋으로 학습
model.train(
    data=str(DATA),
    epochs=80,
    imgsz=320,
    batch=16,
    patience=15,
    project=str(BASE / "runs" / "classify"),
    name="bolt_v2",
    exist_ok=False
)

# 테스트 데이터로 평가
best_model = YOLO(
    str(BASE / "runs" / "classify" / "bolt_v2" / "weights" / "best.pt")
)

metrics = best_model.val(
    data=str(DATA),
    split="test"
)

print("테스트 Top-1 정확도:", metrics.top1)
print("새 모델 학습 및 평가 완료")