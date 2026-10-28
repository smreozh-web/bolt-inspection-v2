from pathlib import Path
from ultralytics import YOLO

# 프로젝트 경로
PROJECT_DIR = Path(__file__).resolve().parent

# 새로 분할한 v2 데이터셋
DATA_YAML = PROJECT_DIR / "bolt_detection_split_v2" / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"데이터셋 파일이 없습니다: {DATA_YAML}")

print("학습 데이터셋:", DATA_YAML)

# YOLO11n 모델 불러오기
model = YOLO("yolo11n.pt")

# YOLO v2 학습
model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    workers=0,
    project=str(PROJECT_DIR / "runs" / "detect"),
    name="bolt_detection_v2",
    exist_ok=False,
    patience=15,
    plots=True
)

print("YOLO v2 학습 완료!")