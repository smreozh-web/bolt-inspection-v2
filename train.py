from ultralytics import YOLO

# YOLO 모델 불러오기
model = YOLO("yolo11n.pt")

# 학습 시작
model.train(
    data="Fastener Object Detection.v1i.yolov11/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
