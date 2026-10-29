from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import torch

from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


# ==========================================
# 1. 기본 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

YOLO_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "bolt_detection_v2"
    / "weights"
    / "best.pt"
)

RESNET_PATH = (
    BASE_DIR
    / "runs"
    / "resnet18"
    / "v11_roi_finetune"
    / "best_resnet18_v11_roi.pth"
)

CAMERA_INDEX = 1

DET_CONF = 0.23
CLS_CONF = 0.60

SAVE_DIR = (
    BASE_DIR
    / "evaluation_results"
    / "camera_v13_diagnostic"
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 제공된 화면을 기준으로 한 초기 보정값.
# 카메라 거리/줌/해상도와 볼트 머리 형상을 유지해야 한다.
#
# 실제 규격 순서:
# M5x12, M5x8, M4x12, M4x8, M3x12, M3x8
#
# head_px: 머리 너비 대표값
# length_split_px: 8mm / 12mm 구분 기준
#
# M5x8은 기존 측정에 머리가 포함되어 있었으므로
# 경계 수정 후 길이 기준을 재확인해야 한다.

SIZE_CALIBRATION = {
    "M3": {
        "head_px": 49.25,
        "length_split_px": 123.0,
    },
    "M4": {
        "head_px": 59.5,
        "length_split_px": 140.0,
    },
    "M5": {
        "head_px": 72.15,
        "length_split_px": 119.0,
    },
}

HISTORY = 7
MATCH_DISTANCE = 65

TRACKS = {}
NEXT_TRACK_ID = 1


# ==========================================
# 2. 실행 장치
# ==========================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("사용 장치:", DEVICE)


# ==========================================
# 3. ResNet 이미지 전처리
# ==========================================

class ResizeWithPadding:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = image.convert("RGB")

        w, h = image.size

        scale = min(
            self.size / w,
            self.size / h,
        )

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        image = image.resize(
            (new_w, new_h),
            Image.Resampling.BILINEAR,
        )

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        image = ImageOps.expand(
            image,
            border=(
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            ),
            fill=(128, 128, 128),
        )

        return image


preprocess = transforms.Compose([
    ResizeWithPadding(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==========================================
# 4. YOLO 모델 로드
# ==========================================

if not YOLO_PATH.exists():
    raise FileNotFoundError(
        f"YOLO 모델 파일이 없습니다: {YOLO_PATH}"
    )

detector = YOLO(str(YOLO_PATH))

print("YOLO 모델 로드 완료")


# ==========================================
# 5. ResNet-18 모델 로드
# ==========================================

if not RESNET_PATH.exists():
    raise FileNotFoundError(
        f"ResNet 모델 파일이 없습니다: {RESNET_PATH}"
    )

checkpoint = torch.load(
    RESNET_PATH,
    map_location=DEVICE,
    weights_only=True,
)

class_names = checkpoint["class_names"]

classifier = models.resnet18(weights=None)

classifier.fc = nn.Linear(
    classifier.fc.in_features,
    len(class_names),
)

classifier.load_state_dict(
    checkpoint["model_state_dict"]
)

classifier = classifier.to(DEVICE)
classifier.eval()

print("ResNet-18 모델 로드 완료")
print("분류 클래스:", class_names)


# ==========================================
# 6. YOLO 검출 박스 ROI 추출
# ==========================================

def extract_fixed_roi(frame, x1, y1, x2, y2):
    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.empty(
            (0, 0, 3),
            dtype=np.uint8,
        )

    return frame[y1:y2, x1:x2].copy()


# ==========================================
# 7. ResNet-18 규격 분류
# ==========================================

@torch.inference_mode()
def classify_bolt(roi):
    rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(rgb)

    tensor = preprocess(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    output = classifier(tensor)

    probabilities = torch.softmax(
        output,
        dim=1,
    )[0]

    confidence, index = probabilities.max(dim=0)

    predicted_class = class_names[index.item()]
    confidence = confidence.item()

    return predicted_class, confidence


# ==========================================
# 8. OpenCV 볼트 치수 측정
# ==========================================

def measure_thread(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # 밝은 볼트와 어두운 배경 분리
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    # 금속 반사로 생긴 작은 구멍 보정
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # 작은 노이즈 제거
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if count <= 1:
        return None

    height, width = mask.shape
    center_x = width / 2

    candidates = []

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 30 or bh < 20:
            continue

        cx = centroids[i][0]

        # 중앙에 가깝고 세로로 긴 영역 우선
        score = (
            area
            + bh * 2
            - abs(cx - center_x) * 5
        )

        candidates.append((score, i))

    if not candidates:
        return None

    best_id = max(candidates)[1]

    bolt_mask = (
        np.uint8(labels == best_id) * 255
    )

    bolt_mask = cv2.morphologyEx(
        bolt_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        ),
    )

    # 각 행에서 볼트 외곽 너비 측정
    widths = np.zeros(
        height,
        dtype=np.float32,
    )

    for row in range(height):
        xs = np.where(
            bolt_mask[row] > 0
        )[0]

        if len(xs) >= 2:
            widths[row] = (
                xs[-1] - xs[0] + 1
            )

    smooth = cv2.GaussianBlur(
        widths.reshape(-1, 1),
        (1, 9),
        0,
    ).flatten()

    valid = np.where(smooth > 5)[0]

    if len(valid) < 20:
        return None

    top = int(valid[0])
    bottom = int(valid[-1])

    total_length = bottom - top + 1

    # 위쪽 머리 영역
    head_end = top + max(
        5,
        total_length // 4,
    )

    head_region = smooth[top:head_end]

    # 아래쪽 나사산 영역
    thread_start = top + max(
        5,
        total_length // 3,
    )

    thread_region = smooth[
        thread_start:bottom + 1
    ]

    head_region = head_region[
        head_region > 5
    ]

    thread_region = thread_region[
        thread_region > 5
    ]

    if (
        len(head_region) < 3
        or len(thread_region) < 5
    ):
        return None

    head_width = float(
        np.median(head_region)
    )

    thread_width = float(
        np.median(thread_region)
    )

    # 머리와 나사산 너비 차이가 부족하면 보류
    if head_width <= thread_width * 1.15:
        return None

    threshold_width = (
        head_width + thread_width
    ) / 2

    # 머리 위쪽의 좁은 부분을 경계로 오인하지 않도록
    # 넓은 머리 구간을 먼저 확인한다.
    boundary = None
    seen_head = False
    head_run = 0

    search_end = top + int(
        total_length * 0.65
    )

    for row in range(
        top,
        min(search_end, bottom - 8),
    ):
        if smooth[row] >= threshold_width:
            head_run += 1

            if head_run >= 5:
                seen_head = True
        else:
            head_run = 0

        section = smooth[row:row + 5]

        # 머리 확인 후 5행 연속 좁아지는 지점
        if seen_head and np.all(
            (section > 5)
            & (section < threshold_width)
        ):
            boundary = row
            break

    if boundary is None:
        return None

    thread_length = bottom - boundary + 1

    return {
        "start_y": y1 + boundary,
        "end_y": y1 + bottom,
        "center_x": (x1 + x2) // 2,
        "width_px": thread_width,
        "length_px": thread_length,
        "head_width_px": head_width,
    }


# ==========================================
# 9. OpenCV 규격 판별
# ==========================================

def classify_by_size(measurement):
    if measurement is None:
        return "UNKNOWN"

    h = float(
        measurement["head_width_px"]
    )
    d = float(
        measurement["width_px"]
    )
    l = float(
        measurement["length_px"]
    )

    if not all(
        np.isfinite(value)
        for value in (h, d, l)
    ):
        return "UNKNOWN"

    # 현재 촬영 조건에서의 허용 범위
    if not (
        35 <= h <= 85
        and 17 <= d <= 48
        and 60 <= l <= 220
    ):
        return "UNKNOWN"

    if h <= d * 1.15:
        return "UNKNOWN"

    # 머리 너비가 가장 가까운 규격 선택
    diameter = min(
        SIZE_CALIBRATION,
        key=lambda name: abs(
            h
            - SIZE_CALIBRATION[name]["head_px"]
        ),
    )

    calibration = SIZE_CALIBRATION[diameter]

    # 지름 규격별 길이 기준 적용
    if l < calibration["length_split_px"]:
        length = "8"
    else:
        length = "12"

    return f"{diameter}x{length}"


# ==========================================
# 10. 최근 프레임 결과 안정화
# ==========================================

def update_tracks(items, frame_no):
    global NEXT_TRACK_ID

    assigned = set()

    for item in items:
        cx, cy = item["center"]
        candidates = []

        for tid, track in TRACKS.items():
            if tid in assigned:
                continue

            if frame_no - track["last_seen"] > 10:
                continue

            tx, ty = track["center"]

            dist = (
                (cx - tx) ** 2
                + (cy - ty) ** 2
            ) ** 0.5

            if dist < MATCH_DISTANCE:
                candidates.append((dist, tid))

        if candidates:
            tid = min(candidates)[1]
        else:
            tid = NEXT_TRACK_ID
            NEXT_TRACK_ID += 1

            TRACKS[tid] = {
                "center": (cx, cy),
                "last_seen": frame_no,
                "history": deque(
                    maxlen=HISTORY
                ),
            }

        track = TRACKS[tid]

        track["center"] = (cx, cy)
        track["last_seen"] = frame_no

        track["history"].append(
            item["size"]
        )

        assigned.add(tid)

        votes = Counter(
            track["history"]
        )

        best, count = votes.most_common(1)[0]

        if (
            len(track["history"]) >= 5
            and count >= 4
        ):
            item["stable"] = best
        else:
            item["stable"] = "CHECKING"

    for tid in list(TRACKS):
        if frame_no - TRACKS[tid]["last_seen"] > 20:
            del TRACKS[tid]


# ==========================================
# 11. 카메라 실행
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "카메라 연결 실패. Camo를 확인하세요."
    )

print("카메라 실행 완료")
print("s: 결과 저장")
print("q: 프로그램 종료")

save_count = 0
frame_no = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("카메라 프레임 읽기 실패")
            break

        frame_no += 1

        original = frame.copy()
        display = frame.copy()

        frame_h, frame_w = frame.shape[:2]

        # YOLO 볼트 검출
        results = detector.predict(
            source=original,
            conf=DET_CONF,
            verbose=False,
        )

        boxes = results[0].boxes

        raw_count = (
            len(boxes)
            if boxes is not None
            else 0
        )

        items = []

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_w, x2)
                y2 = min(frame_h, y2)

                bw = x2 - x1
                bh = y2 - y1

                if bw <= 0 or bh <= 0:
                    continue

                # 기존 레일 촬영 필터
                if (
                    y1 > frame_h * 0.45
                    or bh < bw * 1.3
                    or bh < 60
                ):
                    continue

                roi = extract_fixed_roi(
                    original,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                if roi.size == 0:
                    continue

                measurement = measure_thread(
                    original,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                size_result = classify_by_size(
                    measurement
                )

                # SIZE가 UNKNOWN이어도 AI 분류 수행
                ai_label, ai_conf = classify_bolt(
                    roi
                )

                items.append({
                    "box": (x1, y1, x2, y2),
                    "center": (
                        (x1 + x2) // 2,
                        (y1 + y2) // 2,
                    ),
                    "measurement": measurement,
                    "size": size_result,
                    "ai": ai_label,
                    "ai_conf": ai_conf,
                    "roi": roi,
                })

        # 왼쪽부터 번호 부여
        items.sort(
            key=lambda item: item["center"][0]
        )

        update_tracks(items, frame_no)

        # 30프레임마다 측정값 출력
        if frame_no % 30 == 0:
            print(
                f"\n===== FRAME {frame_no} ====="
            )

            for idx, item in enumerate(items, 1):
                m = item["measurement"]

                if m is None:
                    print(
                        f"#{idx}: MEASURE FAILED"
                    )
                    continue

                print(
                    f"#{idx} | "
                    f"H={m['head_width_px']:.1f} | "
                    f"D={m['width_px']:.1f} | "
                    f"L={m['length_px']:.1f} | "
                    f"AI={item['ai']} | "
                    f"SIZE={item['size']}"
                )

        # ==========================================
        # 12. 검출 박스 및 측정선 표시
        # ==========================================

        for idx, item in enumerate(items, 1):
            x1, y1, x2, y2 = item["box"]

            measurement = item["measurement"]
            stable = item["stable"]

            if stable in ("UNKNOWN", "CHECKING"):
                color = (0, 165, 255)
            else:
                color = (0, 200, 0)

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                display,
                f"#{idx}",
                (
                    max(0, x1),
                    max(25, y1 - 10),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            if measurement is None:
                cv2.putText(
                    display,
                    "MEASURE FAILED",
                    (
                        max(0, x1),
                        min(
                            frame_h - 10,
                            y2 + 22,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    1,
                )
                continue

            mx = measurement["center_x"]
            sy = measurement["start_y"]
            ey = measurement["end_y"]

            half_width = max(
                1,
                int(
                    measurement["width_px"] / 2
                ),
            )

            # 노란색: 나사산 길이
            cv2.line(
                display,
                (mx, sy),
                (mx, ey),
                (0, 255, 255),
                2,
            )

            # 파란색: 나사산 시작점
            cv2.line(
                display,
                (mx - half_width, sy),
                (mx + half_width, sy),
                (255, 0, 0),
                2,
            )

            # 빨간색: 끝점
            cv2.line(
                display,
                (mx - half_width, ey),
                (mx + half_width, ey),
                (0, 0, 255),
                2,
            )

            # 청록색: 나사산 지름
            mid_y = (sy + ey) // 2

            cv2.line(
                display,
                (mx - half_width, mid_y),
                (mx + half_width, mid_y),
                (255, 255, 0),
                2,
            )

            h = measurement["head_width_px"]
            d = measurement["width_px"]
            l = measurement["length_px"]
            r = l / max(d, 1)

            detail = (
                f"H:{h:.1f} "
                f"D:{d:.1f} "
                f"L:{l:.0f}px "
                f"R:{r:.2f}"
            )

            cv2.putText(
                display,
                detail,
                (
                    max(0, x1),
                    min(
                        frame_h - 10,
                        y2 + 22,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        # ==========================================
        # 13. 검출 개수 표시
        # ==========================================

        size_ok_count = sum(
            item["size"] != "UNKNOWN"
            for item in items
        )

        cv2.putText(
            display,
            f"Bolt Count: {len(items)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            (
                f"YOLO:{raw_count} "
                f"FILTERED:{len(items)} "
                f"SIZE_OK:{size_ok_count}"
            ),
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # ==========================================
        # 14. 오른쪽 AI / SIZE 결과 패널
        # ==========================================

        PANEL_WIDTH = 310

        panel_height = max(
            frame_h,
            130 + len(items) * 95,
        )

        panel = np.zeros(
            (panel_height, PANEL_WIDTH, 3),
            dtype=np.uint8,
        )

        panel[:] = (35, 35, 35)

        cv2.putText(
            panel,
            "BOLT INSPECTION",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            panel,
            f"Detected: {len(items)}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

        for idx, item in enumerate(items, 1):
            start_y = 95 + (idx - 1) * 95

            ai_label = item["ai"]
            ai_conf = item["ai_conf"]
            size_label = item["stable"]

            if ai_conf >= CLS_CONF:
                ai_text = (
                    f"AI: {ai_label} "
                    f"({ai_conf * 100:.1f}%)"
                )
                ai_color = (100, 255, 100)
            else:
                ai_text = (
                    "AI: UNCERTAIN "
                    f"({ai_conf * 100:.1f}%)"
                )
                ai_color = (0, 165, 255)

            cv2.putText(
                panel,
                f"BOLT #{idx}",
                (15, start_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                panel,
                ai_text,
                (15, start_y + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                ai_color,
                1,
            )

            cv2.putText(
                panel,
                f"SIZE: {size_label}",
                (15, start_y + 51),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 220, 100),
                1,
            )

            cv2.line(
                panel,
                (15, start_y + 66),
                (
                    PANEL_WIDTH - 15,
                    start_y + 66,
                ),
                (90, 90, 90),
                1,
            )

        # 카메라 영상 높이를 패널과 맞춤
        if panel_height > frame_h:
            padded_display = cv2.copyMakeBorder(
                display,
                0,
                panel_height - frame_h,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(35, 35, 35),
            )
        else:
            padded_display = display

        final_display = np.hstack(
            (padded_display, panel)
        )

        cv2.imshow(
            "Bolt Inspection v13 - AI and SIZE",
            final_display,
        )

        # ==========================================
        # 15. 키보드 조작
        # ==========================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            save_count += 1

            frame_path = (
                SAVE_DIR
                / f"frame_{save_count:03d}.jpg"
            )

            # 결과 패널까지 포함해서 저장
            cv2.imwrite(
                str(frame_path),
                final_display,
            )

            # 볼트 ROI 개별 저장
            for i, item in enumerate(items):
                roi_path = (
                    SAVE_DIR
                    / (
                        f"roi_{save_count:03d}"
                        f"_{i:02d}.jpg"
                    )
                )

                cv2.imwrite(
                    str(roi_path),
                    item["roi"],
                )

            print(
                f"{save_count}번째 결과 저장 완료"
            )

finally:
    cap.release()
    cv2.destroyAllWindows()

print("프로그램 정상 종료")
