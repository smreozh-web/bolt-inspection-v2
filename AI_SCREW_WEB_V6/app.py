
from pathlib import Path
from datetime import datetime
import os
import time
import threading
import hmac
import secrets
from collections import defaultdict, deque
from urllib.parse import urlsplit

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, session, redirect, url_for

# ==========================================
# 기본 설정
# ==========================================

ROOT = Path(__file__).resolve().parent
PROJECT = Path(
    os.environ.get("SCREW_PROJECT", str(ROOT.parent))
)

# -1: 카메라 0~3 자동 탐색
# Camo Studio 번호를 알면 0, 1, 2 등으로 고정 가능
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "-1"))

YOLO_FILE = (
    PROJECT
    / "runs/detect/bolt_detection_v2/weights/best.pt"
)

ALT_YOLO_FILE = (
    PROJECT
    / "runs/detect/train-2/weights/best.pt"
)

RESNET_FILE = (
    PROJECT
    / "runs"
    / "resnet18"
    / "v11_roi_finetune"
    / "best_resnet18_v11_roi.pth"
)

app = Flask(__name__)

# 로그인 전용 화면 + 세션 기반 접근 제어
AUTH_USERNAME = "admin"
AUTH_PASSWORD = os.environ.get("SCREW_WEB_PASSWORD", "")
if not AUTH_PASSWORD:
    raise RuntimeError("SCREW_WEB_PASSWORD가 없습니다. run_secure.py로 실행하세요.")

app.secret_key = secrets.token_hex(32)  # 서버 재시작 시 기존 로그인 세션 만료
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=__import__("datetime").timedelta(hours=8),
)
login_attempts = defaultdict(deque)
login_lock = threading.Lock()
model_inference_lock = threading.Lock()


def valid_csrf():
    supplied = request.headers.get("X-CSRF-Token", "") or request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    return bool(supplied and expected and hmac.compare_digest(supplied, expected))


@app.before_request
def require_authentication():
    if request.endpoint in ("login", "static"):
        return None
    if not session.get("authenticated"):
        if request.path.startswith("/api/") or request.path == "/video_feed":
            return jsonify({"error": "로그인이 필요합니다."}), 401
        return redirect(url_for("login"))
    if request.method not in ("GET", "HEAD", "OPTIONS") and not valid_csrf():
        return jsonify({"error": "보안 토큰이 올바르지 않습니다. 페이지를 새로고침하세요."}), 403


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if session.get("authenticated"):
            return redirect(url_for("index"))
        session.setdefault("login_csrf", secrets.token_urlsafe(32))
        return render_template("login.html", error=None)

    if not hmac.compare_digest(request.form.get("csrf_token", ""), session.get("login_csrf", "")):
        return render_template("login.html", error="페이지를 새로고침하고 다시 시도해주세요."), 400

    # IP별 로그인 시도 제한 (최근 5분 동안 최대 8회)
    client = request.remote_addr or "unknown"
    now = time.monotonic()
    with login_lock:
        attempts = login_attempts[client]
        while attempts and now - attempts[0] > 300:
            attempts.popleft()
        limited = len(attempts) >= 8
        if not limited:
            attempts.append(now)
    if limited:
        return render_template("login.html", error="로그인 시도가 많습니다. 5분 뒤 다시 시도해주세요."), 429

    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if not (hmac.compare_digest(username, AUTH_USERNAME) and hmac.compare_digest(password, AUTH_PASSWORD)):
        time.sleep(0.35)
        return render_template("login.html", error="아이디 또는 비밀번호가 올바르지 않습니다."), 401

    with login_lock:
        login_attempts.pop(client, None)
    session.clear()
    session["authenticated"] = True
    session["csrf_token"] = secrets.token_urlsafe(32)
    session.permanent = True
    return redirect(url_for("index"))


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.after_request
def disable_sensitive_cache(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    # 로컬 http 접속도 허용해야 하므로 HTTPS 요청에서만 Secure 쿠키 설정
    # Cloudflare Tunnel은 원본 서버에 HTTP로 접속하므로 Host 기준으로 판별
    if request.host.split(":")[0].endswith(".trycloudflare.com"):
        cookie = response.headers.get("Set-Cookie")
        if cookie and "Secure" not in cookie:
            response.headers["Set-Cookie"] = cookie + "; Secure"
    return response


SUPPORTED_SPECS = ["M3x8", "M3x12", "M4x8", "M4x12", "M5x8", "M5x12"]
TARGET_SPEC = "M3x8"
CONFIDENCE_THRESHOLD = 0.80

def normalized_spec(name):
    return str(name).strip().lower().replace("×", "x").replace("*", "x").replace(" ", "").replace("_", "")

def judge(spec, confidence, target, threshold):
    if confidence is None or spec == "나사":
        return "확인 필요"
    if confidence < threshold:
        return "확인 필요"
    return "정상투입" if normalized_spec(spec) == normalized_spec(target) else "오투입"


state = {
    "camera": "연결 대기",
    "target_spec": TARGET_SPEC,
    "threshold": CONFIDENCE_THRESHOLD,
    "overall": "미검출",
    "target_probability": None,
    "model": "확인 중",
    "count": 0,
    "detections": [],
    "updated": None,
    "error": None
}

lock = threading.Lock()

latest_jpg = None
camera = None
yolo = None
resnet = None
classes = []
transform = None


# ==========================================
# AI 모델 불러오기
# ==========================================

def load_models():
    global yolo, resnet, classes, transform

    from ultralytics import YOLO
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image, ImageOps

    yolo_path = (
        YOLO_FILE if YOLO_FILE.exists()
        else ALT_YOLO_FILE
    )

    if yolo_path.exists():
        try:
            yolo = YOLO(str(yolo_path))
            print(f"[YOLO] 로드 성공: {yolo_path}", flush=True)
        except Exception as e:
            print(f"[YOLO] 로드 실패: {e}", flush=True)
    else:
        print("[YOLO] 모델 파일 없음", flush=True)

    if RESNET_FILE.exists():
        try:
            ck = torch.load(
                RESNET_FILE,
                map_location="cpu",
                weights_only=True
            )

            classes = ck["class_names"]

            resnet = models.resnet18(weights=None)
            resnet.fc = nn.Linear(
                resnet.fc.in_features,
                len(classes)
            )

            resnet.load_state_dict(
                ck["model_state_dict"]
            )
            resnet.eval()

            class ResizeWithPadding:
                def __call__(self, image):
                    image = image.convert("RGB")
                    w, h = image.size

                    scale = min(224 / w, 224 / h)

                    nw = max(1, round(w * scale))
                    nh = max(1, round(h * scale))

                    image = image.resize(
                        (nw, nh),
                        Image.Resampling.BILINEAR
                    )

                    dw = 224 - nw
                    dh = 224 - nh

                    return ImageOps.expand(
                        image,
                        (
                            dw // 2,
                            dh // 2,
                            dw - dw // 2,
                            dh - dh // 2
                        ),
                        fill=(128, 128, 128)
                    )

            transform = transforms.Compose([
                ResizeWithPadding(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [.485, .456, .406],
                    [.229, .224, .225]
                )
            ])

            print("[ResNet18] 로드 성공", flush=True)

        except Exception as e:
            resnet = None
            print(f"[ResNet18] 로드 실패: {e}", flush=True)

    else:
        print("[ResNet18] 모델 파일 없음", flush=True)

    with lock:
        state["model"] = (
            ("YOLO 준비" if yolo is not None else "YOLO 미준비")
            + " · "
            + ("ResNet18 준비" if resnet is not None
               else "ResNet18 미준비")
        )


# ==========================================
# Windows DirectShow 카메라 이름 기반 선택
# ==========================================

selected_camera_name = "Camo Camera"
selected_camera_index = None
camera_selection_version = 0


def camera_devices():
    # Flask 요청 스레드와 camera_loop 스레드는 각각 COM 초기화가 필요합니다.
    # pygrabber가 사용하는 comtypes에서 이 스레드의 COM을 초기화하고 정리합니다.
    import comtypes

    # COM을 먼저 초기화한 뒤 pygrabber를 import해야 합니다.
    # pygrabber의 import 과정에서도 COM 장치를 사용할 수 있습니다.
    comtypes.CoInitialize()
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    finally:
        comtypes.CoUninitialize()


def open_camera():
    with lock:
        wanted_name = selected_camera_name
        wanted_index = selected_camera_index

    try:
        devices = camera_devices()
    except Exception as exc:
        print(f"[카메라] 장치 목록 조회 실패: {exc}", flush=True)
        return None, None

    # pygrabber 장치 목록의 순서와 OpenCV VideoCapture 인덱스는
    # Windows 환경에서 서로 다를 수 있습니다.
    # 이 PC에서는 별도 DirectShow 테스트로 0번이 Camo임을 확인했습니다.
    if wanted_name == "Camo Camera":
        camo_devices = [(i, n) for i, n in enumerate(devices) if "camo" in n.lower()]
        if not camo_devices:
            print("[카메라] Camo Camera 장치 이름을 찾지 못했습니다.", flush=True)
            return None, None
        matches = [(0, "Camo Camera (OpenCV 검증된 0번)")]
        print(f"[카메라] Camo 목록 위치: {[i for i, _ in camo_devices]}; "
              "OpenCV MSMF 0번으로 연결합니다.", flush=True)
    else:
        matches = [(i, n) for i, n in enumerate(devices)
                   if i == wanted_index and n == wanted_name]

    if not matches:
        print(f"[카메라] 선택한 장치를 찾지 못했습니다: {wanted_name}", flush=True)
        return None, None

    for index, name in matches:
        # Camo는 DirectShow에서 검은 프레임을 반환하므로 MSMF 사용.
        # 일반 웹캠은 기존 DirectShow 설정을 그대로 유지.
        backend = cv2.CAP_MSMF if index == 0 else cv2.CAP_DSHOW
        backend_name = "MSMF" if backend == cv2.CAP_MSMF else "DirectShow"
        print(f"[카메라] {backend_name} 연결 시도: {index} / {name}", flush=True)
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"[카메라] 연결 성공: {index} / {name}", flush=True)
                return cap, frame
            time.sleep(0.15)
        cap.release()
    return None, None


@app.get("/api/cameras")
def list_cameras():
    try:
        devices = camera_devices()
    except Exception as exc:
        return jsonify({"error": "카메라 목록을 읽지 못했습니다. pygrabber 설치를 확인하세요: " + str(exc)}), 503
    result = []
    camo_added = False
    for i, name in enumerate(devices):
        if "camo" in name.lower():
            if not camo_added:
                result.append({"id": "camo", "label": "Camo Camera"})
                camo_added = True
        else:
            result.append({"id": str(i), "label": name})
    with lock:
        selected = "camo" if selected_camera_name == "Camo Camera" else str(selected_camera_index)
    return jsonify({"devices": result, "selected": selected})


@app.post("/api/cameras")
def select_camera():
    global selected_camera_name, selected_camera_index, camera_selection_version, latest_jpg
    payload = request.get_json(silent=True) or {}
    requested = str(payload.get("id", ""))
    try:
        devices = camera_devices()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503
    if requested == "camo":
        if not any("camo" in name.lower() for name in devices):
            return jsonify({"error": "Camo Camera가 감지되지 않습니다."}), 404
        name, index = "Camo Camera", None
    else:
        if not requested.isdecimal() or int(requested) >= len(devices):
            return jsonify({"error": "선택한 카메라가 없습니다."}), 400
        index = int(requested)
        name = devices[index]
        if "camo" in name.lower():
            return jsonify({"error": "Camo Camera 항목을 선택해주세요."}), 400
    with lock:
        selected_camera_name = name
        selected_camera_index = index
        camera_selection_version += 1
        latest_jpg = None
        state.update(camera="카메라 전환 중", count=0, detections=[], overall="미검출",
                     target_probability=None, updated=None, error=None)
    return jsonify({"message": "카메라 전환을 시작했습니다.", "selected": requested})


# ==========================================
# 실시간 카메라 및 AI 검사
# ==========================================

def camera_loop():
    global latest_jpg, camera

    active_selection_version = -1

    try:
        load_models()
    except Exception as e:
        with lock:
            state["model"] = "모델 로드 오류"
            state["error"] = str(e)

        print(f"[AI] 모델 로드 오류: {e}", flush=True)

    while True:
        with lock:
            wanted_version = camera_selection_version
        if wanted_version != active_selection_version:
            if camera is not None:
                camera.release()
                camera = None
            active_selection_version = wanted_version

        # 카메라가 없으면 다시 탐색
        if camera is None or not camera.isOpened():

            with lock:
                state["camera"] = "카메라 탐색 중"
                state["count"] = 0
                state["detections"] = []
                state["overall"] = "미검출"
                state["target_probability"] = None
                latest_jpg = None

            camera, frame = open_camera()
            with lock:
                changed_during_open = active_selection_version != camera_selection_version
            if changed_during_open:
                if camera is not None:
                    camera.release()
                    camera = None
                continue

            if camera is None:

                with lock:
                    state["camera"] = "연결 안 됨"
                    state["error"] = (
                        "카메라 영상을 받을 수 없습니다. "
                        "선택한 카메라와 Camo Studio 상태를 확인하세요."
                    )

                print(
                    "[카메라] 연결 실패. 3초 후 재시도",
                    flush=True
                )

                time.sleep(3)
                continue

        else:

            ok, frame = camera.read()

            if (
                not ok
                or frame is None
                or frame.size == 0
            ):

                with lock:
                    state["camera"] = "영상 수신 실패"
                    state["count"] = 0
                    state["detections"] = []
                    latest_jpg = None

                print(
                    "[카메라] 영상 수신 실패. 다시 연결합니다.",
                    flush=True
                )

                camera.release()
                camera = None

                time.sleep(0.5)
                continue

        # 실제 프레임 수신 성공
        with lock:
            state["camera"] = "연결됨 (영상 수신 중)"

            previous_error = state.get("error") or ""

            if (
                "카메라" in previous_error
                or "영상 수신" in previous_error
            ):
                state["error"] = None

        # AI 추론이 오래 걸려도 우선 카메라 원본을 브라우저로 전송합니다.


        with lock:
            target_spec = state["target_spec"]
            threshold = state["threshold"]
        detections = []

        # YOLO 탐지 및 ResNet18 분류
        if yolo is not None:

            try:
                import torch
                from PIL import Image

                with model_inference_lock:
                    result = yolo.predict(
                        frame, conf=0.35, imgsz=640, verbose=False, device="cpu"
                    )[0]

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    label = "나사"
                    confidence = None
                    target_probability = None

                    # 이미지 범위에 맞게 좌표 보정
                    h, w = frame.shape[:2]

                    x1 = max(0, min(x1, w))
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h))
                    y2 = max(0, min(y2, h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # 탐지된 나사 영역 분류
                    if (
                        resnet is not None
                        and transform is not None
                    ):

                        crop = frame[y1:y2, x1:x2]

                        if crop.size > 0:

                            rgb = cv2.cvtColor(
                                crop,
                                cv2.COLOR_BGR2RGB
                            )

                            image = Image.fromarray(rgb)

                            inp = transform(image).unsqueeze(0)

                            with model_inference_lock, torch.inference_mode():

                                output = resnet(inp)

                                probs = torch.softmax(
                                    output,
                                    dim=1
                                )[0]

                                score, idx = probs.max(0)

                                label = str(
                                    classes[int(idx)]
                                )

                                confidence = round(float(score), 4)
                                # 선택한 규격에 해당하는 softmax 출력값
                                for class_index, class_name in enumerate(classes):
                                    if normalized_spec(class_name) == normalized_spec(target_spec):
                                        target_probability = round(float(probs[class_index]), 4)
                                        break

                    verdict = judge(label, confidence, target_spec, threshold)
                    # 탐지 박스 그리기


                    detections.append({
                        "spec": label,
                        "confidence": confidence,
                        "target_probability": target_probability,
                        "verdict": verdict,
                        "box": [x1, y1, x2, y2]
                    })

            except Exception as e:

                with lock:
                    state["error"] = "AI 추론 오류: " + str(e)

                print(
                    f"[AI] 추론 오류: {e}",
                    flush=True
                )

        # 볼트 중심 위치를 기준으로 왼쪽부터 정렬
        detections.sort(
            key=lambda d: (d["box"][0] + d["box"][2]) / 2
        )

        # 정렬된 순서대로 박스와 번호 표시
        for number, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["box"]
            detection["number"] = number

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 화면 위쪽에서도 번호가 잘리지 않도록 위치 보정
            text_y = y1 - 10 if y1 >= 35 else y1 + 28
            text_y = min(text_y, frame.shape[0] - 5)

            cv2.putText(
                frame,
                f"#{number}",
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 0),
                5
            )

            cv2.putText(
                frame,
                f"#{number}",
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2
            )

        # 웹 브라우저에 보낼 JPEG 생성
        ok, jpg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if ok:

            with lock:
                if (active_selection_version != camera_selection_version or
                        target_spec != state["target_spec"] or threshold != state["threshold"]):
                    continue
                latest_jpg = jpg.tobytes()

                state["count"] = len(detections)
                state["detections"] = detections
                state["target_probability"] = (
                    detections[0].get("target_probability") if len(detections) == 1 else None
                )
                if not detections:
                    state["overall"] = "미검출"
                elif any(d["verdict"] == "오투입" for d in detections):
                    state["overall"] = "오투입"
                elif any(d["verdict"] == "확인 필요" for d in detections):
                    state["overall"] = "확인 필요"
                else:
                    state["overall"] = "정상투입"
                state["updated"] = (
                    datetime.now().strftime("%H:%M:%S")
                )

        time.sleep(0.06)


# ==========================================
# 접속한 브라우저의 카메라 이미지 검사 (Camo 검사와 독립)
# ==========================================

@app.post("/api/inspect_device")
def inspect_device():
    # 로그인 및 CSRF 검사는 기존 before_request에서 처리합니다.
    if request.content_length is not None and request.content_length > 3_000_000:
        return jsonify({"error": "이미지 용량이 너무 큽니다."}), 413
    data = request.get_data(cache=False)
    if not data or len(data) > 3_000_000:
        return jsonify({"error": "이미지 크기가 올바르지 않습니다."}), 400
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "이미지를 읽을 수 없습니다."}), 400
    height, width = frame.shape[:2]
    if width > 1920 or height > 1920:
        scale = min(1920 / width, 1920 / height)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    if yolo is None:
        return jsonify({"error": "YOLO 모델이 준비되지 않았습니다."}), 503

    import torch
    from PIL import Image
    # 브라우저별 검사 설정: 서버 Camo 설정과 다른 사용자 설정을 변경하지 않습니다.
    target_spec = session.get("device_target_spec", TARGET_SPEC)
    threshold = session.get("device_threshold", CONFIDENCE_THRESHOLD)
    detections = []
    try:
        with model_inference_lock:
            result = yolo.predict(frame, conf=0.35, imgsz=640, verbose=False, device="cpu")[0]
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                h, w = frame.shape[:2]
                x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue
                label = "나사"
                confidence = None
                target_probability = None
                if resnet is not None and transform is not None:
                    crop = frame[y1:y2, x1:x2]
                    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    inp = transform(image).unsqueeze(0)
                    with torch.inference_mode():
                        probs = torch.softmax(resnet(inp), dim=1)[0]
                    score, idx = probs.max(0)
                    label = str(classes[int(idx)])
                    confidence = round(float(score), 4)
                    for i, name in enumerate(classes):
                        if normalized_spec(name) == normalized_spec(target_spec):
                            target_probability = round(float(probs[i]), 4)
                            break
                detections.append({
                    "spec": label, "confidence": confidence,
                    "target_probability": target_probability,
                    "verdict": judge(label, confidence, target_spec, threshold),
                    "box": [x1, y1, x2, y2],
                })
    except Exception as exc:
        app.logger.exception("브라우저 카메라 검사 실패")
        return jsonify({"error": "AI 검사 중 오류가 발생했습니다."}), 500

    if not detections:
        overall = "미검출"
    elif any(d["verdict"] == "오투입" for d in detections):
        overall = "오투입"
    elif any(d["verdict"] == "확인 필요" for d in detections):
        overall = "확인 필요"
    else:
        overall = "정상투입"
    return jsonify({
        "count": len(detections), "detections": detections,
        "overall": overall,
        "target_probability": detections[0]["target_probability"] if len(detections) == 1 else None,
        "updated": datetime.now().strftime("%H:%M:%S"),
        "camera": "접속 기기 카메라", "model": state.get("model", ""),
        "target_spec": target_spec, "threshold": threshold,
        "error": None,
    })


@app.route("/api/device_settings", methods=["GET", "POST"])
def device_settings():
    """각 로그인 브라우저 세션의 검사 설정을 독립적으로 관리합니다."""
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        target = str(payload.get("target_spec", ""))
        try:
            threshold = float(payload.get("threshold"))
        except (TypeError, ValueError):
            return jsonify({"error": "확률 기준이 올바르지 않습니다."}), 400
        if target not in SUPPORTED_SPECS or not 0.5 <= threshold <= 0.99:
            return jsonify({"error": "규격 또는 확률 기준이 올바르지 않습니다."}), 400
        session["device_target_spec"] = target
        session["device_threshold"] = threshold
    return jsonify({
        "target_spec": session.get("device_target_spec", TARGET_SPEC),
        "threshold": session.get("device_threshold", CONFIDENCE_THRESHOLD),
    })


# ==========================================
# 웹 페이지
# ==========================================

@app.get("/")
def index():
    return render_template("index.html", csrf_token=session.get("csrf_token", ""))


# ==========================================
# 상태 정보 API
# ==========================================

@app.get("/api/status")
def status():

    with lock:
        return jsonify(dict(state))


# ==========================================
# 실시간 영상 스트리밍
# ==========================================

@app.post("/api/settings")
def settings():
    payload = request.get_json(silent=True) or {}
    target = str(payload.get("target_spec", ""))
    try:
        threshold = float(payload.get("threshold"))
    except (TypeError, ValueError):
        return jsonify({"error": "확률 기준이 올바르지 않습니다."}), 400
    if target not in SUPPORTED_SPECS or not 0.5 <= threshold <= 0.99:
        return jsonify({"error": "규격 또는 확률 기준이 올바르지 않습니다."}), 400
    with lock:
        state["target_spec"] = target
        state["threshold"] = threshold
        state["overall"] = "판정 갱신 중"
        state["target_probability"] = None
        state["detections"] = []
        state["count"] = 0
        return jsonify({"target_spec": target, "threshold": threshold})


@app.get("/video_feed")
def video_feed():

    def frames():

        while True:

            with lock:
                jpg = latest_jpg

            if jpg is not None:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )

            time.sleep(0.12)

    return Response(
        frames(),
        mimetype=(
            "multipart/x-mixed-replace; boundary=frame"
        ),
        headers={
            "Cache-Control": "no-store"
        }
    )


# ==========================================
# 서버 실행
# ==========================================

if __name__ == "__main__":

    threading.Thread(
        target=camera_loop,
        daemon=True
    ).start()

    print("======================================")
    print(" AI Screw Inspection Web Server")
    print("======================================")
    print("로컬 접속: http://127.0.0.1:5001")
    print("외부 접속: 인증 설정 후 Cloudflare Tunnel 사용")
    print("======================================")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        threaded=True
    )
