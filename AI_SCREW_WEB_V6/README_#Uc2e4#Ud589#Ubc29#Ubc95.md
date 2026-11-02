# AI SCREW — 동일 Wi-Fi 접속 웹사이트

## 1. 설치 위치
`AI_SCREW_LAN_WEB` 폴더를 기존 `bolt-inspection-main` 프로젝트 폴더 **안**에 넣습니다.

예시:
```
bolt-inspection-main/
  runs/
  train.py
  detect_bolt_resnet18_v11.py
  AI_SCREW_LAN_WEB/
    app.py
    templates/index.html
    requirements.txt
```

## 2. PyCharm 실행
1. 기존 프로젝트를 PyCharm에서 엽니다.
2. 기존 Python 인터프리터 `C:\ai_screw_env\Scripts\python.exe`를 선택합니다.
3. Flask가 없다면 PyCharm 하단 Terminal에서 `C:\ai_screw_env\Scripts\python.exe -m pip install flask`를 한 번 실행합니다. 나머지 패키지는 기존 학습 환경을 활용합니다.
4. `AI_SCREW_LAN_WEB/app.py`를 오른쪽 클릭 → **Run 'app'**.
5. 서버 PC에서 `http://127.0.0.1:5000` 접속.

## 3. 스마트폰·다른 노트북 접속
1. 서버 PC와 접속 기기를 **같은 Wi-Fi**에 연결합니다.
2. Windows 시작 메뉴에서 `cmd` 실행 → `ipconfig` 입력 → Wi-Fi 어댑터의 `IPv4 주소` 확인.
3. 다른 기기의 브라우저에 `http://PC의_IPv4_주소:5000` 입력. 예: `http://192.168.0.10:5000` (예시일 뿐 실제 주소 사용).
4. Windows 방화벽 경고가 나오면 **개인 네트워크**에 대해서만 Python 접근을 허용합니다. 학교 공용 Wi-Fi는 장치 간 접속이 차단될 수 있습니다.

## 4. 카메라
- Windows PC에서 Camo Studio를 켜고 스마트폰 카메라를 연결합니다.
- 기본 카메라 번호는 `1`입니다. 연결되지 않으면 `app_v1.py` 맨 위 `CAMERA_INDEX` 환경변수 기본값을 `0` 또는 `2`로 바꿔 확인합니다.
- **카메라는 서버 PC에서만 열고**, 다른 기기는 처리된 영상을 수신합니다.

## 5. AI 모델 경로
기존 프로젝트 기준으로 다음 파일을 자동 탐색합니다.
- 우선 YOLO: `runs/detect/bolt_detection_v2/weights/best.pt`
- 대체 YOLO: `runs/detect/train-2/weights/best.pt` (7종 형태 탐지 모델이므로 실제 단일 bolt 모델과 정확도가 다릅니다)
- ResNet18: `runs/resnet18/v8_weighted/best_resnet18_v8.pth`

모델이 없으면 카메라 화면부터 테스트할 수 있습니다. 두 모델이 준비되면 탐지 박스와 6종 분류가 나타납니다.

## 6. 유의사항
- Windows PC가 켜져 있고 `app_v1.py`가 실행 중이어야 접속 가능합니다.
- CPU에서는 처리 속도가 느릴 수 있습니다.
- 신뢰도는 분류 모델의 출력값이며, 정확도 보증이 아닙니다.
- 같은 Wi-Fi 접속용 HTTP 개발 서버입니다. 인터넷에 직접 포트를 개방하지 마세요. 같은 네트워크의 다른 사람도 영상에 접속할 수 있으므로 신뢰할 수 있는 개인 네트워크에서만 사용하세요.
- 카메라 화면이 안 보이면 Camo 가상 카메라가 다른 앱에서 점유 중인지 확인하세요.
