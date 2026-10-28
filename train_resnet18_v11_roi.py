
from pathlib import Path
from collections import Counter, defaultdict
import copy
import json
import random
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

# ==========================================
# 1. 기본 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

OLD_DATA_DIR = BASE_DIR / "screw_dataset_v6_roi"
NEW_DATA_DIR = BASE_DIR / "bolt"

OLD_MODEL_PATH = (
    BASE_DIR / "runs" / "resnet18"
    / "v10_roi_finetune"
    / "best_resnet18_v10_roi.pth"
)

SAVE_DIR = (
    BASE_DIR / "runs" / "resnet18"
    / "v11_roi_finetune"
)

CLASSES = [
    "M3x8", "M3x12",
    "M4x8", "M4x12",
    "M5x8", "M5x12"
]

SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.00003
PATIENCE = 6

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

random.seed(SEED)
torch.manual_seed(SEED)

print("사용 장치:", DEVICE)

# ==========================================
# 2. 입력 경로 확인
# ==========================================

for path in [OLD_DATA_DIR, NEW_DATA_DIR, OLD_MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"필요한 파일 또는 폴더가 없습니다: {path}"
        )

for cls in CLASSES:
    for folder in [OLD_DATA_DIR, NEW_DATA_DIR]:
        if not (folder / cls).is_dir():
            raise FileNotFoundError(
                f"규격 폴더가 없습니다: {folder / cls}"
            )

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 3. 전처리
# ==========================================

class ResizeWithPadding:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = image.convert("RGB")
        w, h = image.size

        scale = self.size / max(w, h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        image = image.resize(
            (nw, nh),
            resample=Image.Resampling.BILINEAR
        )

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            (128, 128, 128)
        )

        canvas.paste(
            image,
            (
                (self.size - nw) // 2,
                (self.size - nh) // 2
            )
        )

        return canvas


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    ResizeWithPadding(224),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        )
    ], p=0.5),
    transforms.RandomRotation(
        degrees=4,
        interpolation=InterpolationMode.BILINEAR,
        fill=(128, 128, 128)
    ),
    transforms.ToTensor(),
    normalize
])

eval_transform = transforms.Compose([
    ResizeWithPadding(224),
    transforms.ToTensor(),
    normalize
])

# ==========================================
# 4. 사진 목록 읽기
# ==========================================

EXTENSIONS = {".jpg", ".jpeg", ".png"}

def get_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    )


def get_capture_group(path):
    """
    roi_018_03.jpg -> 촬영 회차 018

    동일 회차의 6개 볼트 사진이
    학습/검증/테스트에 나뉘어 들어가지
    않도록 하기 위한 그룹 식별자.
    """
    match = re.match(
        r"^roi_(\d+)_(\d+)$",
        path.stem,
        flags=re.IGNORECASE
    )

    if match:
        return f"capture_{match.group(1)}"

    return None


old_samples = []
new_groups = defaultdict(list)

for class_idx, cls in enumerate(CLASSES):
    for path in get_images(OLD_DATA_DIR / cls):
        old_samples.append((path, class_idx))

    for path in get_images(NEW_DATA_DIR / cls):
        group = get_capture_group(path)

        if group is None:
            raise ValueError(
                f"촬영 회차를 확인할 수 없는 파일: {path.name}\n"
                "roi_018_03.jpg 형식의 파일명인지 확인하세요."
            )

        new_groups[group].append((path, class_idx))

if not old_samples:
    raise RuntimeError("기존 ROI 데이터가 없습니다.")

if len(new_groups) < 5:
    raise RuntimeError(
        "새 카메라 ROI의 서로 다른 촬영 회차가 "
        "5개 미만입니다."
    )

# ==========================================
# 5. 촬영 회차 단위 데이터 분할
# ==========================================

group_names = sorted(new_groups.keys())
random.shuffle(group_names)

n_groups = len(group_names)

test_count = max(2, round(n_groups * 0.20))
val_count = max(1, round(n_groups * 0.15))

if test_count + val_count >= n_groups:
    raise RuntimeError(
        "학습에 사용할 촬영 회차가 부족합니다."
    )

test_group_names = set(group_names[:test_count])

val_group_names = set(
    group_names[test_count:test_count + val_count]
)

train_group_names = set(
    group_names[test_count + val_count:]
)

train_samples = list(old_samples)
val_samples = []
test_samples = []

for group, samples in new_groups.items():
    if group in test_group_names:
        test_samples.extend(samples)
    elif group in val_group_names:
        val_samples.extend(samples)
    else:
        train_samples.extend(samples)

if not val_samples or not test_samples:
    raise RuntimeError(
        "검증 또는 테스트 데이터가 비어 있습니다."
    )

print("\n========== 데이터 분할 ==========")
print("기존 ROI:", len(old_samples))
print("새 ROI:", sum(len(v) for v in new_groups.values()))
print("새 촬영 회차:", n_groups)
print("학습 이미지:", len(train_samples))
print("검증 이미지:", len(val_samples))
print("테스트 이미지:", len(test_samples))
print("테스트 회차:", sorted(test_group_names))

# ==========================================
# 6. Dataset / DataLoader
# ==========================================

class BoltDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            image = img.convert("RGB")

        return self.transform(image), label


train_dataset = BoltDataset(
    train_samples, train_transform
)

val_dataset = BoltDataset(
    val_samples, eval_transform
)

test_dataset = BoltDataset(
    test_samples, eval_transform
)

# 클래스별 데이터 수 차이를 보정
class_counts = Counter(
    label for _, label in train_samples
)

sample_weights = [
    1.0 / class_counts[label]
    for _, label in train_samples
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# 7. 기존 v10 모델 로드
# ==========================================

checkpoint = torch.load(
    OLD_MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

class_names = checkpoint["class_names"]

if list(class_names) != CLASSES:
    raise ValueError(
        f"클래스 순서 불일치!\n"
        f"체크포인트: {class_names}\n"
        f"현재 설정: {CLASSES}"
    )


def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASSES)
    )
    return model


model = create_model()
model.load_state_dict(
    checkpoint["model_state_dict"]
)
model = model.to(DEVICE)

# v10 모델 성능 비교용 원본
baseline = create_model()
baseline.load_state_dict(
    checkpoint["model_state_dict"]
)
baseline = baseline.to(DEVICE)
baseline.eval()

print("\n기존 ResNet v10 모델 로드 완료")

# ==========================================
# 8. 평가 함수
# ==========================================

def evaluate(model, loader):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total += batch_size
            correct += (
                predictions == labels
            ).sum().item()

            total_loss += (
                loss.item() * batch_size
            )

    accuracy = correct / total if total else 0.0
    avg_loss = total_loss / total if total else 0.0

    return accuracy, avg_loss


v10_test_acc, _ = evaluate(
    baseline, test_loader
)

print(
    f"학습 전 v10 테스트 정확도: "
    f"{v10_test_acc * 100:.1f}%"
)

# ==========================================
# 9. 추가 학습
# ==========================================

# 기존 특징 추출층은 유지하고,
# 마지막 분류층과 layer4만 추가 학습
for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.fc.parameters():
    parameter.requires_grad = True

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
best_state = None
best_epoch = 0
wait = 0
history = []

print("\n========== v11 학습 시작 ==========")

for epoch in range(EPOCHS):
    model.train()

    # 고정된 층의 BatchNorm 통계는 유지
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    train_total = 0
    train_correct = 0
    train_loss_sum = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        train_total += batch_size
        train_correct += (
            predictions == labels
        ).sum().item()

        train_loss_sum += (
            loss.item() * batch_size
        )

    train_acc = train_correct / train_total
    train_loss = train_loss_sum / train_total

    val_acc, val_loss = evaluate(
        model, val_loader
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Acc: {train_acc * 100:.1f}% | "
        f"Val Acc: {val_acc * 100:.1f}% | "
        f"Val Loss: {val_loss:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_accuracy": train_acc,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "val_loss": val_loss
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_state = copy.deepcopy(
            model.state_dict()
        )
        wait = 0

        print("  -> 최고 모델 갱신")
    else:
        wait += 1

    if wait >= PATIENCE:
        print("Early Stopping")
        break

# ==========================================
# 10. 최고 모델 저장
# ==========================================

if best_state is None:
    raise RuntimeError(
        "저장할 모델이 없습니다."
    )

model.load_state_dict(best_state)

model_path = (
    SAVE_DIR / "best_resnet18_v11_roi.pth"
)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": CLASSES,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss
}, model_path)

# ==========================================
# 11. v10 vs v11 테스트
# ==========================================

v11_test_acc, _ = evaluate(
    model, test_loader
)

print("\n========== 최종 비교 ==========")
print(
    f"v10 테스트 정확도: "
    f"{v10_test_acc * 100:.1f}%"
)
print(
    f"v11 테스트 정확도: "
    f"{v11_test_acc * 100:.1f}%"
)
print("최고 모델 Epoch:", best_epoch)
print("v11 저장 위치:", model_path)

if v11_test_acc > v10_test_acc:
    print("테스트 정확도가 개선되었습니다.")
elif v11_test_acc == v10_test_acc:
    print("테스트 정확도가 동일합니다.")
else:
    print("테스트 정확도가 낮아졌습니다.")

# 사용된 이미지 목록 기록
def serialize_samples(samples):
    return [
        {
            "path": str(path),
            "class": CLASSES[label]
        }
        for path, label in samples
    ]


report = {
    "v10_test_accuracy": v10_test_acc,
    "v11_test_accuracy": v11_test_acc,
    "best_epoch": best_epoch,
    "train_samples": serialize_samples(train_samples),
    "val_samples": serialize_samples(val_samples),
    "test_samples": serialize_samples(test_samples),
    "history": history
}

with open(
    SAVE_DIR / "training_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n기존 v10 모델은 변경되지 않았습니다.")
print("YOLO 및 OpenCV 코드는 변경되지 않았습니다.")
