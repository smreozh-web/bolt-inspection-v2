
from pathlib import Path
from collections import Counter, defaultdict
import copy
import json
import random
import re

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

# ==========================================
# 1. 기본 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "screw_dataset_v6_roi"

OLD_MODEL_PATH = (
    BASE_DIR / "runs" / "resnet18"
    / "v8_weighted" / "best_resnet18_v8.pth"
)

SAVE_DIR = (
    BASE_DIR / "runs" / "resnet18"
    / "v10_roi_finetune"
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "M3x8", "M3x12",
    "M4x8", "M4x12",
    "M5x8", "M5x12"
]

SEED = 42
EPOCHS = 25
BATCH_SIZE = 16
LR = 0.00001
PATIENCE = 7

random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("사용 장치:", DEVICE)

# ==========================================
# 2. 이미지 전처리
# ==========================================

class ResizeWithPadding:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = image.convert("RGB")

        width, height = image.size
        scale = self.size / max(width, height)

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR
        )

        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2

        right = self.size - new_width - left
        bottom = self.size - new_height - top

        return ImageOps.expand(
            image,
            border=(left, top, right, bottom),
            fill=(128, 128, 128)
        )


train_transform = transforms.Compose([
    ResizeWithPadding(224),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    ResizeWithPadding(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 3. 같은 볼트 개체끼리 그룹화
# ==========================================

def get_object_id(path):
    match = re.search(r"개체(\d+)_", path.name)

    if match:
        return match.group(1)

    return path.stem


train_samples = []
val_samples = []

for class_index, class_name in enumerate(CLASSES):

    folder = DATA_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(
            f"데이터 폴더 없음: {folder}"
        )

    images = sorted(folder.glob("*.jpg"))

    groups = defaultdict(list)

    for path in images:
        object_id = get_object_id(path)
        groups[object_id].append(path)

    group_ids = sorted(groups.keys())
    random.Random(SEED).shuffle(group_ids)

    if len(group_ids) < 2:
        raise ValueError(
            f"{class_name}: 서로 다른 개체가 2개 이상 필요합니다."
        )

    val_count = max(1, round(len(group_ids) * 0.2))
    val_count = min(val_count, len(group_ids) - 1)

    val_ids = set(group_ids[:val_count])

    for object_id, paths in groups.items():
        for path in paths:
            sample = (path, class_index)

            if object_id in val_ids:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

print("\n========== 데이터 분할 ==========")

for i, class_name in enumerate(CLASSES):
    train_n = sum(y == i for _, y in train_samples)
    val_n = sum(y == i for _, y in val_samples)

    print(
        f"{class_name}: "
        f"학습 {train_n}장 / 검증 {val_n}장"
    )

print("학습 총합:", len(train_samples))
print("검증 총합:", len(val_samples))

# ==========================================
# 4. Dataset / DataLoader
# ==========================================

class BoltDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        return image, label


train_dataset = BoltDataset(
    train_samples,
    train_transform
)

val_dataset = BoltDataset(
    val_samples,
    val_transform
)

# 데이터가 적은 클래스도 균형 있게 학습
counts = Counter(y for _, y in train_samples)

weights = [
    1.0 / counts[y]
    for _, y in train_samples
]

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
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

# ==========================================
# 5. 기존 ResNet-18 모델 불러오기
# ==========================================

if not OLD_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"기존 모델이 없습니다: {OLD_MODEL_PATH}"
    )

checkpoint = torch.load(
    OLD_MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

old_classes = checkpoint["class_names"]

if old_classes != CLASSES:
    raise ValueError(
        f"클래스 순서가 다릅니다.\n"
        f"기존: {old_classes}\n"
        f"현재: {CLASSES}"
    )

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES)
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)

print("\n기존 ResNet-18 가중치 로드 완료")

# ==========================================
# 6. 학습 설정
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0001
)

best_accuracy = -1.0
best_loss = float("inf")
best_state = None
patience_counter = 0

history = []

# ==========================================
# 7. 추가 학습
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    train_loss /= max(1, train_total)
    val_loss /= max(1, val_total)

    train_acc = train_correct / max(1, train_total)
    val_acc = val_correct / max(1, val_total)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Acc: {train_acc * 100:.1f}% | "
        f"Val Acc: {val_acc * 100:.1f}% | "
        f"Val Loss: {val_loss:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    improved = (
        val_acc > best_accuracy
        or (
            val_acc == best_accuracy
            and val_loss < best_loss
        )
    )

    if improved:

        best_accuracy = val_acc
        best_loss = val_loss
        best_state = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": CLASSES,
                "epoch": epoch + 1,
                "val_accuracy": best_accuracy
            },
            SAVE_DIR / "best_resnet18_v10_roi.pth"
        )

        print("  -> 최고 모델 저장")

    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly Stopping")
        break

# ==========================================
# 8. 학습 결과 저장
# ==========================================

with open(
    SAVE_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        history,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n========== 학습 완료 ==========")
print(f"최고 검증 정확도: {best_accuracy * 100:.1f}%")
print("저장 위치:", SAVE_DIR)
print("기존 v8 모델은 변경되지 않았습니다.")
