
from pathlib import Path
from collections import defaultdict
import copy
import hashlib
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image, ImageOps

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# 1. 기본 설정
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

VERTICAL_DIR = BASE_DIR / "screw_dataset_v5_vertical"
CLS_DIR = BASE_DIR / "screw_cls_v2"
EXTRA_DIR = BASE_DIR / "screw_dataset"

SAVE_DIR = BASE_DIR / "runs" / "resnet18" / "v8_weighted"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "M3x8", "M3x12",
    "M4x8", "M4x12",
    "M5x8", "M5x12"
]

SEED = 42
EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
PATIENCE = 7

# 세로형 데이터의 학습 가중치
VERTICAL_WEIGHT = 3.0
OTHER_WEIGHT = 1.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("사용 장치:", DEVICE)


# ==========================================
# 2. 이미지 전처리
# ==========================================
class ResizeWithPadding:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        w, h = image.size
        scale = min(self.size / w, self.size / h)

        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))

        image = image.resize((new_w, new_h))

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        return ImageOps.expand(
            image,
            border=(
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2
            ),
            fill=(128, 128, 128)
        )


preprocess = transforms.Compose([
    ResizeWithPadding(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. 데이터 수집 함수
# ==========================================
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_images(folder):
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def image_hash(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB").resize((64, 64))
        data = np.asarray(img)

    return hashlib.sha256(data.tobytes()).hexdigest()


samples = {
    "train": [],
    "val": [],
    "test": []
}

# 출처와 샘플링 가중치를 따로 저장
train_weights = []

seen_hashes = {}
seen_names = {}
stats = defaultdict(int)


def add_sample(path, label, split, source):
    path = Path(path)

    try:
        key = image_hash(path)
    except Exception as e:
        print("이미지 읽기 실패:", path, e)
        stats["read_error"] += 1
        return

    if key in seen_hashes:
        old_split, old_label, old_path = seen_hashes[key]

        if old_label != label:
            print("경고: 다른 규격에서 동일 이미지 발견!")
            print("기존:", old_path)
            print("추가:", path)

        stats["duplicate"] += 1
        return

    name_key = (label, path.name)

    if name_key in seen_names:
        stats["same_filename"] += 1
        return

    seen_hashes[key] = (split, label, path)
    seen_names[name_key] = path

    samples[split].append((path, label))
    stats[f"{source}_{split}"] += 1

    if split == "train":
        if source == "vertical":
            train_weights.append(VERTICAL_WEIGHT)
        else:
            train_weights.append(OTHER_WEIGHT)


# ==========================================
# 4. 세로형 데이터셋 개체별 분할
# ==========================================
print("\n[1] screw_dataset_v5_vertical 수집")

for class_index, class_name in enumerate(CLASSES):
    class_dir = VERTICAL_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(class_dir)

    groups = defaultdict(list)

    for path in get_images(class_dir):
        group_name = path.stem.split("_")[0]
        groups[group_name].append(path)

    group_names = sorted(groups.keys())

    print(f"\n[{class_name}] 발견된 개체 그룹")

    for group_name in group_names:
        print(
            f"  {group_name}: "
            f"{len(groups[group_name])}장"
        )

    if len(group_names) != 5:
        raise ValueError(
            f"{class_name}: 개체 그룹이 "
            f"{len(group_names)}개입니다. "
            "정확히 5개가 필요합니다."
        )

    rng = random.Random(SEED)
    rng.shuffle(group_names)

    train_groups = group_names[:3]
    val_groups = group_names[3:4]
    test_groups = group_names[4:5]

    for group in train_groups:
        for path in groups[group]:
            add_sample(
                path, class_index, "train", "vertical"
            )

    for group in val_groups:
        for path in groups[group]:
            add_sample(
                path, class_index, "val", "vertical"
            )

    for group in test_groups:
        for path in groups[group]:
            add_sample(
                path, class_index, "test", "vertical"
            )

    print(
        f"{class_name}: "
        f"Train={train_groups}, "
        f"Val={val_groups}, "
        f"Test={test_groups}"
    )


# ==========================================
# 5. screw_cls_v2 추가
# ==========================================
print("\n[2] screw_cls_v2 추가")

for class_index, class_name in enumerate(CLASSES):
    for original_split in ("train", "val", "test"):
        folder = CLS_DIR / original_split / class_name

        for path in get_images(folder):
            add_sample(
                path,
                class_index,
                "train",
                "cls_v2"
            )


# ==========================================
# 6. screw_dataset 추가
# ==========================================
print("\n[3] screw_dataset 추가")

for class_index, class_name in enumerate(CLASSES):
    folder = EXTRA_DIR / class_name

    for path in get_images(folder):
        add_sample(
            path,
            class_index,
            "train",
            "extra"
        )


# ==========================================
# 7. 데이터 수 확인
# ==========================================
print("\n===== 데이터 수집 결과 =====")

for split in ("train", "val", "test"):
    print(f"{split}: {len(samples[split])}장")

print("중복 이미지 제외:", stats["duplicate"])
print("동일 파일명 제외:", stats["same_filename"])
print("읽기 실패:", stats["read_error"])

for source in ("vertical", "cls_v2", "extra"):
    print(
        source,
        "Train:", stats[f"{source}_train"],
        "Val:", stats[f"{source}_val"],
        "Test:", stats[f"{source}_test"]
    )

for split in ("train", "val", "test"):
    if not samples[split]:
        raise ValueError(f"{split} 데이터가 없습니다.")

if len(train_weights) != len(samples["train"]):
    raise ValueError("학습 이미지와 가중치 수가 다릅니다.")

print("\n===== 학습 가중치 =====")
print("세로형:", VERTICAL_WEIGHT)
print("기타:", OTHER_WEIGHT)

vertical_count = stats["vertical_train"]
other_count = (
    stats["cls_v2_train"] + stats["extra_train"]
)

weighted_vertical = vertical_count * VERTICAL_WEIGHT
weighted_other = other_count * OTHER_WEIGHT
total_weight = weighted_vertical + weighted_other

print(
    "세로형 예상 샘플링 비율:",
    f"{weighted_vertical / total_weight * 100:.1f}%"
)


# ==========================================
# 8. Dataset / DataLoader
# ==========================================
class BoltDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]

        with Image.open(path) as image:
            image = preprocess(image)

        return image, label


# 세로형 이미지를 3배 높은 확률로 선택
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(train_weights),
    num_samples=len(samples["train"]),
    replacement=True
)

train_loader = DataLoader(
    BoltDataset(samples["train"]),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    BoltDataset(samples["val"]),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    BoltDataset(samples["test"]),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 9. ResNet-18 모델
# ==========================================
model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 10. 학습 및 평가 함수
# ==========================================
def run_epoch(loader, training=False):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(predictions.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_true, all_pred)

    macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0
    )

    return avg_loss, accuracy, macro_f1, all_true, all_pred


# ==========================================
# 11. 모델 학습
# ==========================================
best_f1 = -1
best_loss = float("inf")
best_state = None
wait = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc, train_f1, _, _ = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_acc, val_f1, _, _ = run_epoch(
        val_loader,
        training=False
    )

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    improved = (
        val_f1 > best_f1 + 1e-6
        or (
            abs(val_f1 - best_f1) <= 1e-6
            and val_loss < best_loss
        )
    )

    if improved:
        best_f1 = val_f1
        best_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0

        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": CLASSES,
                "val_macro_f1": best_f1,
                "image_size": 224,
                "roi_size": [1200, 2000],
                "vertical_weight": VERTICAL_WEIGHT
            },
            SAVE_DIR / "best_resnet18_v8.pth"
        )

        print("  → 최고 성능 모델 저장")

    else:
        wait += 1

    if wait >= PATIENCE:
        print("Early stopping")
        break


# ==========================================
# 12. 최종 테스트
# ==========================================
if best_state is None:
    raise RuntimeError("학습된 모델이 없습니다.")

model.load_state_dict(best_state)

test_loss, test_acc, test_f1, y_true, y_pred = run_epoch(
    test_loader,
    training=False
)

print("\n===== ResNet-18 v8 테스트 결과 =====")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro-F1: {test_f1:.4f}")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0
)

print(report)

(SAVE_DIR / "test_report.txt").write_text(
    f"Accuracy: {test_acc:.4f}\n"
    f"Macro-F1: {test_f1:.4f}\n\n"
    + report,
    encoding="utf-8"
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES)))
)

fig, ax = plt.subplots(figsize=(8, 7))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title("ResNet-18 v8 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    SAVE_DIR / "confusion_matrix_v8.png",
    dpi=200
)

plt.close()

print("\n학습 및 평가 완료!")
print("모델 저장 위치:", SAVE_DIR)
