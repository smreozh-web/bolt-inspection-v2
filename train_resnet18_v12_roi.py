
from pathlib import Path
from collections import Counter, defaultdict
import copy
import csv
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

# 이미 학습한 v11 모델에서 시작
V11_MODEL_PATH = (
    BASE_DIR / "runs" / "resnet18"
    / "v11_roi_finetune"
    / "best_resnet18_v11_roi.pth"
)

# v12는 별도 폴더에 저장
SAVE_DIR = (
    BASE_DIR / "runs" / "resnet18"
    / "v12_roi_finetune"
)

CLASSES = [
    "M3x8", "M3x12",
    "M4x8", "M4x12",
    "M5x8", "M5x12"
]

SEED = 42
EPOCHS = 25
BATCH_SIZE = 16
PATIENCE = 7

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

random.seed(SEED)
torch.manual_seed(SEED)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("사용 장치:", DEVICE)


# ==========================================
# 2. 파일 확인
# ==========================================

for path in [OLD_DATA_DIR, NEW_DATA_DIR, V11_MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"파일 또는 폴더 없음: {path}")

for folder in [OLD_DATA_DIR, NEW_DATA_DIR]:
    for cls in CLASSES:
        if not (folder / cls).is_dir():
            raise FileNotFoundError(f"규격 폴더 없음: {folder / cls}")


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
            ((self.size - nw) // 2,
             (self.size - nh) // 2)
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
# 4. 데이터 읽기
# ==========================================

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def get_capture_group(path):
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
                f"촬영 회차 확인 불가: {path.name}"
            )

        new_groups[group].append((path, class_idx))


# ==========================================
# 5. v11과 동일한 데이터 분할 재현
# ==========================================

# v11 학습 때 사용한 seed와 분할 방식을 유지
group_names = sorted(new_groups.keys())
random.shuffle(group_names)

n_groups = len(group_names)

if n_groups < 5:
    raise RuntimeError("촬영 회차가 5개 미만입니다.")

test_count = max(2, round(n_groups * 0.20))
val_count = max(1, round(n_groups * 0.15))

if test_count + val_count >= n_groups:
    raise RuntimeError("학습 회차가 부족합니다.")

test_groups = set(group_names[:test_count])

val_groups = set(
    group_names[test_count:test_count + val_count]
)

train_groups = set(
    group_names[test_count + val_count:]
)

train_samples = list(old_samples)
val_samples = []
test_samples = []

for group, samples in new_groups.items():
    if group in test_groups:
        test_samples.extend(samples)
    elif group in val_groups:
        val_samples.extend(samples)
    else:
        train_samples.extend(samples)

if not train_samples or not val_samples or not test_samples:
    raise RuntimeError("데이터 분할 결과가 비어 있습니다.")

print("\n========== 데이터 분할 ==========")
print("학습:", len(train_samples))
print("검증:", len(val_samples))
print("테스트:", len(test_samples))
print("테스트 회차:", sorted(test_groups))


# ==========================================
# 6. 데이터 로더
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


train_dataset = BoltDataset(train_samples, train_transform)
val_dataset = BoltDataset(val_samples, eval_transform)
test_dataset = BoltDataset(test_samples, eval_transform)

class_counts = Counter(
    label for _, label in train_samples
)

weights = [
    1.0 / class_counts[label]
    for _, label in train_samples
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 7. v11 모델 불러오기
# ==========================================

checkpoint = torch.load(
    V11_MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

if list(checkpoint["class_names"]) != CLASSES:
    raise ValueError("v11 모델의 클래스 순서가 다릅니다.")


def create_model():
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(
        net.fc.in_features,
        len(CLASSES)
    )
    return net


model = create_model()
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)

# 비교용 v11 원본
baseline = create_model()
baseline.load_state_dict(checkpoint["model_state_dict"])
baseline = baseline.to(DEVICE)
baseline.eval()

print("\nv11 모델 불러오기 완료")


# ==========================================
# 8. 평가 함수
# ==========================================

criterion = nn.CrossEntropyLoss()


def evaluate(net, loader):
    net.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            n = labels.size(0)

            total += n
            correct += (predictions == labels).sum().item()
            loss_sum += loss.item() * n

            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    return {
        "accuracy": correct / total if total else 0.0,
        "loss": loss_sum / total if total else 0.0,
        "true": true_labels,
        "pred": predicted_labels
    }


# ==========================================
# 9. v11 기준 성능
# ==========================================

v11_result = evaluate(baseline, test_loader)

print(
    "\nv11 테스트 정확도:",
    f"{v11_result['accuracy'] * 100:.1f}%"
)


# ==========================================
# 10. v12 추가 학습 설정
# ==========================================

# 전체 모델을 먼저 고정
for parameter in model.parameters():
    parameter.requires_grad = False

# layer3, layer4, fc만 학습
for layer in [model.layer3, model.layer4, model.fc]:
    for parameter in layer.parameters():
        parameter.requires_grad = True

# 앞쪽 특징은 보수적으로, 마지막 분류층은 더 학습
optimizer = torch.optim.AdamW([
    {
        "params": model.layer3.parameters(),
        "lr": 0.000005
    },
    {
        "params": model.layer4.parameters(),
        "lr": 0.00001
    },
    {
        "params": model.fc.parameters(),
        "lr": 0.00003
    }
], weight_decay=0.0001)

best_val_loss = float("inf")
best_state = None
best_epoch = 0
wait = 0
history = []


# ==========================================
# 11. v12 학습
# ==========================================

print("\n========== v12 학습 시작 ==========")

for epoch in range(EPOCHS):
    model.train()

    # 적은 데이터로 BatchNorm 통계가 흔들리지 않도록 고정
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

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
        n = labels.size(0)

        train_total += n
        train_correct += (
            predictions == labels
        ).sum().item()

        train_loss_sum += loss.item() * n

    train_acc = train_correct / train_total
    train_loss = train_loss_sum / train_total

    val_result = evaluate(model, val_loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Acc: {train_acc * 100:.1f}% | "
        f"Val Acc: {val_result['accuracy'] * 100:.1f}% | "
        f"Val Loss: {val_result['loss']:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_accuracy": train_acc,
        "train_loss": train_loss,
        "val_accuracy": val_result["accuracy"],
        "val_loss": val_result["loss"]
    })

    if val_result["loss"] < best_val_loss:
        best_val_loss = val_result["loss"]
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())
        wait = 0

        print("  -> 최고 모델 갱신")
    else:
        wait += 1

    if wait >= PATIENCE:
        print("Early Stopping")
        break


# ==========================================
# 12. v12 모델 저장
# ==========================================

if best_state is None:
    raise RuntimeError("저장할 모델이 없습니다.")

model.load_state_dict(best_state)

model_path = SAVE_DIR / "best_resnet18_v12_roi.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": CLASSES,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss
}, model_path)


# ==========================================
# 13. v11 vs v12 평가
# ==========================================

v12_result = evaluate(model, test_loader)


def make_confusion_matrix(result):
    matrix = [
        [0 for _ in CLASSES]
        for _ in CLASSES
    ]

    for actual, predicted in zip(
        result["true"], result["pred"]
    ):
        matrix[actual][predicted] += 1

    return matrix


def print_class_metrics(result, name):
    matrix = make_confusion_matrix(result)

    print(f"\n========== {name} 클래스별 평가 ==========")
    print("클래스       Precision   Recall      F1     Support")

    metrics = {}

    for i, cls in enumerate(CLASSES):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)

        precision = (
            tp / predicted_count
            if predicted_count else 0.0
        )

        recall = (
            tp / support
            if support else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

        print(
            f"{cls:<10} "
            f"{precision:>8.3f} "
            f"{recall:>8.3f} "
            f"{f1:>8.3f} "
            f"{support:>8}"
        )

    return matrix, metrics


v11_matrix, v11_metrics = print_class_metrics(
    v11_result, "v11"
)

v12_matrix, v12_metrics = print_class_metrics(
    v12_result, "v12"
)


# ==========================================
# 14. 혼동행렬 CSV 저장
# ==========================================

def save_matrix(filename, matrix):
    with open(
        SAVE_DIR / filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Actual / Predicted"] + CLASSES)

        for cls, row in zip(CLASSES, matrix):
            writer.writerow([cls] + row)


save_matrix("confusion_matrix_v11.csv", v11_matrix)
save_matrix("confusion_matrix_v12.csv", v12_matrix)


# ==========================================
# 15. 평가 결과 저장
# ==========================================

report = {
    "v11_test_accuracy": v11_result["accuracy"],
    "v12_test_accuracy": v12_result["accuracy"],
    "best_epoch": best_epoch,
    "v11_class_metrics": v11_metrics,
    "v12_class_metrics": v12_metrics,
    "history": history,
    "test_groups": sorted(test_groups),
    "test_samples": [
        {
            "path": str(path),
            "class": CLASSES[label]
        }
        for path, label in test_samples
    ]
}

with open(
    SAVE_DIR / "training_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n========== 최종 비교 ==========")
print(f"v11 정확도: {v11_result['accuracy'] * 100:.1f}%")
print(f"v12 정확도: {v12_result['accuracy'] * 100:.1f}%")
print("v12 최고 Epoch:", best_epoch)
print("v12 모델:", model_path)

if v12_result["accuracy"] > v11_result["accuracy"]:
    print("테스트 정확도가 개선되었습니다.")
elif v12_result["accuracy"] == v11_result["accuracy"]:
    print("테스트 정확도가 동일합니다.")
else:
    print("테스트 정확도가 낮아졌습니다.")

print("\n기존 v11 모델은 변경하지 않았습니다.")
print("YOLO 및 OpenCV 코드는 변경하지 않았습니다.")
