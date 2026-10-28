
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "runs" / "resnet18"
    / "v10_roi_finetune" / "best_resnet18_v10_roi.pth"
)

DATA_DIR = BASE_DIR / "camera_resnet_dataset"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

class ResizeWithPadding:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = image.convert("RGB")
        w, h = image.size

        scale = self.size / max(w, h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        image = image.resize((nw, nh))

        canvas = Image.new(
            "RGB", (self.size, self.size),
            (128, 128, 128)
        )

        canvas.paste(
            image,
            ((self.size - nw) // 2,
             (self.size - nh) // 2)
        )

        return canvas


transform = transforms.Compose([
    ResizeWithPadding(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("사용 장치:", DEVICE)
print("클래스 순서:", class_names)
print("-" * 55)

correct = 0
total = 0

with torch.no_grad():
    for true_label in sorted(
        p.name for p in DATA_DIR.iterdir() if p.is_dir()
    ):
        for image_path in sorted(
            (DATA_DIR / true_label).glob("*.jpg")
        ):
            image = Image.open(image_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(DEVICE)

            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)[0]

            index = int(torch.argmax(probabilities))
            predicted = class_names[index]
            confidence = float(probabilities[index]) * 100

            total += 1
            correct += int(predicted == true_label)

            print(
                f"실제: {true_label:6s} | "
                f"예측: {predicted:6s} | "
                f"확률: {confidence:5.1f}% | "
                f"{image_path.name}"
            )

print("-" * 55)
print(f"정확도: {correct}/{total} = "
      f"{correct / total * 100:.1f}%"
      if total else "테스트 이미지 없음")
