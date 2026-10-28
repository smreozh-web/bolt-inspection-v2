from pathlib import Path
import random
import shutil

BASE_DIR = Path(__file__).resolve().parent

# 새 Roboflow 데이터셋 폴더 자동 검색
folders = [
    p for p in BASE_DIR.iterdir()
    if p.is_dir()
    and "bolt_detection_v1.v2-" in p.name
]

if len(folders) != 1:
    raise RuntimeError(
        f"v2 데이터셋 폴더를 하나만 찾아야 합니다. 검색 결과: {folders}"
    )

SOURCE = folders[0]
OUTPUT = BASE_DIR / "bolt_detection_split_v2"

images_dir = SOURCE / "train" / "images"
labels_dir = SOURCE / "train" / "labels"

images = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

if not images:
    raise RuntimeError("이미지를 찾지 못했습니다.")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} 폴더가 이미 있습니다. 기존 데이터를 보호하기 위해 중단합니다."
    )

random.seed(42)
random.shuffle(images)

n = len(images)
n_train = int(n * 0.7)
n_valid = int(n * 0.2)

splits = {
    "train": images[:n_train],
    "valid": images[n_train:n_train + n_valid],
    "test": images[n_train + n_valid:]
}

for split, files in splits.items():
    out_images = OUTPUT / split / "images"
    out_labels = OUTPUT / split / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for image in files:
        shutil.copy2(image, out_images / image.name)

        label = labels_dir / (image.stem + ".txt")

        if label.exists():
            shutil.copy2(label, out_labels / label.name)

    print(f"{split}: {len(files)}장")

yaml_text = """train: train/images
val: valid/images
test: test/images

nc: 1
names: ['bolt']
"""

(OUTPUT / "data.yaml").write_text(
    yaml_text,
    encoding="utf-8"
)

print("\nv2 데이터셋 분할 완료!")
print("저장 위치:", OUTPUT)