
from pathlib import Path
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "screw_dataset_v3"
OUTPUT_DIR = BASE_DIR / "screw_dataset_v5_vertical"

CLASSES = [
    "M3x8", "M3x12",
    "M4x8", "M4x12",
    "M5x8", "M5x12"
]

# 기존 가로 이미지 기준 자르기 설정
CROP_WIDTH = 2000
CROP_HEIGHT = 1200

CENTER_X = 1850
CENTER_Y = 1550

for class_name in CLASSES:
    source_folder = SOURCE_DIR / class_name
    output_folder = OUTPUT_DIR / class_name
    output_folder.mkdir(parents=True, exist_ok=True)

    count = 0

    for image_path in sorted(source_folder.rglob("*")):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png"
        ]:
            continue

        with Image.open(image_path) as image:
            # 원본 이미지의 실제 저장 방향 확인
            original_size = image.size

            # 기존 좌표계에서 먼저 자르기
            if original_size != (4000, 3000):
                print(
                    "해상도 확인 필요:",
                    image_path.name,
                    original_size
                )
                continue

            image = image.convert("RGB")

            left = CENTER_X - CROP_WIDTH // 2
            top = CENTER_Y - CROP_HEIGHT // 2

            cropped = image.crop((
                left,
                top,
                left + CROP_WIDTH,
                top + CROP_HEIGHT
            ))

            # 원본 EXIF 방향을 잘라낸 이미지에 적용
            orientation = Image.open(image_path).getexif().get(274, 1)

            if orientation == 6:
                cropped = cropped.transpose(
                    Image.Transpose.ROTATE_270
                )
            elif orientation == 8:
                cropped = cropped.transpose(
                    Image.Transpose.ROTATE_90
                )
            elif orientation == 3:
                cropped = cropped.transpose(
                    Image.Transpose.ROTATE_180
                )

            # 폴더가 다른 사진의 이름 중복 방지
            relative_path = image_path.relative_to(source_folder)
            safe_name = "_".join(relative_path.parts)

            cropped.save(
                output_folder / safe_name,
                quality=95
            )

            count += 1

    print(f"{class_name}: {count}장 저장")

print("세로 방향 이미지 가공 완료!")