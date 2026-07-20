"""
split_dataset.py

Takes your flat per-class folders in data/raw/ (Plastic/, Paper/, Other/)
and splits each into train/ and val/ subsets, copying files into the
structure MobileNetV3 training expects (via torchvision's ImageFolder,
which requires one subfolder per class under both train/ and val/):

    data/classification/
    ├── train/
    │   ├── Plastic/
    │   ├── Paper/
    │   └── Other/
    └── val/
        ├── Plastic/
        ├── Paper/
        └── Other/

Your original data/raw/ folders are left untouched (files are COPIED,
not moved), so you can re-run this safely if you add more images later.

Run this once before train_classifier.py.
"""

import random
import shutil
from pathlib import Path

# --- Config -----------------------------------------------------------
RAW_DATA_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data/classification")
CLASSES = ["Plastic", "Paper", "Other"]
VAL_SPLIT = 0.2          # 20% of each class goes to validation
RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ------------------------------------------------------------------------


def list_images(class_dir: Path):
    return sorted([
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ])


def main():
    random.seed(RANDOM_SEED)

    if OUTPUT_DIR.exists():
        print(f"Output dir {OUTPUT_DIR} already exists -- files will be "
              f"overwritten/re-added, existing images are not deleted.")

    total_train = 0
    total_val = 0

    for class_name in CLASSES:
        class_dir = RAW_DATA_DIR / class_name
        if not class_dir.exists():
            print(f"WARNING: {class_dir} does not exist, skipping. "
                  f"Did you copy your dataset into data/raw/{class_name}/ ?")
            continue

        images = list_images(class_dir)
        if not images:
            print(f"WARNING: no images found in {class_dir}, skipping.")
            continue

        random.shuffle(images)
        val_count = max(1, int(len(images) * VAL_SPLIT))
        val_images = images[:val_count]
        train_images = images[val_count:]

        train_out = OUTPUT_DIR / "train" / class_name
        val_out = OUTPUT_DIR / "val" / class_name
        train_out.mkdir(parents=True, exist_ok=True)
        val_out.mkdir(parents=True, exist_ok=True)

        for img_path in train_images:
            shutil.copy2(img_path, train_out / img_path.name)
        for img_path in val_images:
            shutil.copy2(img_path, val_out / img_path.name)

        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val "
              f"(from {len(images)} total)")
        total_train += len(train_images)
        total_val += len(val_images)

    print(f"\nDone. Total: {total_train} train images, {total_val} val images.")
    print(f"Output written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
