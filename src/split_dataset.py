"""
split_dataset.py

Splits data/raw/<class>/ into data/classification/train/<class>/ and
data/classification/val/<class>/ at an 80/20 ratio.

Fully data-driven: whatever class folders exist under data/raw/ get split.
No hardcoded class list, so this works unchanged whether you have 6 classes,
10 classes, or add/remove classes later.

Run from the project root:
    python src\\split_dataset.py
"""

import random
import shutil
from pathlib import Path

RAW_ROOT = Path("data/raw")
OUT_ROOT = Path("data/classification")
TRAIN_DIR = OUT_ROOT / "train"
VAL_DIR = OUT_ROOT / "val"

VAL_RATIO = 0.2
SEED = 42
MIN_IMAGES_WARN = 20  # below this, warn -- not enough data to train that class well
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main():
    random.seed(SEED)

    if not RAW_ROOT.exists():
        print(f"ERROR: {RAW_ROOT} not found.")
        return

    class_dirs = sorted(p for p in RAW_ROOT.iterdir() if p.is_dir())
    if not class_dirs:
        print(f"ERROR: no class folders found under {RAW_ROOT}.")
        return

    print(f"Found {len(class_dirs)} class folders: {[d.name for d in class_dirs]}\n")

    # Clear previous split so re-runs don't mix old + new images
    if OUT_ROOT.exists():
        print(f"Removing existing {OUT_ROOT} before re-splitting...")
        shutil.rmtree(OUT_ROOT)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    skipped_classes = []
    summary = []

    for class_dir in class_dirs:
        images = [p for p in class_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

        if len(images) == 0:
            print(f"WARNING: '{class_dir.name}' has 0 images -- SKIPPING this class entirely. "
                  f"Training will proceed without it. Fix this before a real training run.")
            skipped_classes.append(class_dir.name)
            continue

        if len(images) < MIN_IMAGES_WARN:
            print(f"WARNING: '{class_dir.name}' only has {len(images)} images "
                  f"(recommend at least {MIN_IMAGES_WARN}+). Proceeding anyway.")

        random.shuffle(images)
        val_count = max(1, int(len(images) * VAL_RATIO))
        val_images = images[:val_count]
        train_images = images[val_count:]

        train_class_dir = TRAIN_DIR / class_dir.name
        val_class_dir = VAL_DIR / class_dir.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)

        summary.append((class_dir.name, len(train_images), len(val_images)))
        print(f"{class_dir.name}: {len(train_images)} train / {len(val_images)} val")

    print("\n=== Summary ===")
    total_train = sum(s[1] for s in summary)
    total_val = sum(s[2] for s in summary)
    for name, tr, va in summary:
        print(f"  {name}: {tr} train, {va} val")
    print(f"  TOTAL: {total_train} train, {total_val} val, "
          f"{len(summary)} classes used, {len(skipped_classes)} classes skipped")

    if skipped_classes:
        print(f"\nSkipped (empty) classes: {skipped_classes}")
        print("These will NOT appear in the trained model at all until you add images.")


if __name__ == "__main__":
    main()