"""
test_all_images.py

Classifies EVERY image in a test folder. Nothing is skipped.

Crop strategy per image (first that applies):
  1. held-object crop   (hand + YOLO box overlap, same as detect.py)
  2. largest object crop (YOLO found a non-person object, no hand found)
  3. whole image        (nothing found)

Labels (optional, needed for accuracy):
  - If the folder has class subfolders (plastic/, paper/, ...), folder names are the labels.
  - Otherwise pass a CSV as the second argument: first column filename, second column label.

Usage (from project root, venv active):
    python src\\test_all_images.py "data\\test_external\\extras"
    python src\\test_all_images.py "data\\test_external\\extras" labels.csv

Writes results_<foldername>.csv in the current folder.
"""

import contextlib
import csv
import io
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

import classifier
from roi_selector import compute_scores
from test_external_with_pipeline import (
    MODEL_WEIGHTS, MAX_HANDS, HAND_DETECTION_CONFIDENCE,
    ensure_hand_model, run_yolo_on_frame, get_hand_regions,
    filter_held_objects, crop_object, load_image_exif_safe,
)

# Return the raw top-1 label always (no "Uncertain") so accuracy is meaningful.
classifier.CONFIDENCE_THRESHOLD = 0.0
classifier.DEBUG_SAVE_CROPS = False

EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

ALIASES = {
    "shoes": "footwear", "shoe": "footwear",
    "clothes": "cloth", "clothing": "cloth", "textile": "cloth", "textile trash": "cloth",
    "cardboard": "paper",
    "brown-glass": "glass", "green-glass": "glass", "white-glass": "glass",
    "biological": "organic", "food organics": "organic", "vegetation": "organic",
    "battery": "hazard", "batteries": "hazard",
}


def norm(label):
    l = str(label).strip().lower()
    return ALIASES.get(l, l)


def collect_items(root, labels_csv):
    items = []
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        for d in sorted(subdirs):
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in EXT:
                    items.append((p, norm(d.name)))
        return items

    csv_labels = {}
    if labels_csv:
        with open(labels_csv, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        for row in rows[1:]:
            if len(row) >= 2:
                csv_labels[Path(row[0].strip()).name] = norm(row[1])

    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in EXT:
            items.append((p, csv_labels.get(p.name)))
    return items


def classify(frame):
    """Returns (method, label, confidence)."""
    detections = run_yolo_on_frame(YOLO_MODEL, frame)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_regions = get_hand_regions(HAND_LANDMARKER.detect(mp_image), w, h)

    objects = [d for d in detections if d.class_name != "person"]
    objects = compute_scores(objects, w, h) if objects else objects
    held = filter_held_objects(objects, hand_regions)

    crop, method = None, "whole image"
    if held:
        crop = crop_object(frame, max(held, key=lambda d: d.score))
        method = "held-object crop"
    elif objects:
        crop = crop_object(frame, max(objects, key=lambda d: d.score))
        method = "largest object crop"

    if crop is None or crop.size == 0:
        crop, method = frame, "whole image"

    with contextlib.redirect_stdout(io.StringIO()):
        label, conf = classifier.predict(crop)
    return method, label, conf


def main():
    global YOLO_MODEL, HAND_LANDMARKER

    if len(sys.argv) < 2:
        print("Usage: python src\\test_all_images.py <folder> [labels.csv]")
        return
    root = Path(sys.argv[1])
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else None
    if not root.exists():
        print(f"ERROR: {root} not found.")
        return

    YOLO_MODEL = YOLO(MODEL_WEIGHTS)
    HAND_LANDMARKER = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_hand_model()),
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
    )

    items = collect_items(root, labels_csv)
    has_labels = any(t is not None for _, t in items)
    print(f"\n{len(items)} images. Labels found: {'yes' if has_labels else 'NO (predictions only)'}\n")

    rows, method_counts, pred_counts = [], {}, {}
    y_true, y_pred = [], []

    for path, true_label in items:
        try:
            frame = load_image_exif_safe(path)
            method, label, conf = classify(frame)
        except Exception as e:
            print(f"{path.name}: ERROR {e}")
            continue

        method_counts[method] = method_counts.get(method, 0) + 1
        pred_counts[label] = pred_counts.get(label, 0) + 1
        rows.append([path.name, method, label, f"{conf:.4f}", true_label or ""])

        mark = ""
        if true_label is not None:
            mark = "  OK" if label == true_label else f"  WRONG (true: {true_label})"
            y_true.append(true_label)
            y_pred.append(label)
        print(f"{path.name}: {label} ({conf:.0%}) [{method}]{mark}")

    out_csv = Path(f"results_{root.name}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "crop_method", "predicted", "confidence", "true_label"])
        w.writerows(rows)

    print("\n=== Crop methods used ===")
    for k, v in method_counts.items():
        print(f"  {k}: {v}")
    print("\n=== Predicted class counts ===")
    for k, v in sorted(pred_counts.items()):
        print(f"  {k}: {v}")

    if y_true:
        known = set(classifier.CLASS_NAMES)
        unknown = sorted({t for t in y_true if t not in known})
        if unknown:
            print(f"\nNOTE: labels not in the model's classes (always counted wrong): {unknown}")
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        print(f"\n=== ACCURACY: {acc:.1%} on {len(y_true)} labeled images ===")
        try:
            from sklearn.metrics import classification_report
            print(classification_report(y_true, y_pred, digits=3, zero_division=0))
        except Exception:
            pass

    print(f"\nSaved per-image results to {out_csv}")
    HAND_LANDMARKER.close()


if __name__ == "__main__":
    main()
