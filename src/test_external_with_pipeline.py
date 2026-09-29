"""
test_external_with_pipeline.py

Exact match to detect.py's held-object logic, adapted for static test
images instead of a live webcam feed. Reuses your actual roi_selector.py
(compute_scores) and classifier.py (predict) directly -- not reimplemented.

Only real difference from detect.py: no DisplayState / debouncing /
majority-vote smoothing, since each test image is independent (there's
no "previous frame" to smooth against). Each image gets exactly one
raw classifier prediction, same as detect.py would produce on the very
first frame a fresh object is detected.

Usage:
    python src\\test_external_with_pipeline.py <path_to_test_folder>

Must be run from src\\ context (or project root with src on path) since
it imports roi_selector and classifier, same as detect.py does.
"""

import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO
from PIL import Image, ImageOps

from roi_selector import Detection, compute_scores
from classifier import predict


def load_image_exif_safe(path, max_dimension=1600):
    """
    Loads an image respecting EXIF orientation metadata, then returns it as
    a BGR numpy array (OpenCV's expected format).

    Also downscales large images (phone photos are often 3000-4000px+ per
    side) to a max dimension -- processing hundreds of full-resolution
    images back-to-back without this was causing memory exhaustion crashes
    partway through a run. Downscaling doesn't hurt detection accuracy
    meaningfully here since YOLO/MediaPipe resize internally anyway.
    """
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)  # applies EXIF rotation, if any
    pil_img = pil_img.convert("RGB")

    if max(pil_img.size) > max_dimension:
        pil_img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    rgb_array = np.array(pil_img)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array

# ---- Config: copied exactly from detect.py --------------------------------

MODEL_WEIGHTS = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25

MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.5
HAND_REGION_PADDING_RATIO = 0.25
HAND_REGION_PADDING_MIN_PX = 20

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ---- Functions copied exactly from detect.py ------------------------------


def ensure_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        print("Downloading hand_landmarker.task (one-time)...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    return HAND_MODEL_PATH


def run_yolo_on_frame(model, frame):
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            class_name=class_name,
            confidence=conf,
        ))
    return detections


def get_hand_regions(hand_result, frame_width, frame_height):
    hand_regions = []
    if not hand_result.hand_landmarks:
        return hand_regions

    for landmarks in hand_result.hand_landmarks:
        xs = [lm.x * frame_width for lm in landmarks]
        ys = [lm.y * frame_height for lm in landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        pad_x = max((x2 - x1) * HAND_REGION_PADDING_RATIO, HAND_REGION_PADDING_MIN_PX)
        pad_y = max((y2 - y1) * HAND_REGION_PADDING_RATIO, HAND_REGION_PADDING_MIN_PX)
        hand_regions.append((
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(frame_width, x2 + pad_x),
            min(frame_height, y2 + pad_y),
        ))
    return hand_regions


def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def filter_held_objects(objects, hand_regions):
    if not hand_regions:
        return []
    held = []
    for obj in objects:
        obj_box = (obj.x1, obj.y1, obj.x2, obj.y2)
        if any(boxes_overlap(obj_box, hand_box) for hand_box in hand_regions):
            held.append(obj)
    return held


def crop_object(frame, obj, padding_ratio=0.1):
    frame_h, frame_w = frame.shape[:2]
    box_w = obj.x2 - obj.x1
    box_h = obj.y2 - obj.y1
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    x1 = max(0, int(obj.x1 - pad_x))
    y1 = max(0, int(obj.y1 - pad_y))
    x2 = min(frame_w, int(obj.x2 + pad_x))
    y2 = min(frame_h, int(obj.y2 + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ---- Test-script-specific driver ------------------------------------------


def main():
    if len(sys.argv) < 2:
        print("Usage: python src\\test_external_with_pipeline.py <path_to_test_folder>")
        return

    test_root = Path(sys.argv[1])
    if not test_root.exists():
        print(f"ERROR: {test_root} not found.")
        return

    print(f"Loading {MODEL_WEIGHTS} ...")
    yolo_model = YOLO(MODEL_WEIGHTS)

    hand_model_path = ensure_hand_model()
    # IMAGE mode (not VIDEO) since these are independent static images,
    # not a continuous stream -- detect.py uses VIDEO mode + detect_for_video
    # because it processes a live continuous feed; that distinction doesn't
    # apply here, so plain .detect() is the correct adaptation.
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model_path),
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    images = sorted(p for p in test_root.iterdir()
                     if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
    print(f"\nProcessing {len(images)} images...\n")

    counts = {}
    no_held_object = []

    for img_path in images:
        try:
            frame = load_image_exif_safe(img_path)
        except Exception as e:
            print(f"{img_path.name}: could not read image ({e}), skipping")
            continue
        if frame is None or frame.size == 0:
            print(f"{img_path.name}: could not read image, skipping")
            continue

        frame_height, frame_width = frame.shape[:2]

        detections = run_yolo_on_frame(yolo_model, frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_result = hand_landmarker.detect(mp_image)
        hand_regions = get_hand_regions(hand_result, frame_width, frame_height)

        people = [d for d in detections if d.class_name == "person"]
        objects = [d for d in detections if d.class_name != "person"]
        objects = compute_scores(objects, frame_width, frame_height) if objects else objects

        held_objects = filter_held_objects(objects, hand_regions)

        if not held_objects:
            no_held_object.append(img_path.name)
            print(f"{img_path.name}: no held object detected (background only) -- SKIPPED "
                  f"[hands={len(hand_regions)}, objects_seen={len(objects)}]")
            continue

        primary_obj = max(held_objects, key=lambda d: d.score)
        cropped = crop_object(frame, primary_obj)

        if cropped is None or cropped.size == 0:
            print(f"{img_path.name}: empty crop, skipping")
            continue

        label, confidence = predict(cropped)
        counts[label] = counts.get(label, 0) + 1
        print(f"{img_path.name}: {label} ({confidence:.2%}) "
              f"[{len(held_objects)} held object(s), primary was '{primary_obj.class_name}' "
              f"YOLO-detected]")

    print("\n=== Summary (held-object classifications only) ===")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print(f"\n  Skipped (no held object detected): {len(no_held_object)} / {len(images)}")
    if no_held_object:
        print(f"  Skipped files: {no_held_object}")

    hand_landmarker.close()


if __name__ == "__main__":
    main()