"""
detect.py

Phase 1 demo: runs YOLOv8n (pre-trained COCO weights, no custom training)
on a laptop webcam feed.

DISPLAY RULE (as requested):
    - If any non-person object is detected -> show ONLY a RED box around
      the closest such object. Person box and background zone are hidden.
    - Else if a person is detected (no object) -> show a GREEN box around
      the person.
    - Else (nothing detected) -> show a BLUE "scanning zone" box in the
      center of the frame, meaning "waiting for something to appear."

IMPORTANT - READ THIS BEFORE TRUSTING THE LABELS ON SCREEN:
    The text label next to the red box (e.g. "Plastic", "Paper", "Other")
    comes from a hardcoded lookup table (COCO_TO_WASTE_CLASS below), NOT
    from real classification. YOLOv8n-COCO only knows its original 80
    object classes (bottle, cup, book, backpack, etc). It has never been
    trained on "plastic", "paper", or "other" as concepts, and cannot be
    made to recognize them via config changes, thresholds, or code tricks.

    This lookup table only relabels a few COCO classes we already know
    (bottle, cup, book) with more readable names for this demo. Any real
    trash item that doesn't happen to look like one of those 3 COCO
    classes (candy wrappers, loose paper, cans, etc.) will either go
    undetected or get misclassified as an unrelated COCO class (e.g.
    "cell phone") -- this is expected, not a bug, and is exactly why a
    real Plastic/Paper/Other system needs its own labeled dataset and a
    custom-trained model in a later phase. This demo proves the display
    logic and distance heuristic only.

Controls:
    q  - quit
"""

import cv2
from ultralytics import YOLO

from roi_selector import Detection, compute_scores

# --- Config ---------------------------------------------------------------
MODEL_WEIGHTS = "yolov8n.pt"     # auto-downloads on first run
CONFIDENCE_THRESHOLD = 0.4
WEBCAM_INDEX = 0                 # change if you have multiple cameras
DEBUG_PRINT_ALL_RAW = True       # prints every raw guess YOLO makes, each frame

# Colors are in OpenCV's BGR order, not RGB.
PERSON_COLOR = (0, 255, 0)       # green
OBJECT_COLOR = (0, 0, 255)       # red
SCAN_ZONE_COLOR = (255, 0, 0)    # blue

# Best-effort label lookup ONLY -- see big warning in the module docstring.
# Anything not in this table falls back to "Other".
COCO_TO_WASTE_CLASS = {
    "bottle": "Plastic",
    "cup": "Paper",     # rough stand-in, NOT accurate for real paper cups
    "book": "Paper",
}
DEFAULT_WASTE_CLASS = "Other"
# ---------------------------------------------------------------------------


def run_yolo_on_frame(model, frame):
    """Runs YOLO on a single frame, returns a list of Detection objects."""
    results = model(frame, verbose=False)[0]

    detections = []
    raw_seen = []
    for box in results.boxes:
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        raw_seen.append(f"{class_name}({conf:.2f})")

        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            class_name=class_name,
            confidence=conf,
        ))

    if DEBUG_PRINT_ALL_RAW:
        print("raw guesses this frame:", raw_seen if raw_seen else "(none)")

    return detections


def draw_scan_zone(frame, frame_width, frame_height):
    """Draws a fixed blue 'waiting for something' zone in the frame center."""
    margin_x = int(frame_width * 0.2)
    margin_y = int(frame_height * 0.2)
    p1 = (margin_x, margin_y)
    p2 = (frame_width - margin_x, frame_height - margin_y)
    cv2.rectangle(frame, p1, p2, SCAN_ZONE_COLOR, 2)
    cv2.putText(frame, "scanning...", (p1[0], max(0, p1[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, SCAN_ZONE_COLOR, 2)
    return frame


def draw_person(frame, person_det):
    p1 = (int(person_det.x1), int(person_det.y1))
    p2 = (int(person_det.x2), int(person_det.y2))
    cv2.rectangle(frame, p1, p2, PERSON_COLOR, 2)
    label = f"person ({person_det.confidence:.2f})"
    cv2.putText(frame, label, (p1[0], max(0, p1[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PERSON_COLOR, 2)
    return frame


def draw_object(frame, obj_det):
    waste_label = COCO_TO_WASTE_CLASS.get(obj_det.class_name, DEFAULT_WASTE_CLASS)
    p1 = (int(obj_det.x1), int(obj_det.y1))
    p2 = (int(obj_det.x2), int(obj_det.y2))
    cv2.rectangle(frame, p1, p2, OBJECT_COLOR, 2)
    label = f"{waste_label} [{obj_det.class_name} {obj_det.confidence:.2f}]"
    cv2.putText(frame, label, (p1[0], max(0, p1[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, OBJECT_COLOR, 2)
    return frame


def render_frame(frame, detections, frame_width, frame_height):
    """
    Applies the display rule:
        object(s) present -> ONLY red box on the closest object
        else person present -> ONLY green box on person
        else -> blue scanning zone
    """
    people = [d for d in detections if d.class_name == "person"]
    objects = [d for d in detections if d.class_name != "person"]

    if objects:
        # Score only the non-person objects, show just the closest one.
        objects = compute_scores(objects, frame_width, frame_height)
        closest = max(objects, key=lambda d: d.score)
        frame = draw_object(frame, closest)
    elif people:
        # If more than one person, just show the largest (closest) one.
        people = compute_scores(people, frame_width, frame_height)
        closest_person = max(people, key=lambda d: d.score)
        frame = draw_person(frame, closest_person)
    else:
        frame = draw_scan_zone(frame, frame_width, frame_height)

    return frame


def main():
    print(f"Loading {MODEL_WEIGHTS} ...")
    model = YOLO(MODEL_WEIGHTS)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {WEBCAM_INDEX}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {frame_width}x{frame_height}")
    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        detections = run_yolo_on_frame(model, frame)
        frame = render_frame(frame, detections, frame_width, frame_height)

        cv2.imshow("Smart Bin - Phase 1 Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()