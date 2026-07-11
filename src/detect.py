"""
detect.py

Phase 1 demo: runs YOLOv8n (pre-trained COCO weights, no custom training)
on a laptop webcam feed, and uses roi_selector.py to guess which detected
object is closest to the camera ("the waste item being thrown away").

Controls:
    q  - quit

Notes:
- This does NOT classify into Plastic/Paper/Other yet. That mapping comes
  in a later phase once we have a custom-trained model. For now every
  detection just shows its raw COCO class name.
- COCO has no "paper trash" class. For testing the distance heuristic,
  a good substitute pairing is:
      - "book"   standing in for paper, held close to the camera
      - "bottle" standing in for plastic, placed farther in the background
"""

import cv2
from ultralytics import YOLO

from roi_selector import Detection, compute_scores

# --- Config ---------------------------------------------------------------
MODEL_WEIGHTS = "yolov8n.pt"     # auto-downloads on first run
CONFIDENCE_THRESHOLD = 0.25      # lowered so weak/uncertain guesses still show up
WEBCAM_INDEX = 0                 # change if you have multiple cameras
DEBUG_PRINT_ALL_RAW = True       # prints every raw guess, even below threshold

ROI_COLOR = (0, 255, 0)          # green = the closest object ("waste")
BACKGROUND_COLOR = (0, 0, 255)   # red = everything else, ignored
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
        # Prints EVERY guess YOLO made this frame, even ones below threshold.
        # If an object shows up here with a class name, YOLO "saw" something
        # in that shape. If nothing prints at all, YOLO found nothing
        # resembling ANY of its 80 known classes in the frame.
        print("raw guesses this frame:", raw_seen if raw_seen else "(none)")

    return detections


def draw_detections(frame, detections):
    """Draws each detection's box, colored green if it's the ROI, red otherwise."""
    for det in detections:
        color = ROI_COLOR if det.is_roi else BACKGROUND_COLOR
        label_prefix = "CLOSEST" if det.is_roi else "background"

        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, color, 2)

        label = f"{label_prefix}: {det.class_name} ({det.confidence:.2f}) score={det.score:.3f}"
        cv2.putText(frame, label, (p1[0], max(0, p1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
        detections = compute_scores(detections, frame_width, frame_height)
        frame = draw_detections(frame, detections)

        cv2.imshow("Smart Bin - Phase 1 Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()