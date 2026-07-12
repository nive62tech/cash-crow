"""
detect.py

Phase 1 demo: YOLOv8n (pre-trained COCO weights) + MediaPipe Hands.

DISPLAY RULE:
    - Something held in a hand -> RED box(es), labeled "Object". Stays
      drawn continuously while held, using the last known position
      during brief single-frame detection flicker (see OBJECT_ABSENCE_
      GRACE_SECONDS below) so it doesn't flash on/off.
    - No held object, but a person is visible -> GREEN box for the
      first PERSON_DISPLAY_SECONDS after they're first seen, then
      hidden. It only reappears if the person is fully ABSENT for
      PERSON_ABSENCE_RESET_SECONDS or longer and then reappears --
      ordinary movement within frame (which causes brief detection
      flicker) does NOT count as "a new person" and will NOT bring
      the box back.
    - Nothing at all -> BLUE "scanning" zone for BACKGROUND_DISPLAY_
      SECONDS, same idea.

WHY THE RED BOX JUST SAYS "OBJECT":
    YOLOv8n only knows its 80 original COCO classes. It has no concept
    of "plastic"/"paper"/"trash". Real classification needs a custom-
    trained model (Phase 3). This label is intentionally generic.

HOW "HELD BY HAND" IS DECIDED:
    MediaPipe Hand Landmarker finds hand positions each frame. We build
    an (invisible) padded region around each hand and check which YOLO
    object boxes overlap it. Only overlapping objects count as "held".

Controls:
    q  - quit
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

from roi_selector import Detection, compute_scores

# --- Config ---------------------------------------------------------------
MODEL_WEIGHTS = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
WEBCAM_INDEX = 0
DEBUG_PRINT_ALL_RAW = True
DEBUG_PRINT_STATE = True

MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.5
HAND_REGION_PADDING_RATIO = 1.0
HAND_REGION_PADDING_MIN_PX = 80

PERSON_DISPLAY_SECONDS = 2.0        # how long the green box shows for a (new) person
BACKGROUND_DISPLAY_SECONDS = 5.0    # how long the blue scan zone shows

# How long something can go briefly undetected before we treat it as
# truly gone. These are DIFFERENT on purpose:
#   - objects: short grace, just enough to smooth normal frame-to-frame
#     detection flicker while a real object is actively held.
#   - person: much longer, so ordinary movement/turning/brief occlusion
#     within frame is NOT mistaken for "they left and a new person
#     arrived". Only a real, sustained absence counts as "new person".
OBJECT_ABSENCE_GRACE_SECONDS = 1.0
PERSON_ABSENCE_RESET_SECONDS = 4.0

# Colors are OpenCV BGR, not RGB.
PERSON_COLOR = (144, 238, 144)     # light green
OBJECT_COLOR = (0, 0, 255)         # red
SCAN_ZONE_COLOR = (255, 0, 0)      # blue

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
# ---------------------------------------------------------------------------


def ensure_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        print("Downloading hand_landmarker.task (one-time)...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    return HAND_MODEL_PATH


class DisplayState:
    """
    Tracks which of 'object' / 'person' / 'background' is active, how
    long we've been in that state, and (for objects specifically) the
    last known bounding boxes -- so the red box keeps being drawn in
    its last known position during a brief single-frame detection gap,
    instead of flashing off every time YOLO misses one frame.
    """
    def __init__(self):
        self.current = None
        self.start_time = None
        self.last_object_seen = None
        self.last_person_seen = None
        self.last_held_objects = []   # cached boxes, refreshed whenever seen

    def update(self, held_objects, people):
        now = time.time()

        if held_objects:
            self.last_object_seen = now
            self.last_held_objects = held_objects
        if people:
            self.last_person_seen = now

        object_active = (
            self.last_object_seen is not None
            and (now - self.last_object_seen) <= OBJECT_ABSENCE_GRACE_SECONDS
        )
        person_active = (
            self.last_person_seen is not None
            and (now - self.last_person_seen) <= PERSON_ABSENCE_RESET_SECONDS
        )

        if object_active:
            desired = "object"
        elif person_active:
            desired = "person"
        else:
            desired = "background"

        if desired != self.current:
            self.current = desired
            self.start_time = now
            if desired != "object":
                self.last_held_objects = []  # clear cache once we truly leave object state

        elapsed = now - self.start_time
        return desired, elapsed


def run_yolo_on_frame(model, frame):
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


def draw_scan_zone(frame, frame_width, frame_height):
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
    cv2.putText(frame, "person", (p1[0], max(0, p1[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PERSON_COLOR, 2)
    return frame


def draw_held_objects(frame, held_objects):
    for obj in held_objects:
        p1 = (int(obj.x1), int(obj.y1))
        p2 = (int(obj.x2), int(obj.y2))
        cv2.rectangle(frame, p1, p2, OBJECT_COLOR, 2)
        cv2.putText(frame, "Object", (p1[0], max(0, p1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, OBJECT_COLOR, 2)
    return frame


def render_frame(frame, detections, hand_regions, frame_width, frame_height, state):
    people = [d for d in detections if d.class_name == "person"]
    objects = [d for d in detections if d.class_name != "person"]

    objects = compute_scores(objects, frame_width, frame_height) if objects else objects
    held_objects = filter_held_objects(objects, hand_regions)

    desired_state, elapsed = state.update(held_objects, people)

    drew = "nothing"
    if desired_state == "object":
        # Use this frame's detections if we have them, otherwise fall back
        # to the last known position so the box doesn't flash off during
        # a brief single-frame gap.
        objs_to_draw = held_objects if held_objects else state.last_held_objects
        frame = draw_held_objects(frame, objs_to_draw)
        drew = f"RED object box x{len(objs_to_draw)}"
    elif desired_state == "person":
        if elapsed <= PERSON_DISPLAY_SECONDS and people:
            people_scored = compute_scores(people, frame_width, frame_height)
            closest_person = max(people_scored, key=lambda d: d.score)
            frame = draw_person(frame, closest_person)
            drew = f"GREEN person box (elapsed={elapsed:.1f}s)"
        else:
            drew = f"nothing (person state, elapsed={elapsed:.1f}s > {PERSON_DISPLAY_SECONDS}s limit)"
    else:
        if elapsed <= BACKGROUND_DISPLAY_SECONDS:
            frame = draw_scan_zone(frame, frame_width, frame_height)
            drew = f"BLUE scan zone (elapsed={elapsed:.1f}s)"
        else:
            drew = f"nothing (background state, elapsed={elapsed:.1f}s > {BACKGROUND_DISPLAY_SECONDS}s limit)"

    if DEBUG_PRINT_STATE:
        print(f"hands_detected={len(hand_regions)} objects_seen={len(objects)} "
              f"held_objects={len(held_objects)} state={desired_state} -> drew: {drew}")

    return frame


def main():
    print(f"Loading {MODEL_WEIGHTS} ...")
    model = YOLO(MODEL_WEIGHTS)

    hand_model_path = ensure_hand_model()
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model_path),
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {WEBCAM_INDEX}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {frame_width}x{frame_height}")
    print("Press 'q' to quit.")

    display_state = DisplayState()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        detections = run_yolo_on_frame(model, frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_regions = get_hand_regions(hand_result, frame_width, frame_height)

        frame = render_frame(frame, detections, hand_regions, frame_width, frame_height, display_state)

        cv2.imshow("Smart Bin - Phase 1 Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()


if __name__ == "__main__":
    main()