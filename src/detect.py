"""
detect.py

Phase 2: YOLOv8n (pre-trained COCO weights) + MediaPipe Hands +
MobileNetV3-Small waste classifier.

DISPLAY RULE:
    - Something held in a hand -> RED box(es), labeled with the real
      predicted class (Plastic/Paper/Other) and confidence, smoothed
      over recent frames via majority vote. Stays drawn continuously
      while held, using the last known position during brief
      single-frame detection flicker (see OBJECT_ABSENCE_GRACE_SECONDS
      below) so it doesn't flash on/off.
    - No held object, but a person is visible -> GREEN box for the
      first PERSON_DISPLAY_SECONDS after they're first seen, then
      hidden. It only reappears if the person is fully ABSENT for
      PERSON_ABSENCE_RESET_SECONDS or longer and then reappears --
      ordinary movement within frame (which causes brief detection
      flicker) does NOT count as "a new person" and will NOT bring
      the box back.
    - Nothing at all -> BLUE "scanning" zone for BACKGROUND_DISPLAY_
      SECONDS, same idea.

HOW CLASSIFICATION WORKS (Phase 2):
    YOLOv8n only knows its 80 original COCO classes -- it has no
    concept of "plastic"/"paper"/"trash", it only answers "where is an
    object". A SEPARATE model, MobileNetV3-Small (classifier.py),
    answers "given just this cropped picture, what material is it".
    Whenever a held object is freshly detected, its region is cropped
    from the frame and passed through classifier.py's predict(). Raw
    per-frame predictions are kept in a short rolling history and
    majority-voted, so the displayed label doesn't flip on a single
    noisy frame -- the same debouncing philosophy already used for the
    red/green/blue detection state itself.

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
from collections import deque, Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

from roi_selector import Detection, compute_scores
from classifier import predict

# --- Config ---------------------------------------------------------------
MODEL_WEIGHTS = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
WEBCAM_INDEX = 0
DEBUG_PRINT_ALL_RAW = True
DEBUG_PRINT_STATE = True

MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.5
HAND_REGION_PADDING_RATIO = 0.25   # tightened: was 1.0 -- was catching things
                                     # merely NEAR the hand (like your own face)
                                     # instead of things actually gripped by
                                     # fingers/palm
HAND_REGION_PADDING_MIN_PX = 20     # was 80 -- same reason

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

# How many recent frame-level predictions to hold onto for majority-vote
# smoothing, so a single flickery frame can't flip the displayed label.
PREDICTION_HISTORY_SIZE = 5

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

    Also tracks a short rolling history of classifier predictions made
    while in the 'object' state, so the displayed class label can be
    majority-voted across recent frames rather than trusting any single
    frame's raw prediction.
    """
    def __init__(self):
        self.current = None
        self.start_time = None
        self.last_object_seen = None
        self.last_person_seen = None
        self.last_held_objects = []   # cached boxes, refreshed whenever seen
        self.prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)

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
                self.prediction_history.clear()  # don't carry stale votes into next object

        elapsed = now - self.start_time
        return desired, elapsed

    def add_prediction(self, label, confidence):
        """Record one frame's raw classifier prediction for smoothing."""
        self.prediction_history.append((label, confidence))

    def get_smoothed_prediction(self):
        """
        Majority-vote across recent frames' predictions, rather than
        trusting any single frame -- avoids the displayed label flipping
        every time one frame's confidence dips near the threshold.
        Returns (label, avg_confidence_for_that_label) or (None, 0.0)
        if nothing recorded yet.
        """
        if not self.prediction_history:
            return None, 0.0

        labels = [label for label, _ in self.prediction_history]
        winning_label, _ = Counter(labels).most_common(1)[0]

        matching_confidences = [
            conf for label, conf in self.prediction_history if label == winning_label
        ]
        avg_confidence = sum(matching_confidences) / len(matching_confidences)

        return winning_label, avg_confidence


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


def crop_object(frame, obj, padding_ratio=0.1):
    """
    Crops the region of `frame` covered by `obj`'s bounding box, with a
    little extra padding around the edges so the classifier isn't fed
    an overly tight crop that clips the object.
    Returns None if the resulting crop would be empty/invalid.
    """
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


def draw_held_objects(frame, held_objects, label_text="Object"):
    for obj in held_objects:
        p1 = (int(obj.x1), int(obj.y1))
        p2 = (int(obj.x2), int(obj.y2))
        cv2.rectangle(frame, p1, p2, OBJECT_COLOR, 2)
        cv2.putText(frame, label_text, (p1[0], max(0, p1[1] - 10)),
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

        # Only run the classifier on a REAL fresh detection this frame
        # (not on the cached fallback box) -- classifying a stale crop
        # from an old frame position would be wasted work at best and
        # misleading at worst.
        if held_objects:
            # Pick the single highest-scoring held object as "the" item
            # being classified -- scoring already exists from compute_scores().
            primary_obj = max(held_objects, key=lambda d: d.score)
            cropped = crop_object(frame, primary_obj)
            if cropped is not None and cropped.size > 0:
                raw_label, raw_confidence = predict(cropped)
                state.add_prediction(raw_label, raw_confidence)

        smoothed_label, smoothed_confidence = state.get_smoothed_prediction()
        if smoothed_label is None:
            label_text = "Object"
        else:
            label_text = f"{smoothed_label} ({smoothed_confidence:.0%})"

        frame = draw_held_objects(frame, objs_to_draw, label_text)
        drew = f"RED object box x{len(objs_to_draw)} label={label_text}"
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