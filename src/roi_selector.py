"""
roi_selector.py

Phase 1 heuristic: given a list of YOLO detections in a single frame,
estimate which one is closest to the camera using ONLY bounding box
geometry (no depth sensor). No ML here — just simple, explainable math.

Heuristic:
    score = size_weight * normalized_area + position_weight * normalized_bottom_y

- normalized_area:  box_area / frame_area          -> bigger box = closer
- normalized_bottom_y: box_bottom_edge / frame_h    -> lower in frame = closer

The detection with the HIGHEST score is treated as the "waste" object
(the ROI). All others are treated as background and ignored.
"""

from dataclasses import dataclass


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    score: float = 0.0          # closeness score, filled in by compute_scores()
    is_roi: bool = False        # True for the single closest detection


def compute_scores(detections, frame_width, frame_height,
                    size_weight=0.6, position_weight=0.4):
    """
    Takes a list of Detection objects (score/is_roi not yet set) and
    returns the same list with .score and .is_roi filled in.

    size_weight + position_weight should sum to 1.0, but this isn't
    enforced strictly in case you want to experiment with the ratio.
    """
    if not detections:
        return detections

    frame_area = float(frame_width * frame_height)

    for det in detections:
        box_w = max(0.0, det.x2 - det.x1)
        box_h = max(0.0, det.y2 - det.y1)
        box_area = box_w * box_h

        normalized_area = box_area / frame_area
        normalized_bottom_y = det.y2 / float(frame_height)

        det.score = (size_weight * normalized_area) + (position_weight * normalized_bottom_y)

    # Mark the single highest-scoring detection as the ROI ("waste" item).
    best = max(detections, key=lambda d: d.score)
    for det in detections:
        det.is_roi = (det is best)

    return detections