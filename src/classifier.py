"""
classifier.py

Phase 2 inference wrapper. Loads the trained MobileNetV3-Small checkpoint
once at import time and exposes predict(cropped_image) -> (label, confidence).

Generalized to however many classes are in the checkpoint's "class_names"
list -- no hardcoded 3-class or 6-class assumption, so this works unchanged
as your class list grows (currently 10: Plastic, Paper, Glass, Metal,
Organic, Cloth, Shoes, Hazard, Non-Recyclable, None).

IMPORTANT: preprocessing here MUST exactly match train_classifier.py's val
transform (Resize 224x224, ToTensor, Normalize ImageNet mean/std). A
mismatch here causes silently-wrong predictions -- this bit us before,
don't reintroduce it.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ---- Config -------------------------------------------------------------

MODEL_PATH = Path("models/mobilenetv3_waste_v3.pth")
CONFIDENCE_THRESHOLD = 0.4  # ASSUMPTION -- original doc never specified this
                             # exact number for the classifier (only that
                             # sub-threshold predictions return "Uncertain").
                             # Tune this against your live testing.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEBUG_SAVE_CROPS = True
DEBUG_CROPS_DIR = Path("debug_crops")

# ---- Load checkpoint once at import time --------------------------------

_checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_NAMES = _checkpoint["class_names"]
NUM_CLASSES = len(CLASS_NAMES)

_model = models.mobilenet_v3_small(weights=None)
_in_features = _model.classifier[-1].in_features
_model.classifier[-1] = nn.Linear(_in_features, NUM_CLASSES)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model.to(DEVICE)
_model.eval()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if DEBUG_SAVE_CROPS:
    DEBUG_CROPS_DIR.mkdir(exist_ok=True)

_debug_crop_counter = 0

print(f"[classifier] Loaded {MODEL_PATH} with {NUM_CLASSES} classes: {CLASS_NAMES}")


def predict(cropped_image):
    """
    cropped_image: a PIL Image (or numpy array convertible to one) of the
                   cropped held-object region from detect.py.

    Returns: (label: str, confidence: float)
             label is "Uncertain" if top confidence < CONFIDENCE_THRESHOLD.
    """
    global _debug_crop_counter

    if not isinstance(cropped_image, Image.Image):
        cropped_image = Image.fromarray(cropped_image)

    if DEBUG_SAVE_CROPS:
        _debug_crop_counter += 1
        debug_path = DEBUG_CROPS_DIR / f"crop_{_debug_crop_counter:05d}.png"
        cropped_image.save(debug_path)

    input_tensor = _transform(cropped_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = _model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_confidence, top_index = torch.max(probabilities, dim=0)
    top_confidence = top_confidence.item()
    raw_label = CLASS_NAMES[top_index.item()]

    # Full per-class breakdown, sorted high to low -- this was essential for
    # real diagnosis before, keep it.
    breakdown = sorted(
        zip(CLASS_NAMES, probabilities.tolist()), key=lambda x: x[1], reverse=True
    )
    breakdown_str = ", ".join(f"{name}={prob:.2f}" for name, prob in breakdown)
    print(f"[classifier] {breakdown_str} -> raw_pred={raw_label} ({top_confidence:.2f})")

    if top_confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain", top_confidence

    return raw_label, top_confidence