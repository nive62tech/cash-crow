"""
classifier.py

Loads the trained MobileNetV3-Small waste classifier once at import/
startup time, and exposes a simple predict(cropped_image) function.

This mirrors the exact architecture and preprocessing used in
train_classifier.py, so predictions here match what was measured
during training/validation.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import numpy as np

# --- Config (must match train_classifier.py) -----------------------------
MODEL_PATH = Path("models/mobilenetv3_waste.pth")
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6   # tweak this later based on live testing
DEBUG_SAVE_CROPS = True      # saves every crop fed to the model, for inspection
DEBUG_DIR = Path("debug_crops")
# --------------------------------------------------------------------------

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same preprocessing as val_transform in train_classifier.py
_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_model(num_classes):
    # Same architecture as build_model() in train_classifier.py, but we
    # don't need ImageNet pretrained weights here -- we're about to
    # overwrite everything with our own trained checkpoint anyway.
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _load_checkpoint():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {MODEL_PATH.resolve()}. "
            f"Make sure you're running from the project root."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=_device)
    class_names = checkpoint["class_names"]

    model = _build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()  # inference mode: disables dropout etc.

    return model, class_names


# Loaded once at import time, not on every predict() call.
_model, _class_names = _load_checkpoint()
print(f"[classifier.py] Loaded model. Classes: {_class_names}")

_debug_counter = 0


def predict(cropped_image):
    """
    Args:
        cropped_image: a cropped region of the frame containing the held
            object. Accepts either:
              - a NumPy array in BGR format (as OpenCV gives you from
                frame[y1:y2, x1:x2]), or
              - a PIL.Image in RGB.

    Returns:
        (label: str, confidence: float)
        label is one of _class_names, or "Uncertain" if confidence is
        below CONFIDENCE_THRESHOLD.
    """
    global _debug_counter

    # Convert OpenCV BGR numpy array -> PIL RGB image
    if isinstance(cropped_image, np.ndarray):
        rgb = cropped_image[:, :, ::-1]  # BGR -> RGB
        pil_image = Image.fromarray(rgb)
    else:
        pil_image = cropped_image

    if DEBUG_SAVE_CROPS:
        DEBUG_DIR.mkdir(exist_ok=True)
        _debug_counter += 1
        pil_image.save(DEBUG_DIR / f"crop_{_debug_counter:04d}.jpg")

    tensor = _preprocess(pil_image).unsqueeze(0).to(_device)  # add batch dim

    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probs, dim=0)

    confidence = confidence.item()
    label = _class_names[predicted_idx.item()]

    # Full breakdown -- shows whether the model is confidently wrong or
    # genuinely torn between classes.
    prob_str = ", ".join(f"{name}={p.item():.2f}" for name, p in zip(_class_names, probs))
    print(f"[classifier] {prob_str} -> raw_pred={label} ({confidence:.2f})")

    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain", confidence

    return label, confidence