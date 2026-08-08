"""
classifier.py

Loads teammate's EfficientNet-B0 FP32 TFLite waste classifier and
exposes the same predict(cropped_image) interface as before, so
detect.py does not need to change.

His model outputs 5 classes: plastic, paper, metal, organic_waste, none.
We map these down to our 3 bin compartments:
    plastic        -> Plastic
    paper          -> Paper
    metal          -> Other
    organic_waste  -> Other
    none           -> Uncertain (no recognizable object in the crop)
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- Config ----------------------------------------------------------------
MODEL_PATH = Path("models/waste_classifier_fp32.tflite")
CLASSES_PATH = Path("models/classes.json")
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6
DEBUG_SAVE_CROPS = True
DEBUG_DIR = Path("debug_crops")

# Maps his 5 raw classes -> our 3 bin labels.
CLASS_MAP = {
    "plastic": "Plastic",
    "paper": "Paper",
    "metal": "Other",
    "organic_waste": "Other",
    "none": "Uncertain",
}
# -----------------------------------------------------------------------------

# Load class index -> name mapping from his classes.json
with open(CLASSES_PATH, "r") as f:
    _raw_classes = json.load(f)
    # classes.json is expected to map index -> name, e.g. {"0": "plastic", ...}
    # sorted by index so position 0 = index 0, etc.
    _class_names_raw = [_raw_classes[str(i)] for i in range(len(_raw_classes))]

_interpreter = Interpreter(model_path=str(MODEL_PATH))
_interpreter.allocate_tensors()
_input_details = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()

print(f"[classifier.py] Loaded EfficientNet-B0 TFLite model. Raw classes: {_class_names_raw}")

_debug_counter = 0


def predict(cropped_image):
    """
    Args:
        cropped_image: NumPy array in BGR (OpenCV) or PIL.Image in RGB.

    Returns:
        (label: str, confidence: float)
        label is one of "Plastic", "Paper", "Other", or "Uncertain".
    """
    global _debug_counter

    if isinstance(cropped_image, np.ndarray):
        rgb = cropped_image[:, :, ::-1]  # BGR -> RGB
        pil_image = Image.fromarray(rgb)
    else:
        pil_image = cropped_image

    if DEBUG_SAVE_CROPS:
        DEBUG_DIR.mkdir(exist_ok=True)
        _debug_counter += 1
        pil_image.save(DEBUG_DIR / f"crop_{_debug_counter:04d}.jpg")

    resized = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(resized).astype(np.float32)  # 0-255 range, EfficientNet's
                                                   # built-in Rescaling layer
                                                   # handles normalization
    arr = np.expand_dims(arr, axis=0)  # add batch dim -> (1, 224, 224, 3)

    _interpreter.set_tensor(_input_details[0]["index"], arr)
    _interpreter.invoke()
    output = _interpreter.get_tensor(_output_details[0]["index"])[0]

    predicted_idx = int(np.argmax(output))
    confidence = float(output[predicted_idx])
    raw_label = _class_names_raw[predicted_idx]

    prob_str = ", ".join(f"{name}={p:.2f}" for name, p in zip(_class_names_raw, output))
    print(f"[classifier] {prob_str} -> raw_pred={raw_label} ({confidence:.2f})")

    mapped_label = CLASS_MAP.get(raw_label, "Uncertain")

    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain", confidence

    return mapped_label, confidence