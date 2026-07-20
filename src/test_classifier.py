"""
Quick sanity check for classifier.py -- run this BEFORE integrating
into detect.py, to confirm the model loads and predicts correctly on
a known image.
"""

from classifier import predict
from PIL import Image

# Point this at any real image from your val set to sanity-check
# predictions against a known label.
test_image_path = "data/classification/val/Plastic/Plastic_108.jpg"

image = Image.open(test_image_path).convert("RGB")
label, confidence = predict(image)

print(f"Predicted: {label} (confidence: {confidence:.3f})")