"""
evaluate_classifier.py

Runs the trained model against data/classification/val/ and prints a
per-class precision/recall/f1 breakdown -- not just overall accuracy.

This matters because overall accuracy hid a real per-class weakness before
(the original Plastic problem was masked by 92.3% overall accuracy). Thin
classes like non-recyclable (332 train images) are the most likely place
for that to happen again.

Run from the project root:
    python src\\evaluate_classifier.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = Path("models/mobilenetv3_waste_v3.pth")
VAL_DIR = Path("data/classification/val")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_tf)

    if val_dataset.classes != class_names:
        print("WARNING: val folder class order doesn't match checkpoint's "
              "class_names. Results below may be mislabeled.")

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("=== Per-class report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    print("=== Confusion matrix (rows=true, cols=predicted) ===")
    cm = confusion_matrix(all_labels, all_preds)
    header = "".join(f"{name[:8]:>10}" for name in class_names)
    print(f"{'':>15}{header}")
    for name, row in zip(class_names, cm):
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"{name[:14]:>15}{row_str}")


if __name__ == "__main__":
    main()
