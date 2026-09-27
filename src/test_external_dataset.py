"""
test_external_dataset.py

Tests the trained model against ANY external test folder, auto-detecting
whether it's labeled or unlabeled:

  - LABELED (has subfolders like test_data/plastic/*.jpg, test_data/paper/*.jpg):
    runs full per-class precision/recall/f1, same as evaluate_classifier.py.

  - UNLABELED (just test_data/*.jpg directly, no subfolders):
    runs predict() on every image, prints the predicted label + confidence
    for each file, and a summary count of how many landed in each class.

Usage:
    python src\\test_external_dataset.py <path_to_test_folder>

Example:
    python src\\test_external_dataset.py D:\\Downloads\\teammate_test_set
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = Path("models/mobilenetv3_waste_v3.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names


def predict_image(model, class_names, transform, image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return class_names[idx.item()], conf.item()


def is_labeled(test_root: Path) -> bool:
    """True if test_root contains subfolders with images (labeled),
    False if it contains images directly (unlabeled)."""
    subdirs = [p for p in test_root.iterdir() if p.is_dir()]
    return len(subdirs) > 0


def run_labeled(model, class_names, transform, test_root: Path):
    from sklearn.metrics import classification_report, confusion_matrix

    label_to_idx = {name: i for i, name in enumerate(class_names)}
    all_preds, all_labels, unknown_folders = [], [], []

    for class_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        folder_name = class_dir.name
        if folder_name not in label_to_idx:
            unknown_folders.append(folder_name)
            continue
        true_idx = label_to_idx[folder_name]

        images = [p for p in class_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        for img_path in images:
            pred_label, _ = predict_image(model, class_names, transform, img_path)
            all_preds.append(label_to_idx.get(pred_label, -1))
            all_labels.append(true_idx)

    if unknown_folders:
        print(f"WARNING: these test folders don't match any trained class "
              f"name and were skipped: {unknown_folders}")
        print(f"Trained class names are: {class_names}")
        print("If these are just named differently (e.g. 'shoes' vs "
              "'footwear'), rename the test folders to match and re-run.\n")

    if not all_labels:
        print("ERROR: no test images matched any known class. Nothing to evaluate.")
        return

    print("=== Per-class report (external test set) ===")
    present_classes = sorted(set(all_labels) | set(p for p in all_preds if p >= 0))
    present_names = [class_names[i] for i in present_classes]
    print(classification_report(all_labels, all_preds, labels=present_classes,
                                 target_names=present_names, digits=3, zero_division=0))


def run_unlabeled(model, class_names, transform, test_root: Path):
    images = [p for p in test_root.iterdir()
              if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

    if not images:
        print(f"ERROR: no images found directly in {test_root}.")
        return

    counts = {name: 0 for name in class_names}
    print(f"Running predictions on {len(images)} images...\n")

    for img_path in sorted(images):
        pred_label, conf = predict_image(model, class_names, transform, img_path)
        counts[pred_label] += 1
        print(f"{img_path.name}: {pred_label} ({conf:.2%})")

    print("\n=== Summary ===")
    for name, count in counts.items():
        print(f"  {name}: {count}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src\\test_external_dataset.py <path_to_test_folder>")
        return

    test_root = Path(sys.argv[1])
    if not test_root.exists():
        print(f"ERROR: {test_root} not found.")
        return

    model, class_names = load_model()
    print(f"Loaded model with {len(class_names)} classes: {class_names}\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if is_labeled(test_root):
        print(f"Detected LABELED test set (subfolders found) at {test_root}\n")
        run_labeled(model, class_names, transform, test_root)
    else:
        print(f"Detected UNLABELED test set (flat folder) at {test_root}\n")
        run_unlabeled(model, class_names, transform, test_root)


if __name__ == "__main__":
    main()
