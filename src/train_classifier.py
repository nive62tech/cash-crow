"""
train_classifier.py (v3)

Trains MobileNetV3-Small on data/classification/{train,val}/<class>/ for
however many classes exist (auto-detected via ImageFolder -- no hardcoded
class count, so this works for the 10-class scheme without edits).

Carries forward from v2:
  - class-weighted CrossEntropyLoss (inverse frequency) to counter imbalance
  - RandomResizedCrop(224, scale=(0.7, 1.0)) instead of plain Resize
  - wider ColorJitter (0.3) for lighting variation
  - checkpoint saved as {"model_state_dict":..., "class_names": [...]} so
    classifier.py never has to guess class order

Run from the project root:
    python src\\train_classifier.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---- Config -----------------------------------------------------------------

DATA_DIR = Path("data/classification")
MODEL_OUT = Path("models/mobilenetv3_waste_v3.pth")
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Val transform MUST exactly match what classifier.py uses at inference time.
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes: int):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(dataset: datasets.ImageFolder, num_classes: int):
    counts = [0] * num_classes
    for _, label in dataset.samples:
        counts[label] += 1

    total = sum(counts)
    weights = []
    for c in counts:
        if c == 0:
            # Class has 0 images (shouldn't happen if split_dataset.py warned
            # you and you fixed it -- but don't divide by zero if it slipped through)
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * c))

    print("Class counts and weights:")
    for name, count, weight in zip(dataset.classes, counts, weights):
        print(f"  {name}: {count} images, weight={weight:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        print(f"ERROR: {train_dir} or {val_dir} not found. Run split_dataset.py first.")
        return

    train_tf, val_tf = build_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)

    class_names = train_dataset.classes  # alphabetical, ImageFolder default
    num_classes = len(class_names)
    print(f"Training on {num_classes} classes: {class_names}\n")

    if train_dataset.classes != val_dataset.classes:
        print("ERROR: train and val class folders don't match. Re-run split_dataset.py.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(num_classes).to(DEVICE)

    class_weights = compute_class_weights(train_dataset, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        print(f"Epoch {epoch}/{EPOCHS}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "class_names": class_names},
                MODEL_OUT,
            )
            print(f"  -> new best ({val_acc:.4f}), saved to {MODEL_OUT}")

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {MODEL_OUT}")

    # Also dump class list to json for quick reference without loading the checkpoint
    with open(MODEL_OUT.with_suffix(".classes.json"), "w") as f:
        json.dump(class_names, f, indent=2)


if __name__ == "__main__":
    main()