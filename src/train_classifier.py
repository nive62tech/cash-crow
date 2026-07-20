"""
train_classifier.py

Fine-tunes MobileNetV3-Small (pre-trained on ImageNet) to classify
cropped waste images into your 3 classes: Plastic, Paper, Other.

WHY MOBILENETV3-SMALL, AND WHY THIS IS DIFFERENT FROM detect.py:
    detect.py (YOLOv8n) answers "where is an object in this frame".
    This script trains a SEPARATE, second model that answers "given
    just a cropped picture of one object, what material is it". In the
    final pipeline: YOLO finds the box -> the box is cropped out of the
    frame -> that crop is fed into THIS trained model -> it outputs
    Plastic/Paper/Other. MobileNetV3-Small is used here specifically
    because it's small/fast enough to run on a Raspberry Pi on just the
    small cropped region, without needing to re-run over the whole frame.

BEFORE RUNNING THIS:
    Run split_dataset.py first -- this script expects data already
    organized into data/classification/train/<class>/ and
    data/classification/val/<class>/, which that script creates for you.

Run:
    python train_classifier.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# --- Config -------------------------------------------------------------
DATA_DIR = Path("../data/classification")
MODEL_OUT_DIR = Path("../models")
MODEL_OUT_PATH = MODEL_OUT_DIR / "mobilenetv3_waste.pth"

IMAGE_SIZE = 224           # MobileNetV3's expected input size
BATCH_SIZE = 8             # lowered from 16 -- you hit an out-of-memory
                            # crash on CPU; lower this further (e.g. 4) if
                            # it still happens
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 3            # Plastic, Paper, Other
# --------------------------------------------------------------------------


def build_dataloaders():
    # Training gets light augmentation (flips/rotation/color jitter) so the
    # model doesn't just memorize your exact photos -- real trash won't
    # always appear at the same angle/lighting as your training images.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation should NOT be augmented -- we want to measure real
    # performance, not performance on artificially altered images.
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

    print(f"Classes found (order matters for inference later!): {train_dataset.classes}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes, device):
    # Load MobileNetV3-Small pretrained on ImageNet (1000 classes), then
    # replace its final classification layer with one sized for OUR
    # 3 classes. This is standard transfer learning: reuse the general
    # visual features the model already learned, only retrain the final
    # decision layer for our specific task.
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, is_training):
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = build_dataloaders()
    model = build_model(NUM_CLASSES, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, is_training=False)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{NUM_EPOCHS} ({elapsed:.1f}s) - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} - "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, MODEL_OUT_PATH)
            print(f"  -> new best val_acc={val_acc:.3f}, saved to {MODEL_OUT_PATH}")

    print(f"\nTraining done. Best val accuracy: {best_val_accuracy:.3f}")
    print(f"Best model saved at: {MODEL_OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()