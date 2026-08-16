"""
train_classifier.py (v2 -- addresses Plastic/Paper/Other confusion)

Fine-tunes MobileNetV3-Small to classify cropped waste images into
Plastic, Paper, Other.

WHAT CHANGED FROM v1 AND WHY:
    1. CLASS-WEIGHTED LOSS: the original training treated every image
       equally, but "Other" had 1,355 images vs Plastic's 921 and
       Paper's 961. This biases the model toward predicting the
       majority class whenever it's unsure -- exactly the symptom
       observed live (real Plastic items reading as Other/Paper).
       Weighting the loss makes mistakes on minority classes "cost"
       more during training, forcing the model to actually learn to
       distinguish them instead of defaulting to the safe majority
       guess.
    2. MORE EPOCHS (25 instead of 15): more passes over the data gives
       the model more chances to refine the harder Plastic/Paper
       boundary, since that boundary was clearly still weak at 15
       epochs (92.3% val accuracy overall was masking a much weaker
       per-class result on Plastic specifically).
    3. STRONGER AUGMENTATION: added random crop/zoom, since your real
       webcam crops are less tightly-framed than the clean Kaggle
       training photos. This is a cheap way to make the model more
       tolerant of imperfect real-world framing without needing new
       data.

BEFORE RUNNING THIS:
    Same as before -- data/classification/train and val must already
    exist (they do, from your first run of split_dataset.py).

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
DATA_DIR = Path("data/classification")
MODEL_OUT_DIR = Path("models")
MODEL_OUT_PATH = MODEL_OUT_DIR / "mobilenetv3_waste_v2.pth"  # new filename --
                                                                # keeps your
                                                                # original
                                                                # checkpoint
                                                                # safe as a
                                                                # fallback

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 25           # was 15
LEARNING_RATE = 0.001
NUM_CLASSES = 3
# --------------------------------------------------------------------------


def build_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),  # was
            # a plain Resize -- RandomResizedCrop simulates the loose,
            # off-center framing real webcam crops have, so the model
            # sees more realistic variation during training
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # widened
            # from 0.2 -- real lighting varies more than the clean
            # Kaggle photos did
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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

    return train_loader, val_loader, train_dataset.classes, train_dataset


def compute_class_weights(train_dataset, device):
    """
    Computes inverse-frequency weights so the loss penalizes mistakes
    on under-represented classes more heavily. This directly targets
    the "everything defaults to Other" bias, since Other had ~40% more
    training images than Plastic or Paper.
    """
    from collections import Counter
    counts = Counter(train_dataset.targets)
    total = sum(counts.values())
    num_classes = len(counts)

    weights = []
    for class_idx in range(num_classes):
        class_count = counts[class_idx]
        weight = total / (num_classes * class_count)
        weights.append(weight)

    print(f"Class weights (by index, matches train_dataset.classes order): {weights}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def build_model(num_classes, device):
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

    train_loader, val_loader, class_names, train_dataset = build_dataloaders()
    model = build_model(NUM_CLASSES, device)

    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # was unweighted
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