"""
train.py
========
Training pipeline for DeepShield deepfake detector.

Model: ResNet50 or EfficientNet-B0 (pretrained on ImageNet)
       Final FC layer replaced with 2-class head.
       Dropout added for regularisation.

Optimizer : Adam (lr=0.0001)
Loss      : CrossEntropyLoss
Schedule  : ReduceLROnPlateau
Early Stop: Patience=5 epochs
Epochs    : Up to 15

Usage:
  python train.py --model resnet50 --data_dir dataset/split --epochs 15
  python train.py --model efficientnet --data_dir dataset/split --epochs 15
"""

import os
import sys

# ── Path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works on Colab/server)
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.preprocessing import get_dataloaders


# ─────────────────────────────────────────────
# 1. Model Builders
# ─────────────────────────────────────────────
def build_resnet50(num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    """
    ResNet50 pretrained on ImageNet.
    Replace final FC with Dropout + Linear(2048 → num_classes).
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all layers first (feature extraction phase)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully-connected layer
    in_features = model.fc.in_features        # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )

    # Unfreeze last residual block + new head for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def build_efficientnet_b0(num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet.
    Replace classifier head with Dropout + Linear.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    # EfficientNet classifier: [Dropout, Linear(1280, 1000)]
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout / 2),
        nn.Linear(256, num_classes),
    )

    # Unfreeze last feature block + classifier
    for param in model.features[-1].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def get_model(model_name: str, device: str) -> nn.Module:
    if model_name == "resnet50":
        model = build_resnet50()
    elif model_name in ("efficientnet", "efficientnet_b0"):
        model = build_efficientnet_b0()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'resnet50' or 'efficientnet'.")

    return model.to(device)


# ─────────────────────────────────────────────
# 2. Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    Saves the best model weights automatically.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4,
                 save_path: str = "models/deepfake_model.pth"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_path  = save_path
        self.counter    = 0
        self.best_loss  = np.inf
        self.best_weights = None
        self.should_stop  = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss    = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter      = 0
            # Save checkpoint
            os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else ".", exist_ok=True)
            torch.save(self.best_weights, self.save_path)
            print(f"  ✓ Model improved → saved to {self.save_path}")
        else:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                print("  ⚑ Early stopping triggered.")

    def restore_best(self, model: nn.Module):
        if self.best_weights:
            model.load_state_dict(self.best_weights)


# ─────────────────────────────────────────────
# 3. One Epoch of Training
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    pbar = tqdm(loader, desc="  Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler:   # AMP mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{100*correct/total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ─────────────────────────────────────────────
# 4. Validation Loop
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────
# 5. Plot Training Curves
# ─────────────────────────────────────────────
def plot_training_history(history: dict, save_path: str = "models/training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DeepShield Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history["train_acc"], label="Train Accuracy", color="#1565C0", lw=2)
    axes[0].plot(history["val_acc"],   label="Val Accuracy",   color="#E53935", lw=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history["train_loss"], label="Train Loss", color="#1565C0", lw=2)
    axes[1].plot(history["val_loss"],   label="Val Loss",   color="#E53935", lw=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training plot saved: {save_path}")


# ─────────────────────────────────────────────
# 6. Main Training Loop
# ─────────────────────────────────────────────
def train(args):
    # ── Device setup ──────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  DeepShield Training Pipeline")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device.upper()}")
    print(f"  Epochs : {args.epochs}")
    print(f"  LR     : {args.lr}")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        train_dir  = os.path.join(args.data_dir, "train"),
        val_dir    = os.path.join(args.data_dir, "val"),
        test_dir   = os.path.join(args.data_dir, "test"),
        batch_size = args.batch_size,
        num_workers= args.num_workers,
    )

    # ── Model ─────────────────────────────────────
    model = get_model(args.model, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    # ── Loss + Optimizer ──────────────────────────
    # Compute class weights to handle imbalance
    real_count = sum(1 for _, l in train_loader.dataset.samples if l == 0)
    fake_count = len(train_loader.dataset.samples) - real_count
    total      = real_count + fake_count
    weights    = torch.tensor([total/real_count, total/fake_count], dtype=torch.float32).to(device)

    criterion  = nn.CrossEntropyLoss(weight=weights)
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler  = ReduceLROnPlateau(optimizer, mode="min", patience=3,
                                   factor=0.5)
    early_stop = EarlyStopping(patience=5, save_path=args.save_path)

    # AMP scaler (GPU only)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # ── History ───────────────────────────────────
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    # ── Epoch Loop ────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device, scaler)
        v_loss, v_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"  Train → loss: {t_loss:.4f} | acc: {t_acc*100:.2f}%")
        print(f"  Val   → loss: {v_loss:.4f} | acc: {v_acc*100:.2f}%")

        scheduler.step(v_loss)
        early_stop(v_loss, model)

        if early_stop.should_stop:
            print("\n⚑ Early stopping. Restoring best weights.")
            early_stop.restore_best(model)
            break

    # ── Save history & plot ────────────────────────
    history_path = args.save_path.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history,
                          save_path=args.save_path.replace(".pth", "_plot.png"))

    print(f"\n✓ Training complete. Best model: {args.save_path}")
    print(f"  Best val loss : {early_stop.best_loss:.4f}")
    return model, history


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train DeepShield deepfake detector")
    p.add_argument("--model",       default="resnet50",
                   choices=["resnet50", "efficientnet"],
                   help="Model architecture")
    p.add_argument("--data_dir",    default="dataset/split",
                   help="Root dataset directory (must contain train/ val/ test/)")
    p.add_argument("--save_path",   default="models/deepfake_model.pth",
                   help="Where to save trained model")
    p.add_argument("--epochs",      type=int, default=15)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=0.0001)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)