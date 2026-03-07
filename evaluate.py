"""
evaluate.py
===========
Evaluate a trained DeepShield model on the test set.

Outputs:
  - Classification report (precision, recall, F1, accuracy)
  - Confusion matrix (saved as PNG)
  - ROC curve (saved as PNG)
  - Per-class accuracy breakdown

Usage:
  python evaluate.py --model_path models/deepfake_model.pth \
                     --model_name resnet50 \
                     --test_dir dataset/split/test
"""

import os
import sys

# ── Path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils.preprocessing import get_val_transforms, DeepfakeDataset
from torch.utils.data import DataLoader
from train import get_model


CLASS_NAMES = ["Real", "Fake"]


# ─────────────────────────────────────────────
# 1. Load Model
# ─────────────────────────────────────────────
def load_model(model_path: str, model_name: str, device: str) -> torch.nn.Module:
    model = get_model(model_name, device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model


# ─────────────────────────────────────────────
# 2. Run Inference on Test Set
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, test_loader, device):
    """
    Returns:
        all_preds  : numpy array of predicted classes
        all_labels : numpy array of true classes
        all_probs  : numpy array [N, 2] of class probabilities
    """
    all_preds  = []
    all_labels = []
    all_probs  = []

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return (np.array(all_preds),
            np.array(all_labels),
            np.vstack(all_probs))


# ─────────────────────────────────────────────
# 3. Print Metrics
# ─────────────────────────────────────────────
def print_metrics(preds, labels, probs):
    print("\n" + "="*55)
    print("  DeepShield — Evaluation Metrics")
    print("="*55)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", pos_label=1)
    rec  = recall_score(labels, preds, average="binary", pos_label=1)
    f1   = f1_score(labels, preds, average="binary", pos_label=1)

    print(f"\n  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%  (Fake class)")
    print(f"  Recall    : {rec*100:.2f}%  (Fake class)")
    print(f"  F1-Score  : {f1*100:.2f}%  (Fake class)")

    print("\n" + "-"*55)
    print("  Detailed Classification Report")
    print("-"*55)
    print(classification_report(labels, preds,
                                 target_names=CLASS_NAMES, digits=4))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────
# 4. Plot Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(preds, labels, save_path="models/confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DeepShield — Confusion Matrix", fontsize=14, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5, linecolor="grey")
    axes[0].set_title("Raw Counts")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5, linecolor="grey",
                vmin=0, vmax=1)
    axes[1].set_title("Normalised")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


# ─────────────────────────────────────────────
# 5. Plot ROC Curve
# ─────────────────────────────────────────────
def plot_roc_curve(labels, probs, save_path="models/roc_curve.png"):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1565C0", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    plt.fill_between(fpr, tpr, alpha=0.15, color="#1565C0")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("DeepShield — ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved:         {save_path}")
    print(f"  AUC-ROC: {roc_auc:.4f}")


# ─────────────────────────────────────────────
# 6. Model Comparison Utility
# ─────────────────────────────────────────────
def compare_models(results: dict, save_path="models/model_comparison.png"):
    """
    Bar chart comparing ResNet50 vs EfficientNet-B0 metrics.
    results = {
        "ResNet50"       : {"accuracy": .., "precision": .., "recall": .., "f1": ..},
        "EfficientNet-B0": {...}
    }
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#1565C0", "#E53935"]

    for i, (model_name, scores) in enumerate(results.items()):
        vals = [scores[m] * 100 for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x + width/2)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison: ResNet50 vs EfficientNet-B0", fontsize=13)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison chart: {save_path}")


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DeepShield model")
    p.add_argument("--model_path",  default="models/deepfake_model.pth")
    p.add_argument("--model_name",  default="resnet50",
                   choices=["resnet50", "efficientnet"])
    p.add_argument("--test_dir",    default="dataset/split/test")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  default="models")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(args.model_path, args.model_name, device)

    # Build test loader
    test_ds = DeepfakeDataset(args.test_dir, transform=get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Inference
    preds, labels, probs = run_inference(model, test_loader, device)

    # Print metrics
    metrics = print_metrics(preds, labels, probs)

    # Save plots
    plot_confusion_matrix(preds, labels,
                          save_path=os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_roc_curve(labels, probs,
                   save_path=os.path.join(args.output_dir, "roc_curve.png"))

    print("\n✓ Evaluation complete.")
