"""Evaluation utilities for SD-MKD experiments.

This module intentionally stays lightweight and framework-agnostic so it can be
reused by CLI scripts or notebooks for quick validation runs.
"""

from __future__ import annotations

from typing import Tuple

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_model(model: torch.nn.Module, data_loader, device: torch.device) -> Tuple[float, float, torch.Tensor]:
    """Compute accuracy, weighted F1, and confusion matrix for a classifier.

    Args:
        model: A torch.nn.Module returning class logits.
        data_loader: Iterable yielding (inputs, labels).
        device: Torch device for evaluation.

    Returns:
        Tuple of (accuracy, weighted_f1, confusion_matrix_tensor).
    """

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds_t = torch.cat(all_preds)
    all_labels_t = torch.cat(all_labels)
    preds_np = all_preds_t.numpy()
    labels_np = all_labels_t.numpy()

    acc = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average="weighted")
    cm = confusion_matrix(labels_np, preds_np)
    return acc, f1, torch.as_tensor(cm)


def summarize_metrics(acc: float, f1: float, cm: torch.Tensor) -> str:
    """Create a compact, human-readable summary string for evaluation metrics."""

    return (
        f"Accuracy: {acc:.4f}\n"
        f"Weighted F1: {f1:.4f}\n"
        f"Confusion matrix:\n{cm}"
    )


def evaluate_saved_student(checkpoint_path: str, num_classes: int, device: torch.device, data_loader) -> Tuple[float, float, torch.Tensor]:
    """Load a saved student checkpoint and evaluate it on a data loader."""

    from student_model import StudentNet  # Imported here to avoid heavy deps at module load.

    model = StudentNet(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    return evaluate_model(model, data_loader, device)


def plot_training_curves(log_dict, save_path: str | None = None):
    """Placeholder helper for visualizing training/validation curves.

    Expects a dictionary like:
        {
            "train_loss": [...],
            "val_loss": [...],
            "val_acc": [...],
        }
    Only renders if matplotlib is available; otherwise it prints a notice.
    """

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping curve plotting")
        return

    keys = [k for k in ("train_loss", "val_loss", "val_acc") if k in log_dict]
    if not keys:
        print("No known keys found in log_dict; nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(next(iter(log_dict.values()), [])) + 1)
    for key in keys:
        ax.plot(epochs, log_dict[key], label=key)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.set_title("Training curves")
    ax.grid(True, linestyle="--", alpha=0.4)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
