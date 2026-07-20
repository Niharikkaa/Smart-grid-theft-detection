"""Evaluation metrics, curves, confusion matrix and ROC curve for Notebook 6."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class Metrics(NamedTuple):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return (
            f"Accuracy  : {self.accuracy:.4f}\n"
            f"Precision : {self.precision:.4f}\n"
            f"Recall    : {self.recall:.4f}\n"
            f"F1 Score  : {self.f1:.4f}\n"
            f"ROC-AUC   : {self.roc_auc:.4f}"
        )


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    """Compute the standard binary classification metric suite."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_prob = np.asarray(y_pred_prob).reshape(-1)
    y_pred = (y_pred_prob >= threshold).astype(int)

    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_pred_prob),
    )


def print_classification_report(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> None:
    """Print sklearn's per-class precision/recall/F1 classification report."""
    y_pred = (np.asarray(y_pred_prob).reshape(-1) >= threshold).astype(int)
    print(classification_report(y_true, y_pred, target_names=["Honest (0)", "Theft (1)"]))


def plot_training_curves(history, save_path: Path | None = None) -> None:
    """Plot training/validation accuracy and loss curves side by side."""
    hist = history
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(hist["accuracy"], label="Train")
    axes[0].plot(hist["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False)

    axes[1].plot(hist["loss"], label="Train")
    axes[1].plot(hist["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Crossentropy")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5, save_path: Path | None = None) -> None:
    """Plot a labeled confusion matrix heatmap."""
    y_pred = (np.asarray(y_pred_prob).reshape(-1) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Honest (0)", "Theft (1)"])
    ax.set_yticklabels(["Honest (0)", "Theft (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray, save_path: Path | None = None) -> None:
    """Plot the ROC curve with the AUC in the legend."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curve(
    y_true,
    y_pred_prob,
    save_path=None,
):
    precision, recall, _ = precision_recall_curve(
        y_true,
        y_pred_prob,
    )

    fig, ax = plt.subplots(figsize=(5.5,5))

    ax.plot(
        recall,
        precision,
        linewidth=2,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    plt.show()