"""Shared metric computation and plotting helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def plot_curves(history: dict, title: str, out_path: Path) -> Path:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="train loss", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="validation loss", lw=2)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Loss curve"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(epochs, history["train_acc"], label="train accuracy", lw=2)
    axes[1].plot(epochs, history["val_acc"], label="validation accuracy", lw=2)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy curve"); axes[1].legend(); axes[1].grid(alpha=.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion(cm: dict, title: str, out_path: Path) -> Path:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(mat, cmap="Blues")
    labels = ["<=50K (0)", ">50K (1)"]
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    total = mat.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}\n({100*mat[i, j]/total:.1f}%)",
                    ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def plot_roc(curves: dict, title: str, out_path: Path) -> Path:
    """curves: {label: (y_true, y_prob)}"""
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for label, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc_score(y_true, y_prob):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def load_splits():
    d = np.load(C.DATA_PROC / "processed_splits.npz")
    return (d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"])
