"""Step 2 - Train and fully evaluate the baseline deep feed-forward network."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from evaluate import (classification_metrics, load_splits, plot_confusion,
                      plot_curves, plot_roc)
from model import MLP, TrainConfig, architecture_table, predict_proba, train_model

BASELINE = TrainConfig(
    name="baseline",
    hidden_sizes=(128, 64),
    dropout=0.3,
    batch_norm=False,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=256,
    max_epochs=100,
    early_stopping=True,
    patience=10,
    note="Two hidden layers, Adam, BCEWithLogitsLoss, early stopping on validation loss.",
)


def main() -> None:
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

    model, res = train_model(BASELINE, X_train, y_train, X_val, y_val, verbose=True)

    arch = architecture_table(model, X_train.shape[1])
    val_metrics = classification_metrics(y_val, predict_proba(model, X_val))
    test_prob = predict_proba(model, X_test)
    test_metrics = classification_metrics(y_test, test_prob)

    plot_curves(res.history, "Step 2 - Baseline network training curves",
                C.FIGURES / "step2_curves.png")
    plot_confusion(test_metrics["confusion_matrix"],
                   "Step 2 - Baseline confusion matrix (test set)",
                   C.FIGURES / "step2_confusion_matrix.png")
    plot_roc({"Baseline MLP": (y_test, test_prob)},
             "Step 2 - ROC curve (test set)", C.FIGURES / "step2_roc.png")

    torch.save({"state_dict": model.state_dict(), "config": BASELINE.as_dict(),
                "in_features": int(X_train.shape[1])},
               C.ARTIFACTS / "baseline_model.pt")

    out = {
        "config": BASELINE.as_dict(),
        "architecture_table": arch,
        "total_trainable_parameters": res.n_params,
        "epochs_run": res.epochs_run,
        "best_epoch": res.best_epoch,
        "train_time_sec": res.train_time_sec,
        "history": res.history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (C.RESULTS / "step2_metrics.json").write_text(json.dumps(out, indent=2))

    # Markdown summary
    L = ["# Step 2 - Baseline Deep Feed-Forward Network", "",
         "## Architecture", "",
         "| Layer | Type | Output units | Activation | Parameters |", "|---|---|---|---|---|"]
    for r in arch:
        L.append(f"| {r['layer']} | {r['type']} | {r['output_units']} | "
                 f"{r['activation']} | {r['params']:,} |")
    L += ["", f"**Total trainable parameters: {res.n_params:,}**", "",
          f"Loss: BCEWithLogitsLoss | Optimizer: Adam(lr={BASELINE.lr}) | "
          f"Batch size: {BASELINE.batch_size} | Max epochs: {BASELINE.max_epochs} | "
          f"Early stopping patience: {BASELINE.patience}", "",
          f"Trained {res.epochs_run} epochs in {res.train_time_sec:.1f}s; "
          f"best validation loss at epoch {res.best_epoch}.", "",
          "## Test-set performance", "",
          "| Metric | Value |", "|---|---|"]
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        L.append(f"| {k} | {test_metrics[k]:.4f} |")
    cm = test_metrics["confusion_matrix"]
    L += ["", "## Confusion matrix (test)", "",
          "| | Predicted <=50K | Predicted >50K |", "|---|---|---|",
          f"| **Actual <=50K** | {cm['tn']:,} (TN) | {cm['fp']:,} (FP) |",
          f"| **Actual >50K** | {cm['fn']:,} (FN) | {cm['tp']:,} (TP) |", ""]
    (C.RESULTS / "step2_metrics.md").write_text("\n".join(L) + "\n")

    print("\nTest metrics:")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:10s} {test_metrics[k]:.4f}")
    print(f"  confusion  {cm}")
    print(f"Total trainable parameters: {res.n_params:,}")
    print("Step 2 complete.")


if __name__ == "__main__":
    main()
