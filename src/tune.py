"""Step 3 - Manual hyper-parameter tuning: 10 controlled configurations.

Each configuration changes a small number of factors relative to the baseline so
the effect of depth/width, learning rate, batch size and regularisation can be
attributed rather than guessed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from evaluate import (classification_metrics, load_splits, plot_confusion,
                      plot_curves, plot_roc)
from model import TrainConfig, architecture_table, predict_proba, train_model

# --------------------------------------------------------------------------- #
# The configs I tried. C1 is the same as my Step 2 baseline, used as the reference.
# --------------------------------------------------------------------------- #
CONFIGS = [
    TrainConfig(name="C1-baseline", hidden_sizes=(128, 64), dropout=0.3, lr=1e-3,
                batch_size=256, weight_decay=0.0, batch_norm=False,
                note="Reference: the Step 2 baseline."),
    TrainConfig(name="C2-shallow-narrow", hidden_sizes=(32, 16), dropout=0.3, lr=1e-3,
                batch_size=256, note="Width sweep: much smaller capacity."),
    TrainConfig(name="C3-wide", hidden_sizes=(512, 256), dropout=0.3, lr=1e-3,
                batch_size=256, note="Width sweep: ~10x more parameters."),
    TrainConfig(name="C4-deep-3layer", hidden_sizes=(256, 128, 64), dropout=0.3, lr=1e-3,
                batch_size=256, note="Depth sweep: 3 hidden layers."),
    TrainConfig(name="C5-deep-4layer", hidden_sizes=(256, 128, 64, 32), dropout=0.3, lr=1e-3,
                batch_size=256, note="Depth sweep: 4 hidden layers."),
    TrainConfig(name="C6-lr-high", hidden_sizes=(128, 64), dropout=0.3, lr=1e-2,
                batch_size=256, note="Learning-rate sweep: 10x higher."),
    TrainConfig(name="C7-lr-low", hidden_sizes=(128, 64), dropout=0.3, lr=3e-4,
                batch_size=256, note="Learning-rate sweep: ~3x lower."),
    TrainConfig(name="C8-batch-small", hidden_sizes=(128, 64), dropout=0.3, lr=1e-3,
                batch_size=64, note="Batch-size sweep: 64 (noisier gradients)."),
    TrainConfig(name="C9-batch-large", hidden_sizes=(128, 64), dropout=0.3, lr=1e-3,
                batch_size=1024, note="Batch-size sweep: 1024 (smoother gradients)."),
    TrainConfig(name="C10-no-reg", hidden_sizes=(256, 128), dropout=0.0, lr=1e-3,
                batch_size=256, weight_decay=0.0, early_stopping=False, max_epochs=60,
                note="Regularisation ablation: no dropout, no weight decay, "
                     "no early stopping -> expect overfitting."),
    TrainConfig(name="C11-dropout-heavy", hidden_sizes=(256, 128), dropout=0.5, lr=1e-3,
                batch_size=256, note="Regularisation: heavy dropout p=0.5."),
    TrainConfig(name="C12-batchnorm-wd", hidden_sizes=(256, 128), dropout=0.2, lr=1e-3,
                batch_size=256, weight_decay=1e-4, batch_norm=True,
                note="Regularisation: batch norm + weight decay + light dropout."),
]


def main() -> None:
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    rows, histories, detail = [], {}, {}
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] {cfg.name}: {cfg.note}")
        model, res = train_model(cfg, X_train, y_train, X_val, y_val, verbose=False)
        vm = classification_metrics(y_val, predict_proba(model, X_val))
        tm = classification_metrics(y_test, predict_proba(model, X_test))
        histories[cfg.name] = res.history
        detail[cfg.name] = {"config": cfg.as_dict(), "val_metrics": vm,
                            "test_metrics": tm, "history": res.history,
                            "epochs_run": res.epochs_run, "best_epoch": res.best_epoch,
                            "train_time_sec": res.train_time_sec, "n_params": res.n_params}
        # gap between train and val accuracy shows overfitting
        gap = res.history["train_acc"][-1] - res.history["val_acc"][-1]
        rows.append({
            "config": cfg.name,
            "hidden_layers": len(cfg.hidden_sizes),
            "hidden_sizes": "-".join(map(str, cfg.hidden_sizes)),
            "lr": cfg.lr, "batch_size": cfg.batch_size, "dropout": cfg.dropout,
            "weight_decay": cfg.weight_decay, "batch_norm": cfg.batch_norm,
            "early_stopping": cfg.early_stopping,
            "params": res.n_params,
            "epochs_run": res.epochs_run, "best_epoch": res.best_epoch,
            "train_time_s": res.train_time_sec,
            "val_loss": round(res.best_val_loss, 4),
            "val_accuracy": round(vm["accuracy"], 4),
            "val_f1": round(vm["f1"], 4),
            "val_roc_auc": round(vm["roc_auc"], 4),
            "train_val_acc_gap": round(gap, 4),
            "test_roc_auc": round(tm["roc_auc"], 4),
        })
        print(f"    val acc {vm['accuracy']:.4f} | val F1 {vm['f1']:.4f} | "
              f"val AUC {vm['roc_auc']:.4f} | {res.train_time_sec:.1f}s | "
              f"{res.n_params:,} params")

    df = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
    df.to_csv(C.RESULTS / "step3_tuning.csv", index=False)

    best_name = df.iloc[0]["config"]
    best_cfg = next(c for c in CONFIGS if c.name == best_name)
    print(f"\nBest configuration by validation ROC-AUC: {best_name}")

    # All the configs land within a few ten-thousandths of each other, which is
    # close enough that randomness could explain it. So before I call any factor
    # important, I retrain one config under different seeds to see how much the
    # score moves on its own. Anything smaller than that is not a real effect.
    import numpy as np
    from model import set_seed
    noise_runs = []
    base = CONFIGS[0]
    for seed in (0, 7, 123):
        cfg_s = TrainConfig(**{**base.as_dict(), "hidden_sizes": tuple(base.hidden_sizes),
                               "name": f"noise-seed{seed}"})
        import model as _m
        _orig = _m.C.SEED
        _m.C.SEED = seed
        m_s, r_s = train_model(cfg_s, X_train, y_train, X_val, y_val, verbose=False)
        _m.C.SEED = _orig
        vm_s = classification_metrics(y_val, predict_proba(m_s, X_val))
        noise_runs.append({"seed": seed, "val_roc_auc": vm_s["roc_auc"],
                           "val_accuracy": vm_s["accuracy"], "val_f1": vm_s["f1"]})
        print(f"    noise probe seed={seed}: val AUC {vm_s['roc_auc']:.4f}")
    noise_band = {
        m: float(max(r[m] for r in noise_runs) - min(r[m] for r in noise_runs))
        for m in ("val_roc_auc", "val_accuracy", "val_f1")
    }
    print(f"  seed-only spread: {noise_band}")

    # how much does each factor actually move the score?
    def grp(names):
        return df[df["config"].isin(names)]["val_roc_auc"]

    factor_spread = {
        "learning rate (C1 / C6 / C7)":
            float(grp(["C1-baseline", "C6-lr-high", "C7-lr-low"]).max()
                  - grp(["C1-baseline", "C6-lr-high", "C7-lr-low"]).min()),
        "depth (C1 / C4 / C5)":
            float(grp(["C1-baseline", "C4-deep-3layer", "C5-deep-4layer"]).max()
                  - grp(["C1-baseline", "C4-deep-3layer", "C5-deep-4layer"]).min()),
        "width (C2 / C1 / C3)":
            float(grp(["C2-shallow-narrow", "C1-baseline", "C3-wide"]).max()
                  - grp(["C2-shallow-narrow", "C1-baseline", "C3-wide"]).min()),
        "batch size (C8 / C1 / C9)":
            float(grp(["C8-batch-small", "C1-baseline", "C9-batch-large"]).max()
                  - grp(["C8-batch-small", "C1-baseline", "C9-batch-large"]).min()),
        "regularisation (C10 / C11 / C12)":
            float(grp(["C10-no-reg", "C11-dropout-heavy", "C12-batchnorm-wd"]).max()
                  - grp(["C10-no-reg", "C11-dropout-heavy", "C12-batchnorm-wd"]).min()),
    }
    ranked = sorted(factor_spread.items(), key=lambda kv: kv[1], reverse=True)

    FACTOR_GROUPS = {
        "learning rate (C1 / C6 / C7)": ["C1-baseline", "C6-lr-high", "C7-lr-low"],
        "depth (C1 / C4 / C5)": ["C1-baseline", "C4-deep-3layer", "C5-deep-4layer"],
        "width (C2 / C1 / C3)": ["C2-shallow-narrow", "C1-baseline", "C3-wide"],
        "batch size (C8 / C1 / C9)": ["C8-batch-small", "C1-baseline", "C9-batch-large"],
        "regularisation (C10 / C11 / C12)": ["C10-no-reg", "C11-dropout-heavy",
                                             "C12-batchnorm-wd"],
    }
    factor_multi = {}
    for fname, names in FACTOR_GROUPS.items():
        sub = df[df["config"].isin(names)]
        factor_multi[fname] = {
            "val_roc_auc_spread": float(sub["val_roc_auc"].max() - sub["val_roc_auc"].min()),
            "val_accuracy_spread": float(sub["val_accuracy"].max() - sub["val_accuracy"].min()),
            "val_f1_spread": float(sub["val_f1"].max() - sub["val_f1"].min()),
            "train_time_spread_s": float(sub["train_time_s"].max() - sub["train_time_s"].min()),
            "exceeds_seed_noise_auc": bool(
                (sub["val_roc_auc"].max() - sub["val_roc_auc"].min()) > noise_band["val_roc_auc"]),
            "exceeds_seed_noise_f1": bool(
                (sub["val_f1"].max() - sub["val_f1"].min()) > noise_band["val_f1"]),
        }
    # the unregularized run keeps its last-epoch weights, so the gap shows up here
    overfit_evidence = {
        n: {"final_train_acc": detail[n]["history"]["train_acc"][-1],
            "final_val_acc": detail[n]["history"]["val_acc"][-1],
            "final_train_loss": detail[n]["history"]["train_loss"][-1],
            "final_val_loss": detail[n]["history"]["val_loss"][-1],
            "min_val_loss": min(detail[n]["history"]["val_loss"]),
            "val_loss_rise_after_best": detail[n]["history"]["val_loss"][-1]
                                        - min(detail[n]["history"]["val_loss"])}
        for n in ("C1-baseline", "C10-no-reg", "C11-dropout-heavy", "C12-batchnorm-wd")
    }

    # retrain the best config and save it
    model, res = train_model(best_cfg, X_train, y_train, X_val, y_val, verbose=False)
    test_prob = predict_proba(model, X_test)
    best_test = classification_metrics(y_test, test_prob)
    torch.save({"state_dict": model.state_dict(), "config": best_cfg.as_dict(),
                "in_features": int(X_train.shape[1])},
               C.ARTIFACTS / "best_nn_model.pt")
    plot_curves(res.history, f"Step 3 - Best configuration ({best_name}) curves",
                C.FIGURES / "step3_best_curves.png")
    plot_confusion(best_test["confusion_matrix"],
                   f"Step 3 - Best NN confusion matrix ({best_name}, test)",
                   C.FIGURES / "step3_best_confusion_matrix.png")

    # plots
    fig, ax = plt.subplots(figsize=(11, 5))
    d = df.sort_values("val_roc_auc")
    ax.barh(d["config"], d["val_roc_auc"], color="#4C78A8")
    ax.set_xlim(min(d["val_roc_auc"]) - 0.01, max(d["val_roc_auc"]) + 0.003)
    ax.set_xlabel("Validation ROC-AUC")
    ax.set_title("Step 3 - Validation ROC-AUC by configuration")
    for y_, v in zip(range(len(d)), d["val_roc_auc"]):
        ax.text(v + 0.0004, y_, f"{v:.4f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "step3_config_auc.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in ["C1-baseline", "C10-no-reg", "C11-dropout-heavy", "C12-batchnorm-wd"]:
        h = histories[name]
        ax.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], lw=2, label=f"{name} (val)")
        ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], lw=1, ls="--", alpha=.6,
                label=f"{name} (train)")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss")
    ax.set_title("Step 3 - Effect of regularisation on the train/validation gap")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "step3_regularisation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = [k for k, _ in ranked][::-1]
    vals = [v for _, v in ranked][::-1]
    ax.barh(labels, vals, color="#E45756")
    ax.set_xlabel("Spread in validation ROC-AUC across the factor's levels")
    ax.set_title("Step 3 - Which factor mattered most")
    for y_, v in zip(range(len(vals)), vals):
        ax.text(v + max(vals) * 0.01, y_, f"{v:.4f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "step3_factor_effects.png", dpi=150); plt.close(fig)

    # save the results
    payload = {
        "n_configurations": len(CONFIGS),
        "table": rows,
        "best_config": best_cfg.as_dict(),
        "best_config_name": best_name,
        "best_config_test_metrics": best_test,
        "best_config_train_time_sec": res.train_time_sec,
        "best_config_history": res.history,
        "factor_effect_spread_val_roc_auc": dict(ranked),
        "factor_effect_multi_metric": factor_multi,
        "seed_noise_runs": noise_runs,
        "seed_noise_band": noise_band,
        "overfitting_evidence": overfit_evidence,
        "most_important_factor": ranked[0][0],
        "per_config_detail": detail,
    }
    (C.RESULTS / "step3_tuning.json").write_text(json.dumps(payload, indent=2))

    cols = ["config", "hidden_sizes", "lr", "batch_size", "dropout", "weight_decay",
            "batch_norm", "early_stopping", "params", "epochs_run", "train_time_s",
            "val_loss", "val_accuracy", "val_f1", "val_roc_auc", "train_val_acc_gap"]
    md = ["# Step 3 - Manual Tuning (12 configurations)", "",
          "Sorted by validation ROC-AUC. All runs use the identical Step-1 split and seed 42.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md += ["", "## Seed-only noise floor", "",
           "Configuration C1 retrained under three different seeds on the same split:", "",
           "| Seed | val ROC-AUC | val accuracy | val F1 |", "|---|---|---|---|"]
    md += [f"| {r['seed']} | {r['val_roc_auc']:.4f} | {r['val_accuracy']:.4f} | "
           f"{r['val_f1']:.4f} |" for r in noise_runs]
    md += ["", f"Seed-only spread: ROC-AUC {noise_band['val_roc_auc']:.4f}, "
           f"accuracy {noise_band['val_accuracy']:.4f}, F1 {noise_band['val_f1']:.4f}. "
           "A factor only counts as real if its spread exceeds this band.", "",
           "## Which factor mattered most", "",
           "| Factor | ROC-AUC spread | Accuracy spread | F1 spread | Train-time spread (s) | "
           "> seed noise (AUC) |", "|---|---|---|---|---|---|"]
    for k, v in factor_multi.items():
        md.append(f"| {k} | {v['val_roc_auc_spread']:.4f} | {v['val_accuracy_spread']:.4f} | "
                  f"{v['val_f1_spread']:.4f} | {v['train_time_spread_s']:.1f} | "
                  f"{'yes' if v['exceeds_seed_noise_auc'] else 'no'} |")
    md += ["", f"**Largest raw ROC-AUC spread: {ranked[0][0]}**", "",
           "## Overfitting evidence (final-epoch weights)", "",
           "| Config | final train acc | final val acc | final val loss | min val loss | "
           "val-loss rise |", "|---|---|---|---|---|---|"]
    for n, v in overfit_evidence.items():
        md.append(f"| {n} | {v['final_train_acc']:.4f} | {v['final_val_acc']:.4f} | "
                  f"{v['final_val_loss']:.4f} | {v['min_val_loss']:.4f} | "
                  f"{v['val_loss_rise_after_best']:+.4f} |")
    md += [""]
    (C.RESULTS / "step3_tuning.md").write_text("\n".join(md) + "\n")

    print("\nFactor effect (spread in val ROC-AUC):")
    for k, v in ranked:
        print(f"  {k:36s} {v:.4f}")
    print(f"\nBest test metrics: acc={best_test['accuracy']:.4f} "
          f"F1={best_test['f1']:.4f} AUC={best_test['roc_auc']:.4f}")
    print("Step 3 complete.")


if __name__ == "__main__":
    main()
