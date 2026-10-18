"""Step 4 (part 1) - Run the FLAML AutoML framework on exactly the same split.

Fairness decisions (documented in the report):
  * FLAML receives the identical Step-1 train rows and the identical validation
    rows (eval_method='holdout' with X_val/y_val passed explicitly), so no row of
    the test set is ever seen and no re-splitting happens.
  * retrain_full=False. FLAML's default is to refit the winning config on
    train+validation after the search; that would hand AutoML ~33% more training
    data than the hand-built network ever saw and make the comparison unfair.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from evaluate import classification_metrics

TIME_BUDGET = 300  # seconds
ESTIMATORS = ["lgbm", "xgboost", "xgb_limitdepth", "rf", "extra_tree", "lrl1"]


def load_raw_splits():
    tr = pd.read_csv(C.DATA_PROC / "raw_train.csv")
    va = pd.read_csv(C.DATA_PROC / "raw_val.csv")
    te = pd.read_csv(C.DATA_PROC / "raw_test.csv")
    y = "income_binary"
    return (tr.drop(columns=[y]), tr[y].values,
            va.drop(columns=[y]), va[y].values,
            te.drop(columns=[y]), te[y].values)


def main() -> None:
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_raw_splits()
    print(f"FLAML input - train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    log_file = str(C.ARTIFACTS / "flaml_search.log")
    automl = AutoML()
    settings = dict(
        task="classification",
        metric="roc_auc",
        time_budget=TIME_BUDGET,
        estimator_list=ESTIMATORS,
        eval_method="holdout",
        X_val=X_va, y_val=y_va,        # exactly the Step-1 validation split
        retrain_full=False,            # do NOT refit on train+val (fairness)
        model_history=True,            # keep each estimator's best model for the leaderboard
        log_file_name=log_file,
        seed=C.SEED,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.perf_counter()
    automl.fit(X_train=X_tr, y_train=y_tr, **settings)
    search_time = time.perf_counter() - t0
    print(f"\nSearch finished in {search_time:.1f}s. Best estimator: {automl.best_estimator}")

    # ---- leaderboard: each estimator's best configuration --------------------
    # best_model_for_estimator() returns the *inner* learner, which expects the
    # features FLAML's own DataTransformer produces (object -> pandas category,
    # NaN -> '__NAN__'), not the raw frame. Transform once and reuse.
    ft = automl.feature_transformer
    X_va_t, X_te_t = ft.transform(X_va), ft.transform(X_te)

    leaderboard = []
    for est in ESTIMATORS:
        loss = automl.best_loss_per_estimator.get(est)
        if loss is None or not np.isfinite(loss):
            leaderboard.append({"estimator": est, "status": "no successful trial"})
            continue
        row = {
            "estimator": est,
            "val_roc_auc_flaml": round(1.0 - float(loss), 5),  # FLAML loss = 1 - roc_auc
            "best_config": automl.best_config_per_estimator.get(est),
            "config_train_time_s": round(float(
                automl.best_config_train_time or 0) if est == automl.best_estimator else 0.0, 3),
        }
        try:
            mdl = automl.best_model_for_estimator(est)
            if mdl is not None:
                p_va = mdl.predict_proba(X_va_t)[:, 1]
                p_te = mdl.predict_proba(X_te_t)[:, 1]
                vm = classification_metrics(y_va, p_va)
                tm = classification_metrics(y_te, p_te)
                row.update({
                    "val_accuracy": round(vm["accuracy"], 4),
                    "val_f1": round(vm["f1"], 4),
                    "val_roc_auc": round(vm["roc_auc"], 4),
                    "test_accuracy": round(tm["accuracy"], 4),
                    "test_f1": round(tm["f1"], 4),
                    "test_roc_auc": round(tm["roc_auc"], 4),
                })
        except Exception as exc:
            row["eval_error"] = str(exc)
        leaderboard.append(row)

    leaderboard.sort(key=lambda r: r.get("val_roc_auc", r.get("val_roc_auc_flaml", -1)),
                     reverse=True)

    # ---- trial history from the FLAML log ------------------------------------
    trials = []
    try:
        with open(log_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "record_id" in rec:
                    trials.append({
                        "record_id": rec.get("record_id"),
                        "iter_per_learner": rec.get("iter_per_learner"),
                        "learner": rec.get("learner"),
                        "validation_loss": rec.get("validation_loss"),
                        "val_roc_auc": (None if rec.get("validation_loss") is None
                                        else round(1 - rec["validation_loss"], 5)),
                        "wall_clock_time": rec.get("wall_clock_time"),
                        "config": rec.get("config"),
                    })
    except Exception as exc:
        print(f"  (could not parse FLAML log: {exc})")

    # ---- selected model: full test evaluation + persistence -------------------
    p_te = automl.predict_proba(X_te)[:, 1]
    p_va = automl.predict_proba(X_va)[:, 1]
    test_metrics = classification_metrics(y_te, p_te)
    val_metrics = classification_metrics(y_va, p_va)

    model_path = C.ARTIFACTS / "automl_model.joblib"
    joblib.dump(automl, model_path)                       # full pipeline incl. transformer
    inner_path = C.ARTIFACTS / "automl_best_estimator.joblib"
    joblib.dump(automl.model, inner_path)

    payload = {
        "framework": "FLAML",
        "flaml_version": __import__("flaml").__version__,
        "settings": {k: v for k, v in settings.items()
                     if k not in ("X_val", "y_val")},
        "time_budget_s": TIME_BUDGET,
        "actual_search_time_s": round(search_time, 2),
        "time_to_find_best_model_s": round(float(automl.time_to_find_best_model), 2),
        "best_config_train_time_s": round(float(automl.best_config_train_time), 3),
        "selected_estimator": automl.best_estimator,
        "selected_hyperparameters": automl.best_config,
        "selected_model_repr": str(automl.model.model)[:2000],
        "n_trials": len(trials),
        "leaderboard": leaderboard,
        "trials": trials,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_size_bytes_full": model_path.stat().st_size,
        "model_size_bytes_estimator_only": inner_path.stat().st_size,
    }
    (C.RESULTS / "step4_automl.json").write_text(json.dumps(payload, indent=2, default=str))
    np.save(C.ARTIFACTS / "automl_test_proba.npy", p_te)

    # ---- Markdown -------------------------------------------------------------
    L = ["# Step 4 - AutoML (FLAML) Results", "",
         f"Framework: **FLAML {payload['flaml_version']}** | metric: **roc_auc** | "
         f"time budget: **{TIME_BUDGET}s** | actual search: **{search_time:.1f}s** | "
         f"trials: **{len(trials)}** | seed: **{C.SEED}**", "",
         "Validation is the *same* Step-1 validation split (`eval_method='holdout'`, "
         "`X_val`/`y_val` passed explicitly). `retrain_full=False` so FLAML never trains "
         "on validation rows.", "",
         "## Leaderboard (best configuration per estimator)", "",
         "| Rank | Estimator | Val ROC-AUC | Val acc | Val F1 | Test ROC-AUC | Test acc | Test F1 |",
         "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(leaderboard, 1):
        if "val_roc_auc" not in r:
            L.append(f"| {i} | {r['estimator']} | - | - | - | - | - | - |")
            continue
        L.append(f"| {i} | {r['estimator']} | {r['val_roc_auc']:.4f} | {r['val_accuracy']:.4f} | "
                 f"{r['val_f1']:.4f} | {r['test_roc_auc']:.4f} | {r['test_accuracy']:.4f} | "
                 f"{r['test_f1']:.4f} |")
    L += ["", f"## Selected model: `{automl.best_estimator}`", "",
          "Hyperparameters chosen by FLAML:", "", "| Hyperparameter | Value |", "|---|---|"]
    for k, v in (automl.best_config or {}).items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Selected model test metrics", "", "| Metric | Value |", "|---|---|"]
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        L.append(f"| {k} | {test_metrics[k]:.4f} |")
    cm = test_metrics["confusion_matrix"]
    L += ["", "| | Predicted <=50K | Predicted >50K |", "|---|---|---|",
          f"| **Actual <=50K** | {cm['tn']:,} | {cm['fp']:,} |",
          f"| **Actual >50K** | {cm['fn']:,} | {cm['tp']:,} |", ""]
    (C.RESULTS / "step4_automl.md").write_text("\n".join(L) + "\n")

    print("\nLeaderboard:")
    for r in leaderboard:
        print("  ", {k: v for k, v in r.items() if k != "best_config"})
    print(f"\nSelected: {automl.best_estimator}")
    print(f"Test: acc={test_metrics['accuracy']:.4f} F1={test_metrics['f1']:.4f} "
          f"AUC={test_metrics['roc_auc']:.4f}")
    print("Step 4 (AutoML run) complete.")


if __name__ == "__main__":
    main()
