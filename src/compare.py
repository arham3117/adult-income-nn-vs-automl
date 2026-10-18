"""Step 4 (part 2) - Head-to-head: hand-built network vs. FLAML AutoML.

Compared on accuracy, F1, ROC-AUC, training time, model size on disk and
inference latency. Sizes and latencies are measured END TO END (model plus the
preprocessing it needs), because that is what actually has to be deployed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from evaluate import classification_metrics, load_splits, plot_roc
from model import MLP, predict_proba

N_LATENCY_SINGLE = 200   # single-row requests, the realistic serving pattern
N_LATENCY_BATCH = 20     # full-test-set passes


def load_nn(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = MLP(ckpt["in_features"], tuple(cfg["hidden_sizes"]), cfg["dropout"], cfg["batch_norm"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, cfg


def time_calls(fn, n: int) -> dict:
    fn()  # warm-up (JIT/threadpool/page-cache)
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times)
    return {"mean_ms": round(float(a.mean()), 4),
            "median_ms": round(float(np.median(a)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
            "std_ms": round(float(a.std()), 4), "n": n}


def main() -> None:
    _, _, _, _, X_test_t, y_test = load_splits()
    pre = joblib.load(C.ARTIFACTS / "preprocessor.joblib")
    raw_test = pd.read_csv(C.DATA_PROC / "raw_test.csv")
    X_raw = raw_test.drop(columns=["income_binary"])
    y_raw = raw_test["income_binary"].values
    assert np.array_equal(y_raw, y_test.astype(int)), "test splits must be identical"

    step2 = json.loads((C.RESULTS / "step2_metrics.json").read_text())
    step3 = json.loads((C.RESULTS / "step3_tuning.json").read_text())
    step4 = json.loads((C.RESULTS / "step4_automl.json").read_text())

    # ---------------- neural network (best tuned config) ---------------------
    nn_model, nn_cfg = load_nn(C.ARTIFACTS / "best_nn_model.pt")
    nn_prob = predict_proba(nn_model, X_test_t)
    nn_metrics = classification_metrics(y_test, nn_prob)

    nn_model_bytes = (C.ARTIFACTS / "best_nn_model.pt").stat().st_size
    pre_bytes = (C.ARTIFACTS / "preprocessor.joblib").stat().st_size
    nn_total_bytes = nn_model_bytes + pre_bytes

    # ---------------- Step-2 baseline network (third contender) --------------
    # The assignment asks explicitly for Step 2 to be compared with AutoML, so the
    # untuned baseline is carried through the same size/latency measurement rather
    # than only the Step-3 winner.
    b_model, b_cfg = load_nn(C.ARTIFACTS / "baseline_model.pt")
    b_prob = predict_proba(b_model, X_test_t)
    b_metrics = classification_metrics(y_test, b_prob)
    b_model_bytes = (C.ARTIFACTS / "baseline_model.pt").stat().st_size
    b_total_bytes = b_model_bytes + pre_bytes

    one_raw = X_raw.iloc[[0]]

    def b_single():
        x = pre.transform(one_raw)
        with torch.no_grad():
            torch.sigmoid(b_model(torch.tensor(np.asarray(x), dtype=torch.float32)))

    def b_batch():
        x = pre.transform(X_raw)
        with torch.no_grad():
            torch.sigmoid(b_model(torch.tensor(np.asarray(x), dtype=torch.float32)))

    def nn_single():
        x = pre.transform(one_raw)
        with torch.no_grad():
            torch.sigmoid(nn_model(torch.tensor(np.asarray(x), dtype=torch.float32)))

    def nn_batch():
        x = pre.transform(X_raw)
        with torch.no_grad():
            torch.sigmoid(nn_model(torch.tensor(np.asarray(x), dtype=torch.float32)))

    nn_lat_single = time_calls(nn_single, N_LATENCY_SINGLE)
    nn_lat_batch = time_calls(nn_batch, N_LATENCY_BATCH)
    b_lat_single = time_calls(b_single, N_LATENCY_SINGLE)
    b_lat_batch = time_calls(b_batch, N_LATENCY_BATCH)

    # ---------------- FLAML AutoML -------------------------------------------
    automl = joblib.load(C.ARTIFACTS / "automl_model.joblib")
    ft = automl.feature_transformer

    # --- FLAML category-code bug, and the fix -------------------------------
    # flaml.automl.data.DataTransformer.transform() re-derives the pandas
    # Categorical levels from whatever frame it is handed instead of reusing the
    # levels seen at fit time. A one-row request therefore gets codes numbered
    # 0..k for only the values in that row, which do not line up with the codes
    # the model was trained on - predictions come back silently wrong (we measured
    # 0.27 vs 0.98 for the same record). We therefore pin the training-time
    # CategoricalDtypes and re-cast after transforming. Casting to a fixed
    # CategoricalDtype also maps any unseen level to NaN, which XGBoost handles
    # natively as a missing value - this is the Step 5 unseen-category mechanism.
    X_tr_raw = pd.read_csv(C.DATA_PROC / "raw_train.csv").drop(columns=["income_binary"])
    t_train = ft.transform(X_tr_raw)
    pinned_dtypes = {c: t_train[c].dtype for c in t_train.columns
                     if str(t_train[c].dtype) == "category"}

    def am_transform(X: pd.DataFrame) -> pd.DataFrame:
        t = ft.transform(X).copy()
        for col, dt in pinned_dtypes.items():
            if col in t.columns:
                t[col] = t[col].astype(dt)
        return t

    est_inner = automl.model
    am_prob = est_inner.predict_proba(am_transform(X_raw))[:, 1]
    am_metrics = classification_metrics(y_raw, am_prob)

    # Row-by-row must now agree with the batch pass; assert it rather than hope.
    _row_probs = np.array([est_inner.predict_proba(am_transform(X_raw.iloc[[i]]))[0, 1]
                           for i in range(10)])
    assert np.allclose(_row_probs, am_prob[:10], atol=1e-6), \
        "single-row and batch predictions disagree"

    am_naive_prob = automl.predict_proba(X_raw)[:, 1]
    am_naive_metrics = classification_metrics(y_raw, am_naive_prob)

    # The pickled AutoML object is ~35 MB, but most of that is search state
    # (model_history keeps every estimator's best model). What you would actually
    # deploy is the winning estimator + transformer + pinned dtypes.
    am_lean_path = C.ARTIFACTS / "automl_deployable.joblib"
    joblib.dump({"feature_transformer": ft, "estimator": est_inner,
                 "pinned_dtypes": pinned_dtypes,
                 "estimator_name": automl.best_estimator,
                 "best_config": automl.best_config}, am_lean_path, compress=0)
    am_full_bytes = (C.ARTIFACTS / "automl_model.joblib").stat().st_size
    am_bytes = am_lean_path.stat().st_size
    am_est_bytes = (C.ARTIFACTS / "automl_best_estimator.joblib").stat().st_size

    def am_single():
        est_inner.predict_proba(am_transform(one_raw))

    def am_batch():
        est_inner.predict_proba(am_transform(X_raw))

    am_lat_single = time_calls(am_single, N_LATENCY_SINGLE)
    am_lat_batch = time_calls(am_batch, N_LATENCY_BATCH)

    # ---------------- training time ------------------------------------------
    # For the NN the honest number is the whole manual search (12 configs + the
    # final retrain), because that human-directed search is what produced the
    # model - the same role the FLAML time budget plays for AutoML.
    nn_sweep_time = sum(r["train_time_s"] for r in step3["table"])
    nn_final_time = step3["best_config_train_time_sec"]
    am_search_time = step4["actual_search_time_s"]
    am_fit_time = step4["best_config_train_time_s"]

    b_fit_time = step2["train_time_sec"]
    nn_params = next(r["params"] for r in step3["table"]
                     if r["config"] == step3["best_config_name"])

    def row(metric, step2_v, nn_v, am_v):
        return {"metric": metric, "nn_step2_baseline": step2_v,
                "neural_network": nn_v, "automl_flaml": am_v}

    rows = [
        row("Test accuracy", round(b_metrics["accuracy"], 4),
            round(nn_metrics["accuracy"], 4), round(am_metrics["accuracy"], 4)),
        row("Test precision", round(b_metrics["precision"], 4),
            round(nn_metrics["precision"], 4), round(am_metrics["precision"], 4)),
        row("Test recall", round(b_metrics["recall"], 4),
            round(nn_metrics["recall"], 4), round(am_metrics["recall"], 4)),
        row("Test F1", round(b_metrics["f1"], 4),
            round(nn_metrics["f1"], 4), round(am_metrics["f1"], 4)),
        row("Test ROC-AUC", round(b_metrics["roc_auc"], 4),
            round(nn_metrics["roc_auc"], 4), round(am_metrics["roc_auc"], 4)),
        row("Final-model fit time (s)", round(b_fit_time, 2),
            round(nn_final_time, 2), round(am_fit_time, 2)),
        row("Total search time (s)", round(b_fit_time, 2),
            round(nn_sweep_time, 2), round(am_search_time, 2)),
        row("Model file size (KB)", round(b_model_bytes / 1024, 1),
            round(nn_model_bytes / 1024, 1), round(am_est_bytes / 1024, 1)),
        row("Deployable size incl. preprocessing (KB)", round(b_total_bytes / 1024, 1),
            round(nn_total_bytes / 1024, 1), round(am_bytes / 1024, 1)),
        row("Full pickled search object (KB)", "n/a", "n/a",
            round(am_full_bytes / 1024, 1)),
        row("Latency, single row (ms, mean)", b_lat_single["mean_ms"],
            nn_lat_single["mean_ms"], am_lat_single["mean_ms"]),
        row("Latency, single row (ms, p95)", b_lat_single["p95_ms"],
            nn_lat_single["p95_ms"], am_lat_single["p95_ms"]),
        row(f"Latency, batch of {len(X_raw):,} (ms, mean)", b_lat_batch["mean_ms"],
            nn_lat_batch["mean_ms"], am_lat_batch["mean_ms"]),
        row("Throughput, batch (rows/s)",
            round(len(X_raw) / (b_lat_batch["mean_ms"] / 1000)),
            round(len(X_raw) / (nn_lat_batch["mean_ms"] / 1000)),
            round(len(X_raw) / (am_lat_batch["mean_ms"] / 1000))),
        row("Trainable parameters / trees", step2["total_trainable_parameters"], nn_params,
            (step4["selected_hyperparameters"] or {}).get("n_estimators", "n/a")),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "step4_comparison.csv", index=False)

    # ---------------- winner --------------------------------------------------
    winner = "automl_flaml" if am_metrics["roc_auc"] > nn_metrics["roc_auc"] else "neural_network"
    margin = abs(am_metrics["roc_auc"] - nn_metrics["roc_auc"])
    seed_noise = step3["seed_noise_band"]["val_roc_auc"]

    plot_roc({"Hand-built MLP": (y_test, nn_prob),
              f"FLAML ({step4['selected_estimator']})": (y_raw, am_prob)},
             "Step 4 - ROC comparison on the test set",
             C.FIGURES / "step4_roc_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    keys = ["accuracy", "f1", "roc_auc"]
    for ax, k in zip(axes, keys):
        vals = [nn_metrics[k], am_metrics[k]]
        bars = ax.bar(["Hand-built\nMLP", f"FLAML\n{step4['selected_estimator']}"], vals,
                      color=["#4C78A8", "#F58518"])
        ax.set_ylim(min(vals) - 0.06, max(vals) + 0.03)
        ax.set_title(f"Test {k.upper()}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
                    ha="center", fontsize=10)
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("Step 4 - Hand-built network vs. AutoML")
    fig.tight_layout(); fig.savefig(C.FIGURES / "step4_metric_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(["MLP", "FLAML"], [nn_total_bytes / 1024, am_bytes / 1024],
                color=["#4C78A8", "#F58518"])
    axes[0].set_ylabel("KB"); axes[0].set_title("Deployable size on disk\n(model + preprocessing)")
    axes[0].grid(axis="y", alpha=.3)
    axes[1].bar(["MLP", "FLAML"], [nn_lat_single["mean_ms"], am_lat_single["mean_ms"]],
                color=["#4C78A8", "#F58518"])
    axes[1].set_ylabel("ms"); axes[1].set_title("Single-row inference latency\n(mean, end to end)")
    axes[1].grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "step4_cost_comparison.png", dpi=150)
    plt.close(fig)

    payload = {
        "comparison_table": rows,
        "neural_network": {
            "config_name": step3["best_config_name"], "config": nn_cfg,
            "test_metrics": nn_metrics,
            "baseline_step2_test_metrics": step2["test_metrics"],
            "latency_single_row": nn_lat_single, "latency_batch": nn_lat_batch,
            "model_bytes": nn_model_bytes, "preprocessor_bytes": pre_bytes,
            "deployable_bytes": nn_total_bytes,
            "final_fit_time_s": nn_final_time, "total_search_time_s": nn_sweep_time,
        },
        "automl": {
            "framework": "FLAML", "selected_estimator": step4["selected_estimator"],
            "hyperparameters": step4["selected_hyperparameters"],
            "test_metrics": am_metrics,
            "latency_single_row": am_lat_single, "latency_batch": am_lat_batch,
            "model_bytes": am_est_bytes, "deployable_bytes": am_bytes,
            "final_fit_time_s": am_fit_time, "total_search_time_s": am_search_time,
            "full_search_object_bytes": am_full_bytes,
            "test_metrics_without_dtype_pinning": am_naive_metrics,
            "dtype_pinning_note": (
                "FLAML's DataTransformer re-derives Categorical levels per call, so "
                "un-pinned single-row inference is silently wrong and the batch score is "
                "slightly off. Pinning the training-time dtypes fixes both: test ROC-AUC "
                f"{am_naive_metrics['roc_auc']:.4f} -> {am_metrics['roc_auc']:.4f}."),
        },
        "neural_network_step2_baseline": {
            "config": step2["config"], "test_metrics": b_metrics,
            "latency_single_row": b_lat_single, "latency_batch": b_lat_batch,
            "model_bytes": b_model_bytes, "deployable_bytes": b_total_bytes,
            "fit_time_s": b_fit_time,
            "n_params": step2["total_trainable_parameters"],
        },
        "step2_vs_automl": {
            "roc_auc_margin": round(am_metrics["roc_auc"] - b_metrics["roc_auc"], 5),
            "f1_margin": round(am_metrics["f1"] - b_metrics["f1"], 5),
            "accuracy_margin": round(am_metrics["accuracy"] - b_metrics["accuracy"], 5),
            "automl_slower_x": round(am_lat_single["mean_ms"] / b_lat_single["mean_ms"], 1),
            "automl_larger_x": round(am_bytes / b_total_bytes, 1),
        },
        "winner": winner,
        "roc_auc_margin": round(margin, 5),
        "seed_noise_band_roc_auc": round(seed_noise, 5),
        "margin_exceeds_seed_noise": bool(margin > seed_noise),
        "selection_rule": "Primary criterion: test ROC-AUC (threshold-independent and robust to "
                          "the 24/76 class imbalance). F1 is used as a tie-break, then deployable "
                          "size and latency.",
    }
    (C.RESULTS / "step4_comparison.json").write_text(json.dumps(payload, indent=2, default=str))
    (C.ARTIFACTS / "winner.json").write_text(json.dumps(
        {"winner": winner, "selected_estimator": step4["selected_estimator"],
         "nn_config_name": step3["best_config_name"]}, indent=2))

    L = ["# Step 4 - Hand-built Network vs. AutoML", "",
         f"Step-2 baseline MLP  |  best tuned MLP (`{step3['best_config_name']}`)  |  "
         f"AutoML: FLAML, selected `{step4['selected_estimator']}`", "",
         "| Metric | Step-2 baseline MLP | Best tuned MLP | FLAML AutoML |",
         "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['metric']} | {r['nn_step2_baseline']} | {r['neural_network']} | "
                 f"{r['automl_flaml']} |")
    L += ["", f"**Winner: {'FLAML AutoML' if winner == 'automl_flaml' else 'Hand-built MLP'}** "
          f"(test ROC-AUC margin {margin:.4f}; seed-only noise band is {seed_noise:.4f}, so the "
          f"margin is {'larger than' if margin > seed_noise else 'within'} run-to-run noise).", "",
          payload["selection_rule"], ""]
    (C.RESULTS / "step4_comparison.md").write_text("\n".join(L) + "\n")

    print(df.to_string(index=False))
    print(f"\nWinner: {winner} (margin {margin:.4f} vs noise {seed_noise:.4f})")
    print("Step 4 (comparison) complete.")


if __name__ == "__main__":
    main()
