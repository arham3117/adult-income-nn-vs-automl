"""Step 5 (part 1) - Export the winning model together with its preprocessing.

Produces a single self-contained bundle in artifacts/deployment/ so the serving
code never needs the training code, the dataset, or FLAML's search state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from model import MLP


def build_schema(X_train: pd.DataFrame) -> dict:
    """Raw-input contract: the 12 unprocessed columns a caller must send."""
    return {
        "raw_columns": C.FEATURE_COLUMNS,
        "numeric_features": C.NUMERIC_FEATURES,
        "categorical_features": C.CATEGORICAL_FEATURES,
        "known_categories": {
            c: sorted(X_train[c].dropna().unique().tolist())
            for c in C.CATEGORICAL_FEATURES
        },
        "numeric_ranges": {
            c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                "median": float(X_train[c].median())}
            for c in C.NUMERIC_FEATURES
        },
        "numeric_defaults": {c: float(X_train[c].median()) for c in C.NUMERIC_FEATURES},
        "target": {"name": "income", "positive_class": ">50K",
                   "classes": {"0": "<=50K", "1": ">50K"}},
        "decision_threshold": 0.5,
    }


def main() -> None:
    winner_info = json.loads((C.ARTIFACTS / "winner.json").read_text())
    winner = winner_info["winner"]
    comparison = json.loads((C.RESULTS / "step4_comparison.json").read_text())

    X_train = pd.read_csv(C.DATA_PROC / "raw_train.csv").drop(columns=["income_binary"])
    schema = build_schema(X_train)

    bundle: dict = {
        "winner": winner,
        "created_from": "src/export_model.py",
        "schema": schema,
        "seed": C.SEED,
    }

    if winner == "automl_flaml":
        lean = joblib.load(C.ARTIFACTS / "automl_deployable.joblib")
        bundle.update({
            "kind": "flaml",
            "feature_transformer": lean["feature_transformer"],
            "estimator": lean["estimator"],
            # Training-time Categorical levels. Re-casting to these after FLAML's
            # transform is mandatory: FLAML re-derives levels per call, which
            # silently corrupts single-row inference. It also maps unseen levels
            # to NaN, which XGBoost treats as missing.
            "pinned_dtypes": lean["pinned_dtypes"],
            "estimator_name": lean["estimator_name"],
            "hyperparameters": lean["best_config"],
            "test_metrics": comparison["automl"]["test_metrics"],
        })
    else:
        ckpt = torch.load(C.ARTIFACTS / "best_nn_model.pt", map_location="cpu",
                          weights_only=False)
        bundle.update({
            "kind": "pytorch_mlp",
            "preprocessor": joblib.load(C.ARTIFACTS / "preprocessor.joblib"),
            "state_dict": ckpt["state_dict"],
            "model_config": ckpt["config"],
            "in_features": ckpt["in_features"],
            "test_metrics": comparison["neural_network"]["test_metrics"],
        })

    out = C.DEPLOY_DIR / "model_bundle.joblib"
    joblib.dump(bundle, out, compress=3)
    (C.DEPLOY_DIR / "schema.json").write_text(json.dumps(schema, indent=2))

    meta = {
        "winner": winner,
        "kind": bundle["kind"],
        "bundle_path": str(out.relative_to(C.ROOT)),
        "bundle_size_bytes": out.stat().st_size,
        "bundle_size_mb": round(out.stat().st_size / 1e6, 2),
        "test_metrics": bundle["test_metrics"],
        "estimator_name": bundle.get("estimator_name"),
        "hyperparameters": bundle.get("hyperparameters"),
    }
    (C.DEPLOY_DIR / "bundle_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))
    print("Export complete.")


if __name__ == "__main__":
    main()
