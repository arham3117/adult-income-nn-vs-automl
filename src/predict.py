"""Step 5 (part 2) - predict() over RAW, unprocessed input.

A caller sends the 12 original columns as plain strings/numbers exactly as they
appear in the census file. Everything else - imputation, encoding, scaling,
unseen-category handling - happens here, using the artefacts fitted on the
training split.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

BUNDLE_PATH = C.DEPLOY_DIR / "model_bundle.joblib"


@lru_cache(maxsize=1)
def load_bundle(path: str = str(BUNDLE_PATH)) -> dict:
    """Load once and cache - model loading must not be on the request path."""
    bundle = joblib.load(path)
    if bundle["kind"] == "pytorch_mlp":
        import torch
        from model import MLP
        cfg = bundle["model_config"]
        net = MLP(bundle["in_features"], tuple(cfg["hidden_sizes"]),
                  cfg["dropout"], cfg["batch_norm"])
        net.load_state_dict(bundle["state_dict"])
        net.eval()
        bundle = {**bundle, "_net": net}
    return bundle


def _coerce(records: Iterable[dict], schema: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Validate/repair raw dicts into the exact training frame layout.

    Returns the frame plus a per-record note of every repair applied, so the API
    can tell the caller what it did rather than silently guessing.
    """
    cols = schema["raw_columns"]
    rows, notes = [], []
    for rec in records:
        rec = {str(k).strip(): v for k, v in dict(rec).items()}
        note: dict[str, Any] = {"missing_fields": [], "unseen_categories": {},
                                "coercion_failures": [], "out_of_range": {}}
        row = {}
        for col in cols:
            val = rec.get(col, None)
            if isinstance(val, str):
                val = val.strip()
                if val in ("", "?", "NA", "N/A", "null", "None"):
                    val = None
            if val is None:
                note["missing_fields"].append(col)

            if col in schema["numeric_features"]:
                if val is None:
                    # Median from the TRAINING split - never from the request.
                    row[col] = schema["numeric_defaults"][col]
                else:
                    try:
                        row[col] = float(val)
                    except (TypeError, ValueError):
                        note["coercion_failures"].append(col)
                        row[col] = schema["numeric_defaults"][col]
                rng = schema["numeric_ranges"][col]
                if row[col] < rng["min"] or row[col] > rng["max"]:
                    # Kept, not clipped: trees extrapolate flat beyond a split.
                    note["out_of_range"][col] = row[col]
            else:
                if val is not None:
                    val = str(val)
                    if val not in schema["known_categories"][col]:
                        note["unseen_categories"][col] = val
                row[col] = val
        rows.append(row)
        notes.append(note)
    return pd.DataFrame(rows, columns=cols), notes


def _proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    if bundle["kind"] == "flaml":
        t = bundle["feature_transformer"].transform(X).copy()
        # Re-cast to the TRAINING-time Categorical levels. Without this FLAML
        # renumbers the codes per request and predictions are silently wrong;
        # with it, any unseen level becomes NaN -> handled as missing by XGBoost.
        for col, dt in bundle["pinned_dtypes"].items():
            if col in t.columns:
                t[col] = t[col].astype(dt)
        return bundle["estimator"].predict_proba(t)[:, 1]

    import torch
    Xt = bundle["preprocessor"].transform(X)
    with torch.no_grad():
        logits = bundle["_net"](torch.tensor(np.asarray(Xt), dtype=torch.float32))
        return torch.sigmoid(logits).numpy().reshape(-1)


def predict(records: list[dict] | dict, bundle: dict | None = None,
            threshold: float | None = None) -> list[dict]:
    """Predict income class from raw, unprocessed records.

    >>> predict({"age": 39, "workclass": "State-gov", "education-num": 13, ...})
    [{'prediction': '<=50K', 'probability_gt_50k': 0.02, ...}]
    """
    bundle = bundle or load_bundle()
    schema = bundle["schema"]
    thr = schema["decision_threshold"] if threshold is None else threshold
    if isinstance(records, dict):
        records = [records]
    if not records:
        return []

    X, notes = _coerce(records, schema)
    probs = _proba(bundle, X)

    out = []
    for p, note in zip(probs, notes):
        p = float(p)
        out.append({
            "prediction": ">50K" if p >= thr else "<=50K",
            "predicted_class": int(p >= thr),
            "probability_gt_50k": round(p, 6),
            "probability_le_50k": round(1 - p, 6),
            "confidence": round(max(p, 1 - p), 6),
            "threshold": thr,
            "input_warnings": {k: v for k, v in note.items() if v},
        })
    return out


if __name__ == "__main__":
    sample = {
        "age": 39, "workclass": "State-gov", "education-num": 13,
        "marital-status": "Never-married", "occupation": "Adm-clerical",
        "relationship": "Not-in-family", "race": "White", "sex": "Male",
        "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
        "native-country": "United-States",
    }
    print(json.dumps(predict(sample), indent=2))
    weird = {**sample, "native-country": "Wakanda", "occupation": None, "age": "not-a-number"}
    print(json.dumps(predict(weird), indent=2))
