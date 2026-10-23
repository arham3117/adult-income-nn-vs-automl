"""Step 1 - Data acquisition and a reproducible preprocessing pipeline.

Run:  python3 src/data_prep.py
Writes raw/processed splits, the fitted preprocessor, and a summary report.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


# --------------------------------------------------------------------------- #
# 1. Acquisition
# --------------------------------------------------------------------------- #
def download_raw() -> pd.DataFrame:
    """Fetch the UCI Adult dataset; fall back to OpenML 1590 if UCI is down."""
    zip_path = C.DATA_RAW / "adult.zip"
    if not zip_path.exists():
        import requests

        print(f"Downloading {C.UCI_URL} ...")
        resp = requests.get(C.UCI_URL, timeout=90)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            print(f"  archive contents: {names}")
            frames = []
            for member, skiprows in (("adult.data", 0), ("adult.test", 1)):
                if member not in names:
                    continue
                with zf.open(member) as fh:
                    raw = fh.read().decode("utf-8")
                df = pd.read_csv(
                    io.StringIO(raw),
                    header=None,
                    names=C.RAW_COLUMNS,
                    skiprows=skiprows,
                    skipinitialspace=True,
                    na_values=[C.MISSING_TOKEN],
                )
                df["source_file"] = member
                frames.append(df)
            data = pd.concat(frames, ignore_index=True)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"UCI parse failed ({exc}); falling back to OpenML 1590.")
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        data = bunch.frame.rename(columns={"class": "income"})
        data.columns = [c.replace("_", "-") for c in data.columns]
        data["source_file"] = "openml-1590"

    data.to_csv(C.DATA_RAW / "adult_combined_raw.csv", index=False)
    return data


# --------------------------------------------------------------------------- #
# 2. Cleaning
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise labels/whitespace, drop leaky or redundant columns, de-duplicate."""
    log: dict = {"rows_raw": int(len(df))}

    # Strip whitespace from every object column (the UCI files are ', '-separated).
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # adult.test writes labels with a trailing period: '<=50K.' -> '<=50K'.
    df[C.TARGET] = df[C.TARGET].astype(str).str.rstrip(".").str.strip()
    log["label_values_after_normalisation"] = sorted(df[C.TARGET].unique().tolist())

    # some rows in adult.test are blank, drop anything without a label
    df = df[df[C.TARGET].isin(["<=50K", ">50K"])].copy()
    log["rows_after_label_filter"] = int(len(df))

    df["income_binary"] = (df[C.TARGET] == ">50K").astype(int)

    # '?' was already read as NaN, just record where it showed up
    missing = df[C.RAW_COLUMNS].isna().sum()
    log["missing_per_column"] = {k: int(v) for k, v in missing.items() if v > 0}

    # Drop exact duplicates across the feature+label space (census data has a few).
    before = len(df)
    df = df.drop_duplicates(subset=C.RAW_COLUMNS).reset_index(drop=True)
    log["duplicate_rows_dropped"] = int(before - len(df))
    log["rows_final"] = int(len(df))

    log["dropped_columns"] = C.DROP_COLUMNS
    df = df.drop(columns=C.DROP_COLUMNS + ["source_file"], errors="ignore")

    return df, log


# --------------------------------------------------------------------------- #
# 3. Split + preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Numeric: median impute + standardise. Categorical: 'Missing' impute + one-hot.

    `handle_unknown='ignore'` is deliberate: at inference an unseen category becomes
    an all-zero block instead of raising, which is the mechanism used in Step 5.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value=C.MISSING_CATEGORY),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, C.NUMERIC_FEATURES),
            ("cat", categorical, C.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def stratified_split(df: pd.DataFrame):
    """60 / 20 / 20 stratified split with a fixed seed, done in two steps."""
    X = df[C.FEATURE_COLUMNS]
    y = df["income_binary"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=C.TEST_SIZE + C.VAL_SIZE, stratify=y, random_state=C.SEED
    )
    # Split the held-out 40% in half -> 20% val, 20% test.
    rel = C.TEST_SIZE / (C.TEST_SIZE + C.VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel, stratify=y_temp, random_state=C.SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    np.random.seed(C.SEED)

    raw = download_raw()
    df, clean_log = clean(raw)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)          # fit on TRAIN ONLY
    Xt_val = pre.transform(X_val)
    Xt_test = pre.transform(X_test)
    feat_names = pre.get_feature_names_out().tolist()

    # Persist raw splits (needed by FLAML and by the deployment demo) ...
    for name, Xd, yd in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        out = Xd.copy()
        out["income_binary"] = yd.values
        out.to_csv(C.DATA_PROC / f"raw_{name}.csv", index=False)

    # ... and the transformed matrices.
    np.savez_compressed(
        C.DATA_PROC / "processed_splits.npz",
        X_train=Xt_train.astype(np.float32), y_train=y_train.values.astype(np.float32),
        X_val=Xt_val.astype(np.float32), y_val=y_val.values.astype(np.float32),
        X_test=Xt_test.astype(np.float32), y_test=y_test.values.astype(np.float32),
    )
    joblib.dump(pre, C.ARTIFACTS / "preprocessor.joblib")
    (C.ARTIFACTS / "feature_names.json").write_text(json.dumps(feat_names, indent=2))

    def balance(y):
        vc = pd.Series(y).value_counts().sort_index()
        return {
            "n": int(len(y)),
            "n_le_50k": int(vc.get(0, 0)),
            "n_gt_50k": int(vc.get(1, 0)),
            "pct_gt_50k": round(100 * float(np.mean(y)), 3),
        }

    summary = {
        "seed": C.SEED,
        "cleaning": clean_log,
        "split_scheme": "60% train / 20% validation / 20% test, stratified on income",
        "splits": {
            "train": balance(y_train), "val": balance(y_val), "test": balance(y_test),
            "total": balance(df["income_binary"]),
        },
        "n_features_raw": len(C.FEATURE_COLUMNS),
        "n_features_after_encoding": int(Xt_train.shape[1]),
        "numeric_features": C.NUMERIC_FEATURES,
        "categorical_features": C.CATEGORICAL_FEATURES,
        "categories_per_categorical_feature": {
            col: int(X_train[col].nunique(dropna=True)) for col in C.CATEGORICAL_FEATURES
        },
        "preprocessing_decisions": [
            "Merged adult.data and adult.test into one pool, then re-split ourselves so that "
            "train/val/test come from an identical distribution and the split is reproducible.",
            "Stripped whitespace on all string columns and removed the trailing '.' that "
            "adult.test appends to its labels.",
            "Mapped the target to binary: '<=50K' -> 0, '>50K' -> 1.",
            f"Dropped {C.DROP_COLUMNS}: fnlwgt is a census sampling weight (a survey design "
            "artefact, not a person-level predictor); education is a string duplicate of the "
            "ordinal education-num.",
            f"Dropped {clean_log['duplicate_rows_dropped']} exact duplicate rows to avoid the "
            "same record appearing in both train and test.",
            "'?' tokens were read as NaN and imputed with an explicit 'Missing' category rather "
            "than the mode, because missingness in workclass/occupation is informative "
            "(it co-occurs with 'Never-worked'/retired respondents).",
            "Numeric features: median imputation (robust to the heavy right skew of "
            "capital-gain) followed by StandardScaler, which matters for gradient descent.",
            "Categorical features: OneHotEncoder(handle_unknown='ignore') so an unseen category "
            "at inference yields an all-zero block instead of an error.",
            "The ColumnTransformer is fit on the training split only; validation and test are "
            "transformed with those fitted statistics to prevent data leakage.",
            "Kept capital-gain/capital-loss despite ~91% zeros: they are among the strongest "
            "signals for the >50K class.",
        ],
    }
    (C.RESULTS / "step1_data_summary.json").write_text(json.dumps(summary, indent=2))

    # Human-readable Markdown table.
    lines = ["# Step 1 - Dataset and Preprocessing Summary", "",
             f"Random seed: **{C.SEED}**  |  Split: **60/20/20 stratified**", "",
             "## Split sizes and class balance", "",
             "| Split | Rows | <=50K (0) | >50K (1) | % >50K |", "|---|---|---|---|---|"]
    for k in ("train", "val", "test", "total"):
        s = summary["splits"][k]
        lines.append(f"| {k} | {s['n']:,} | {s['n_le_50k']:,} | {s['n_gt_50k']:,} | {s['pct_gt_50k']}% |")
    lines += ["", f"Raw features used: **{len(C.FEATURE_COLUMNS)}** "
              f"({len(C.NUMERIC_FEATURES)} numeric, {len(C.CATEGORICAL_FEATURES)} categorical)",
              f"Features after one-hot encoding: **{Xt_train.shape[1]}**", "",
              "## Missing values in the raw data", ""]
    if clean_log["missing_per_column"]:
        lines += ["| Column | Missing rows |", "|---|---|"]
        lines += [f"| {k} | {v:,} |" for k, v in clean_log["missing_per_column"].items()]
    lines += ["", "## Preprocessing decisions", ""]
    lines += [f"{i}. {d}" for i, d in enumerate(summary["preprocessing_decisions"], 1)]
    (C.RESULTS / "step1_data_summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(summary["splits"], indent=2))
    print(f"Encoded feature count: {Xt_train.shape[1]}")
    print("Step 1 complete.")


if __name__ == "__main__":
    main()
