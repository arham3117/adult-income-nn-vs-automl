# Plan — Intro to AI Coding Assignment (Take-home, 15 pts)

## 0. Source requirements (from `docs/Coding_Assignment.pdf`)

Build a deep neural network **from scratch** on the UCI Adult Census Income dataset, benchmark it
against an **AutoML** system, and **deploy** the better model as a working inference service.
Six graded steps:

| # | Requirement | Pts |
|---|---|---|
| 1 | Obtain Adult dataset (UCI id 2 / OpenML 1590). Reproducible preprocessing: missing values, categorical encoding, numeric scaling, **stratified** train/val/test split, fixed seed. Report split sizes, class balance, every preprocessing decision. | 3 |
| 2 | Deep feed-forward NN in PyTorch/TensorFlow — own architecture, loss, optimizer, training loop. No pretrained models, no AutoML. **≥2 hidden layers**. Report (a) architecture table + trainable-parameter count, (b) train/val loss & accuracy curves, (c) test accuracy, precision, recall, F1, ROC-AUC + confusion matrix. | 3 |
| 3 | Manual tuning: **≥8 configurations** varying depth/width, learning rate, batch size, regularization (dropout, weight decay, batch norm, early stopping). Table of all configs with validation metrics + training time. State which factor mattered most and why. | 3 |
| 4 | AutoML (AutoGluon / FLAML / auto-sklearn) on **exactly the same split**. Report leaderboard, selected model + hyperparameters. Compare vs. hand-built NN on accuracy, F1, ROC-AUC, training time, **model size on disk**, **inference latency**. | 3 |
| 5 | Deploy the better model. Export model **+ preprocessing pipeline**, write a `predict` function that accepts **raw unprocessed input**, wrap in **FastAPI** (Colab/localhost). Demonstrate **5 sample predictions**, measure **average end-to-end latency per request**, describe handling of an **unseen feature value** at inference. | 3 |
| 6 | Word-document report answering each step: every assumption (with reasoning), every tool (with reasoning), screenshot of running deployment. Submit document + code as a **zip**. | — |

Note from the PDF: AI tools/resources are allowed, but the submitted report must reflect our own
views and experimental findings.

## 1. Environment & tooling decisions

- **Python 3.9.6** (system interpreter). Already present: `torch 2.8.0`, `scikit-learn 1.6.1`,
  `pandas 2.3.3`, `numpy 2.0.2`, `matplotlib`, `seaborn`, `fastapi`, `pydantic 2`, `joblib`,
  `python-docx`, `requests`.
- **To install:** `uvicorn` (ASGI server for FastAPI), `flaml[automl]` (AutoML — chosen over
  AutoGluon/auto-sklearn because it is lightweight, installs cleanly on Py3.9/macOS-arm64, and
  exposes an explicit time budget + leaderboard).
- **Framework choice: PyTorch** — explicit training loop is required by step 2, and PyTorch makes the
  hand-written loop (forward/backward/step) visible and auditable.
- **Data source: direct UCI download** (`archive.ics.uci.edu/static/public/2/adult.zip`) with an
  OpenML fallback; avoids a hard dependency on the `openml`/`ucimlrepo` packages.
- Fixed global seed **42** everywhere (`random`, `numpy`, `torch`), deterministic split.

## 2. Repository layout to create

```
Intro-AI/
├── plan.md
├── requirements.txt
├── README.md
├── docs/Coding_Assignment.pdf
├── data/            raw/ (downloaded zip+csv), processed/ (split parquet/npz)
├── src/
│   ├── config.py          paths, seed, hyperparameter grids
│   ├── data_prep.py       step 1 — download, clean, split, fit preprocessor
│   ├── model.py           configurable MLP definition
│   ├── train.py           step 2 — baseline training loop + full evaluation
│   ├── tune.py            step 3 — 8+ configuration sweep
│   ├── automl_flaml.py    step 4 — FLAML run + leaderboard
│   ├── compare.py         step 4 — head-to-head comparison table
│   ├── export_model.py    step 5 — bundle winner + preprocessor
│   ├── predict.py         step 5 — raw-input predict function
│   ├── app.py             step 5 — FastAPI service
│   └── client_demo.py     step 5 — 5 sample requests + latency measurement
├── artifacts/       models, preprocessor, metrics JSON, deployment bundle
├── figures/         curves, confusion matrices, ROC, comparison charts
├── results/         CSV/Markdown tables
├── reports/         report .docx + deployment screenshots
└── make_report.py   step 6 — assembles the Word document
```

## 3. Step-by-step execution

### Step 1 — Data & preprocessing (`src/data_prep.py`)
1. Download `adult.zip` from UCI → `data/raw/`; parse `adult.data` + `adult.test` with the 15 column
   names; fall back to OpenML 1590 if the download fails.
2. Clean: strip whitespace, normalise the test-split label (`<=50K.` → `<=50K`), map target to
   `{<=50K:0, >50K:1}`, drop exact duplicate rows, treat `?` as `NaN`.
3. Decisions to record: drop `fnlwgt` (survey sampling weight, not a person-level predictor);
   keep `education-num` and drop the redundant string `education`; keep `capital-gain`/`capital-loss`
   despite heavy zero-inflation.
4. Missing values: `workclass`, `occupation`, `native-country` only → impute with the explicit
   category `"Missing"` (informative missingness) rather than the mode. Numeric → median imputer
   (defensive; none missing in practice).
5. Encode: `OneHotEncoder(handle_unknown="ignore")` — this is also the mechanism that makes an
   unseen category safe at inference (step 5 question). Scale numerics with `StandardScaler`.
6. Split **60/20/20 stratified** on the target with `random_state=42`, done in two calls
   (train+temp, then val/test). **Fit the preprocessor on train only**, transform val/test.
7. Persist: raw split frames, transformed matrices, fitted `ColumnTransformer`, and a
   `results/step1_data_summary.md` + JSON with split sizes, class balance per split, feature counts.

### Step 2 — Baseline network (`src/model.py`, `src/train.py`)
1. `MLP` class: configurable hidden sizes, dropout, batch-norm, ReLU. Baseline = **2 hidden layers
   (128 → 64)**, dropout 0.3, `BCEWithLogitsLoss`, Adam lr 1e-3, batch 256, up to 100 epochs with
   early stopping (patience 10) on validation loss.
2. Hand-written training loop: manual `zero_grad` / `forward` / `loss.backward()` / `step`, per-epoch
   train & val loss/accuracy history.
3. Outputs: architecture table (layer, output units, activation, params) + total trainable params;
   `figures/step2_curves.png` (loss + accuracy); test accuracy, precision, recall, F1, ROC-AUC,
   confusion matrix (`figures/step2_confusion_matrix.png`, `figures/step2_roc.png`);
   `artifacts/baseline_model.pt`, `results/step2_metrics.json`.

### Step 3 — Manual tuning sweep (`src/tune.py`)
1. **10 configurations** spanning: depth (2/3/4 hidden layers), width (64→512), learning rate
   (1e-2 / 1e-3 / 3e-4), batch size (64 / 256 / 1024), dropout (0.0 / 0.2 / 0.5), weight decay
   (0 / 1e-4 / 1e-2), batch-norm on/off, early stopping on/off. Each config is a controlled
   variation so the effect of each factor is attributable.
2. Record per config: val accuracy, F1, ROC-AUC, val loss, epochs run, wall-clock training time,
   parameter count. Save `results/step3_tuning.csv` + Markdown table, plus a comparison figure.
3. Analyse which factor mattered most (expected: learning rate / regularization) and write the
   reasoning into the results file; retrain and persist the best config as
   `artifacts/best_nn_model.pt` with its test metrics.

### Step 4 — AutoML benchmark (`src/automl_flaml.py`, `src/compare.py`)
1. Run FLAML `AutoML.fit` on the **same raw train split** (FLAML does its own encoding; the
   *rows* are identical, which is what "same split" requires), `metric="roc_auc"`,
   `task="classification"`, fixed time budget (e.g. 300 s), `seed=42`, validation via the same
   held-out val set (`X_val`/`y_val` passed explicitly, not resplit).
2. Report: leaderboard of all estimators tried (from `automl.best_config_per_estimator` +
   the FLAML log file), selected model, its hyperparameters, and its test metrics.
3. `compare.py`: head-to-head table — accuracy, F1, ROC-AUC, training time, model size on disk
   (bytes of the saved artifact incl. preprocessor), inference latency (mean ms/row over N
   repetitions, batch and single-row). Bar-chart figure. Pick the winner (primary criterion:
   test ROC-AUC, tie-broken by F1, then latency/size).

### Step 5 — Deployment (`src/export_model.py`, `src/predict.py`, `src/app.py`, `src/client_demo.py`)
1. Export a **single bundle**: winner model + fitted preprocessor + feature schema + metadata
   (`artifacts/deployment/`).
2. `predict_raw(records)`: accepts raw unprocessed dicts (the 14 original columns, strings not
   encoded), applies the exact training preprocessor, returns label + probability. Handles unseen
   categories via `handle_unknown="ignore"`, missing keys via the `"Missing"` category / median
   imputation, and validates/coerces types.
3. FastAPI app: `GET /health`, `GET /schema`, `POST /predict` (single), `POST /predict_batch`,
   Pydantic request model that permits raw string inputs. Run with uvicorn on localhost:8000.
4. `client_demo.py`: fires **5 sample raw records** (including one with an unseen
   `native-country` value to prove graceful handling), prints predictions, measures average
   end-to-end latency per request over a warm-up + N timed runs → `results/step5_demo.json`.
5. Capture deployment screenshots (terminal with uvicorn running + `/docs` Swagger page +
   demo output) into `reports/screenshots/`.

### Step 6 — Report & submission (`make_report.py`)
1. Generate `reports/Intro_AI_Coding_Assignment_Report.docx` with `python-docx`: one section per
   step, all tables, all figures embedded, deployment screenshots, plus dedicated
   **Assumptions** (with reasoning) and **Tools Used** (with reasoning) sections, and a short
   reflection/conclusion.
2. Zip code + report → `submission/Intro_AI_Coding_Assignment.zip` (excluding raw data and
   virtualenv bulk, including a README with run instructions).

## 4. Execution order & verification

1. Install `uvicorn`, `flaml[automl]`; write `requirements.txt`.
2. Run step 1 → verify split sizes/class balance printed and artifacts written.
3. Run step 2 → verify curves + metrics files exist; sanity-check accuracy ≈ 0.85+.
4. Run step 3 → verify ≥8 rows in the tuning table; best config retrained.
5. Run step 4 → verify leaderboard + comparison table.
6. Run step 5 → start uvicorn in background, run client demo, confirm 5 predictions + latency,
   screenshot, stop server.
7. Run step 6 → verify the .docx opens and contains every figure/table; build the zip.

## 5. Risks & mitigations

- **FLAML install on Py3.9/arm64** — if `flaml[automl]` fails, fall back to `flaml` core with
  lightgbm/xgboost installed separately; if AutoML is entirely unavailable, substitute
  auto-sklearn → last-resort documented fallback is a scikit-learn `HalvingRandomSearchCV`
  ensemble search, clearly labelled as a deviation.
- **UCI download blocked** — OpenML 1590 fallback via `sklearn.datasets.fetch_openml`.
- **Long runtimes** — cap the tuning sweep with early stopping and a bounded FLAML time budget so
  the whole pipeline reruns end-to-end in minutes.
- **Screenshots on headless run** — capture with macOS `screencapture` while uvicorn is live; if
  unavailable, save rendered terminal/Swagger output as images.

---

# Execution log — what actually happened

All six steps completed. Deviations from the plan above, and why:

| Planned | Actual | Reason |
|---|---|---|
| ≥8 tuning configurations | **12** | Needed 3 levels per factor to attribute effects cleanly. |
| — | **Added a seed-noise floor** (baseline retrained under 3 seeds) | Configurations landed within ~0.001 ROC-AUC of each other; without a noise measurement the "most important factor" ranking would have been noise. |
| — | **Fixed the no-regularisation ablation** | The first version restored best-validation weights even when `early_stopping=False`, so the ablation silently kept early stopping's main benefit and showed no effect. After the fix, regularisation went from indistinguishable to the dominant factor. |
| FLAML with defaults | `retrain_full=False`, explicit `X_val`/`y_val` | Defaults would have refit on train+val and re-split internally, giving AutoML ~33% more data and breaking the "same split" requirement. |
| Model size = saved artifact | Lean bundle measured separately from the 33.8 MB pickled search object | The full `AutoML` pickle is mostly search state (`model_history`); reporting it as model size would unfairly penalise AutoML. |
| — | **Found and fixed a FLAML inference bug** | `DataTransformer.transform()` re-derives Categorical levels per call, so single-row serving was silently wrong (0.2739 vs 0.9820 for the same record). Pinning training-time dtypes fixed it and raised batch test ROC-AUC 0.9266 → 0.9277. |
| macOS `screencapture` screenshots | Headless Chromium (Playwright) + rendered console output | OS-level screen capture was blocked by macOS permissions; the browser images are genuine renders of the live server. |

## Results

| | Hand-built MLP | FLAML AutoML (`xgb_limitdepth`) |
|---|---|---|
| Test accuracy | 0.8545 | **0.8722** |
| Test F1 | 0.6593 | **0.7097** |
| Test ROC-AUC | 0.9110 | **0.9277** |
| Deployable size | **0.71 MB** | 13.86 MB |
| Single-row latency | **1.20 ms** | 3.51 ms |

- Step 1: 48,790 rows after cleaning (52 duplicates dropped), 60/20/20 stratified, class balance 23.94% in every split, 91 encoded features.
- Step 2: 2 hidden layers (128→64), 20,097 trainable parameters, test ROC-AUC 0.9104.
- Step 3: regularisation was the only factor whose effect (0.0247 AUC) clearly exceeded the 0.0006 seed-noise band.
- Step 4: FLAML tried 6 estimator families; `xgb_limitdepth` won. AutoML wins on quality by 0.0167 AUC; the network wins on size (20×) and latency (3×).
- Step 5: FLAML model deployed via FastAPI; 5 live predictions, **average end-to-end latency 4.9–5.1 ms/request**.
- Step 6: `reports/Intro_AI_Coding_Assignment_Report.docx` (20 tables, 16 figures, 6 deployment screenshots); `submission/Intro_AI_Coding_Assignment.zip` (8.9 MB, 69 files).
