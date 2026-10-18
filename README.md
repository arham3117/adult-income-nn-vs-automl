# Intro to AI — Coding Assignment (Take-Home)

A deep feed-forward neural network built from scratch in PyTorch on the **UCI Adult Census
Income** dataset, benchmarked against the **FLAML AutoML** framework on an identical data
split, with the better model deployed as a **FastAPI** inference service.

**Author:** Arham · **Course:** Introduction to Artificial Intelligence

---

## Table of contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Prerequisites](#2-prerequisites)
3. [Setup (step by step)](#3-setup-step-by-step)
4. [Quick start — run the deployed API in ~2 minutes](#4-quick-start--run-the-deployed-api-in-2-minutes)
5. [Full reproduction from scratch](#5-full-reproduction-from-scratch)
6. [Where to find each graded deliverable](#6-where-to-find-each-graded-deliverable)
7. [Project layout](#7-project-layout)
8. [Troubleshooting](#8-troubleshooting)
9. [Notes on findings](#9-notes-on-findings)

---

## 1. Results at a glance

| | Step-2 baseline MLP | Best tuned MLP | **FLAML AutoML** |
|---|---|---|---|
| Test accuracy | 0.8541 | 0.8545 | **0.8722** |
| Test F1 | 0.6696 | 0.6593 | **0.7097** |
| Test ROC-AUC | 0.9104 | 0.9110 | **0.9277** |
| Deployable size | 88 KB | 708 KB | 13.9 MB |
| Single-row latency | 1.16 ms | 1.21 ms | 3.46 ms |

AutoML (`xgb_limitdepth`) wins on every predictive metric and is the model deployed.
The neural network wins on size and speed. Average end-to-end API latency: **~4.9 ms/request**.

**The full write-up is the Word report:**
[`reports/Intro_AI_Coding_Assignment_Report.docx`](reports/Intro_AI_Coding_Assignment_Report.docx)

---

## 2. Prerequisites

### Python version — please read, this is the most common failure

> **This project requires Python 3.9 – 3.12.**
>
> It pins `numpy==2.0.2`, which ships pre-built wheels only up to **CPython 3.12**. On
> **Python 3.13 or newer** `pip` falls back to compiling NumPy from source and the install
> **fails** with `subprocess-exited-with-error`. Verified: installs cleanly on 3.9.6,
> fails on 3.14.8.

Check what you have:

```bash
python3 --version
```

If that prints 3.13 or newer, install a supported version alongside it (this does not
replace your system Python):

- **macOS:** `brew install python@3.12` → then use `python3.12` in step 3.2
- **Ubuntu/Debian:** `sudo apt install python3.12 python3.12-venv` → use `python3.12`
- **Windows:** download 3.12 from [python.org](https://www.python.org/downloads/) → use `py -3.12`
- **Any OS (conda):** `conda create -n introai python=3.12 && conda activate introai`

### System dependency — macOS only

`xgboost` and `lightgbm` link against the OpenMP runtime, which macOS does not ship.
Without it you get `XGBoostError: Library not loaded: @rpath/libomp.dylib`.

```bash
brew install libomp
```

Linux and Windows users can skip this — OpenMP is already present.

### Disk and time

- ~3 GB free (PyTorch alone is ~1 GB)
- Setup: ~5 minutes · Running the API: instant · Full retraining: ~10 minutes

---

## 3. Setup (step by step)

### 3.1 Clone the repository

```bash
git clone https://github.com/arham3117/adult-income-nn-vs-automl.git
cd adult-income-nn-vs-automl
```

### 3.2 Create a virtual environment

A virtualenv keeps these pinned versions from colliding with anything else on your machine.
**Substitute the interpreter you confirmed in step 2** (e.g. `python3.12` instead of `python3`).

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now be prefixed with `(.venv)`. Confirm the version is in range:

```bash
python --version          # must be 3.9.x - 3.12.x
```

> Inside an activated venv, use `python` / `pip` (no `3` suffix) — they now point at the
> venv. This is why the commands below say `python`, not `python3`.

### 3.3 Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Every version is pinned. This matters beyond reproducibility: the saved model bundle is a
pickle of scikit-learn and FLAML objects, and loading it under very different library
versions can warn or fail.

### 3.4 Verify the install

```bash
python -c "import torch, sklearn, flaml, xgboost, fastapi, docx; print('All dependencies OK')"
```

Expected output: `All dependencies OK`

If this is the first command that errors, go to [Troubleshooting](#8-troubleshooting) —
do not continue, every later step depends on it.

---

## 4. Quick start — run the deployed API in ~2 minutes

**No training required.** The trained model is committed at
`artifacts/deployment/model_bundle.joblib`, so the service runs straight from a fresh clone.

### 4.1 Start the server

From the **project root**, with the venv activated:

```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

You should see:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
[startup] bundle loaded: kind=flaml winner=automl_flaml estimator=xgb_limitdepth
```

The `[startup] bundle loaded` line confirms the trained model was found and loaded. It may
appear a moment after the "Uvicorn running" line because of output buffering — that is normal.

### 4.2 Open the interactive docs

Visit **<http://127.0.0.1:8000/docs>** in a browser. This is the auto-generated Swagger UI —
you can expand `POST /predict`, click **Try it out**, and send a request without writing code.

### 4.3 Send a prediction

In a **second terminal** (leave the server running):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":52,"workclass":"Self-emp-inc","education-num":14,
       "marital-status":"Married-civ-spouse","occupation":"Exec-managerial",
       "relationship":"Husband","race":"White","sex":"Male",
       "capital-gain":15024,"capital-loss":0,"hours-per-week":60,
       "native-country":"United-States"}'
```

Expected:

```json
{"prediction":">50K","predicted_class":1,"probability_gt_50k":0.9996, ...}
```

Input is **raw and unprocessed** — the 12 original census columns as plain strings and
numbers. All imputation, encoding and scaling happen server-side.

### 4.4 Run the five-sample demo

Still in the second terminal, with the server running:

```bash
python src/client_demo.py
```

This fires the five sample records from Step 5 of the report — including one with an unseen
category and one with missing fields — and measures average end-to-end latency over 100
requests.

### 4.5 Available endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness; reports which model is loaded and its test ROC-AUC |
| `/schema` | GET | The raw-input contract: columns, known categories, valid ranges |
| `/predict` | POST | One raw record in, one prediction out |
| `/predict_batch` | POST | Many records in one call, with per-record latency |
| `/docs` | GET | Interactive Swagger UI |

Stop the server with `Ctrl+C`.

---

## 5. Full reproduction from scratch

Only needed to regenerate every result, figure and table. **Run from the project root with
the venv activated, in this exact order** — each step reads artefacts the previous one wrote.

```bash
python src/data_prep.py        # Step 1 — download, clean, stratified split, preprocessing  (~1 min)
python src/train.py            # Step 2 — baseline network + full evaluation                (~30 s)
python src/tune.py             # Step 3 — 12-config sweep + seed-noise floor                (~2 min)
python src/automl_flaml.py     # Step 4a — FLAML search (fixed 300 s budget)                (~5 min)
python src/compare.py          # Step 4b — 3-way comparison: size, latency, quality         (~1 min)
python src/export_model.py     # Step 5a — export the winning deployment bundle             (~10 s)
```

Then, to regenerate the deployment evidence — **two terminals**:

```bash
# Terminal 1 — leave running
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000

# Terminal 2
python src/client_demo.py           # 5 predictions + latency  -> results/step5_demo.json
python src/capture_screenshots.py   # screenshots of the live service
```

`capture_screenshots.py` additionally needs a headless browser (one-time, ~90 MB):

```bash
python -m playwright install chromium
```

Finally, rebuild the Word report from the results:

```bash
python make_report.py          # -> reports/Intro_AI_Coding_Assignment_Report.docx
```

**Reproducibility.** Everything is seeded (`SEED = 42` in `src/config.py`) and the split is
deterministic, so metrics reproduce exactly. Wall-clock timings (training time, latency) will
differ on your hardware — that is expected.

> `src/data_prep.py` downloads the dataset from the UCI repository on first run and caches it
> in `data/raw/`. It needs internet access once; it falls back to OpenML (id 1590) if UCI is
> unreachable. `data/` and the large intermediate FLAML artefacts are intentionally not
> committed — re-running the steps above regenerates them.

---

## 6. Where to find each graded deliverable

| Assignment step | Deliverable | Location |
|---|---|---|
| 1 — Data & preprocessing | Split sizes, class balance, every decision | `results/step1_data_summary.md` · report §1 |
| 2 — Network from scratch | Architecture table, params, curves, metrics, confusion matrix | `results/step2_metrics.md` · `figures/step2_*.png` · report §2 |
| 3 — Manual tuning | 12-config table, training times, dominant factor | `results/step3_tuning.md` · `results/step3_tuning.csv` · report §3 |
| 4 — AutoML benchmark | Leaderboard, hyperparameters, 3-way comparison | `results/step4_automl.md` · `results/step4_comparison.md` · report §4 |
| 5 — Deployment | 5 predictions, latency, unseen-feature handling | `results/step5_demo.md` · `reports/screenshots/` · report §5 |
| 6 — Report | Word document, assumptions, tools, screenshots | `reports/Intro_AI_Coding_Assignment_Report.docx` |

Source code for each step is in `src/`, named to match (`data_prep.py`, `train.py`, `tune.py`,
`automl_flaml.py`, `compare.py`, `export_model.py`, `predict.py`, `app.py`).

---

## 7. Project layout

```
adult-income-nn-vs-automl/
├── README.md                     this file
├── requirements.txt              pinned dependencies
├── plan.md                       the plan the work followed + execution log
├── make_report.py                Step 6 — builds the .docx from results/*.json
│
├── src/
│   ├── config.py                 paths, seed, column schema
│   ├── data_prep.py              Step 1 — acquisition, cleaning, split, ColumnTransformer
│   ├── model.py                  configurable MLP + hand-written training loop
│   ├── evaluate.py               metrics and plotting helpers
│   ├── train.py                  Step 2 — baseline network
│   ├── tune.py                   Step 3 — tuning sweep, noise floor, factor analysis
│   ├── automl_flaml.py           Step 4a — FLAML run + leaderboard
│   ├── compare.py                Step 4b — quality / size / latency comparison
│   ├── export_model.py           Step 5a — deployment bundle
│   ├── predict.py                Step 5b — predict() over raw unprocessed input
│   ├── app.py                    Step 5c — FastAPI service
│   ├── client_demo.py            Step 5d — 5 live predictions + latency benchmark
│   └── capture_screenshots.py    Step 5e — screenshots of the running deployment
│
├── artifacts/
│   ├── deployment/
│   │   ├── model_bundle.joblib   the deployed model + preprocessing (committed)
│   │   └── schema.json           raw-input contract
│   ├── preprocessor.joblib       fitted ColumnTransformer
│   └── *.pt                      trained PyTorch checkpoints
│
├── results/                      metrics and tables (JSON + Markdown) per step
├── figures/                      curves, confusion matrices, ROC, comparisons
├── reports/
│   ├── Intro_AI_Coding_Assignment_Report.docx     ← the submitted report
│   └── screenshots/              6 captures of the running deployment
└── docs/Coding_Assignment.pdf    the original assignment brief
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `subprocess-exited-with-error` while installing numpy | Python 3.13+ — no numpy 2.0.2 wheel, tries to compile | Use Python 3.9–3.12 ([§2](#2-prerequisites)) |
| `XGBoostError: Library not loaded: @rpath/libomp.dylib` | macOS missing OpenMP runtime | `brew install libomp` |
| `ModuleNotFoundError: No module named 'torch'` | venv not activated, or installed into a different interpreter | `source .venv/bin/activate`, re-run §3.3 |
| `ModuleNotFoundError: No module named 'src'` | Running from the wrong directory | `cd` to the project root first |
| `ERROR: [Errno 48] Address already in use` | Port 8000 is taken | Use another port: `--port 8001` |
| `FileNotFoundError: model_bundle.joblib` | Bundle missing from the clone | Run `python src/export_model.py` (needs §5 steps 1–4 first) |
| `FileNotFoundError: processed_splits.npz` | Running step 2+ before step 1 | Run `python src/data_prep.py` first |
| Playwright `Executable doesn't exist` | Browser not downloaded | `python -m playwright install chromium` |
| UCI download times out | Network/firewall | Re-run; the code falls back to OpenML automatically |
| `.venv\Scripts\Activate.ps1 cannot be loaded` (Windows) | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

Still stuck? `python -c "import sys; print(sys.version); print(sys.executable)"` confirms
which interpreter is actually running — a mismatch here explains most import errors.

---

## 9. Notes on findings

Three results worth highlighting, all documented in full in the report:

**A FLAML correctness bug, found and fixed.** `flaml.automl.data.DataTransformer.transform()`
re-derives pandas Categorical levels from whatever DataFrame it receives rather than reusing
the levels from fit time. Single-row inference — exactly what an API does — therefore gets
category codes that do not match training, and predictions come back **silently wrong** with
no error (measured: 0.2739 served individually vs 0.9820 in batch for the same record). The
deployment bundle pins the training-time `CategoricalDtype`s and re-casts after transforming.
This fixed single-row serving and also raised batch test ROC-AUC from 0.9266 to 0.9277.
`src/compare.py` asserts row-by-row and batch predictions agree on every run. See report §5.1.

**Manual tuning bought essentially nothing.** The Step-2 baseline is statistically
indistinguishable from the best of 12 tuned configurations (0.9104 vs 0.9110 ROC-AUC, against
a measured seed-only noise band of 0.0006) and has a *better* F1. Before ranking any
hyper-parameter factor, the baseline was retrained under three seeds to establish that noise
floor — only regularisation moved the metric by more than it. See report §3.2–3.3.

**AutoML won on quality, the network won on cost.** FLAML leads by 0.0167 ROC-AUC — real,
since it is ~28× the noise band — but is 3× slower per request and 20× larger on disk than the
tuned network. The report states explicitly that for a latency- or memory-constrained target
the network would be the right thing to ship instead.
