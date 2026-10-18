# Step 4 - Hand-built Network vs. AutoML

Step-2 baseline MLP  |  best tuned MLP (`C3-wide`)  |  AutoML: FLAML, selected `xgb_limitdepth`

| Metric | Step-2 baseline MLP | Best tuned MLP | FLAML AutoML |
|---|---|---|---|
| Test accuracy | 0.8541 | 0.8545 | 0.8722 |
| Test precision | 0.731 | 0.75 | 0.7779 |
| Test recall | 0.6177 | 0.5882 | 0.6524 |
| Test F1 | 0.6696 | 0.6593 | 0.7097 |
| Test ROC-AUC | 0.9104 | 0.911 | 0.9277 |
| Final-model fit time (s) | 1.96 | 2.83 | 6.36 |
| Total search time (s) | 1.96 | 30.09 | 300.1 |
| Model file size (KB) | 82.0 | 701.5 | 14186.8 |
| Deployable size incl. preprocessing (KB) | 88.1 | 707.6 | 14192.3 |
| Full pickled search object (KB) | n/a | n/a | 34610.7 |
| Latency, single row (ms, mean) | 1.1574 | 1.2097 | 3.4567 |
| Latency, single row (ms, p95) | 1.2642 | 1.3715 | 3.7955 |
| Latency, batch of 9,758 (ms, mean) | 7.2532 | 9.9448 | 65.0028 |
| Throughput, batch (rows/s) | 1345337 | 981216 | 150117 |
| Trainable parameters / trees | 20097 | 178689 | 2770 |

**Winner: FLAML AutoML** (test ROC-AUC margin 0.0167; seed-only noise band is 0.0006, so the margin is larger than run-to-run noise).

Primary criterion: test ROC-AUC (threshold-independent and robust to the 24/76 class imbalance). F1 is used as a tie-break, then deployable size and latency.

