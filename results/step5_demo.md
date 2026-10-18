# Step 5 - Deployment Demonstration

Service: FastAPI + uvicorn on `http://127.0.0.1:8000`  |  model: `xgb_limitdepth` (automl_flaml), test ROC-AUC 0.9277

## Five sample predictions (raw, unprocessed input)

| # | Scenario | Prediction | P(>50K) | Client latency (ms) | Notes |
|---|---|---|---|---|---|
| 1 | Young clerical worker, low hours | <=50K | 0.0005 | 5.95 | - |
| 2 | Senior executive, married, long hours | >50K | 0.9996 | 5.40 | - |
| 3 | Mid-career professional, doctorate | >50K | 0.9949 | 4.99 | - |
| 4 | UNSEEN category: native-country='Wakanda' (never in training) | >50K | 0.6317 | 5.24 | unseen_categories: {'native-country': 'Wakanda'} |
| 5 | MISSING fields: occupation and workclass omitted entirely | <=50K | 0.0117 | 5.02 | missing_fields: ['workclass', 'occupation'] |

## End-to-end latency

100 single-record POSTs against a warm server.

| Statistic | Value (ms) |
|---|---|
| mean | 4.913 |
| median | 4.824 |
| p95 | 5.659 |
| min | 4.531 |
| max | 5.847 |
| stdev | 0.292 |
| server side mean | 3.809 |
| http overhead | 1.103 |

**Average end-to-end latency per request: 4.913 ms** (of which 3.809 ms is model + preprocessing and 1.103 ms is HTTP/serialisation overhead).

Batch endpoint: 5 records in 5.471 ms = 1.094 ms/record.

