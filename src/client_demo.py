"""Step 5 (part 4) - Drive the live API: 5 sample predictions + latency.

Latency here is END TO END from the client: HTTP request out, JSON response in,
measured against a server that is already warm.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

BASE = "http://127.0.0.1:8000"
N_TIMED = 100

# Five raw records, chosen to span the decision space and the edge cases.
SAMPLES = [
    {"_label": "1. Young clerical worker, low hours",
     "age": 25, "workclass": "Private", "education-num": 9,
     "marital-status": "Never-married", "occupation": "Other-service",
     "relationship": "Own-child", "race": "White", "sex": "Female",
     "capital-gain": 0, "capital-loss": 0, "hours-per-week": 25,
     "native-country": "United-States"},
    {"_label": "2. Senior executive, married, long hours",
     "age": 52, "workclass": "Self-emp-inc", "education-num": 14,
     "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
     "relationship": "Husband", "race": "White", "sex": "Male",
     "capital-gain": 15024, "capital-loss": 0, "hours-per-week": 60,
     "native-country": "United-States"},
    {"_label": "3. Mid-career professional, doctorate",
     "age": 41, "workclass": "Federal-gov", "education-num": 16,
     "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty",
     "relationship": "Husband", "race": "Asian-Pac-Islander", "sex": "Male",
     "capital-gain": 0, "capital-loss": 1902, "hours-per-week": 45,
     "native-country": "India"},
    {"_label": "4. UNSEEN category: native-country='Wakanda' (never in training)",
     "age": 37, "workclass": "Private", "education-num": 13,
     "marital-status": "Married-civ-spouse", "occupation": "Tech-support",
     "relationship": "Husband", "race": "Black", "sex": "Male",
     "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40,
     "native-country": "Wakanda"},
    {"_label": "5. MISSING fields: occupation and workclass omitted entirely",
     "age": 60, "education-num": 7, "marital-status": "Widowed",
     "relationship": "Not-in-family", "race": "White", "sex": "Female",
     "capital-gain": 0, "capital-loss": 0, "hours-per-week": 20,
     "native-country": "United-States"},
]


def main() -> None:
    health = requests.get(f"{BASE}/health", timeout=30).json()
    print("Health:", json.dumps(health), "\n")

    demos = []
    for s in SAMPLES:
        label = s["_label"]
        payload = {k: v for k, v in s.items() if not k.startswith("_")}
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/predict", json=payload, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        body = r.json()
        demos.append({"label": label, "request": payload, "response": body,
                      "client_latency_ms": round(elapsed, 3)})
        print(f"{label}\n   -> {body['prediction']} "
              f"(P(>50K)={body['probability_gt_50k']:.4f}) "
              f"| server {body['server_latency_ms']:.2f} ms | client {elapsed:.2f} ms")
        if body.get("input_warnings"):
            print(f"      warnings: {body['input_warnings']}")
        print()

    # ---- end-to-end latency over a warm server ------------------------------
    warm = {k: v for k, v in SAMPLES[1].items() if not k.startswith("_")}
    for _ in range(10):
        requests.post(f"{BASE}/predict", json=warm, timeout=30)

    times, server_times = [], []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        resp = requests.post(f"{BASE}/predict", json=warm, timeout=30)
        times.append((time.perf_counter() - t0) * 1000)
        server_times.append(resp.json()["server_latency_ms"])

    times_sorted = sorted(times)
    latency = {
        "n_requests": N_TIMED,
        "endpoint": "/predict (single record)",
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "p95_ms": round(times_sorted[int(0.95 * N_TIMED) - 1], 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "stdev_ms": round(statistics.stdev(times), 3),
        "server_side_mean_ms": round(statistics.mean(server_times), 3),
        "http_overhead_ms": round(statistics.mean(times) - statistics.mean(server_times), 3),
    }

    # Batch endpoint for throughput context.
    batch_payload = {"records": [{k: v for k, v in s.items() if not k.startswith("_")}
                                 for s in SAMPLES]}
    t0 = time.perf_counter()
    br = requests.post(f"{BASE}/predict_batch", json=batch_payload, timeout=60).json()
    batch_ms = (time.perf_counter() - t0) * 1000
    batch = {"n": br["n"], "client_total_ms": round(batch_ms, 3),
             "client_ms_per_record": round(batch_ms / br["n"], 3),
             "server_ms_per_record": br["server_latency_ms_per_record"]}

    out = {"health": health, "samples": demos, "latency": latency, "batch": batch}
    (C.RESULTS / "step5_demo.json").write_text(json.dumps(out, indent=2))

    L = ["# Step 5 - Deployment Demonstration", "",
         f"Service: FastAPI + uvicorn on `{BASE}`  |  model: "
         f"`{health['estimator']}` ({health['winner']}), test ROC-AUC {health['test_roc_auc']}", "",
         "## Five sample predictions (raw, unprocessed input)", "",
         "| # | Scenario | Prediction | P(>50K) | Client latency (ms) | Notes |",
         "|---|---|---|---|---|---|"]
    for i, d in enumerate(demos, 1):
        w = d["response"].get("input_warnings") or {}
        note = "; ".join(f"{k}: {v}" for k, v in w.items()) if w else "-"
        L.append(f"| {i} | {d['label'].split('. ', 1)[1]} | "
                 f"{d['response']['prediction']} | "
                 f"{d['response']['probability_gt_50k']:.4f} | "
                 f"{d['client_latency_ms']:.2f} | {note} |")
    L += ["", "## End-to-end latency", "",
          f"{N_TIMED} single-record POSTs against a warm server.", "",
          "| Statistic | Value (ms) |", "|---|---|"]
    for k in ("mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "stdev_ms",
              "server_side_mean_ms", "http_overhead_ms"):
        L.append(f"| {k.replace('_ms','').replace('_',' ')} | {latency[k]} |")
    L += ["", f"**Average end-to-end latency per request: {latency['mean_ms']} ms** "
          f"(of which {latency['server_side_mean_ms']} ms is model + preprocessing and "
          f"{latency['http_overhead_ms']} ms is HTTP/serialisation overhead).", "",
          f"Batch endpoint: {batch['n']} records in {batch['client_total_ms']} ms "
          f"= {batch['client_ms_per_record']} ms/record.", ""]
    (C.RESULTS / "step5_demo.md").write_text("\n".join(L) + "\n")

    print(f"Average end-to-end latency: {latency['mean_ms']} ms "
          f"(median {latency['median_ms']}, p95 {latency['p95_ms']})")
    print(f"  server-side {latency['server_side_mean_ms']} ms | "
          f"HTTP overhead {latency['http_overhead_ms']} ms")
    print("Step 5 demo complete.")


if __name__ == "__main__":
    main()
