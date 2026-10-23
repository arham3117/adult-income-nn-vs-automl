"""Step 5 (part 3) - FastAPI inference service.

Run locally:
    python3 -m uvicorn src.app:app --host 127.0.0.1 --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from predict import load_bundle, predict

app = FastAPI(
    title="Adult Census Income - Inference API",
    description=("Serves the model that won the Step-4 benchmark. Accepts RAW, "
                 "unprocessed census fields; all preprocessing is applied server side."),
    version="1.0.0",
)


class RawRecord(BaseModel):
    """The 12 raw columns. Everything is optional so the service can demonstrate
    its missing-field handling; strings are accepted exactly as in the raw file."""
    age: Optional[Any] = Field(None, examples=[39])
    workclass: Optional[str] = Field(None, examples=["State-gov"])
    education_num: Optional[Any] = Field(None, alias="education-num", examples=[13])
    marital_status: Optional[str] = Field(None, alias="marital-status",
                                          examples=["Never-married"])
    occupation: Optional[str] = Field(None, examples=["Adm-clerical"])
    relationship: Optional[str] = Field(None, examples=["Not-in-family"])
    race: Optional[str] = Field(None, examples=["White"])
    sex: Optional[str] = Field(None, examples=["Male"])
    capital_gain: Optional[Any] = Field(None, alias="capital-gain", examples=[2174])
    capital_loss: Optional[Any] = Field(None, alias="capital-loss", examples=[0])
    hours_per_week: Optional[Any] = Field(None, alias="hours-per-week", examples=[40])
    native_country: Optional[str] = Field(None, alias="native-country",
                                          examples=["United-States"])

    model_config = {"populate_by_name": True, "extra": "allow"}

    def to_raw(self) -> dict:
        # by_alias gives back the hyphenated names like 'education-num'
        return {k: v for k, v in self.model_dump(by_alias=True).items() if k != "extra"}


class BatchRequest(BaseModel):
    records: list[RawRecord]
    threshold: Optional[float] = None


@app.on_event("startup")
def _warm() -> None:
    """Load and warm the model at startup so the first request is not an outlier."""
    b = load_bundle()
    predict({"age": 39}, bundle=b)
    print(f"[startup] bundle loaded: kind={b['kind']} "
          f"winner={b['winner']} estimator={b.get('estimator_name')}")


@app.get("/health")
def health() -> dict:
    b = load_bundle()
    return {"status": "ok", "model_kind": b["kind"], "winner": b["winner"],
            "estimator": b.get("estimator_name"),
            "test_roc_auc": round(b["test_metrics"]["roc_auc"], 4)}


@app.get("/schema")
def get_schema() -> dict:
    return load_bundle()["schema"]


@app.post("/predict")
def predict_one(record: RawRecord, threshold: Optional[float] = None) -> dict:
    t0 = time.perf_counter()
    try:
        result = predict(record.to_raw(), bundle=load_bundle(), threshold=threshold)[0]
    except Exception as exc:  # surface the reason instead of a bare 500
        raise HTTPException(status_code=400, detail=f"prediction failed: {exc}") from exc
    result["server_latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    return result


@app.post("/predict_batch")
def predict_batch(req: BatchRequest) -> dict:
    t0 = time.perf_counter()
    try:
        results = predict([r.to_raw() for r in req.records], bundle=load_bundle(),
                          threshold=req.threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"prediction failed: {exc}") from exc
    total = (time.perf_counter() - t0) * 1000
    return {"n": len(results), "predictions": results,
            "server_latency_ms": round(total, 3),
            "server_latency_ms_per_record": round(total / max(len(results), 1), 3)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
