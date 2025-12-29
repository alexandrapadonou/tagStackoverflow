# api/main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from api.inference import InferenceService
import os
from api.model_fetch import ensure_models


MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_BLOB_URL = os.getenv("MODEL_BLOB_URL", "https://tagmodel.blob.core.windows.net/modellight/model_artifacts.zip?sp=rw&st=2025-12-24T11:13:49Z&se=2026-01-09T19:28:49Z&spr=https&sv=2024-11-04&sr=b&sig=Y1DIEEL%2Blg88ujVpN3DiGHbpO1IQaIxeqcaxy%2FqfUWc%3D")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    topk: Optional[int] = Field(None, ge=1, le=50)
    threshold: Optional[float] = None


class PredictResponse(BaseModel):
    tags: List[str]

app = FastAPI(title="StackOverflow Tagger API", version="1.0.0")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
svc = InferenceService(MODEL_DIR)

@app.on_event("startup")
def startup():
    health()
    ensure_models(MODEL_DIR, MODEL_BLOB_URL)
    svc.load()

@app.get("/health")
def health():
    return {"status": "ok", "model_dir": MODEL_DIR}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    tags = svc.predict_tags(req.text, topk=req.topk, threshold=req.threshold)
    return PredictResponse(tags=tags)