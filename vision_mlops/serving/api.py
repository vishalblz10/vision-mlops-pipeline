"""FastAPI model server with Prometheus instrumentation.

Endpoints map onto the Kubernetes deployment in ``deploy/``: ``/healthz`` is
the liveness probe, ``/readyz`` the readiness probe (503 until a model is
loaded), ``/metrics`` is scraped by Prometheus, and ``/predict`` serves
predictions for an uploaded image.
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from vision_mlops import __version__
from vision_mlops.data import CLASSES, IMG_SIZE
from vision_mlops.model import load_model

PREDICTIONS = Counter(
    "vmops_predictions_total",
    "Predictions served, labelled by predicted class.",
    ["predicted_class"],
)
LATENCY = Histogram(
    "vmops_request_latency_seconds",
    "End-to-end /predict latency.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
CONFIDENCE = Histogram(
    "vmops_prediction_confidence",
    "Top-1 softmax confidence of served predictions (low values hint at drift).",
    buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
MODEL_READY = Gauge("vmops_model_ready", "1 when a model is loaded and serving.")

DEFAULT_MODEL_PATH = "models/model.pt"


def create_app(model_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        app.state.model = load_model(path) if os.path.exists(path) else None
        MODEL_READY.set(1 if app.state.model is not None else 0)
        yield
        MODEL_READY.set(0)

    app = FastAPI(title="vision-mlops model server", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "alive"}

    @app.get("/readyz")
    def readyz() -> dict:
        if app.state.model is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        if app.state.model is None:
            raise HTTPException(503, "model not loaded")

        start = time.perf_counter()
        payload = await file.read()
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception:
            raise HTTPException(422, "could not decode the uploaded file as an image")

        image = image.resize((IMG_SIZE, IMG_SIZE))
        x = (
            torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        with torch.no_grad():
            probs = torch.softmax(app.state.model(x), dim=1)[0]

        class_id = int(probs.argmax())
        confidence = float(probs[class_id])
        label = CLASSES[class_id]
        elapsed = time.perf_counter() - start

        PREDICTIONS.labels(predicted_class=label).inc()
        LATENCY.observe(elapsed)
        CONFIDENCE.observe(confidence)

        return {
            "label": label,
            "confidence": round(confidence, 6),
            "probabilities": {CLASSES[i]: round(float(p), 6) for i, p in enumerate(probs)},
            "latency_ms": round(elapsed * 1000, 2),
            "model_version": __version__,
        }

    return app


app = create_app()
