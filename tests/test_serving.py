import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vision_mlops.data import CLASSES, IMG_SIZE, make_dataset
from vision_mlops.model import save_model
from vision_mlops.serving.api import create_app


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    from vision_mlops.train import train_model

    X, y = make_dataset(300, seed=42)
    model, _ = train_model(X, y, epochs=2, seed=42)
    path = tmp_path_factory.mktemp("serving") / "model.pt"
    save_model(model, path)
    return path


@pytest.fixture()
def client(model_path):
    with TestClient(create_app(str(model_path))) as test_client:
        yield test_client


def png_bytes() -> bytes:
    X, _ = make_dataset(1, seed=11)
    array = (X[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readyz_when_loaded(client):
    assert client.get("/readyz").status_code == 200


def test_predict(client):
    response = client.post("/predict", files={"file": ("img.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == set(CLASSES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_resizes_any_image(client):
    image = Image.new("RGB", (IMG_SIZE * 4, IMG_SIZE * 2), color=(200, 50, 50))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    response = client.post("/predict", files={"file": ("img.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code == 200


def test_garbage_upload_is_422(client):
    response = client.post("/predict", files={"file": ("junk.bin", b"not an image", "application/octet-stream")})
    assert response.status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", files={"file": ("img.png", png_bytes(), "image/png")})
    text = client.get("/metrics").text
    assert "vmops_predictions_total" in text
    assert "vmops_request_latency_seconds" in text
    assert "vmops_model_ready 1.0" in text


def test_missing_model_not_ready(tmp_path):
    with TestClient(create_app(str(tmp_path / "missing.pt"))) as client:
        assert client.get("/readyz").status_code == 503
        response = client.post("/predict", files={"file": ("img.png", png_bytes(), "image/png")})
        assert response.status_code == 503
