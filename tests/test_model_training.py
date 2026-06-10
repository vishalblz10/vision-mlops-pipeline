import numpy as np
import torch

from vision_mlops.data import CLASSES, IMG_SIZE
from vision_mlops.evaluate import predict_probs
from vision_mlops.model import ShapeNet, load_model, save_model


def test_forward_shape():
    model = ShapeNet()
    x = torch.zeros(4, 3, IMG_SIZE, IMG_SIZE)
    out = model(x)
    assert out.shape == (4, len(CLASSES))


def test_loss_decreases(trained_model):
    _, history = trained_model
    assert history[-1]["loss"] < history[0]["loss"]


def test_holdout_accuracy(trained_model, shapes_data):
    model, _ = trained_model
    _, _, X_test, y_test = shapes_data
    probs = predict_probs(model, X_test)
    accuracy = float((probs.argmax(dim=1) == y_test).float().mean())
    assert accuracy > 0.8


def test_save_load_roundtrip(tmp_path, trained_model, shapes_data):
    model, _ = trained_model
    _, _, X_test, _ = shapes_data
    path = tmp_path / "model.pt"
    save_model(model, path)

    reloaded = load_model(path)
    original = predict_probs(model, X_test[:16])
    restored = predict_probs(reloaded, X_test[:16])
    np.testing.assert_allclose(original, restored, atol=1e-6)
