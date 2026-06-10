import onnxruntime as ort
import torch

from vision_mlops.evaluate import predict_probs
from vision_mlops.export import export_onnx, quantize_dynamic, serialized_size_bytes, verify_onnx


def test_export_creates_file(tmp_path, trained_model):
    model, _ = trained_model
    path = export_onnx(model, tmp_path / "model.onnx")
    assert path.exists()
    assert path.stat().st_size > 0


def test_onnx_parity(tmp_path, trained_model):
    model, _ = trained_model
    path = export_onnx(model, tmp_path / "model.onnx")
    max_diff = verify_onnx(model, path)
    assert max_diff < 1e-3


def test_onnx_dynamic_batch(tmp_path, trained_model, shapes_data):
    model, _ = trained_model
    _, _, X_test, _ = shapes_data
    path = export_onnx(model, tmp_path / "model.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 5):
        (out,) = session.run(None, {"images": X_test[:batch].numpy()})
        assert out.shape == (batch, 3)


def test_quantized_model_smaller_and_agrees(trained_model, shapes_data):
    model, _ = trained_model
    _, _, X_test, _ = shapes_data

    quantized = quantize_dynamic(model)
    assert serialized_size_bytes(quantized) < serialized_size_bytes(model)

    original_pred = predict_probs(model, X_test).argmax(dim=1)
    with torch.no_grad():
        quant_pred = quantized(X_test).argmax(dim=1)
    agreement = float((original_pred == quant_pred).float().mean())
    assert agreement >= 0.9
