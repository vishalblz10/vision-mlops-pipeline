"""Deployment exports: ONNX (for KServe & friends) and dynamic INT8 quantization.

The ONNX artifact is what the KServe InferenceService serves; the quantized
variant is the starting point for on-device targets (the same step where a
Core ML / LiteRT conversion would plug in).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from vision_mlops.data import IMG_SIZE


def export_onnx(model: nn.Module, path: str | Path, opset: int = 17) -> Path:
    """Export the model to ONNX with a dynamic batch dimension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    return path


def verify_onnx(model: nn.Module, path: str | Path, n: int = 8, atol: float = 1e-3) -> float:
    """Check ONNX Runtime outputs match PyTorch; returns the max abs difference."""
    x = torch.rand(n, 3, IMG_SIZE, IMG_SIZE)
    model.eval()
    with torch.no_grad():
        torch_out = model(x).numpy()

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (onnx_out,) = session.run(None, {"images": x.numpy()})

    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > atol:
        raise ValueError(f"ONNX output diverges from PyTorch (max abs diff {max_diff:.2e})")
    return max_diff


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization of the Linear layers (the parameter bulk)."""
    # torch leaves the engine unset ("none") on some platforms, e.g. macOS arm64.
    if torch.backends.quantized.engine == "none":
        for engine in ("fbgemm", "qnnpack"):
            if engine in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = engine
                break
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def serialized_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes
