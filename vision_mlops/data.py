"""Synthetic shapes dataset for the reference computer-vision model.

Images are 32x32 RGB renders of a single bright shape (circle, square or
triangle) on a dark noisy background. The generator is fully seeded, so every
pipeline stage — training, evaluation, drift simulation — is reproducible on
any machine without downloading data.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

CLASSES = ("circle", "square", "triangle")
IMG_SIZE = 32


def _render_shape(class_id: int, rng: np.random.Generator, brightness_shift: float) -> np.ndarray:
    background = tuple(int(c) for c in rng.integers(0, 60, size=3))
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), background)
    draw = ImageDraw.Draw(img)

    color = tuple(int(c) for c in rng.integers(120, 256, size=3))
    size = int(rng.integers(10, 24))
    x0 = int(rng.integers(2, IMG_SIZE - size - 2))
    y0 = int(rng.integers(2, IMG_SIZE - size - 2))
    x1, y1 = x0 + size, y0 + size

    shape = CLASSES[class_id]
    if shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=color)
    elif shape == "square":
        draw.rectangle([x0, y0, x1, y1], fill=color)
    else:  # triangle
        draw.polygon([(x0, y1), (x1, y1), ((x0 + x1) // 2, y0)], fill=color)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    noise = rng.normal(0.0, 0.02, size=arr.shape).astype(np.float32)
    return np.clip(arr + noise + brightness_shift, 0.0, 1.0)


def make_dataset(
    n: int, seed: int = 42, brightness_shift: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``n`` labelled images as ``(X, y)`` tensors.

    ``X`` has shape ``(n, 3, 32, 32)`` in ``[0, 1]``; ``y`` holds class ids.
    ``brightness_shift`` uniformly brightens (or darkens) every image — used to
    simulate upstream data drift such as a misconfigured camera exposure.
    """
    rng = np.random.default_rng(seed)
    images = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        class_id = int(rng.integers(0, len(CLASSES)))
        images[i] = _render_shape(class_id, rng, brightness_shift)
        labels[i] = class_id

    X = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(labels)
    return X, y


def extract_features(X: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-image scalar features that production monitoring watches for drift."""
    arr = X.detach().cpu().numpy()
    return {
        "brightness": arr.mean(axis=(1, 2, 3)),
        "contrast": arr.std(axis=(1, 2, 3)),
    }
