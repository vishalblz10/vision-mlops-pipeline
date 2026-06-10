"""Reference PyTorch model: a compact CNN classifier for the shapes dataset."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from vision_mlops.data import CLASSES, IMG_SIZE


class ShapeNet(nn.Module):
    """Three conv blocks plus a small MLP head; ~160k parameters, CPU-friendly."""

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE // 8) ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def save_model(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": list(CLASSES)}, path)
    return path


def load_model(path: str | Path) -> ShapeNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ShapeNet(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
