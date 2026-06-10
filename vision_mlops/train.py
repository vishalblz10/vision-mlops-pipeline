"""Seeded training loop for the reference model."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vision_mlops.model import ShapeNet


def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[ShapeNet, list[dict]]:
    """Train a ShapeNet on ``(X, y)``; returns the model and per-epoch history."""
    torch.manual_seed(seed)
    model = ShapeNet()
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    model.train()
    for epoch in range(epochs):
        total_loss, correct, seen = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            correct += int((logits.argmax(dim=1) == yb).sum())
            seen += len(xb)
        history.append(
            {"epoch": epoch + 1, "loss": total_loss / seen, "accuracy": correct / seen}
        )

    model.eval()
    return model, history
