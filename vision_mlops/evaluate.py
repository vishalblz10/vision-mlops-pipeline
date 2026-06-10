"""Classification evaluation: accuracy, macro precision/recall/F1, confusion matrix."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from vision_mlops.data import CLASSES


@torch.no_grad()
def predict_probs(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i : i + batch_size])
        probs.append(torch.softmax(logits, dim=1))
    return torch.cat(probs)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = len(CLASSES)
) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": {
            CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    probs = predict_probs(model, X)
    y_pred = probs.argmax(dim=1).numpy()
    metrics = compute_metrics(y.numpy(), y_pred)
    metrics["mean_confidence"] = float(probs.max(dim=1).values.mean())
    return metrics
