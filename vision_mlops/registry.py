"""MLflow tracking and model-registry helpers.

Runs are logged to the configured tracking URI; promotion uses registered-model
aliases (the modern replacement for the deprecated stage transitions), so the
serving layer can always resolve ``models:/<name>@production``.
"""

from __future__ import annotations

import mlflow
from mlflow import MlflowClient
from torch import nn

DEFAULT_MODEL_NAME = "shapes-classifier"
DEFAULT_EXPERIMENT = "vision-mlops"


def log_training_run(
    model: nn.Module,
    params: dict,
    metrics: dict,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> str:
    """Log one training run (params, scalar metrics, model artifact); returns run id."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    scalar_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(scalar_metrics)
        mlflow.pytorch.log_model(model, name="model")
        return run.info.run_id


def register_model_version(run_id: str, model_name: str = DEFAULT_MODEL_NAME) -> int:
    """Register the run's model artifact as a new version; returns the version."""
    version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    return int(version.version)


def promote(model_name: str, version: int, alias: str = "production") -> None:
    """Point the alias (e.g. ``production``) at a registered version."""
    MlflowClient().set_registered_model_alias(model_name, alias, str(version))


def resolve_alias(model_name: str = DEFAULT_MODEL_NAME, alias: str = "production") -> int:
    """Return the version the alias currently points at."""
    return int(MlflowClient().get_model_version_by_alias(model_name, alias).version)


def load_by_alias(model_name: str = DEFAULT_MODEL_NAME, alias: str = "production") -> nn.Module:
    """Load the aliased model version back as a PyTorch module."""
    return mlflow.pytorch.load_model(f"models:/{model_name}@{alias}")
