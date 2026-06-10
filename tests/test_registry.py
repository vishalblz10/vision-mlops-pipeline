import numpy as np
import pytest

from vision_mlops.evaluate import predict_probs
from vision_mlops.registry import (
    load_by_alias,
    log_training_run,
    promote,
    register_model_version,
    resolve_alias,
)


@pytest.fixture()
def tracking_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path}/mlflow.db"


def test_full_registry_lifecycle(trained_model, shapes_data, tracking_uri):
    model, _ = trained_model
    _, _, X_test, _ = shapes_data

    run_id = log_training_run(
        model,
        params={"epochs": 3, "seed": 42},
        metrics={"accuracy": 0.95, "f1_macro": 0.94, "per_class": {"ignored": "dict"}},
        tracking_uri=tracking_uri,
    )
    assert isinstance(run_id, str) and run_id

    version = register_model_version(run_id, "shapes-test")
    assert version == 1

    promote("shapes-test", version, alias="production")
    assert resolve_alias("shapes-test", "production") == 1

    loaded = load_by_alias("shapes-test", "production")
    original = predict_probs(model, X_test[:8])
    restored = predict_probs(loaded, X_test[:8])
    np.testing.assert_allclose(original, restored, atol=1e-5)
