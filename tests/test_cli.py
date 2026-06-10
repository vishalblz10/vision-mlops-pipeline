import json

from vision_mlops.cli import main


def test_train_writes_model_and_metrics(tmp_path):
    model_path = tmp_path / "model.pt"
    metrics_path = tmp_path / "metrics.json"
    code = main(
        [
            "train",
            "--samples", "300",
            "--holdout", "100",
            "--epochs", "2",
            "--out", str(model_path),
            "--metrics-out", str(metrics_path),
        ]
    )
    assert code == 0
    assert model_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "accuracy" in metrics


def test_gate_passes_on_good_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "accuracy": 0.97,
                "f1_macro": 0.97,
                "per_class": {"circle": {"recall": 0.97}},
            }
        )
    )
    assert main(["gate", "--candidate", str(metrics_path)]) == 0


def test_gate_blocks_on_bad_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"accuracy": 0.5, "f1_macro": 0.5, "per_class": {}}))
    assert main(["gate", "--candidate", str(metrics_path)]) == 1


def test_drift_command_detects_shift():
    assert main(["drift", "--samples", "300", "--shift", "0.3"]) == 1
    assert main(["drift", "--samples", "300", "--shift", "0.0"]) == 0
