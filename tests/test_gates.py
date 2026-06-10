import pytest

from vision_mlops.gates import GateConfig, evaluate_gate

GOOD = {
    "accuracy": 0.95,
    "f1_macro": 0.94,
    "per_class": {
        "circle": {"recall": 0.96},
        "square": {"recall": 0.93},
        "triangle": {"recall": 0.95},
    },
}


def test_passing_candidate():
    decision = evaluate_gate(GOOD, GateConfig())
    assert decision.passed
    assert not decision.failures


def test_low_accuracy_fails():
    candidate = {**GOOD, "accuracy": 0.85}
    decision = evaluate_gate(candidate, GateConfig())
    assert not decision.passed
    assert any("accuracy" in check.name for check in decision.failures)


def test_low_per_class_recall_fails():
    candidate = {
        **GOOD,
        "per_class": {**GOOD["per_class"], "triangle": {"recall": 0.5}},
    }
    decision = evaluate_gate(candidate, GateConfig())
    assert not decision.passed
    assert any("triangle" in check.name for check in decision.failures)


def test_regression_against_baseline_fails():
    baseline = {"accuracy": 0.99}
    decision = evaluate_gate(GOOD, GateConfig(), baseline=baseline)
    assert not decision.passed


def test_small_regression_within_tolerance_passes():
    baseline = {"accuracy": 0.96}
    decision = evaluate_gate(GOOD, GateConfig(), baseline=baseline)
    assert decision.passed


def test_decision_to_dict():
    payload = evaluate_gate(GOOD, GateConfig()).to_dict()
    assert payload["passed"] is True
    assert all({"name", "passed", "detail"} <= set(c) for c in payload["checks"])


def test_from_yaml(tmp_path):
    config_file = tmp_path / "gates.yaml"
    config_file.write_text(
        "min_accuracy: 0.5\nmin_f1_macro: 0.5\nmin_per_class_recall: 0.4\nmax_accuracy_drop: 0.1\n"
    )
    config = GateConfig.from_yaml(config_file)
    assert config.min_accuracy == 0.5
    assert config.max_accuracy_drop == pytest.approx(0.1)
