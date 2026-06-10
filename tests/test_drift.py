import numpy as np

from vision_mlops.data import extract_features, make_dataset
from vision_mlops.drift import detect_drift, population_stability_index


def test_identical_distributions_no_drift():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 2000)
    psi = population_stability_index(values, values.copy())
    assert psi < 0.01

    report = detect_drift({"f": values}, {"f": values.copy()})
    assert not report.drifted


def test_shifted_distribution_high_psi():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 2000)
    current = rng.normal(1, 1, 2000)
    psi = population_stability_index(reference, current)
    assert psi > 0.25


def test_same_distribution_different_seed_low_psi():
    reference = np.random.default_rng(2).normal(0, 1, 3000)
    current = np.random.default_rng(3).normal(0, 1, 3000)
    psi = population_stability_index(reference, current)
    assert psi < 0.1


def test_constant_feature_does_not_crash():
    reference = np.full(100, 5.0)
    current = np.full(100, 5.0)
    psi = population_stability_index(reference, current)
    assert psi < 0.01


def test_drift_report_structure():
    rng = np.random.default_rng(4)
    ref = {"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)}
    cur = {"a": rng.normal(3, 1, 500), "b": ref["b"].copy()}
    report = detect_drift(ref, cur)
    assert report.drifted
    by_name = {f.feature: f for f in report.features}
    assert by_name["a"].drifted
    assert not by_name["b"].drifted
    payload = report.to_dict()
    assert payload["drifted"] is True
    assert len(payload["features"]) == 2


def test_brightness_shift_triggers_drift():
    X_ref, _ = make_dataset(400, seed=42)
    X_cur, _ = make_dataset(400, seed=43, brightness_shift=0.3)
    report = detect_drift(extract_features(X_ref), extract_features(X_cur))
    assert report.drifted
