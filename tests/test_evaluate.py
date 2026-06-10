import numpy as np

from vision_mlops.data import CLASSES
from vision_mlops.evaluate import compute_metrics, confusion_matrix, evaluate_model


def test_confusion_matrix_known_values():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    expected = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    np.testing.assert_array_equal(cm, expected)


def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    metrics = compute_metrics(y, y, num_classes=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    for name in CLASSES:
        assert metrics["per_class"][name]["recall"] == 1.0


def test_metrics_structure():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])
    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    assert set(metrics) >= {"accuracy", "precision_macro", "recall_macro", "f1_macro", "per_class", "confusion_matrix"}
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"][CLASSES[2]]["support"] == 2


def test_evaluate_model_on_holdout(trained_model, shapes_data):
    model, _ = trained_model
    _, _, X_test, y_test = shapes_data
    result = evaluate_model(model, X_test, y_test)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["mean_confidence"] <= 1.0
    assert len(result["confusion_matrix"]) == len(CLASSES)
