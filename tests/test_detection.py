import numpy as np
import pytest

from vision_mlops.detection import (
    average_precision,
    evaluate_detections,
    iou,
    match_image,
    precision_recall_curve,
)


def test_iou_identical_boxes():
    box = (10.0, 10.0, 50.0, 50.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_known_overlap():
    # 10x10 boxes offset by 5 in x: intersection 5x10=50, union 200-50=150
    value = iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert value == pytest.approx(50 / 150)


def test_match_image_duplicate_is_fp():
    gt = [(0, 0, 10, 10)]
    preds = [(0, 0, 10, 10), (1, 0, 11, 10)]
    scores = [0.9, 0.8]
    sorted_scores, tp_flags = match_image(preds, scores, gt)
    assert list(sorted_scores) == [0.9, 0.8]
    assert list(tp_flags) == [True, False]


def test_precision_recall_curve():
    tp_flags = np.array([True, False, True])
    precision, recall = precision_recall_curve(tp_flags, n_gt=2)
    np.testing.assert_allclose(precision, [1.0, 0.5, 2 / 3])
    np.testing.assert_allclose(recall, [0.5, 0.5, 1.0])


def test_average_precision_known_value():
    # TP@0.9, FP@0.8, TP@0.7 with 2 GTs -> AP = 0.5*1.0 + 0.5*(2/3) = 5/6
    precision = np.array([1.0, 0.5, 2 / 3])
    recall = np.array([0.5, 0.5, 1.0])
    assert average_precision(precision, recall) == pytest.approx(5 / 6)


def test_perfect_detections_map_is_one():
    predictions = [{"boxes": [(0, 0, 10, 10), (20, 20, 40, 40)], "scores": [0.95, 0.9], "labels": [0, 1]}]
    ground_truths = [{"boxes": [(0, 0, 10, 10), (20, 20, 40, 40)], "labels": [0, 1]}]
    result = evaluate_detections(predictions, ground_truths)
    assert result["map50"] == pytest.approx(1.0)
    assert result["per_class_ap"][0] == pytest.approx(1.0)
    assert result["per_class_ap"][1] == pytest.approx(1.0)


def test_partial_detections():
    predictions = [
        {
            "boxes": [(0, 0, 10, 10), (100, 100, 110, 110), (20, 20, 30, 30)],
            "scores": [0.9, 0.8, 0.7],
            "labels": [0, 0, 0],
        }
    ]
    ground_truths = [{"boxes": [(0, 0, 10, 10), (20, 20, 30, 30)], "labels": [0, 0]}]
    result = evaluate_detections(predictions, ground_truths)
    assert result["map50"] == pytest.approx(5 / 6)
    assert result["recall"] == pytest.approx(1.0)
