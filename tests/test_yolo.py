import importlib.util

import pytest

from vision_mlops.yolo import Detection, YoloDetector, results_to_detections

ULTRALYTICS_INSTALLED = importlib.util.find_spec("ultralytics") is not None


def test_results_to_detections_converts_and_sorts():
    detections = results_to_detections(
        boxes_xyxy=[[0, 0, 10, 10], [5, 5, 20, 20]],
        confidences=[0.4, 0.9],
        class_ids=[0, 1],
        class_names={0: "person", 1: "car"},
    )
    assert [d.label for d in detections] == ["car", "person"]
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[1].box == (0.0, 0.0, 10.0, 10.0)
    assert all(isinstance(d, Detection) for d in detections)


def test_empty_results():
    assert results_to_detections([], [], [], {}) == []


@pytest.mark.skipif(ULTRALYTICS_INSTALLED, reason="ultralytics is installed")
def test_detector_raises_without_ultralytics():
    with pytest.raises(ImportError, match="ultralytics"):
        YoloDetector()


@pytest.mark.skipif(not ULTRALYTICS_INSTALLED, reason="ultralytics not installed")
def test_detector_runs_real_inference():
    import numpy as np

    detector = YoloDetector()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    detections = detector.detect(image)
    assert isinstance(detections, list)
