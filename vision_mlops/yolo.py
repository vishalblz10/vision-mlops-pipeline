"""YOLO detector adapter behind a stable interface.

The ultralytics backend is an optional extra (``pip install '.[yolo]'``) so the
core pipeline stays light; the output post-processing is dependency-free and
unit-tested without weights. Detections from this adapter feed straight into
``vision_mlops.detection.evaluate_detections`` for mAP scoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    label: str
    confidence: float


def results_to_detections(boxes_xyxy, confidences, class_ids, class_names) -> list[Detection]:
    """Convert raw detector outputs (arrays/sequences) into ``Detection`` objects,
    sorted by descending confidence."""
    detections = [
        Detection(
            box=tuple(float(v) for v in box),
            label=str(class_names[int(cls)]),
            confidence=float(conf),
        )
        for box, conf, cls in zip(boxes_xyxy, confidences, class_ids)
    ]
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


class YoloDetector:
    """Thin wrapper around an ultralytics YOLO model."""

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed; install the extra with `pip install '.[yolo]'`"
            ) from exc
        self._model = YOLO(weights)
        self.confidence = confidence

    def detect(self, image) -> list[Detection]:
        """Run detection on an image (path, ndarray or PIL image)."""
        result = self._model.predict(image, conf=self.confidence, verbose=False)[0]
        boxes = result.boxes
        return results_to_detections(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            result.names,
        )
