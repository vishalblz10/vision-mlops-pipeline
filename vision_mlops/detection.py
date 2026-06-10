"""Object-detection evaluation: IoU, greedy matching, precision/recall and mAP.

Framework-agnostic (pure numpy) so it can score any detector's output — e.g.
a YOLO model — against ground-truth annotations. Boxes are ``[x1, y1, x2, y2]``
in pixels.
"""

from __future__ import annotations

import numpy as np


def iou(box_a, box_b) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return float(inter / (area_a + area_b - inter))


def match_image(pred_boxes, pred_scores, gt_boxes, iou_threshold: float = 0.5):
    """Greedily match one image's predictions to ground truths for one class.

    Predictions are processed in descending score order; each ground-truth box
    may be matched at most once, so duplicate detections count as false
    positives. Returns ``(scores, tp_flags)`` aligned to the processing order.
    """
    order = np.argsort(-np.asarray(pred_scores, dtype=float))
    scores = np.asarray(pred_scores, dtype=float)[order]
    tp = np.zeros(len(order), dtype=bool)
    matched: set[int] = set()

    for rank, idx in enumerate(order):
        best_iou, best_gt = 0.0, None
        for g, gt in enumerate(gt_boxes):
            if g in matched:
                continue
            value = iou(pred_boxes[idx], gt)
            if value > best_iou:
                best_iou, best_gt = value, g
        if best_gt is not None and best_iou >= iou_threshold:
            matched.add(best_gt)
            tp[rank] = True
    return scores, tp


def precision_recall_curve(tp_flags: np.ndarray, n_gt: int):
    """Cumulative precision/recall over predictions sorted by score."""
    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(~tp_flags)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    return precision, recall


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under the precision envelope (VOC2010-style continuous AP)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def evaluate_detections(predictions, ground_truths, iou_threshold: float = 0.5) -> dict:
    """Score a detector over a dataset.

    ``predictions``: per-image dicts with ``boxes``, ``scores`` and ``labels``.
    ``ground_truths``: per-image dicts with ``boxes`` and ``labels``.
    Returns mAP at the given IoU threshold plus per-class AP and overall
    precision/recall at the full operating point.
    """
    classes = sorted(
        {int(label) for gt in ground_truths for label in gt["labels"]}
        | {int(label) for pred in predictions for label in pred["labels"]}
    )

    per_class_ap: dict[int, float] = {}
    total_tp = total_pred = total_gt = 0

    for cls in classes:
        all_scores: list[np.ndarray] = []
        all_tp: list[np.ndarray] = []
        n_gt = 0
        for pred, gt in zip(predictions, ground_truths):
            pred_idx = [i for i, label in enumerate(pred["labels"]) if int(label) == cls]
            gt_boxes = [b for b, label in zip(gt["boxes"], gt["labels"]) if int(label) == cls]
            n_gt += len(gt_boxes)

            boxes = [pred["boxes"][i] for i in pred_idx]
            scores = [pred["scores"][i] for i in pred_idx]
            if boxes:
                s, tp = match_image(boxes, scores, gt_boxes, iou_threshold)
                all_scores.append(s)
                all_tp.append(tp)

        if all_scores:
            scores = np.concatenate(all_scores)
            tp = np.concatenate(all_tp)
            order = np.argsort(-scores)
            tp = tp[order]
        else:
            tp = np.zeros(0, dtype=bool)

        precision, recall = precision_recall_curve(tp, n_gt)
        per_class_ap[cls] = average_precision(precision, recall) if n_gt else 0.0

        total_tp += int(tp.sum())
        total_pred += len(tp)
        total_gt += n_gt

    mean_ap = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return {
        f"map{int(iou_threshold * 100)}": mean_ap,
        "per_class_ap": per_class_ap,
        "precision": total_tp / total_pred if total_pred else 0.0,
        "recall": total_tp / total_gt if total_gt else 0.0,
    }
