"""Mask measurements; model predictions never supply their own accuracy truth.

IoU/Dice use the explicit convention that two empty masks score one. Boundary
matching uses a Euclidean pixel tolerance on the original, unresized image.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def binary_mask(value: Any) -> np.ndarray:
    """Validate a nonempty H x W mask containing only 0/1 or 0/255 values."""
    mask = np.asarray(value)
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("Expected a nonempty two-dimensional native mask")
    if mask.dtype.kind not in "buif" or not np.isfinite(mask).all():
        raise ValueError("Mask must contain finite binary numbers")
    values = np.unique(mask)
    if not (np.isin(values, [0, 1]).all() or np.isin(values, [0, 255]).all()):
        raise ValueError("Mask is not binary; threshold probabilities explicitly")
    return mask != 0


def _pair(prediction: Any, reference: Any) -> tuple[np.ndarray, np.ndarray]:
    predicted, target = binary_mask(prediction), binary_mask(reference)
    if predicted.shape != target.shape:
        raise ValueError("Prediction and reference resolutions differ; no implicit resizing")
    return predicted, target


def mask_iou(prediction: Any, reference: Any) -> float:
    predicted, target = _pair(prediction, reference)
    union = int(np.count_nonzero(predicted | target))
    return float(np.count_nonzero(predicted & target) / union) if union else 1.0


def dice_coefficient(prediction: Any, reference: Any) -> float:
    predicted, target = _pair(prediction, reference)
    denominator = int(np.count_nonzero(predicted) + np.count_nonzero(target))
    return float(2 * np.count_nonzero(predicted & target) / denominator) if denominator else 1.0


def mask_boundary(value: Any) -> np.ndarray:
    """One-pixel internal boundary from an eight-neighbour binary erosion."""
    mask = binary_mask(value)
    height, width = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    eroded = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + height, dx:dx + width]
    return mask & ~eroded


def _dilate_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    dilated = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                dilated |= padded[radius + dy:radius + dy + height,
                                  radius + dx:radius + dx + width]
    return dilated


def boundary_f1(prediction: Any, reference: Any, tolerance_px: int = 2) -> float:
    if isinstance(tolerance_px, bool) or not isinstance(tolerance_px, int) or not 0 <= tolerance_px <= 32:
        raise ValueError("Boundary tolerance must be an integer from 0 to 32 pixels")
    predicted, target = _pair(prediction, reference)
    predicted_boundary, target_boundary = mask_boundary(predicted), mask_boundary(target)
    count_predicted, count_target = (int(np.count_nonzero(x)) for x in (predicted_boundary, target_boundary))
    if not count_predicted or not count_target:
        return 1.0 if count_predicted == count_target else 0.0
    precision = float(np.count_nonzero(predicted_boundary & _dilate_disk(target_boundary, tolerance_px)) / count_predicted)
    recall = float(np.count_nonzero(target_boundary & _dilate_disk(predicted_boundary, tolerance_px)) / count_target)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def describe_mask(value: Any, confidence: float | None = None) -> dict[str, Any]:
    mask = binary_mask(value)
    if confidence is not None and (not np.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("SAM3 grounding confidence must lie in [0, 1]")
    ys, xs = np.nonzero(mask)
    return {
        "height": mask.shape[0], "width": mask.shape[1],
        "foreground_pixels": int(len(xs)),
        "image_coverage_fraction": float(len(xs) / mask.size),
        "mask_bbox_xyxy_exclusive": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if len(xs) else None,
        "grounding_confidence": float(confidence) if confidence is not None else None,
        "accuracy_established": False,
        "iou": None, "dice": None, "boundary_f1": None,
    }


def evaluate_mask(prediction: Any, *, confidence: float | None = None,
                  reference: Any = None, reference_provenance: dict[str, Any] | None = None,
                  tolerance_px: int = 2) -> dict[str, Any]:
    """Report accuracy only with explicit independently established reference provenance.

    Required provenance: verified_independent=true, derived_from_prediction=false,
    kind='independent_manual_annotation' or 'independent_dataset_annotation',
    and nonempty source_id, reviewer, evidence. These attestations still require
    real review; they are not inferred from agreement with a model prediction.
    """
    result = describe_mask(prediction, confidence)
    if reference is None:
        if reference_provenance is not None:
            raise ValueError("Reference provenance supplied without a reference mask")
        result["accuracy_status"] = "not_established_no_independent_reference"
        return result
    provenance = reference_provenance or {}
    if (provenance.get("verified_independent") is not True
            or provenance.get("derived_from_prediction") is not False
            or provenance.get("kind") not in {"independent_manual_annotation", "independent_dataset_annotation"}
            or any(not isinstance(provenance.get(key), str) or not provenance[key].strip()
                   for key in ("source_id", "reviewer", "evidence"))):
        raise ValueError("Accuracy requires independently verified reference provenance; predictions are not ground truth")
    predicted, target = _pair(prediction, reference)
    result.update({
        "accuracy_established": True, "accuracy_status": "independent_reference_provided",
        "iou": mask_iou(predicted, target), "dice": dice_coefficient(predicted, target),
        "boundary_f1": boundary_f1(predicted, target, tolerance_px),
        "boundary_tolerance_px": tolerance_px, "reference_provenance": dict(provenance),
        "empty_mask_convention": "both_empty=1; one_empty=0",
    })
    return result
