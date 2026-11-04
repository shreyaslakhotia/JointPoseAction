"""Explicit positive-scale similarity registration with auditable evidence.

Bounding-box normalization is deliberately absent: canonical model coordinates
do not establish a calibrated physical pose or metric scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def points3(value: Any, *, minimum: int = 1) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"Expected at least {minimum} three-dimensional points")
    if not np.isfinite(points).all():
        raise ValueError("Points must be finite")
    return points


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    source_frame: str
    target_frame: str
    source_units: str
    target_units: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Similarity scale must be finite and positive")
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)):
            raise ValueError("Rotation must be a proper orthonormal matrix")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("Translation must be a finite three-vector")
        if any(not isinstance(x, str) or not x.strip() for x in
               (self.source_frame, self.target_frame, self.source_units, self.target_units)):
            raise ValueError("Explicit source/target frames and units are required")
        if not self.provenance:
            raise ValueError("Alignment provenance is required")
        rotation, translation = rotation.copy(), translation.copy()
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: Any) -> np.ndarray:
        return self.scale * (points3(points) @ self.rotation.T) + self.translation

    def as_dict(self) -> dict[str, Any]:
        return {"scale": float(self.scale), "rotation": self.rotation.tolist(),
                "translation": self.translation.tolist(), "source_frame": self.source_frame,
                "target_frame": self.target_frame, "source_units": self.source_units,
                "target_units": self.target_units, "provenance": dict(self.provenance)}


def fit_landmark_similarity(source: Any, target: Any, *, source_frame: str,
                            target_frame: str, source_units: str, target_units: str,
                            evidence: Mapping[str, Any], estimate_scale: bool = True
                            ) -> tuple[SimilarityTransform, dict[str, Any]]:
    """Fit paired independent 3-D landmarks; residuals are fit error, not accuracy.

    Three non-collinear paired landmarks are the minimum. Source and target
    landmark IDs must correspond by row; no nearest-neighbour pairing is inferred.
    """
    if (evidence.get("independent_paired_landmarks") is not True
            or any(not isinstance(evidence.get(k), str) or not evidence[k].strip()
                   for k in ("source_id", "reviewer", "correspondence_evidence"))):
        raise ValueError("Independent paired landmark evidence is required")
    x, y = points3(source, minimum=3), points3(target, minimum=3)
    if x.shape != y.shape:
        raise ValueError("Source and target landmark arrays must have equal shape")
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    if np.linalg.matrix_rank(xc) < 2 or np.linalg.matrix_rank(yc) < 2:
        raise ValueError("Collinear or coincident landmarks cannot establish similarity")
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.ones(3)
    sign[-1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.sum(xc * xc) / len(x))
    scale = float(np.sum(singular * sign) / variance) if estimate_scale else 1.0
    transform = SimilarityTransform(scale, rotation, y.mean(axis=0) - scale * rotation @ x.mean(axis=0),
                                    source_frame, target_frame, source_units, target_units,
                                    {"method": "independent_paired_3d_landmarks",
                                     "estimate_scale": bool(estimate_scale), **dict(evidence)})
    residual = np.linalg.norm(transform.apply(x) - y, axis=1)
    return transform, {"landmarks": len(x), "fit_rmse": float(np.sqrt(np.mean(residual ** 2))),
                       "fit_max_error": float(residual.max()), "units": target_units,
                       "accuracy_established": False,
                       "interpretation": "in-sample landmark fit residual; independent check landmarks required for accuracy"}


def validate_alignment_views(optimization_camera_ids: Sequence[str],
                             input_camera_ids: Sequence[str],
                             heldout_camera_ids: Sequence[str]) -> None:
    for values in (optimization_camera_ids, input_camera_ids, heldout_camera_ids):
        if isinstance(values, str) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Camera IDs must be explicit nonempty strings in a sequence")
    optimized, inputs, heldout = map(set, (optimization_camera_ids, input_camera_ids, heldout_camera_ids))
    if inputs & heldout:
        raise ValueError("Input and held-out cameras overlap")
    if not optimized <= inputs or optimized & heldout:
        raise ValueError("Alignment optimization may use input cameras only")
