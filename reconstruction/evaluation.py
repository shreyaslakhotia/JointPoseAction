"""Silhouette agreement and bounded geometric diagnostics, with explicit holdout.

Predicted masks provide agreement measurements, never independent accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from segmentation_pilot.metrics import binary_mask, boundary_f1, dice_coefficient, mask_iou
from .alignment import points3, validate_alignment_views

CAMERA_IDS = ("C10095", "C10115", "C10118", "C10119", "C10379", "C10390", "C10395", "C10404")


@dataclass(frozen=True)
class ViewPartition:
    input_camera_ids: tuple[str, ...]
    heldout_camera_ids: tuple[str, ...]
    expected_camera_ids: tuple[str, ...] = CAMERA_IDS
    alignment_camera_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for values in (self.input_camera_ids, self.heldout_camera_ids, self.expected_camera_ids,
                       self.alignment_camera_ids):
            if isinstance(values, str) or any(not isinstance(x, str) or not x.strip() for x in values):
                raise ValueError("Camera IDs must be explicit nonempty strings in a sequence")
            if len(set(values)) != len(values):
                raise ValueError("Camera lists must not contain duplicates")
        if not self.input_camera_ids or not self.heldout_camera_ids:
            raise ValueError("Both input and held-out camera lists must be explicit and nonempty")
        if set(self.input_camera_ids) | set(self.heldout_camera_ids) != set(self.expected_camera_ids):
            raise ValueError("Input and held-out camera lists must partition expected cameras")
        validate_alignment_views(self.alignment_camera_ids, self.input_camera_ids, self.heldout_camera_ids)


@dataclass(frozen=True)
class Observation:
    camera: Any
    mask: np.ndarray
    mask_kind: str
    source_sha256: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = binary_mask(self.mask)
        width, height = self.camera.image_size
        if mask.shape != (height, width):
            raise ValueError("Observation mask must match native camera resolution")
        if self.mask_kind not in {"predicted", "independent_verified"}:
            raise ValueError("Mask kind must be predicted or independent_verified")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            raise ValueError("Observation source SHA-256 is required")
        if self.mask_kind == "independent_verified":
            p = self.provenance
            if (p.get("verified_independent") is not True or p.get("derived_from_prediction") is not False
                    or p.get("kind") not in {"independent_manual_annotation", "independent_dataset_annotation"}
                    or any(not isinstance(p.get(k), str) or not p[k].strip()
                           for k in ("source_id", "reviewer", "evidence"))):
                raise ValueError("Independent references require explicit independent annotation provenance")
        object.__setattr__(self, "mask", mask.copy())


def evaluate_silhouettes(reconstruction: Any, observations: Sequence[Observation],
                         partition: ViewPartition, *, boundary_tolerance_px: int = 2,
                         renderer: Callable[[Any, Any], np.ndarray] | None = None,
                         representation_kind: str = "occupancy") -> dict[str, Any]:
    """Render occupied volume or an explicit mesh renderer; never scatter points.

    Missing views remain missing. Means use available views separately for input
    and held-out cameras and independently for predicted/verified references.
    A mesh renderer hook must return a native HxW binary rasterized silhouette.
    """
    if representation_kind not in {"occupancy", "mesh"}:
        raise ValueError("Silhouette evaluation requires occupied volume or a rasterized mesh, not point scatter")
    if representation_kind == "occupancy":
        from .visual_hull import VoxelGrid
        if not isinstance(reconstruction, VoxelGrid):
            raise ValueError("Occupancy silhouette rendering requires a VoxelGrid")
        if renderer is None:
            from .rendering import render_occupancy
            renderer = render_occupancy
    elif renderer is None:
        raise ValueError("Mesh evaluation requires an explicit surface renderer")
    else:
        vertices = points3(getattr(reconstruction, "vertices", None), minimum=3)
        faces = np.asarray(getattr(reconstruction, "faces", None))
        if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.dtype.kind not in "iu"
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Mesh requires finite vertices and valid integer triangle faces")
    metadata = getattr(reconstruction, "metadata", {})
    construction_ids = metadata.get("retained_camera_ids", metadata.get("input_camera_ids"))
    if (not isinstance(construction_ids, (list, tuple)) or not construction_ids
            or any(not isinstance(value, str) for value in construction_ids)
            or len(construction_ids) != len(set(construction_ids))):
        raise ValueError("Reconstruction metadata must record nonempty input/retained camera IDs")
    validate_alignment_views(construction_ids, partition.input_camera_ids, partition.heldout_camera_ids)
    artifact_alignment = metadata.get("alignment_camera_ids", [])
    validate_alignment_views(artifact_alignment, partition.input_camera_ids, partition.heldout_camera_ids)
    if set(artifact_alignment) != set(partition.alignment_camera_ids):
        raise ValueError("Artifact alignment camera IDs disagree with evaluation partition")
    indexed = {}
    for observation in observations:
        camera_id = observation.camera.camera_id
        if camera_id in indexed or camera_id not in partition.expected_camera_ids:
            raise ValueError("Duplicate or unexpected observation camera")
        if (getattr(reconstruction, "coordinate_frame", None) != observation.camera.coordinate_frame
                or getattr(reconstruction, "units", None) != observation.camera.units):
            raise ValueError("Reconstruction and camera frames/units must agree explicitly")
        indexed[camera_id] = observation
    rows = []
    for camera_id in partition.expected_camera_ids:
        split = "input" if camera_id in partition.input_camera_ids else "heldout"
        observation = indexed.get(camera_id)
        if observation is None:
            rows.append({"camera_id": camera_id, "partition": split, "status": "missing", "metrics": None})
            continue
        rendered = binary_mask(renderer(reconstruction, observation.camera))
        if rendered.shape != observation.mask.shape:
            raise ValueError("Renderer changed native image resolution")
        independent = observation.mask_kind == "independent_verified"
        rows.append({"camera_id": camera_id, "partition": split, "status": "available",
                     "reference_kind": observation.mask_kind, "reference_sha256": observation.source_sha256,
                     "reference_provenance": dict(observation.provenance),
                     "measurement_kind": "independent_reference_accuracy" if independent else "predicted_mask_agreement",
                     "accuracy_established": independent,
                     "metrics": {"iou": mask_iou(rendered, observation.mask),
                                 "dice": dice_coefficient(rendered, observation.mask),
                                 "boundary_f1": boundary_f1(rendered, observation.mask, boundary_tolerance_px)},
                     "rendered_foreground_pixels": int(rendered.sum()),
                     "reference_foreground_pixels": int(observation.mask.sum())})
    summaries = {}
    for split in ("input", "heldout"):
        for kind in ("predicted", "independent_verified"):
            available = [r for r in rows if r["partition"] == split and r.get("reference_kind") == kind]
            summaries[f"{split}_{kind}"] = {"available_cameras": len(available),
                "camera_ids": [r["camera_id"] for r in available],
                "mean": {metric: float(np.mean([r["metrics"][metric] for r in available]))
                         if available else None for metric in ("iou", "dice", "boundary_f1")}}
    return {"schema_version": 1, "per_camera": rows, "summaries": summaries,
            "expected_cameras": len(partition.expected_camera_ids),
            "available_cameras": len(indexed), "missing_camera_ids": [r["camera_id"] for r in rows if r["status"] == "missing"],
            "input_camera_ids": list(partition.input_camera_ids), "heldout_camera_ids": list(partition.heldout_camera_ids),
            "reconstruction_input_camera_ids": list(construction_ids),
            "alignment_camera_ids": list(partition.alignment_camera_ids),
            "boundary_tolerance_px": boundary_tolerance_px,
            "empty_mask_convention": "both_empty=1; one_empty=0",
            "aggregation": "arithmetic mean over available cameras within partition and reference kind; missing views never zero-imputed"}


def point_projection_coverage(points: Any, camera: Any) -> dict[str, Any]:
    """Point visibility diagnostic, explicitly not a filled silhouette metric."""
    from .cameras import project
    points = points3(points)
    pixels, depth = project(camera, points)
    width, height = camera.image_size
    visible = ((depth > 0) & np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0)
               & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    return {"measurement_kind": "point_projection_coverage_not_silhouette", "points": len(points),
            "points_inside_image_in_front": int(visible.sum()), "fraction": float(visible.mean())}


def temporal_chamfer(first: Any, second: Any, *, first_frame: str, second_frame: str,
                     first_units: str, second_units: str,
                     registration: Mapping[str, Any], max_pair_evaluations: int = 10_000_000,
                     chunk_size: int = 256) -> dict[str, Any]:
    """Symmetric mean nearest-neighbour distance, in common declared units.

    This is geometry change after supplied registration, not semantic tracking,
    point-index identity, correspondence accuracy, or an independent 3-D score.
    """
    if not first_frame or first_frame != second_frame or not first_units or first_units != second_units:
        raise ValueError("Temporal shapes require the same explicit frame and units")
    if (registration.get("applied") is not True
            or any(not isinstance(registration.get(key), str) or not registration[key].strip()
                   for key in ("method", "evidence"))):
        raise ValueError("Explicit applied registration metadata is required")
    a, b = points3(first), points3(second)
    if (isinstance(max_pair_evaluations, bool) or not isinstance(max_pair_evaluations, int)
            or not 1 <= max_pair_evaluations <= 10_000_000
            or isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 1024
            or len(a) * len(b) > max_pair_evaluations or max(len(a), len(b)) > 10000):
        raise ValueError("Chamfer computation exceeds bounded CPU budget")
    def directed(x: np.ndarray, y: np.ndarray) -> float:
        total = 0.0
        for start in range(0, len(x), chunk_size):
            distances2 = np.sum((x[start:start + chunk_size, None] - y[None]) ** 2, axis=2)
            total += float(np.sqrt(distances2.min(axis=1)).sum())
        return total / len(x)
    ab, ba = directed(a, b), directed(b, a)
    return {"measurement_kind": "registered_temporal_geometry_change", "symmetric_mean_chamfer": (ab + ba) / 2,
            "first_to_second_mean": ab, "second_to_first_mean": ba, "squared_distances": False,
            "units": first_units, "coordinate_frame": first_frame, "registration": dict(registration),
            "first_points": len(a), "second_points": len(b), "semantic_tracking_established": False}


def runtime_record(*, status: str, elapsed_seconds: float, peak_memory_bytes: int | None,
                   output_bytes: int, failure_reason: str | None = None,
                   memory_measurement: str = "not_measured") -> dict[str, Any]:
    if status not in {"completed", "failed", "blocked"} or not np.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("Invalid run status or elapsed time")
    if (isinstance(output_bytes, bool) or not isinstance(output_bytes, int) or output_bytes < 0
            or (peak_memory_bytes is not None and (isinstance(peak_memory_bytes, bool)
                or not isinstance(peak_memory_bytes, int) or peak_memory_bytes < 0))):
        raise ValueError("Byte counts must be nonnegative integers or unknown")
    if status != "completed" and not failure_reason:
        raise ValueError("Failed or blocked runs require a reason")
    return {"status": status, "elapsed_seconds": float(elapsed_seconds),
            "peak_memory_bytes": peak_memory_bytes, "memory_measurement": memory_measurement,
            "output_bytes": output_bytes, "failure_reason": failure_reason}
