"""Evidence-gated checks for original Assembly101 fixed RGB cameras.

The official Nimble ModelViewMatrix maps world to camera. Toolkit camera.py
inverts it for storage, then applies the inverse in world_to_eye. Distortion
coefficients below are named, never copied positionally from toolkit tuples.
This module does not infer synchronization, metric units, or calibration
accuracy from a valid rotation matrix or a plausible projection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TOOLKIT_REVISION = "8d4b7f1bce3f99874514199aae7504f78ebf3517"
TOOLKIT_URL = "https://github.com/facebookresearch/assemblyhands-toolkit"
CAMERA_IDS = ("C10095", "C10115", "C10118", "C10119", "C10379", "C10390", "C10395", "C10404")
DISTORTION_CV2_ORDER = ("k1", "k2", "p1", "p2", "k3")


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def load_nimble_calibration(path, recording_id, *, source_revision=TOOLKIT_REVISION, image_sizes=None):
    """Parse one recording-specific original (distorted) calibration JSON.

    image_sizes, when provided, maps camera IDs to decoded (width, height).
    A structurally valid report is deliberately not a scientific readiness gate.
    """
    path = Path(path)
    if path.name != f"{recording_id}.json":
        raise ValueError("Calibration filename must exactly match recording_id")
    if source_revision != TOOLKIT_REVISION:
        raise ValueError("Unreviewed calibration source revision")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Official Nimble calibration must be a list")
    cameras = {}
    for row in payload:
        camera = row.get("Camera", row)
        serial = camera.get("SerialNo", "")
        if not serial.startswith("C"):
            continue  # Four moving fisheye HMC cameras are outside this pilot.
        if not serial.endswith("_rgb") or serial[:-4] not in CAMERA_IDS:
            raise ValueError(f"Unexpected fixed camera serial: {serial}")
        camera_id = serial[:-4]
        if camera_id in cameras:
            raise ValueError(f"Duplicate camera: {camera_id}")
        if row.get("MountingLocation", "Exocentric") != "Exocentric":
            raise ValueError("Fixed camera must be Exocentric")
        if camera.get("DistortionModel") != "OpenCV":
            raise ValueError("Only official fixed OpenCV cameras are supported")
        size = (int(camera["ImageSizeX"]), int(camera["ImageSizeY"]))
        if size != (1920, 1080):
            raise ValueError(f"Unexpected original image size: {size}")
        if image_sizes is not None and tuple(image_sizes[camera_id]) != size:
            raise ValueError(f"Decoded image/calibration size mismatch: {camera_id}")
        fx, fy, cx, cy = _finite_array([camera[k] for k in ("fx", "fy", "cx", "cy")], (4,), "intrinsics")
        if fx <= 0 or fy <= 0 or not (0 <= cx < size[0] and 0 <= cy < size[1]):
            raise ValueError("Invalid focal lengths or principal point")
        transform = _finite_array(camera["ModelViewMatrix"], (4, 4), "ModelViewMatrix")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
            raise ValueError("Invalid homogeneous transform last row")
        rotation = transform[:3, :3]
        determinant = float(np.linalg.det(rotation))
        orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        if abs(determinant - 1) > 1e-5 or orthogonality > 1e-5:
            raise ValueError("Invalid rotation matrix")
        distortion = _finite_array([camera[k] for k in DISTORTION_CV2_ORDER], (5,), "distortion")
        for key in ("k4", "k5", "k6", "p3", "p4"):
            if key in camera and (not np.isfinite(camera[key]) or camera[key] != 0):
                raise ValueError(f"Unsupported nonzero unused coefficient {key}")
        cameras[camera_id] = {
            "camera_id": camera_id, "serial": serial, "image_size": list(size),
            "K": [[float(fx), 0., float(cx)], [0., float(fy), float(cy)], [0., 0., 1.]],
            "distortion_cv2_order": list(DISTORTION_CV2_ORDER), "distortion": distortion.tolist(),
            "world_to_camera": transform.tolist(), "projection_direction": "world_to_camera",
            "rotation_determinant": determinant, "rotation_orthogonality_max_abs": orthogonality,
            "units_status": "unverified", "coordinate_convention": "OpenCV right/down/forward; pixel centers",
        }
    if set(cameras) != set(CAMERA_IDS):
        raise ValueError(f"Need all eight fixed cameras; missing {sorted(set(CAMERA_IDS) - set(cameras))}")
    return {
        "recording_id": recording_id, "source_revision": source_revision,
        "source_url": f"{TOOLKIT_URL}/blob/{source_revision}/calib/nimble_json_calib/{path.name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cameras": cameras,
        "structural_checks_passed": True, "decoded_resolution_checked": image_sizes is not None,
        "calibration_accuracy_verified": False, "units_status": "unverified", "segmentation_ready": False,
        "limitations": ["Rotation consistency does not establish image reprojection accuracy.",
                        "No video synchronization or metric units inferred from calibration."],
    }


def project_world_points(camera, points):
    """Project N world points to distorted native pixels; return pixels and Z.

    Nonpositive-depth points produce NaN pixels, not plausible image locations.
    Input coordinates must share the calibration translation's unknown units.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("World points must be finite N x 3")
    if camera.get("projection_direction") != "world_to_camera":
        raise ValueError("Explicit reviewed projection direction required")
    transform = _finite_array(camera["world_to_camera"], (4, 4), "world_to_camera")
    intrinsic = _finite_array(camera["K"], (3, 3), "K")
    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    depth = camera_points[:, 2]
    pixels = np.full((len(points), 2), np.nan)
    front = depth > 0
    x, y = (camera_points[front, :2] / depth[front, None]).T
    k1, k2, p1, p2, k3 = camera["distortion"]
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*radial + 2*p2*x*y + p1*(r2 + 2*y*y)
    pixels[front] = np.column_stack((intrinsic[0, 0]*xd + intrinsic[0, 2],
                                     intrinsic[1, 1]*yd + intrinsic[1, 2]))
    return pixels, depth


def reprojection_diagnostics(camera, world_points, observed_pixels, *, independence_evidence):
    """Measure residuals only for externally observed 2D correspondences.

    Projections saved by the pose producer are not independent pixel references.
    Declared evidence and its origin remain part of the report for manual review.
    """
    if not independence_evidence or independence_evidence.get("independent") is not True or not independence_evidence.get("source"):
        raise ValueError("Independent 2D reference provenance required")
    if independence_evidence.get("derived_from_same_calibration") is not False:
        raise ValueError("Same-calibration projected labels cannot validate calibration")
    pixels, depth = project_world_points(camera, world_points)
    observed = _finite_array(observed_pixels, pixels.shape, "observed_pixels")
    if len(pixels) < 3 or not np.isfinite(pixels).all():
        raise ValueError("Need at least three positive-depth reference correspondences")
    residual = np.linalg.norm(pixels - observed, axis=1)
    return {"count": len(residual), "residual_units": "native_image_pixels",
            "median_px": float(np.median(residual)), "mean_px": float(residual.mean()),
            "rmse_px": float(np.sqrt(np.mean(residual**2))), "max_px": float(residual.max()),
            "positive_depth_count": int((depth > 0).sum()), "evidence": independence_evidence,
            "accuracy_acceptance": "requires_predeclared_threshold_and_reference_review"}


def audit_temporal_correspondence(views, evidence=None, *, max_residual_seconds=1/60):
    """Audit an explicitly measured constant-offset mapping over a pilot window.

    views[camera_id] = {frame_indices: [...], timestamps_seconds: [...]}.
    evidence = {method, source, shared_reference_id, independent: true,
      correspondences: {camera_id: [{frame_index, reference_seconds, role}]}}.
    Each camera needs >=2 fit and >=2 held_out observations. A held-out event
    ID must be shared across all cameras. Validation checks mapping residuals,
    not equal frame indices, FPS, or stream start times. The report covers only
    the observed reference span; future drift is not certified.
    """
    if not np.isfinite(max_residual_seconds) or max_residual_seconds <= 0:
        raise ValueError("Positive temporal residual threshold required")
    if set(views) != set(CAMERA_IDS):
        raise ValueError("Temporal audit requires all eight exact camera IDs")
    arrays = {}
    for camera_id, view in views.items():
        indices = np.asarray(view["frame_indices"])
        times = np.asarray(view["timestamps_seconds"], dtype=float)
        if indices.ndim != 1 or times.shape != indices.shape or len(times) < 2:
            raise ValueError("Need corresponding frame indices and timestamps")
        if not np.isfinite(times).all() or not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all() or np.any(np.diff(indices) <= 0) or np.any(np.diff(times) <= 0):
            raise ValueError("Frame indices and timestamps must be finite and strictly increasing")
        arrays[camera_id] = {int(i): float(t) for i, t in zip(indices, times)}
    report = {"segmentation_ready": False, "synchronization_verified": False,
              "tolerance_seconds": float(max_residual_seconds), "per_camera": {},
              "reason": "Independent shared-event correspondences are required; equal indices/FPS/PTS origins are insufficient."}
    if not evidence or evidence.get("independent") is not True or any(not evidence.get(k) for k in ("method", "source", "shared_reference_id")):
        return report
    correspondences = evidence.get("correspondences", {})
    if set(correspondences) != set(CAMERA_IDS):
        return report
    common_held_out = None
    all_passed = True
    for camera_id in CAMERA_IDS:
        rows = correspondences[camera_id]
        fit = [r for r in rows if r.get("role") == "fit"]
        held = [r for r in rows if r.get("role") == "held_out"]
        if len(fit) < 2 or len(held) < 2:
            report["reason"] = f"Insufficient fit/held-out events for {camera_id}"
            return report
        held_ids = {r.get("event_id") for r in held if r.get("event_id")}
        common_held_out = held_ids if common_held_out is None else common_held_out & held_ids
        if len({r.get("event_id") for r in rows}) != len(rows) or any(not r.get("event_id") for r in rows):
            raise ValueError("Reference event IDs must be unique per view and nonempty")
        if len({r["frame_index"] for r in rows}) != len(rows):
            raise ValueError("A source frame cannot be reused for independent reference events")
        for row in rows:
            if row["frame_index"] not in arrays[camera_id] or not np.isfinite(row["reference_seconds"]):
                raise ValueError("Reference event must identify a decoded frame and finite common time")
        offset = float(np.median([r["reference_seconds"] - arrays[camera_id][r["frame_index"]] for r in fit]))
        residual = [arrays[camera_id][r["frame_index"]] + offset - r["reference_seconds"] for r in held]
        maximum = float(np.max(np.abs(residual)))
        passed = maximum <= max_residual_seconds
        all_passed &= passed
        report["per_camera"][camera_id] = {"source_to_reference_offset_seconds": offset,
            "held_out_count": len(held), "held_out_max_abs_residual_seconds": maximum,
            "fit_count": len(fit), "passed": passed,
            "reference_span_seconds": [float(min(r["reference_seconds"] for r in rows)), float(max(r["reference_seconds"] for r in rows))]}
    if not common_held_out or len(common_held_out) < 2:
        report["reason"] = "Need two shared held-out reference events across all cameras"
        return report
    # Shared event labels must also refer to the same common-clock times.
    for event_id in common_held_out:
        event_times = [r["reference_seconds"] for rows in correspondences.values() for r in rows if r["event_id"] == event_id]
        if np.ptp(event_times) > 1e-9:
            raise ValueError("Shared reference event has inconsistent common timestamps")
    report.update({"synchronization_verified": bool(all_passed), "segmentation_ready": bool(all_passed),
                   "reason": "Held-out measured constant-offset mapping passed" if all_passed else "Temporal residual exceeds declared tolerance",
                   "evidence": evidence, "scope": "Observed reference span only; no full-recording drift guarantee"})
    return report
