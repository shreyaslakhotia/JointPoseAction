"""Metadata-only adapters for future learned reconstruction runs.

There are deliberately no imports of model packages, downloads or inference calls.
An input recipe is not a processed image and a valid artifact record is not proof
of geometric accuracy. Alignment belongs to the separate evaluation stage.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


MODEL_MANIFEST = Path(__file__).resolve().parents[1] / "reproducibility/upstreams/reconstruction_models.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KEYS = ("dataset_revision", "recording_id", "object_id", "camera_id", "mask_run_id")


def file_record(path: str | Path) -> dict[str, Any]:
    """Hash an existing file without copying it or trusting its extension."""
    p = Path(path).resolve(strict=True)
    if not p.is_file():
        raise ValueError(f"Expected a regular file: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(p), "sha256": digest.hexdigest(), "bytes": p.stat().st_size}


def camera_metadata_from_camera(camera, calibration_path: str | Path) -> dict[str, Any]:
    """Bridge the common Camera contract to an adapter recipe without guessing.

    Named units are retained verbatim; their physical meaning is not validated
    by this conversion. No resize, rectification or coordinate change occurs.
    """
    from .cameras import Camera
    if not isinstance(camera, Camera):
        raise ValueError('Expected the common validated Camera representation')
    return {'camera_id':camera.camera_id,'K':camera.K.tolist(),
            'world_to_camera':camera.world_to_camera.tolist(),
            'image_size_wh':list(camera.image_size),
            'convention':'opencv_x_right_y_down_z_forward','pixel_convention':'integer_centers',
            'distortion_model':'opencv_brown_conrady',
            'distortion_coefficients':dict(zip(('k1','k2','p1','p2','k3'),camera.distortion)),
            'coordinate_frame':camera.coordinate_frame,'units':camera.units,
            'calibration_file':file_record(calibration_path)}


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _verify_record(record: Mapping[str, Any]) -> None:
    actual = file_record(record["path"])
    if actual["sha256"] != record["sha256"] or actual["bytes"] != record["bytes"]:
        raise ValueError(f"File changed since provenance was recorded: {record['path']}")


def model_spec(method_id: str, manifest_path: str | Path = MODEL_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    try:
        return _json_copy(manifest["models"][method_id])
    except KeyError as exc:
        raise ValueError(f"Unknown reconstruction method: {method_id}") from exc


def crop_resize_transform(
    image_size_wh: Sequence[int], crop_xyxy: Sequence[int] | None = None,
    target_size_wh: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Describe integer half-open crop and half-pixel-center resize exactly.

    Coordinates refer to pixel centers at integer positions. This is an image
    grid transform, not undistortion, rectification, or a model camera estimate.
    """
    def positive_pair(value: Sequence[int], name: str) -> tuple[int, int]:
        if len(value) != 2 or any(isinstance(x, bool) or int(x) != x or x <= 0 for x in value):
            raise ValueError(f"{name} must contain two positive integers")
        return int(value[0]), int(value[1])

    width, height = positive_pair(image_size_wh, "image_size_wh")
    box = [0, 0, width, height] if crop_xyxy is None else list(crop_xyxy)
    if len(box) != 4 or any(isinstance(x, bool) or int(x) != x for x in box):
        raise ValueError("crop_xyxy must contain four integer half-open bounds")
    x0, y0, x1, y1 = map(int, box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Crop lies outside source image or is empty")
    out_w, out_h = positive_pair([x1 - x0, y1 - y0] if target_size_wh is None else target_size_wh, "target_size_wh")
    sx, sy = out_w / (x1 - x0), out_h / (y1 - y0)
    crop = np.array([[1., 0., -x0], [0., 1., -y0], [0., 0., 1.]])
    resize = np.array([[sx, 0., (sx - 1) / 2], [0., sy, (sy - 1) / 2], [0., 0., 1.]])
    affine = resize @ crop
    return {
        "image_size_wh": [width, height], "crop_xyxy_half_open": [x0, y0, x1, y1],
        "target_size_wh": [out_w, out_h], "pixel_center_convention": "integer_centers_half_pixel_resize",
        "original_to_crop": crop.tolist(), "crop_to_resized": resize.tolist(),
        "original_to_resized": affine.tolist(), "resized_to_original": np.linalg.inv(affine).tolist(),
        "rgb_interpolation_planned": "bilinear_half_pixel",
        "mask_interpolation_planned": "nearest_neighbor_half_pixel",
        "undistortion_executed": False, "processing_executed": False,
    }


def _camera_record(metadata: Mapping[str, Any], size_wh: Sequence[int], affine: np.ndarray) -> dict[str, Any]:
    camera = _json_copy(metadata)
    required = ("camera_id", "K", "world_to_camera", "image_size_wh", "convention", "pixel_convention", "distortion_model",
                "distortion_coefficients", "coordinate_frame", "units", "calibration_file")
    if any(key not in camera for key in required):
        raise ValueError(f"Camera metadata requires: {', '.join(required)}")
    if list(camera["image_size_wh"]) != list(size_wh):
        raise ValueError("Camera image dimensions differ from original RGB")
    if camera["convention"] != "opencv_x_right_y_down_z_forward" or camera["pixel_convention"] != "integer_centers":
        raise ValueError("Explicit OpenCV camera axes and integer pixel centers required; convert other conventions upstream")
    K = np.asarray(camera["K"], dtype=float)
    T = np.asarray(camera["world_to_camera"], dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("Invalid camera K")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or abs(np.linalg.det(K)) < 1e-12:
        raise ValueError("Camera K has invalid focal lengths or is singular")
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("Invalid world_to_camera matrix")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        raise ValueError("Camera rotation is not a proper rigid rotation")
    if camera["distortion_model"] not in ("none", "opencv_brown_conrady"):
        raise ValueError("Unsupported distortion schema; explicitly convert it before preparation")
    coefficients = camera["distortion_coefficients"]
    if not isinstance(coefficients, dict) or not np.isfinite(list(coefficients.values())).all():
        raise ValueError("Distortion coefficients require finite named values")
    if camera["distortion_model"] == "none" and any(coefficients.values()):
        raise ValueError("Nonzero distortion cannot be declared none")
    if camera["distortion_model"] == "opencv_brown_conrady" and set(coefficients) != {"k1", "k2", "p1", "p2", "k3"}:
        raise ValueError("Brown-Conrady coefficients must be named k1,k2,p1,p2,k3")
    units=camera['units']
    if (not isinstance(camera["coordinate_frame"], str) or not camera["coordinate_frame"]
            or not isinstance(units,str) or not (units in ('m','cm','mm','unknown','canonical_unitless') or units.startswith('synthetic_'))):
        raise ValueError("Explicit camera coordinate frame and units required")
    _verify_record(camera["calibration_file"])
    camera["resized_grid_K"] = (affine @ K).tolist()
    camera["resized_grid_K_scope"] = "image_grid_only_distortion_preserved_not_an_inferred_model_camera"
    camera["calibration_accuracy_verified_by_adapter"] = False
    return camera


def prepare_input_bundle(
    destination: str | Path, method_id: str, rgb_path: str | Path, mask_path: str | Path, *,
    camera_metadata: Mapping[str, Any], source_identifiers: Mapping[str, Any],
    crop_xyxy: Sequence[int] | None = None, target_size_wh: Sequence[int] | None = None,
    background_policy: str = "mask_as_alpha", model_preprocessing: Mapping[str, Any] | None = None,
    manifest_path: str | Path = MODEL_MANIFEST,
) -> dict[str, Any]:
    """Write only a JSON recipe referencing unmodified source RGB and binary mask.

    A later approved model wrapper must execute and record every preprocessing
    operation, including model-internal recentering. This function never marks
    real-data readiness, mask accuracy, metric scale, or camera alignment true.
    """
    spec = model_spec(method_id, manifest_path)
    source = _json_copy(source_identifiers)
    if any(not isinstance(source.get(key), str) or not source[key] for key in _SOURCE_KEYS):
        raise ValueError(f"Source identifiers require nonempty {', '.join(_SOURCE_KEYS)}")
    timestamp = source.get("source_timestamp")
    if not isinstance(timestamp, dict) or not isinstance(timestamp.get("basis"), str) or not timestamp["basis"]:
        raise ValueError("Source timestamp requires an explicit basis and seconds")
    seconds = timestamp.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Source timestamp seconds must be finite and nonnegative")
    if camera_metadata.get("camera_id") != source["camera_id"]:
        raise ValueError("Camera identifier mismatch")
    if background_policy not in ("preserve_rgb", "mask_as_alpha", "composite_white"):
        raise ValueError("Unknown background policy")
    rgb = file_record(rgb_path)
    mask_record = file_record(mask_path)
    with Image.open(rgb_path) as image:
        if image.mode != "RGB":
            raise ValueError("Source RGB must be RGB; resolve existing alpha explicitly upstream")
        size = list(image.size)
        image.verify()
    with Image.open(mask_path) as image:
        if image.mode not in ("1", "L"):
            raise ValueError("Mask must be a single-channel binary image")
        mask = np.asarray(image).copy()
        if list(image.size) != size:
            raise ValueError("RGB and mask dimensions differ")
    if not np.isin(mask, [0, 1, 255]).all() or not np.any(mask):
        raise ValueError("Mask must be nonempty binary; do not threshold probability/overlay images here")
    if np.any(mask == 1) and np.any(mask == 255):
        raise ValueError("Mixed mask foreground encoding is ambiguous")
    transform = crop_resize_transform(size, crop_xyxy, target_size_wh)
    x0, y0, x1, y1 = transform["crop_xyxy_half_open"]
    if np.count_nonzero(mask[y0:y1, x0:x1]) != np.count_nonzero(mask):
        raise ValueError("Crop removes selected foreground pixels")
    camera = _camera_record(camera_metadata, size, np.array(transform["original_to_resized"]))
    bundle = {
        "schema_version": 1, "purpose": "metadata_only_learned_reconstruction_input_recipe",
        "method_id": method_id, "model_source": spec["source"], "model_weights": spec["weights"],
        "model_manifest": file_record(manifest_path), "source_identifiers": source,
        "rgb": rgb, "mask": mask_record, "foreground_pixels": int(np.count_nonzero(mask)),
        "mask_status": "predicted_silhouette_not_ground_truth", "camera": camera,
        "preprocessing": {"grid_transform": transform, "background_policy": background_policy,
                          "background_rgb": [255, 255, 255] if background_policy == "composite_white" else None,
                          "alpha_recipe": "binary_mask_0_or_255" if background_policy == "mask_as_alpha" else None,
                          "model_internal_operations_planned": _json_copy(model_preprocessing or {}),
                          "model_internal_operations_verified_executed": False},
        "input_preparation_executed": False, "inference_executed": False,
        "reconstruction_ready": False, "camera_alignment_verified": False, "metric_scale_verified": False,
    }
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return bundle


def verify_input_bundle(path: str | Path) -> dict[str, Any]:
    """Detect source/calibration/manifest changes; does not grant run permission."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1 or bundle.get("purpose") != "metadata_only_learned_reconstruction_input_recipe":
        raise ValueError("Unsupported input recipe")
    for record in (bundle["rgb"], bundle["mask"], bundle["model_manifest"], bundle["camera"]["calibration_file"]):
        _verify_record(record)
    for flag in ("input_preparation_executed", "inference_executed", "reconstruction_ready", "camera_alignment_verified", "metric_scale_verified"):
        if bundle.get(flag) is not False:
            raise ValueError("Metadata-only recipes must not claim execution or scientific readiness")
    spec = model_spec(bundle["method_id"], bundle["model_manifest"]["path"])
    if bundle["model_source"] != spec["source"] or bundle["model_weights"] != spec["weights"]:
        raise ValueError("Recipe model identifiers do not match pinned manifest")
    transform = bundle["preprocessing"]["grid_transform"]
    expected = crop_resize_transform(transform["image_size_wh"], transform["crop_xyxy_half_open"], transform["target_size_wh"])
    if transform != expected:
        raise ValueError("Recipe grid transform is inconsistent")
    if bundle["camera"]["camera_id"] != bundle["source_identifiers"]["camera_id"]:
        raise ValueError("Recipe camera identity is inconsistent")
    camera = _camera_record(bundle["camera"], transform["image_size_wh"], np.array(transform["original_to_resized"]))
    if camera != bundle["camera"]:
        raise ValueError("Recipe camera/grid intrinsics are inconsistent")
    return bundle


def validate_external_artifact(
    artifact_path: str | Path, metadata: Mapping[str, Any], *, input_bundle_path: str | Path,
) -> dict[str, Any]:
    """Validate provenance for a future externally generated, unaligned artifact.

    This is file/metadata validation only. Gaussian PLY is explicitly distinct
    from surface points. Geometry loading/rendering and alignment are separate.
    A successful return never authorizes a real experiment.
    """
    bundle = verify_input_bundle(input_bundle_path)
    record = _json_copy(metadata)
    artifact = file_record(artifact_path)
    if artifact["bytes"] == 0:
        raise ValueError("Empty external artifact")
    if record.get("artifact_sha256") != artifact["sha256"]:
        raise ValueError("Artifact SHA-256 mismatch")
    if record.get("input_bundle_sha256") != file_record(input_bundle_path)["sha256"]:
        raise ValueError("Input recipe SHA-256 mismatch")
    for key, expected in (("method_id", bundle["method_id"]), ("source_commit", bundle["model_source"]["commit"]),
                          ("weights_revision", bundle["model_weights"]["revision"])):
        if record.get(key) != expected:
            raise ValueError(f"Artifact {key} mismatch")
    if not isinstance(record.get("checkpoint_sha256"), list) or not record["checkpoint_sha256"] or any(not isinstance(x, str) or not _SHA256.fullmatch(x) for x in record["checkpoint_sha256"]):
        raise ValueError("Exact checkpoint file SHA-256 identifiers required")
    if not isinstance(record.get("configuration_sha256"), str) or not _SHA256.fullmatch(record["configuration_sha256"]):
        raise ValueError("Configuration SHA-256 required")
    if not isinstance(record.get("configuration_file"), dict):
        raise ValueError("An exact configuration file record is required")
    _verify_record(record["configuration_file"])
    if record["configuration_file"]["sha256"] != record["configuration_sha256"]:
        raise ValueError("Configuration file and identifier disagree")
    representation = record.get("representation")
    if representation not in ("triangle_mesh", "point_cloud", "gaussian_splat"):
        raise ValueError("Explicit supported representation required")
    allowed = {"triangle_mesh": {".obj", ".glb", ".ply"}, "point_cloud": {".ply"}, "gaussian_splat": {".ply"}}
    if Path(artifact_path).suffix.lower() not in allowed[representation]:
        raise ValueError("Representation and artifact extension disagree")
    if record.get("coordinate_frame") != "model_native_unaligned" or record.get("units") not in ("canonical_unitless", "unknown"):
        raise ValueError("Native learned outputs cannot claim calibrated frame or metric units")
    for flag, expected in (("alignment_required", True), ("camera_alignment_verified", False), ("metric_scale_verified", False)):
        if record.get(flag) is not expected:
            raise ValueError(f"Unaligned native artifact requires {flag}={expected}")
    for key in ("runtime_seconds", "peak_gpu_memory_bytes"):
        value = record.get(key)
        if value is None:
            if not record.get(f"{key}_unavailable_reason"):
                raise ValueError(f"Missing {key} requires an explicit reason")
        elif isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid resource measurement: {key}")
    return {"schema_version": 1, "validation_scope": "file_identity_and_metadata_only", "artifact": artifact,
            "metadata": record, "source_identifiers": bundle["source_identifiers"],
            "geometry_validated": False, "checkpoint_payload_hashes_verified_here": False, "benchmark_ready": False,
            "warning": "Model-native coordinates and learned layout are not calibrated metric geometry"}
