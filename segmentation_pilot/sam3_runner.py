"""Bounded, provenance-preserving official SAM3 image inference.

This module imports no torch/SAM3 until all input and scientific gates pass.
It performs no download, training, instance selection, reconstruction or union.
Run ``python -m segmentation_pilot.sam3_runner --help`` for the interface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .metrics import describe_mask

SOURCE_REVISION = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
CHECKPOINT_REPOSITORY = "facebook/sam3"
CHECKPOINT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
CHECKPOINT_BYTES = 3450062241
CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
STATIC_SCOPE = "static_object_state_only"
DIAGNOSTIC_SCOPE = "diagnostic_2d_review"
DATASET_REVISION = "bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967"
SURVEY_CAMERAS = {"C10095", "C10115", "C10118", "C10119", "C10379", "C10390", "C10395", "C10404"}
STATIC_THRESHOLDS = {
    "maximum_displacement_native_px": 2.0,
    "maximum_forward_backward_native_px": 0.75,
    "minimum_valid_tracking_fraction": 0.95,
    "minimum_landmark_span_fraction_of_roi_each_axis": 0.2,
    "minimum_manually_identified_landmarks": 4,
    "claim_elapsed_seconds": [0.0, 1.0],
    "thresholds_predeclared_not_tuned_on_results": True,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"Expected nonempty {field}, at most 512 characters")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Finite numeric {field} required")
    return float(value)


def _hashed_json_evidence(gate: dict, stem: str, parent: Path) -> tuple[dict, dict]:
    path = Path(_text(gate.get(stem + "_path"), stem + "_path"))
    if not path.is_absolute():
        path = parent / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Existing {stem} JSON file required")
    expected = gate.get(stem + "_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Explicit SHA-256 required for {stem}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{stem} SHA-256 mismatch")
    report = read_json(path)
    if not isinstance(report, dict):
        raise ValueError(f"{stem} must be a JSON object")
    return report, {"path": str(path), "sha256": actual}


def _static_thresholds_unchanged(values: Any) -> bool:
    if not isinstance(values, dict) or set(values) != set(STATIC_THRESHOLDS):
        return False
    for key, expected in STATIC_THRESHOLDS.items():
        actual = values[key]
        if isinstance(expected, bool):
            if actual is not expected:
                return False
        elif isinstance(expected, list):
            if (not isinstance(actual, list) or len(actual) != len(expected)
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in actual)
                    or actual != expected):
                return False
        elif isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != expected:
            return False
    return True


def _validate_static_row(row: dict, gate: dict, parent: Path, expected_sha: str) -> dict:
    """Validate measured local stability plus separately attested cross-view state.

    Local optical flow does not establish cross-camera identity, configuration,
    calibration or synchronized clocks. Their manual attestations remain separate
    and no boolean alone can bypass the hashed measurement reports.
    """
    if (gate.get("static_object_state_correspondence_verified") is not True
            or gate.get("manual_identity_configuration_verified") is not True
            or gate.get("temporal_sync_verified") is not False
            or not isinstance(gate.get("object_ids"), list) or row["object_id"] not in gate["object_ids"]):
        raise ValueError("Static-state gate requires manual identity/configuration, exact object membership and temporal_sync_verified=false")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("Static-state images require an explicit source image SHA-256")
    pts = _finite_number(row.get("source_pts_seconds"), "source_pts_seconds")
    if not 0 <= pts <= 1:
        raise ValueError("Static-state source PTS must lie within the predeclared first 0..1 seconds")
    report, stability_source = _hashed_json_evidence(gate, "static_stability_evidence", parent)
    if not _static_thresholds_unchanged(report.get("thresholds")):
        raise ValueError("Static stability thresholds differ from the predeclared contract")
    if report.get("temporal_sync_verified") is not False:
        raise ValueError("Local stability evidence must not claim temporal synchronization")
    recordings = report.get("recordings")
    if not isinstance(recordings, list):
        raise ValueError("Static stability report requires measured recording records")
    matches = [r for r in recordings if isinstance(r, dict) and r.get("recording_id") == row["recording_id"]
               and r.get("object_id") == row["object_id"]]
    if len(matches) != 1:
        raise ValueError("Static stability recording/object identity mismatch or ambiguity")
    recording = matches[0]
    identity = recording.get("identity_evidence", {})
    if (recording.get("temporal_sync_verified") is not False or not isinstance(identity, dict)
            or identity.get("manually_verified") is not True):
        raise ValueError("Manual object identity and explicit unsynchronized local evidence required")
    for key in ("source", "description"):
        _text(identity.get(key), "static identity " + key)
    views = recording.get("views")
    if not isinstance(views, list):
        raise ValueError("Measured static stability views required")
    matches = [v for v in views if isinstance(v, dict) and v.get("camera_id") == row["camera_id"]]
    if len(matches) != 1:
        raise ValueError("Static stability camera mismatch or ambiguity")
    view = matches[0]
    if view.get("status") != "measured" or view.get("static_state_local_stability_passed") is not True:
        raise ValueError("Requested camera failed or lacks measured local static stability")
    if view.get("reference_frame_sha256") != expected_sha:
        raise ValueError("Static stability reference image SHA-256 differs from requested image")
    reference_pts = _finite_number(view.get("reference_pts_seconds"), "reference_pts_seconds")
    if not math.isclose(reference_pts, pts, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Static stability reference PTS differs from requested image")
    claim = view.get("claim_first_second", {})
    if not isinstance(claim, dict) or claim.get("static_state_local_stability_passed") is not True:
        raise ValueError("First-second measured static stability claim did not pass")
    span = claim.get("elapsed_span_seconds")
    if (not isinstance(span, list) or len(span) != 2
            or _finite_number(span[0], "stability span start") != 0
            or not 1 - 1/30 <= _finite_number(span[1], "stability span end") <= 1 + 1e-9):
        raise ValueError("Full predeclared first-second local stability span required")
    point_names, per_point = view.get("point_names"), claim.get("per_point")
    if (not isinstance(point_names, list) or len(point_names) < 4
            or any(not isinstance(n, str) or not n.strip() for n in point_names)
            or len(set(point_names)) != len(point_names)
            or not isinstance(per_point, list) or len(per_point) != len(point_names)
            or {p.get("point_name") for p in per_point if isinstance(p, dict)} != set(point_names)):
        raise ValueError("At least four uniquely named measured object landmarks required")
    landmark_span = view.get("roi_landmark_span_fraction")
    if (not isinstance(landmark_span, list) or len(landmark_span) != 2
            or any(_finite_number(x, "ROI landmark span") < .2 for x in landmark_span)):
        raise ValueError("Measured object landmarks must span both ROI axes")
    sampled_frames = claim.get("sampled_frames")
    if isinstance(sampled_frames, bool) or not isinstance(sampled_frames, int) or sampled_frames < 2:
        raise ValueError("Measured first-second frame count required")
    for point in per_point:
        if not isinstance(point, dict) or point.get("passed") is not True:
            raise ValueError("Every declared object landmark must pass local stability")
        fraction = _finite_number(point.get("valid_fraction"), "valid tracking fraction")
        drift = _finite_number(point.get("maximum_displacement_native_px"), "maximum landmark displacement")
        fb = _finite_number(point.get("maximum_forward_backward_native_px"), "maximum forward/backward error")
        total, valid = point.get("total_frames"), point.get("valid_frames")
        if (isinstance(total, bool) or not isinstance(total, int) or total != sampled_frames
                or isinstance(valid, bool) or not isinstance(valid, int) or not 0 <= valid <= total
                or not math.isclose(fraction, valid / total, rel_tol=0, abs_tol=1e-9)
                or not .95 <= fraction <= 1 or not 0 <= drift <= 2 or not 0 <= fb <= .75):
            raise ValueError("Recorded landmark measurements fail the unchanged static thresholds")
    calibration, calibration_source = _hashed_json_evidence(gate, "calibration_evidence", parent)
    calibration_recordings = calibration.get("recordings")
    if isinstance(calibration_recordings, list):
        calibration_matches = [r for r in calibration_recordings if isinstance(r, dict)
                               and r.get("recording_id") == row["recording_id"]]
    elif calibration.get("recording_id") == row["recording_id"]:
        calibration_matches = [calibration]
    else:
        calibration_matches = []
    if len(calibration_matches) != 1:
        raise ValueError("Calibration evidence does not match the exact recording")
    return {"static_stability_evidence": stability_source, "calibration_evidence": calibration_source,
            "reference_frame_sha256": expected_sha, "reference_pts_seconds": reference_pts,
            "camera_id": row["camera_id"], "object_id": row["object_id"],
            "local_stability_measurements_validated": True,
            "manual_cross_view_identity_configuration_attestation": True}


def box_to_normalized_cxcywh(box: Any, width: int, height: int) -> list[float]:
    if isinstance(width, bool) or isinstance(height, bool) or min(width, height) < 1:
        raise ValueError("Positive image dimensions required")
    try:
        values = np.asarray(box, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Box must contain four finite pixel XYXY coordinates") from error
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Box must contain four finite pixel XYXY coordinates")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box must have positive area and lie within the original image")
    return [float((x1 + x2) / (2 * width)), float((y1 + y2) / (2 * height)),
            float((x2 - x1) / width), float((y2 - y1) / height)]


def _validate_diagnostic_manifest(path: Path, manifest: dict, max_images: int) -> tuple[dict, list[dict], dict]:
    """Permit only explicitly requested 2D diagnosis, never pilot acceptance."""
    if (manifest.get("purpose") != DIAGNOSTIC_SCOPE
            or any(manifest.get(key) is not False for key in
                   ("multiview_dataset_ready", "calibration_accuracy_verified", "temporal_sync_verified"))
            or manifest.get("calibration_temporal_gate_passed") is True
            or manifest.get("calibration_static_gate_passed") is True):
        raise ValueError("Explicit diagnostic purpose and false dataset/calibration/synchronization claims are required")
    survey, survey_source = _hashed_json_evidence(manifest, "source_survey", path.parent)
    assessment, assessment_source = _hashed_json_evidence(manifest, "calibration_assessment", path.parent)
    if survey.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Diagnostic source survey dataset revision mismatch")
    if (not isinstance(survey.get("cameras"), list) or len(survey["cameras"]) != 8
            or set(survey["cameras"]) != SURVEY_CAMERAS):
        raise ValueError("Diagnostic source survey must identify the eight fixed camera IDs")
    recordings, assessments = survey.get("recordings"), assessment.get("recordings")
    if not isinstance(recordings, list) or not isinstance(assessments, list):
        raise ValueError("Actual source survey and calibration assessment recording lists required")
    if any(assessment.get(key) is True for key in
           ("multiview_dataset_ready", "calibration_accuracy_verified", "temporal_sync_verified")):
        raise ValueError("Diagnostic calibration assessment must retain unresolved validation claims")
    stability, stability_source = None, None
    if "tracking_stability_path" in manifest or "tracking_stability_sha256" in manifest:
        stability, stability_source = _hashed_json_evidence(manifest, "tracking_stability", path.parent)
        if not isinstance(stability.get("recordings"), list):
            raise ValueError("Tracking evidence must contain the real per-recording measurements")
    rows = manifest.get("images")
    if not isinstance(rows, list) or not 1 <= len(rows) <= max_images:
        raise ValueError(f"Manifest must contain 1 to {max_images} images")
    validated, image_ids, image_evidence = [], set(), {}
    from PIL import Image
    for original in rows:
        if not isinstance(original, dict):
            raise ValueError("Each diagnostic image must be an object")
        row = dict(original)
        for key in ("image_id", "path", "object_id", "recording_id", "camera_id", "prompt"):
            _text(row.get(key), key)
        image_id = row["image_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", image_id) or image_id in image_ids:
            raise ValueError("image_id must be unique and filename-safe (at most 100 characters)")
        image_ids.add(image_id)
        if any(key in row for key in ("action_label", "action_id", "action_cls", "ground_truth_action")):
            raise ValueError("Action labels are not permitted in diagnostic instance selection")
        checksum = row.get("sha256")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("Diagnostic images require an explicit native image SHA-256")
        pts = _finite_number(row.get("source_pts_seconds"), "source_pts_seconds")
        matches = [r for r in recordings if isinstance(r, dict) and r.get("recording_id") == row["recording_id"]]
        if len(matches) != 1:
            raise ValueError("Diagnostic recording missing or ambiguous in actual source survey")
        recording = matches[0]
        views = recording.get("views", [])
        if (not isinstance(views, list) or len(views) != 8
                or {v.get("camera_id") for v in views if isinstance(v, dict)} != SURVEY_CAMERAS):
            raise ValueError("Source recording must retain the eight-camera survey inventory")
        matches = [v for v in views if isinstance(v, dict) and v.get("camera_id") == row["camera_id"]]
        if len(matches) != 1 or matches[0].get("status") != "decoded":
            raise ValueError("Diagnostic camera has no uniquely decoded source survey view")
        view = matches[0]
        if (view.get("source_path") != f"recordings/{row['recording_id']}/{row['camera_id']}_rgb.mp4"
                or view.get("source_revision") != DATASET_REVISION):
            raise ValueError("Diagnostic source video recording/camera/revision mapping changed")
        frames = view.get("frames")
        if not isinstance(frames, list):
            raise ValueError("Native source survey frames required")
        matches = [f for f in frames if isinstance(f, dict) and f.get("sha256") == checksum
                   and isinstance(f.get("pts_seconds"), (int, float))
                   and not isinstance(f.get("pts_seconds"), bool)
                   and math.isclose(f["pts_seconds"], pts, rel_tol=0, abs_tol=1e-9)]
        if len(matches) != 1:
            raise ValueError("Diagnostic image SHA-256/PTS identity is absent or ambiguous in the source camera frames")
        frame = matches[0]
        image_path = Path(row["path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve(strict=True)
        source_path = Path(_text(frame.get("path"), "survey frame path"))
        if not source_path.is_absolute():
            source_path = Path(survey_source["path"]).parent / source_path
        source_path = source_path.resolve(strict=True)
        for checked_path in {image_path, source_path}:
            if not checked_path.is_file() or file_sha256(checked_path) != checksum:
                raise ValueError("Diagnostic native frame file SHA-256 differs from the actual source survey")
        size = frame.get("image_size")
        if (not isinstance(size, list) or len(size) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in size)):
            raise ValueError("Source survey native frame dimensions required")
        with Image.open(image_path) as image:
            if list(image.size) != size:
                raise ValueError("Diagnostic frame dimensions differ from the native source survey")
            image.verify()
        if "image_size" in view and view["image_size"] != size:
            raise ValueError("Source survey frame/view native image dimensions disagree")
        matches = [r for r in assessments if isinstance(r, dict) and r.get("recording_id") == row["recording_id"]]
        if len(matches) != 1 or matches[0].get("calibration_accuracy_verified") is not False:
            raise ValueError("Exact recording calibration assessment must explicitly retain unverified accuracy")
        assessment_record = matches[0]
        if assessment_record.get("temporal_sync_verified") is True or assessment_record.get("multiview_dataset_ready") is True:
            raise ValueError("Diagnostic calibration assessment must not promote synchronization or dataset readiness")
        refinement = row.get("refinement")
        if refinement is not None:
            if not isinstance(refinement, dict) or set(refinement) != {"box_xyxy", "reason", "intervention"}:
                raise ValueError("Refinement requires only box_xyxy, reason and intervention")
            _text(refinement.get("reason"), "refinement reason")
            if refinement.get("intervention") not in {"human", "agent_visual_review"}:
                raise ValueError("Diagnostic refinement must record a human or agent visual intervention")
            box_to_normalized_cxcywh(refinement["box_xyxy"], *size)
        tracking = {"available": False, "status": "not_assessed_for_this_object_view"}
        if stability is not None:
            recording_matches = [r for r in stability["recordings"] if isinstance(r, dict)
                                 and r.get("recording_id") == row["recording_id"] and r.get("object_id") == row["object_id"]]
            if len(recording_matches) > 1:
                raise ValueError("Ambiguous tracking evidence recording/object identity")
            if recording_matches:
                view_matches = [v for v in recording_matches[0].get("views", []) if isinstance(v, dict)
                                and v.get("camera_id") == row["camera_id"]]
                if len(view_matches) > 1:
                    raise ValueError("Ambiguous per-view tracking evidence")
                if view_matches:
                    measured = view_matches[0]
                    tracking = {"available": True, "status": measured.get("status"),
                                "static_state_local_stability_passed": measured.get("static_state_local_stability_passed"),
                                "reference_frame_sha256": measured.get("reference_frame_sha256"),
                                "claim_first_second": measured.get("claim_first_second"),
                                "full_window": measured.get("full_window"),
                                "not_an_acceptance_gate": True}
        row.update(path=str(image_path), correspondence_scope=DIAGNOSTIC_SCOPE, diagnostic_2d_review=True,
                   multiview_dataset_ready=False, calibration_accuracy_verified=False, temporal_sync_verified=False)
        validated.append(row)
        image_evidence[image_id] = {"source_survey_frame_path": str(source_path), "source_frame_sha256": checksum,
                                    "source_pts_seconds": pts, "native_image_size": size,
                                    "calibration_assessment_record": assessment_record,
                                    "tracking_diagnostics": tracking}
    return manifest, validated, {"purpose": DIAGNOSTIC_SCOPE, "diagnostic_2d_review": True,
                                "multiview_dataset_ready": False, "calibration_accuracy_verified": False,
                                "temporal_sync_verified": False, "correspondence_scope": DIAGNOSTIC_SCOPE,
                                "source_survey": survey_source, "calibration_assessment": assessment_source,
                                "tracking_stability": stability_source, "diagnostic_image_evidence": image_evidence,
                                "scope_limitations": ["Diagnostic independent 2D segmentation only; not an accepted calibrated multiview dataset.",
                                                       "Camera calibration accuracy and temporal synchronization remain unverified.",
                                                       "Inference completion does not pass temporal/static gates or authorize reconstruction benchmarks.",
                                                       "Local tracking status is descriptive; no segmentation accuracy is established without independent references."]}


def validate_image_manifest(path: Path, *, max_images: int = 32, diagnostic_2d_review: bool = False) -> tuple[dict, list[dict], dict]:
    """Require explicit recording/camera calibration and temporal evidence.

    Manifest: {calibration_temporal_gate_passed:true, gate_evidence_path:...,
      images:[{image_id,path,object_id,recording_id,camera_id,prompt,
               refinement?:{box_xyxy,reason,intervention}}]}.
    Evidence: {recordings:{recording_id:{calibration_verified:true,
      temporal_correspondence_verified:true,camera_ids:[...],evidence:"..."}}}.
    Alternative static-only manifest: calibration_static_gate_passed=true and
    correspondence_scope='static_object_state_only'. Its recording gate needs
    static_object_state_correspondence_verified=true, temporal_sync_verified=false,
    manual_identity_configuration_verified=true, object_ids, and existing hashed
    static_stability_evidence/calibration_evidence JSON paths. Images additionally
    require sha256 and source_pts_seconds matching measured initial reference
    frames. Local landmark stability never establishes synchronized clocks.
    Diagnostic manifests have purpose='diagnostic_2d_review', all three readiness/
    calibration/synchronization flags false, and hashed source_survey and
    calibration_assessment paths. They are accepted only with the explicit
    diagnostic_2d_review argument/CLI flag and cannot satisfy either normal gate.
    Paths are relative to their manifest. Gates must be established from real
    diagnostics; this function cannot establish physical validity from a flag.
    """
    path = Path(path).resolve(strict=True)
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("Calibration/temporal gate has not passed; segmentation is blocked")
    if diagnostic_2d_review:
        return _validate_diagnostic_manifest(path, manifest, max_images)
    if manifest.get("purpose") == DIAGNOSTIC_SCOPE or manifest.get("diagnostic_2d_review") is True:
        raise ValueError("Diagnostic manifests require explicit --diagnostic-2d-review; validated gates remain closed")
    temporal_branch = manifest.get("calibration_temporal_gate_passed") is True
    static_branch = manifest.get("calibration_static_gate_passed") is True
    if temporal_branch == static_branch:
        raise ValueError("Exactly one verified temporal or explicitly narrow static-state gate is required")
    if static_branch and manifest.get("correspondence_scope") != STATIC_SCOPE:
        raise ValueError("Static-state inference must declare correspondence_scope=static_object_state_only")
    if temporal_branch and manifest.get("correspondence_scope") == STATIC_SCOPE:
        raise ValueError("Static-only scope cannot be passed through the temporal gate")
    gate_path = Path(_text(manifest.get("gate_evidence_path"), "gate_evidence_path"))
    if not gate_path.is_absolute():
        gate_path = path.parent / gate_path
    gate_path = gate_path.resolve(strict=True)
    evidence = read_json(gate_path)
    if not isinstance(evidence, dict) or not isinstance(evidence.get("recordings"), dict):
        raise ValueError("Gate evidence must contain per-recording validations")
    rows = manifest.get("images")
    if not isinstance(rows, list) or not 1 <= len(rows) <= max_images:
        raise ValueError(f"Manifest must contain 1 to {max_images} images")
    image_ids = set()
    validated = []
    static_evidence = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each image must be an object")
        row = dict(row)
        for key in ("image_id", "path", "object_id", "recording_id", "camera_id", "prompt"):
            _text(row.get(key), key)
        image_id = row["image_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", image_id) or image_id in image_ids:
            raise ValueError("image_id must be unique and filename-safe (at most 100 characters)")
        image_ids.add(image_id)
        if any(key in row for key in ("action_label", "action_id", "action_cls", "ground_truth_action")):
            raise ValueError("Action labels are not permitted in the segmentation inference manifest")
        gate = evidence["recordings"].get(row["recording_id"], {})
        if (not isinstance(gate, dict) or gate.get("calibration_verified") is not True
                or (temporal_branch and gate.get("temporal_correspondence_verified") is not True)
                or not isinstance(gate.get("camera_ids"), list)
                or row["camera_id"] not in gate["camera_ids"]):
            raise ValueError(f"No verified recording/camera gate for {image_id}")
        _text(gate.get("evidence"), "recording gate evidence")
        image_path = Path(row["path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve(strict=True)
        if not image_path.is_file():
            raise ValueError("Image path must be an existing file")
        row["path"] = str(image_path)
        expected_sha = row.get("sha256")
        actual_sha = file_sha256(image_path)
        if expected_sha is not None and expected_sha != actual_sha:
            raise ValueError(f"Image SHA-256 mismatch: {image_id}")
        if static_branch:
            static_evidence[image_id] = _validate_static_row(row, gate, gate_path.parent, expected_sha)
            row["correspondence_scope"] = STATIC_SCOPE
            row["temporal_sync_verified"] = False
        row["sha256"] = actual_sha
        refinement = row.get("refinement")
        if refinement is not None:
            if not isinstance(refinement, dict) or set(refinement) != {"box_xyxy", "reason", "intervention"}:
                raise ValueError("Refinement requires only box_xyxy, reason and intervention")
            _text(refinement.get("reason"), "refinement reason")
            if refinement.get("intervention") not in {"human", "agent_visual_review"}:
                raise ValueError("Refinement intervention must identify human or agent_visual_review")
            # Check numeric form now; image-dependent bounds are checked after decode.
            box = np.asarray(refinement["box_xyxy"], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("Refinement box must contain four finite pixel coordinates")
        validated.append(row)
    return manifest, validated, {"path": str(gate_path), "sha256": file_sha256(gate_path),
                                "recordings": evidence["recordings"],
                                "correspondence_scope": STATIC_SCOPE if static_branch else "verified_temporal_correspondence",
                                "temporal_sync_verified": False if static_branch else None,
                                "static_image_evidence": static_evidence,
                                "scope_limitations": (["Static physical object state only; absolute and cross-camera synchronization remain unverified.",
                                                        "Local landmark stability does not independently prove whole-object rigidity or cross-view identity.",
                                                        "Separate manual identity/configuration and calibration evidence is required; no reconstruction benchmark readiness is asserted."]
                                                       if static_branch else [])}


def verify_model_inputs(source: Path, checkpoint: Path) -> dict:
    source, checkpoint = Path(source).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    git = ["git", "-c", f"safe.directory={source.as_posix()}", "-C", str(source)]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Official SAM3 source revision mismatch: {revision}")
    subprocess.run(git + ["diff", "--no-ext-diff", "--exit-code", "HEAD", "--"], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    untracked = subprocess.check_output(git + ["ls-files", "--others", "--exclude-standard", "--", "sam3"], text=True)
    if any(name.endswith(".py") for name in untracked.splitlines()):
        raise ValueError("Untracked Python code in official SAM3 source")
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("Official SAM3 checkpoint byte size mismatch")
    checksum = file_sha256(checkpoint)
    if checksum != CHECKPOINT_SHA256:
        raise ValueError("Official SAM3 checkpoint SHA-256 mismatch")
    source_files = {}
    for name in ("sam3/model_builder.py", "sam3/model/sam3_image_processor.py", "pyproject.toml"):
        source_files[name] = file_sha256(source / name)
    return {"source_repository": "https://github.com/facebookresearch/sam3", "source_revision": revision,
            "source_path": str(source), "source_files_sha256": source_files,
            "checkpoint_repository": CHECKPOINT_REPOSITORY, "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_filename": "sam3.pt", "checkpoint_path": str(checkpoint),
            "checkpoint_bytes": CHECKPOINT_BYTES, "checkpoint_sha256": checksum}


def pack_native_masks(masks: Any, height: int, width: int) -> dict[str, np.ndarray]:
    """Pack each independent candidate without union, filtering or postprocessing."""
    value = np.asarray(masks)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.dtype != np.bool_ or value.ndim != 3 or value.shape[1:] != (height, width):
        raise ValueError("Native SAM3 masks must be Boolean N x H x W or N x 1 x H x W")
    flattened = value.reshape(len(value), height * width)
    return {"packed_masks": np.packbits(flattened, axis=1, bitorder="little"),
            "mask_shape": np.asarray(value.shape, dtype=np.int64)}


def unpack_native_masks(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as record:
        shape = tuple(int(x) for x in record["mask_shape"])
        if len(shape) != 3 or min(shape[1:]) < 1 or shape[0] < 0:
            raise ValueError("Invalid native-mask shape")
        packed = record["packed_masks"]
        expected = (shape[0], math.ceil(shape[1] * shape[2] / 8))
        if packed.dtype != np.uint8 or packed.shape != expected:
            raise ValueError("Invalid packed-mask data")
        return np.unpackbits(packed, axis=1, count=shape[1] * shape[2], bitorder="little").reshape(shape).astype(bool)


def save_stage(output: Path, name: str, masks: Any, boxes: Any, scores: Any,
               *, height: int, width: int, prompt: str, refinement: dict | None = None,
               probabilities: Any = None, evaluation_threshold: float = 0.5,
               correspondence_scope: str = "unspecified", temporal_sync_verified: bool | None = None,
               diagnostic_2d_review: bool = False) -> dict:
    """Persist model arrays before any next call mutates processor state."""
    if not 0 <= evaluation_threshold <= 1:
        raise ValueError("Evaluation confidence threshold must lie in [0, 1]")
    if diagnostic_2d_review and (correspondence_scope != DIAGNOSTIC_SCOPE or temporal_sync_verified is not False):
        raise ValueError("Diagnostic candidate outputs must retain their explicit unsynchronized 2D-only scope")
    packed = pack_native_masks(masks, height, width)
    native_masks = np.asarray(masks).reshape(-1, height, width)
    boxes, scores = np.asarray(boxes), np.asarray(scores)
    count = len(native_masks)
    if (boxes.shape != (count, 4) or scores.shape != (count,)
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Candidate masks, finite pixel boxes and grounding scores do not align")
    stage = output / name
    stage.mkdir(parents=False, exist_ok=False)
    np.savez_compressed(stage / "native_masks.npz", **packed)
    probability_info = None
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.shape == (count, 1, height, width):
            probabilities = probabilities[:, 0]
        if (probabilities.shape != (count, height, width) or not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()):
            raise ValueError("Processor masks_logits must contain finite sigmoid probabilities")
        if not np.array_equal(probabilities > 0.5, native_masks):
            raise ValueError("Binary masks differ from native sigmoid probabilities > 0.5")
        np.savez_compressed(stage / "native_probabilities.npz", probabilities=probabilities)
        probability_info = {"path": "native_probabilities.npz", "semantics": "sigmoid_probability_not_logits",
                            "sha256": file_sha256(stage / "native_probabilities.npz")}
    candidates = []
    for index in range(count):
        metrics = describe_mask(native_masks[index], float(scores[index]))
        candidates.append({"candidate_id": index, "box_xyxy_pixels": boxes[index].tolist(),
                           "above_evaluation_threshold": bool(scores[index] > evaluation_threshold),
                           "diagnostic_2d_review": diagnostic_2d_review, "multiview_dataset_ready": False,
                           "calibration_accuracy_verified": False if diagnostic_2d_review else None,
                           "temporal_sync_verified": temporal_sync_verified, **metrics})
    record = {"stage": name, "prompt": prompt, "refinement": refinement,
              "correspondence_scope": correspondence_scope, "temporal_sync_verified": temporal_sync_verified,
              "diagnostic_2d_review": diagnostic_2d_review, "multiview_dataset_ready": False,
              "calibration_accuracy_verified": False if diagnostic_2d_review else None,
              "candidate_count": count, "evaluation_threshold": evaluation_threshold,
              "candidate_storage_threshold": 0.0, "binary_mask_threshold": 0.5,
              "score_semantics": "class_sigmoid_times_presence_sigmoid; not measured accuracy",
              "native_masks": {"path": "native_masks.npz", "bitorder": "little", "shape": [count, height, width],
                               "sha256": file_sha256(stage / "native_masks.npz")},
              "native_probabilities": probability_info, "candidates": candidates,
              "selected_candidate_id": None, "review_status": "pending_independent_visual_review",
              "instance_union_performed": False, "accuracy_established": False}
    (stage / "candidates.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def run(args) -> dict:
    manifest_path = Path(args.images).resolve(strict=True)
    diagnostic = getattr(args, "diagnostic_2d_review", False)
    manifest, images, gate = validate_image_manifest(manifest_path, max_images=args.max_images, diagnostic_2d_review=diagnostic)
    model_provenance = verify_model_inputs(Path(args.source), Path(args.checkpoint))
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output must be a new directory; prior text/refined results are immutable")
    # Heavy imports are deliberately below all provenance/gate checks.
    from PIL import Image
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference must run inside an authorized Slurm GPU allocation")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SLURM_JOB_ID required; do not execute inference on a login node")
    if "sam3" in sys.modules:
        raise RuntimeError("SAM3 was imported before source verification")
    sys.path.insert(0, str(Path(args.source).resolve()))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    decoded = []
    for row in images:
        with Image.open(row["path"]) as image:
            width, height = image.size
            if width * height > args.max_image_pixels or min(width, height) < 1:
                raise ValueError("Image resolution exceeds the declared bound")
            if row.get("refinement"):
                box_to_normalized_cxcywh(row["refinement"]["box_xyxy"], width, height)
            decoded.append((width, height))
    maximum_packed_bytes = sum(math.ceil(w * h / 8) * args.max_candidates * (2 if row.get("refinement") else 1)
                               for row, (w, h) in zip(images, decoded))
    maximum_probability_bytes = sum(w * h * 4 * args.max_candidates * (2 if row.get("refinement") else 1)
                                    for row, (w, h) in zip(images, decoded)) if args.save_probabilities else 0
    if maximum_packed_bytes + maximum_probability_bytes > args.max_output_bytes:
        raise ValueError("Worst-case native-mask/probability output exceeds the declared byte bound")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    provenance = {**model_provenance, "images_manifest_path": str(manifest_path),
                  "images_manifest_sha256": file_sha256(manifest_path), "gate_evidence": gate,
                  "purpose": DIAGNOSTIC_SCOPE if diagnostic else "gated_inference_for_review",
                  "diagnostic_2d_review": diagnostic, "multiview_dataset_ready": False,
                  "calibration_accuracy_verified": False if diagnostic else None,
                  "correspondence_scope": gate["correspondence_scope"],
                  "temporal_sync_verified": gate["temporal_sync_verified"],
                  "scope_limitations": gate["scope_limitations"],
                  "slurm_job_id": os.environ["SLURM_JOB_ID"], "python": sys.version,
                  "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0), "precision": "bfloat16_autocast",
                  "save_probabilities": args.save_probabilities, "max_candidates": args.max_candidates,
                  "worst_case_packed_mask_bytes": maximum_packed_bytes,
                  "worst_case_probability_bytes": maximum_probability_bytes,
                  "training_performed": False, "reconstruction_performed": False}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    try:
        torch.cuda.reset_peak_memory_stats()
        model = build_sam3_image_model(checkpoint_path=str(Path(args.checkpoint).resolve()), load_from_HF=False,
                                      device="cuda", eval_mode=True, enable_inst_interactivity=False, compile=False)
        processor = Sam3Processor(model, confidence_threshold=0.0)
        records = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row, (width, height) in zip(images, decoded):
                image_output = output / row["image_id"]
                image_output.mkdir()
                with Image.open(row["path"]) as original:
                    image = original.convert("RGB")
                    image.save(image_output / "frame.png")
                state = processor.set_image(image)
                stages = []

                def persist(stage_name, refinement=None):
                    count = len(state["scores"])
                    if count > args.max_candidates:
                        raise ValueError(f"Model returned {count} candidates, exceeding the declared bound")
                    # .numpy() cannot represent bf16; float32 preserves its values exactly.
                    arrays = {key: state[key].detach().cpu().float().numpy() for key in ("boxes", "scores")}
                    arrays["masks"] = state["masks"].detach().cpu().numpy().copy()
                    probabilities = state["masks_logits"].detach().cpu().float().numpy().copy() if args.save_probabilities else None
                    saved = save_stage(image_output, stage_name, arrays["masks"], arrays["boxes"], arrays["scores"],
                                       height=height, width=width, prompt=row["prompt"], refinement=refinement,
                                       probabilities=probabilities, evaluation_threshold=args.confidence_threshold,
                                       correspondence_scope=gate["correspondence_scope"],
                                       temporal_sync_verified=gate["temporal_sync_verified"], diagnostic_2d_review=diagnostic)
                    stages.append({"stage": stage_name, "candidate_count": saved["candidate_count"],
                                   "candidates_path": f"{row['image_id']}/{stage_name}/candidates.json"})

                state = processor.set_text_prompt(row["prompt"], state)
                persist("text_only")
                refinement = row.get("refinement")
                if refinement:
                    box = box_to_normalized_cxcywh(refinement["box_xyxy"], width, height)
                    state = processor.add_geometric_prompt(box, True, state)
                    persist("refined", refinement)
                records.append({**row, "frame_path": f"{row['image_id']}/frame.png",
                                "frame_sha256": file_sha256(image_output / "frame.png"),
                                "width": width, "height": height, "stages": stages})
                actual_bytes = sum(item.stat().st_size for item in output.rglob("*") if item.is_file())
                if actual_bytes > args.max_output_bytes:
                    raise ValueError("Output byte bound exceeded; aborting remaining images")
                del state
        summary = {"status": "diagnostic_2d_inference_completed_review_pending" if diagnostic else "inference_completed_review_pending",
                   "purpose": DIAGNOSTIC_SCOPE if diagnostic else "gated_inference_for_review",
                   "diagnostic_2d_review": diagnostic, "multiview_dataset_ready": False,
                   "calibration_accuracy_verified": False if diagnostic else None,
                   "images": records,
                   "correspondence_scope": gate["correspondence_scope"],
                   "temporal_sync_verified": gate["temporal_sync_verified"],
                   "scope_limitations": gate["scope_limitations"],
                   "image_count": len(records), "elapsed_seconds": time.monotonic() - started,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "accuracy_established": False, "independent_reference_available": False}
        (output / "completed.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return summary
    except Exception as error:
        (output / "failed.json").write_text(json.dumps({"status": "failed", "error_type": type(error).__name__,
                                                       "error": str(error)}, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Validated images JSON manifest; schema in validate_image_manifest docstring")
    parser.add_argument("--checkpoint", required=True, help="Pinned official sam3.pt; no automatic downloads")
    parser.add_argument("--source", required=True, help="Clean checkout of the pinned official SAM3 Git revision")
    parser.add_argument("--output", required=True, help="New dedicated project-storage output directory")
    parser.add_argument("--diagnostic-2d-review", action="store_true",
                        help="Explicit independent 2D diagnosis only; requires diagnostic manifest, keeps calibrated multiview/static acceptance gates closed")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Descriptive threshold; all returned positive-score candidates are saved")
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--max-image-pixels", type=int, default=4096 * 2160)
    parser.add_argument("--max-output-bytes", type=int, default=2 * 1024 ** 3)
    parser.add_argument("--save-probabilities", action="store_true", help="Optional native sigmoid arrays; subject to worst-case output-size bound")
    args = parser.parse_args()
    if (not 1 <= args.max_images <= 64 or not 1 <= args.max_candidates <= 400
            or not 1 <= args.max_image_pixels <= 4096 * 2160
            or not 1 <= args.max_output_bytes <= 8 * 1024 ** 3
            or not 0 <= args.confidence_threshold <= 1):
        parser.error("Requested bounds exceed the bounded pilot limits")
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "image_count", "elapsed_seconds", "accuracy_established")}))


if __name__ == "__main__":
    main()
