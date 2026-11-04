"""Bounded optical-flow diagnostics of manually identified object landmarks.

These tests characterize local image stability, not cross-view identity,
synchronization, metric geometry, object rigidity, or segmentation accuracy.
The predeclared acceptance scope is each source camera's first 0..1 seconds;
the complete <=3-second trace is retained separately and never hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from segmentation_pilot.acquire_frames import (
    ByteBudget, HTTPRangeIO, http_fetcher, check_project_quota, save_bytes, DATASET_REVISION,
)
from segmentation_pilot.calibration import CAMERA_IDS

MAX_DISPLACEMENT_PX = 2.0
MAX_FORWARD_BACKWARD_PX = .75
MIN_VALID_FRACTION = .95
MIN_ROI_SPAN_FRACTION = .2
CLAIM_SECONDS = 1.0
MAX_NETWORK_BYTES = 805_306_368
MAX_OUTPUT_BYTES = 134_217_728


def thresholds():
    return {"maximum_displacement_native_px": MAX_DISPLACEMENT_PX,
            "maximum_forward_backward_native_px": MAX_FORWARD_BACKWARD_PX,
            "minimum_valid_tracking_fraction": MIN_VALID_FRACTION,
            "minimum_landmark_span_fraction_of_roi_each_axis": MIN_ROI_SPAN_FRACTION,
            "minimum_manually_identified_landmarks": 4, "claim_elapsed_seconds": [0., CLAIM_SECONDS],
            "thresholds_predeclared_not_tuned_on_results": True}


def validate_landmark_spec(spec, image_size=(1920, 1080)):
    if spec.get("camera_id") not in CAMERA_IDS or spec.get("manual_object_landmarks_verified") is not True:
        raise ValueError("Known camera and manually verified object landmarks required")
    points = np.asarray(spec["points_xy"], dtype=np.float64)
    roi = np.asarray(spec["roi_xyxy"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("At least four finite manually selected object landmarks required")
    names = spec.get("point_names", [])
    if len(names) != len(points) or len(set(names)) != len(names) or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("Unique physical-part landmark names required")
    if roi.shape != (4,) or not np.isfinite(roi).all():
        raise ValueError("Finite native image ROI required")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= image_size[0] and 0 <= y0 < y1 <= image_size[1]):
        raise ValueError("ROI must lie within native image")
    if np.any(points < roi[:2]) or np.any(points > roi[2:]):
        raise ValueError("Object landmarks must lie within declared object ROI")
    span = np.ptp(points, axis=0)/(roi[2:]-roi[:2])
    if np.any(span < MIN_ROI_SPAN_FRACTION):
        raise ValueError("Landmarks must be distributed in both object ROI axes")
    digest = spec.get("reference_frame_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Exact survey reference image SHA-256 required")
    return points, roi, span


def summarize_trace(trace, point_names, *, end_seconds=CLAIM_SECONDS):
    """Aggregate retained measurements; unavailable tracks never count as zero motion."""
    frames = [r for r in trace if 0 <= r["elapsed_seconds"] <= end_seconds+1e-9]
    if len(frames) < 2 or frames[0]["elapsed_seconds"] != 0 or frames[-1]["elapsed_seconds"] < end_seconds-1/30:
        return {"static_state_local_stability_passed": False, "reason": "Requested local time span not observed",
                "elapsed_span_seconds": [frames[0]["elapsed_seconds"], frames[-1]["elapsed_seconds"]] if frames else []}
    per_point = []
    for index, name in enumerate(point_names):
        rows = [frame["points"][index] for frame in frames]
        valid = [r for r in rows if r["valid"]]
        drift = [r["displacement_px"] for r in valid if r.get("displacement_px") is not None]
        fb = [r["forward_backward_px"] for r in rows if r.get("forward_backward_px") is not None]
        fraction = len(valid)/len(rows)
        maximum_drift = max(drift) if drift else None
        maximum_fb = max(fb) if fb else None
        passed = (fraction >= MIN_VALID_FRACTION and maximum_drift is not None and maximum_drift <= MAX_DISPLACEMENT_PX
                  and maximum_fb is not None and maximum_fb <= MAX_FORWARD_BACKWARD_PX)
        per_point.append({"point_name": name, "valid_fraction": fraction, "valid_frames": len(valid),
                          "total_frames": len(rows), "maximum_displacement_native_px": maximum_drift,
                          "maximum_forward_backward_native_px": maximum_fb,
                          "tracking_lost_or_occluded_frames": len(rows)-len(valid),
                          "passed": bool(passed)})
    return {"static_state_local_stability_passed": bool(all(p["passed"] for p in per_point)),
            "elapsed_span_seconds": [frames[0]["elapsed_seconds"], frames[-1]["elapsed_seconds"]],
            "sampled_frames": len(frames), "per_point": per_point,
            "scope": "Local 2D landmark stability only; no absolute or cross-camera clock alignment"}


def track_frames(frames, initial_points, point_names, roi_xyxy):
    """Track (native grayscale image, metadata) generator using forward/backward LK.

    A failed track is not reacquired from background. Occlusion is a possible
    reason for loss, not a ground-truth diagnosis inferred from optical flow.
    """
    import cv2
    cv2.setNumThreads(4)
    initial = np.asarray(initial_points, dtype=np.float32)
    current = initial[:, None, :].copy()
    active = np.ones(len(initial), dtype=bool)
    previous, trace, previous_time = None, [], None
    roi = np.asarray(roi_xyxy, dtype=float)
    parameters = {"winSize": (21, 21), "maxLevel": 3,
                  "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}
    for image, metadata in frames:
        elapsed = float(metadata["elapsed_seconds"])
        if previous_time is not None and elapsed <= previous_time:
            raise ValueError("Strictly increasing decoded timestamps required")
        if image.shape != (1080, 1920) or image.dtype != np.uint8:
            raise ValueError("Native 1920x1080 uint8 grayscale frames required")
        previous_time = elapsed
        if previous is None:
            if elapsed != 0:
                raise ValueError("Tracking reference must be first decoded source frame")
            rows = [{"point_name": n, "xy": p.tolist(), "valid": True, "forward_backward_px": 0.,
                     "displacement_px": 0., "status": "manual_reference"} for n, p in zip(point_names, initial)]
        else:
            old = current.copy()
            forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(previous, image, old, None, **parameters)
            if forward is None:
                forward = old.copy()
                forward_status = np.zeros((len(old), 1), dtype=np.uint8)
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(image, previous, forward, None, **parameters)
            if backward is None:
                backward = old.copy()
                backward_status = np.zeros((len(old), 1), dtype=np.uint8)
            fb = np.linalg.norm(backward[:, 0]-old[:, 0], axis=1)
            displacement = np.linalg.norm(forward[:, 0]-initial, axis=1)
            in_image = ((forward[:, 0, 0] >= 0) & (forward[:, 0, 0] < 1920)
                        & (forward[:, 0, 1] >= 0) & (forward[:, 0, 1] < 1080))
            # ROI is only a selection safeguard, not an object mask or GT boundary.
            in_roi = np.all((forward[:, 0] >= roi[:2]) & (forward[:, 0] <= roi[2:]), axis=1)
            finite = np.isfinite(forward[:, 0]).all(axis=1) & np.isfinite(fb) & np.isfinite(displacement)
            old_active = active.copy()
            active &= (forward_status.ravel() != 0) & (backward_status.ravel() != 0) & finite & in_image & in_roi & (fb <= MAX_FORWARD_BACKWARD_PX)
            rows = []
            for index, name in enumerate(point_names):
                measured = bool(old_active[index] and finite[index])
                rows.append({"point_name": name, "xy": forward[index, 0].tolist() if measured else None,
                             "valid": bool(active[index]), "forward_backward_px": float(fb[index]) if measured else None,
                             "displacement_px": float(displacement[index]) if measured else None,
                             "status": "tracked" if active[index] else "tracking_lost_possible_occlusion_or_low_texture",
                             "forward_lk_status": bool(forward_status[index, 0]),
                             "backward_lk_status": bool(backward_status[index, 0])})
            current[active] = forward[active]
            # Failed tracks retain their last coordinates but remain inactive.
        trace.append({**metadata, "points": rows})
        previous = image
    return {"thresholds": thresholds(), "claim_first_second": summarize_trace(trace, point_names),
            "full_window": summarize_trace(trace, point_names, end_seconds=3.-1/60), "trace": trace,
            "temporal_sync_verified": False, "segmentation_ready": False}


def _stream_frames(stream, reference_frame, *, seconds=3.):
    import av
    import cv2
    from PIL import Image
    reference_path = Path(reference_frame["path"])
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != reference_frame["sha256"]:
        raise ValueError("Reference PNG SHA-256 mismatch")
    with Image.open(reference_path) as stored:
        reference_rgb = np.asarray(stored.convert("RGB"))
    first_time, previous_time = None, None
    with av.open(stream, mode="r", format="mp4", buffer_size=32768) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Exactly one original video stream required")
        video = container.streams.video[0]
        for ordinal, frame in enumerate(container.decode(video)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Native frame PTS required")
            now = float(frame.pts*frame.time_base)
            if previous_time is not None and now <= previous_time:
                raise ValueError("Nonmonotonic native video PTS")
            previous_time = now
            if first_time is None:
                first_time = now
            elapsed = now-first_time
            if elapsed >= seconds:
                break
            if ordinal > 360:
                raise ValueError("Unexpected source frame density")
            rgb = np.asarray(frame.to_image().convert("RGB"))
            if rgb.shape != (1080, 1920, 3):
                raise ValueError("Native video dimensions differ from calibration")
            if ordinal == 0:
                if frame.pts != reference_frame["pts"] or now != reference_frame["pts_seconds"] or not np.array_equal(rgb, reference_rgb):
                    raise ValueError("Decoded initial reference differs from manually annotated survey frame")
            yield cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), {"decoded_frame_ordinal_from_stream_start": ordinal,
                "pts": int(frame.pts), "pts_seconds": now, "elapsed_seconds": elapsed,
                "time_base_numerator": frame.time_base.numerator, "time_base_denominator": frame.time_base.denominator}


def run(survey_path, landmarks_path, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU Slurm execution required")
    root, output = Path("/projects/jointposeaction").resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(root/"experiments/segmentation_pilot"):
        raise ValueError("New dedicated pilot output directory required")
    quota = check_project_quota(root, MAX_OUTPUT_BYTES)
    survey = json.loads(Path(survey_path).read_text())
    annotations = json.loads(Path(landmarks_path).read_text())
    if survey.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Survey dataset revision changed")
    network_budget, output_budget = ByteBudget(MAX_NETWORK_BYTES), ByteBudget(MAX_OUTPUT_BYTES)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "dataset_revision": DATASET_REVISION, "thresholds": thresholds(), "quota_before": quota,
              "landmarks_sha256": hashlib.sha256(Path(landmarks_path).read_bytes()).hexdigest(),
              "survey_sha256": hashlib.sha256(Path(survey_path).read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "job_id": os.environ["SLURM_JOB_ID"], "recordings": [],
              "temporal_sync_verified": False, "segmentation_ready": False,
              "limitations": ["Local object landmark stability is not proof of cross-camera correspondence or physical synchronization.",
                              "Untracked/occluded/moving parts may invalidate whole-object rigidity despite stable selected landmarks.",
                              "Cross-view object identity, configuration and camera validity require separate evidence.",
                              "Optical flow is a descriptive measurement, not independent segmentation ground truth.",
                              "Acceptance applies only to local first 0..1-second intervals; full 3-second traces remain available."]}
    import httpx
    from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
    token = get_token()
    try:
        with httpx.Client(timeout=httpx.Timeout(45., read=60.)) as client:
            for annotation in annotations["recordings"]:
                recording_id = annotation["recording_id"]
                source = next(r for r in survey["recordings"] if r["recording_id"] == recording_id)
                identity = annotation.get("identity_evidence", {})
                if identity.get("manually_verified") is not True or not identity.get("source") or not identity.get("description") or not annotation.get("object_id"):
                    raise ValueError("Explicit manual physical object identity evidence required")
                camera_ids = [v["camera_id"] for v in annotation["views"]]
                if len(set(camera_ids)) != len(camera_ids) or not camera_ids:
                    raise ValueError("Nonempty unique manually assessed views required")
                row = {"recording_id": recording_id, "object_id": annotation["object_id"],
                       "identity_evidence": identity, "views": [], "unassessed_cameras": sorted(set(CAMERA_IDS)-set(camera_ids)),
                       "segmentation_ready": False, "temporal_sync_verified": False}
                report["recordings"].append(row)
                for spec in annotation["views"]:
                    camera_id, before = spec["camera_id"], network_budget.used
                    result = {"camera_id": camera_id, "status": "failed", "static_state_local_stability_passed": False}
                    row["views"].append(result)
                    try:
                        points, roi, span = validate_landmark_spec(spec)
                        view = next(v for v in source["views"] if v["camera_id"] == camera_id)
                        reference = next(f for f in view["frames"] if f["requested_elapsed_seconds"] == 0.)
                        if reference["sha256"] != spec["reference_frame_sha256"]:
                            raise ValueError("Manual landmark reference does not match initial survey frame")
                        path = f"recordings/{recording_id}/{camera_id}_rgb.mp4"
                        if view["source_path"] != path:
                            raise ValueError("Survey video camera mapping changed")
                        meta = get_hf_file_metadata(hf_hub_url("cvml-nus/assembly101", path, repo_type="dataset", revision=DATASET_REVISION), token=token)
                        if meta.commit_hash != DATASET_REVISION or meta.size != view["source_bytes"] or meta.etag != view["source_etag"]:
                            raise ValueError("Immutable source video metadata changed")
                        result.update(source_path=path, source_etag=meta.etag, source_bytes=meta.size,
                                      source_revision=meta.commit_hash)
                        stream = HTTPRangeIO(meta.size, http_fetcher(client, meta.location, meta.size, token, budget=network_budget), network_budget)
                        diagnostics = track_frames(_stream_frames(stream, reference), points, spec["point_names"], roi)
                        trace_file = output/recording_id/f"{camera_id}_landmark_trace.json"
                        saved = save_bytes(trace_file, (json.dumps(diagnostics, indent=2)+"\n").encode(), output_budget)
                        result.update({"status": "measured", "reference_frame_sha256": reference["sha256"],
                            "reference_pts": reference["pts"], "reference_pts_seconds": reference["pts_seconds"],
                            "roi_xyxy": roi.tolist(), "initial_points_xy": points.tolist(), "point_names": spec["point_names"],
                            "roi_landmark_span_fraction": span.tolist(), "claim_first_second": diagnostics["claim_first_second"],
                            "full_window": diagnostics["full_window"], "trace": saved,
                            "static_state_local_stability_passed": diagnostics["claim_first_second"]["static_state_local_stability_passed"]})
                    except Exception as error:
                        result["error"] = {"type": type(error).__name__, "reason": "Strict bounded landmark stability check failed; thresholds unchanged"}
                    result["network_payload_bytes"] = network_budget.used-before
                    print(json.dumps({"recording_id": recording_id, "camera_id": camera_id, "status": result["status"],
                                      "local_first_second_passed": result["static_state_local_stability_passed"]}), flush=True)
    finally:
        report.update(network_payload_bytes=network_budget.used, output_bytes_before_manifest=output_budget.used)
        save_bytes(output/"stability.json", (json.dumps(report, indent=2)+"\n").encode(), output_budget)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--landmarks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.survey, args.landmarks, args.output)
    print(json.dumps({"report": str(Path(args.output)/"stability.json"), "recordings": len(report["recordings"]), "segmentation_ready": False}))


if __name__ == "__main__":
    main()
