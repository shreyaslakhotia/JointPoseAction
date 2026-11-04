"""Descriptive two-image epipolar checks; no reconstruction or sync claims.

Image matches are candidate reference correspondences, not ground truth. Fitted
RANSAC selection never uses the supplied calibration fundamental matrix. Image
undistortion uses the supplied intrinsic/distortion parameters, so this is not
an independent validation of those parameters. Planar scenes and repeated toys
can produce misleading matches. Every readiness gate remains closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from segmentation_pilot.calibration import CAMERA_IDS, load_nimble_calibration


def fundamental_matrix(camera1, camera2):
    """F maps an undistorted native-pixel point in view1 to a line in view2."""
    if camera1.get("projection_direction") != "world_to_camera" or camera2.get("projection_direction") != "world_to_camera":
        raise ValueError("Reviewed world-to-camera conventions required")
    first, second = (np.asarray(c["world_to_camera"], dtype=float) for c in (camera1, camera2))
    if first.shape != (4, 4) or second.shape != (4, 4) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Finite 4 x 4 camera transforms required")
    relative = second @ np.linalg.inv(first)
    rotation, translation = relative[:3, :3], relative[:3, 3]
    if np.linalg.norm(translation) < 1e-12:
        raise ValueError("Zero-baseline cameras have no nondegenerate fundamental matrix")
    tx, ty, tz = translation
    cross = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]], dtype=float)
    intrinsic1, intrinsic2 = (np.asarray(c["K"], dtype=float) for c in (camera1, camera2))
    if any(k.shape != (3, 3) or not np.isfinite(k).all() for k in (intrinsic1, intrinsic2)):
        raise ValueError("Finite 3 x 3 intrinsics required")
    matrix = np.linalg.inv(intrinsic2).T @ cross @ rotation @ np.linalg.inv(intrinsic1)
    norm = np.linalg.norm(matrix)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Degenerate calibration fundamental matrix")
    return matrix/norm


def undistort_pixels(camera, points):
    """Return native-K undistorted pixels using named official coefficients."""
    import cv2
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Finite N x 2 pixel points required")
    if not len(points):
        return points.copy()
    if camera.get("distortion_cv2_order") != ["k1", "k2", "p1", "p2", "k3"]:
        raise ValueError("Explicit named OpenCV coefficient mapping required")
    intrinsic = np.asarray(camera["K"], dtype=np.float64)
    distortion = np.asarray(camera["distortion"], dtype=np.float64)
    if intrinsic.shape != (3, 3) or distortion.shape != (5,) or not np.isfinite(intrinsic).all() or not np.isfinite(distortion).all():
        raise ValueError("Finite intrinsic and five named distortion parameters required")
    return cv2.undistortPoints(points[:, None, :], intrinsic, distortion, P=intrinsic).reshape(-1, 2)


def symmetric_epipolar_errors(matrix, points1, points2):
    """Arithmetic mean of both point-to-corresponding-line distances, pixels."""
    matrix = np.asarray(matrix, dtype=float)
    first, second = np.asarray(points1, dtype=float), np.asarray(points2, dtype=float)
    if matrix.shape != (3, 3) or first.ndim != 2 or first.shape[1] != 2 or second.shape != first.shape:
        raise ValueError("Need 3 x 3 F and matching N x 2 point arrays")
    if not np.isfinite(matrix).all() or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Finite epipolar inputs required")
    first_h = np.column_stack((first, np.ones(len(first))))
    second_h = np.column_stack((second, np.ones(len(second))))
    lines2, lines1 = first_h @ matrix.T, second_h @ matrix
    numerator = np.abs(np.sum(second_h*lines2, axis=1))
    lengths1, lengths2 = np.linalg.norm(lines1[:, :2], axis=1), np.linalg.norm(lines2[:, :2], axis=1)
    if np.any(lengths1 <= 1e-15) or np.any(lengths2 <= 1e-15):
        raise ValueError("Degenerate epipolar line")
    return .5*numerator*(1/lengths1+1/lengths2)


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "median_px": float(np.median(values)),
            "p95_px": float(np.percentile(values, 95)), "mean_px": float(values.mean()),
            "max_px": float(values.max())} if len(values) else {"count": 0}


def _native_image(frame):
    import cv2
    path = Path(frame["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
        raise ValueError("Survey image SHA-256 mismatch")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (1080, 1920):
        raise ValueError("Native original survey image required")
    return image


def match_pair(frame1, frame2, camera1, camera2, *, ratio=.75, max_features=4000, ransac_threshold_px=1.):
    """SIFT ratio/mutual matches, then image-fitted RANSAC independent of Fcal."""
    import cv2
    cv2.setRNGSeed(20261104)
    cv2.setNumThreads(4)
    image1, image2 = _native_image(frame1), _native_image(frame2)
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    result = {"source_image_sha256": [frame1["sha256"], frame2["sha256"]],
              "source_image_paths": [frame1["path"], frame2["path"]],
              "source_pts_seconds": [frame1["pts_seconds"], frame2["pts_seconds"]],
              "synchronization_verified": False, "segmentation_ready": False,
              "keypoint_counts": [len(keypoints1), len(keypoints2)],
              "descriptor": "SIFT native grayscale", "ratio_threshold": ratio,
              "ransac_threshold_px": ransac_threshold_px, "ransac_seed": 20261104,
              "calibration_F_used_to_select_matches": False,
              "residual_definition": "mean of two undistorted-native-pixel point-to-epipolar-line distances"}
    if descriptors1 is None or descriptors2 is None or min(len(descriptors1), len(descriptors2)) < 2:
        return {**result, "status": "insufficient_descriptors"}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratio_matches(first, second):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(first, second, k=2)
                if len(pair) == 2 and pair[0].distance < ratio*pair[1].distance}
    forward, reverse = ratio_matches(descriptors1, descriptors2), ratio_matches(descriptors2, descriptors1)
    pairs = [(a, b) for a, b in forward.items() if reverse.get(b) == a]
    result["mutual_ratio_match_count"] = len(pairs)
    if len(pairs) < 12:
        return {**result, "status": "insufficient_candidate_matches"}
    raw1 = np.array([keypoints1[a].pt for a, _ in pairs])
    raw2 = np.array([keypoints2[b].pt for _, b in pairs])
    points1, points2 = undistort_pixels(camera1, raw1), undistort_pixels(camera2, raw2)
    # This fit uses only matched image points. Never gate on calibration residuals.
    fitted, mask = cv2.findFundamentalMat(points1, points2, method=cv2.FM_RANSAC,
                                         ransacReprojThreshold=ransac_threshold_px,
                                         confidence=.999, maxIters=10000)
    if fitted is None or fitted.shape != (3, 3) or mask is None:
        return {**result, "status": "image_fundamental_fit_failed"}
    inliers = mask.ravel().astype(bool)
    result["image_fitted_ransac_inlier_count"] = int(inliers.sum())
    if inliers.sum() < 8:
        return {**result, "status": "insufficient_image_ransac_inliers"}
    calibration_f = fundamental_matrix(camera1, camera2)
    _, homography_mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_threshold_px)
    homography_count = 0 if homography_mask is None else int(homography_mask.sum())
    ratio_planar = homography_count/max(int(inliers.sum()), 1)
    result.update({"status": "descriptive_correspondences_available",
        "calibration_F": calibration_f.tolist(), "independently_image_fitted_F": fitted.tolist(),
        "calibration_epipolar_residuals_on_image_ransac_inliers": _summary(symmetric_epipolar_errors(calibration_f, points1[inliers], points2[inliers])),
        "image_fit_residuals_on_own_inliers_not_accuracy": _summary(symmetric_epipolar_errors(fitted, points1[inliers], points2[inliers])),
        "homography_inlier_count": homography_count,
        "homography_to_fundamental_inlier_count_ratio": ratio_planar,
        "potential_planar_or_repeated_texture_degeneracy": ratio_planar >= .8,
        "correspondences": [{"raw_pixel_1": a.tolist(), "raw_pixel_2": b.tolist(),
                              "undistorted_pixel_1": c.tolist(), "undistorted_pixel_2": d.tolist(),
                              "image_ransac_inlier": bool(keep)}
                             for a, b, c, d, keep in zip(raw1, raw2, points1, points2, inliers)],
        "ground_truth_correspondences": False,
        "independent_manual_correspondence_review": "pending",
        "calibration_accuracy_verified": False})
    return result


def survey_diagnostics(survey_path, output):
    survey_path, output = Path(survey_path), Path(output)
    if output.exists():
        raise ValueError("Independent epipolar output must be a new file")
    survey = json.loads(survey_path.read_text())
    report = {"schema_version": 1, "method": "known-calibration versus independently image-fitted epipolar lines",
              "survey_path": str(survey_path.resolve()), "survey_sha256": hashlib.sha256(survey_path.read_bytes()).hexdigest(),
              "reference_camera": "C10119", "sample_elapsed_seconds": 1.,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "synchronization_verified": False, "segmentation_ready": False, "recordings": [],
              "limitations": ["Candidate SIFT matches are not ground truth; manual correspondence review remains required.",
                              "RANSAC fit uses image matches, not calibration F, to select candidate inliers.",
                              "Point undistortion uses provided intrinsics and distortion, so their independent accuracy is not established.",
                              "Static-background matches cannot establish physical synchronization.",
                              "Planar scenes, repeated parts and moving hands can mislead image matching.",
                              "These are 2D-to-2D epipolar residuals, not 3D reprojection errors or reconstruction metrics."]}
    for recording in survey["recordings"]:
        row = {"recording_id": recording["recording_id"], "pairs": []}
        report["recordings"].append(row)
        calibration = load_nimble_calibration(recording["calibration_path"], recording["recording_id"])
        if calibration["sha256"] != recording["calibration_sha256"]:
            raise ValueError("Survey calibration hash changed")
        row.update(calibration_path=recording["calibration_path"], calibration_sha256=calibration["sha256"],
                   calibration_source_revision=calibration["source_revision"])
        frames = {v["camera_id"]: next((f for f in v.get("frames", []) if f["requested_elapsed_seconds"] == 1.), None)
                  for v in recording["views"]}
        for camera_id in CAMERA_IDS:
            if camera_id == "C10119":
                continue
            pair = {"camera_ids": ["C10119", camera_id]}
            row["pairs"].append(pair)
            if not frames.get("C10119") or not frames.get(camera_id):
                pair["status"] = "source_frame_unavailable"
                continue
            try:
                pair.update(match_pair(frames["C10119"], frames[camera_id],
                                       calibration["cameras"]["C10119"], calibration["cameras"][camera_id]))
            except Exception as error:
                pair.update(status="diagnostic_failed", error_type=type(error).__name__)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = survey_diagnostics(args.survey, args.output)
    print(json.dumps({"report": args.output, "recordings": len(report["recordings"]), "segmentation_ready": False}))


if __name__ == "__main__":
    main()
