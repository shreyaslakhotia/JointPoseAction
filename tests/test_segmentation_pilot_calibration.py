"""Synthetic camera and timing tests; no scientific claims about real data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from segmentation_pilot.calibration import (
    CAMERA_IDS, load_nimble_calibration, project_world_points,
    reprojection_diagnostics, audit_temporal_correspondence,
)


class CalibrationTests(unittest.TestCase):
    def fixture(self, root, mutate=None):
        rows = []
        for camera_id in CAMERA_IDS:
            camera = {"SerialNo": camera_id + "_rgb", "DistortionModel": "OpenCV",
                      "ImageSizeX": 1920, "ImageSizeY": 1080,
                      "fx": 1000., "fy": 1000., "cx": 960., "cy": 540.,
                      "ModelViewMatrix": np.eye(4).tolist(),
                      "k1": .1, "k2": .2, "k3": .3, "p1": .01, "p2": .02}
            rows.append({"Camera": camera, "MountingLocation": "Exocentric"})
        if mutate:
            mutate(rows)
        path = Path(root) / "recording.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_named_distortion_projection_and_unverified_status(self):
        with tempfile.TemporaryDirectory() as root:
            report = load_nimble_calibration(self.fixture(root), "recording")
            camera = report["cameras"][CAMERA_IDS[0]]
            self.assertEqual(camera["distortion"], [.1, .2, .01, .02, .3])
            projected, depth = project_world_points(camera, [[0, 0, 2], [1, 0, 2], [0, 0, -1]])
            np.testing.assert_allclose(projected[:2], [[960, 540], [1496.09375, 542.5]])
            self.assertTrue(np.isnan(projected[2]).all())
            np.testing.assert_equal(depth, [2, 2, -1])
            self.assertEqual(report["units_status"], "unverified")
            self.assertFalse(report["segmentation_ready"])

    def test_translation_is_world_to_camera(self):
        with tempfile.TemporaryDirectory() as root:
            def change(rows):
                rows[0]["Camera"]["ModelViewMatrix"][0][3] = 1
                for name in ("k1", "k2", "k3", "p1", "p2"):
                    rows[0]["Camera"][name] = 0
            camera = load_nimble_calibration(self.fixture(root, change), "recording")["cameras"][CAMERA_IDS[0]]
            projected, _ = project_world_points(camera, [[0, 0, 2]])
            np.testing.assert_allclose(projected, [[1460, 540]])

    def test_rejects_identity_revision_size_rotation_and_missing_camera(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture(root)
            for kwargs in ({"recording_id": "wrong"}, {"recording_id": "recording", "source_revision": "unreviewed"},
                           {"recording_id": "recording", "image_sizes": {c: [960, 540] for c in CAMERA_IDS}}):
                with self.assertRaises(ValueError):
                    load_nimble_calibration(path, **kwargs)
            for mutate in (lambda rows: rows.pop(), lambda rows: rows[0]["Camera"].update(SerialNo="C99999_rgb"),
                           lambda rows: rows[0]["Camera"]["ModelViewMatrix"][0].__setitem__(0, 2)):
                with self.assertRaises(ValueError):
                    load_nimble_calibration(self.fixture(root, mutate), "recording")

    def test_reprojection_requires_independent_observation(self):
        with tempfile.TemporaryDirectory() as root:
            camera = load_nimble_calibration(self.fixture(root), "recording")["cameras"][CAMERA_IDS[0]]
            xyz = [[0, 0, 1], [.1, 0, 1], [0, .1, 1]]
            uv, _ = project_world_points(camera, xyz)
            with self.assertRaises(ValueError):
                reprojection_diagnostics(camera, xyz, uv, independence_evidence={"independent": True, "source": "projected-labels"})
            result = reprojection_diagnostics(camera, xyz, uv + [3, 4], independence_evidence={
                "independent": True, "source": "synthetic fixture only", "derived_from_same_calibration": False})
            self.assertAlmostEqual(result["rmse_px"], 5)

    def temporal_fixture(self):
        views = {c: {"frame_indices": [0, 1, 2, 3], "timestamps_seconds": [.0, .1, .2, .3]} for c in CAMERA_IDS}
        rows = [{"event_id": str(i), "frame_index": i, "reference_seconds": i/10 + .02,
                 "role": "fit" if i < 2 else "held_out"} for i in range(4)]
        evidence = {"method": "synthetic shared events", "source": "unit fixture", "independent": True,
                    "shared_reference_id": "synthetic clock", "correspondences": {c: copy.deepcopy(rows) for c in CAMERA_IDS}}
        return views, evidence

    def test_equal_fps_indices_are_not_synchronization(self):
        views, _ = self.temporal_fixture()
        report = audit_temporal_correspondence(views)
        self.assertFalse(report["synchronization_verified"])
        self.assertFalse(report["segmentation_ready"])

    def test_measured_offset_requires_heldout_all_views(self):
        views, evidence = self.temporal_fixture()
        report = audit_temporal_correspondence(views, evidence)
        self.assertTrue(report["synchronization_verified"])
        self.assertAlmostEqual(report["per_camera"][CAMERA_IDS[0]]["source_to_reference_offset_seconds"], .02)
        evidence["correspondences"][CAMERA_IDS[0]][-1]["role"] = "fit"
        self.assertFalse(audit_temporal_correspondence(views, evidence)["segmentation_ready"])

    def test_detects_heldout_offset_failure(self):
        views, evidence = self.temporal_fixture()
        views[CAMERA_IDS[0]]["timestamps_seconds"][-1] += .05
        report = audit_temporal_correspondence(views, evidence)
        self.assertFalse(report["segmentation_ready"])
        self.assertAlmostEqual(report["per_camera"][CAMERA_IDS[0]]["held_out_max_abs_residual_seconds"], .05)


if __name__ == "__main__":
    unittest.main()
