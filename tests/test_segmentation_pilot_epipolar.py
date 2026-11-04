"""Synthetic algebra tests only; no real camera calibration is certified."""
import copy
import importlib.util
import unittest

import numpy as np

from segmentation_pilot.calibration import project_world_points
from segmentation_pilot.epipolar import fundamental_matrix, undistort_pixels, symmetric_epipolar_errors


class EpipolarTests(unittest.TestCase):
    def camera(self, translation=(0, 0, 0), distortion=(0, 0, 0, 0, 0)):
        transform = np.eye(4)
        transform[:3, 3] = translation
        return {"K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
                "world_to_camera": transform.tolist(), "projection_direction": "world_to_camera",
                "distortion": list(distortion), "distortion_cv2_order": ["k1", "k2", "p1", "p2", "k3"]}

    def test_synthetic_stereo_and_wrong_correspondences(self):
        first, second = self.camera(), self.camera((-1, 0, 0))
        xyz = np.array([[0, 0, 5], [.5, 1, 4], [1, -1, 8], [-.5, .5, 6]])
        points1, _ = project_world_points(first, xyz)
        points2, _ = project_world_points(second, xyz)
        matrix = fundamental_matrix(first, second)
        np.testing.assert_allclose(symmetric_epipolar_errors(matrix, points1, points2), 0, atol=1e-10)
        np.testing.assert_allclose(symmetric_epipolar_errors(matrix, points1, points2+[0, 7]), 7)
        self.assertEqual(np.linalg.matrix_rank(matrix), 2)
        np.testing.assert_allclose(symmetric_epipolar_errors(-matrix*5, points1, points2+[0, 7]), 7)

    def test_world_frame_rotation_and_relative_transform(self):
        first, second = self.camera((.2, .1, 0)), self.camera((-.8, .1, 0))
        angle = .2
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        second["world_to_camera"] = np.column_stack((rotation, [-.8, .1, 0])).tolist()+[[0, 0, 0, 1]]
        xyz = [[0, .2, 4], [.7, -.4, 7], [-.3, .9, 8]]
        points1, _ = project_world_points(first, xyz)
        points2, _ = project_world_points(second, xyz)
        np.testing.assert_allclose(symmetric_epipolar_errors(fundamental_matrix(first, second), points1, points2), 0, atol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required; execute in project cluster environment")
    def test_distortion_roundtrip_uses_native_k_and_named_order(self):
        camera = self.camera(distortion=(.1, -.03, .004, -.002, .01))
        plain = self.camera()
        xyz = [[0, 0, 5], [1, .5, 4], [-1, -.5, 8]]
        raw, _ = project_world_points(camera, xyz)
        target, _ = project_world_points(plain, xyz)
        np.testing.assert_allclose(undistort_pixels(camera, raw), target, atol=1e-4)
        broken = copy.deepcopy(camera)
        broken["distortion_cv2_order"] = ["k1", "k2", "k3", "p1", "p2"]
        with self.assertRaises(ValueError):
            undistort_pixels(broken, raw)

    def test_zero_baseline_and_unknown_direction_rejected(self):
        camera = self.camera()
        with self.assertRaises(ValueError):
            fundamental_matrix(camera, camera)
        second = self.camera((-1, 0, 0))
        second["projection_direction"] = "camera_to_world"
        with self.assertRaises(ValueError):
            fundamental_matrix(camera, second)

    def test_degenerate_line_and_nonfinite_points_rejected(self):
        with self.assertRaises(ValueError):
            symmetric_epipolar_errors(np.zeros((3, 3)), [[0, 0]], [[1, 1]])
        with self.assertRaises(ValueError):
            symmetric_epipolar_errors(np.eye(3), [[np.nan, 0]], [[1, 1]])


if __name__ == "__main__":
    unittest.main()
