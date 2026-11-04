"""Synthetic local-image stability checks; no real tracking claim."""
import copy
import importlib.util
import unittest

import numpy as np

from segmentation_pilot.stability import validate_landmark_spec, summarize_trace, track_frames, thresholds


class StabilityTests(unittest.TestCase):
    def spec(self):
        return {"camera_id": "C10095", "manual_object_landmarks_verified": True,
                "reference_frame_sha256": "a"*64, "roi_xyxy": [100, 100, 300, 300],
                "points_xy": [[130, 130], [270, 130], [130, 270], [270, 270]],
                "point_names": ["front_left", "front_right", "rear_left", "rear_right"]}

    def test_landmarks_need_explicit_identity_and_distribution(self):
        points, roi, span = validate_landmark_spec(self.spec())
        np.testing.assert_allclose(span, [.7, .7])
        for key, value in (("manual_object_landmarks_verified", False), ("camera_id", "unknown"),
                           ("points_xy", [[130, 130]]*4), ("point_names", ["same"]*4),
                           ("reference_frame_sha256", "missing")):
            spec = copy.deepcopy(self.spec())
            spec[key] = value
            with self.assertRaises(ValueError):
                validate_landmark_spec(spec)

    def trace(self, drift=0., fb=.1, valid=True):
        return [{"elapsed_seconds": i/60, "points": [{"valid": valid, "displacement_px": drift,
                                                      "forward_backward_px": fb} for _ in range(4)]}
                for i in range(181)]

    def test_fixed_thresholds_accept_small_motion_reject_large_motion(self):
        names = self.spec()["point_names"]
        self.assertTrue(summarize_trace(self.trace(1.5), names)["static_state_local_stability_passed"])
        self.assertFalse(summarize_trace(self.trace(2.01), names)["static_state_local_stability_passed"])
        self.assertFalse(summarize_trace(self.trace(fb=.751), names)["static_state_local_stability_passed"])
        self.assertEqual(thresholds()["maximum_displacement_native_px"], 2.)

    def test_tracking_loss_not_counted_as_stationarity(self):
        names = self.spec()["point_names"]
        report = summarize_trace(self.trace(valid=False), names)
        self.assertFalse(report["static_state_local_stability_passed"])
        self.assertIsNone(report["per_point"][0]["maximum_displacement_native_px"])

    def test_first_second_scope_separate_from_later_motion(self):
        trace = self.trace()
        for frame in trace:
            if frame["elapsed_seconds"] > 1:
                for point in frame["points"]:
                    point["displacement_px"] = 10.
        names = self.spec()["point_names"]
        self.assertTrue(summarize_trace(trace, names)["static_state_local_stability_passed"])
        self.assertFalse(summarize_trace(trace, names, end_seconds=3.)["static_state_local_stability_passed"])

    def test_unobserved_claim_span_fails(self):
        report = summarize_trace(self.trace()[:10], self.spec()["point_names"])
        self.assertFalse(report["static_state_local_stability_passed"])

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required in cluster project environment")
    def test_synthetic_translated_image_sequence(self):
        import cv2
        image = np.zeros((1080, 1920), dtype=np.uint8)
        points = np.array([[130, 130], [270, 130], [130, 270], [270, 270]])
        for x, y in points:
            cv2.rectangle(image, (x-6, y-6), (x+6, y+6), 255, -1)
        def frames(displacement):
            for i in range(181):
                shift = displacement*min(i/60, 1.)
                warped = cv2.warpAffine(image, np.float32([[1, 0, shift], [0, 1, 0]]), (1920, 1080))
                yield warped, {"elapsed_seconds": i/60}
        names = self.spec()["point_names"]
        passed = track_frames(frames(1.), points, names, [100, 100, 300, 300])
        failed = track_frames(frames(5.), points, names, [100, 100, 300, 300])
        self.assertTrue(passed["claim_first_second"]["static_state_local_stability_passed"])
        self.assertFalse(failed["claim_first_second"]["static_state_local_stability_passed"])
        self.assertFalse(passed["temporal_sync_verified"])


if __name__ == "__main__":
    unittest.main()
