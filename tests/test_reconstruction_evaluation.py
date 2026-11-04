"""Known synthetic geometry checks; these establish no real-data readiness."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from reconstruction.alignment import SimilarityTransform, fit_landmark_similarity, validate_alignment_views
from reconstruction.cameras import Camera
from reconstruction.evaluation import (Observation, ViewPartition, evaluate_silhouettes,
                                       point_projection_coverage, runtime_record, temporal_chamfer)
from reconstruction.figures import comparison_markdown, silhouette_overlay
from reconstruction.rendering import render_occupancy
from reconstruction.visual_hull import VoxelGrid


def camera(name="a"):
    return Camera(name, np.array([[8., 0, 4], [0, 8, 4], [0, 0, 1]]), np.eye(4), (9, 9),
                  coordinate_frame="synthetic_world", units="synthetic_length")


def grid():
    return VoxelGrid(np.array([-.5, -.5, 2.]), np.array([1., 1., 1.]), np.ones((1, 1, 1), dtype=bool),
                     coordinate_frame="synthetic_world", units="synthetic_length",
                     metadata={"retained_camera_ids": ["a"], "alignment_camera_ids": ["a"]})


class ReconstructionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.partition = ViewPartition(("a",), ("b", "c"), ("a", "b", "c"), ("a",))

    def observation(self, name, mask=None, **kwargs):
        mask = render_occupancy(grid(), camera(name)) if mask is None else mask
        return Observation(camera(name), mask, kwargs.pop("mask_kind", "predicted"), "a" * 64, **kwargs)

    def test_perfect_predicted_agreement_is_not_accuracy(self):
        report = evaluate_silhouettes(grid(), [self.observation("a"), self.observation("b")], self.partition)
        self.assertEqual(report["summaries"]["heldout_predicted"]["mean"]["iou"], 1.)
        self.assertFalse(report["per_camera"][1]["accuracy_established"])
        self.assertEqual(report["per_camera"][1]["measurement_kind"], "predicted_mask_agreement")
        self.assertEqual(report["missing_camera_ids"], ["c"])
        self.assertEqual(report["summaries"]["heldout_predicted"]["available_cameras"], 1)

    def test_known_disjoint_masks_and_no_missing_imputation(self):
        empty = np.zeros((9, 9), dtype=bool)
        report = evaluate_silhouettes(grid(), [self.observation("b", empty)], self.partition)
        self.assertEqual(report["summaries"]["heldout_predicted"]["mean"], {"iou": 0., "dice": 0., "boundary_f1": 0.})
        self.assertIsNone(report["summaries"]["input_predicted"]["mean"]["iou"])

    def test_independent_reference_requires_provenance(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            self.observation("b", mask_kind="independent_verified")
        p = {"verified_independent": True, "derived_from_prediction": False,
             "kind": "independent_manual_annotation", "source_id": "synthetic analytic cube",
             "reviewer": "test fixture", "evidence": "known synthetic mask"}
        observation = self.observation("b", mask_kind="independent_verified", provenance=p)
        report = evaluate_silhouettes(grid(), [observation], self.partition)
        self.assertTrue(report["per_camera"][1]["accuracy_established"])
        self.assertEqual(report["summaries"]["heldout_independent_verified"]["mean"]["iou"], 1.)
        self.assertIsNone(report["summaries"]["heldout_predicted"]["mean"]["iou"])

    def test_partitions_reject_alignment_and_reconstruction_leakage(self):
        for inputs, heldout, align in [(("a", "b"), ("b", "c"), ()), (("a",), ("b", "c"), ("b",))]:
            with self.assertRaises(ValueError):
                ViewPartition(inputs, heldout, ("a", "b", "c"), align)
        with self.assertRaises(ValueError):
            ViewPartition(("a",), ("b",), ("a", "b", "c"))
        with self.assertRaises(ValueError):
            validate_alignment_views(("c",), ("a",), ("b", "c"))

    def test_actual_construction_and_alignment_views_cannot_leak(self):
        for metadata in ({}, {"retained_camera_ids": ["b"], "alignment_camera_ids": ["a"]},
                         {"retained_camera_ids": ["a"], "alignment_camera_ids": ["b"]},
                         {"retained_camera_ids": ["a"]}):
            artifact = VoxelGrid(grid().origin_xyz, grid().voxel_size_xyz, grid().occupancy,
                                 grid().coordinate_frame, grid().units, metadata)
            with self.assertRaises(ValueError):
                evaluate_silhouettes(artifact, [self.observation("b")], self.partition)

    def test_mesh_hook_requires_real_surface_connectivity(self):
        surface = SimpleNamespace(vertices=np.array([[0., 0, 2], [1, 0, 2], [0, 1, 2]]),
                                  faces=np.array([[0, 1, 2]]), coordinate_frame=grid().coordinate_frame,
                                  units=grid().units, metadata=grid().metadata)
        report = evaluate_silhouettes(surface, [self.observation("b")], self.partition,
            representation_kind="mesh", renderer=lambda mesh, cam: render_occupancy(grid(), cam))
        self.assertEqual(report["summaries"]["heldout_predicted"]["mean"]["iou"], 1.)
        surface.faces = np.array([[0, 1, 9]])
        with self.assertRaisesRegex(ValueError, "triangle"):
            evaluate_silhouettes(surface, [self.observation("b")], self.partition,
                representation_kind="mesh", renderer=lambda mesh, cam: render_occupancy(grid(), cam))

    def test_invalid_observations_and_duplicate_views(self):
        with self.assertRaises(ValueError):
            Observation(camera(), np.zeros((8, 9)), "predicted", "a" * 64)
        with self.assertRaises(ValueError):
            Observation(camera(), np.zeros((9, 9)), "predicted", "not a hash")
        with self.assertRaises(ValueError):
            evaluate_silhouettes(grid(), [self.observation("a"), self.observation("a")], self.partition)

    def test_point_scatter_is_not_a_silhouette_renderer(self):
        for kind in ("point_cloud", "occupancy"):
            with self.assertRaises(ValueError):
                evaluate_silhouettes(np.array([[0., 0., 2.]]), [self.observation("a")], self.partition,
                                     representation_kind=kind)
        with self.assertRaisesRegex(ValueError, "explicit"):
            evaluate_silhouettes(grid(), [self.observation("a")], self.partition, representation_kind="mesh")

    def test_frames_and_renderer_resolution_are_checked(self):
        wrong = VoxelGrid(grid().origin_xyz, grid().voxel_size_xyz, grid().occupancy, "other", "synthetic_length", grid().metadata)
        with self.assertRaisesRegex(ValueError, "frames/units"):
            evaluate_silhouettes(wrong, [self.observation("a")], self.partition)
        with self.assertRaisesRegex(ValueError, "resolution"):
            evaluate_silhouettes(grid(), [self.observation("a")], self.partition,
                                 renderer=lambda volume, cam: np.zeros((2, 2), dtype=bool))

    def test_point_coverage_reports_only_visibility(self):
        report = point_projection_coverage([[0, 0, 3], [0, 0, -1], [100, 0, 2]], camera())
        self.assertEqual(report["points_inside_image_in_front"], 1)
        self.assertAlmostEqual(report["fraction"], 1 / 3)
        self.assertIn("not_silhouette", report["measurement_kind"])

    def test_temporal_chamfer_known_translation_and_registration(self):
        kwargs = dict(first_frame="w", second_frame="w", first_units="m", second_units="m",
                      registration={"applied": True, "method": "identity_common_calibration", "evidence": "synthetic"})
        report = temporal_chamfer([[0, 0, 0]], [[0, 0, 2]], **kwargs)
        self.assertEqual(report["symmetric_mean_chamfer"], 2.)
        self.assertFalse(report["semantic_tracking_established"])
        same = temporal_chamfer([[0, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 0]], **kwargs)
        self.assertEqual(same["symmetric_mean_chamfer"], 0.)
        for changes in ({"second_units": "mm"}, {"second_frame": "canonical"}, {"registration": {}}, {"max_pair_evaluations": 0}):
            with self.assertRaises(ValueError):
                temporal_chamfer([[0, 0, 0]], [[0, 0, 1]], **(kwargs | changes))

    def test_resource_records_do_not_invent_missing_memory(self):
        record = runtime_record(status="blocked", elapsed_seconds=0, peak_memory_bytes=None,
                                output_bytes=0, failure_reason="calibration review missing")
        self.assertIsNone(record["peak_memory_bytes"])
        with self.assertRaises(ValueError):
            runtime_record(status="failed", elapsed_seconds=1, peak_memory_bytes=None, output_bytes=0)

    def test_figure_overlay_and_table_from_supplied_data(self):
        from PIL import Image
        report = evaluate_silhouettes(grid(), [self.observation("b")], self.partition)
        table = comparison_markdown({"synthetic": report})
        self.assertIn("prediction agreement", table)
        self.assertIn("unavailable", table)
        with tempfile.TemporaryDirectory() as tmp:
            path = silhouette_overlay(np.zeros((9, 9, 3), dtype=np.uint8), self.observation("b").mask,
                                      self.observation("b").mask, Path(tmp) / "overlay.png")
            with Image.open(path) as image:
                self.assertEqual(image.size, (9, 9))


class SimilarityAlignmentTests(unittest.TestCase):
    def test_recovers_known_positive_scale_rotation_translation(self):
        source = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        target = 2.5 * source @ rotation.T + [3., -2, 5]
        transform, report = fit_landmark_similarity(source, target, source_frame="canonical", target_frame="world",
            source_units="dimensionless", target_units="mm", evidence={"independent_paired_landmarks": True,
            "source_id": "synthetic", "reviewer": "test", "correspondence_evidence": "known paired vertices"})
        np.testing.assert_allclose(transform.apply(source), target, atol=1e-12)
        self.assertAlmostEqual(transform.scale, 2.5)
        self.assertAlmostEqual(np.linalg.det(transform.rotation), 1.)
        self.assertLess(report["fit_rmse"], 1e-12)
        self.assertFalse(report["accuracy_established"])

    def test_rejects_arbitrary_normalization_and_invalid_transform(self):
        with self.assertRaises(ValueError):
            fit_landmark_similarity(np.eye(3), np.eye(3), source_frame="a", target_frame="b",
                                    source_units="unknown", target_units="unknown", evidence={})
        for scale, rotation in [(-1, np.eye(3)), (1, np.diag([-1, 1, 1])), (1, np.eye(3) * 2)]:
            with self.assertRaises(ValueError):
                SimilarityTransform(scale, rotation, np.zeros(3), "a", "b", "m", "m", {"method": "test"})

    def test_rejects_collinear_landmarks(self):
        x = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaisesRegex(ValueError, "Collinear"):
            fit_landmark_similarity(x, x, source_frame="a", target_frame="b", source_units="m", target_units="m",
                evidence={"independent_paired_landmarks": True, "source_id": "synthetic", "reviewer": "test",
                          "correspondence_evidence": "collinear fixture"})


if __name__ == "__main__":
    unittest.main()
