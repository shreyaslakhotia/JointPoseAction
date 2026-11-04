"""CPU-only numerical/provenance tests; no SAM3 import, weights or GPU required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from segmentation_pilot.metrics import (
    binary_mask, boundary_f1, describe_mask, dice_coefficient, evaluate_mask,
    mask_boundary, mask_iou,
)
from segmentation_pilot import sam3_runner as runner


class MaskMetricsTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((9, 13), dtype=bool)
        self.mask[2:7, 3:10] = True

    def test_exact_and_disjoint(self):
        for metric in (mask_iou, dice_coefficient, boundary_f1):
            self.assertEqual(metric(self.mask, self.mask), 1.0)
            self.assertEqual(metric(self.mask, np.zeros_like(self.mask)), 0.0)

    def test_empty_convention(self):
        empty = np.zeros((3, 5), dtype=bool)
        for metric in (mask_iou, dice_coefficient, boundary_f1):
            self.assertEqual(metric(empty, empty), 1.0)

    def test_known_partial_overlap(self):
        left, right = np.zeros((3, 6), bool), np.zeros((3, 6), bool)
        left[:, :4], right[:, 2:] = True, True
        self.assertAlmostEqual(mask_iou(left, right), 1 / 3)
        self.assertAlmostEqual(dice_coefficient(left, right), 0.5)

    def test_binary_formats_and_invalid_probabilities(self):
        np.testing.assert_array_equal(binary_mask(self.mask.astype(np.uint8) * 255), self.mask)
        for value in (np.array([[0, 1, 255]]), np.array([[0.1, 0.9]]),
                      np.array([[np.nan]]), np.zeros((1, 2, 3)), np.empty((0, 2)),
                      np.array([["0", "1"]])):
            with self.assertRaises(ValueError):
                binary_mask(value)

    def test_no_implicit_resize(self):
        with self.assertRaises(ValueError):
            mask_iou(self.mask, np.zeros((13, 9), bool))

    def test_boundary_shift_and_explicit_tolerance(self):
        shifted = np.roll(self.mask, 1, axis=1)
        self.assertLess(boundary_f1(self.mask, shifted, 0), 1)
        self.assertEqual(boundary_f1(self.mask, shifted, 1), 1)
        with self.assertRaises(ValueError):
            boundary_f1(self.mask, shifted, -1)
        with self.assertRaises(ValueError):
            boundary_f1(self.mask, shifted, True)

    def test_full_image_boundary_does_not_disappear(self):
        boundary = mask_boundary(np.ones((4, 6), bool))
        self.assertEqual(int(boundary.sum()), 16)
        self.assertFalse(boundary[1:3, 1:5].any())

    def test_descriptive_measures_are_not_accuracy(self):
        record = evaluate_mask(self.mask, confidence=0.9)
        self.assertFalse(record["accuracy_established"])
        self.assertIsNone(record["iou"])
        self.assertIsNone(record["dice"])
        self.assertIsNone(record["boundary_f1"])
        self.assertEqual(record["foreground_pixels"], 35)
        self.assertAlmostEqual(record["image_coverage_fraction"], 35 / 117)
        self.assertEqual(record["mask_bbox_xyxy_exclusive"], [3, 2, 10, 7])

    def test_self_reference_cannot_be_reported_without_independent_evidence(self):
        with self.assertRaises(ValueError):
            evaluate_mask(self.mask, reference=self.mask)
        provenance = {"kind": "sam3_prediction", "verified_independent": True,
                      "derived_from_prediction": True, "source_id": "a", "reviewer": "a", "evidence": "a"}
        with self.assertRaises(ValueError):
            evaluate_mask(self.mask, reference=self.mask, reference_provenance=provenance)

    def test_explicit_independent_reference(self):
        provenance = {"kind": "independent_manual_annotation", "verified_independent": True,
                      "derived_from_prediction": False, "source_id": "human-native-mask-01",
                      "reviewer": "independent-reviewer", "evidence": "Drawn from original RGB, blind to predictions"}
        record = evaluate_mask(self.mask, reference=self.mask.copy(), reference_provenance=provenance)
        self.assertTrue(record["accuracy_established"])
        self.assertEqual(record["iou"], 1)
        self.assertEqual(record["boundary_tolerance_px"], 2)

    def test_invalid_confidence(self):
        for value in (float("nan"), -0.01, 1.01):
            with self.assertRaises(ValueError):
                describe_mask(self.mask, value)


class NativeMaskTests(unittest.TestCase):
    def test_non_square_box_conversion_and_bounds(self):
        self.assertEqual(runner.box_to_normalized_cxcywh([20, 10, 100, 50], 200, 100), [0.3, 0.3, 0.4, 0.4])
        for box in ([-1, 0, 1, 1], [0, 0, 201, 100], [2, 2, 1, 3],
                    [0, 0, float("nan"), 2], [0, 0, 2]):
            with self.assertRaises(ValueError):
                runner.box_to_normalized_cxcywh(box, 200, 100)

    def test_binary_roundtrip_preserves_holes_and_independent_instances(self):
        masks = np.zeros((2, 1, 7, 11), bool)
        masks[0, 0, 1:6, 1:8] = True
        masks[0, 0, 3, 3] = False
        masks[1, 0, 5, 10] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.npz"
            np.savez_compressed(path, **runner.pack_native_masks(masks, 7, 11))
            restored = runner.unpack_native_masks(path)
        np.testing.assert_array_equal(restored, masks[:, 0])
        self.assertFalse(restored[0, 3, 3])
        self.assertEqual(int(restored[1].sum()), 1)

    def test_zero_candidates_remains_empty_not_fake_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = runner.save_stage(output, "text_only", np.empty((0, 1, 5, 9), bool),
                                       np.empty((0, 4)), np.empty((0,)), height=5, width=9, prompt="toy")
            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(runner.unpack_native_masks(output / "text_only/native_masks.npz").shape, (0, 5, 9))

    def test_original_and_refinement_are_separate_and_immutable(self):
        masks = np.zeros((2, 1, 5, 9), bool)
        masks[0, 0, 1, 1], masks[1, 0, 3, 7] = True, True
        boxes = [[1, 1, 2, 2], [7, 3, 8, 4]]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original = runner.save_stage(output, "text_only", masks, boxes, [0.9, 0.2], height=5, width=9, prompt="toy")
            checksum = runner.file_sha256(output / "text_only/native_masks.npz")
            masks[0] = False
            runner.save_stage(output, "refined", masks, boxes, [0.8, 0.1], height=5, width=9, prompt="toy",
                              refinement={"box_xyxy": [0, 0, 3, 3], "intervention": "human", "reason": "Exclude another instance"})
            self.assertEqual(checksum, runner.file_sha256(output / "text_only/native_masks.npz"))
            self.assertFalse(original["instance_union_performed"])
            self.assertIsNone(original["selected_candidate_id"])
            self.assertEqual(original["candidate_count"], 2)
            self.assertTrue(original["candidates"][0]["above_evaluation_threshold"])
            self.assertFalse(original["candidates"][1]["above_evaluation_threshold"])
            with self.assertRaises(FileExistsError):
                runner.save_stage(output, "text_only", masks, boxes, [0.8, 0.1], height=5, width=9, prompt="toy")

    def test_native_probability_semantics(self):
        probabilities = np.array([[[[0.1, 0.8], [0.5, 0.51]]]], dtype=np.float32)
        masks = probabilities > 0.5
        with tempfile.TemporaryDirectory() as directory:
            record = runner.save_stage(Path(directory), "text_only", masks, [[0, 0, 2, 2]], [0.8],
                                       height=2, width=2, prompt="toy", probabilities=probabilities)
            self.assertEqual(record["native_probabilities"]["semantics"], "sigmoid_probability_not_logits")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                runner.save_stage(Path(directory), "text_only", ~masks, [[0, 0, 2, 2]], [0.8],
                                  height=2, width=2, prompt="toy", probabilities=probabilities)

    def test_candidate_alignment_and_native_binary_required(self):
        with self.assertRaises(ValueError):
            runner.pack_native_masks(np.ones((1, 3, 4), dtype=np.uint8), 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                runner.save_stage(Path(directory), "text_only", np.zeros((1, 3, 4), bool),
                                  np.empty((0, 4)), [0.5], height=3, width=4, prompt="toy")


class InferenceGateTests(unittest.TestCase):
    def fixture(self, root):
        (root / "frame.png").write_bytes(b"synthetic placeholder; not decoded by manifest validation")
        gate = {"recordings": {"recording-a": {"calibration_verified": True,
                 "temporal_correspondence_verified": True, "camera_ids": ["C10119"],
                 "evidence": "Synthetic test gate, not dataset validation"}}}
        manifest = {"calibration_temporal_gate_passed": True, "gate_evidence_path": "gate.json",
                    "images": [{"image_id": "a-C10119", "path": "frame.png", "object_id": "physical-toy-a",
                                "recording_id": "recording-a", "camera_id": "C10119", "prompt": "toy"}]}
        self.write(root, "gate.json", gate)
        self.write(root, "images.json", manifest)
        return manifest, gate

    @staticmethod
    def write(root, name, value):
        (root / name).write_text(json.dumps(value), encoding="utf-8")

    def test_recording_camera_gate_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            _, rows, gate = runner.validate_image_manifest(root / "images.json")
            self.assertEqual(rows[0]["sha256"], runner.file_sha256(root / "frame.png"))
            self.assertEqual(gate["sha256"], runner.file_sha256(root / "gate.json"))

    def test_pending_calibration_or_temporal_gate_blocks(self):
        for field in ("calibration_verified", "temporal_correspondence_verified"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, gate = self.fixture(root)
                gate["recordings"]["recording-a"][field] = False
                self.write(root, "gate.json", gate)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_other_camera_or_recording_cannot_inherit_gate(self):
        for field, value in (("camera_id", "C99999"), ("recording_id", "recording-b")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _ = self.fixture(root)
                manifest["images"][0][field] = value
                self.write(root, "images.json", manifest)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_top_gate_must_be_boolean_true(self):
        for value in (False, "true", 1):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _ = self.fixture(root)
                manifest["calibration_temporal_gate_passed"] = value
                self.write(root, "images.json", manifest)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_duplicate_ids_and_action_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root)
            manifest["images"].append(dict(manifest["images"][0]))
            self.write(root, "images.json", manifest)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json")
            manifest["images"] = manifest["images"][:1]
            manifest["images"][0]["action_id"] = 17
            self.write(root, "images.json", manifest)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json")

    def test_input_tampering_and_refinement_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root)
            manifest["images"][0]["sha256"] = "0" * 64
            self.write(root, "images.json", manifest)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json")
            del manifest["images"][0]["sha256"]
            manifest["images"][0]["refinement"] = {"box_xyxy": [0, 0, 5, 5], "reason": "target box"}
            self.write(root, "images.json", manifest)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json")
            manifest["images"][0]["refinement"]["intervention"] = "agent_visual_review"
            self.write(root, "images.json", manifest)
            runner.validate_image_manifest(root / "images.json")

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.json"
            path.write_text('{"passed":false,"passed":true}')
            with self.assertRaises(ValueError):
                runner.read_json(path)

    def test_checkpoint_hash_and_source_revision_are_checked_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"small-synthetic-checkpoint")
            for name in ("sam3/model_builder.py", "sam3/model/sam3_image_processor.py", "pyproject.toml"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic")
            with mock.patch.object(runner.subprocess, "check_output", side_effect=[runner.SOURCE_REVISION + "\n", ""]), \
                 mock.patch.object(runner.subprocess, "run"), \
                 mock.patch.object(runner, "CHECKPOINT_BYTES", checkpoint.stat().st_size), \
                 mock.patch.object(runner, "CHECKPOINT_SHA256", runner.file_sha256(checkpoint)):
                record = runner.verify_model_inputs(root, checkpoint)
                self.assertEqual(record["source_revision"], runner.SOURCE_REVISION)
            with mock.patch.object(runner.subprocess, "check_output", return_value="wrong-revision"):
                with self.assertRaises(ValueError):
                    runner.verify_model_inputs(root, checkpoint)
            with mock.patch.object(runner.subprocess, "check_output", side_effect=[runner.SOURCE_REVISION, ""]), \
                 mock.patch.object(runner.subprocess, "run"), \
                 mock.patch.object(runner, "CHECKPOINT_BYTES", checkpoint.stat().st_size):
                with self.assertRaises(ValueError):
                    runner.verify_model_inputs(root, checkpoint)

    def test_no_sam3_import_for_cpu_helpers(self):
        self.assertNotIn("sam3.model.sam3_image_processor", sys.modules)


class StaticStateGateTests(unittest.TestCase):
    fixture = InferenceGateTests.fixture
    write = staticmethod(InferenceGateTests.write)

    def static_fixture(self, root):
        manifest, gate = self.fixture(root)
        manifest.pop("calibration_temporal_gate_passed")
        manifest.update(calibration_static_gate_passed=True, correspondence_scope=runner.STATIC_SCOPE)
        row = manifest["images"][0]
        row.update(sha256=runner.file_sha256(root / "frame.png"), source_pts_seconds=0.0)
        point_names = ["corner-a", "corner-b", "corner-c", "corner-d"]
        point_values = [{"point_name": name, "passed": True, "valid_fraction": 1.0,
                         "valid_frames": 61, "total_frames": 61,
                         "maximum_displacement_native_px": 1.0,
                         "maximum_forward_backward_native_px": .2} for name in point_names]
        stability = {"thresholds": dict(runner.STATIC_THRESHOLDS), "temporal_sync_verified": False,
                     "recordings": [{"recording_id": row["recording_id"], "object_id": row["object_id"],
                                     "temporal_sync_verified": False,
                                     "identity_evidence": {"manually_verified": True, "source": "synthetic test reference",
                                                           "description": "Named corners on a synthetic rigid object"},
                                     "views": [{"camera_id": row["camera_id"], "status": "measured",
                                                "static_state_local_stability_passed": True,
                                                "reference_frame_sha256": row["sha256"], "reference_pts_seconds": 0.0,
                                                "point_names": point_names, "roi_landmark_span_fraction": [.7, .6],
                                                "claim_first_second": {"static_state_local_stability_passed": True,
                                                                       "elapsed_span_seconds": [0.0, 1.0],
                                                                       "sampled_frames": 61, "per_point": point_values}}]}]}
        calibration = {"recordings": [{"recording_id": row["recording_id"],
                                       "method": "synthetic independently reviewed calibration evidence"}]}
        record = gate["recordings"][row["recording_id"]]
        record.pop("temporal_correspondence_verified")
        record.update(static_object_state_correspondence_verified=True, temporal_sync_verified=False,
                      manual_identity_configuration_verified=True, object_ids=[row["object_id"]],
                      static_stability_evidence_path="stability.json", calibration_evidence_path="calibration.json")
        self.write(root, "calibration.json", calibration)
        self.write(root, "stability.json", stability)
        record.update(static_stability_evidence_sha256=runner.file_sha256(root / "stability.json"),
                      calibration_evidence_sha256=runner.file_sha256(root / "calibration.json"))
        self.write(root, "gate.json", gate)
        self.write(root, "images.json", manifest)
        return manifest, gate, stability, calibration

    def rewrite_stability(self, root, gate, stability):
        self.write(root, "stability.json", stability)
        gate["recordings"]["recording-a"]["static_stability_evidence_sha256"] = runner.file_sha256(root / "stability.json")
        self.write(root, "gate.json", gate)

    def test_measured_static_branch_is_explicitly_not_synchronized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.static_fixture(root)
            _, rows, evidence = runner.validate_image_manifest(root / "images.json")
            self.assertEqual(evidence["correspondence_scope"], "static_object_state_only")
            self.assertIs(evidence["temporal_sync_verified"], False)
            self.assertIs(rows[0]["temporal_sync_verified"], False)
            self.assertTrue(evidence["static_image_evidence"][rows[0]["image_id"]]["local_stability_measurements_validated"])
            self.assertEqual(len(evidence["static_image_evidence"]), 1)

    def test_static_branch_requires_its_own_scope_and_cannot_mix_gates(self):
        for field, value in (("correspondence_scope", "synchronized_frames"),
                             ("calibration_temporal_gate_passed", True)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _, _, _ = self.static_fixture(root)
                manifest[field] = value
                self.write(root, "images.json", manifest)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_missing_evidence_file_or_hash_blocks_static_branch(self):
        for stem in ("static_stability_evidence", "calibration_evidence"):
            with self.subTest(stem=stem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, gate, _, _ = self.static_fixture(root)
                gate["recordings"]["recording-a"][stem + "_path"] = "absent.json"
                self.write(root, "gate.json", gate)
                with self.assertRaises(FileNotFoundError):
                    runner.validate_image_manifest(root / "images.json")
            with self.subTest(stem=stem + "_sha256"), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, gate, _, _ = self.static_fixture(root)
                gate["recordings"]["recording-a"][stem + "_sha256"] = "0" * 64
                self.write(root, "gate.json", gate)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_failed_or_unmeasured_camera_cannot_pass_static_gate(self):
        for field, value in (("status", "failed"), ("static_state_local_stability_passed", False)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, gate, stability, _ = self.static_fixture(root)
                stability["recordings"][0]["views"][0][field] = value
                self.rewrite_stability(root, gate, stability)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_reference_hash_and_pts_must_match_exact_requested_image(self):
        for field, value in (("reference_frame_sha256", "f" * 64), ("reference_pts_seconds", 1.0)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, gate, stability, _ = self.static_fixture(root)
                stability["recordings"][0]["views"][0][field] = value
                self.rewrite_stability(root, gate, stability)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_static_images_require_hash_and_first_second_source_pts(self):
        for field, value in (("sha256", None), ("source_pts_seconds", 1.1),
                             ("source_pts_seconds", -0.1), ("source_pts_seconds", True)):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _, _, _ = self.static_fixture(root)
                manifest["images"][0][field] = value
                self.write(root, "images.json", manifest)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_stability_recording_object_and_gate_object_membership_must_match(self):
        for target in ("stability-object", "stability-recording", "gate-object"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, gate, stability, _ = self.static_fixture(root)
                if target == "stability-object":
                    stability["recordings"][0]["object_id"] = "another-object"
                elif target == "stability-recording":
                    stability["recordings"][0]["recording_id"] = "another-recording"
                else:
                    gate["recordings"]["recording-a"]["object_ids"] = ["another-object"]
                self.rewrite_stability(root, gate, stability)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_temporal_sync_claim_or_missing_manual_identity_blocks_static_branch(self):
        for field, value in (("temporal_sync_verified", True), ("temporal_sync_verified", None),
                             ("manual_identity_configuration_verified", False), ("calibration_verified", False)):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, gate, _, _ = self.static_fixture(root)
                gate["recordings"]["recording-a"][field] = value
                self.write(root, "gate.json", gate)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_predeclared_thresholds_cannot_be_relaxed_or_disguised_as_boolean(self):
        for field, value in (("maximum_displacement_native_px", 3.0),
                             ("minimum_valid_tracking_fraction", .9),
                             ("claim_elapsed_seconds", [False, 1]),
                             ("thresholds_predeclared_not_tuned_on_results", False)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, gate, stability, _ = self.static_fixture(root)
                stability["thresholds"][field] = value
                self.rewrite_stability(root, gate, stability)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_passing_boolean_cannot_override_failed_numerical_measurements(self):
        for field, value in (("maximum_displacement_native_px", 2.01),
                             ("maximum_forward_backward_native_px", .76),
                             ("valid_fraction", .9)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, gate, stability, _ = self.static_fixture(root)
                stability["recordings"][0]["views"][0]["claim_first_second"]["per_point"][0][field] = value
                self.rewrite_stability(root, gate, stability)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json")

    def test_wrong_recording_calibration_report_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, gate, _, calibration = self.static_fixture(root)
            calibration["recordings"][0]["recording_id"] = "wrong-recording"
            self.write(root, "calibration.json", calibration)
            gate["recordings"]["recording-a"]["calibration_evidence_sha256"] = runner.file_sha256(root / "calibration.json")
            self.write(root, "gate.json", gate)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json")

    def test_static_scope_is_saved_alongside_native_mask_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            record = runner.save_stage(Path(directory), "text_only", np.zeros((1, 2, 2), bool), [[0, 0, 2, 2]], [.8],
                                       height=2, width=2, prompt="toy", correspondence_scope=runner.STATIC_SCOPE,
                                       temporal_sync_verified=False)
            self.assertEqual(record["correspondence_scope"], "static_object_state_only")
            self.assertIs(record["temporal_sync_verified"], False)
            self.assertFalse(record["accuracy_established"])


class DiagnosticReviewTests(unittest.TestCase):
    write = staticmethod(InferenceGateTests.write)

    def fixture(self, root):
        from PIL import Image
        Image.new("RGB", (9, 5), (120, 80, 40)).save(root / "native.png")
        checksum = runner.file_sha256(root / "native.png")
        frame = {"path": "native.png", "sha256": checksum, "pts_seconds": 0.0,
                 "image_size": [9, 5], "synchronization_verified": False}
        survey = {"dataset_revision": runner.DATASET_REVISION, "cameras": sorted(runner.SURVEY_CAMERAS),
                  "segmentation_ready": False, "recordings": [{"recording_id": "recording-a", "views": [
                      {"camera_id": camera, "status": "decoded", "frames": [dict(frame)], "image_size": [9, 5],
                       "source_path": f"recordings/recording-a/{camera}_rgb.mp4", "source_revision": runner.DATASET_REVISION}
                      for camera in sorted(runner.SURVEY_CAMERAS)]}]}
        assessment = {"calibration_accuracy_verified": False, "temporal_sync_verified": False,
                      "multiview_dataset_ready": False,
                      "recordings": [{"recording_id": "recording-a", "calibration_accuracy_verified": False,
                                      "temporal_sync_verified": False, "multiview_dataset_ready": False,
                                      "assessment": "Synthetic schema/diagnostic evidence only; physical calibration unverified"}]}
        self.write(root, "survey.json", survey)
        self.write(root, "assessment.json", assessment)
        manifest = {"purpose": "diagnostic_2d_review", "multiview_dataset_ready": False,
                    "calibration_accuracy_verified": False, "temporal_sync_verified": False,
                    "source_survey_path": "survey.json", "source_survey_sha256": runner.file_sha256(root / "survey.json"),
                    "calibration_assessment_path": "assessment.json",
                    "calibration_assessment_sha256": runner.file_sha256(root / "assessment.json"),
                    "images": [{"image_id": "diagnostic-a-C10119", "path": "native.png", "sha256": checksum,
                                "source_pts_seconds": 0.0, "object_id": "physical-object-a", "recording_id": "recording-a",
                                "camera_id": "C10119", "prompt": "toy"}]}
        self.write(root, "images.json", manifest)
        return manifest, survey, assessment

    def rewrite(self, root, manifest, survey=None, assessment=None):
        if survey is not None:
            self.write(root, "survey.json", survey)
            manifest["source_survey_sha256"] = runner.file_sha256(root / "survey.json")
        if assessment is not None:
            self.write(root, "assessment.json", assessment)
            manifest["calibration_assessment_sha256"] = runner.file_sha256(root / "assessment.json")
        self.write(root, "images.json", manifest)

    def test_diagnostic_native_identity_is_checked_but_acceptance_gates_stay_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            _, rows, evidence = runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)
            self.assertTrue(evidence["diagnostic_2d_review"])
            self.assertEqual(evidence["correspondence_scope"], "diagnostic_2d_review")
            for key in ("multiview_dataset_ready", "calibration_accuracy_verified", "temporal_sync_verified"):
                self.assertIs(evidence[key], False)
                self.assertIs(rows[0][key], False)
            self.assertEqual(evidence["diagnostic_image_evidence"][rows[0]["image_id"]]["native_image_size"], [9, 5])

    def test_explicit_cli_flag_is_required_and_completion_does_not_pass_a_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _, _ = self.fixture(root)
            manifest["status"] = "diagnostic_2d_inference_completed_review_pending"
            self.rewrite(root, manifest)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json")
            # Neither a mixed true gate nor successful inference promotes the source.
            manifest["calibration_temporal_gate_passed"] = True
            self.rewrite(root, manifest)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json")

    def test_all_diagnostic_false_claims_and_purpose_are_mandatory(self):
        for field in ("multiview_dataset_ready", "calibration_accuracy_verified", "temporal_sync_verified", "purpose"):
            for missing in (False, True):
                with self.subTest(field=field, missing=missing), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest, _, _ = self.fixture(root)
                    if missing:
                        manifest.pop(field)
                    else:
                        manifest[field] = True
                    self.rewrite(root, manifest)
                    with self.assertRaises(ValueError):
                        runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)

    def test_normal_manifest_cannot_silently_fall_back_to_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            InferenceGateTests.fixture(self, root)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)

    def test_missing_or_tampered_source_and_calibration_assessment_block_diagnostics(self):
        for stem in ("source_survey", "calibration_assessment"):
            for missing in (False, True):
                with self.subTest(stem=stem, missing=missing), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest, _, _ = self.fixture(root)
                    if missing:
                        manifest[stem + "_path"] = "missing.json"
                    else:
                        manifest[stem + "_sha256"] = "0" * 64
                    self.rewrite(root, manifest)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)

    def test_recording_camera_pts_and_image_hash_must_match_survey(self):
        for field, value in (("recording_id", "wrong-recording"), ("camera_id", "C99999"),
                             ("source_pts_seconds", 1.0), ("sha256", "0" * 64), ("sha256", None)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _, _ = self.fixture(root)
                manifest["images"][0][field] = value
                self.rewrite(root, manifest)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)

    def test_actual_native_image_pixels_and_dimensions_are_verified(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            Image.new("RGB", (9, 5), (0, 0, 0)).save(root / "native.png")
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, survey, _ = self.fixture(root)
            for view in survey["recordings"][0]["views"]:
                view["frames"][0]["image_size"] = [5, 9]
            self.rewrite(root, manifest, survey=survey)
            with self.assertRaises(ValueError):
                runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)

    def test_wrong_recording_or_verified_accuracy_calibration_assessment_rejected(self):
        for field, value in (("recording_id", "wrong-recording"), ("calibration_accuracy_verified", True)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _, assessment = self.fixture(root)
                assessment["recordings"][0][field] = value
                self.rewrite(root, manifest, assessment=assessment)
                with self.assertRaises(ValueError):
                    runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)

    def test_failed_tracking_is_preserved_and_does_not_become_static_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _, _ = self.fixture(root)
            tracking = {"recordings": [{"recording_id": "recording-a", "object_id": "physical-object-a", "views": [
                {"camera_id": "C10119", "status": "measured", "static_state_local_stability_passed": False,
                 "claim_first_second": {"static_state_local_stability_passed": False}}]}]}
            self.write(root, "tracking.json", tracking)
            manifest.update(tracking_stability_path="tracking.json", tracking_stability_sha256=runner.file_sha256(root / "tracking.json"))
            self.rewrite(root, manifest)
            _, rows, evidence = runner.validate_image_manifest(root / "images.json", diagnostic_2d_review=True)
            result = evidence["diagnostic_image_evidence"][rows[0]["image_id"]]["tracking_diagnostics"]
            self.assertTrue(result["available"])
            self.assertIs(result["static_state_local_stability_passed"], False)
            self.assertIs(evidence["multiview_dataset_ready"], False)

    def test_every_diagnostic_candidate_retains_false_readiness_and_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            record = runner.save_stage(Path(directory), "text_only", np.zeros((1, 2, 2), bool), [[0, 0, 2, 2]], [.8],
                                       height=2, width=2, prompt="toy", correspondence_scope=runner.DIAGNOSTIC_SCOPE,
                                       temporal_sync_verified=False, diagnostic_2d_review=True)
            for result in (record, *record["candidates"]):
                self.assertTrue(result["diagnostic_2d_review"])
                self.assertIs(result["multiview_dataset_ready"], False)
                self.assertIs(result["calibration_accuracy_verified"], False)
                self.assertIs(result["temporal_sync_verified"], False)
                self.assertIs(result["accuracy_established"], False)
            self.assertIsNone(record["candidates"][0]["iou"])

    def test_diagnostic_output_cannot_claim_synchronization(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                runner.save_stage(Path(directory), "text_only", np.zeros((1, 2, 2), bool), [[0, 0, 2, 2]], [.8],
                                  height=2, width=2, prompt="toy", correspondence_scope=runner.DIAGNOSTIC_SCOPE,
                                  temporal_sync_verified=True, diagnostic_2d_review=True)


if __name__ == "__main__":
    unittest.main()
