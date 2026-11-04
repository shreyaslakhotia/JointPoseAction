"""CPU-only provenance and crop/camera tests; no model package imports."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from reconstruction.adapters import (
    MODEL_MANIFEST, crop_resize_transform, file_record, model_spec,
    prepare_input_bundle, validate_external_artifact, verify_input_bundle, camera_metadata_from_camera,
)


class ReconstructionAdapterTests(unittest.TestCase):
    def test_common_camera_bridge_preserves_units_distortion_and_extrinsics(self):
        from reconstruction.cameras import Camera
        c=Camera('synthetic-camera-0',np.array([[20,0,6],[0,22,5],[0,0,1.]]),np.eye(4),(12,10),
                 (.01,.02,.003,.004,.005),'synthetic_world','synthetic_length_unit')
        self.camera=camera_metadata_from_camera(c,self.calibration)
        bundle=self.prepare()
        self.assertEqual(bundle['camera']['units'],'synthetic_length_unit')
        self.assertEqual(bundle['camera']['distortion_coefficients'],{'k1':.01,'k2':.02,'p1':.003,'p2':.004,'k3':.005})
        np.testing.assert_array_equal(bundle['camera']['world_to_camera'],c.world_to_camera)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rgb = self.root / "rgb.png"
        self.mask = self.root / "mask.png"
        self.calibration = self.root / "calibration.json"
        Image.new("RGB", (12, 10), (40, 80, 120)).save(self.rgb)
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[3:6, 4:8] = 255
        Image.fromarray(mask).save(self.mask)
        self.calibration.write_text('{"synthetic":true}', encoding="utf-8")
        self.camera = {
            "camera_id": "synthetic-camera-0", "image_size_wh": [12, 10],
            "K": [[20, 0, 6], [0, 22, 5], [0, 0, 1]], "world_to_camera": np.eye(4).tolist(),
            "convention": "opencv_x_right_y_down_z_forward", "pixel_convention": "integer_centers",
            "distortion_model": "none", "distortion_coefficients": {},
            "coordinate_frame": "synthetic_world", "units": "synthetic_unit",
            "calibration_file": file_record(self.calibration),
        }
        self.ids = {"dataset_revision": "synthetic-v1", "recording_id": "synthetic-recording",
                    "object_id": "synthetic-cuboid", "camera_id": "synthetic-camera-0",
                    "mask_run_id": "synthetic-analytic-mask", "source_timestamp": {"basis": "synthetic_time", "seconds": 0.0}}
        self.bundle_path = self.root / "recipe.json"

    def prepare(self, **kwargs):
        return prepare_input_bundle(self.bundle_path, "stable_fast_3d", self.rgb, self.mask,
                                    camera_metadata=self.camera, source_identifiers=self.ids, **kwargs)

    def artifact(self):
        bundle = self.prepare()
        artifact = self.root / "external.obj"
        artifact.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        config = self.root / "run-config.json"
        config.write_text('{"synthetic_external_artifact_test":true}', encoding="utf-8")
        metadata = {
            "artifact_sha256": file_record(artifact)["sha256"],
            "input_bundle_sha256": file_record(self.bundle_path)["sha256"],
            "method_id": bundle["method_id"], "source_commit": bundle["model_source"]["commit"],
            "weights_revision": bundle["model_weights"]["revision"], "checkpoint_sha256": ["a" * 64],
            "configuration_sha256": file_record(config)["sha256"], "configuration_file": file_record(config),
            "representation": "triangle_mesh", "coordinate_frame": "model_native_unaligned",
            "units": "canonical_unitless", "alignment_required": True,
            "camera_alignment_verified": False, "metric_scale_verified": False,
            "runtime_seconds": 0.25, "peak_gpu_memory_bytes": None,
            "peak_gpu_memory_bytes_unavailable_reason": "synthetic structural test; no GPU inference",
        }
        return artifact, metadata

    def test_all_four_sources_pinned_with_no_access_claim(self):
        manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["models"]), {"sam3d_objects", "hunyuan3d_2", "stable_fast_3d", "instantmesh"})
        for name in manifest["models"]:
            spec = model_spec(name)
            self.assertRegex(spec["source"]["commit"], r"^[a-f0-9]{40}$")
            self.assertRegex(spec["weights"]["revision"], r"^[a-f0-9]{40}$")
            self.assertFalse(spec["weights"]["access_verified_for_this_account"])
        with self.assertRaises(ValueError):
            model_spec("invented-method")

    def test_half_pixel_transform_and_intrinsics_preserve_projection(self):
        bundle = self.prepare(crop_xyxy=[2, 1, 10, 9], target_size_wh=[16, 24])
        affine = np.array(bundle["preprocessing"]["grid_transform"]["original_to_resized"])
        K0 = np.array(self.camera["K"])
        K1 = np.array(bundle["camera"]["resized_grid_K"])
        xyz = np.array([0.1, -0.2, 2.])
        p0 = K0 @ xyz
        p1 = K1 @ xyz
        np.testing.assert_allclose(p1 / p1[2], affine @ (p0 / p0[2]))
        np.testing.assert_allclose(affine @ [2, 1, 1], [0.5, 1., 1.])
        inverse = bundle["preprocessing"]["grid_transform"]["resized_to_original"]
        np.testing.assert_allclose(np.array(inverse) @ affine, np.eye(3), atol=1e-12)

    def test_recipe_does_not_process_or_copy_media(self):
        before = {p.name for p in self.root.iterdir()}
        bundle = self.prepare(background_policy="composite_white")
        self.assertEqual({p.name for p in self.root.iterdir()} - before, {"recipe.json"})
        self.assertFalse(bundle["inference_executed"])
        self.assertFalse(bundle["reconstruction_ready"])
        self.assertFalse(bundle["input_preparation_executed"])
        self.assertEqual(bundle["preprocessing"]["background_rgb"], [255, 255, 255])
        self.assertEqual(verify_input_bundle(self.bundle_path), bundle)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_reject_crop_that_discards_foreground(self):
        with self.assertRaisesRegex(ValueError, "foreground"):
            self.prepare(crop_xyxy=[5, 0, 12, 10])

    def test_reject_bad_crop_and_zero_resize(self):
        for crop in ([0, 0, 13, 10], [1, 1, 1, 8], [0.5, 1, 8, 9]):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                crop_resize_transform([12, 10], crop)
        with self.assertRaises(ValueError):
            crop_resize_transform([12, 10], target_size_wh=[0, 4])

    def test_probability_or_overlay_is_not_binary_mask(self):
        Image.fromarray(np.full((10, 12), 128, dtype=np.uint8)).save(self.mask)
        with self.assertRaisesRegex(ValueError, "binary"):
            self.prepare()
        Image.new("RGB", (12, 10), (255, 0, 255)).save(self.mask)
        with self.assertRaisesRegex(ValueError, "single-channel"):
            self.prepare()

    def test_empty_mismatch_and_ambiguous_encoding_rejected(self):
        for data in (np.zeros((10, 12), np.uint8), np.ones((9, 12), np.uint8)):
            Image.fromarray(data).save(self.mask)
            with self.assertRaises(ValueError):
                self.prepare()
        data = np.zeros((10, 12), np.uint8)
        data[3, 4], data[4, 4] = 1, 255
        Image.fromarray(data).save(self.mask)
        with self.assertRaisesRegex(ValueError, "Mixed"):
            self.prepare()

    def test_named_distortion_preserved_without_claiming_undistortion(self):
        self.camera["distortion_model"] = "opencv_brown_conrady"
        self.camera["distortion_coefficients"] = {"k1": 0.1, "k2": 0., "p1": 0.01, "p2": 0., "k3": 0.}
        bundle = self.prepare(target_size_wh=[24, 20])
        self.assertEqual(bundle["camera"]["distortion_coefficients"], self.camera["distortion_coefficients"])
        self.assertFalse(bundle["preprocessing"]["grid_transform"]["undistortion_executed"])

    def test_invalid_camera_calibration_and_identity_rejected(self):
        for field, value in (("image_size_wh", [8, 8]), ("camera_id", "wrong"),
                             ("world_to_camera", np.diag([-1, 1, 1, 1]).tolist()),
                             ("convention", "opengl"), ("pixel_convention", "corner_origin"),
                             ("units", "probably_mm"), ("distortion_model", "guessed_model")):
            with self.subTest(field=field):
                original = copy.deepcopy(self.camera)
                self.camera[field] = value
                with self.assertRaises(ValueError):
                    self.prepare()
                self.camera = original

    def test_timestamp_basis_required_without_claiming_synchronization(self):
        del self.ids["source_timestamp"]
        with self.assertRaisesRegex(ValueError, "timestamp"):
            self.prepare()

    def test_source_mutation_detected(self):
        self.prepare()
        Image.new("RGB", (12, 10), (1, 2, 3)).save(self.rgb)
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_input_bundle(self.bundle_path)

    def test_grid_intrinsic_tamper_detected(self):
        bundle = self.prepare()
        bundle["camera"]["resized_grid_K"][0][0] += 10
        self.bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            verify_input_bundle(self.bundle_path)

    def test_external_artifact_validation_never_proves_geometry(self):
        artifact, metadata = self.artifact()
        result = validate_external_artifact(artifact, metadata, input_bundle_path=self.bundle_path)
        self.assertFalse(result["geometry_validated"])
        self.assertFalse(result["benchmark_ready"])
        self.assertFalse(result["checkpoint_payload_hashes_verified_here"])
        self.assertEqual(result["artifact"]["bytes"], artifact.stat().st_size)

    def test_artifact_alignment_metric_and_source_claims_rejected(self):
        artifact, metadata = self.artifact()
        for key, value in (("units", "m"), ("camera_alignment_verified", True),
                           ("metric_scale_verified", True), ("alignment_required", False),
                           ("coordinate_frame", "assembly101_world"), ("source_commit", "b" * 40),
                           ("artifact_sha256", "0" * 64), ("representation", "point_cloud")):
            candidate = copy.deepcopy(metadata)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_external_artifact(artifact, candidate, input_bundle_path=self.bundle_path)

    def test_resources_missing_reason_and_config_mutation_rejected(self):
        artifact, metadata = self.artifact()
        candidate = copy.deepcopy(metadata)
        del candidate["peak_gpu_memory_bytes_unavailable_reason"]
        with self.assertRaisesRegex(ValueError, "reason"):
            validate_external_artifact(artifact, candidate, input_bundle_path=self.bundle_path)
        Path(metadata["configuration_file"]["path"]).write_text('{}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_external_artifact(artifact, metadata, input_bundle_path=self.bundle_path)


if __name__ == "__main__":
    unittest.main()
