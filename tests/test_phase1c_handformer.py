"""CPU contracts for the official HandFormer adapter; no trained model needed."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from e0.data import segment_features, sha256
from e0.source_profile import dimensionless_profile
from phase1c.handformer import REVISION, compatible_tree, build_model, verify_upstream
from phase1c.handformer_data import (HandFormerDataset, crop_and_resize,
                                   load_compatibility, prepare)


REPO = Path(__file__).resolve().parents[1]


class HandFormerAdapterTests(unittest.TestCase):
    def test_canonical_git_blob_hash_accepts_lf_and_crlf_but_rejects_edits(self):
        canonical = b"def forward(x):\n    return x + 1\n"
        path = "HandFormer/utils.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / path
            target.parent.mkdir(parents=True)
            provenance = root / "provenance.json"
            manifest = {"revision": REVISION, "source_sha256": {
                path: hashlib.sha256(canonical).hexdigest()}}
            provenance.write_text(json.dumps(manifest))

            def git(command, **options):
                if command[-2:] == ["rev-parse", "HEAD"]:
                    return REVISION + "\n"
                self.assertEqual(command[-2:], ["show", REVISION + ":" + path])
                self.assertNotIn("text", options)  # Canonical blob read as bytes.
                return canonical

            with mock.patch("phase1c.handformer.PROVENANCE", provenance), mock.patch(
                    "phase1c.handformer.subprocess.check_output", side_effect=git):
                for content in (canonical, canonical.replace(b"\n", b"\r\n")):
                    target.write_bytes(content)
                    self.assertEqual(verify_upstream(root), manifest)
                for content in (canonical.replace(b"+ 1", b"+ 2"), canonical + b" ",
                                canonical.replace(b"\n", b"\r")):
                    target.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                        verify_upstream(root)
            # Matching a worktree alone is insufficient: the pinned Git blob
            # must independently match the committed canonical fingerprint.
            target.write_bytes(canonical)
            with mock.patch("phase1c.handformer.PROVENANCE", provenance), mock.patch(
                    "phase1c.handformer.subprocess.check_output",
                    side_effect=[REVISION + "\n", canonical + b"edited"]):
                with self.assertRaisesRegex(ValueError, "pinned Git blob hash mismatch"):
                    verify_upstream(root)

    def fixture(self, root):
        profile = dimensionless_profile()
        frames = {}
        confidence = {}
        for frame in range(12):
            hands = np.arange(126, dtype=float).reshape(2, 21, 3) / 10
            hands += frame / 100
            hands[0, 0] = 0  # A valid zero coordinate is not missing.
            frames[frame] = hands
            confidence[frame] = np.array([1.0, 0.0 if frame == 4 else 0.8])
        source = root / "pose.json"
        scores = root / "confidence.json"
        source.write_text(json.dumps({str(k): {"0": v[0].tolist(), "1": v[1].tolist()}
                                      for k, v in frames.items()}))
        scores.write_text(json.dumps({str(k): {"0": float(v[0]), "1": float(v[1])}
                                      for k, v in confidence.items()}))
        row = {"segment_id": "train:0:1", "recording_id": "recording", "split": "train",
               "start_frame": 0, "end_frame": 5, "label": 0}
        _, row["quality"] = segment_features(frames, row, profile, confidences=confidence)
        artifact = {"ready_for_training": True, "classes": [{"label": 0}, {"label": 1}],
                    "segments": [row], "source_manifest": {"pose_profile": profile,
                    "recordings": {"recording": {"path": str(source), "sha256": sha256(source),
                    "confidence": {"path": str(scores), "sha256": sha256(scores)}}}}}
        dataset = root / "dataset.json"
        dataset.write_text(json.dumps(artifact))
        return dataset, artifact, frames, confidence

    def test_prepare_preserves_population_confidence_and_original_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, artifact, frames, confidence = self.fixture(root)
            result = prepare(dataset, root / "cache", project_root=root)
            self.assertEqual(result["additional_exclusions"], [])
            self.assertEqual(result["segment_ids"], ["train:0:1"])
            self.assertLess(result["cache_bytes"], 100_000)
            path = root / "cache/compatibility.json"
            self.assertEqual(prepare(dataset, root / "cache", project_root=root), result)
            # A completed per-recording progress journal can resume a stage that
            # stopped before publishing the final compatibility manifest.
            path.rename(root / "cache/previous-completion.json")
            self.assertEqual(prepare(dataset, root / "cache", project_root=root), result)
            config = {"sanity_mode": True, "preprocessing": {"samples": 12,
                       "evaluation_crop_fraction": 1.0}}
            data = HandFormerDataset(dataset, artifact, "train", config, path)
            x, label, index = data[0]
            self.assertEqual((label, index), (0, 0))
            self.assertEqual(tuple(x.shape), (3, 12, 21, 2))
            row = artifact["segments"][0]
            scale, origin = row["quality"]["source_to_feature_scale"], np.array(row["quality"]["origin_feature_units"])
            self.assertTrue(torch.all(x[:, 4, :, 1] == 0))
            np.testing.assert_allclose(x[:, 0, 0, 0], -origin, atol=1e-5)
            np.testing.assert_allclose(x[:, 1, 8, 0], frames[1][0, 8] * scale - origin, atol=1e-5)
            data.close()
            item = result["recordings"]["recording"]["files"]["xyz"]
            target = root / "cache" / item["path"]
            with target.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_compatibility(path, dataset, artifact, verify_hashes=True)

    def test_internal_missing_frames_zero_fill_without_population_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, artifact, frames, confidence = self.fixture(root)
            del frames[3]
            del confidence[3]
            entry = artifact["source_manifest"]["recordings"]["recording"]
            source, scores = Path(entry["path"]), Path(entry["confidence"]["path"])
            source.write_text(json.dumps({str(k): {"0": v[0].tolist(), "1": v[1].tolist()} for k, v in frames.items()}))
            scores.write_text(json.dumps({str(k): {"0": float(v[0]), "1": float(v[1])} for k, v in confidence.items()}))
            entry["sha256"], entry["confidence"]["sha256"] = sha256(source), sha256(scores)
            row = artifact["segments"][0]
            _, row["quality"] = segment_features(frames, row, dimensionless_profile(), confidences=confidence)
            dataset.write_text(json.dumps(artifact))
            report = prepare(dataset, root / "cache", project_root=root)
            self.assertEqual(report["additional_exclusions"], [])
            self.assertEqual(report["missing_observation_totals"]["segments_with_missing_source_frames"], 1)
            self.assertEqual(report["segment_missing_observations"][row["segment_id"]]["missing_source_frames"], 1)
            data = HandFormerDataset(dataset, artifact, "train", {"sanity_mode": True,
                "preprocessing": {"samples": 12, "evaluation_crop_fraction": 1.0}}, root / "cache/compatibility.json")
            self.assertTrue(torch.all(data[0][0][:, 3] == 0))
            self.assertEqual(len(data), 1)
            data.close()

    def test_preparation_enforces_storage_and_complete_intervals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, artifact, _, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "storage bound"):
                prepare(dataset, root / "tiny", max_cache_bytes=1, project_root=root)
            artifact["segments"][0]["end_frame"] = 20
            dataset.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "full interval"):
                prepare(dataset, root / "missing", project_root=root)

    def test_zero_coordinate_frames_keep_their_temporal_positions(self):
        data = np.ones((3, 4, 21, 2), dtype=np.float32)
        data[:, 1] = 0
        result = crop_and_resize(data, False, samples=4, evaluation_fraction=1.0)
        np.testing.assert_array_equal(result.numpy(), data)
        zeros = crop_and_resize(np.zeros_like(data), False, samples=120)
        self.assertEqual(tuple(zeros.shape), (3, 120, 21, 2))
        self.assertTrue(torch.isfinite(zeros).all())

    def test_temporal_crop_matches_author_explicit_length_interpolation(self):
        data = np.arange(3 * 100 * 21 * 2, dtype=np.float32).reshape(3, 100, 21, 2)
        observed = crop_and_resize(data, False, samples=120, evaluation_fraction=0.85)
        bias = int((1 - 0.85) * 100 / 2)
        expected = crop_and_resize(data[:, bias:100 - bias], False, samples=120, evaluation_fraction=1)
        torch.testing.assert_close(observed, expected)
        np.random.seed(17)
        first = crop_and_resize(data, True)
        np.random.seed(17)
        torch.testing.assert_close(first, crop_and_resize(data, True))

    def test_exact_broadcast_patch_preserves_values_and_gradients(self):
        source = "def f(self, x):\n return repeat(self.action_token, '() n d -> b n d', b=x.shape[0])\n"
        scope = {}
        exec(compile(compatible_tree(source, "fixture.py"), "fixture.py", "exec"), scope)
        token = torch.randn(1, 1, 8, requires_grad=True)
        holder = type("Holder", (), {"action_token": token})()
        observed = scope["f"](holder, torch.zeros(3, 1))
        expected = token.repeat(3, 1, 1)
        torch.testing.assert_close(observed, expected)
        gradient = torch.randn_like(observed)
        actual_grad = torch.autograd.grad(observed, token, gradient, retain_graph=True)[0]
        reference_grad = torch.autograd.grad(expected, token, gradient)[0]
        torch.testing.assert_close(actual_grad, reference_grad)
        with self.assertRaisesRegex(ValueError, "Unknown repeat"):
            compatible_tree("repeat(x, 'unknown', b=2)", "fixture.py")

    def test_official_pinned_architecture_forward_backward(self):
        value = os.environ.get("JPA_HANDFORMER_ROOT")
        root = Path(value) if value else REPO / "data/phase1c_source_review/HandFormer"
        if not (root / ".git").exists():
            self.skipTest("Optional pinned upstream checkout not present; cluster CPU prep requires this check")
        verify_upstream(root)
        config = json.loads((REPO / "configs/b1_handformer.json").read_text())
        config["upstream_path"] = str(root)
        model = build_model(config, {"classes": [{}] * 100})
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3_805_866)
        model.eval()
        x = torch.randn(1, 3, 120, 21, 2)
        output = model(x)
        self.assertEqual(tuple(output.shape), (1, 100))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
