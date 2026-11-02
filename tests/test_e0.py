"""All fixtures are synthetic; no optimizer step, training run or GPU required."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import contextlib
from unittest.mock import patch

import numpy as np
import torch

from e0.acquisition import RangeFile, inspect_archive, validate_range_headers
from e0.data import (check_leakage, choose_classes, load_actions, load_segments,
                     pose_bounds, read_cleaned, require_profile, segment_features, sha256)
from e0.metrics import baselines, compute_metrics
from e0.model import TCN
from e0.preprocess import prepare
from e0.runtime import SegmentDataset, load_artifact, seed_all
from e0.train import epoch


def profile():
    return {"status": "synthetic", "synthetic": True, "schema": "assembly101_cleaned_landmarks3D",
            "hand_keys": ["0", "1"], "hand_order": ["left", "right"], "wrist_index": 5,
            "joint_order": [f"synthetic_joint_{i}" for i in range(21)],
            "units_to_m": 1.0, "coordinate_frame": "synthetic_world", "pose_fps": 60,
            "annotation_fps": 30, "annotation_index_base": 0, "pose_frame_offset": 0,
            "end_inclusive": True, "zero_is_invalid": True, "evidence": ["synthetic fixture only"]}


def recording(person):
    return f"nusar-2021_action_both_{person}-a01_{person}_user_id_2021-02-01_000000"


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    actions_path = root / "actions.csv"
    fields = ["action_id", "verb_id", "noun_id", "action_cls", "verb_cls", "noun_cls"]
    write_csv(actions_path, fields, [dict(zip(fields, row)) for row in
                                   [(10, 1, 1, "a", "v1", "n1"), (20, 2, 2, "b", "v2", "n2")]])
    manifest = {"status": "synthetic", "dataset_revision": "synthetic-not-official",
                "pose_profile": profile(), "actions": {"path": str(actions_path), "sha256": sha256(actions_path)},
                "splits": {}, "recordings": {}}
    columns = ["id", "video", "start_frame", "end_frame", "action_id", "verb_id", "noun_id"]
    for person, split in zip((9001, 9002, 9003), ("train", "validation", "test")):
        rec = recording(person)
        rows = [{"id": 0, "video": f"{rec}/C10119_rgb.mp4", "start_frame": 1, "end_frame": 2,
                 "action_id": 10, "verb_id": 1, "noun_id": 1},
                {"id": 1, "video": f"{rec}/C10119_rgb.mp4", "start_frame": 2, "end_frame": 3,
                 "action_id": 20, "verb_id": 2, "noun_id": 2}]
        rows += [{**rows[0], "id": 2, "video": f"{rec}/C10095_rgb.mp4"}]
        csv_path = root / f"{split}.csv"
        write_csv(csv_path, columns, rows)
        manifest["splits"][split] = {"path": str(csv_path), "sha256": sha256(csv_path)}
        pose_path = root / f"{rec}.json"
        poses = {str(t): {str(h): (np.arange(63).reshape(21, 3) + 1 + h * 100 + t).tolist()
                         for h in range(2)} for t in range(12)}
        pose_path.write_text(json.dumps(poses))
        manifest["recordings"][rec] = {"path": str(pose_path), "sha256": sha256(pose_path),
                                       "split": split, "participant_id": str(person)}
    return manifest


class DataTests(unittest.TestCase):
    def test_cached_feasibility_joins_pose_inventory_and_selects_from_train(self):
        from scripts.data.e0_feasibility import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = fixture(root)
            train_path = Path(m["splits"]["train"]["path"])
            with train_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                columns, original = reader.fieldnames, list(reader)
            rows = [{**original[i % 2], "id": i} for i in range(100)]
            write_csv(train_path, columns, rows)
            files = [{"name": "actions.csv", **m["actions"]}]
            files += [{"name": f"{split}.csv", "path": entry["path"], "sha256": sha256(entry["path"])} for split, entry in m["splits"].items()]
            sample = next(iter(m["recordings"].values()))["path"]
            review = {"revision": "synthetic-only", "files": files,
                      "archive": {"sample_path": sample, "cleaned_members": [
                          {"path": f"landmarks3D/{name}.json", "compressed_bytes": 100, "uncompressed_bytes": 500}
                          for name in m["recordings"]]}}
            review_path, output = root / "review.json", root / "feasibility.json"
            review_path.write_text(json.dumps(review))
            with patch.dict("os.environ", {"SLURM_JOB_ID": "synthetic-test-not-a-job"}), patch("sys.argv", ["test", "--review", str(review_path), "--output", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads(output.read_text())
            self.assertEqual(len(result["selection"]["class_mapping"]), 2)
            self.assertEqual(result["coverage"]["train"]["candidate_segments"], 100)
            self.assertEqual(result["candidate_compressed_pose_bytes"], 300)
            self.assertFalse(result["training_ready"])

    def test_overlapping_segments_retained_and_views_not_multiplied(self):
        with tempfile.TemporaryDirectory() as directory:
            m = fixture(Path(directory))
            actions = load_actions(m["actions"]["path"])
            segments, excluded = load_segments({k: v["path"] for k, v in m["splits"].items()}, actions)
            self.assertEqual(len(segments), 6)
            self.assertEqual(len(excluded), 3)
            self.assertLessEqual(segments[1]["start_frame"], segments[0]["end_frame"])
            self.assertNotEqual(segments[0]["action_id"], segments[1]["action_id"])

    def test_endpoint_offset_and_30_to_60_mapping(self):
        p = profile()
        s = {"start_frame": 3, "end_frame": 5}
        self.assertEqual(pose_bounds(s, p), (6, 12))
        p["end_inclusive"] = False
        self.assertEqual(pose_bounds(s, p), (6, 10))
        p.update(annotation_index_base=1, pose_frame_offset=7)
        self.assertEqual(pose_bounds(s, p), (11, 15))

    def test_distinct_annotation_ids_with_identical_intervals_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            m = fixture(Path(directory))
            path = Path(m["splits"]["train"]["path"])
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows.append({**rows[0], "id": 100})
            write_csv(path, fields, rows)
            segments, _ = load_segments({k: v["path"] for k, v in m["splits"].items()}, load_actions(m["actions"]["path"]))
            self.assertEqual(len(segments), 7)

    def test_missing_nan_zero_and_wrist_masks_no_interpolation(self):
        p = profile()
        frames = {i: np.ones((2, 21, 3)) * (i + 1) for i in range(4)}
        del frames[1]
        frames[2][0, 3] = np.nan
        frames[2][1, 5] = 0
        x, q = segment_features(frames, {"start_frame": 0, "end_frame": 1}, p, 4)
        self.assertEqual(x.shape, (4, 216))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.all(x[1] == 0))
        self.assertEqual(q["missing_sampled_frames"], 1)
        self.assertEqual(x[2, 126 + 3:126 + 6].tolist(), [0, 0, 0])
        self.assertTrue(np.all(x[2, 63:126] == 0))
        self.assertEqual(q["pose_frame_indices"], [0, 1, 2, 3])

    def test_translation_invariance_preserves_wrist_motion(self):
        frames = {i: np.ones((2, 21, 3)) * (i + 1) for i in range(4)}
        segment = {"start_frame": 0, "end_frame": 1}
        a, _ = segment_features(frames, segment, profile(), 4)
        b, _ = segment_features({i: x + 30 for i, x in frames.items()}, segment, profile(), 4)
        np.testing.assert_allclose(a, b)
        self.assertGreater(float(np.abs(a[-1, 126:132]).sum()), 0)

    def test_zero_wrist_can_be_a_valid_origin_and_is_not_dropped(self):
        p = profile()
        p["zero_is_invalid"] = False
        frames = {i: np.ones((2, 21, 3)) for i in range(4)}
        for value in frames.values():
            value[1, 5] = 0
        x, quality = segment_features(frames, {"start_frame": 0, "end_frame": 1}, p, 4)
        self.assertTrue(np.all(x[:, 132 + 21 + 5] == 1))
        self.assertTrue(np.all(x[:, 174 + 21 + 5] == 1))
        self.assertTrue(np.any(x[:, 63:126] != 0))
        self.assertEqual(quality["zero_policy"], "preserved_ambiguity_not_tracking_confidence")
        self.assertEqual(quality["zero_coordinate_joint_count"], 4)

    def test_recording_and_participant_leakage(self):
        s = [{"recording_id": "r1", "participant_id": "p", "split": "train"},
             {"recording_id": "r1", "participant_id": "p", "split": "test"}]
        with self.assertRaisesRegex(ValueError, "Recording leakage"):
            check_leakage(s)
        s[1]["recording_id"] = "r2"
        with self.assertRaisesRegex(ValueError, "Participant overlap"):
            check_leakage(s)
        self.assertIn("p", check_leakage(s, "report_official_overlap")["participant_overlap"])

    def test_vocabulary_uses_training_only(self):
        s = [{"split": "train", "action_id": 10}] * 2 + [{"split": "train", "action_id": 20}]
        s += [{"split": "test", "action_id": 99}] * 100
        self.assertEqual(choose_classes(s, 2), [10])

    def test_unverified_real_contract_refused(self):
        with self.assertRaises(ValueError):
            require_profile(profile())
        p = profile()
        p["units_to_m"] = None
        with self.assertRaisesRegex(ValueError, "units"):
            require_profile(p, allow_unverified=True)

    def test_cleaned_rejects_legacy_and_wrong_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            path.write_text(json.dumps([{"landmarks": {}}]))
            with self.assertRaises(ValueError):
                read_cleaned(path)
            path.write_text(json.dumps({"0": {"0": [[1, 2, 3]], "1": [[1, 2, 3]]}}))
            with self.assertRaises(ValueError):
                read_cleaned(path)

    def test_end_to_end_synthetic_preprocess_and_loader_hash_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = fixture(root)
            out = root / "prepared"
            a = prepare(m, out, samples=8, min_train_examples=1, synthetic=True)
            self.assertEqual(a["summary"]["split_counts"], {"train": 2, "validation": 2, "test": 2})
            self.assertEqual(a["summary"]["exclusions"]["alternate_view"], 3)
            with self.assertRaises(ValueError):
                load_artifact(out / "dataset.json")
            loaded = load_artifact(out / "dataset.json", allow_synthetic=True)
            dataset = SegmentDataset(out / "dataset.json", loaded, "train")
            x, label, _ = dataset[0]
            self.assertEqual(tuple(x.shape), (8, 216))
            self.assertEqual(label, 0)
            (out / loaded["segments"][0]["path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_artifact(out / "dataset.json", allow_synthetic=True)

    def test_real_preprocessing_blocked_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                prepare(fixture(root), root / "out")
            self.assertFalse((root / "out").exists())

    def test_failed_quality_preprocessing_cannot_load_for_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = fixture(root)
            first = next(iter(m["recordings"].values()))
            path = Path(first["path"])
            frames = json.loads(path.read_text())
            for frame in frames.values():
                frame["0"] = np.zeros((21, 3)).tolist()
                frame["1"] = np.zeros((21, 3)).tolist()
            path.write_text(json.dumps(frames))
            first["sha256"] = sha256(path)
            with self.assertRaises(ValueError):
                prepare(m, root / "out", min_train_examples=1, synthetic=True)
            report = json.loads((root / "out/exclusions.json").read_text())
            self.assertTrue(any(x["reason"] == "insufficient_valid_local_joints" for x in report))
            with self.assertRaisesRegex(ValueError, "readiness"):
                load_artifact(root / "out/dataset.json", allow_synthetic=True)


class ModelMetricTests(unittest.TestCase):
    def test_inference_epoch_cpu_never_updates_parameters(self):
        model = TCN(3, 2, width=4, dilations=(1,), dropout=0)
        before = [p.detach().clone() for p in model.parameters()]
        loader = [(torch.ones(2, 8, 3), torch.tensor([0, 1]), torch.tensor([0, 1]))]
        metrics, p, y = epoch(model, loader, torch.device("cpu"))
        self.assertEqual(p.shape, (2, 2))
        self.assertGreater(metrics["loss"], 0)
        self.assertEqual(y.tolist(), [0, 1])
        for expected, actual in zip(before, model.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_cpu_forward_backward_shapes_gradients_without_optimizer(self):
        seed_all(42)
        model = TCN(216, 7, width=8, dilations=(1, 2), dropout=0)
        x = torch.randn(3, 16, 216)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (3, 7))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3, 6])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.ones(3, 216, 16))

    def test_seed_repeatability(self):
        seed_all(12)
        a = TCN(3, 2, width=4, dropout=0)(torch.ones(2, 8, 3)).detach()
        seed_all(12)
        b = TCN(3, 2, width=4, dropout=0)(torch.ones(2, 8, 3)).detach()
        torch.testing.assert_close(a, b)

    def test_known_metrics_macro_f1_and_top5(self):
        classes = [{"action_id": i, "verb_id": i, "noun_id": i} for i in range(6)]
        p = np.eye(6)[[0, 1, 0]] * 0.9 + 0.1 / 6
        m = compute_metrics(p, [0, 1, 2], classes)
        self.assertAlmostEqual(m["top1"], 2 / 3)
        self.assertEqual(m["samples"], 3)
        self.assertEqual(m["per_class"][3]["accuracy"], None)
        self.assertEqual(m["confusion_matrix"][2][0], 1)
        self.assertAlmostEqual(m["macro_f1_all_selected_classes"], (2 / 3 + 1) / 6)
        self.assertIsNone(m["top5_note"])

    def test_marginal_verb_prediction_not_top_action_verb(self):
        classes = [{"action_id": 0, "verb_id": 5, "noun_id": 1},
                   {"action_id": 1, "verb_id": 7, "noun_id": 2},
                   {"action_id": 2, "verb_id": 7, "noun_id": 3}]
        m = compute_metrics([[0.4, 0.3, 0.3]], [1], classes)
        self.assertEqual(m["top1"], 0)
        self.assertEqual(m["verb_top1"], 1)
        self.assertEqual(m["top5_k"], 3)
        self.assertIsNotNone(m["top5_note"])

    def test_metrics_reject_invalid_labels_and_probabilities(self):
        c = [{"action_id": 0, "verb_id": 0, "noun_id": 0}, {"action_id": 1, "verb_id": 1, "noun_id": 1}]
        for p, y in (([[0.5, 0.5]], [2]), ([[2, 1]], [0]), ([[np.nan, 0]], [0])):
            with self.assertRaises(ValueError):
                compute_metrics(p, y, c)

    def test_majority_baseline_selected_from_train(self):
        m = baselines([0, 0, 1], np.array([1, 1, 0]), 2)
        self.assertEqual(m["majority_label_from_train"], 0)
        self.assertAlmostEqual(m["majority_top1"], 1 / 3)

    def test_checkpoint_roundtrip_with_weights_only(self):
        model = TCN(3, 2, width=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            torch.save({"model": model.state_dict(), "provenance": {"torch": str(torch.__version__)}}, path)
            state = torch.load(path, weights_only=True)
            model.load_state_dict(state["model"])


class AcquisitionTests(unittest.TestCase):
    def test_server_ignoring_or_misreporting_range_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_range_headers(200, None, 0, 10, 100)
        with self.assertRaises(RuntimeError):
            validate_range_headers(206, "bytes 0-99/100", 0, 10, 100)
        validate_range_headers(206, "bytes 0-9/100", 0, 10, 100)
    def test_range_budget_refuses_before_fetch(self):
        calls = []
        stream = RangeFile(100, lambda start, count: calls.append(count), max_request=10)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            stream.read(11)
        self.assertFalse(calls)

    def test_truncated_response_rejected(self):
        stream = RangeFile(100, lambda start, count: b"x")
        with self.assertRaisesRegex(RuntimeError, "Truncated"):
            stream.read(10)

    def test_bounded_archive_prefix_and_member_inventory(self):
        archive = io.BytesIO()
        frames = {str(i): {str(h): np.ones((21, 3)).tolist() for h in range(2)} for i in range(5)}
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as writer:
            writer.writestr("assembly101_camera_and_hand_poses/landmarks3D/synthetic.json", json.dumps(frames))
        raw = archive.getvalue()
        stream = RangeFile(len(raw), lambda start, count: raw[start:start + count])
        result = inspect_archive(stream)
        self.assertEqual(len(result["sample_frames"]), 3)
        self.assertFalse(result["full_member_crc_verified"])
        self.assertLess(stream.transferred, 4_000_000)


if __name__ == "__main__":
    unittest.main()
