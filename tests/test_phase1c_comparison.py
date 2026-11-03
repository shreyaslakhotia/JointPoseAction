"""Synthetic tests for matched-population reporting and clustered uncertainty."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from phase1c.compare import compare, file_sha, measured_metrics, paired_recording_bootstrap


class ComparisonTests(unittest.TestCase):
    def fixture(self, root):
        classes = [{"label": index, "action_id": index + 10, "verb_id": index,
                    "noun_id": 0, "action_cls": f"action {index}"} for index in range(2)]
        rows = [{"segment_id": f"test:{i}", "recording_id": f"recording-{i // 2}",
                 "participant_id": "shared", "split": "test", "label": i % 2,
                 "action_id": 10 + i % 2} for i in range(6)]
        dataset = root / "dataset.json"
        dataset.write_text(json.dumps({"classes": classes, "segments": rows}))
        return dataset, rows, classes

    def make_run(self, root, dataset, rows, classes, name, predictions, subset=None):
        run = root / name
        (run / "test").mkdir(parents=True)
        (run / "train").mkdir()
        indices = list(range(len(rows))) if subset is None else subset
        probabilities = np.asarray([[0.9, 0.1] if predictions[i] == 0 else [0.1, 0.9] for i in indices])
        labels = np.asarray([rows[i]["label"] for i in indices], dtype=np.int64)
        np.savez_compressed(run / "test/probabilities.npz", probabilities=probabilities,
                            labels=labels, segment_ids=np.asarray([rows[i]["segment_id"] for i in indices]))
        with (run / "test/predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["segment_id", "recording_id", "true_action_id", "predicted_action_id"])
            writer.writeheader()
            for i in indices:
                writer.writerow({"segment_id": rows[i]["segment_id"], "recording_id": rows[i]["recording_id"],
                                 "true_action_id": rows[i]["action_id"], "predicted_action_id": classes[predictions[i]]["action_id"]})
        (run / "test/metrics.json").write_text(json.dumps({"provenance": {"dataset_sha256": file_sha(dataset)}, "class_mapping": classes}))
        (run / "train/history.jsonl").write_text('{"epoch": 1}\n')
        (run / "train/training_completed.json").write_text(json.dumps({"best_epoch": 1, "selection_criterion": "maximum_validation_macro_f1"}))
        return run

    def test_head_class_accuracy_does_not_hide_coverage(self):
        classes = [{"action_id": i, "verb_id": i, "noun_id": 0} for i in range(3)]
        probabilities = np.tile([0.8, 0.1, 0.1], (10, 1))
        measured = measured_metrics(probabilities, np.asarray([0] * 8 + [1, 2]), classes)
        self.assertAlmostEqual(measured["top1"], 0.8)
        self.assertAlmostEqual(measured["balanced_accuracy"], 1 / 3)
        self.assertEqual(measured["classes_never_predicted"], 2)
        self.assertEqual(measured["classes_zero_correct"], 2)
        self.assertAlmostEqual(measured["weighted_f1"], (16 / 18) * 0.8)

    def test_recording_resampling_is_paired_and_reproducible(self):
        recordings = ["a"] * 20 + ["b"] * 2 + ["c"] * 2
        left = [0] * 24
        right = [1] * 20 + [0] * 4
        first = paired_recording_bootstrap(left, right, recordings, repeats=1000, seed=5)
        self.assertEqual(first, paired_recording_bootstrap(left, right, recordings, repeats=1000, seed=5))
        self.assertAlmostEqual(first["difference_right_minus_left"], 20 / 24)
        # Resampling recordings has positive probability of no improved recording.
        self.assertEqual(first["ci95_percentile"][0], 0)
        self.assertEqual(first["ci95_percentile"][1], 1)
        self.assertEqual(first["recordings"], 3)
        same = paired_recording_bootstrap(right, right, recordings, repeats=100)
        self.assertEqual(same["ci95_percentile"], [0, 0])

    def test_reports_measured_matched_examples_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, rows, classes = self.fixture(root)
            original = self.make_run(root, dataset, rows, classes, "original", [0] * 6)
            balanced = self.make_run(root, dataset, rows, classes, "balanced", [0, 1] * 3)
            original_sha = file_sha(original / "test/probabilities.npz")
            report = compare(dataset, [("B0-original", original), ("B0-balanced", balanced)], root / "comparison",
                             repeats=100, expected_dataset_sha=file_sha(dataset))
            self.assertEqual(report["population"]["policy"], "full_frozen_test")
            self.assertEqual(report["models"]["B0-balanced"]["metrics"]["top1"], 1)
            self.assertEqual(report["paired_accuracy_differences"][0]["difference_right_minus_left"], 0.5)
            self.assertEqual(original_sha, file_sha(original / "test/probabilities.npz"))
            self.assertTrue((root / "comparison/per_class.csv").is_file())
            self.assertTrue((root / "comparison/B0_BALANCED_RESULTS.md").is_file())
            self.assertTrue((root / "comparison/FUSION_READINESS.md").is_file())
            self.assertIn("participants overlap", (root / "comparison/DUAL_BACKBONE_COMPARISON.md").read_text())
            with self.assertRaises(FileExistsError):
                compare(dataset, [("B0-original", original), ("B0-balanced", balanced)], root / "comparison",
                        repeats=100, expected_dataset_sha=file_sha(dataset))

    def test_population_mismatch_requires_explicit_common_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, rows, classes = self.fixture(root)
            original = self.make_run(root, dataset, rows, classes, "original", [0] * 6)
            limited = self.make_run(root, dataset, rows, classes, "limited", [0, 1] * 3, subset=[0, 1, 2, 3])
            arguments = (dataset, [("original", original), ("limited", limited)], root / "comparison")
            with self.assertRaisesRegex(ValueError, "Population mismatch"):
                compare(*arguments, repeats=100, expected_dataset_sha=file_sha(dataset))
            with self.assertRaisesRegex(ValueError, "Population mismatch"):
                compare(*arguments, allow_common_subset=True, repeats=100, expected_dataset_sha=file_sha(dataset))
            report = compare(*arguments, allow_common_subset=True, exclusion_reason="Declared synthetic adapter rejection",
                             repeats=100, expected_dataset_sha=file_sha(dataset))
            self.assertEqual(report["population"]["compared_segments"], 4)
            self.assertEqual(report["population"]["excluded_segment_ids_by_model"]["limited"], ["test:4", "test:5"])
            self.assertEqual(report["models"]["original"]["metrics"]["samples"], 4)

    def test_original_e0_exports_without_segment_array_or_class_mapping_are_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, rows, classes = self.fixture(root)
            original = self.make_run(root, dataset, rows, classes, "original", [0] * 6)
            new = self.make_run(root, dataset, rows, classes, "new", [0, 1] * 3)
            # Exact original e0.evaluate.py NPZ schema: no segment_ids array.
            path = original / "test/probabilities.npz"
            with np.load(path, allow_pickle=False) as arrays:
                probabilities, labels = arrays["probabilities"].copy(), arrays["labels"].copy()
            np.savez_compressed(path, probabilities=probabilities, labels=labels)
            metrics = json.loads((original / "test/metrics.json").read_text())
            del metrics["class_mapping"]
            (original / "test/metrics.json").write_text(json.dumps(metrics))
            report = compare(dataset, [("B0-original", original), ("new", new)], root / "comparison",
                             repeats=100, expected_dataset_sha=file_sha(dataset))
            self.assertEqual(report["models"]["B0-original"]["metrics"]["top1"], .5)

    def test_handformer_deliverable_uses_pinned_source_and_reports_recipe_confounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, rows, classes = self.fixture(root)
            original = self.make_run(root, dataset, rows, classes, "original", [0] * 6)
            handformer = self.make_run(root, dataset, rows, classes, "handformer", [0, 1] * 3)
            compare(dataset, [("B0-original", original), ("B1-HandFormer", handformer)], root / "comparison",
                    repeats=100, expected_dataset_sha=file_sha(dataset))
            text = (root / "comparison/B1_HANDFORMER_RESULTS.md").read_text(encoding="utf-8")
            self.assertIn("640a10a6bd03f85842a2662e67cf30444ab1d673", text)
            self.assertIn("cannot be attributed solely to architecture", text)
            self.assertIn("No RGB features or geometry branch", text)

    def test_rejects_changed_dataset_mapping_csv_labels_and_output_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, rows, classes = self.fixture(root)
            first = self.make_run(root, dataset, rows, classes, "first", [0] * 6)
            second = self.make_run(root, dataset, rows, classes, "second", [0, 1] * 3)
            runs = [("first", first), ("second", second)]
            with self.assertRaisesRegex(ValueError, "frozen approved"):
                compare(dataset, runs, root / "out", repeats=100)
            with self.assertRaisesRegex(ValueError, "separate"):
                compare(dataset, runs, first / "comparison", repeats=100, expected_dataset_sha=file_sha(dataset))
            metrics_path = second / "test/metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["class_mapping"] = list(reversed(classes))
            metrics_path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "class ordering"):
                compare(dataset, runs, root / "out", repeats=100, expected_dataset_sha=file_sha(dataset))
            metrics["class_mapping"] = classes
            metrics_path.write_text(json.dumps(metrics))
            csv_path = second / "test/predictions.csv"
            csv_path.write_text(csv_path.read_text().replace("test:0,recording-0,10,10", "test:0,recording-0,11,10"))
            with self.assertRaisesRegex(ValueError, "true label"):
                compare(dataset, runs, root / "out", repeats=100, expected_dataset_sha=file_sha(dataset))


if __name__ == "__main__":
    unittest.main()
