"""Synthetic fixtures for the read-only results audit."""
import json
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from e0.audit import clock_diagnostics, history_summary, independent_metrics, reconstruct_features
from e0.data import segment_features
from e0.metrics import compute_metrics
from e0.quality import timestamp_cadence
from e0.source_profile import dimensionless_profile


class AuditTests(unittest.TestCase):
    def test_complete_audit_entrypoint_on_synthetic_saved_run_without_optimizer(self):
        import torch
        from e0 import audit
        from e0.model import TCN
        from e0.preprocess import prepare
        from e0.data import sha256
        from test_e0 import fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "data/raw/assembly101-e0"
            raw_root.mkdir(parents=True)
            manifest = fixture(raw_root)
            manifest["pose_profile"] = {**dimensionless_profile(), "status": "synthetic", "synthetic": True}
            for rec, entry in manifest["recordings"].items():
                for kind, value in (("confidence", {str(i): {"0": 1, "1": 1} for i in range(12)}),
                                    ("timestamp", {str(i): 1000 + round(i / 60, 3) for i in range(12)})):
                    path = raw_root / f"{kind}-{rec}.json"
                    path.write_text(json.dumps(value))
                    entry[kind] = {"path": str(path), "sha256": sha256(path)}
            processed = root / "data/processed/synthetic"
            artifact = prepare(manifest, processed, min_train_examples=1, max_classes=2, synthetic=True)
            dataset_path = processed / "dataset.json"
            run = root / "experiments/e0/controlled-baseline"
            for name in ("train", "test", "sanity"):
                (run / name).mkdir(parents=True)
            config = {"epochs": 60, "patience": 10, "class_weighting": "none"}
            provenance = {"dataset_path": str(dataset_path), "dataset_sha256": sha256(dataset_path), "git_commit": "synthetic-only", "config": config}
            (run / "train/provenance.json").write_text(json.dumps(provenance))
            args = {"input_dim": 216, "classes": 2, "width": 4, "dilations": [1], "dropout": 0.}
            model = TCN(**args).eval()
            torch.save({"model": model.state_dict(), "model_args": args, "classes": artifact["classes"],
                        "epoch": 1, "provenance": provenance}, run / "train/best.pt")
            for split in ("validation", "test"):
                rows = [s for s in artifact["segments"] if s["split"] == split]
                features = []
                for row in rows:
                    with np.load(processed / row["path"], allow_pickle=False) as source:
                        features.append(source["features"])
                with torch.no_grad():
                    p = model(torch.from_numpy(np.stack(features))).softmax(1).numpy()
                y = np.array([r["label"] for r in rows])
                metrics = compute_metrics(p, y, artifact["classes"])
                metrics_path = run / ("train/validation_metrics.json" if split == "validation" else "test/metrics.json")
                metrics_path.write_text(json.dumps(metrics))
                if split == "test":
                    np.savez_compressed(run / "test/probabilities.npz", probabilities=p, labels=y)
                    with (run / "test/predictions.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["segment_id", "true_action_id", "predicted_action_id"])
                        writer.writeheader()
                        for row, guess in zip(rows, p.argmax(1)):
                            writer.writerow({"segment_id": row["segment_id"], "true_action_id": row["action_id"],
                                             "predicted_action_id": artifact["classes"][guess]["action_id"]})
            history = [{"epoch": 1, "train": {"loss": 1., "top1": .5}, "validation": {"loss": 1., "top1": .5}}]
            (run / "train/history.jsonl").write_text(json.dumps(history[0]) + "\n")
            (run / "sanity/sanity.json").write_text(json.dumps({"passed": True, "synthetic_fixture": True}))
            output = root / "experiments/e0/audit-synthetic"
            def path(value):
                return root if str(value) in ("/projects/jointposeaction", "/projects/_ssd/jointposeaction") else Path(value)
            with patch.object(audit, "Path", side_effect=path), \
                 patch.dict("os.environ", {"SLURM_JOB_ID": "synthetic"}), \
                 patch("sys.argv", ["audit", "--run", str(run), "--output", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                audit.main()
            report = json.loads((output / "audit.json").read_text())
            self.assertNotIn("failure", report)
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertTrue(report["metrics"]["confusion_matrix_matches_saved"])
            self.assertEqual(report["raw"]["feature_reconstruction_failures"], [])
            self.assertEqual(report["inference"]["test"]["predictions_different_from_saved"], 0)
            self.assertEqual(report["annotations"]["retained_segment_mismatch_count"], 0)
            self.assertTrue(report["annotations"]["training_only_vocabulary_reproduced"])

    def test_independent_metrics_agree_and_expose_imbalance_gap(self):
        y = np.array([0] * 80 + [1] * 10 + [2] * 10)
        p = np.tile([.8, .1, .1], (100, 1))
        independent = independent_metrics(p, y, 3)
        classes = [{"action_id": i, "verb_id": i, "noun_id": i} for i in range(3)]
        production = compute_metrics(p, y, classes)
        self.assertEqual(independent["top1"], .8)
        self.assertAlmostEqual(independent["macro_f1"], production["macro_f1_all_selected_classes"])
        self.assertAlmostEqual(independent["macro_f1"], (2 * .8 / 1.8) / 3)
        self.assertEqual(independent["never_predicted_classes"], [1, 2])
        self.assertEqual(independent["confusion_matrix"], production["confusion_matrix"])

    def test_history_uses_minimum_loss_not_maximum_accuracy(self):
        history = [{"epoch": 1, "train": {}, "validation": {"loss": 2, "top1": .2}},
                   {"epoch": 2, "train": {}, "validation": {"loss": 3, "top1": .3}}]
        result = history_summary(history, {"epochs": 60, "patience": 10, "class_weighting": "none"})
        self.assertEqual(result["minimum_validation_loss_epoch"], 1)
        self.assertEqual(result["maximum_validation_top1_epoch"], 2)

    def test_local_cadence_gate_can_pass_substantial_accumulated_drift(self):
        timestamps = {str(i): 1000 + i / 59 for i in range(3001)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timestamps.json"
            path.write_text(json.dumps(timestamps))
            # This demonstrates a validation gap; it does not assert real data drift.
            self.assertTrue(timestamp_cadence(path, dict.fromkeys(range(3001)))["accepted"])
        diagnostic = clock_diagnostics(timestamps)
        self.assertTrue(diagnostic["passes_original_local_cadence_gate"])
        self.assertTrue(diagnostic["accumulated_drift_over_50ms"])
        self.assertGreater(diagnostic["maximum_accumulated_drift_seconds"], .8)
        ideal = clock_diagnostics({str(i): 1000 + round(i / 60, 3) for i in range(3001)})
        self.assertFalse(ideal["accumulated_drift_over_50ms"])

    def test_independent_features_match_export_contract_with_missing_and_zero_wrist(self):
        raw, frames, confidence, scores = {}, {}, {}, {}
        for i in range(10):
            xyz = np.arange(126).reshape(2, 21, 3).astype(float) + i
            xyz[1, 5] = 0  # Positive score: preserve this valid zero.
            frames[i] = xyz
            raw[str(i)] = {str(h): xyz[h].tolist() for h in range(2)}
            confidence[str(i)] = {"0": 0 if i < 4 else 1, "1": 1}
            scores[i] = np.array([confidence[str(i)]["0"], 1])
        del raw["3"], confidence["3"], frames[3], scores[3]
        row = {"start_frame": 1, "end_frame": 3}
        actual, indices = reconstruct_features(raw, confidence, row, samples=8)
        exported, quality = segment_features(frames, row, dimensionless_profile(), samples=8, confidences=scores)
        np.testing.assert_allclose(actual, exported, atol=1e-6)
        self.assertEqual(indices.tolist(), quality["pose_frame_indices"])
        self.assertTrue((actual[0, :63] == 0).all())


if __name__ == "__main__":
    unittest.main()
