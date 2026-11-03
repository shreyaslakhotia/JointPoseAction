"""Synthetic CPU tests of the controlled training/evaluation contract."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from e0.data import sha256
from e0.model import TCN
from e0.source_profile import dimensionless_profile
from phase1c import train
from phase1c.evaluate import evaluate_experiment
from phase1c.metrics import compute_metrics


def classes():
    return [{"label": i, "action_id": 10 + i, "verb_id": i, "noun_id": i,
             "action_cls": f"synthetic action {i}"} for i in range(2)]


def synthetic_artifact(root):
    rows = []
    for split in ("train", "validation", "test"):
        for label in range(2):
            for example in range(2):
                path = root / f"{split}-{label}-{example}.npz"
                x = np.full((8, 2), label * 2 - 1, dtype=np.float32)
                x[:, 1] += np.linspace(0, .1 + example / 10, 8, dtype=np.float32)
                np.savez_compressed(path, features=x)
                rows.append({"segment_id": f"{split}:{label}:{example}", "split": split,
                             "label": label, "action_id": label + 10, "verb_id": label, "noun_id": label,
                             "recording_id": f"synthetic-{split}", "participant_id": f"person-{split}",
                             "path": path.name, "sha256": sha256(path), "quality": {"valid_local_fraction": 1.0}})
    artifact = {"ready_for_training": True, "source_manifest": {"status": "synthetic", "pose_profile": dimensionless_profile()},
                "classes": classes(), "segments": rows, "summary": {"features": 2, "samples_per_segment": 8,
                "leakage": {"participant_policy": "report_official_overlap"},
                "split_counts": {s: 4 for s in ("train", "validation", "test")}}}
    path = root / "dataset.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path, artifact


def tiny_config(dataset_path):
    return {"model_family": "tcn", "expected_dataset_sha256": sha256(dataset_path), "seed": 42,
            "batch_size": 4, "epochs": 2, "patience": 2, "learning_rate": .001, "weight_decay": .0001,
            "optimizer": {"name": "AdamW"}, "scheduler": {"name": "cosine"},
            "gradient_clip": 1.0, "class_weighting": "inverse_sqrt_frequency", "selection": "validation_macro_f1",
            "sanity": {"classes": 2, "per_class": 2, "steps": 1},
            "model": {"width": 4, "dilations": [1], "dropout": 0.0}}


class Phase1CTrainingTests(unittest.TestCase):
    def test_weights_are_training_only_sqrt_smoothed_and_normalized(self):
        rows = [{"split": "train", "label": 0}] * 100 + [{"split": "train", "label": 1}] * 25
        counts, weights = train.class_weights_from_training(rows, 2, "inverse_sqrt_frequency")
        self.assertEqual(counts.tolist(), [100, 25])
        np.testing.assert_allclose(weights.numpy(), [2 / 3, 4 / 3], rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "training rows exclusively"):
            train.class_weights_from_training(rows + [{"split": "test", "label": 1}], 2, "inverse_sqrt_frequency")
        with self.assertRaisesRegex(ValueError, "training support"):
            train.class_weights_from_training(rows[:100], 2, "inverse_sqrt_frequency")

    def test_weighted_objective_and_plain_ce_have_separate_correct_denominators(self):
        model = torch.nn.Identity()
        logits = torch.tensor([[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]])
        labels = torch.tensor([0, 0, 1])
        loader = DataLoader(TensorDataset(logits, labels, torch.arange(3)), batch_size=2)
        weights = torch.tensor([.5, 2.])
        result, _, _ = train.epoch(model, loader, "cpu", classes(), class_weights=weights)
        losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        self.assertAlmostEqual(result["cross_entropy"], float(losses.mean()), places=6)
        self.assertAlmostEqual(result["optimization_loss"], float((losses * weights[labels]).sum() / weights[labels].sum()), places=6)
        self.assertNotEqual(result["cross_entropy"], result["optimization_loss"])

    def test_published_smoothed_temperature_objective_matches_official_formula(self):
        logits = torch.tensor([[.5, -.5], [-.2, .7]])
        labels = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(logits, labels, torch.arange(2)), batch_size=2)
        result, probabilities, _ = train.epoch(torch.nn.Identity(), loader, "cpu", classes(),
                                              loss_options={"label_smoothing": .1, "temperature": 1.1})
        logp = torch.nn.functional.log_softmax(logits / 1.1, dim=-1)
        expected = (.9 * -logp[torch.arange(2), labels] + .1 * -logp.mean(-1)).mean()
        self.assertAlmostEqual(result["optimization_loss"], float(expected), places=6)
        np.testing.assert_allclose(probabilities, logits.softmax(1).numpy(), rtol=1e-6)

    def test_metrics_expose_class_collapse_hidden_by_micro_accuracy(self):
        p = np.array([[.9, .1]] * 10)
        y = np.array([0] * 9 + [1])
        metrics = compute_metrics(p, y, classes())
        self.assertAlmostEqual(metrics["top1"], .9)
        self.assertAlmostEqual(metrics["balanced_accuracy"], .5)
        self.assertEqual(metrics["classes_never_predicted"], 1)
        self.assertEqual(metrics["classes_zero_correct"], 1)
        self.assertAlmostEqual(metrics["weighted_f1"], .9 * 18 / 19)
        self.assertEqual(metrics["per_class"][1]["predicted"], 0)

    def test_b0_major_settings_and_architecture_match_frozen_original(self):
        root = Path(__file__).resolve().parents[1]
        original = json.loads((root / "configs/e0_tcn.json").read_text())
        balanced = json.loads((root / "configs/b0_balanced.json").read_text())
        for key in ("seed", "batch_size", "epochs", "patience", "learning_rate", "weight_decay", "model"):
            self.assertEqual(original[key], balanced[key])
        artifact = {"summary": {"features": 216}, "classes": list(range(100))}
        model = train.build_model(balanced, artifact)
        self.assertIsInstance(model, TCN)
        self.assertEqual(sum(x.numel() for x in model.parameters()), 95268)
        self.assertEqual(balanced["selection"], "validation_macro_f1")

    def test_upstream_warmup_and_multistep_indices_are_preserved(self):
        model = torch.nn.Linear(2, 2)
        config = {"learning_rate": .025, "weight_decay": .0005,
                  "optimizer": {"name": "SGD", "momentum": .9, "nesterov": True},
                  "scheduler": {"name": "warmup_multistep", "warmup_epochs": 5, "milestones": [25, 40], "gamma": .1}}
        optimizer = train.make_optimizer(model, config)
        for index, expected in ((0, .005), (4, .025), (24, .025), (25, .0025), (40, .00025)):
            train.set_epoch_learning_rate(optimizer, config, index)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)

    def test_failed_sanity_stops_before_optimizer_baseline_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = synthetic_artifact(root)
            config = tiny_config(path)
            with patch.object(train, "provenance", return_value={"git_commit": "synthetic", "dataset_sha256": sha256(path)}), \
                 patch.object(train, "bounded_sanity", return_value={"passed": False}), \
                 patch.object(train, "make_optimizer") as optimizer:
                with self.assertRaisesRegex(RuntimeError, "sanity gate failed"):
                    train.fit_experiment(path, config, root / "train", device="cpu", allow_synthetic=True)
                optimizer.assert_not_called()
            self.assertFalse((root / "train/best.pt").exists())

    def test_synthetic_fit_and_final_evaluation_freeze_population_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, artifact = synthetic_artifact(root)
            config = tiny_config(path)
            seen_splits = []
            original_factory = train.build_dataset
            def factory(*args, **kwargs):
                seen_splits.append(args[2])
                return original_factory(*args, **kwargs)
            def record(*args):
                return {"git_commit": "synthetic-test-only", "dataset_sha256": sha256(path), "config": args[1]}
            with patch.object(train, "provenance", side_effect=record), \
                 patch.object(train, "bounded_sanity", return_value={"passed": True, "seconds": 0}), \
                 patch.object(train, "build_dataset", side_effect=factory):
                completed = train.fit_experiment(path, config, root / "train", device="cpu", allow_synthetic=True)
            self.assertNotIn("test", seen_splits)
            self.assertFalse(completed["test_evaluated"])
            checkpoint = torch.load(root / "train/best.pt", map_location="cpu", weights_only=True)
            history = [json.loads(x) for x in (root / "train/history.jsonl").read_text().splitlines()]
            expected_best = max(history, key=lambda row: row["validation"]["macro_f1_all_selected_classes"])
            self.assertEqual(checkpoint["epoch"], expected_best["epoch"])
            with patch("phase1c.evaluate.provenance", side_effect=record):
                metrics = evaluate_experiment(path, root / "train/best.pt", root / "test", allow_synthetic=True, plots=False)
            self.assertEqual(metrics["samples"], 4)
            self.assertEqual(metrics["class_mapping"], artifact["classes"])
            with np.load(root / "test/probabilities.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["probabilities"].shape, (4, 2))
                self.assertEqual(archive["segment_ids"].tolist(), [row["segment_id"] for row in artifact["segments"] if row["split"] == "test"])
            with patch.object(train, "provenance", side_effect=record):
                with self.assertRaises(FileExistsError):
                    train.fit_experiment(path, config, root / "train", device="cpu", allow_synthetic=True)
            checkpoint["classes"][0]["action_id"] = 999
            torch.save(checkpoint, root / "train/tampered.pt")
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                evaluate_experiment(path, root / "train/tampered.pt", root / "rejected", allow_synthetic=True, plots=False)


if __name__ == "__main__":
    unittest.main()
