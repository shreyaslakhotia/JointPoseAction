"""Synthetic/mocked unattended-chain tests; no Slurm submission or real training."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from e0 import pipeline
from e0.report import write_report
from e0.sanity import fit, select_examples


class PipelineTests(unittest.TestCase):
    def test_successful_chain_runs_sanity_baseline_then_one_final_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "experiments/e0/controlled-baseline"
            run.mkdir(parents=True)
            pipeline.save(run / "submission.json", {"git_commit": "pinned", "jobs": {"prepare": "1", "train": "2", "evaluate": "3"}})
            calls = []
            def execute(args, **kwargs):
                calls.append(args[2])
                if args[2] == "e0.sanity":
                    (run / "sanity").mkdir()
                    pipeline.save(run / "sanity/sanity.json", {"passed": True})
                elif args[2] == "e0.train":
                    (run / "train").mkdir()
                    pipeline.save(run / "train/training_completed.json", {"best_epoch": 1})
            with patch.object(pipeline, "git_commit", return_value="pinned"), \
                 patch.object(pipeline, "prepared_dataset", return_value=root / "data/processed/dataset.json"), \
                 patch("e0.runtime.load_artifact", return_value={"source_manifest": {"pose_profile": {"feature_units": "dimensionless_wrist_index_length"}}}), \
                 patch.object(pipeline.subprocess, "run", side_effect=execute), \
                 patch("e0.report.write_report") as report:
                with patch.dict("os.environ", {"JPA_E0_COMMIT": "pinned", "SLURM_JOB_ID": "2"}):
                    pipeline.run_stage(root, root / "repo", run, "train")
                self.assertEqual(pipeline.read(run / "status.json")["state"], "training_complete")
                with patch.dict("os.environ", {"JPA_E0_COMMIT": "pinned", "SLURM_JOB_ID": "3"}):
                    pipeline.run_stage(root, root / "repo", run, "evaluate")
            self.assertEqual(calls, ["e0.sanity", "e0.train", "e0.evaluate"])
            report.assert_called_once()
            self.assertEqual(pipeline.read(run / "status.json")["state"], "complete")

    def test_evaluation_requires_training_completion_even_if_sanity_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "experiments/e0/controlled-baseline"
            (run / "sanity").mkdir(parents=True)
            pipeline.save(run / "submission.json", {"git_commit": "pinned", "jobs": {"prepare": "1", "train": "2", "evaluate": "3"}})
            pipeline.save(run / "sanity/sanity.json", {"passed": True})
            with patch.object(pipeline, "git_commit", return_value="pinned"), \
                 patch.object(pipeline, "prepared_dataset", return_value=root / "data/processed/dataset.json"), \
                 patch.dict("os.environ", {"JPA_E0_COMMIT": "pinned", "SLURM_JOB_ID": "3"}), \
                 patch.object(pipeline.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "completion gate"):
                    pipeline.run_stage(root, root / "repo", run, "evaluate")
            execute.assert_not_called()

    def test_attach_orders_dependencies_and_repeated_submission_never_duplicates_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls, jobs = [], iter(("800", "801"))
            def command(args):
                calls.append(args)
                return next(jobs) if args[0] == "sbatch" else ""
            with patch.object(pipeline, "git_commit", return_value="pinned"), \
                 patch.object(pipeline, "validate_preparation_job", return_value="RUNNING"), \
                 patch.object(pipeline, "command", side_effect=command), contextlib.redirect_stdout(io.StringIO()):
                first = pipeline.submit(root, root / "repo", "242798")
                second = pipeline.submit(root, root / "repo", "242798")
            self.assertEqual(first, second)
            submissions = [x for x in calls if x[0] == "sbatch"]
            self.assertEqual(len(submissions), 2)
            self.assertIn("--dependency=afterok:242798", submissions[0])
            self.assertIn("--dependency=afterok:800", submissions[1])
            self.assertTrue(all("--hold" in x and "--kill-on-invalid-dep=yes" in x for x in submissions))
            self.assertEqual([x for x in calls if x[0] == "scontrol"],
                             [["scontrol", "release", "800"], ["scontrol", "release", "801"]])
            self.assertFalse(any("e0_acquire_prepare.sbatch" in x[-1] for x in submissions))

    def test_partial_submission_resumes_missing_evaluation_without_second_gpu_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls, responses = [], iter(("800", subprocess.CalledProcessError(1, "sbatch"), "801"))
            def command(args):
                calls.append(args)
                if args[0] != "sbatch":
                    return ""
                value = next(responses)
                if isinstance(value, Exception):
                    raise value
                return value
            with patch.object(pipeline, "git_commit", return_value="pinned"), \
                 patch.object(pipeline, "validate_preparation_job", return_value="RUNNING"), \
                 patch.object(pipeline, "command", side_effect=command), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.submit(root, root / "repo", "242798")
                self.assertFalse(any(x[0] == "scontrol" for x in calls))
                state = pipeline.submit(root, root / "repo", "242798")
            self.assertEqual(state["jobs"], {"prepare": "242798", "train": "800", "evaluate": "801"})
            self.assertEqual(sum(x[0] == "sbatch" and "e0_pipeline_train.sbatch" in x[-1] for x in calls), 1)

    def test_completed_preparation_uses_ready_report_without_expired_slurm_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls, jobs = [], iter(("800", "801"))
            def command(args):
                calls.append(args)
                return next(jobs) if args[0] == "sbatch" else ""
            with patch.object(pipeline, "git_commit", return_value="pinned"), \
                 patch.object(pipeline, "validate_preparation_job", return_value="COMPLETED"), \
                 patch.object(pipeline, "prepared_dataset", return_value=root / "data/processed/dataset.json") as gate, \
                 patch.object(pipeline, "command", side_effect=command), contextlib.redirect_stdout(io.StringIO()):
                pipeline.submit(root, root / "repo", "242798")
            gate.assert_called_once_with(root, "242798")
            first = next(x for x in calls if x[0] == "sbatch")
            self.assertFalse(any(x.startswith("--dependency") for x in first))

    def test_unready_preparation_prevents_sanity_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "experiments/e0/controlled-baseline"
            run.mkdir(parents=True)
            pipeline.save(run / "submission.json", {"git_commit": "pinned", "jobs": {"prepare": "1", "train": "2"}})
            report = root / "data/annotations/e0-preparation-1/acquisition.json"
            report.parent.mkdir(parents=True)
            pipeline.save(report, {"training_ready": False})
            with patch.object(pipeline, "git_commit", return_value="pinned"), \
                 patch.dict("os.environ", {"JPA_E0_COMMIT": "pinned", "SLURM_JOB_ID": "2"}), \
                 patch.object(pipeline.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "readiness"):
                    pipeline.run_stage(root, root / "repo", run, "train")
            execute.assert_not_called()
            self.assertEqual(pipeline.read(run / "status.json")["state"], "failed")

    def test_failed_sanity_prevents_baseline_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "experiments/e0/controlled-baseline"
            (run / "sanity").mkdir(parents=True)
            pipeline.save(run / "submission.json", {"git_commit": "pinned", "jobs": {"prepare": "1", "train": "2"}})
            pipeline.save(run / "sanity/sanity.json", {"passed": False})
            with patch.object(pipeline, "git_commit", return_value="pinned"), \
                 patch.object(pipeline, "prepared_dataset", return_value=root / "data/processed/dataset.json"), \
                 patch.dict("os.environ", {"JPA_E0_COMMIT": "pinned", "SLURM_JOB_ID": "2"}), \
                 patch("e0.runtime.load_artifact", return_value={"source_manifest": {"pose_profile": {"feature_units": "dimensionless_wrist_index_length"}}}), \
                 patch.object(pipeline.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "Sanity gate"):
                    pipeline.run_stage(root, root / "repo", run, "train")
            self.assertEqual(execute.call_count, 1)
            self.assertIn("e0.sanity", execute.call_args.args[0])

    def test_measured_report_uses_actual_confusions_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for name in ("train", "test", "sanity"):
                (run / name).mkdir()
            dataset = run / "dataset.json"
            pipeline.save(dataset, {"classes": [{"action_cls": "a"}, {"action_cls": "b"}],
                                    "summary": {"split_counts": {"train": 50, "validation": 10, "test": 10}, "exclusions": {}}})
            pipeline.save(run / "submission.json", {"git_commit": "synthetic", "jobs": {"prepare": "1", "train": "2", "evaluate": "3"}})
            pipeline.save(run / "sanity/sanity.json", {"passed": True})
            pipeline.save(run / "train/training_completed.json", {"best_epoch": 4})
            pipeline.save(run / "test/metrics.json", {"top1": .7, "top5": 1., "macro_f1_all_selected_classes": .69,
                "baselines": {"majority_top1": .6, "uniform_chance_top1": .5}, "confusion_matrix": [[3, 1], [2, 4]],
                "per_class": [{"label": 0, "f1": .66, "support": 4}, {"label": 1, "f1": .72, "support": 6}],
                "provenance": {"dataset_sha256": "synthetic-only"}})
            result = write_report(run, dataset)
            self.assertEqual(result["top1"], .7)
            self.assertEqual(result["common_confusions"][0], {"count": 2, "true_action": "b", "predicted_action": "a"})
            self.assertIn("70.00%", (run / "RESULTS.md").read_text(encoding="utf-8"))


class SanityTests(unittest.TestCase):
    def test_sanity_selection_skips_duplicate_feature_vectors(self):
        class Dataset:
            rows = [{"label": i // 3, "segment_id": str(i)} for i in range(6)]
            def __getitem__(self, i):
                return torch.tensor([float((0, 0, 2, 3, 3, 5)[i])]), self.rows[i]["label"], i
        selected = select_examples(Dataset(), classes=2, per_class=2)
        self.assertEqual([x[2] for x in selected], ["0", "2", "3", "5"])

    def test_sanity_optimizer_is_bounded_and_does_not_claim_pass_after_one_step(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 2)
        before = model.weight.detach().clone()
        result = fit(model, torch.tensor([[-1., -1.], [1., 1.]]), torch.tensor([0, 1]), steps=1)
        self.assertEqual(result["steps"], 1)
        self.assertFalse(result["passed"])
        self.assertFalse(torch.equal(before, model.weight))


if __name__ == "__main__":
    unittest.main()
