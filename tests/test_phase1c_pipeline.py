"""Scheduler failure/re-entry, frozen inputs and allocation contract checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

from phase1c import pipeline, protocol


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.captured_stdout = redirect_stdout(io.StringIO())
        self.captured_stdout.__enter__()

    def tearDown(self):
        self.captured_stdout.__exit__(None, None, None)

    def submission_fixture(self, base):
        root = Path(base)
        repo = root / "repo"
        repo.mkdir()
        original = root / "experiments/e0/controlled-baseline/train"
        original.mkdir(parents=True)
        dataset = root / "data/processed/e0/dataset.json"
        dataset.parent.mkdir(parents=True)
        dataset.write_text("{}")
        (original / "provenance.json").write_text(json.dumps({"dataset_path": str(dataset)}))
        return root, repo

    def failed_preparation_fixture(self, root):
        run = root / "experiments/phase1c/recognition-baselines"
        contract = {"dataset_sha256": protocol.DATASET_SHA256, "counts": protocol.SPLIT_COUNTS}
        state = {"git_commit": pipeline.SOURCE_HASH_FAILURE_COMMIT, "protocol": contract,
                 "jobs": {"prepare": "101", "b0": "102", "b1": "103", "evaluate": "104"},
                 "released": list(pipeline.STAGES), "run": str(run),
                 "original": str(root / "experiments/e0/controlled-baseline")}
        protocol.save(run / "submission.json", state)
        protocol.save(run / "status-prepare.json", {"state": "failed", "error": "HandFormer source hash mismatch: HandFormer/utils.py"})
        protocol.save(run / "handformer-cache/compatibility.json", {
            "ready": True, "dataset_sha256": contract["dataset_sha256"], "split_counts": contract["counts"]})
        accounting = "\n".join(f"{job}|user|p1c-{name}|{status}|{elapsed}" for job, name, status, elapsed in (
            ("101", "prepare", "FAILED", "386"), ("102", "b0", "CANCELLED", "0"),
            ("103", "b1", "CANCELLED", "0"), ("104", "eval", "CANCELLED", "0")))
        return run, state, contract, accounting

    def test_known_preparation_recovery_archives_history_and_reuses_one_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root, repo = self.submission_fixture(temp)
            run, previous, contract, accounting = self.failed_preparation_fixture(root)
            def commands(args):
                return accounting if args[0] == "sacct" else ""
            with patch.object(pipeline, "source_commit", return_value="b" * 40), \
                 patch.object(pipeline, "dataset_contract", return_value=contract), \
                 patch.object(pipeline, "active_jobs", return_value=[]), \
                 patch.object(pipeline.getpass, "getuser", return_value="user"), \
                 patch.object(pipeline, "command", side_effect=commands), \
                 patch.object(pipeline, "submit_job", side_effect=["201", "202", "203", "204"]) as submit:
                current = pipeline.submit(root, repo, recover=True)
                self.assertEqual(submit.call_count, 4)
                self.assertEqual(current["recovery"]["previous_jobs"], previous["jobs"])
                self.assertEqual(current["jobs"]["b0"], "202")
                archive = protocol.read(current["recovery"]["history"])
                self.assertEqual(archive["submission"], previous)
                self.assertEqual(archive["slurm_accounting"], accounting)
            with patch.object(pipeline, "source_commit", return_value="b" * 40), \
                 patch.object(pipeline, "dataset_contract", return_value=contract), \
                 patch.object(pipeline, "active_jobs", return_value=[]), \
                 patch.object(pipeline, "command") as command, patch.object(pipeline, "submit_job") as submit:
                self.assertEqual(pipeline.submit(root, repo, recover=True), current)
                submit.assert_not_called()
                command.assert_not_called()
            with self.assertRaisesRegex(ValueError, "already used"):
                pipeline.recover_preparation(run, current, "c" * 40, contract)

    def test_recovery_refuses_started_models_or_different_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _ = self.submission_fixture(temp)
            run, state, contract, accounting = self.failed_preparation_fixture(root)
            (run / "b0/train").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "model run started"):
                pipeline.recover_preparation(run, state, "b" * 40, contract)
            self.assertFalse((run / "recovery-history").exists())
        with tempfile.TemporaryDirectory() as temp:
            root, _ = self.submission_fixture(temp)
            run, state, contract, accounting = self.failed_preparation_fixture(root)
            protocol.save(run / "status-prepare.json", {"state": "failed", "error": "label integrity failure"})
            with self.assertRaisesRegex(ValueError, "differs"):
                pipeline.recover_preparation(run, state, "b" * 40, contract)
            protocol.save(run / "status-prepare.json", {"state": "failed", "error": "HandFormer source hash mismatch: HandFormer/utils.py"})
            with patch.object(pipeline, "command", return_value=accounting.replace("p1c-b0|CANCELLED|0", "p1c-b0|COMPLETED|100")), \
                 patch.object(pipeline.getpass, "getuser", return_value="user"):
                with self.assertRaisesRegex(ValueError, "without execution"):
                    pipeline.recover_preparation(run, state, "b" * 40, contract)

    def test_partial_submission_resumes_without_duplicate_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root, repo = self.submission_fixture(temp)
            with patch.object(pipeline, "source_commit", return_value="a" * 40), \
                 patch.object(pipeline, "dataset_contract", return_value={"frozen": True}), \
                 patch.object(pipeline, "active_jobs", return_value=[]), \
                 patch.object(pipeline, "command", return_value="") as command, \
                 patch.object(pipeline, "submit_job", side_effect=["101", "102", RuntimeError("temporary scheduler failure")]):
                with self.assertRaisesRegex(RuntimeError, "temporary"):
                    pipeline.submit(root, repo)
                command.assert_not_called()  # all partial stages remain held
            path = root / "experiments/phase1c/recognition-baselines/submission.json"
            self.assertEqual(protocol.read(path)["jobs"], {"prepare": "101", "b0": "102"})
            with patch.object(pipeline, "source_commit", return_value="a" * 40), \
                 patch.object(pipeline, "dataset_contract", return_value={"frozen": True}), \
                 patch.object(pipeline, "active_jobs", return_value=["101|PENDING|prepare", "102|PENDING|b0"]), \
                 patch.object(pipeline, "command", return_value="") as command, \
                 patch.object(pipeline, "submit_job", side_effect=["103", "104"]) as submit:
                state = pipeline.submit(root, repo)
                self.assertEqual(submit.call_count, 2)
                self.assertEqual(submit.call_args_list[0].args[2], "101")
                self.assertEqual(submit.call_args_list[1].args[2], "102:103")
                self.assertEqual(state["released"], list(pipeline.STAGES))
                self.assertEqual(command.call_count, 4)
            with patch.object(pipeline, "source_commit", return_value="a" * 40), \
                 patch.object(pipeline, "dataset_contract", return_value={"frozen": True}), \
                 patch.object(pipeline, "active_jobs", return_value=[]), \
                 patch.object(pipeline, "submit_job") as submit, patch.object(pipeline, "command") as command:
                pipeline.submit(root, repo)
                submit.assert_not_called()
                command.assert_not_called()

    def test_capacity_guard_before_any_job_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root, repo = self.submission_fixture(temp)
            with patch.object(pipeline, "source_commit", return_value="a" * 40), \
                 patch.object(pipeline, "dataset_contract", return_value={}), \
                 patch.object(pipeline, "active_jobs", return_value=["1|RUNNING|other", "2|PENDING|other"]), \
                 patch.object(pipeline, "submit_job") as submit:
                with self.assertRaisesRegex(ValueError, "Five-job"):
                    pipeline.submit(root, repo)
                submit.assert_not_called()

    def test_original_fingerprint_detects_artifact_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("train/provenance.json", "train/best.pt", "train/history.jsonl",
                         "train/training_completed.json", "test/metrics.json",
                         "test/probabilities.npz", "test/predictions.csv", "RESULTS.md"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("frozen")
            snapshot = protocol.original_fingerprint(root)
            protocol.assert_original_unchanged(root, snapshot)
            (root / "train/best.pt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                protocol.assert_original_unchanged(root, snapshot)

    def test_gpu_scripts_follow_cluster_resource_rules(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("b0", "b1"):
            script = (root / f"scripts/cluster/phase1c_{name}.sbatch").read_text()
            self.assertIn("--gres=gpu:l40:1", script)
            self.assertNotIn("--cpus-per-task", script)
            self.assertNotIn("--mem", script)
            self.assertIn("--qos=ug\n", script)

    def test_completed_evaluation_can_be_reused_only_with_unchanged_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "best.pt"
            checkpoint.write_text("frozen checkpoint")
            output = root / "test"
            self.assertFalse(pipeline.completed_evaluation(output, checkpoint))
            output.mkdir()
            names = ("metrics.json", "predictions.csv", "probabilities.npz", "per_class.csv")
            for name in names:
                (output / name).write_text("export")
            protocol.save(output / "evaluation_completed.json", {
                "test_evaluated": True, "dataset_sha256": protocol.DATASET_SHA256,
                "checkpoint_sha256": protocol.digest(checkpoint),
                "export_sha256": {name: protocol.digest(output / name) for name in names}})
            self.assertTrue(pipeline.completed_evaluation(output, checkpoint))
            (output / "predictions.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "export changed"):
                pipeline.completed_evaluation(output, checkpoint)

    def test_upstream_setup_uses_committed_manifest_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root, repo = self.submission_fixture(temp)
            source = {"repository": "https://example.test/HandFormer.git", "revision": "f" * 40}
            protocol.save(repo / "reproducibility/upstreams/handformer.json", source)
            with patch.object(pipeline.subprocess, "run") as run, \
                 patch.object(pipeline, "command", side_effect=[source["revision"], ""]):
                target = pipeline.setup_upstream(root, repo)
                self.assertEqual(run.call_args_list[0].args[0][3], source["repository"])
                self.assertEqual(run.call_args_list[1].args[0][-1], source["revision"])
                self.assertEqual(target, root / "upstreams/HandFormer")

    def test_preparation_integrates_real_model_factory_signature(self):
        import torch
        from phase1c import train
        from torch.utils.data import TensorDataset
        with tempfile.TemporaryDirectory() as temp:
            root, repo = self.submission_fixture(temp)
            run = root / "experiments/phase1c/recognition-baselines"
            run.mkdir(parents=True)
            for name in ("b0_balanced", "b1_handformer"):
                protocol.save(repo / f"configs/{name}.json", {"model_family": "tcn"})
            artifact = {"classes": list(range(100)), "summary": {"features": 216}}
            state = {"original": str(root / "experiments/e0/controlled-baseline"),
                     "protocol": {"dataset_path": str(root / "data/processed/e0/dataset.json")}}
            data = TensorDataset(torch.zeros(2, 216), torch.zeros(2), torch.zeros(2))
            with patch("e0.runtime.load_artifact", return_value=artifact), \
                 patch.object(pipeline, "original_fingerprint", return_value={}), \
                 patch.object(pipeline, "assert_original_unchanged"), \
                 patch.object(pipeline, "used_project_bytes", return_value=1000), \
                 patch.object(pipeline, "setup_upstream", return_value=root / "upstreams/HandFormer"), \
                 patch("phase1c.handformer.verify_upstream"), \
                 patch.object(pipeline, "run_python"), \
                 patch.object(train, "build_model", autospec=True, side_effect=lambda config, artifact: torch.nn.Linear(216, 100)), \
                 patch.object(train, "build_dataset", return_value=data):
                pipeline.prepare(root, repo, run, state)
            prepared = protocol.read(run / "prepared.json")
            self.assertTrue(prepared["ready"])
            self.assertEqual(prepared["cpu_smoke"]["b1"]["output_shape"], [2, 100])

    def test_source_verification_fails_before_any_cache_processing(self):
        with tempfile.TemporaryDirectory() as temp:
            root, repo = self.submission_fixture(temp)
            run = root / "experiments/phase1c/recognition-baselines"
            state = {"original": str(root / "experiments/e0/controlled-baseline"),
                     "protocol": {"dataset_path": str(root / "data/processed/e0/dataset.json")}}
            with patch("e0.runtime.load_artifact", return_value={}), \
                 patch.object(pipeline, "original_fingerprint", return_value={}), \
                 patch.object(pipeline, "used_project_bytes", return_value=0), \
                 patch.object(pipeline, "setup_upstream", return_value=root / "upstreams/HandFormer"), \
                 patch("phase1c.handformer.verify_upstream", side_effect=ValueError("source mismatch")), \
                 patch.object(pipeline, "run_python") as processing:
                with self.assertRaisesRegex(ValueError, "source mismatch"):
                    pipeline.prepare(root, repo, run, state)
                processing.assert_not_called()


if __name__ == "__main__":
    unittest.main()
