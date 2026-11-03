"""Four-job Slurm workflow: prepare -> two bounded model runs -> evaluation."""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .protocol import (DATASET_SHA256, ALLOCATION_BYTES, TRAINING_RESERVE_BYTES,
                       assert_original_unchanged, dataset_contract, digest,
                       original_fingerprint, read, save)

STAGES = ("prepare", "b0", "b1", "evaluate")
SOURCE_HASH_FAILURE_COMMIT = "78e4bbfd34cc55eabe6c41cd857a5e88a03ffa2d"


def command(args):
    return subprocess.check_output(args, text=True).strip()


def source_commit(repo):
    git = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    if command(git + ["status", "--porcelain"]):
        raise ValueError("Commit source changes before submitting Phase 1C")
    return command(git + ["rev-parse", "HEAD"])


def active_jobs():
    return command(["squeue", "--me", "--noheader", "--format=%A|%T|%j"]).splitlines()


def submit_job(script, exports, dependency=None):
    args = ["sbatch", "--parsable", "--hold"]
    if dependency:
        args += [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
    args += ["--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()), str(script)]
    response = command(args)
    job = response.split(";")[0]
    if not re.fullmatch(r"\d+", job):
        raise ValueError("Unexpected sbatch reply; inspect scheduler before retrying")
    return job


def recover_preparation(run, state, commit, contract):
    """One explicit non-scientific recovery, only before either model has run."""
    if state.get("recovery"):
        if state["git_commit"] != commit or state["protocol"] != contract:
            raise ValueError("Preparation recovery already used; inspect instead of retrying again")
        return state  # Repeated invocation resumes the same recorded recovery.
    if state["git_commit"] != SOURCE_HASH_FAILURE_COMMIT or state["protocol"] != contract or set(state["jobs"]) != set(STAGES):
        raise ValueError("Recovery is restricted to the diagnosed source-hash preparation failure")
    failed_status = read(run / "status-prepare.json")
    if failed_status.get("state") != "failed" or failed_status.get("error") != "HandFormer source hash mismatch: HandFormer/utils.py":
        raise ValueError("Preparation failure differs from the diagnosed line-ending issue")
    if (run / "prepared.json").exists() or any((run / name / "train").exists() for name in ("b0", "b1")):
        raise ValueError("Recovery cannot replace preparation after a model run started")
    accounting = command(["sacct", "-X", "-j", ",".join(state["jobs"][s] for s in STAGES),
                          "--noheader", "--parsable2",
                          "--format=JobIDRaw,User%100,JobName%100,State,ElapsedRaw"])
    rows = {}
    for line in accounting.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) < 5 or fields[0] in rows:
            raise ValueError("Unexpected Slurm accounting rows during recovery")
        rows[fields[0]] = fields[1:5]
    expected_names = {"prepare": "p1c-prepare", "b0": "p1c-b0", "b1": "p1c-b1", "evaluate": "p1c-eval"}
    if set(rows) != set(state["jobs"].values()):
        raise ValueError("All previous Phase 1C jobs must be accounted for before recovery")
    for stage, job in state["jobs"].items():
        user, name, status, elapsed = rows[job]
        if user != getpass.getuser() or name != expected_names[stage]:
            raise ValueError("Previous job ownership/name mismatch")
        if stage == "prepare":
            if status != "FAILED":
                raise ValueError("Only failed preparation can be recovered")
        elif not status.startswith("CANCELLED") or elapsed != "0":
            raise ValueError("Dependent jobs must have been cancelled without execution")
    cache = read(run / "handformer-cache/compatibility.json")
    if cache.get("ready") is not True or cache.get("dataset_sha256") != contract["dataset_sha256"] or cache.get("split_counts") != contract["counts"]:
        raise ValueError("Completed cache must match the unchanged frozen dataset")
    if len(active_jobs()) + len(STAGES) > 5:
        raise ValueError("Five-job submission limit: wait for four free slots before recovery")
    archive = run / "recovery-history" / f"preparation-{state['jobs']['prepare']}.json"
    record = {"submission": state, "failed_status": failed_status, "slurm_accounting": accounting,
              "reason": "Windows_CRLF_fingerprints_vs_canonical_LF_source"}
    if archive.exists():
        if read(archive)["submission"] != state:
            raise ValueError("Existing recovery history disagrees with the failed submission")
    else:
        save(archive, record)
    return {**state, "git_commit": commit, "jobs": {}, "released": [],
            "recovery": {"attempt": 1, "reason": record["reason"], "previous_commit": state["git_commit"],
                         "previous_jobs": state["jobs"], "history": str(archive),
                         "cached_preparation_reused": True,
                         "created_utc": datetime.now(timezone.utc).isoformat()}}


def submit(root, repo, recover=False):
    """Submission reads only small metadata and does not run CPU preprocessing."""
    commit = source_commit(repo)
    run = root / "experiments/phase1c/recognition-baselines"
    original = root / "experiments/e0/controlled-baseline"
    dataset = Path(read(original / "train/provenance.json")["dataset_path"]).resolve()
    if not dataset.is_relative_to(root / "data/processed"):
        raise ValueError("Dataset must remain in the dedicated project's processed data")
    contract = dataset_contract(dataset)
    run.mkdir(parents=True, exist_ok=True)
    lock = run / ".submission-lock"
    # Exclusive creation protects repeated user execution; it does not consume a job.
    with lock.open("x") as stream:
        stream.close()
        try:
            path = run / "submission.json"
            if recover and not path.exists():
                raise ValueError("Recovery requires an existing failed submission")
            state = read(path) if path.exists() else {
                "git_commit": commit, "root": str(root), "run": str(run),
                "original": str(original), "protocol": contract,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "jobs": {}, "released": [], "training_runs_authorized": ["b0", "b1"]}
            if recover:
                state = recover_preparation(run, state, commit, contract)
            if state["git_commit"] != commit or state["protocol"] != contract:
                raise ValueError("Recorded Phase 1C pins another source/dataset; inspect it rather than submitting duplicates")
            missing = [name for name in STAGES if name not in state["jobs"]]
            queue = active_jobs()
            if len(queue) + len(missing) > 5:
                raise ValueError(f"Five-job submission limit: {len(queue)} jobs active, {len(missing)} stages still needed. No new jobs submitted; rerun when slots are free.")
            exports = {"JPA_PHASE1C_RUN": str(run), "JPA_PHASE1C_COMMIT": commit}
            save(path, state)
            for stage in missing:
                dependency = None
                if stage in ("b0", "b1"):
                    dependency = state["jobs"]["prepare"]
                elif stage == "evaluate":
                    dependency = ":".join(state["jobs"][name] for name in ("b0", "b1"))
                state["jobs"][stage] = submit_job(
                    repo / f"scripts/cluster/phase1c_{stage}.sbatch", exports, dependency)
                save(path, state)
            # No stage can execute until every ID is durable. Partial submissions
            # stay held and resume without resubmitting any recorded training job.
            for stage in STAGES:
                if stage not in state["released"]:
                    command(["scontrol", "release", state["jobs"][stage]])
                    state["released"].append(stage)
                    save(path, state)
            print(json.dumps(state, indent=2))
            return state
        finally:
            lock.unlink()


def used_project_bytes(root):
    # Resolve the storage-manager symlink: du on the link itself reports 512 B.
    return int(command(["du", "-sk", str(root.resolve())]).split()[0]) * 1024


def run_python(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True)


def completed_evaluation(output, checkpoint, dataset_sha=DATASET_SHA256):
    """Reuse only complete hash-verified exports when a later report stage failed."""
    output = Path(output)
    marker = output / "evaluation_completed.json"
    if not marker.exists():
        return False
    record = read(marker)
    if record.get("test_evaluated") is not True or record.get("dataset_sha256") != dataset_sha or record.get("checkpoint_sha256") != digest(checkpoint):
        raise ValueError("Completed evaluation does not match the frozen checkpoint/dataset")
    names = ("metrics.json", "predictions.csv", "probabilities.npz", "per_class.csv")
    if set(record.get("export_sha256", {})) != set(names):
        raise ValueError("Evaluation export hash record is incomplete")
    if any(digest(output / name) != record["export_sha256"][name] for name in names):
        raise ValueError("Completed evaluation export changed")
    return True


def setup_upstream(root, repo):
    source = read(repo / "reproducibility/upstreams/handformer.json")
    path = root / "upstreams/HandFormer"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Source only, never data/features. Download size is logged after cloning.
        subprocess.run(["git", "clone", "--no-checkout", source["repository"], str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", source["revision"]], check=True)
    git = ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path)]
    if command(git + ["rev-parse", "HEAD"]) != source["revision"] or command(git + ["status", "--porcelain"]):
        raise ValueError("HandFormer checkout must be clean and at the pinned revision")
    print(json.dumps({"upstream_path": str(path), "upstream_commit": source["revision"],
                      "source_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file())}), flush=True)
    return path


def prepare(root, repo, run, state):
    from e0.runtime import load_artifact
    from .train import build_model, build_dataset
    from .handformer import verify_upstream
    import torch
    dataset = state["protocol"]["dataset_path"]
    artifact = load_artifact(dataset)
    snapshot = original_fingerprint(Path(state["original"]))
    snapshot_path = run / "original_fingerprint.json"
    if snapshot_path.exists() and read(snapshot_path) != snapshot:
        raise ValueError("Original baseline changed since Phase 1C preparation began")
    save(snapshot_path, snapshot)
    before = used_project_bytes(root)
    if before + TRAINING_RESERVE_BYTES >= ALLOCATION_BYTES:
        raise ValueError("Dedicated 50 GB allocation lacks training/report headroom")
    upstream = setup_upstream(root, repo)
    # Detect source portability/integrity problems before a long cache pass.
    verify_upstream(upstream)
    config_paths = {"b0": repo / "configs/b0_balanced.json", "b1": repo / "configs/b1_handformer.json"}
    # Source path belongs in runtime compatibility, never in the committed model recipe.
    os.environ["JPA_HANDFORMER_ROOT"] = str(upstream)
    cache = run / "handformer-cache"
    run_python("phase1c.handformer_data", "--dataset", dataset, "--output", cache,
               "--project-root", root, "--max-cache-bytes", 6_000_000_000)
    compatibility = cache / "compatibility.json"
    # No test inference in preparation. CPU import, shapes and backward compatibility
    # are verified on two training examples for each unchanged recognition backbone.
    smoke = {}
    for name, config_path in config_paths.items():
        cfg = read(config_path)
        model = build_model(cfg, artifact)
        data = build_dataset(dataset, artifact, "train", cfg, compatibility=str(compatibility))
        x = torch.stack([data[i][0] for i in range(2)])
        model.eval()
        logits = model(x)
        if logits.shape != (2, 100) or not torch.isfinite(logits).all():
            raise ValueError(f"{name} CPU forward compatibility failed")
        logits.square().mean().backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError(f"{name} CPU backward compatibility failed")
        smoke[name] = {"input_shape": list(x.shape), "output_shape": list(logits.shape),
                       "parameters": sum(p.numel() for p in model.parameters())}
        del model, data, logits, x
    after = used_project_bytes(root)
    if after + TRAINING_RESERVE_BYTES > ALLOCATION_BYTES:
        raise ValueError("Prepared cache leaves insufficient space for controlled training")
    assert_original_unchanged(state["original"], snapshot)
    save(run / "prepared.json", {"ready": True, "dataset_sha256": DATASET_SHA256,
         "compatibility": str(compatibility), "upstream": str(upstream), "cpu_smoke": smoke,
         "used_bytes_before": before, "used_bytes_after": after,
         "allocation_bytes": ALLOCATION_BYTES, "training_reserve_bytes": TRAINING_RESERVE_BYTES,
         "config_sha256": {name: digest(path) for name, path in config_paths.items()}})


def stage(root, repo, run, name):
    if not run.resolve().is_relative_to(root / "experiments/phase1c"):
        raise ValueError("Unexpected Phase 1C output location")
    state = read(run / "submission.json")
    if source_commit(repo) != state["git_commit"] or os.environ.get("JPA_PHASE1C_COMMIT") != state["git_commit"]:
        raise ValueError("Source commit changed after submission; no scientific run started")
    if os.environ.get("SLURM_JOB_ID") != state["jobs"][name]:
        raise ValueError("Stage must run in its recorded Slurm allocation")
    dataset_contract(state["protocol"]["dataset_path"])
    status_path = run / f"status-{name}.json"
    status = {"stage": name, "state": "running", "job": state["jobs"][name],
              "started_utc": datetime.now(timezone.utc).isoformat()}
    save(status_path, status)
    try:
        if name == "prepare":
            prepare(root, repo, run, state)
        else:
            prepared = read(run / "prepared.json")
            if prepared.get("ready") is not True or prepared["dataset_sha256"] != DATASET_SHA256:
                raise ValueError("CPU preparation gate missing")
            original_snapshot = read(run / "original_fingerprint.json")
            assert_original_unchanged(state["original"], original_snapshot)
            os.environ["JPA_HANDFORMER_ROOT"] = prepared["upstream"]
            from .handformer_data import load_compatibility
            load_compatibility(prepared["compatibility"], state["protocol"]["dataset_path"],
                               read(state["protocol"]["dataset_path"]), verify_hashes=True)
            common = ["--dataset", state["protocol"]["dataset_path"],
                      "--compatibility", prepared["compatibility"]]
            if name in ("b0", "b1"):
                cfg = repo / ("configs/b0_balanced.json" if name == "b0" else "configs/b1_handformer.json")
                if digest(cfg) != prepared["config_sha256"][name]:
                    raise ValueError("Training configuration changed after preparation")
                run_python("phase1c.train", *common, "--config", cfg,
                           "--output", run / name / "train")
            else:
                for model in ("b0", "b1"):
                    if not (run / model / "train/training_completed.json").is_file():
                        raise ValueError(f"{model} training completion marker missing")
                    checkpoint, output = run / model / "train/best.pt", run / model / "test"
                    if not completed_evaluation(output, checkpoint):
                        run_python("phase1c.evaluate", *common,
                                   "--checkpoint", checkpoint, "--output", output)
                run_python("phase1c.compare", "--dataset", state["protocol"]["dataset_path"],
                           "--original", state["original"], "--balanced", run / "b0",
                           "--handformer", run / "b1", "--output", run / "comparison")
            assert_original_unchanged(state["original"], original_snapshot)
        status["state"] = "complete"
    except Exception as error:
        status.update(state="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--recover-preparation", action="store_true",
                               help="One recovery of the diagnosed CRLF/LF source-hash failure; reuses completed cache")
    command_parser = sub.add_parser("stage")
    command_parser.add_argument("name", choices=STAGES)
    args = parser.parse_args()
    root = Path("/projects/jointposeaction").resolve()
    if root != Path("/projects/_ssd/jointposeaction"):
        raise RuntimeError("Dedicated jointposeaction SSD root required")
    repo = root / "repo"
    if args.command == "submit":
        submit(root, repo, recover=args.recover_preparation)
    else:
        stage(root, repo, Path(os.environ["JPA_PHASE1C_RUN"]), args.name)


if __name__ == "__main__":
    main()
