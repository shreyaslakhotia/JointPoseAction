"""One controlled Slurm dependency chain; login submission uses the standard library."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def command(args):
    return subprocess.check_output(args, text=True).strip()


def git_commit(repo):
    git = ["git", "-c", f"safe.directory={repo.as_posix()}"]
    if command(git + ["status", "--porcelain"]):
        raise ValueError("Commit source changes before submitting a recorded E0 pipeline")
    return command(git + ["rev-parse", "HEAD"])


def validate_preparation_job(job):
    if not re.fullmatch(r"[0-9]+", str(job)):
        raise ValueError("Numeric preparation job ID required")
    rows = command(["sacct", "-X", "-j", str(job), "--noheader", "--parsable2",
                    "--format=User%100,JobName%100,State"]).splitlines()
    if not rows:
        raise ValueError("Preparation job not found in Slurm accounting")
    user, name, state = [x.strip() for x in rows[0].split("|")[:3]]
    if user != getpass.getuser() or name != "e0-prepare" or state not in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "COMPLETED"):
        raise ValueError("Preparation job must be your successful or active e0-prepare job")
    return state


def submit(root, repo, prepare_job=None):
    commit = git_commit(repo)
    run = root / "experiments/e0/controlled-baseline"
    run.mkdir(parents=True, exist_ok=True)
    lock = run / ".submission-lock"
    with lock.open("x") as guard:
        # Exclusive creation is the lock; close the handle so cleanup is portable
        # to Windows as well as Linux. The file stays present until finally.
        guard.close()
        try:
            state_path = run / "submission.json"
            state = read(state_path) if state_path.exists() else {"git_commit": commit, "jobs": {}, "run": str(run)}
            if state["git_commit"] != commit:
                raise ValueError("Existing controlled pipeline pins another commit; review it instead of submitting another baseline")
            jobs = state["jobs"]
            if prepare_job and jobs.get("prepare") and jobs["prepare"] != str(prepare_job):
                raise ValueError("Existing pipeline is attached to a different preparation job")
            if "prepare" not in jobs:
                if prepare_job is None:
                    active = command(["squeue", "--me", "-h", "-n", "e0-prepare", "-o", "%i"]).splitlines()
                    if len(active) > 1:
                        raise ValueError("Multiple preparation jobs; supply --prepare-job explicitly")
                    prepare_job = active[0].strip() if active else None
                if prepare_job:
                    validate_preparation_job(prepare_job)
                    jobs["prepare"] = str(prepare_job)
                else:
                    jobs["prepare"] = submit_job(repo / "scripts/cluster/e0_acquire_prepare.sbatch")
                save(state_path, state)
            common = {"JPA_E0_PIPELINE_DIR": str(run), "JPA_E0_PREP_JOB": jobs["prepare"], "JPA_E0_COMMIT": commit}
            if "train" not in jobs:
                completed = validate_preparation_job(jobs["prepare"]) == "COMPLETED"
                if completed:
                    prepared_dataset(root, jobs["prepare"])
                # Completed jobs can age out of Slurm's dependency cache. Their
                # durable ready report remains a gate checked again at execution.
                state["preparation_completed_at_submission"] = completed
                jobs["train"] = submit_job(repo / "scripts/cluster/e0_pipeline_train.sbatch", None if completed else jobs["prepare"], common, hold=True)
                save(state_path, state)
            if "evaluate" not in jobs:
                jobs["evaluate"] = submit_job(repo / "scripts/cluster/e0_pipeline_evaluate.sbatch", jobs["train"], common, hold=True)
                save(state_path, state)
            # Release only after all stage IDs are durably recorded; prevents a
            # fast dependency from starting before its handoff record exists.
            if not (run / "status.json").exists():
                save(run / "status.json", {"stage": "preparation_or_queue", "state": "submitted", "jobs": jobs})
            released = state.setdefault("released", [])
            for stage in ("train", "evaluate"):
                if stage not in released:
                    command(["scontrol", "release", jobs[stage]])
                    released.append(stage)
                    save(state_path, state)
            print(json.dumps(state, indent=2))
            return state
        finally:
            lock.unlink()


def submit_job(script, dependency=None, exports=None, hold=False):
    args = ["sbatch", "--parsable"]
    if hold:
        args.append("--hold")
    if dependency:
        args += [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
    if exports:
        args += ["--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items())]
    reply = command(args + [str(script)])
    job = reply.split(";")[0]
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("Unexpected sbatch reply; inspect scheduler state before retrying")
    return job


def prepared_dataset(root, job):
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("Invalid preparation ID")
    report = read(root / f"data/annotations/e0-preparation-{job}/acquisition.json")
    if report.get("training_ready") is not True or "failure" in report:
        raise ValueError("Preparation did not pass all dataset readiness checks")
    path = Path(report["dataset"]).resolve()
    if not path.is_relative_to(root / "data/processed"):
        raise ValueError("Prepared dataset must stay within this project's processed data")
    return path


def run_stage(root, repo, run, stage):
    if not run.resolve().is_relative_to(root / "experiments/e0"):
        raise ValueError("Pipeline outputs must stay inside dedicated E0 experiments")
    state = read(run / "submission.json")
    if git_commit(repo) != state["git_commit"] or os.environ.get("JPA_E0_COMMIT") != state["git_commit"]:
        raise ValueError("Source commit changed since pipeline submission")
    expected_job = state["jobs"]["train" if stage == "train" else "evaluate"]
    if os.environ.get("SLURM_JOB_ID") != expected_job:
        raise ValueError("Stage must run inside its recorded Slurm allocation")
    status = {"stage": stage, "state": "running", "slurm_job_id": expected_job}
    save(run / "status.json", status)
    try:
        dataset = prepared_dataset(root, state["jobs"]["prepare"])
        if stage == "train":
            from .runtime import load_artifact
            artifact = load_artifact(dataset)
            if artifact["source_manifest"]["pose_profile"].get("feature_units") != "dimensionless_wrist_index_length":
                raise ValueError("Unexpected prepared feature protocol")
            subprocess.run([sys.executable, "-m", "e0.sanity", "--dataset", str(dataset),
                            "--config", str(repo / "configs/e0_tcn.json"), "--output", str(run / "sanity")], check=True)
            if read(run / "sanity/sanity.json").get("passed") is not True:
                raise ValueError("Sanity gate did not pass")
            subprocess.run([sys.executable, "-m", "e0.train", "--dataset", str(dataset),
                            "--config", str(repo / "configs/e0_tcn.json"), "--output", str(run / "train")], check=True)
            if not (run / "train/training_completed.json").is_file():
                raise ValueError("Training completion marker missing")
            status.update(state="training_complete", dataset=str(dataset), checkpoint=str(run / "train/best.pt"))
        else:
            if not (run / "train/training_completed.json").is_file() or read(run / "sanity/sanity.json").get("passed") is not True:
                raise ValueError("Training/sanity completion gate missing")
            subprocess.run([sys.executable, "-m", "e0.evaluate", "--dataset", str(dataset),
                            "--checkpoint", str(run / "train/best.pt"), "--output", str(run / "test")], check=True)
            from .report import write_report
            write_report(run, dataset)
            status.update(state="complete", dataset=str(dataset), results=str(run / "RESULTS.md"))
    except Exception as error:
        status.update(state="failed", error_type=type(error).__name__)
        raise
    finally:
        save(run / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("submit")
    p.add_argument("--prepare-job")
    p = sub.add_parser("stage")
    p.add_argument("stage", choices=("train", "evaluate"))
    root = Path("/projects/jointposeaction").resolve()
    args = parser.parse_args()
    if root != Path("/projects/_ssd/jointposeaction"):
        raise RuntimeError("Dedicated project root required")
    repo = root / "repo"
    if args.action == "submit":
        submit(root, repo, args.prepare_job)
    else:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Stage execution requires a Slurm job")
        run_stage(root, repo, Path(os.environ["JPA_E0_PIPELINE_DIR"]), args.stage)


if __name__ == "__main__":
    main()
