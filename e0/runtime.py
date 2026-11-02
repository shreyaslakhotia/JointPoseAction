"""Prepared artifact loader and reproducibility helpers shared by train/evaluate."""
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import SPLITS, check_leakage, require_profile, sha256


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_artifact(path, allow_synthetic=False):
    path = Path(path).resolve()
    artifact = json.loads(path.read_text())
    if artifact.get("ready_for_training") is not True:
        raise ValueError("Preprocessing did not finish its readiness checks")
    source = artifact["source_manifest"]
    synthetic = source.get("status") == "synthetic"
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic data cannot be reported as a real E0 run")
    require_profile(source["pose_profile"], allow_unverified=synthetic and allow_synthetic)
    if source.get("status") not in (["validated", "synthetic"] if allow_synthetic else ["validated"]):
        raise ValueError("Unvalidated dataset")
    classes = artifact["classes"]
    if len(classes) < 2 or [x["label"] for x in classes] != list(range(len(classes))):
        raise ValueError("Class labels must be contiguous with at least two classes")
    segments = artifact["segments"]
    if len({x["segment_id"] for x in segments}) != len(segments):
        raise ValueError("Duplicate segment identifiers")
    check_leakage(segments, artifact["summary"]["leakage"]["participant_policy"])
    for segment in segments:
        if segment["split"] not in SPLITS or not 0 <= segment["label"] < len(classes):
            raise ValueError("Invalid split/label")
        expected = classes[segment["label"]]
        if any(segment[k] != expected[k] for k in ("action_id", "verb_id", "noun_id")):
            raise ValueError("Segment does not match label vocabulary")
        target = (path.parent / segment["path"]).resolve()
        if not target.is_relative_to(path.parent) or sha256(target) != segment["sha256"]:
            raise ValueError("Sample path/hash integrity failure")
    if any(not any(x["split"] == split for x in segments) for split in SPLITS):
        raise ValueError("Empty dataset split")
    return artifact


class SegmentDataset(Dataset):
    def __init__(self, path, artifact, split):
        self.root = Path(path).resolve().parent
        self.rows = [x for x in artifact["segments"] if x["split"] == split]
        self.shape = (artifact["summary"]["samples_per_segment"], artifact["summary"]["features"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.root / row["path"], allow_pickle=False) as sample:
            x = sample["features"].copy()
        if x.shape != self.shape or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("Malformed prepared features")
        return torch.from_numpy(x), row["label"], index


def provenance(dataset_path, config):
    repo = Path(__file__).resolve().parents[1]
    command = ["git", "-c", f"safe.directory={repo.as_posix()}"]
    commit = subprocess.check_output(command + ["rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(command + ["status", "--porcelain"], cwd=repo, text=True)
    if dirty:
        raise ValueError("Commit/review source changes before a recorded training run")
    import os
    import platform
    import importlib.metadata
    packages = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions() if item.metadata["Name"]}
    return {"git_commit": commit, "dataset_path": str(Path(dataset_path).resolve()),
            "dataset_sha256": sha256(dataset_path), "config": config,
            "python": platform.python_version(), "packages": packages,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
