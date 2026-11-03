"""Immutable dataset contract and original-baseline preservation helpers."""
import hashlib
import json
from pathlib import Path

DATASET_SHA256 = "f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59"
SPLIT_COUNTS = {"train": 22083, "validation": 6880, "test": 9760}
ALLOCATION_BYTES = 50_000_000_000
TRAINING_RESERVE_BYTES = 2_000_000_000


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def dataset_contract(path, expected_hash=DATASET_SHA256):
    """No feature reads or statistics fitting; safe on the login submission path."""
    path = Path(path).resolve()
    if digest(path) != expected_hash:
        raise ValueError("Frozen E0 dataset hash differs; do not change the evaluation population")
    artifact = read(path)
    rows = artifact["segments"]
    counts = {s: sum(row["split"] == s for row in rows) for s in SPLIT_COUNTS}
    if counts != SPLIT_COUNTS or len(artifact["classes"]) != 100:
        raise ValueError("Unexpected frozen E0 population")
    if len({row["segment_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate action segment IDs")
    record_splits = {}
    for row in rows:
        if row["split"] not in SPLIT_COUNTS:
            raise ValueError("Unknown split")
        old = record_splits.setdefault(row["recording_id"], row["split"])
        if old != row["split"]:
            raise ValueError("Recording-level split leakage")
        if artifact["classes"][row["label"]]["action_id"] != row["action_id"]:
            raise ValueError("Frozen label-map mismatch")
    return {"dataset_path": str(path), "dataset_sha256": expected_hash,
            "counts": counts, "classes": artifact["classes"],
            "recordings": len(record_splits), "recording_disjoint": True,
            "participant_independence_claimed": False,
            "absolute_pose_video_clock_verified": False,
            "same_segment_observations_different_temporal_preprocessing": True}


def original_fingerprint(original):
    original = Path(original).resolve()
    needed = ["train/provenance.json", "train/best.pt", "train/history.jsonl",
              "train/training_completed.json", "test/metrics.json",
              "test/probabilities.npz", "test/predictions.csv", "RESULTS.md"]
    for relative in needed:
        if not (original / relative).is_file():
            raise ValueError(f"Original E0 artifact missing: {relative}")
    return {p.relative_to(original).as_posix(): digest(p)
            for p in sorted(original.rglob("*")) if p.is_file()}


def assert_original_unchanged(original, fingerprint):
    if original_fingerprint(original) != fingerprint:
        raise ValueError("Frozen original E0 artifacts changed; stop Phase 1C")
