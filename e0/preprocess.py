"""CPU-only segment preprocessing. Real data require a validated source contract."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .data import (SPLITS, check_leakage, choose_classes, load_actions, load_segments,
                   read_cleaned, require_profile, segment_features, sha256)
from .quality import EXCLUDED_RECORDINGS, load_confidences, timestamp_cadence


def prepare(manifest, output, samples=64, min_train_examples=50, max_classes=100,
            min_valid_fraction=0.25, participant_policy="report_official_overlap", synthetic=False):
    require_profile(manifest["pose_profile"], allow_unverified=synthetic)
    if not 0 <= min_valid_fraction <= 1 or min_train_examples < 1 or max_classes < 2:
        raise ValueError("Invalid preprocessing configuration")
    if manifest.get("status") != ("synthetic" if synthetic else "validated") or not manifest.get("dataset_revision"):
        raise ValueError("Validated revision-pinned dataset manifest required")
    sources = [manifest["actions"], *manifest["splits"].values()]
    sources += list(manifest["recordings"].values())
    sources += [entry[k] for entry in manifest["recordings"].values()
                for k in ("confidence", "timestamp") if k in entry]
    for source in sources:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {source['path']}")
    actions = load_actions(manifest["actions"]["path"])
    segments, exclusions = load_segments({k: v["path"] for k, v in manifest["splits"].items()}, actions)
    # Check the entire supplied canonical-view split metadata, before exclusions
    # can hide a recording shared between official files.
    official_leakage = check_leakage(segments, "report_official_overlap")
    selected = [s for s in segments if s["recording_id"] in manifest["recordings"]
                and s["recording_id"] not in EXCLUDED_RECORDINGS]
    leakage = check_leakage(selected, participant_policy)
    vocabulary = choose_classes(selected, min_train_examples, max_classes)
    mapping = {a: i for i, a in enumerate(vocabulary)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples").mkdir()
    retained = []
    by_recording = {}
    for segment in segments:
        reason = None
        if segment["recording_id"] in EXCLUDED_RECORDINGS:
            reason = EXCLUDED_RECORDINGS[segment["recording_id"]]
        elif segment["recording_id"] not in manifest["recordings"]:
            reason = "recording_not_selected"
        elif segment["action_id"] not in mapping:
            reason = "outside_training_selected_vocabulary"
        if reason:
            exclusions.append({**segment, "reason": reason})
            continue
        by_recording.setdefault(segment["recording_id"], []).append(segment)
    for recording in sorted(by_recording):
        entry = manifest["recordings"][recording]
        frames = read_cleaned(entry["path"])
        confidences = None
        if not synthetic or "confidence" in entry or "timestamp" in entry:
            if "confidence" not in entry or "timestamp" not in entry:
                raise ValueError("Real recordings require hashed timestamp and confidence companions")
            timing = timestamp_cadence(entry["timestamp"]["path"], frames)
            if not timing["accepted"]:
                exclusions.extend({**s, "reason": timing["reason"], "timing": timing} for s in by_recording[recording])
                continue
            confidences = load_confidences(entry["confidence"]["path"])
            if set(confidences) != set(frames):
                exclusions.extend({**s, "reason": "confidence_pose_keys_mismatch"} for s in by_recording[recording])
                continue
        for segment in by_recording[recording]:
            if entry["split"] != segment["split"] or entry["participant_id"] != segment["participant_id"]:
                raise ValueError("Selected recording manifest disagrees with official segment membership")
            try:
                features, quality = segment_features(frames, segment, manifest["pose_profile"], samples, confidences)
            except ValueError as error:
                if str(error) != "interval_outside_pose_coverage":
                    raise
                exclusions.append({**segment, "reason": str(error)})
                continue
            if quality["valid_local_fraction"] < min_valid_fraction or quality["valid_local_fraction"] == 0:
                exclusions.append({**segment, "reason": "insufficient_valid_local_joints", "quality": quality})
                continue
            name = f"samples/{len(retained):08d}.npz"
            np.savez_compressed(output / name, features=features,
                                frame_indices=np.asarray(quality["pose_frame_indices"], dtype=np.int64))
            retained.append({**segment, "label": mapping[segment["action_id"]],
                             "path": name, "sha256": sha256(output / name), "quality": quality})
        del frames  # one recording at a time; no complete pose corpus in memory
    classes = [{"label": mapping[a], **actions[a]} for a in vocabulary]
    summary = {"split_counts": dict(Counter(x["split"] for x in retained)),
               "class_counts": {s: dict(Counter(str(x["action_id"]) for x in retained if x["split"] == s)) for s in SPLITS},
               "recording_counts": {s: len({x["recording_id"] for x in retained if x["split"] == s}) for s in SPLITS},
               "participant_counts": {s: len({x["participant_id"] for x in retained if x["split"] == s}) for s in SPLITS},
               "exclusions": dict(Counter(x["reason"] for x in exclusions)),
               "leakage": leakage, "official_metadata_leakage": official_leakage,
               "classes": len(classes), "features": 216,
               "samples_per_segment": samples, "train_only_class_selection": True,
               "min_train_examples": min_train_examples, "max_classes": max_classes,
               "min_valid_local_fraction": min_valid_fraction,
               "overlapping_segments": "retained independently", "temporal_filter": None}
    artifact = {"ready_for_training": False, "source_manifest": manifest, "classes": classes,
                "segments": retained, "summary": summary}
    # Save complete exclusions, including alternate-view rows, before readiness
    # failure; never leave a failure without an accounting of considered events.
    (output / "exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    (output / "dataset.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if len(classes) < 2 or any(summary["split_counts"].get(s, 0) == 0 for s in SPLITS):
        raise ValueError("Dataset not train-ready: fewer than two classes or an empty split; see saved summary")
    if set(summary["class_counts"]["train"]) != {str(a) for a in vocabulary}:
        raise ValueError("A selected class lost all training examples; review exclusions")
    if any(count < min_train_examples for count in summary["class_counts"]["train"].values()):
        raise ValueError("Selected classes fell below minimum valid training count; review exclusions")
    artifact["ready_for_training"] = True
    (output / "dataset.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--min-train-examples", type=int, default=50)
    parser.add_argument("--max-classes", type=int, default=100)
    parser.add_argument("--min-valid-fraction", type=float, default=0.25)
    parser.add_argument("--participant-policy", choices=["require_disjoint", "report_official_overlap"], default="report_official_overlap")
    args = parser.parse_args()
    artifact = prepare(json.loads(Path(args.manifest).read_text()), args.output,
                       args.samples, args.min_train_examples, args.max_classes,
                       args.min_valid_fraction, args.participant_policy)
    print(json.dumps(artifact["summary"], indent=2))


if __name__ == "__main__":
    main()
