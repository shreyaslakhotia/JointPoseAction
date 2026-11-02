"""Analyze already-acquired official metadata; no network, pose download or training."""
import argparse
from collections import Counter, defaultdict
import heapq
import json
import os
from pathlib import Path

import numpy as np

from e0.data import check_leakage, choose_classes, load_actions, load_segments, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Process the full CSVs in a CPU Slurm job")
    review_path = Path(args.review)
    review = json.loads(review_path.read_text())
    files = {entry["name"]: entry for entry in review["files"]}
    for entry in files.values():
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Official annotation hash changed")
    actions = load_actions(files["actions.csv"]["path"])
    segments, excluded = load_segments({s: files[f"{s}.csv"]["path"] for s in ("train", "validation", "test")}, actions)
    leakage = check_leakage(segments, "report_official_overlap")
    members = {Path(x["path"]).stem: x for x in review["archive"]["cleaned_members"]}
    available = [x for x in segments if x["recording_id"] in members]
    vocabulary = choose_classes(available, min_train_examples=50, max_classes=100)
    selected = [x for x in available if x["action_id"] in vocabulary]
    coverage = {}
    for split in ("train", "validation", "test"):
        all_rows = [x for x in segments if x["split"] == split]
        rows = [x for x in selected if x["split"] == split]
        recordings = sorted({x["recording_id"] for x in rows})
        coverage[split] = {"official_segments": len(all_rows),
                           "pose_available_segments": sum(x["recording_id"] in members for x in all_rows),
                           "pose_available_recordings": len({x["recording_id"] for x in all_rows if x["recording_id"] in members}),
                           "missing_pose_recordings": sorted({x["recording_id"] for x in all_rows if x["recording_id"] not in members}),
                           "candidate_segments": len(rows), "candidate_recordings": recordings,
                           "candidate_participants": sorted({x["participant_id"] for x in rows}),
                           "toy_categories": sorted({x["toy_name"] for x in rows if x["toy_name"]}),
                           "class_counts": dict(Counter(str(x["action_id"]) for x in rows))}
    selected_recordings = sorted({x["recording_id"] for x in selected})
    candidate_members = [{"recording_id": recording, **members[recording]} for recording in selected_recordings]
    by_recording = defaultdict(list)
    for row in segments:
        by_recording[row["recording_id"]].append(row)
    overlapping_pairs = 0
    for rows in by_recording.values():
        ends = []
        for row in sorted(rows, key=lambda x: x["start_frame"]):
            while ends and ends[0] < row["start_frame"]:
                heapq.heappop(ends)
            overlapping_pairs += len(ends)
            heapq.heappush(ends, row["end_frame"])
    sample = json.loads(Path(review["archive"]["sample_path"]).read_text())
    sample_quality = []
    for key in sorted(sample, key=int):
        hands = np.asarray([sample[key]["0"], sample[key]["1"]], dtype=float)
        sample_quality.append({"frame_id": int(key), "wrists_source_units": hands[:, 5].tolist(),
                               "zero_joint_indices_by_source_hand": [np.where((hand == 0).all(-1))[0].tolist() for hand in hands],
                               "finite": bool(np.isfinite(hands).all()),
                               "wrist_to_index_proximal_source_units": np.linalg.norm(hands[:, 8] - hands[:, 5], axis=-1).tolist()})
    compressed = sum(x["compressed_bytes"] for x in candidate_members)
    uncompressed = sum(x["uncompressed_bytes"] for x in candidate_members)
    report = {"dataset_revision": review["revision"], "review_sha256": sha256(review_path),
              "selection": {"basis": "all pose-available official recordings; vocabulary from training counts only",
                            "minimum_training_segments": 50, "maximum_classes": 100,
                            "class_mapping": [actions[x] for x in vocabulary]},
              "coverage": coverage, "leakage": leakage,
              "overlapping_interval_pairs_inclusive_reference": overlapping_pairs,
              "alternate_view_rows": len(excluded), "candidate_pose_members": candidate_members,
              "candidate_compressed_pose_bytes": compressed, "candidate_uncompressed_pose_bytes": uncompressed,
              "feature_bytes_uncompressed_upper_bound": len(selected) * 64 * 216 * 4,
              "partial_sample_diagnostics": sample_quality,
              "sample_frame_ids_are_not_recording_bounds": True,
              "training_ready": False,
              "blockers": ["Physical units and exact cleaned coordinate frame need producer/release evidence",
                           "Finite zero wrist is ambiguous; preserved, never presumed missing",
                           "Full representative recording and nominal 30/60 endpoint/origin checks pending",
                           "Actual pose transfer/storage plan requires review before acquisition"]}
    output = Path(args.output)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    brief = {"report": str(output), "classes": len(vocabulary),
             "coverage": {s: {k: v for k, v in row.items() if k not in ("candidate_recordings", "candidate_participants", "class_counts", "missing_pose_recordings")} for s, row in coverage.items()},
             "storage": {k: report[k] for k in ("candidate_compressed_pose_bytes", "candidate_uncompressed_pose_bytes", "feature_bytes_uncompressed_upper_bound")},
             "sample": sample_quality, "training_ready": False}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
