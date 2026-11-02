"""Official CSV segments and an explicitly profiled cleaned-landmarks adapter."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

SPLITS = ("train", "validation", "test")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_actions(path):
    actions = {}
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"action_id", "verb_id", "noun_id", "action_cls", "verb_cls", "noun_cls"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("actions.csv does not match documented official columns")
        for row in reader:
            item = {k: row[k] for k in required}
            for key in ("action_id", "verb_id", "noun_id"):
                item[key] = int(item[key])
                if item[key] < 0:
                    raise ValueError("Negative class ID")
            if item["action_id"] in actions:
                raise ValueError("Duplicate action mapping")
            actions[item["action_id"]] = item
    if not actions:
        raise ValueError("No action mappings")
    return actions


def load_segments(files, actions, canonical_view="C10119_rgb.mp4"):
    """Keep one documented source view; overlapping intervals remain independent."""
    if set(files) != set(SPLITS):
        raise ValueError("Explicit train/validation/test CSV paths required")
    segments, exclusions = [], []
    seen = set()
    columns = {"id", "video", "start_frame", "end_frame", "action_id", "verb_id", "noun_id"}
    for split in SPLITS:
        with open(files[split], newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not columns.issubset(reader.fieldnames or []):
                raise ValueError(f"Unsupported official CSV header: {split}")
            for line, row in enumerate(reader, start=2):
                source_id = f"{split}:{line}:{row['id']}"
                parts = row["video"].split("/")
                if len(parts) != 2 or not parts[0].startswith("nusar-"):
                    raise ValueError(f"Invalid recording/view path at {source_id}")
                recording, view = parts
                match = re.search(r"_([0-9]+)_user_id_", recording)
                if not match:
                    raise ValueError(f"Cannot identify participant: {recording}")
                action, verb, noun = (int(row[k]) for k in ("action_id", "verb_id", "noun_id"))
                if action not in actions or (verb, noun) != (actions[action]["verb_id"], actions[action]["noun_id"]):
                    raise ValueError(f"Label mapping mismatch at {source_id}")
                start, end = int(row["start_frame"]), int(row["end_frame"])
                if start < 0 or end < start:
                    raise ValueError(f"Invalid interval at {source_id}")
                if view != canonical_view:
                    exclusions.append({"source_id": source_id, "recording_id": recording,
                                       "reason": "alternate_view", "split": split})
                    continue
                # Annotation identity, not interval/class, defines an example.
                # Equal/overlapping intervals with distinct IDs remain distinct.
                key = (split, recording, row["id"])
                if key in seen:
                    raise ValueError(f"Duplicate canonical-view event: {source_id}")
                seen.add(key)
                segments.append({"segment_id": source_id, "recording_id": recording,
                                 "participant_id": match[1], "source_view": view, "split": split,
                                 "start_frame": start, "end_frame": end,
                                 "toy_id": row.get("toy_id"), "toy_name": row.get("toy_name"),
                                 "action_id": action, "verb_id": verb, "noun_id": noun})
    return segments, exclusions


def check_leakage(segments, participant_policy="require_disjoint"):
    if participant_policy not in ("require_disjoint", "report_official_overlap"):
        raise ValueError("Explicit participant leakage policy required")
    recording_splits, participant_splits = defaultdict(set), defaultdict(set)
    for segment in segments:
        recording_splits[segment["recording_id"]].add(segment["split"])
        participant_splits[segment["participant_id"]].add(segment["split"])
    recordings = {k: sorted(v) for k, v in recording_splits.items() if len(v) > 1}
    participants = {k: sorted(v) for k, v in participant_splits.items() if len(v) > 1}
    if recordings:
        raise ValueError(f"Recording leakage: {recordings}")
    if participants and participant_policy == "require_disjoint":
        raise ValueError(f"Participant overlap: {participants}")
    return {"recording_overlap": recordings, "participant_overlap": participants,
            "participant_policy": participant_policy}


def choose_classes(segments, min_train_examples, max_classes=None):
    counts = Counter(s["action_id"] for s in segments if s["split"] == "train")
    # No validation/test counts enter selection or label reindexing.
    eligible = [k for k in counts if counts[k] >= min_train_examples]
    eligible.sort(key=lambda k: (-counts[k], k))
    if max_classes is not None:
        eligible = eligible[:max_classes]
    if not eligible:
        raise ValueError("Training-only selection produced no classes")
    return sorted(eligible)


def require_profile(profile, allow_unverified=False):
    required = {"schema", "hand_keys", "hand_order", "wrist_index", "joint_order", "units_to_m",
                "coordinate_frame", "pose_fps", "annotation_fps", "annotation_index_base",
                "pose_frame_offset", "end_inclusive", "zero_is_invalid", "evidence"}
    if not required.issubset(profile):
        raise ValueError(f"Missing pose contract fields: {sorted(required - profile.keys())}")
    if profile["schema"] != "assembly101_cleaned_landmarks3D":
        raise ValueError("Only documented cleaned landmarks3D adapter is supported")
    if profile["hand_keys"] != ["0", "1"] or profile["hand_order"] != ["left", "right"] or profile["wrist_index"] != 5:
        raise ValueError("Expected source-order [left,right] and wrist 5 from pinned reader")
    if len(profile["joint_order"]) != 21 or len(set(profile["joint_order"])) != 21:
        raise ValueError("Explicit unique 21-joint source order required")
    if (profile["pose_fps"], profile["annotation_fps"]) != (60, 30):
        raise ValueError("This adapter supports only verified 60/30 FPS products")
    if profile["annotation_index_base"] not in (0, 1) or not isinstance(profile["pose_frame_offset"], int):
        raise ValueError("Explicit integer index origin/offset required")
    if not isinstance(profile["end_inclusive"], bool) or not isinstance(profile["zero_is_invalid"], bool):
        raise ValueError("Endpoint and zero policies must be explicit")
    dimensionless = profile.get("feature_units") == "dimensionless_wrist_index_length"
    if dimensionless:
        if profile["units_to_m"] is not None or profile.get("scale_joints") != [5, 8]:
            raise ValueError("Dimensionless features require explicit wrist/index normalization, no metric conversion")
    elif not isinstance(profile["units_to_m"], (int, float)) or not 0 < profile["units_to_m"] < 1000:
        raise ValueError("Verified units conversion required")
    if not profile["coordinate_frame"] or not profile["evidence"]:
        raise ValueError("Coordinate-frame and verification evidence required")
    if not allow_unverified and (profile.get("status") != "validated_real_sample" or profile.get("synthetic")):
        raise ValueError("Real preprocessing blocked until pose/sample/time contract is validated")
    if not allow_unverified and profile["zero_is_invalid"] and not profile.get("zero_invalid_evidence"):
        raise ValueError("Zero-as-missing requires authoritative evidence; finite zeros must otherwise be preserved")


def pose_bounds(segment, profile):
    base, offset = profile["annotation_index_base"], profile["pose_frame_offset"]
    first = 2 * (segment["start_frame"] - base) + offset
    stop = 2 * (segment["end_frame"] - base + int(profile["end_inclusive"])) + offset
    if first < 0 or stop <= first:
        raise ValueError("Invalid mapped interval")
    return first, stop


def read_cleaned(path):
    """Frame-key -> {'0': 21x3, '1': 21x3}; no legacy-list fallback."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("Expected nonempty cleaned frame-keyed object")
    frames = {}
    for key, row in value.items():
        if not key.isdigit() or int(key) in frames or not isinstance(row, dict) or set(row) != {"0", "1"}:
            raise ValueError("Unexpected cleaned frame/hand keys")
        xyz = np.asarray([row["0"], row["1"]], dtype=np.float64)
        if xyz.shape != (2, 21, 3):
            raise ValueError("Expected two 21x3 hands")
        frames[int(key)] = xyz
    return frames


def segment_features(frames, segment, profile, samples=64, confidences=None):
    """Nearest-frame sampling without interpolation, smoothing or gap filling."""
    if samples < 2:
        raise ValueError("At least two temporal samples required")
    first, stop = pose_bounds(segment, profile)
    if first < min(frames) or stop - 1 > max(frames):
        raise ValueError("interval_outside_pose_coverage")
    indices = np.rint(np.linspace(first, stop - 1, samples)).astype(np.int64)
    xyz = np.zeros((samples, 2, 21, 3), dtype=np.float64)
    available = np.zeros(samples, dtype=bool)
    for t, frame in enumerate(indices):
        if int(frame) in frames:
            xyz[t] = frames[int(frame)]
            available[t] = True
    zero_coordinates = available[:, None, None] & np.isfinite(xyz).all(-1) & (xyz == 0).all(-1)
    valid = np.isfinite(xyz).all(-1) & available[:, None, None]
    if confidences is not None:
        # Zero score is an explicit conservative exclusion policy, not a calibrated
        # probability or a zero-coordinate heuristic. Missing scores are unavailable.
        score_valid = np.zeros((samples, 2), dtype=bool)
        for t, frame in enumerate(indices):
            if int(frame) in confidences:
                score_valid[t] = np.asarray(confidences[int(frame)]) > 0
        valid &= score_valid[..., None]
    if profile["zero_is_invalid"]:
        valid &= np.any(xyz != 0, axis=-1)
    dimensionless = profile.get("feature_units") == "dimensionless_wrist_index_length"
    scale = profile["units_to_m"]
    if dimensionless:
        lengths = np.linalg.norm(xyz[:, :, 8] - xyz[:, :, 5], axis=-1)
        lengths = lengths[valid[:, :, 8] & valid[:, :, 5] & (lengths > 0)]
        if not len(lengths):
            raise ValueError("no_valid_hand_scale")
        scale = 1.0 / float(np.median(lengths))
    xyz = np.where(valid[..., None], xyz, 0) * scale
    wrist = profile["wrist_index"]
    wrist_valid = valid[:, :, wrist]
    local_valid = valid & wrist_valid[..., None]
    local = xyz - xyz[:, :, wrist:wrist + 1]
    local = np.where(local_valid[..., None], local, 0)
    roots = xyz[:, :, wrist]
    # Shared segment origin preserves relative wrist positions and motion.
    # This is translation normalization, not temporal denoising/statistics fitting.
    origin = np.zeros(3)
    for t in range(samples):
        if wrist_valid[t].any():
            origin = roots[t, wrist_valid[t]].mean(0)
            break
    roots = np.where(wrist_valid[..., None], roots - origin, 0)
    features = np.concatenate([local.reshape(samples, -1), roots.reshape(samples, -1),
                               valid.reshape(samples, -1), local_valid.reshape(samples, -1)], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite normalized features")
    return features, {"pose_frame_indices": indices.tolist(), "pose_seconds": (indices / 60).tolist(),
                      "pose_interval_half_open": [first, stop],
                      "valid_joint_fraction": float(valid.mean()),
                      "valid_local_fraction": float(local_valid.mean()),
                      "missing_sampled_frames": int((~available).sum()),
                      "zero_coordinate_joint_count": int(zero_coordinates.sum()),
                      "zero_policy": "invalid_by_verified_rule" if profile["zero_is_invalid"] else "preserved_ambiguity_not_tracking_confidence",
                      "confidence_available": confidences is not None,
                      "confidence_policy": "positive_source_hand_score" if confidences is not None else None,
                      "feature_units": "dimensionless_wrist_index_length" if dimensionless else "metres",
                      "source_to_feature_scale": scale,
                      "origin_feature_units": origin.tolist(),
                      **({} if dimensionless else {"origin_m": origin.tolist()})}
