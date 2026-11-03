"""Read-only post-run audit. No network, GPU, optimizer or model selection."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .data import sha256


def independent_metrics(probabilities, labels, classes):
    p, y = np.asarray(probabilities), np.asarray(labels, dtype=int)
    if p.shape != (len(y), classes) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid saved probability matrix")
    if not np.allclose(p.sum(1), 1, atol=1e-5) or (y < 0).any() or (y >= classes).any():
        raise ValueError("Invalid probability normalization or labels")
    predicted = p.argmax(1)
    support = np.bincount(y, minlength=classes)
    prediction_counts = np.bincount(predicted, minlength=classes)
    precision, recall, f1, correct_counts = [], [], [], []
    # Independent per-label Boolean sets, rather than using production confusion/F1 code.
    for k in range(classes):
        tp = int(((y == k) & (predicted == k)).sum())
        pr = tp / prediction_counts[k] if prediction_counts[k] else 0.
        re = tp / support[k] if support[k] else 0.
        precision.append(pr)
        recall.append(re)
        f1.append(2 * pr * re / (pr + re) if pr + re else 0.)
        correct_counts.append(tp)
    ranking = np.argsort(-p, axis=1, kind="stable")[:, :min(5, classes)]
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for actual, guess in zip(y, predicted):
        matrix[int(actual), int(guess)] += 1
    frequency = prediction_counts / len(y)
    return {"samples": len(y), "top1": float((predicted == y).mean()),
            "top5": float((ranking == y[:, None]).any(1).mean()),
            "macro_f1": float(np.mean(f1)), "weighted_f1": float(sum(f * int(n) for f, n in zip(f1, support)) / len(y)),
            "balanced_accuracy": float(np.mean(recall)),
            "never_predicted_classes": np.flatnonzero(prediction_counts == 0).tolist(),
            "zero_true_positive_classes": [i for i, n in enumerate(correct_counts) if not n],
            "support": support.tolist(), "prediction_counts": prediction_counts.tolist(),
            "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": matrix.tolist(),
            "top_10_prediction_mass": float(np.sort(prediction_counts)[-10:].sum() / len(y)),
            "effective_predicted_classes": float(np.exp(-sum(v * math.log(v) for v in frequency if v))),
            "maximum_probability_sum_error": float(np.max(np.abs(p.sum(1) - 1)))}


def history_summary(history, config):
    if not history:
        raise ValueError("Empty training history")
    best = min(history, key=lambda e: e["validation"]["loss"])
    last = history[-1]
    return {"epochs_executed": len(history), "configured_maximum_epochs": config["epochs"],
            "configured_patience": config["patience"], "class_weighting": config["class_weighting"],
            "minimum_validation_loss_epoch": best["epoch"],
            "maximum_validation_top1_epoch": max(history, key=lambda e: e["validation"]["top1"])["epoch"],
            "first_epoch": history[0], "selected_epoch": best, "last_epoch": last,
            "epochs_after_best_loss": last["epoch"] - best["epoch"],
            "training_metrics_are_online_with_dropout": True,
            "checkpoint_selection_is_validation_loss_not_macro_f1": True}


def clock_diagnostics(timestamps):
    converted = {int(k): v for k, v in timestamps.items()}
    if len(converted) != len(timestamps):
        raise ValueError("Duplicate numeric timestamp keys")
    keys = sorted(converted)
    times = np.asarray([converted[k] for k in keys], dtype=float)
    ids = np.asarray(keys, dtype=np.int64)
    if len(keys) < 2:
        raise ValueError("Too few timestamps")
    local = np.diff(times) - np.diff(ids) / 60.
    drift = (times - times[0]) - (ids - ids[0]) / 60.
    return {"minimum_frame": keys[0], "maximum_frame": keys[-1], "frames": len(keys),
            "timestamp_span": float(times[-1] - times[0]), "frame_id_span_at_60fps": float((ids[-1] - ids[0]) / 60),
            "maximum_local_step_residual": float(np.max(np.abs(local))),
            "maximum_accumulated_drift_seconds": float(np.max(np.abs(drift))),
            "end_drift_seconds": float(drift[-1]),
            "passes_original_local_cadence_gate": bool((np.abs(local) <= .002).all()),
            "accumulated_drift_over_50ms": bool(np.max(np.abs(drift)) > .05),
            "absolute_video_clock_offset_independently_verified": False}


def reconstruct_features(raw, confidence, row, samples=64):
    """Independent per-frame reconstruction of declared dimensionless feature layout."""
    first, stop = 2 * row["start_frame"], 2 * (row["end_frame"] + 1)
    raw = {int(k): v for k, v in raw.items()}
    confidence = {int(k): v for k, v in confidence.items()}
    indices = np.rint(np.linspace(first, stop - 1, samples)).astype(int)
    positions, masks = [], []
    for frame in indices:
        value = raw.get(int(frame))
        score = confidence.get(int(frame))
        hands, valid_hands = [], []
        for h in ("0", "1"):
            hand = np.asarray(value[h], dtype=float) if value is not None else np.zeros((21, 3))
            valid = np.isfinite(hand).all(1) if value is not None and score is not None and score[h] > 0 else np.zeros(21, dtype=bool)
            hands.append(hand)
            valid_hands.append(valid)
        positions.append(hands)
        masks.append(valid_hands)
    positions, masks = np.asarray(positions), np.asarray(masks)
    distances = []
    for t in range(samples):
        for h in range(2):
            if masks[t, h, 5] and masks[t, h, 8]:
                value = np.linalg.norm(positions[t, h, 8] - positions[t, h, 5])
                if value > 0:
                    distances.append(value)
    if not distances:
        raise ValueError("No valid independent scale")
    multiplier = 1. / float(np.median(distances))
    coordinates = np.where(masks[..., None], positions, 0.) * multiplier
    origin = np.zeros(3)
    for t in range(samples):
        available = [coordinates[t, h, 5] for h in range(2) if masks[t, h, 5]]
        if available:
            origin = np.mean(available, axis=0)
            break
    output = []
    for t in range(samples):
        local, roots, local_masks = [], [], []
        for h in range(2):
            local_valid = masks[t, h] & masks[t, h, 5]
            local.extend(np.where(local_valid[:, None], coordinates[t, h] - coordinates[t, h, 5], 0).ravel())
            roots.extend(coordinates[t, h, 5] - origin if masks[t, h, 5] else np.zeros(3))
            local_masks.extend(local_valid)
        output.append([*local, *roots, *masks[t].ravel(), *local_masks])
    return np.asarray(output, dtype=np.float32), indices


def scan_features(dataset_path, artifact):
    base = Path(dataset_path).parent
    groups = defaultdict(list)
    stats, violations, by_recording = [], Counter(), defaultdict(list)
    shape = (artifact["summary"]["samples_per_segment"], artifact["summary"]["features"])
    for i, row in enumerate(artifact["segments"]):
        path = (base / row["path"]).resolve()
        if not path.is_relative_to(base) or sha256(path) != row["sha256"]:
            raise ValueError("Prepared feature path/hash mismatch")
        with np.load(path, allow_pickle=False) as item:
            x, indices = item["features"], item["frame_indices"]
        if x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("Malformed prepared feature array")
        expected = np.rint(np.linspace(2 * row["start_frame"], 2 * row["end_frame"] + 1, shape[0])).astype(int)
        violations["frame_indices_mismatch"] += int(not np.array_equal(indices, expected))
        violations["mask_not_binary"] += int(not np.isin(x[:, 132:], (0, 1)).all())
        masks = x[:, 132:174].reshape(shape[0], 2, 21)
        local_masks = x[:, 174:].reshape(shape[0], 2, 21)
        local = x[:, :126].reshape(shape[0], 2, 21, 3)
        roots = x[:, 126:132].reshape(shape[0], 2, 3)
        violations["nonzero_masked_local_coordinates"] += int(np.any(local[local_masks == 0] != 0))
        violations["nonzero_masked_roots"] += int(np.any(roots[masks[:, :, 5] == 0] != 0))
        digest = hashlib.sha256(x.tobytes()).hexdigest()
        groups[digest].append((row["split"], row["label"], i))
        stat = {"split": row["split"], "label": row["label"], "validity": float(local_masks.mean()),
                "local_temporal_std": float(local.std(0).mean()), "root_temporal_std": float(roots.std(0).mean()),
                "coordinate_abs_max": float(np.abs(x[:, :132]).max()),
                "duration_seconds": (row["end_frame"] - row["start_frame"] + 1) / 30.,
                "source_to_feature_scale": row["quality"]["source_to_feature_scale"]}
        stats.append(stat)
        by_recording[row["recording_id"]].append(row)
    duplicates = [v for v in groups.values() if len(v) > 1]
    summary = {"scanned_segments": len(stats), "violations": dict(violations),
               "exact_duplicate_feature_groups": len(duplicates),
               "duplicate_groups_with_different_labels": sum(len({v[1] for v in group}) > 1 for group in duplicates),
               "duplicate_groups_crossing_splits": sum(len({v[0] for v in group}) > 1 for group in duplicates),
               "unique_feature_vectors": len(groups),
               "per_split": {}}
    for split in ("train", "validation", "test"):
        rows = [s for s in stats if s["split"] == split]
        quantiles = {k: np.quantile([s[k] for s in rows], [0, .25, .5, .75, .95, 1]).tolist()
                     for k in ("validity", "local_temporal_std", "root_temporal_std", "coordinate_abs_max", "duration_seconds", "source_to_feature_scale")}
        n = len(rows)
        upper_correct = sum(max(Counter(v[1] for v in group if v[0] == split).values(), default=0) for group in groups.values())
        summary["per_split"][split] = {"samples": n, "quantiles_min_q25_median_q75_q95_max": quantiles,
                                        "exact_input_ambiguity_top1_upper_bound": upper_correct / n,
                                        "almost_stationary_segments": sum(s["local_temporal_std"] < 1e-6 and s["root_temporal_std"] < 1e-6 for s in rows)}
    return summary, stats, by_recording


def raw_audit(dataset_path, artifact, by_recording):
    base = Path(dataset_path).parent
    records, failures = [], []
    for number, (rec, entry) in enumerate(sorted(artifact["source_manifest"]["recordings"].items()), 1):
        for value in (entry, entry["timestamp"], entry["confidence"]):
            if sha256(value["path"]) != value["sha256"]:
                raise ValueError("Raw source hash changed")
        raw = json.loads(Path(entry["path"]).read_text())
        timestamps = json.loads(Path(entry["timestamp"]["path"]).read_text())
        confidence = json.loads(Path(entry["confidence"]["path"]).read_text())
        diag = clock_diagnostics(timestamps)
        keys_match = {int(k) for k in raw} == {int(k) for k in timestamps} == {int(k) for k in confidence}
        selected = sorted(by_recording.get(rec, []), key=lambda s: (s["start_frame"], s["segment_id"]))
        probes = [selected[0], selected[-1]] if len(selected) > 1 else selected
        max_difference = 0.
        for row in probes:
            expected, indices = reconstruct_features(raw, confidence, row, artifact["summary"]["samples_per_segment"])
            with np.load(base / row["path"], allow_pickle=False) as actual:
                difference = float(np.max(np.abs(expected - actual["features"])))
                equal_indices = bool(np.array_equal(indices, actual["frame_indices"]))
            max_difference = max(max_difference, difference)
            if difference > 1e-5 or not equal_indices:
                failures.append({"recording_id": rec, "segment_id": row["segment_id"], "maximum_feature_difference": difference,
                                 "frame_indices_equal": equal_indices})
        ids = [int(k) for k in raw]
        row = {"recording_id": rec, "split": entry["split"], **diag,
               "companion_keys_match": keys_match, "retained_segments": len(selected),
               "reconstructed_segments": len(probes), "maximum_feature_difference": max_difference,
               "pose_minimum_frame": min(ids), "pose_maximum_frame": max(ids)}
        records.append(row)
        if number % 25 == 0:
            print(json.dumps({"raw_recordings_audited": number, "total": len(artifact["source_manifest"]["recordings"])}), flush=True)
    return {"recordings": records, "source_hashes_verified": True, "feature_reconstruction_failures": failures,
            "absolute_pose_video_clock_alignment_independently_verified": False,
            "companion_key_mismatch_recordings": sum(not r["companion_keys_match"] for r in records),
            "retained_recordings_with_drift_over_50ms": sum(r["retained_segments"] > 0 and r["accumulated_drift_over_50ms"] for r in records),
            "maximum_accumulated_drift_seconds": max(r["maximum_accumulated_drift_seconds"] for r in records)}


def exclusion_audit(dataset_path, artifact):
    counts, per_split, per_recording = Counter(), defaultdict(Counter), defaultdict(Counter)
    per_class = defaultdict(Counter)
    vocabulary = {c["action_id"] for c in artifact["classes"]}
    exclusions = json.loads((Path(dataset_path).parent / "exclusions.json").read_text())
    for row in exclusions:
        counts[row["reason"]] += 1
        per_split[row["split"]][row["reason"]] += 1
        if row["reason"] != "alternate_view":
            per_recording[row["recording_id"]][row["reason"]] += 1
            if row.get("action_id") in vocabulary:
                per_class[row["split"], row["action_id"]][row["reason"]] += 1
    return {"counts": dict(counts), "per_split": {k: dict(v) for k, v in per_split.items()},
            "per_class_selected_vocabulary": [{"action_id": c["action_id"], "action_cls": c["action_cls"],
                "split": split, "retained": artifact["summary"]["class_counts"][split].get(str(c["action_id"]), 0),
                "excluded": sum(per_class[split, c["action_id"]].values()),
                "reasons": dict(per_class[split, c["action_id"]])}
                for c in artifact["classes"] for split in ("train", "validation", "test")],
            "recordings_with_exclusions": len(per_recording),
            "largest_non_view_exclusion_counts": [{"recording_id": k, "counts": dict(v)} for k, v in sorted(per_recording.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:20]],
            "stored_summary_counts_match": dict(counts) == artifact["summary"]["exclusions"]}


def annotation_audit(artifact):
    """Stream original CSVs independently of the production segment loader."""
    source = artifact["source_manifest"]
    for entry in (source["actions"], *source["splits"].values()):
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Official annotation hash changed")
    with open(source["actions"]["path"], newline="", encoding="utf-8-sig") as stream:
        actions = {int(row["action_id"]): row for row in csv.DictReader(stream)}
    mapping_mismatches = []
    for label, row in enumerate(artifact["classes"]):
        actual = actions[row["action_id"]]
        if row["label"] != label or any(row[k] != int(actual[k]) for k in ("action_id", "verb_id", "noun_id")) or row["action_cls"] != actual["action_cls"]:
            mapping_mismatches.append(row["action_id"])
    retained = {r["segment_id"]: r for r in artifact["segments"]}
    found, mismatches, recording_splits, training_counts, canonical_counts = set(), [], defaultdict(set), Counter(), Counter()
    for split, entry in source["splits"].items():
        with open(entry["path"], newline="", encoding="utf-8-sig") as stream:
            for number, raw in enumerate(csv.DictReader(stream), 2):
                if not raw["video"].endswith("/C10119_rgb.mp4"):
                    continue
                rec = raw["video"].rsplit("/", 1)[0]
                recording_splits[rec].add(split)
                canonical_counts[split] += 1
                if split == "train" and rec in source["recordings"]:
                    training_counts[int(raw["action_id"])] += 1
                sid = f"{split}:{number}:{raw['id']}"
                if sid not in retained:
                    continue
                found.add(sid)
                row = retained[sid]
                wrong = any(row[k] != int(raw[k]) for k in ("action_id", "verb_id", "noun_id", "start_frame", "end_frame"))
                wrong |= row["recording_id"] != rec or row["split"] != split
                wrong |= row["participant_id"] != rec.split("_user_id_")[0].rsplit("_", 1)[-1]
                wrong |= artifact["classes"][row["label"]]["action_id"] != int(raw["action_id"])
                if wrong:
                    mismatches.append(sid)
    minimum, maximum = artifact["summary"]["min_train_examples"], artifact["summary"]["max_classes"]
    eligible = [a for a in training_counts if training_counts[a] >= minimum]
    selected = sorted(sorted(eligible, key=lambda a: (-training_counts[a], a))[:maximum])
    return {"original_csv_hashes_verified": True, "canonical_counts": dict(canonical_counts),
            "class_mapping_mismatches": mapping_mismatches, "retained_segment_mismatch_count": len(mismatches),
            "retained_mismatch_examples": mismatches[:20], "retained_ids_not_found_count": len(set(retained) - found),
            "duplicate_retained_ids": len(artifact["segments"]) - len(retained),
            "official_recordings_crossing_splits": {r: sorted(s) for r, s in recording_splits.items() if len(s) > 1},
            "training_only_vocabulary_reproduced": selected == [c["action_id"] for c in artifact["classes"]]}


def inference_audit(dataset_path, artifact, checkpoint, saved_test_p):
    import torch
    from torch.utils.data import DataLoader, Subset
    from .model import TCN
    from .runtime import SegmentDataset
    # The Slurm wrapper sets OMP/MKL thread limits before importing either backend.
    model = TCN(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    results = {}
    perturbations = {}
    for split in ("train", "validation", "test"):
        data = SegmentDataset(dataset_path, artifact, split)
        if split == "test":
            per_label, indices = Counter(), []
            for i, row in enumerate(data.rows):
                if per_label[row["label"]] < 3:
                    indices.append(i)
                    per_label[row["label"]] += 1
            chosen = Subset(data, indices)
        else:
            indices, chosen = list(range(len(data))), data
        probabilities, truth = [], []
        with torch.inference_mode():
            for features, labels, _ in DataLoader(chosen, batch_size=64):
                probabilities.append(model(features).softmax(1).numpy())
                truth.extend(labels.tolist())
                if split == "validation":
                    order = torch.randperm(features.shape[1], generator=torch.Generator().manual_seed(42))
                    for name, changed in (("time_reversed", features.flip(1)), ("time_shuffled", features[:, order]),
                                          ("coordinates_zeroed_masks_retained", torch.cat((torch.zeros_like(features[:, :, :132]), features[:, :, 132:]), dim=2))):
                        perturbations.setdefault(name, []).append(model(changed).softmax(1).numpy())
        p = np.concatenate(probabilities)
        result = independent_metrics(p, truth, len(artifact["classes"]))
        if split == "test":
            reference = saved_test_p[indices]
            result["maximum_difference_from_saved_probabilities"] = float(np.max(np.abs(p - reference)))
            result["predictions_different_from_saved"] = int((p.argmax(1) != reference.argmax(1)).sum())
            result["purpose"] = "stratified checkpoint/export integrity sample, not a full test estimate"
        results[split] = result
        if split == "validation":
            controls = {}
            for name, parts in perturbations.items():
                changed = np.concatenate(parts)
                control = independent_metrics(changed, truth, len(artifact["classes"]))
                controls[name] = {"top1": control["top1"], "macro_f1": control["macro_f1"],
                                  "prediction_change_fraction": float((changed.argmax(1) != p.argmax(1)).mean())}
            results["validation_perturbations"] = {"controls": controls, "selection_or_training_performed": False,
                "interpretation_limit": "Frozen-model perturbations can be out of distribution; these are sensitivity diagnostics, not trained ablation models"}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this audit in a CPU Slurm job")
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    root = Path("/projects/jointposeaction").resolve()
    if root != Path("/projects/_ssd/jointposeaction") or not run.is_relative_to(root / "experiments/e0") or not output.is_relative_to(root / "experiments/e0") or output.is_relative_to(run):
        raise ValueError("Use dedicated project inputs and a separate audit output")
    output.mkdir(parents=True, exist_ok=False)
    report = {"optimizer_steps": 0, "gpu_used": False, "baseline_files_modified": False,
              "audit_source_sha256": sha256(__file__)}
    try:
        import torch
        provenance = json.loads((run / "train/provenance.json").read_text())
        dataset_path = Path(provenance["dataset_path"]).resolve()
        if not dataset_path.is_relative_to(root / "data/processed"):
            raise ValueError("Unexpected dataset path")
        artifact = json.loads(dataset_path.read_text())
        for entry in artifact["source_manifest"]["recordings"].values():
            for source in (entry, entry["timestamp"], entry["confidence"]):
                if not Path(source["path"]).resolve().is_relative_to(root / "data/raw/assembly101-e0"):
                    raise ValueError("Raw source must stay within this project's acquired E0 corpus")
        report.update(dataset_sha256=sha256(dataset_path), training_commit=provenance["git_commit"], summary=artifact["summary"])
        checkpoint = torch.load(run / "train/best.pt", weights_only=True, map_location="cpu")
        if report["dataset_sha256"] != provenance["dataset_sha256"] or report["dataset_sha256"] != checkpoint["provenance"]["dataset_sha256"] or checkpoint["classes"] != artifact["classes"]:
            raise ValueError("Checkpoint/provenance/dataset label map mismatch")
        with np.load(run / "test/probabilities.npz", allow_pickle=False) as saved:
            p, y = saved["probabilities"], saved["labels"]
        rows = [s for s in artifact["segments"] if s["split"] == "test"]
        if not np.array_equal(y, [s["label"] for s in rows]):
            raise ValueError("Saved probability labels disagree with dataset row order")
        metrics = independent_metrics(p, y, len(artifact["classes"]))
        stored = json.loads((run / "test/metrics.json").read_text())
        metrics["production_metrics_differences"] = {"top1": metrics["top1"] - stored["top1"], "top5": metrics["top5"] - stored["top5"],
            "macro_f1": metrics["macro_f1"] - stored["macro_f1_all_selected_classes"]}
        metrics["confusion_matrix_matches_saved"] = metrics["confusion_matrix"] == stored["confusion_matrix"]
        with (run / "test/predictions.csv").open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        if len(predictions) != len(rows):
            raise ValueError("Prediction table has wrong row count")
        metrics["prediction_csv_mismatches"] = sum(
            item["segment_id"] != row["segment_id"] or int(item["true_action_id"]) != row["action_id"]
            or int(item["predicted_action_id"]) != artifact["classes"][int(guess)]["action_id"]
            for item, row, guess in zip(predictions, rows, p.argmax(1)))
        report["metrics"] = metrics
        report["saved_verb_top1"] = stored.get("verb_top1")
        report["saved_noun_top1"] = stored.get("noun_top1")
        report["sanity"] = json.loads((run / "sanity/sanity.json").read_text())
        history = [json.loads(line) for line in (run / "train/history.jsonl").read_text().splitlines() if line]
        report["training"] = history_summary(history, provenance["config"])
        report["training"]["checkpoint_epoch"] = checkpoint["epoch"]
        report["training"]["checkpoint_selection_consistent"] = checkpoint["epoch"] == report["training"]["minimum_validation_loss_epoch"]
        report["annotations"] = annotation_audit(artifact)
        report["exclusions"] = exclusion_audit(dataset_path, artifact)
        features, stats, by_recording = scan_features(dataset_path, artifact)
        report["features"] = features
        report["raw"] = raw_audit(dataset_path, artifact, by_recording)
        report["inference"] = inference_audit(dataset_path, artifact, checkpoint, p)
        saved_val = json.loads((run / "train/validation_metrics.json").read_text())
        val = report["inference"]["validation"]
        report["validation_reproduction_differences"] = {"top1": val["top1"] - saved_val["top1"],
            "macro_f1": val["macro_f1"] - saved_val["macro_f1_all_selected_classes"]}
        train_counts = np.bincount([r["label"] for r in artifact["segments"] if r["split"] == "train"], minlength=len(artifact["classes"]))
        report["per_class"] = [{**c, "training_support": int(train_counts[i]), "test_support": metrics["support"][i],
                                "test_predictions": metrics["prediction_counts"][i], "test_recall": metrics["recall"][i],
                                "test_f1": metrics["f1"][i]} for i, c in enumerate(artifact["classes"])]
        top = np.argsort(-np.asarray(metrics["prediction_counts"]), kind="stable")[:10]
        report["most_predicted_classes"] = [report["per_class"][i] for i in top]
        a, b = np.log1p(train_counts), np.asarray(metrics["f1"])
        a, b = a - a.mean(), b - b.mean()
        denominator = math.sqrt(float((a * a).sum() * (b * b).sum()))
        report["training_log_frequency_vs_test_f1_correlation"] = float((a * b).sum() / denominator) if denominator else None
        test_stats = [s for s in stats if s["split"] == "test"]
        report["test_validity_groups"] = []
        correct = p.argmax(1) == y
        for low, high in ((0., .5), (.5, .75), (.75, 1.0001)):
            selected = [i for i, s in enumerate(test_stats) if low <= s["validity"] < high]
            report["test_validity_groups"].append({"range": [low, high], "samples": len(selected),
                "top1": float(correct[selected].mean()) if selected else None})
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "reason": str(error)}
    finally:
        (output / "audit.json").write_text(json.dumps(report, indent=2))
        compact = {"report": str(output / "audit.json"), "dataset_sha256": report.get("dataset_sha256"),
                   "training": report.get("training"), "failure": report.get("failure")}
        if "metrics" in report:
            m = report["metrics"]
            compact["metrics"] = {k: m[k] for k in ("top1", "top5", "macro_f1", "weighted_f1", "balanced_accuracy", "production_metrics_differences", "confusion_matrix_matches_saved", "prediction_csv_mismatches", "never_predicted_classes", "zero_true_positive_classes", "top_10_prediction_mass", "effective_predicted_classes")}
        compact["verb_noun_top1"] = [report.get("saved_verb_top1"), report.get("saved_noun_top1")]
        compact["annotations"] = report.get("annotations")
        compact["exclusions"] = report.get("exclusions", {}).get("per_split")
        compact["feature_checks"] = {k: v for k, v in report.get("features", {}).items() if k != "per_split"}
        compact["raw_checks"] = {k: v for k, v in report.get("raw", {}).items() if k != "recordings"}
        compact["full_checkpoint_inference"] = {split: {k: v for k, v in value.items() if k in ("samples", "top1", "macro_f1", "balanced_accuracy", "maximum_difference_from_saved_probabilities", "predictions_different_from_saved")}
                                                  for split, value in report.get("inference", {}).items() if split != "validation_perturbations"}
        compact["validation_reproduction_differences"] = report.get("validation_reproduction_differences")
        compact["validation_sensitivity"] = report.get("inference", {}).get("validation_perturbations")
        compact["most_predicted_classes"] = report.get("most_predicted_classes")
        compact["test_validity_groups"] = report.get("test_validity_groups")
        print(json.dumps(compact, indent=2), flush=True)
    if "failure" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
