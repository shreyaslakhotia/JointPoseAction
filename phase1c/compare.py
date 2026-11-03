"""Matched-example comparison of frozen checkpoints; never selects or trains models."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from e0.metrics import compute_metrics


FROZEN_DATASET_SHA256 = "f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59"


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def measured_metrics(probabilities, labels, classes):
    """All selected classes enter macro-F1; balanced accuracy uses supported classes."""
    metrics = compute_metrics(probabilities, labels, classes)
    confusion = np.asarray(metrics["confusion_matrix"])
    support, predicted = confusion.sum(1), confusion.sum(0)
    correct = confusion.diagonal()
    recall = np.divide(correct, support, out=np.zeros(len(classes)), where=support > 0)
    f1 = np.asarray([row["f1"] for row in metrics["per_class"]])
    metrics.update(weighted_f1=float((f1 * support).sum() / support.sum()),
                   balanced_accuracy=float(recall[support > 0].mean()),
                   classes_never_predicted=int((predicted == 0).sum()),
                   classes_zero_correct=int((correct == 0).sum()),
                   supported_classes_zero_correct=int(((correct == 0) & (support > 0)).sum()),
                   prediction_concentration_top10=float(np.sort(predicted)[-min(10, len(classes)):].sum() / support.sum()))
    for row in metrics["per_class"]:
        index = row["label"]
        row.update(predictions=int(predicted[index]), correct=int(correct[index]), recall=float(recall[index]))
    return metrics


def paired_recording_bootstrap(left_correct, right_correct, recording_ids, repeats=10000, seed=20261103):
    """Percentile CI for right-minus-left segment-weighted accuracy, clustered by recording."""
    left, right = np.asarray(left_correct), np.asarray(right_correct)
    ids = np.asarray(recording_ids, dtype=str)
    if left.ndim != 1 or left.shape != right.shape or left.shape != ids.shape or not len(ids):
        raise ValueError("Paired predictions and recording identifiers must align")
    if not np.isin(left, [0, 1]).all() or not np.isin(right, [0, 1]).all():
        raise ValueError("Expected correctness indicators")
    if repeats < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    groups, inverse = np.unique(ids, return_inverse=True)
    if len(groups) < 2:
        raise ValueError("At least two recordings required for clustered uncertainty")
    counts = np.bincount(inverse)
    differences = np.bincount(inverse, weights=right.astype(float) - left.astype(float))
    generator = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    for start in range(0, repeats, 256):
        sampled = generator.integers(0, len(groups), size=(min(256, repeats - start), len(groups)))
        estimates[start:start + len(sampled)] = differences[sampled].sum(1) / counts[sampled].sum(1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {"difference_right_minus_left": float((right.astype(float) - left.astype(float)).mean()),
            "ci95_percentile": [float(lower), float(upper)], "recordings": len(groups),
            "segments": len(ids), "replicates": repeats, "seed": seed,
            "sampling_unit": "recording; all its action segments retained together",
            "estimand": "segment-weighted accuracy difference on the matched held-out population",
            "limitation": "recording clusters are assumed independent; shared participants may retain dependence; conditional on this fixed trained pair, not training-seed uncertainty"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(name, run_path, artifact, dataset_sha):
    run = Path(run_path).resolve()
    metrics_path = run / "test/metrics.json"
    metrics = read_json(metrics_path)
    provenance = metrics.get("provenance", {})
    if provenance.get("dataset_sha256") != dataset_sha:
        raise ValueError(f"{name}: evaluation dataset hash differs")
    if "class_mapping" in metrics and metrics["class_mapping"] != artifact["classes"]:
        raise ValueError(f"{name}: class ordering differs")
    with (run / "test/predictions.csv").open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    with np.load(run / "test/probabilities.npz", allow_pickle=False) as arrays:
        probabilities, labels = arrays["probabilities"].copy(), arrays["labels"].copy()
        if "segment_ids" in arrays and list(arrays["segment_ids"].astype(str)) != [row["segment_id"] for row in exported]:
            raise ValueError(f"{name}: probability/CSV segment ordering differs")
    if probabilities.shape != (len(exported), len(artifact["classes"])) or labels.shape != (len(exported),):
        raise ValueError(f"{name}: prediction array shape mismatch")
    source = {row["segment_id"]: row for row in artifact["segments"] if row["split"] == "test"}
    if len({row["segment_id"] for row in exported}) != len(exported):
        raise ValueError(f"{name}: duplicate exported segment")
    if not exported or set(row["segment_id"] for row in exported) - source.keys():
        raise ValueError(f"{name}: empty or non-test prediction population")
    measured_metrics(probabilities, labels, artifact["classes"])
    indices = {}
    for index, row in enumerate(exported):
        original = source[row["segment_id"]]
        if labels[index] != original["label"] or str(row["true_action_id"]) != str(original["action_id"]):
            raise ValueError(f"{name}: true label mismatch")
        predicted_action = artifact["classes"][int(probabilities[index].argmax())]["action_id"]
        if str(row["predicted_action_id"]) != str(predicted_action):
            raise ValueError(f"{name}: predicted label/probabilities mismatch")
        if row.get("recording_id") != original["recording_id"]:
            raise ValueError(f"{name}: recording identity mismatch")
        indices[row["segment_id"]] = index
    resources = {}
    for relative, key in (("train/provenance.json", "training_provenance"),
                          ("train/training_completed.json", "training_completion"),
                          ("submission.json", "submission")):
        if (run / relative).is_file():
            resources[key] = read_json(run / relative)
    hashes = {relative: file_sha(run / relative) for relative in
              ("test/metrics.json", "test/predictions.csv", "test/probabilities.npz",
               "train/provenance.json", "train/training_completed.json", "train/history.jsonl", "train/best.pt")
              if (run / relative).is_file()}
    return {"name": name, "run": str(run), "indices": indices,
            "probabilities": probabilities, "labels": labels, "resources": resources, "input_hashes": hashes}


def compare(dataset_path, runs, output, *, allow_common_subset=False, exclusion_reason=None,
            repeats=10000, seed=20261103, expected_dataset_sha=FROZEN_DATASET_SHA256):
    dataset_path, output = Path(dataset_path).resolve(), Path(output).resolve()
    dataset_sha = file_sha(dataset_path)
    if dataset_sha != expected_dataset_sha:
        raise ValueError("Dataset differs from the frozen approved manifest")
    artifact = read_json(dataset_path)
    if len(runs) < 2 or len(dict(runs)) != len(runs):
        raise ValueError("At least two uniquely named runs required")
    for _, run in runs:
        root = Path(run).resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Comparison output must be separate from every frozen run")
    loaded = [load_run(name, path, artifact, dataset_sha) for name, path in runs]
    test_rows = [row for row in artifact["segments"] if row["split"] == "test"]
    all_ids = {row["segment_id"] for row in test_rows}
    common = set.intersection(*(set(run["indices"]) for run in loaded))
    full_population = all(set(run["indices"]) == all_ids for run in loaded)
    if not full_population and not (allow_common_subset and exclusion_reason and exclusion_reason.strip()):
        raise ValueError("Population mismatch: explicit common-subset policy and exclusion reason required")
    if not common:
        raise ValueError("No shared evaluation examples")
    rows = [row for row in test_rows if row["segment_id"] in common]
    selected_ids = [row["segment_id"] for row in rows]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    classes = artifact["classes"]
    report = {"status": "measured", "dataset_sha256": dataset_sha, "dataset_path": str(dataset_path),
              "class_mapping": classes,
              "split_counts": {split: sum(row["split"] == split for row in artifact["segments"])
                               for split in ("train", "validation", "test")},
              "population": {"policy": "full_frozen_test" if full_population else "explicit_common_subset",
              "original_test_segments": len(test_rows), "compared_segments": len(rows),
              "recordings": len({row["recording_id"] for row in rows}), "exclusion_reason": exclusion_reason,
              "segment_ids": selected_ids, "excluded_segment_ids_by_model": {
                  run["name"]: sorted(all_ids - run["indices"].keys()) for run in loaded}},
              "limitations": ["Official recording-disjoint splits; participants overlap across splits.",
                  "The original test results were already inspected; comparisons are diagnostic, never a selection criterion.",
                  "Absolute pose-to-video synchronization and metric calibration remain unverified.",
                  "Architecture comparisons may also reflect documented temporal preprocessing differences.",
                  "Recording bootstrap preserves overlapping segments within each recording; it does not quantify training-seed uncertainty."],
              "models": {}, "paired_accuracy_differences": []}
    correctness = {}
    for run in loaded:
        order = [run["indices"][identifier] for identifier in selected_ids]
        probabilities = run["probabilities"][order]
        metrics = measured_metrics(probabilities, labels, classes)
        correctness[run["name"]] = probabilities.argmax(1) == labels
        report["models"][run["name"]] = {"run": run["run"], "metrics": metrics,
                    "resources": run["resources"], "input_hashes": run["input_hashes"],
                    "learning_curves": {"history": str(Path(run["run"]) / "train/history.jsonl"),
                                        "plots": str(Path(run["run"]) / "test")}}
    for left, right in itertools.combinations(correctness, 2):
        result = paired_recording_bootstrap(correctness[left], correctness[right],
                                             [row["recording_id"] for row in rows], repeats, seed)
        report["paired_accuracy_differences"].append({"left": left, "right": right, **result})
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "per_class.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "label", "action_id", "support", "predictions", "correct", "accuracy", "recall", "f1"])
        writer.writeheader()
        for name, model in report["models"].items():
            for row in model["metrics"]["per_class"]:
                writer.writerow({"model": name, **row})
    (output / "DUAL_BACKBONE_COMPARISON.md").write_text(markdown_report(report), encoding="utf-8")
    for name, filename in (("B0-balanced", "B0_BALANCED_RESULTS.md"), ("B1-HandFormer", "B1_HANDFORMER_RESULTS.md")):
        if name in report["models"]:
            (output / filename).write_text(model_report(name, report), encoding="utf-8")
    fusion = Path(__file__).resolve().parents[1] / "docs/experiments/FUSION_READINESS.md"
    shutil.copyfile(fusion, output / "FUSION_READINESS.md")
    # Read-only guarantee: detect external modification as well as accidental writes.
    for run in loaded:
        if any(file_sha(Path(run["run"]) / path) != digest for path, digest in run["input_hashes"].items()):
            raise ValueError("A frozen input artifact changed during comparison")
    return report


def markdown_report(report):
    pct = lambda value: f"{100 * value:.2f}%"
    population = report["population"]
    lines = ["# Phase 1C dual-backbone comparison", "", "Measured frozen-checkpoint results; test performance is not used for selection.", "",
             f"Population: {population['policy']}; {population['compared_segments']} segments, {population['recordings']} recordings.",
             f"Dataset SHA-256: `{report['dataset_sha256']}`.", "",
             "| Model | Top-1 | Top-5 | Macro-F1 | Weighted-F1 | Balanced accuracy |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["models"].items():
        metrics = row["metrics"]
        lines.append(f"| {name} | " + " | ".join(pct(metrics[key]) for key in
                     ("top1", "top5", "macro_f1_all_selected_classes", "weighted_f1", "balanced_accuracy")) + " |")
    lines += ["", "| Model | Never predicted classes | Zero-correct classes | Top-ten prediction concentration | Verb Top-1 | Noun Top-1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["models"].items():
        metrics = row["metrics"]
        lines.append(f"| {name} | {metrics['classes_never_predicted']} | {metrics['classes_zero_correct']} | {pct(metrics['prediction_concentration_top10'])} | {pct(metrics['verb_top1'])} | {pct(metrics['noun_top1'])} |")
    if "B0-original" in report["models"]:
        baseline = report["models"]["B0-original"]["metrics"]
        lines += ["", "Changes relative to B0-original on this same population (percentage points):", ""]
        for name, row in report["models"].items():
            if name == "B0-original":
                continue
            metrics = row["metrics"]
            top1_delta = 100 * (metrics["top1"] - baseline["top1"])
            f1_delta = 100 * (metrics["macro_f1_all_selected_classes"] - baseline["macro_f1_all_selected_classes"])
            coverage_delta = baseline["classes_zero_correct"] - metrics["classes_zero_correct"]
            lines.append(f"- {name}: Top-1 {top1_delta:+.2f} pp; macro-F1 {f1_delta:+.2f} pp; change in classes with at least one correct prediction {coverage_delta:+d}. These are descriptive differences, not a causal attribution to architecture or weighting.")
    lines += ["", "## Paired accuracy uncertainty", "",
              "Recording-cluster percentile intervals resample entire recordings. Values below are percentage points; positive favors the right model.", ""]
    for pair in report["paired_accuracy_differences"]:
        lo, hi = pair["ci95_percentile"]
        lines.append(f"- {pair['right']} minus {pair['left']}: {100 * pair['difference_right_minus_left']:+.2f} pp; 95% interval [{100 * lo:+.2f}, {100 * hi:+.2f}] pp ({pair['replicates']} replicates, seed {pair['seed']}).")
    lines += ["", "## Checkpoint and resource record", ""]
    for name, row in report["models"].items():
        resources = row["resources"]
        completion, provenance = resources.get("training_completion", {}), resources.get("training_provenance", {})
        criterion = completion.get("selection_criterion", provenance.get("selection_criterion", "minimum_validation_cross_entropy" if name == "B0-original" else "not recorded"))
        lines += [f"- {name}: best epoch {completion.get('best_epoch', 'not recorded')}; criterion `{criterion}`; parameters {provenance.get('parameter_count', 'not recorded')}; training seconds {completion.get('training_seconds', 'not recorded')}; GPU `{provenance.get('gpu', 'not recorded')}`; peak CUDA bytes {completion.get('peak_cuda_memory_bytes', 'not recorded')}; training Slurm job `{completion.get('slurm_job_id', provenance.get('slurm_job_id', 'not recorded'))}`. History: `{row['learning_curves']['history']}`; plots: `{row['learning_curves']['plots']}`."]
    lines += ["", "## Error patterns", "", "Full per-class recall/F1/support are in `per_class.csv`; complete confusion matrices and provenance are in `comparison.json`.", ""]
    for name, row in report["models"].items():
        matrix = row["metrics"]["confusion_matrix"]
        confusions = sorted(((value, i, j) for i, line in enumerate(matrix) for j, value in enumerate(line) if i != j and value), reverse=True)[:5]
        action = lambda index: report["class_mapping"][index].get("action_cls", str(report["class_mapping"][index]["action_id"]))
        lines.append(f"{name}:")
        lines += [f"- {action(i)} → {action(j)}: {value}." for value, i, j in confusions]
        lines.append("")
    lines += ["## Interpretation limits", ""] + [f"- {item}" for item in report["limitations"]]
    if population["policy"] != "full_frozen_test":
        lines += [f"- Explicit common-subset reason: {population['exclusion_reason']}. Excluded IDs are preserved per model in JSON. Original published full-test metrics remain unchanged."]
    return "\n".join(lines) + "\n"


def model_report(name, report):
    """Model-specific measured deliverable; no synthetic or pending scores inserted."""
    row = report["models"][name]
    resources = row["resources"]
    provenance, completion = resources.get("training_provenance", {}), resources.get("training_completion", {})
    config = provenance.get("config", {})
    weights = provenance.get("class_weights")
    pct = lambda value: f"{100 * value:.2f}%"
    lines = [f"# {name} measured results", "", f"Dataset SHA-256: `{report['dataset_sha256']}`.",
             f"Frozen source split counts: `{json.dumps(report['split_counts'], sort_keys=True)}`; vocabulary: {len(report['class_mapping'])} classes.",
             f"Held-out comparison population: {report['population']['policy']}, {report['population']['compared_segments']} segments.", "",
             "| Model | Top-1 | Top-5 | Macro-F1 | Weighted-F1 | Balanced accuracy |",
             "|---|---:|---:|---:|---:|---:|"]
    for model_name in ("B0-original", name):
        if model_name not in report["models"]:
            continue
        metrics = report["models"][model_name]["metrics"]
        lines.append(f"| {model_name} | " + " | ".join(pct(metrics[key]) for key in
                     ("top1", "top5", "macro_f1_all_selected_classes", "weighted_f1", "balanced_accuracy")) + " |")
    metrics = row["metrics"]
    lines += ["", f"Never-predicted classes: {metrics['classes_never_predicted']}; zero-correct classes: {metrics['classes_zero_correct']}; top-ten prediction concentration: {pct(metrics['prediction_concentration_top10'])}.",
              f"Probability-marginal verb Top-1: {pct(metrics['verb_top1'])}; noun Top-1: {pct(metrics['noun_top1'])}.", "",
              f"Checkpoint selection: `{completion.get('selection_criterion', provenance.get('selection_criterion', 'not recorded'))}`; best epoch: {completion.get('best_epoch', 'not recorded')}; epochs completed: {completion.get('epochs_completed', 'not recorded')}.",
              f"Parameters: {provenance.get('parameter_count', 'not recorded')}; training seconds: {completion.get('training_seconds', 'not recorded')}; sanity seconds: {completion.get('sanity_seconds', 'not recorded')}; GPU: `{provenance.get('gpu', 'not recorded')}`; peak allocated CUDA bytes: {completion.get('peak_cuda_memory_bytes', 'not recorded')}.",
              f"Training Slurm job: `{completion.get('slurm_job_id', provenance.get('slurm_job_id', 'not recorded'))}`; source commit: `{provenance.get('git_commit', 'not recorded')}`.",
              "", "## Training recipe and interpretation", ""]
    if name == "B0-balanced":
        lines += ["The controlled TCN treatment uses training-only inverse-square-root class-frequency weights, normalized to arithmetic mean one. The architecture, 64-step/216-channel pose representation and frozen population are retained. Checkpoint selection changes from the original minimum validation cross-entropy to maximum validation macro-F1. Consequently the measured difference includes both the prescribed imbalance treatment and the changed selection rule; it is not an isolated estimate of weighting's causal effect.", ""]
    else:
        upstream_path = Path(__file__).resolve().parents[1] / "reproducibility/upstreams/handformer.json"
        upstream = read_json(upstream_path)
        lines += [f"Official source: `{upstream.get('repository', upstream.get('url'))}`; pinned revision: `{upstream.get('revision', upstream.get('commit'))}`. The official pose-only HF_Pose architecture uses a 100-class head and its temporal/micro-action pathway through the compatibility adapter.", "",
                  "B1 uses the released pose-only training recipe with its documented adaptations, including an unweighted smoothed/temperature-scaled loss. B0-balanced uses class-weighted ordinary cross-entropy. Temporal sampling, normalization and loss differ between model families, so their difference cannot be attributed solely to architecture.", "",
                  "Documented preprocessing deviations:", ""]
        lines += [f"- {value}." for value in upstream.get("preprocessing_deviations", [])]
        lines += ["", "Compatibility-only source transformations:", ""]
        lines += [f"- {value}." for value in upstream.get("compatibility_transformations", [])]
        lines += ["", "The original missing-coordinate mask remains explicit in the source cache, but the official pose encoder receives coordinates without extra mask channels. Positive-confidence finite zero coordinates are preserved. No RGB features or geometry branch are included.", ""]
    if "B0-original" in report["models"]:
        baseline = report["models"]["B0-original"]["metrics"]
        delta_top1 = 100 * (metrics["top1"] - baseline["top1"])
        delta_f1 = 100 * (metrics["macro_f1_all_selected_classes"] - baseline["macro_f1_all_selected_classes"])
        lines += [f"Measured change versus the original on matched examples: Top-1 {delta_top1:+.2f} percentage points; macro-F1 {delta_f1:+.2f} percentage points. Both changes are reported, including any accuracy/coverage tradeoff. See the paired recording-cluster confidence interval and common confusion patterns in `DUAL_BACKBONE_COMPARISON.md`.", ""]
    lines += ["Captured configuration:", "", "```json", json.dumps(config, indent=2), "```", "",
              f"Training-only class counts, in frozen label order: `{json.dumps(provenance.get('class_counts'))}`.",
              f"Applied class weights, same order (`null` means unweighted): `{json.dumps(weights)}`.", "",
              "## Reproducibility and limitations", "",
              f"Run directory: `{row['run']}`. Learning history: `{row['learning_curves']['history']}`; measured loss/accuracy/F1/coverage plots: `{row['learning_curves']['plots']}`.",
              "The run retains its validation-selected checkpoint, train provenance, frozen test probabilities/predictions and evaluation metrics. `comparison.json` records their hashes, complete metrics and resource metadata; `per_class.csv` contains support, recall and F1 for every selected class. The original E0 run is not modified.", ""]
    lines += [f"- {value}" for value in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--balanced", required=True)
    parser.add_argument("--handformer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-common-subset", action="store_true")
    parser.add_argument("--exclusion-reason")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()
    report = compare(args.dataset, [("B0-original", args.original), ("B0-balanced", args.balanced),
                                    ("B1-HandFormer", args.handformer)], args.output,
                     allow_common_subset=args.allow_common_subset, exclusion_reason=args.exclusion_reason,
                     repeats=args.bootstrap_repeats)
    print(json.dumps({"output": str(Path(args.output).resolve()), "population": report["population"]["policy"],
                      "models": {name: row["metrics"]["top1"] for name, row in report["models"].items()}}))


if __name__ == "__main__":
    main()
