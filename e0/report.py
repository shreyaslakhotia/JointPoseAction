"""Concise measured E0 report assembled after the single final evaluation."""
import json
from pathlib import Path


def write_report(run, dataset_path):
    run = Path(run)
    artifact = json.loads(Path(dataset_path).read_text())
    metrics = json.loads((run / "test/metrics.json").read_text())
    completion = json.loads((run / "train/training_completed.json").read_text())
    submission = json.loads((run / "submission.json").read_text())
    sanity = json.loads((run / "sanity/sanity.json").read_text())
    classes = artifact["classes"]
    matrix = metrics["confusion_matrix"]
    confusions = sorted([(int(n), i, j) for i, row in enumerate(matrix) for j, n in enumerate(row) if i != j and n], reverse=True)[:10]
    weakest = sorted(metrics["per_class"], key=lambda x: (x["f1"], -x["support"]))[:10]
    report = {"top1": metrics["top1"], "top5": metrics["top5"],
              "macro_f1": metrics["macro_f1_all_selected_classes"], "baselines": metrics["baselines"],
              "best_epoch": completion["best_epoch"], "jobs": submission["jobs"],
              "split_counts": artifact["summary"]["split_counts"],
              "exclusions": artifact["summary"]["exclusions"], "classes": len(classes),
              "weakest_classes": weakest,
              "common_confusions": [{"count": n, "true_action": classes[i]["action_cls"],
                                     "predicted_action": classes[j]["action_cls"]} for n, i, j in confusions],
              "sanity_passed": sanity["passed"], "dataset_sha256": metrics["provenance"]["dataset_sha256"],
              "git_commit": submission["git_commit"], "scope": "selected Assembly101 pose-only subset; official recording-disjoint splits"}
    (run / "results_summary.json").write_text(json.dumps(report, indent=2))
    pct = lambda n: f"{100 * n:.2f}%"
    lines = ["# E0 controlled baseline results", "", report["scope"] + ". Participants overlap across splits.", "",
             f"Classes: {len(classes)}. Retained segments: {json.dumps(report['split_counts'])}.",
             f"Training-only sanity passed: {sanity['passed']}. Best epoch: {completion['best_epoch']} (validation cross-entropy).", "",
             "| Held-out measure | Result |", "|---|---:|",
             f"| Top-1 | {pct(metrics['top1'])} |", f"| Top-5 | {pct(metrics['top5'])} |",
             f"| Macro-F1 | {metrics['macro_f1_all_selected_classes']:.4f} |",
             f"| Training-majority Top-1 | {pct(metrics['baselines']['majority_top1'])} |",
             f"| Uniform chance Top-1 | {pct(metrics['baselines']['uniform_chance_top1'])} |", "",
             "## Error analysis", "", "Largest off-diagonal confusion counts:", ""]
    lines += [f"- {x['true_action']} → {x['predicted_action']}: {x['count']} segments." for x in report["common_confusions"]]
    lines += ["", "Lowest per-class F1 (support shown to expose small evaluation groups):", ""]
    lines += [f"- {classes[x['label']]['action_cls']}: F1 {x['f1']:.3f}, support {x['support']}." for x in weakest]
    lines += ["", "These are descriptive error counts; this run does not establish their causal explanation.", "",
              "## Reproducibility", "", f"Commit: `{submission['git_commit']}`.",
              f"Dataset SHA-256: `{report['dataset_sha256']}`.", f"Jobs: `{json.dumps(submission['jobs'])}`.",
              "Configuration: repo `configs/e0_tcn.json`; captured in `train/provenance.json`.",
              "Measured curves: `test/loss.png`, `test/top1.png`; confusion and class coverage plots are in `test/`.",
              "Prediction table/probabilities and full metrics are in `test/`. Training history is `train/history.jsonl`.",
              "The validation-selected checkpoint is `train/best.pt`. Test evaluation occurs only after training completion.",
              "Features are dimensionless; metric hand/object calibration and participant-independent generalization are not claimed.",
              "No broad hyperparameter search or E3 experiment was performed.", ""]
    (run / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return report
