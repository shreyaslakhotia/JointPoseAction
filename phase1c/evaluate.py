"""One final held-out evaluation of an immutable validation-selected model."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from e0.data import sha256
from e0.metrics import baselines
from e0.runtime import load_artifact, provenance
from .metrics import compute_metrics
from .train import build_dataset, build_model, epoch, save_json, validate_config


def evaluate_experiment(dataset_path, checkpoint_path, output, compatibility=None, *, allow_synthetic=False, plots=True):
    checkpoint_path = Path(checkpoint_path)
    marker = checkpoint_path.parent / "training_completed.json"
    if not marker.is_file():
        raise ValueError("Final evaluation requires the completed training marker")
    completed = json.loads(marker.read_text(encoding="utf-8"))
    if completed.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("Training completion/checkpoint hash mismatch")
    artifact = load_artifact(dataset_path, allow_synthetic=allow_synthetic)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = validate_config(checkpoint["config"])
    expected = sha256(dataset_path)
    if checkpoint["selection"] != "maximum_validation_macro_f1":
        raise ValueError("Checkpoint was not selected by the frozen validation criterion")
    if config.get("expected_dataset_sha256") != expected or checkpoint["provenance"]["dataset_sha256"] != expected:
        raise ValueError("Evaluation dataset differs from the frozen training population")
    if checkpoint["classes"] != artifact["classes"] or checkpoint["model_family"] != config["model_family"]:
        raise ValueError("Evaluation class mapping or model family differs from training")
    compatibility = compatibility or checkpoint["provenance"].get("compatibility_path")
    if compatibility:
        if checkpoint["provenance"].get("compatibility_sha256") != sha256(compatibility):
            raise ValueError("Compatibility preprocessing differs from training")
    elif checkpoint["provenance"].get("compatibility_sha256"):
        raise ValueError("Missing pinned compatibility preprocessing")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = build_model(config, artifact)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = build_dataset(dataset_path, artifact, "test", config, compatibility)
    summary, probabilities, labels = epoch(model, DataLoader(dataset, batch_size=config["batch_size"], num_workers=0),
                                            torch.device("cpu"), artifact["classes"])
    metrics = compute_metrics(probabilities, labels, artifact["classes"])
    metrics.update(cross_entropy=summary["cross_entropy"], best_epoch=checkpoint["epoch"],
                   selection_criterion=checkpoint["selection"], model_family=config["model_family"],
                   parameter_count=checkpoint["provenance"]["parameter_count"],
                   evaluation_seconds=time.perf_counter() - started)
    metrics["baselines"] = baselines([x["label"] for x in artifact["segments"] if x["split"] == "train"], labels, len(artifact["classes"]))
    metrics["provenance"] = provenance(dataset_path, {"checkpoint": str(checkpoint_path.resolve()),
                                                    "checkpoint_sha256": sha256(checkpoint_path),
                                                    "training_commit": checkpoint["provenance"]["git_commit"],
                                                    "compatibility_sha256": checkpoint["provenance"].get("compatibility_sha256")})
    save_json(output / "metrics.json", metrics)
    np.savez_compressed(output / "probabilities.npz", probabilities=probabilities, labels=labels,
                        segment_ids=np.asarray([row["segment_id"] for row in dataset.rows]))
    fields = ["segment_id", "recording_id", "participant_id", "true_action_id", "predicted_action_id", "valid_local_fraction"]
    with (output / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, prediction in zip(dataset.rows, probabilities.argmax(1)):
            writer.writerow({"segment_id": row["segment_id"], "recording_id": row["recording_id"],
                             "participant_id": row["participant_id"], "true_action_id": row["action_id"],
                             "predicted_action_id": artifact["classes"][prediction]["action_id"],
                             "valid_local_fraction": row.get("quality", {}).get("valid_local_fraction")})
    with (output / "per_class.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = list(metrics["per_class"][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    if plots:
        plot_results(checkpoint_path.parent / "history.jsonl", metrics, output)
    save_json(output / "evaluation_completed.json", {"dataset_sha256": expected, "checkpoint_sha256": sha256(checkpoint_path),
                                                    "metrics_sha256": sha256(output / "metrics.json"),
                                                    "export_sha256": {name: sha256(output / name) for name in
                                                        ("metrics.json", "predictions.csv", "probabilities.npz", "per_class.csv")},
                                                    "test_evaluated": True, "samples": len(labels)})
    return metrics


def plot_results(history_path, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
    for measure in ("cross_entropy", "top1", "macro_f1_all_selected_classes", "balanced_accuracy"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for split in ("train", "validation"):
            ax.plot([row["epoch"] for row in history], [row[split][measure] for row in history], label=split)
        ax.set(xlabel="Epoch", ylabel=measure)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{measure}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(metrics["confusion_matrix"], cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title="Held-out action confusion")
    fig.tight_layout()
    fig.savefig(output / "confusion.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compatibility")
    args = parser.parse_args()
    print(json.dumps(evaluate_experiment(args.dataset, args.checkpoint, args.output, args.compatibility), indent=2), flush=True)


if __name__ == "__main__":
    main()
