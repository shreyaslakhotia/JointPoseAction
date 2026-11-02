"""Held-out evaluation of a validation-selected checkpoint; never selects epochs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import sha256
from .metrics import baselines, compute_metrics
from .model import TCN
from .runtime import SegmentDataset, load_artifact, provenance
from .train import epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = load_artifact(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["selection"] != "minimum_validation_cross_entropy":
        raise ValueError("Checkpoint was not selected by validation")
    if checkpoint["provenance"]["dataset_sha256"] != sha256(args.dataset) or checkpoint["classes"] != artifact["classes"]:
        raise ValueError("Evaluation data/vocabulary differ from training")
    model = TCN(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model"])
    dataset = SegmentDataset(args.dataset, artifact, "test")
    _, probabilities, labels = epoch(model, DataLoader(dataset, batch_size=64), torch.device("cpu"))
    metrics = compute_metrics(probabilities, labels, artifact["classes"])
    train_y = [x["label"] for x in artifact["segments"] if x["split"] == "train"]
    metrics["baselines"] = baselines(train_y, labels, len(artifact["classes"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metrics["provenance"] = provenance(args.dataset, {"checkpoint": str(Path(args.checkpoint).resolve()),
                                                       "checkpoint_sha256": sha256(args.checkpoint),
                                                       "training_commit": checkpoint["provenance"]["git_commit"]})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output / "probabilities.npz", probabilities=probabilities, labels=labels)
    with open(output / "predictions.csv", "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["segment_id", "recording_id", "participant_id", "true_action_id", "predicted_action_id", "valid_local_fraction"])
        writer.writeheader()
        for row, prediction in zip(dataset.rows, probabilities.argmax(1)):
            writer.writerow({"segment_id": row["segment_id"], "recording_id": row["recording_id"],
                             "participant_id": row["participant_id"], "true_action_id": row["action_id"],
                             "predicted_action_id": artifact["classes"][prediction]["action_id"],
                             "valid_local_fraction": row["quality"]["valid_local_fraction"]})
    from .plots import plot_results
    plot_results(Path(args.checkpoint).parent / "history.jsonl", artifact, metrics, output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
