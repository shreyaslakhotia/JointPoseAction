"""Explicitly invoked single-run training; no job submission or test selection."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from .model import TCN
from .runtime import SegmentDataset, load_artifact, provenance, seed_all


def epoch(model, loader, device, optimizer=None, class_weights=None):
    model.train(optimizer is not None)
    total_loss, loss_denominator, total, correct = 0.0, 0.0, 0, 0
    probabilities, truth = [], []
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")
    for features, labels, _ in loader:
        features, labels = features.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(features)
            loss_sum = loss_fn(logits, labels)
            denominator = class_weights[labels].sum() if class_weights is not None else len(labels)
            loss = loss_sum / denominator
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training/evaluation loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        total_loss += loss_sum.item()
        loss_denominator += float(denominator)
        total += len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        probabilities.extend(logits.detach().softmax(1).cpu().numpy())
        truth.extend(labels.cpu().numpy())
    if not total:
        raise ValueError("Empty data loader")
    return {"loss": total_loss / loss_denominator, "top1": correct / total}, np.asarray(probabilities), np.asarray(truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if any(config[k] < 1 for k in ("batch_size", "epochs", "patience")) or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid training hyperparameters")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Real training requires an explicitly authorized Slurm GPU job")
    seed_all(config["seed"])
    artifact = load_artifact(args.dataset)
    record = provenance(args.dataset, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "dataset_summary.json").write_text(json.dumps(artifact["summary"], indent=2))
    train = SegmentDataset(args.dataset, artifact, "train")
    validation = SegmentDataset(args.dataset, artifact, "validation")
    device = torch.device("cuda")
    model_args = {"input_dim": artifact["summary"]["features"], "classes": len(artifact["classes"]), **config["model"]}
    model = TCN(**model_args).to(device)
    record.update(model_args=model_args, parameter_count=sum(p.numel() for p in model.parameters()),
                  class_mapping=artifact["classes"])
    (output / "provenance.json").write_text(json.dumps(record, indent=2))
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True,
                              generator=generator, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=config["batch_size"], num_workers=0)
    weights = None
    if config["class_weighting"] == "inverse_frequency":
        counts = np.bincount([x["label"] for x in train.rows], minlength=len(artifact["classes"]))
        if (counts == 0).any():
            raise ValueError("Empty training class")
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    elif config["class_weighting"] != "none":
        raise ValueError("Unknown weighting")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    best, stale = float("inf"), 0
    with open(output / "history.jsonl", "x", encoding="utf-8") as stream:
        for number in range(config["epochs"]):
            train_metrics, _, _ = epoch(model, train_loader, device, optimizer, weights)
            val_metrics, _, _ = epoch(model, validation_loader, device)
            line = {"epoch": number + 1, "train": train_metrics, "validation": val_metrics,
                    "learning_rate": optimizer.param_groups[0]["lr"]}
            stream.write(json.dumps(line) + "\n")
            stream.flush()
            print(json.dumps(line), flush=True)
            if val_metrics["loss"] < best:
                best, stale = val_metrics["loss"], 0
                torch.save({"model": model.state_dict(), "model_args": model_args,
                            "classes": artifact["classes"], "epoch": number + 1,
                            "selection": "minimum_validation_cross_entropy",
                            "validation_loss": best, "provenance": record}, output / "best.pt")
            else:
                stale += 1
            scheduler.step()
            if stale >= config["patience"]:
                break
    checkpoint = torch.load(output / "best.pt", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model"])
    _, p, y = epoch(model, validation_loader, device)
    (output / "validation_metrics.json").write_text(json.dumps(compute_metrics(p, y, artifact["classes"]), indent=2))
    (output / "training_completed.json").write_text(json.dumps({"best_epoch": checkpoint["epoch"], "test_evaluated": False}))


if __name__ == "__main__":
    main()
