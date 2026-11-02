"""Bounded training-only overfit gate; fresh baseline initialization follows."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .model import TCN
from .runtime import SegmentDataset, load_artifact, provenance, seed_all


def select_examples(dataset, classes=4, per_class=4):
    labels = sorted({r["label"] for r in dataset.rows})[:classes]
    chosen, seen = [], set()
    for label in labels:
        count = 0
        for i, row in enumerate(dataset.rows):
            if row["label"] != label:
                continue
            x, y, _ = dataset[i]
            digest = hashlib.sha256(x.numpy().tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            chosen.append((x, y, row["segment_id"]))
            count += 1
            if count == per_class:
                break
        if count != per_class:
            raise ValueError("Insufficient distinct training features for bounded overfit test")
    if len(labels) < 2:
        raise ValueError("Sanity test requires at least two training classes")
    return chosen


def fit(model, features, labels, steps=500):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        initial = float(loss_fn(model(features), labels))
    history, passed = [], False
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite sanity loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                logits = model(features)
                value = float(loss_fn(logits, labels))
                accuracy = float((logits.argmax(1) == labels).float().mean())
            history.append({"step": step, "loss": value, "top1": accuracy})
            passed = accuracy >= .95 and value <= initial * .25
            if passed:
                break
    return {"passed": passed, "initial_loss": initial, "steps": step,
            "final_loss": history[-1]["loss"], "final_top1": history[-1]["top1"], "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Real overfit gate requires the authorized Slurm GPU job")
    config = json.loads(Path(args.config).read_text())
    seed_all(config["seed"])
    artifact = load_artifact(args.dataset)
    dataset = SegmentDataset(args.dataset, artifact, "train")
    chosen = select_examples(dataset)
    device = torch.device("cuda")
    model_args = {"input_dim": artifact["summary"]["features"], "classes": len(artifact["classes"]),
                  **config["model"], "dropout": 0.0}
    model = TCN(**model_args).to(device)
    result = fit(model, torch.stack([x for x, _, _ in chosen]).to(device),
                 torch.tensor([y for _, y, _ in chosen], device=device))
    result.update(samples=len(chosen), segment_ids=[sid for _, _, sid in chosen],
                  selection_split="train", validation_used=False, test_used=False,
                  baseline_reuses_sanity_weights=False, model_args=model_args,
                  provenance=provenance(args.dataset, {"purpose": "bounded_overfit_gate", "maximum_steps": 500}))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "sanity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("Sanity gate failed; baseline training must not start")


if __name__ == "__main__":
    main()
