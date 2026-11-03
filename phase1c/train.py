"""Single controlled fit per model, preceded by a bounded training-only sanity gate.

The existing E0 implementation and artifacts remain untouched. The public CLI
requires a Slurm GPU allocation; CPU callers are restricted to synthetic tests.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from e0.data import sha256
from e0.model import TCN
from e0.runtime import SegmentDataset, load_artifact, provenance, seed_all
from e0.sanity import select_examples
from .metrics import compute_metrics


def validate_config(config):
    if config.get("model_family") not in ("tcn", "handformer"):
        raise ValueError("Unknown recognition model family")
    if config.get("selection") != "validation_macro_f1":
        raise ValueError("Phase 1C selects checkpoints using validation macro-F1")
    for key in ("batch_size", "epochs", "patience"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid learning rate or weight decay")
    if config.get("class_weighting", "none") not in ("none", "inverse_sqrt_frequency"):
        raise ValueError("Only prespecified unweighted or inverse-square-root weighting is allowed")
    if config.get("gradient_clip", 1.0) is not None and config.get("gradient_clip", 1.0) <= 0:
        raise ValueError("Invalid gradient clipping limit")
    loss_options = config.get("loss", {})
    if not 0 <= loss_options.get("label_smoothing", 0.0) < 1 or loss_options.get("temperature", 1.0) <= 0:
        raise ValueError("Invalid loss smoothing or temperature")
    sanity = config.get("sanity", {})
    if not 1 <= sanity.get("steps", 500) <= 500 or not 2 <= sanity.get("classes", 4) <= 4 or not 1 <= sanity.get("per_class", 4) <= 4:
        raise ValueError("Sanity gate must remain bounded to <=16 examples / 500 updates")
    return config


def build_model(config, artifact):
    if config["model_family"] == "tcn":
        return TCN(input_dim=artifact["summary"]["features"], classes=len(artifact["classes"]), **config["model"])
    from .handformer import build_model as handformer_model
    return handformer_model(config, artifact)


def build_dataset(dataset_path, artifact, split, config, compatibility=None):
    if config["model_family"] == "tcn":
        dataset = SegmentDataset(dataset_path, artifact, split)
    else:
        from .handformer import build_dataset as handformer_dataset
        dataset = handformer_dataset(dataset_path, artifact, split, config, compatibility)
    expected = [row for row in artifact["segments"] if row["split"] == split]
    if len(dataset) != len(expected) or len(dataset.rows) != len(expected):
        raise ValueError("Adapter changed the frozen example population")
    keys = ("segment_id", "recording_id", "participant_id", "label", "action_id", "split")
    if any(any(row[key] != original[key] for key in keys) for row, original in zip(dataset.rows, expected)):
        raise ValueError("Adapter changed frozen segment ordering or labels")
    return dataset


def class_weights_from_training(rows, classes, treatment):
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Class weights must use training rows exclusively")
    counts = np.bincount([row["label"] for row in rows], minlength=classes)
    if len(counts) != classes or (counts == 0).any():
        raise ValueError("Every selected class needs training support")
    if treatment == "none":
        return counts, None
    if treatment != "inverse_sqrt_frequency":
        raise ValueError("Unknown class-weight treatment")
    weights = 1 / np.sqrt(counts.astype(np.float64))
    weights /= weights.mean()
    return counts, torch.as_tensor(weights, dtype=torch.float32)


def make_optimizer(model, config):
    options = dict(config.get("optimizer", {"name": "AdamW"}))
    name = options.pop("name")
    if name not in ("Adam", "AdamW", "SGD"):
        raise ValueError("Unsupported optimizer")
    return getattr(torch.optim, name)(model.parameters(), lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"], **options)


def make_scheduler(optimizer, config):
    options = dict(config.get("scheduler", {"name": "cosine"}))
    name = options.pop("name")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], **options)
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **options)
    if name == "warmup_multistep":
        return None  # Exact upstream epoch-start schedule, applied by set_epoch_learning_rate.
    if name == "none":
        return None
    raise ValueError("Unsupported scheduler")


def set_epoch_learning_rate(optimizer, config, epoch_index):
    schedule = config.get("scheduler", {})
    if schedule.get("name") != "warmup_multistep":
        return
    warmup, milestones = schedule.get("warmup_epochs", 5), schedule.get("milestones", [25, 40])
    factor = (epoch_index + 1) / warmup if epoch_index < warmup else schedule.get("gamma", .1) ** sum(epoch_index >= point for point in milestones)
    for group in optimizer.param_groups:
        group["lr"] = config["learning_rate"] * factor


def epoch(model, loader, device, classes, optimizer=None, class_weights=None, gradient_clip=1.0, loss_options=None):
    """Record plain CE separately from the weighted mean optimized in fitting.

    PyTorch's weighted mean divides the summed weighted losses by the sum of
    weights for labels in each batch. Epoch optimization_loss uses the analogous
    aggregate denominator; it is not presented as ordinary cross-entropy.
    """
    model.train(optimizer is not None)
    summed_ce, summed_objective, objective_denominator, total = 0.0, 0.0, 0.0, 0
    probabilities, truth = [], []
    options = loss_options or {}
    for features, labels, _ in loader:
        features, labels = features.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(features)
            losses = nn.functional.cross_entropy(logits, labels, reduction="none")
            objective_losses = nn.functional.cross_entropy(logits / options.get("temperature", 1.0), labels,
                                                           reduction="none", label_smoothing=options.get("label_smoothing", 0.0))
            weights = class_weights[labels] if class_weights is not None else torch.ones_like(losses)
            numerator, denominator = (objective_losses * weights).sum(), weights.sum()
            loss = numerator / denominator
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite recognition loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
        summed_ce += float(losses.detach().sum())
        summed_objective += float(numerator.detach())
        objective_denominator += float(denominator.detach())
        total += len(labels)
        probabilities.append(logits.detach().softmax(1).cpu().numpy())
        truth.append(labels.cpu().numpy())
    if not total:
        raise ValueError("Empty split")
    p, y = np.concatenate(probabilities), np.concatenate(truth)
    metrics = compute_metrics(p, y, classes)
    # Keep history lightweight; full validation/test diagnostics are exported once.
    summary = {key: metrics[key] for key in ("top1", "top5", "macro_f1_all_selected_classes", "weighted_f1", "balanced_accuracy",
                                             "classes_never_predicted", "classes_zero_correct", "prediction_concentration_top10")}
    summary.update(loss=summed_ce / total, cross_entropy=summed_ce / total,
                   optimization_loss=summed_objective / objective_denominator,
                   optimization_is_weighted=class_weights is not None,
                   objective_label_smoothing=options.get("label_smoothing", 0.0),
                   objective_temperature=options.get("temperature", 1.0))
    return summary, p, y


def bounded_sanity(config, artifact, dataset, device):
    sanity = config.get("sanity", {})
    seed_all(config["seed"])
    selected = select_examples(dataset, classes=sanity.get("classes", 4), per_class=sanity.get("per_class", 4))
    model = build_model(config, artifact).to(device)
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.p = 0.0
        elif isinstance(layer, nn.MultiheadAttention):
            # Attention dropout is a scalar, not a child nn.Dropout module.
            layer.dropout = 0.0
    features = torch.stack([x for x, _, _ in selected]).to(device)
    labels = torch.tensor([y for _, y, _ in selected], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=sanity.get("learning_rate", .003), weight_decay=0)
    model.eval()
    with torch.no_grad():
        initial = float(nn.functional.cross_entropy(model(features), labels))
    passed, history = False, []
    limit = sanity.get("steps", 500)
    started = time.perf_counter()
    print(json.dumps({"stage": "sanity", "samples": len(selected), "maximum_steps": limit}), flush=True)
    for step in range(1, limit + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features), labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite sanity loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0 or step == limit:
            model.eval()
            with torch.no_grad():
                logits = model(features)
                value = float(nn.functional.cross_entropy(logits, labels))
                accuracy = float((logits.argmax(1) == labels).float().mean())
            history.append({"step": step, "loss": value, "top1": accuracy})
            if step % 50 == 0 or accuracy >= .95 or step == limit:
                print(json.dumps({"stage": "sanity", **history[-1]}), flush=True)
            passed = accuracy >= .95 and value <= initial * .25
            if passed:
                break
    return {"passed": passed, "samples": len(selected), "segment_ids": [sid for _, _, sid in selected],
            "selection_split": "train", "validation_used": False, "test_used": False,
            "dropout_disabled_for_sanity_only": True, "optimizer": "AdamW", "weight_decay": 0,
            "learning_rate": sanity.get("learning_rate", .003), "initial_loss": initial,
            "steps": step, "history": history, "seconds": time.perf_counter() - started,
            "baseline_reuses_sanity_weights": False}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fit_experiment(dataset_path, config, output, compatibility=None, *, device="cuda", allow_synthetic=False):
    config = validate_config(copy.deepcopy(config))
    artifact = load_artifact(dataset_path, allow_synthetic=allow_synthetic)
    synthetic = artifact["source_manifest"].get("status") == "synthetic"
    if str(device) != "cuda" and not synthetic:
        raise RuntimeError("Real training requires the approved Slurm GPU job")
    if not synthetic and (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise RuntimeError("Real training requires the approved Slurm GPU job")
    dataset_hash = sha256(dataset_path)
    if config.get("expected_dataset_sha256") != dataset_hash:
        raise ValueError("Frozen dataset hash differs from the approved population")
    output = Path(output)
    if (output / "training_completed.json").exists():
        raise FileExistsError("Completed training is immutable; no duplicate fit")
    output.mkdir(parents=True, exist_ok=False)
    record = provenance(dataset_path, config)
    record.update(model_family=config["model_family"], selection_criterion="maximum_validation_macro_f1",
                  compatibility_path=str(Path(compatibility).resolve()) if compatibility else None,
                  compatibility_sha256=sha256(compatibility) if compatibility else None,
                  class_mapping=artifact["classes"])
    train = build_dataset(dataset_path, artifact, "train", config, compatibility)
    validation = build_dataset(dataset_path, artifact, "validation", config, compatibility)
    counts, weights = class_weights_from_training(train.rows, len(artifact["classes"]), config.get("class_weighting", "none"))
    record.update(class_counts=counts.tolist(), class_weights=weights.tolist() if weights is not None else None,
                  weight_source="training_segments_only", weight_normalization="arithmetic_mean_one" if weights is not None else None)
    save_json(output / "provenance.json", record)
    save_json(output / "dataset_summary.json", artifact["summary"])
    sanity_config = copy.deepcopy(config)
    sanity_config["sanity_mode"] = True
    sanity_dataset = build_dataset(dataset_path, artifact, "train", sanity_config, compatibility)
    sanity = bounded_sanity(config, artifact, sanity_dataset, torch.device(device))
    save_json(output / "sanity.json", sanity)
    if not sanity["passed"]:
        raise RuntimeError("Bounded training-only sanity gate failed; full training did not start")
    # Reinitialize every stochastic component: no sanity weights/optimizer/RNG are reused.
    seed_all(config["seed"])
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_model(config, artifact).to(device)
    record.update(parameter_count=sum(p.numel() for p in model.parameters()),
                  model_args={"input_dim": artifact["summary"]["features"], "classes": len(artifact["classes"]), **config["model"]}
                  if config["model_family"] == "tcn" else config["model"])
    save_json(output / "provenance.json", record)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(validation, batch_size=config["batch_size"], num_workers=0)
    weights = weights.to(device) if weights is not None else None
    best, stale, started = -float("inf"), 0, time.perf_counter()
    with (output / "history.jsonl").open("x", encoding="utf-8") as stream:
        for number in range(config["epochs"]):
            epoch_started = time.perf_counter()
            set_epoch_learning_rate(optimizer, config, number)
            train_metrics, _, _ = epoch(model, train_loader, device, artifact["classes"], optimizer, weights,
                                         config.get("gradient_clip", 1.0), config.get("loss"))
            val_metrics, _, _ = epoch(model, val_loader, device, artifact["classes"])
            line = {"epoch": number + 1, "train": train_metrics, "validation": val_metrics,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.perf_counter() - epoch_started,
                    "elapsed_training_seconds": time.perf_counter() - started}
            stream.write(json.dumps(line) + "\n")
            stream.flush()
            print(json.dumps(line), flush=True)
            score = val_metrics["macro_f1_all_selected_classes"]
            if score > best:
                best, stale = score, 0
                torch.save({"model": model.state_dict(), "model_args": record["model_args"],
                            "model_family": config["model_family"], "config": config,
                            "classes": artifact["classes"], "epoch": number + 1,
                            "selection": "maximum_validation_macro_f1", "validation_metrics": val_metrics,
                            "provenance": record}, output / "best.pt")
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step()
            if stale >= config["patience"]:
                break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    _, p, y = epoch(model, val_loader, device, artifact["classes"])
    save_json(output / "validation_metrics.json", compute_metrics(p, y, artifact["classes"]))
    result = {"best_epoch": checkpoint["epoch"], "selection_criterion": checkpoint["selection"],
              "best_validation_macro_f1": best, "epochs_completed": number + 1,
              "training_seconds": time.perf_counter() - started, "sanity_seconds": sanity["seconds"],
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "test_evaluated": False,
              "dataset_sha256": dataset_hash, "checkpoint_sha256": sha256(output / "best.pt")}
    save_json(output / "training_completed.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compatibility")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Training CLI requires an explicitly approved Slurm GPU job")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(fit_experiment(args.dataset, config, args.output, args.compatibility), indent=2), flush=True)


if __name__ == "__main__":
    main()
