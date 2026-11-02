"""Segment metrics and probability-marginal verb/noun predictions."""
import numpy as np


def compute_metrics(probabilities, labels, classes):
    p, original = np.asarray(probabilities, dtype=np.float64), np.asarray(labels)
    if original.ndim != 1 or original.dtype.kind not in "iu":
        raise ValueError("Expected integer label vector")
    y = original.astype(np.int64)
    c = len(classes)
    if p.ndim != 2 or p.shape != (len(y), c) or not len(y) or not np.isfinite(p).all():
        raise ValueError("Invalid/empty probabilities or labels")
    if (y < 0).any() or (y >= c).any() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid probability normalization or label range")
    pred = p.argmax(1)
    confusion = np.zeros((c, c), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    support, predicted = confusion.sum(1), confusion.sum(0)
    tp = confusion.diagonal()
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(c, dtype=float), where=(support + predicted) != 0)
    k = min(5, c)
    topk = np.argsort(-p, axis=1, kind="stable")[:, :k]
    result = {"samples": len(y), "classes": c, "top1": float((pred == y).mean()),
              "top5": float((topk == y[:, None]).any(1).mean()), "top5_k": k,
              "top5_note": "trivial when class count <= 5" if c <= 5 else None,
              "macro_f1_all_selected_classes": float(f1.mean()),
              "macro_f1_supported_classes": float(f1[support > 0].mean()),
              "confusion_matrix": confusion.tolist(),
              "per_class": [{"label": i, "action_id": item["action_id"],
                             "support": int(support[i]), "f1": float(f1[i]),
                             "accuracy": float(tp[i] / support[i]) if support[i] else None}
                            for i, item in enumerate(classes)]}
    for key in ("verb_id", "noun_id"):
        ids = sorted({item[key] for item in classes})
        marginals = np.stack([p[:, [i for i, item in enumerate(classes) if item[key] == value]].sum(1) for value in ids], axis=1)
        predicted_ids = np.asarray(ids)[marginals.argmax(1)]
        truth = np.asarray([classes[i][key] for i in y])
        result[key.replace("_id", "_top1")] = float((predicted_ids == truth).mean())
    return result


def baselines(train_labels, test_labels, classes):
    counts = np.bincount(train_labels, minlength=classes)
    if not counts.sum() or not len(test_labels):
        raise ValueError("Baselines require train and evaluation samples")
    majority = int(counts.argmax())
    return {"majority_label_from_train": majority,
            "majority_top1": float((np.asarray(test_labels) == majority).mean()),
            "uniform_chance_top1": 1 / classes, "uniform_chance_top5": min(5, classes) / classes,
            "training_frequency_chance_top1": float((counts / counts.sum())[test_labels].mean())}
