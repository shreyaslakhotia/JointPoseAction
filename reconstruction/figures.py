"""Publication figure helpers from supplied observations; no generated results."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from segmentation_pilot.metrics import binary_mask, mask_boundary


def silhouette_overlay(rgb: Any, reference: Any, rendered: Any, output: str | Path) -> Path:
    """Save native-resolution RGB with cyan reference / orange rendered contours."""
    from PIL import Image
    image = np.asarray(rgb)
    target, prediction = binary_mask(reference), binary_mask(rendered)
    if image.dtype != np.uint8 or image.shape != (*target.shape, 3) or target.shape != prediction.shape:
        raise ValueError("Overlay requires native uint8 RGB and same-resolution binary masks")
    canvas = image.copy()
    canvas[mask_boundary(target)] = (0, 220, 255)
    canvas[mask_boundary(prediction)] = (255, 140, 0)
    canvas[mask_boundary(target) & mask_boundary(prediction)] = (255, 255, 255)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
    return path


def plot_camera_metrics(report: Mapping[str, Any], output: str | Path, *, title: str) -> Path:
    """Plot available per-camera IoU; missing views explicitly labelled missing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["per_camera"]
    if not rows or not title:
        raise ValueError("A report and explicit figure title are required")
    labels = [f"{r['camera_id']}\n{r['partition']}" for r in rows]
    values = [r["metrics"]["iou"] if r["status"] == "available" else np.nan for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["#377eb8" if r["partition"] == "input" else "#e68613" for r in rows]
    ax.bar(np.arange(len(rows)), values, color=colors)
    for index, row in enumerate(rows):
        if row["status"] == "missing":
            ax.text(index, 0.03, "missing", ha="center", rotation=90, fontsize=8)
    ax.set(ylim=(0, 1.08), ylabel="Silhouette IoU", title=title)
    ax.set_xticks(np.arange(len(rows)), labels, fontsize=8)
    kinds = {r.get("measurement_kind") for r in rows if r["status"] == "available"}
    caption = "Predicted-mask agreement; not independent accuracy" if kinds == {"predicted_mask_agreement"} else "Reference kinds are distinguished in the source report"
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def comparison_markdown(named_reports: Mapping[str, Mapping[str, Any]]) -> str:
    """Show each reference kind and split separately, preserving unavailable values."""
    lines = ["| Method | Partition | Reference | Available views | IoU | Dice | Boundary F1 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for name, report in named_reports.items():
        safe_name = str(name).replace("|", "\\|").replace("\n", " ")
        for split in ("input", "heldout"):
            for kind in ("predicted", "independent_verified"):
                summary = report["summaries"][f"{split}_{kind}"]
                values = ["unavailable" if summary["mean"][key] is None else f"{summary['mean'][key]:.4f}"
                          for key in ("iou", "dice", "boundary_f1")]
                label = "prediction agreement" if kind == "predicted" else "independent reference"
                lines.append(f"| {safe_name} | {split} | {label} | {summary['available_cameras']} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"
