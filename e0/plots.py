"""Static diagnostics generated only when measured history/predictions exist."""
import json
import numpy as np


def plot_results(history_path, artifact, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    for metric in ("loss", "top1"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for split in ("train", "validation"):
            ax.plot([x["epoch"] for x in history], [x[split][metric] for x in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = np.asarray(metrics["confusion_matrix"])
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title="Held-out segment confusion counts")
    fig.tight_layout()
    fig.savefig(output / "confusion.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [str(x["action_id"]) for x in artifact["classes"]]
    width = 0.25
    x = np.arange(len(labels))
    for offset, split in enumerate(("train", "validation", "test")):
        counts = artifact["summary"]["class_counts"][split]
        ax.bar(x + offset * width, [counts.get(label, 0) for label in labels], width, label=split)
    ax.set_xticks(x + width, labels, rotation=90)
    ax.set(xlabel="Official action ID", ylabel="Segments")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "class_counts.png", dpi=160)
    plt.close(fig)
