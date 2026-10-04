"""Report-ready figures. EDA image examples are drawn from training only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # File generation also works on machines without a display.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from .config import CLASSES
from .data import resolve_image, split_counts

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220})
BLUE, GOLD, GRAY = "#3267A8", "#B77A1D", "#727D88"
NAMES = [name.replace("_", " ") for name in CLASSES]


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_eda(rows, root, output, seed=42):
    output = Path(output)
    counts = [sum(r["class"] == name for r in rows) for name in CLASSES]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    bars = ax.barh(NAMES, counts, color=BLUE)
    ax.bar_label(bars, padding=4)
    ax.set(xlabel="Number of images", title=f"NEU class distribution (n = {len(rows)})")
    ax.set_xlim(0, max(counts) * 1.18)
    save_figure(fig, output / "class_distribution.png")

    groups = split_counts(rows)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x = np.arange(len(CLASSES))
    for i, (split, color, hatch) in enumerate(zip(("train", "val", "test"), (BLUE, GOLD, GRAY), ("", "//", ".."))):
        bars = ax.bar(x + (i - 1) * 0.25, [groups[split][n] for n in CLASSES], width=0.25,
                      label=split, color=color, hatch=hatch, edgecolor="white")
        ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_xticks(x, [n.replace(" ", "\n") for n in NAMES])
    ax.set(ylabel="Number of images", title="Frozen train / validation / test distribution")
    ax.legend()
    ax.margins(y=0.15)
    save_figure(fig, output / "split_distribution.png")

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(3, 6, figsize=(14, 7), layout="constrained")
    for column, name in enumerate(CLASSES):
        samples = [r for r in rows if r["class"] == name and r["split"] == "train"]
        for index, row in enumerate(rng.choice(samples, size=min(3, len(samples)), replace=False)):
            with Image.open(resolve_image(root, row["path"])) as image:
                axes[index, column].imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
        for index in range(3):
            axes[index, column].axis("off")
        axes[0, column].set_title(NAMES[column], fontsize=11)
    fig.suptitle("Representative training images (original resolution)")
    save_figure(fig, output / "sample_grid.png")


def plot_history(history, output):
    for metric, title in (("accuracy", "Accuracy"), ("loss", "Cross-entropy loss")):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        epochs = np.arange(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], color=BLUE, label="Training", marker="o", markersize=3)
        ax.plot(epochs, history[f"val_{metric}"], color=GOLD, label="Validation", linestyle="--", marker="s", markersize=3)
        ax.set(xlabel="Epoch", ylabel=title, title=f"Training and validation {title.lower()}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0.5, len(epochs) + 0.5)
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        save_figure(fig, Path(output) / f"training_{metric}.png")


def plot_confusion(matrix, path, normalized=False):
    values = matrix.astype(float)
    if normalized:
        values = np.divide(values, values.sum(axis=1, keepdims=True), out=np.zeros_like(values),
                           where=values.sum(axis=1, keepdims=True) != 0)
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    chart = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    fig.colorbar(chart, ax=ax, label="Fraction of true class" if normalized else "Image count")
    ax.set_xticks(range(6), NAMES, rotation=35, ha="right")
    ax.set_yticks(range(6), NAMES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Held-out test confusion matrix")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{values[i, j]:.2f}" if normalized else str(matrix[i, j]),
                    ha="center", va="center", color="white" if values[i, j] > values.max() / 2 else "black")
    save_figure(fig, path)


def plot_predictions(rows, root, probabilities, path, seed=42):
    rng = np.random.default_rng(seed)
    # Random examples independent of correctness; selection is not cherry-picked.
    indices = rng.choice(len(rows), size=min(12, len(rows)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        row = rows[index]
        prediction = int(probabilities[index].argmax())
        with Image.open(resolve_image(root, row["path"])) as image:
            ax.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"True: {row['class']}\nPred: {CLASSES[prediction]}\n"
                     f"Confidence: {probabilities[index, prediction]:.1%}", fontsize=10)
    fig.suptitle("Random held-out test predictions; confidence is a softmax score")
    save_figure(fig, path)
