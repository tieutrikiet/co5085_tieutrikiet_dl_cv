from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

# Shared color palette for plots
BLUE, BLUE_DARK, GRAY = "#2a78d6", "#104281", "#c9c8c3"
TEXT_GRAY = "#52514e"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": TEXT_GRAY,
        "xtick.color": TEXT_GRAY,
        "ytick.color": TEXT_GRAY,
        "axes.grid": False,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "font.size": 9,
    })


def save_fig(fig, fig_dir: Path, name: str, show: bool = False) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_learning_curves(histories: Dict[str, dict], colors: Dict[str, str]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, h in histories.items():
        epochs = np.arange(1, len(h["val_loss"]) + 1)
        color = colors.get(name, BLUE)
        for ax, key in zip(axes, ["loss", "acc"]):
            ax.plot(epochs, h[f"train_{key}"], color=color, ls="--", lw=1.2)
            ax.plot(epochs, h[f"val_{key}"], color=color, lw=1.8, label=name)
        best = int(np.argmax(h["val_acc"]))
        axes[1].scatter(best + 1, h["val_acc"][best], color=color, s=25, zorder=3)  # selected checkpoint

    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes:
        ax.grid(True, axis="y")
    style = [Line2D([], [], color=TEXT_GRAY, ls="--", label="train"),
             Line2D([], [], color=TEXT_GRAY, label="val")]
    axes[0].legend(handles=style, frameon=False)
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names: Sequence[str], title: str, ax, show_ylabels: bool = True):
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(n), normalize="true")  # row-normalized (recall)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j] >= 0.005:  # hide ~0 cells for readability
                ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.5 else TEXT_GRAY)
    ax.set_xticks(range(n), class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n), class_names if show_ylabels else [], fontsize=7)
    ax.set(xlabel="Predicted", title=title)
    if show_ylabels:
        ax.set_ylabel("True")
    return cm


def plot_confusion_matrices(outputs: Dict[str, tuple], class_names: Sequence[str]):
    """outputs: {name: (y_true, y_pred, probs)} -> (fig, {name: cm})"""
    fig, axes = plt.subplots(1, len(outputs), figsize=(5 * len(outputs), 4.8), squeeze=False)
    cms = {}
    for k, (ax, (name, (y_true, y_pred, _))) in enumerate(zip(axes[0], outputs.items())):
        cms[name] = plot_confusion_matrix(y_true, y_pred, class_names, f"{name} — test (%)", ax,
                                          show_ylabels=(k == 0))
    fig.tight_layout()
    return fig, cms


def plot_misclassified(images: np.ndarray, y_true, y_pred, probs, class_names: Sequence[str],
                       title: str, n: int = 16, cols: int = 8):
    wrong = np.flatnonzero(y_pred != y_true)
    conf = probs[wrong, y_pred[wrong]]
    wrong = wrong[np.argsort(-conf)][:n]  # most confident mistakes
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"T: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]} ({probs[i, y_pred[i]]:.0%})",
                     fontsize=6.5)
    fig.suptitle(f"{title}: {np.sum(y_pred != y_true)} lỗi / {len(y_true)} — {len(wrong)} lỗi tự tin nhất")
    fig.tight_layout()
    return fig
