from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def describe_dataset(train_images, train_labels, test_images, test_labels) -> None:
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)
    print("Unique labels in the training set:", np.unique(train_labels).tolist())
    print("Pixel range:", train_images.min(), "to", train_images.max())
    print("Test pixel range:", test_images.min(), "to", test_images.max())


def describe_split(**arrays) -> None:
    for name, arr in arrays.items():
        print(f"{name} shape:", arr.shape)


def describe_pixels(images: np.ndarray) -> None:
    x = images.astype(np.float32) / 255.0  # scale to [0, 1], as ToTensor() will
    print(f"Pixel mean (train, [0,1] scale): {x.mean():.4f}")
    print(f"Pixel std  (train, [0,1] scale): {x.std():.4f}")
    print(f"Fraction of exactly-zero pixels : {(images == 0).mean():.1%}")


def plot_samples_per_class(images: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
                           samples_per_class: int = 8, seed: int = 42, title: str = "Training samples"):
    rng = np.random.default_rng(seed)
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, samples_per_class,
                             figsize=(samples_per_class * 0.9, n_classes * 0.95))
    for k, name in enumerate(class_names):
        picks = rng.choice(np.flatnonzero(labels == k), samples_per_class, replace=False)
        for j, i in enumerate(picks):
            ax = axes[k, j]
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[k, 0].set_ylabel(f"{k}: {name}", rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(f"{title} ({samples_per_class} per class)", y=0.995)
    fig.tight_layout()
    return fig
