import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Fashion-MNIST metadata (label order from the official GitHub repository)
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
FASHION_MNIST_SHAPE = (1, 28, 28)  # (C, H, W)


# ---------------------------------------------------------------- file loading

def read_idx(filename: Path) -> np.ndarray:
    """Read an IDX file and return it as a NumPy array."""
    with open(filename, "rb") as f:
        zero, data_type, dims = struct.unpack(">HBB", f.read(4))
        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_fashion_mnist(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Fashion-MNIST dataset from the specified directory following Github repository structure."""
    train_images = read_idx(data_dir / "train-images-idx3-ubyte")
    train_labels = read_idx(data_dir / "train-labels-idx1-ubyte")
    test_images = read_idx(data_dir / "t10k-images-idx3-ubyte")
    test_labels = read_idx(data_dir / "t10k-labels-idx1-ubyte")
    return train_images, train_labels, test_images, test_labels


# ---------------------------------------------------------------- split

def load_or_create_split(labels: np.ndarray, split_path: Path, seed: int, val_ratio: float):
    """Reuse the saved split so later experiments use exactly the same samples."""
    if split_path.exists():
        with np.load(split_path) as split_indices:
            assert int(split_indices["seed"]) == seed, "Saved split seed does not match the current seed"
            return split_indices["train_indices"], split_indices["val_indices"]

    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,  # Ensure the split maintains the label distribution
        shuffle=True,
    )
    train_indices, val_indices = np.sort(train_indices), np.sort(val_indices)
    np.savez(split_path, seed=seed, train_indices=train_indices, val_indices=val_indices)
    return train_indices, val_indices


# ---------------------------------------------------------------- preprocessing

def pixel_stats(images: np.ndarray) -> Tuple[float, float]:
    """Mean/std on the [0, 1] scale — compute on the training split only."""
    x = images.astype(np.float32) / 255.0
    return float(x.mean()), float(x.std())


def preprocess(images: np.ndarray, mean: float, std: float) -> torch.Tensor:
    # uint8 [N, H, W] -> normalized float32 [N, 1, H, W]
    x = torch.tensor(images, dtype=torch.float32).div(255.0).unsqueeze(1)
    return (x - mean) / std


def to_tensor_dataset(images: np.ndarray, labels: np.ndarray, mean: float, std: float) -> TensorDataset:
    return TensorDataset(preprocess(images, mean, std), torch.tensor(labels, dtype=torch.long))


# ---------------------------------------------------------------- loaders

def make_train_loader(train_ds: TensorDataset, batch_size: int, seed: int) -> DataLoader:
    """A fresh generator with the same seed gives every model the same mini-batch order"""
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def make_eval_loaders(datasets: Dict[str, TensorDataset], batch_size: int) -> Dict[str, DataLoader]:
    """Unshuffled loaders for evaluation; 'train' measures train accuracy in eval mode."""
    return {split: DataLoader(ds, batch_size=batch_size) for split, ds in datasets.items()}
