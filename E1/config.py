from pathlib import Path

import numpy as np

from common.datasets import FASHION_MNIST_CLASSES, FASHION_MNIST_SHAPE

# Paths are anchored to this file, so the pipeline runs from any working directory
E1_DIR = Path(__file__).resolve().parent
REPO_DIR = E1_DIR.parent

CONFIG = {
    "dataset": "FashionMNIST",
    "data_dir": REPO_DIR.parent / "fashion-mnist",
    "seed": 42,
    "val_ratio": 0.1,  # 10%
    "batch_size": 128,
    "epochs": 50,
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "metric": "val_accuracy",
    "patience": 5,
}

SPLIT_PATH = E1_DIR / f"split_indices_seed{CONFIG['seed']}.npz"
FIG_DIR = REPO_DIR / "figures"
RESULTS_DIR = E1_DIR / "results"

CLASS_NAMES = FASHION_MNIST_CLASSES
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SHAPE = FASHION_MNIST_SHAPE      # (C, H, W)
INPUT_DIM = int(np.prod(IMAGE_SHAPE))  # flattened size

SAMPLES_PER_CLASS = 8
MODEL_COLORS = {"Softmax": "#DE7272", "MLP": "#55A9A0", "CNN": "#F2A900"}
