"""Food-101 data pipeline for A1: split manifest, integrity checks, lazy Dataset and DataLoaders.

Images are decoded from disk per sample, so the 101k-image dataset is never held in memory.
The Dataset lives in a module (not the notebook) so DataLoader workers can pickle it under
macOS's `spawn` start method.
"""
import csv
import hashlib
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = ["image_id", "class_index", "split", "sha256", "width", "height", "mode"]


# ---------------------------------------------------------------- metadata

def read_classes(data_dir: Path) -> List[str]:
    """Class names sorted lexicographically (index = position)."""
    names = (data_dir / "meta" / "classes.txt").read_text().split()
    return sorted(names)


def read_official_split(data_dir: Path, name: str) -> List[str]:
    """Image IDs such as `apple_pie/1005649` from meta/{train,test}.txt."""
    return (data_dir / "meta" / f"{name}.txt").read_text().split()


def image_path(data_dir: Path, image_id: str) -> Path:
    return data_dir / "images" / f"{image_id}.jpg"


# ---------------------------------------------------------------- split + manifest

def assign_splits(data_dir: Path, classes: Sequence[str], val_per_class: int, split_seed: int) -> List[dict]:
    """Carve validation out of the official train list, per class, with a per-class RNG.

    Within each class: sort IDs, permute with default_rng(split_seed + class_index),
    first `val_per_class` -> val, rest -> train. The official test list is kept as-is.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    by_class: Dict[str, List[str]] = {c: [] for c in classes}
    for image_id in read_official_split(data_dir, "train"):
        by_class[image_id.split("/")[0]].append(image_id)

    rows = []
    for name in classes:
        idx = class_to_idx[name]
        ids = sorted(by_class[name])
        order = np.random.default_rng(split_seed + idx).permutation(len(ids))
        for rank, j in enumerate(order):
            rows.append({"image_id": ids[j], "class_index": idx, "split": "val" if rank < val_per_class else "train"})
    for image_id in sorted(read_official_split(data_dir, "test")):
        rows.append({"image_id": image_id, "class_index": class_to_idx[image_id.split("/")[0]], "split": "test"})
    return rows


def inspect_image(path: Path) -> dict:
    """SHA-256 of the raw bytes plus a decode check (size/mode; error string if unreadable)."""
    data = path.read_bytes()
    info = {"sha256": hashlib.sha256(data).hexdigest()}
    try:
        with Image.open(path) as img:
            img.load()
            info.update(width=img.width, height=img.height, mode=img.mode)
    except Exception as exc:  # corrupt / unreadable file
        info.update(width=-1, height=-1, mode=f"ERROR: {exc}")
    return info


def build_manifest(data_dir: Path, classes: Sequence[str], val_per_class: int, split_seed: int,
                   workers: int = 16) -> List[dict]:
    rows = assign_splits(data_dir, classes, val_per_class, split_seed)
    paths = [image_path(data_dir, r["image_id"]) for r in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:  # I/O-bound: threads are enough
        for row, info in zip(rows, pool.map(inspect_image, paths, chunksize=256)):
            row.update(info)
    return rows


def save_manifest(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["class_index"], r["width"], r["height"] = int(r["class_index"]), int(r["width"]), int(r["height"])
    return rows


def load_or_build_manifest(path: Path, data_dir: Path, classes: Sequence[str], val_per_class: int,
                           split_seed: int) -> List[dict]:
    """Reuse the locked manifest so every experiment sees exactly the same split."""
    if path.exists():
        return load_manifest(path)
    rows = build_manifest(data_dir, classes, val_per_class, split_seed)
    save_manifest(rows, path)
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- integrity checks

def check_manifest(rows: List[dict], num_classes: int, expected: Dict[str, int]) -> Dict[str, object]:
    """Counts per split/class, disjoint IDs, unreadable files and exact-hash duplicates."""
    report: Dict[str, object] = {}
    per_split = Counter(r["split"] for r in rows)
    report["counts"] = dict(per_split)
    report["counts_ok"] = all(per_split[s] == expected[s] * num_classes for s in SPLITS)

    per_class = Counter((r["split"], r["class_index"]) for r in rows)
    report["per_class_ok"] = all(per_class[(s, c)] == expected[s] for s in SPLITS for c in range(num_classes))

    ids = {s: {r["image_id"] for r in rows if r["split"] == s} for s in SPLITS}
    report["disjoint_ids"] = all(not (ids[a] & ids[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:])

    report["unreadable"] = [r["image_id"] for r in rows if r["width"] < 0]
    report["non_rgb"] = Counter(r["mode"] for r in rows if r["mode"] != "RGB" and r["width"] >= 0)

    by_hash: Dict[str, List[dict]] = {}
    for r in rows:
        by_hash.setdefault(r["sha256"], []).append(r)
    dup_groups = [g for g in by_hash.values() if len(g) > 1]
    report["duplicate_groups"] = len(dup_groups)
    report["cross_split_duplicates"] = [
        [(r["image_id"], r["split"]) for r in g] for g in dup_groups if len({r["split"] for r in g}) > 1
    ]
    return report


# ---------------------------------------------------------------- transforms

def build_transforms(train: bool, image_size: int = 224, resize_size: int = 256):
    """Shared protocol: RRC(0.8-1.0) + flip for train; resize 256 -> center crop 224 for eval."""
    if train:
        spatial = [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.75, 4 / 3),
                                interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
        ]
    else:
        spatial = [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(image_size),
        ]
    return T.Compose([
        T.ToImage(),
        *spatial,
        T.ToDtype(torch.float32, scale=True),  # uint8 -> [0, 1]
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[.., 3, H, W] normalized -> [0, 1] for plotting."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


# ---------------------------------------------------------------- dataset + loaders

class Food101Dataset(Dataset):
    """Lazy dataset: keeps only (path, label) pairs; decodes each JPEG on access."""

    def __init__(self, rows: List[dict], data_dir: Path, transform=None):
        self.image_ids = [r["image_id"] for r in rows]
        self.labels = [r["class_index"] for r in rows]
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        with Image.open(image_path(self.data_dir, self.image_ids[i])) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


def seed_worker(worker_id: int) -> None:
    """Derive Python/NumPy seeds from the torch seed each worker receives from the loader generator."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_datasets(rows: List[dict], data_dir: Path, image_size: int = 224) -> Dict[str, Food101Dataset]:
    return {
        s: Food101Dataset([r for r in rows if r["split"] == s], data_dir, build_transforms(s == "train", image_size))
        for s in SPLITS
    }


def make_loaders(datasets: Dict[str, Food101Dataset], batch_size: int, seed: int, num_workers: int = 4,
                 pin_memory: bool = False, persistent_workers: Optional[bool] = None) -> Dict[str, DataLoader]:
    """Train shuffles each epoch with a seeded generator and keeps the last incomplete batch."""
    persistent = num_workers > 0 if persistent_workers is None else persistent_workers
    loaders = {}
    for split, ds in datasets.items():
        g = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            worker_init_fn=seed_worker,
            generator=g,
        )
    return loaders
