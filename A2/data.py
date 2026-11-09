"""Dữ liệu WIDERFace: đọc và làm sạch nhãn YOLO5Face, tạo và kiểm tra manifest.

Nhãn sinh từ `train2yolo.py` / `val2yolo.py`: mỗi ảnh một file `<stem>.txt`, mỗi dòng một mặt,
15 cột `0 cx cy w h x1 y1 ... x5 y5`, chuẩn hoá về [0, 1].

`train2yolo.py` chia cả giá trị -1 (thiếu landmark) cho kích thước ảnh, nên landmark thiếu ở train
là số âm nhỏ (vd. -0.00097). Mặt nào có toạ độ landmark âm được coi là không có landmark.
"""
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

N_COLS = 15                    # class + 4 bbox + 10 landmark
SPLITS = ("train", "val")      # test không có nhãn công khai, chỉ dùng để suy luận
SIZE_BINS = (10, 32, 96)       # nhóm kích thước mặt (px, theo cạnh ngắn của box)
MANIFEST_FIELDS = ["split", "image_path", "label_path", "orig_name", "width", "height",
                   "n_faces", "n_faces_lmk", "n_dropped", "sha256"]


def parse_label_file(path: Path) -> np.ndarray:
    """Đọc một file nhãn thành mảng (N, 15) float32; file rỗng -> (0, 15)."""
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != N_COLS:
            raise ValueError(f"{path}:{lineno}: cần {N_COLS} cột, có {len(values)}")
        rows.append([float(v) for v in values])
    return np.asarray(rows, dtype=np.float32).reshape(-1, N_COLS)


def clean_labels(labels: np.ndarray, img_w: int, img_h: int, min_px: float) -> Tuple[np.ndarray, dict]:
    """Bỏ box quá nhỏ, cắt box vào trong ảnh, đưa landmark thiếu về -1.

    Trả về mảng (M, 15) đã làm sạch và {"n_dropped", "n_lmk"}.
    """
    labels = labels.copy()

    # Bỏ box có cạnh < min_px (gồm cả w hoặc h <= 0)
    keep = (labels[:, 3] * img_w >= min_px) & (labels[:, 4] * img_h >= min_px)
    n_dropped = int((~keep).sum())
    labels = labels[keep]

    # Cắt box vào ảnh: cxcywh -> xyxy -> kẹp [0, 1] -> cxcywh
    cx, cy, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
    x1, y1 = np.clip(cx - w / 2, 0, 1), np.clip(cy - h / 2, 0, 1)
    x2, y2 = np.clip(cx + w / 2, 0, 1), np.clip(cy + h / 2, 0, 1)
    labels[:, 1], labels[:, 2] = (x1 + x2) / 2, (y1 + y2) / 2
    labels[:, 3], labels[:, 4] = x2 - x1, y2 - y1

    # Chỉ coi là có landmark khi cả 10 toạ độ đều không âm
    has_lmk = (labels[:, 5:] >= 0).all(axis=1)
    labels[~has_lmk, 5:] = -1

    return labels, {"n_dropped": n_dropped, "n_lmk": int(has_lmk.sum())}


def val_original_names(wider_dir: Path) -> List[str]:
    """Tên gốc ảnh val (vd. `0--Parade/0_Parade_marchingband_1_465.jpg`) theo thứ tự của val2yolo.

    val2yolo đánh số 0..N theo thứ tự dòng `#` trong val/label.txt, nên phần tử i là tên gốc của `val/i.jpg`.
    """
    lines = (wider_dir / "val" / "label.txt").read_text().splitlines()
    return [line.split()[-1] for line in lines if line.startswith("#")]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sort_key(path: Path):
    """Ảnh val tên 0.jpg, 1.jpg, ...: sắp theo số; train sắp theo tên."""
    return (0, int(path.stem), "") if path.stem.isdigit() else (1, 0, path.stem)


def inspect_sample(args) -> dict:
    """Kích thước ảnh (chỉ đọc header), SHA-256 và thống kê nhãn đã làm sạch của một ảnh."""
    split, img_path, yolo_dir, min_px = args
    label_path = img_path.with_suffix(".txt")
    with Image.open(img_path) as img:      # chỉ đọc header, không decode cả ảnh
        width, height = img.size
    labels, stats = clean_labels(parse_label_file(label_path), width, height, min_px)
    return {
        "split": split,
        "image_path": img_path.relative_to(yolo_dir).as_posix(),
        "label_path": label_path.relative_to(yolo_dir).as_posix(),
        "width": width,
        "height": height,
        "n_faces": len(labels),
        "n_faces_lmk": stats["n_lmk"],
        "n_dropped": stats["n_dropped"],
        "sha256": file_sha256(img_path),
    }


def build_manifest(yolo_dir: Path, wider_dir: Path, min_px: float, workers: int = 16) -> List[dict]:
    """Mỗi ảnh trong widerface_yolo/{train,val} một dòng; ảnh val có thêm tên gốc WIDERFace."""
    val_names = val_original_names(wider_dir)
    rows = []
    for split in SPLITS:
        images = sorted((yolo_dir / split).glob("*.jpg"), key=_sort_key)
        print(f"[manifest] {split}: {len(images)} ảnh")
        jobs = [(split, p, yolo_dir, min_px) for p in images]
        with ThreadPoolExecutor(max_workers=workers) as pool:  # chủ yếu là I/O nên dùng thread
            for i, row in enumerate(pool.map(inspect_sample, jobs, chunksize=64), start=1):
                if split == "val":
                    row["orig_name"] = val_names[int(Path(row["image_path"]).stem)]
                else:
                    row["orig_name"] = Path(row["image_path"]).name
                rows.append(row)
                if i % 2000 == 0 or i == len(jobs):
                    print(f"[manifest] {split}: {i}/{len(jobs)}")
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
        for key in ("width", "height", "n_faces", "n_faces_lmk", "n_dropped"):
            r[key] = int(r[key])
    return rows


def load_or_build_manifest(path: Path, yolo_dir: Path, wider_dir: Path, min_px: float,
                           rebuild: bool = False) -> List[dict]:
    """Dùng lại manifest đã khoá để mọi thí nghiệm thấy cùng một bộ dữ liệu."""
    if path.exists() and not rebuild:
        return load_manifest(path)
    rows = build_manifest(yolo_dir, wider_dir, min_px)
    save_manifest(rows, path)
    return rows


def load_clean_labels(row: dict, yolo_dir: Path, min_px: float) -> np.ndarray:
    """Nhãn (N, 15) đã làm sạch của một dòng manifest."""
    labels = parse_label_file(yolo_dir / row["label_path"])
    return clean_labels(labels, row["width"], row["height"], min_px)[0]


def face_size_histogram(rows: List[dict], yolo_dir: Path, min_px: float) -> np.ndarray:
    """Đếm số mặt theo nhóm kích thước: <10, 10-32, 32-96, >=96 px."""
    counts = np.zeros(len(SIZE_BINS) + 1, dtype=np.int64)
    for r in rows:
        labels = load_clean_labels(r, yolo_dir, min_px)
        side = np.minimum(labels[:, 3] * r["width"], labels[:, 4] * r["height"])
        counts += np.bincount(np.digitize(side, SIZE_BINS), minlength=len(counts))
    return counts


def check_manifest(rows: List[dict], yolo_dir: Path, min_px: float) -> dict:
    """In số ảnh, số mặt, box bị bỏ, tỉ lệ có landmark và phân bố kích thước mặt theo từng split."""
    bin_names = [f"<{SIZE_BINS[0]}"] + [f"{a}-{b}" for a, b in zip(SIZE_BINS, SIZE_BINS[1:])] + [f">={SIZE_BINS[-1]}"]
    summary = {}
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        n_faces = sum(r["n_faces"] for r in part)
        n_lmk = sum(r["n_faces_lmk"] for r in part)
        n_dropped = sum(r["n_dropped"] for r in part)
        sizes = face_size_histogram(part, yolo_dir, min_px)
        summary[split] = {
            "images": len(part),
            "faces_raw": n_faces + n_dropped,
            "faces": n_faces,
            "dropped": n_dropped,
            "faces_lmk": n_lmk,
            "no_face_images": sum(r["n_faces"] == 0 for r in part),
            "size_bins": dict(zip(bin_names, sizes.tolist())),
        }
        print(f"\n[{split}] ảnh={len(part)}  mặt={n_faces + n_dropped} -> {n_faces} "
              f"(bỏ {n_dropped} box < {min_px}px)  ảnh không có mặt={summary[split]['no_face_images']}")
        print(f"  có landmark: {n_lmk} ({n_lmk / max(n_faces, 1):.1%})")
        print("  kích thước mặt (px): " + "  ".join(f"{k}: {v} ({v / max(n_faces, 1):.1%})"
                                                    for k, v in summary[split]["size_bins"].items()))
    return summary
