"""Dataset, augmentation và DataLoader cho WIDERFace (định dạng YOLO5Face).

Ảnh được đọc từ local theo từng mẫu, không nạp cả tập vào RAM. Class Dataset nằm trong module
để worker của DataLoader import được khi macOS dùng `spawn`.

Bên trong, nhãn được giữ ở toạ độ pixel: box `xyxy` (N, 4), landmark (N, 10) và mặt nạ `has_lmk` (N,).
Đầu ra mỗi mẫu là nhãn (N, 16) giống yolov5-face: `[batch_idx, cls, cx, cy, w, h, x1, y1, ..., x5, y5]`,
chuẩn hoá về [0, 1]; landmark thiếu là -1.
"""
import math
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import load_clean_labels

PAD_VALUE = 114                    # màu xám dùng để đệm ảnh, giống yolov5
FLIP_PAIRS = ((0, 1), (3, 4))      # mắt trái/phải, khoé miệng trái/phải đổi chỗ khi lật ngang


def to_pixel(labels: np.ndarray, w: int, h: int):
    """Nhãn chuẩn hoá (N, 15) -> box xyxy pixel, landmark pixel, mặt nạ có landmark."""
    cx, cy, bw, bh = labels[:, 1] * w, labels[:, 2] * h, labels[:, 3] * w, labels[:, 4] * h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    has_lmk = labels[:, 5] >= 0
    lmk = labels[:, 5:].copy()
    lmk[:, 0::2] *= w
    lmk[:, 1::2] *= h
    lmk[~has_lmk] = -1
    return boxes.astype(np.float32), lmk.astype(np.float32), has_lmk


def to_normalized(boxes: np.ndarray, lmk: np.ndarray, has_lmk: np.ndarray, w: int, h: int) -> np.ndarray:
    """Ngược lại của `to_pixel`: trả về (N, 15) với cột class = 0."""
    out = np.zeros((len(boxes), 15), dtype=np.float32)
    out[:, 1] = (boxes[:, 0] + boxes[:, 2]) / 2 / w
    out[:, 2] = (boxes[:, 1] + boxes[:, 3]) / 2 / h
    out[:, 3] = (boxes[:, 2] - boxes[:, 0]) / w
    out[:, 4] = (boxes[:, 3] - boxes[:, 1]) / h
    out[:, 5::2] = lmk[:, 0::2] / w
    out[:, 6::2] = lmk[:, 1::2] / h
    out[~has_lmk, 5:] = -1
    return out


def letterbox(img: np.ndarray, size: int):
    """Đệm ảnh (đã resize sẵn) thành hình vuông `size`, ảnh nằm giữa. Trả về ảnh và (pad_x, pad_y)."""
    h, w = img.shape[:2]
    pad_x, pad_y = (size - w) // 2, (size - h) // 2
    out = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    out[pad_y:pad_y + h, pad_x:pad_x + w] = img
    return out, (pad_x, pad_y)


def augment_hsv(img: np.ndarray, hgain: float, sgain: float, vgain: float) -> None:
    """Đổi ngẫu nhiên hue/saturation/value, sửa trực tiếp trên ảnh BGR."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=img)


def random_affine(img, boxes, lmk, has_lmk, out_size: int, aug: dict, min_px: float):
    """Tịnh tiến, co giãn, nghiêng (và xoay nếu bật) rồi cắt về `out_size` x `out_size`.

    Box được biến đổi qua 4 góc rồi lấy khung bao; mặt có landmark rơi ra ngoài ảnh thì bỏ landmark.
    """
    h, w = img.shape[:2]
    C = np.eye(3)
    C[0, 2], C[1, 2] = -w / 2, -h / 2                       # đưa tâm ảnh về gốc toạ độ
    R = np.eye(3)
    angle = random.uniform(-aug["degrees"], aug["degrees"])
    scale = random.uniform(1 - aug["scale"], 1 + aug["scale"])
    R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
    S = np.eye(3)
    S[0, 1] = math.tan(math.radians(random.uniform(-aug["shear"], aug["shear"])))
    S[1, 0] = math.tan(math.radians(random.uniform(-aug["shear"], aug["shear"])))
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - aug["translate"], 0.5 + aug["translate"]) * out_size
    T[1, 2] = random.uniform(0.5 - aug["translate"], 0.5 + aug["translate"]) * out_size
    M = (T @ S @ R @ C)[:2]
    img = cv2.warpAffine(img, M, (out_size, out_size), borderValue=(PAD_VALUE,) * 3)

    n = len(boxes)
    if n:
        # Box: biến đổi 4 góc rồi lấy khung bao
        corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
        corners = np.c_[corners, np.ones(len(corners))] @ M.T
        corners = corners.reshape(n, 4, 2)
        new = np.c_[corners.min(1), corners.max(1)].clip(0, out_size)

        # Landmark: biến đổi từng điểm; điểm nào ra ngoài ảnh thì bỏ landmark của mặt đó
        pts = np.c_[lmk.reshape(-1, 2), np.ones(n * 5)] @ M.T
        pts = pts.reshape(n, 10)
        inside = ((pts >= 0) & (pts < out_size)).all(axis=1)
        has_lmk = has_lmk & inside
        pts[~has_lmk] = -1

        # Giữ box đủ lớn và không bị cắt quá nhiều so với kích thước sau co giãn
        bw, bh = new[:, 2] - new[:, 0], new[:, 3] - new[:, 1]
        area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * scale ** 2
        keep = (bw >= min_px) & (bh >= min_px) & (bw * bh / (area0 + 1e-9) > 0.1)
        boxes, lmk, has_lmk = new[keep].astype(np.float32), pts[keep].astype(np.float32), has_lmk[keep]
    return img, boxes, lmk, has_lmk


def flip_lr(img, boxes, lmk, has_lmk):
    """Lật ngang ảnh, box và landmark; đổi chỗ các điểm trái/phải để nhãn vẫn đúng nghĩa."""
    w = img.shape[1]
    img = np.ascontiguousarray(img[:, ::-1])
    boxes = boxes.copy()
    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    lmk = lmk.copy()
    lmk[has_lmk, 0::2] = w - lmk[has_lmk, 0::2]
    pts = lmk.reshape(-1, 5, 2)
    for a, b in FLIP_PAIRS:
        pts[:, [a, b]] = pts[:, [b, a]]
    return img, boxes, pts.reshape(-1, 10), has_lmk


class WiderFaceDataset(Dataset):
    """Đọc ảnh và nhãn theo manifest.

    `augment=True`: mosaic (theo xác suất) hoặc letterbox, rồi affine ngẫu nhiên, HSV và lật ngang.
    `augment=False`: chỉ resize cạnh dài về `img_size` và letterbox, dùng cho đánh giá.
    """

    def __init__(self, rows: List[dict], yolo_dir: Path, img_size: int, min_px: float,
                 augment: bool = False, aug: dict = None):
        self.rows = rows
        self.yolo_dir = Path(yolo_dir)
        self.img_size = img_size
        self.min_px = min_px
        self.augment = augment
        self.aug = aug or {}

    def __len__(self):
        return len(self.rows)

    def load_image(self, i: int):
        """Ảnh BGR đã resize để cạnh dài = img_size, kèm nhãn pixel theo ảnh đã resize."""
        row = self.rows[i]
        img = cv2.imread(str(self.yolo_dir / row["image_path"]))
        if img is None:
            raise FileNotFoundError(self.yolo_dir / row["image_path"])
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (round(w0 * r), round(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
        boxes, lmk, has_lmk = to_pixel(load_clean_labels(row, self.yolo_dir, self.min_px), w, h)
        return img, boxes, lmk, has_lmk, (h0, w0), r

    def load_letterbox(self, i: int):
        img, boxes, lmk, has_lmk, shape0, r = self.load_image(i)
        img, (px, py) = letterbox(img, self.img_size)
        boxes = boxes + [px, py, px, py]
        lmk = np.where(has_lmk[:, None], lmk + [px, py] * 5, -1).astype(np.float32)
        return img, boxes, lmk, has_lmk, shape0, r, (px, py)

    def load_mosaic(self, i: int):
        """Ghép 4 ảnh vào khung 2s x 2s quanh một tâm ngẫu nhiên; affine sau đó cắt về s x s."""
        s = self.img_size
        xc, yc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))
        canvas = np.full((2 * s, 2 * s, 3), PAD_VALUE, dtype=np.uint8)
        all_boxes, all_lmk, all_has = [], [], []
        for k, idx in enumerate([i] + random.choices(range(len(self)), k=3)):
            img, boxes, lmk, has_lmk, _, _ = self.load_image(idx)
            h, w = img.shape[:2]
            if k == 0:      # trên trái
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif k == 1:    # trên phải
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:    # dưới trái
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:           # dưới phải
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            dx, dy = x1a - x1b, y1a - y1b
            all_boxes.append(boxes + [dx, dy, dx, dy])
            all_lmk.append(np.where(has_lmk[:, None], lmk + [dx, dy] * 5, -1))
            all_has.append(has_lmk)

        boxes = np.concatenate(all_boxes).astype(np.float32)
        lmk = np.concatenate(all_lmk).astype(np.float32)
        has_lmk = np.concatenate(all_has)
        # Phần ảnh bị cắt nằm ngoài khung: kẹp box về khung, landmark ra ngoài khung thì bỏ
        boxes = boxes.clip(0, 2 * s)
        has_lmk = has_lmk & ((lmk >= 0) & (lmk < 2 * s)).all(axis=1)
        lmk[~has_lmk] = -1
        return canvas, boxes, lmk, has_lmk

    def __getitem__(self, i: int):
        row = self.rows[i]
        meta = {"image_path": row["image_path"], "orig_name": row["orig_name"]}
        if self.augment:
            if random.random() < self.aug["mosaic"]:
                img, boxes, lmk, has_lmk = self.load_mosaic(i)
            else:
                img, boxes, lmk, has_lmk, *_ = self.load_letterbox(i)
            img, boxes, lmk, has_lmk = random_affine(img, boxes, lmk, has_lmk, self.img_size, self.aug, self.min_px)
            augment_hsv(img, self.aug["hsv_h"], self.aug["hsv_s"], self.aug["hsv_v"])
            if random.random() < self.aug["fliplr"]:
                img, boxes, lmk, has_lmk = flip_lr(img, boxes, lmk, has_lmk)
        else:
            img, boxes, lmk, has_lmk, shape0, r, pad = self.load_letterbox(i)
            # Thông tin để đưa dự đoán về toạ độ ảnh gốc khi đánh giá
            meta.update(shape0=shape0, ratio=r, pad=pad)

        labels = to_normalized(boxes, lmk, has_lmk, self.img_size, self.img_size)
        out = torch.zeros((len(labels), 16))
        out[:, 1:] = torch.from_numpy(labels)
        img = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))  # BGR HWC -> RGB CHW
        return img, out, meta


def collate_fn(batch):
    """Gộp batch: ảnh uint8 (B, 3, S, S), nhãn (sum N, 16) với cột 0 là chỉ số ảnh trong batch."""
    imgs, labels, metas = zip(*batch)
    for i, lab in enumerate(labels):
        lab[:, 0] = i
    return torch.stack(imgs), torch.cat(labels), list(metas)


def seed_worker(worker_id: int) -> None:
    """Mỗi worker có seed riêng cho `random` và `numpy`, suy ra từ seed của torch."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def plot_batch(imgs: torch.Tensor, labels: torch.Tensor, path: Path, title: str, n: int = 8) -> None:
    """Vẽ n ảnh đầu của batch kèm box (xanh) và landmark (5 màu) để kiểm tra augmentation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["red", "orange", "lime", "cyan", "magenta"]   # mắt trái, mắt phải, mũi, miệng trái, miệng phải
    n = min(n, len(imgs))
    cols = 4
    fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(4 * cols, 4 * math.ceil(n / cols)))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if k >= n:
            continue
        s = imgs.shape[-1]
        ax.imshow(imgs[k].permute(1, 2, 0).numpy())
        lab = labels[labels[:, 0] == k].numpy()
        for row in lab:
            cx, cy, w, h = row[2:6] * s
            ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, color="deepskyblue", lw=0.8))
            if row[6] >= 0:
                pts = row[6:16].reshape(5, 2) * s
                ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=4)
        ax.set_title(f"{len(lab)} mặt", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_datasets(rows: List[dict], yolo_dir: Path, cfg: dict, aug: dict) -> Dict[str, WiderFaceDataset]:
    return {
        "train": WiderFaceDataset([r for r in rows if r["split"] == "train"], yolo_dir, cfg["img_size"],
                                  cfg["min_face_px"], augment=True, aug=aug),
        "val": WiderFaceDataset([r for r in rows if r["split"] == "val"], yolo_dir, cfg["img_size"],
                                cfg["min_face_px"], augment=False),
    }


def make_loaders(datasets: Dict[str, WiderFaceDataset], batch_size: int, seed: int, num_workers: int,
                 pin_memory: bool = False) -> Dict[str, DataLoader]:
    """Train xáo trộn có seed; val giữ thứ tự cố định."""
    common = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
                  pin_memory=pin_memory, persistent_workers=num_workers > 0, worker_init_fn=seed_worker)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return {
        "train": DataLoader(datasets["train"], shuffle=True, generator=gen, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
    }
