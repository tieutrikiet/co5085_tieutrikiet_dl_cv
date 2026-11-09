# A2/config.py
from pathlib import Path

A2_DIR = Path(__file__).resolve().parent
REPO_DIR = A2_DIR.parent
YOLO_DIR = REPO_DIR.parent / "widerface_yolo"   # train/, val/ (định dạng YOLO)
WIDER_DIR = REPO_DIR.parent / "widerface"       # bản gốc, dùng cho val/label.txt và test
MANIFEST_PATH = A2_DIR / "split_manifest.csv"
RESULTS_DIR = A2_DIR / "results"
CKPT_DIR = A2_DIR / "checkpoints"
FIG_DIR = A2_DIR / "figures"

CONFIG = {
    "min_face_px": 2,
    "img_size": 640,              # tạm dùng giá trị của paper, chốt lại sau khi đo thời gian
    "batch_size": 16,
    "num_workers": 8,
    "seed": 42,
}

# Augmentation lấy theo data/hyp.scratch.yaml của yolov5-face
AUGMENT = {
    "mosaic": 0.5,       # xác suất ghép 4 ảnh
    "translate": 0.1,    # tịnh tiến ± tỉ lệ cạnh ảnh
    "scale": 0.5,        # phóng to/thu nhỏ ± 50%
    "shear": 0.5,        # độ nghiêng ± độ
    "degrees": 0.0,      # không xoay
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,       # lật ngang, có đổi chỗ landmark trái/phải
}

EXPERIMENTS = [("yolo5face_n", "landmark"), ("yolo5face_n", "no_landmark")]