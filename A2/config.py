# A2/config.py
from pathlib import Path

A2_DIR = Path(__file__).resolve().parent
REPO_DIR = A2_DIR.parent
YOLO_DIR = REPO_DIR.parent / "widerface_yolo"   # train/, val/ (định dạng YOLO)
WIDER_DIR = REPO_DIR.parent / "widerface"       # bản gốc, dùng cho val/label.txt và test
MANIFEST_PATH = A2_DIR / "split_manifest.csv"
RESULTS_DIR = A2_DIR / "results"
EXPERIMENT_DIR = RESULTS_DIR / "experiments"     # kết quả thí nghiệm thật
CHECK_DIR = RESULTS_DIR / "_checks"              # kết quả thử nghiệm nhỏ, không dùng trong báo cáo
CKPT_DIR = A2_DIR / "checkpoints"
FIG_DIR = A2_DIR / "figures"

CONFIG = {
    "min_face_px": 2,
    "img_size": 640,              # tạm dùng giá trị của paper, chốt lại sau khi đo thời gian
    "batch_size": 16,
    "num_workers": 8,
    "seed": 42,
    "epochs": 20,                 # tạm, chốt lại sau khi ước lượng thời gian
    "log_every": 100,             # in tiến độ mỗi 100 batch
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

# Loss và optimizer lấy theo data/hyp.scratch.yaml + train.py của yolov5-face
LOSS = {
    "box": 0.05,         # hệ số CIoU loss
    "obj": 1.0,          # hệ số objectness loss
    "cls": 0.5,          # hệ số class loss (= 0 thực tế vì chỉ có 1 lớp)
    "landmark": 0.005,   # hệ số wing loss của landmark
    "anchor_t": 4.0,     # mặt được gán cho anchor nếu tỉ lệ kích thước < 4 lần
}

OPTIM = {
    "lr0": 0.01,             # SGD
    "lrf": 0.2,              # LR cuối = lr0 * lrf (cosine)
    "momentum": 0.937,
    "weight_decay": 5e-4,    # chỉ áp cho weight của conv
    "nominal_batch": 64,     # gom gradient để batch hiệu dụng ~64
    "warmup_epochs": 3.0,
    "warmup_min_iters": 1000,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "ema_decay": 0.9999,
    "amp": True,             # FP16 autocast
}

EXPERIMENTS = [("yolo5face_n", "landmark"), ("yolo5face_n", "no_landmark")]


def experiment_key(model: str, variant: str) -> str:
    """Tên ngắn của một cấu hình, dùng cho `--only`: ví dụ yolo5face_n_landmark."""
    return f"{model}_{variant}"


def run_name(model: str, variant: str, cfg: dict) -> str:
    return f"{experiment_key(model, variant)}_img{cfg['img_size']}_ep{cfg['epochs']}_seed{cfg['seed']}"
