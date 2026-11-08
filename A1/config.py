"""A1 config: đường dẫn, hyperparameter chung và ma trận 5 cấu hình thí nghiệm."""
from pathlib import Path

A1_DIR = Path(__file__).resolve().parent
REPO_DIR = A1_DIR.parent

DATA_DIR = REPO_DIR.parent / "food-101"          # dataset nằm ở thư mục cha của repo
MANIFEST_PATH = A1_DIR / "split_manifest.csv"
RESULTS_DIR = A1_DIR / "results"
EXPERIMENT_DIR = RESULTS_DIR / "experiments"     # kết quả thí nghiệm thật (commit)
CHECK_DIR = RESULTS_DIR / "_checks"              # kết quả thử nghiệm nhỏ (không dùng trong báo cáo)
CKPT_DIR = A1_DIR / "checkpoints"                # *.pt, không commit (.gitignore)
FIG_DIR = A1_DIR / "figures"

CONFIG = {
    "dataset": "Food-101",
    "data_dir": DATA_DIR,
    "manifest_path": MANIFEST_PATH,
    "split_seed": 2026,           # per-class RNG: default_rng(split_seed + class_index)
    "per_class": {"train": 600, "val": 150, "test": 250},
    "image_size": 224,
    "physical_batch": 32,
    "accum_steps": 4,             # effective batch = 32 x 4 = 128
    "num_workers": 8,
    "seed": 42,
    "epochs": 10,                 # ngân sách rút gọn, dùng chung cho cả 5 cấu hình
    "warmup_epochs": 1,           # 10% số epoch
    "lr_head": 3e-4,              # LR cố định, không dò
    "backbone_lr_mult": {"scratch": 1.0, "pretrained": 0.1},
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "grad_clip": 1.0,
    "head_init_std": 0.01,
    "amp": True,                  # FP16 autocast; dùng chung cho mọi cấu hình
    "log_every": 200,             # in tiến độ mỗi 200 batch
}

# (arch, init, freeze)
EXPERIMENTS = [
    ("resnet50", "scratch",    "full"),
    ("resnet50", "pretrained", "full"),
    ("vit_b_16", "pretrained", "full"),
    ("resnet50", "pretrained", "head"),
    ("resnet50", "pretrained", "partial"),
]

# Chạy cấu hình nhanh và quan trọng trước: head, partial, ResNet full, ViT full, ResNet scratch
RUN_ORDER = [EXPERIMENTS[i] for i in (3, 4, 1, 2, 0)]

# Module (tên trong TorchVision) được TRAIN ở mỗi chế độ; phần còn lại bị freeze
TRAINABLE_MODULES = {
    "resnet50": {
        "full":    None,                 # None = train toàn bộ
        "partial": ["layer4", "fc"],     # freeze conv1, bn1, layer1-3
        "head":    ["fc"],               # freeze toàn bộ backbone
    },
    "vit_b_16": {
        "full":    None,                 # phạm vi đã chốt chỉ dùng ViT full
    },
}


def experiment_key(arch: str, init: str, freeze: str) -> str:
    """Tên ngắn của một cấu hình, dùng cho `--only`: ví dụ resnet50_pretrained_head."""
    return f"{arch}_{init}_{freeze}"


def run_name(arch: str, init: str, freeze: str, cfg: dict, seed: int) -> str:
    return f"{experiment_key(arch, init, freeze)}_lr{cfg['lr_head']:.0e}_ep{cfg['epochs']}_seed{seed}"
