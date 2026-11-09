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
    "img_size": None, 
    "seed": 42,
}

EXPERIMENTS = [("yolo5face_n", "landmark"), ("yolo5face_n", "no_landmark")]