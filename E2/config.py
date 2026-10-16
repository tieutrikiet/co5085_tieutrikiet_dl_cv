from pathlib import Path

# E2 dùng lại đúng CONFIG (seed, split, batch size, lr, patience, ...) của E1 để kết quả so sánh được
from E1.config import CLASS_NAMES, CONFIG, NUM_CLASSES, SPLIT_PATH

E2_DIR = Path(__file__).resolve().parent
REPO_DIR = E2_DIR.parent

E2_CONFIG = dict(
    embed_dim=64,
    depth=2,
    num_heads=4,
    mlp_ratio=2,
    dropout=0.1,
    pool="cls",
    head="softmax",
)

RESULTS_DIR = E2_DIR / "results"
FIG_DIR = E2_DIR / "figures"
E1_RESULTS_PATH = REPO_DIR / "E1" / "results" / "results.json"  # baseline MLP / CNN của E1

ATTN_COLORS = {"scratch": "#2a78d6", "torch": "#F2A900"}
TOK_COLORS = {"patch7": "#DE7272", "row": "#55A9A0", "cnn": "#F2A900"}

# Lưới token (H', W') của từng tokenizer, dùng để đặt attention map lên ảnh 28x28
TOKEN_GRID = {"patch7": (4, 4), "row": (28, 1), "cnn": (7, 7)}
