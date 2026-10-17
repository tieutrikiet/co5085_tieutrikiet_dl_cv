from pathlib import Path

# E3 dùng lại đúng CONFIG (seed, split, batch size, lr, patience, ...) của E1 để kết quả so sánh được
from E1.config import CLASS_NAMES, CONFIG, NUM_CLASSES, SPLIT_PATH

E3_DIR = Path(__file__).resolve().parent
REPO_DIR = E3_DIR.parent

E3_CONFIG = dict(
    hidden_size=128,
    num_layers=1,
    bidirectional=False,
    dropout=0.1,
    pool="last",
    head="softmax",
    batch_first=True,
)

BASELINES = ["MLP", "CNN"]  # train lại từ E1 trên cùng split, seed, thứ tự batch

RESULTS_DIR = E3_DIR / "results"
FIG_DIR = E3_DIR / "figures"
E1_RESULTS_PATH = REPO_DIR / "E1" / "results" / "results.json"  # số liệu E1 đã báo cáo, để đối chiếu baseline

RNN_COLORS = {"lstm": "#2a78d6", "gru": "#F2A900"}
SEQ_COLORS = {"row": "#DE7272", "col": "#55A9A0", "patch": "#8a6fd1"}
