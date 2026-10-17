"""Hình riêng của E3: minh họa cách biểu diễn chuỗi và learning curves theo cặp."""
from typing import Dict

import matplotlib.pyplot as plt

from common.plotting import plot_learning_curves

from .classifiers import RNNS
from .config import RNN_COLORS, SEQ_COLORS
from .sequences import SCAN_ORDER, SEQUENCES, create_sequence


def plot_sequences(image, sequences=SEQUENCES):
    """Hàng trên: ảnh và thứ tự quét; hàng dưới: chuỗi [T, D] mà RNN thực sự nhận vào."""
    fig, axes = plt.subplots(2, len(sequences), figsize=(3.2 * len(sequences), 6.4))
    for k, (name, spec) in enumerate(sequences.items()):
        seq = create_sequence(image[None], **spec)[0]   # [1, C, H, W] -> [T, D]
        T, D = seq.shape

        ax = axes[0, k]
        ax.imshow(image[0], cmap="gray")
        if spec["mode"] == "patch":                     # vẽ lưới patch và số thứ tự bước t
            p = spec["patch_size"]
            g = image.shape[-1] // p
            for i in range(1, g):
                ax.axhline(i * p - 0.5, color="#DE7272", lw=0.8)
                ax.axvline(i * p - 0.5, color="#DE7272", lw=0.8)
            for t in range(T):
                r, c = divmod(t, g)
                ax.text(c * p + (p - 1) / 2, r * p + (p - 1) / 2, str(t), color="#DE7272",
                        ha="center", va="center", fontsize=9, fontweight="bold")
        ax.set_title(f"{name}: T={T}, D={D}\n({SCAN_ORDER[spec['mode']]})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1, k]
        ax.imshow(seq.numpy(), cmap="gray", aspect="auto")
        ax.set(xlabel="D (đặc trưng mỗi bước)", ylabel="t (bước thời gian)")
    fig.tight_layout()
    return fig


def plot_lstm_vs_gru_curves(histories: Dict[str, dict], seq_name: str):
    """LSTM vs GRU trên cùng một cách biểu diễn"""
    names = [f"{r}_{seq_name}" for r in RNNS]
    fig = plot_learning_curves({n: histories[n] for n in names}, {n: RNN_COLORS[n.split("_")[0]] for n in names})
    fig.suptitle(f"{seq_name}: LSTM vs. GRU", y=1.02)
    return fig


def plot_sequence_curves(histories: Dict[str, dict], rnn_type: str):
    """Cùng một RNN, so các cách biểu diễn"""
    names = {s: f"{rnn_type}_{s}" for s in SEQUENCES}
    fig = plot_learning_curves({s: histories[n] for s, n in names.items()}, SEQ_COLORS)
    fig.suptitle(f"{rnn_type.upper()}: so sánh cách biểu diễn chuỗi", y=1.02)
    return fig
