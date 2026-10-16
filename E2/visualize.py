"""Hình riêng của E2: learning curves theo cặp, accuracy vs. chi phí, attention rollout."""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common.plotting import plot_learning_curves

from .attention import ATTENTIONS
from .config import ATTN_COLORS, CLASS_NAMES, TOK_COLORS, TOKEN_GRID


def plot_scratch_vs_torch_curves(histories: Dict[str, dict], tok: str):
    names = [f"{tok}_{a}" for a in ATTENTIONS]
    fig = plot_learning_curves({n: histories[n] for n in names},
                               {n: ATTN_COLORS[n.split("_")[1]] for n in names})
    fig.suptitle(f"{tok}: MSA tự viết vs. nn.MultiheadAttention", y=1.02)
    return fig


def plot_tokenizer_curves(histories: Dict[str, dict], tokenizers):
    fig = plot_learning_curves({t: histories[f"{t}_scratch"] for t in tokenizers}, TOK_COLORS)
    fig.suptitle("So sánh tokenization (MSA tự viết)", y=1.02)
    return fig


def plot_acc_vs_cost(results: List[dict], e1: Optional[Dict[str, dict]] = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for r in results:
        color = TOK_COLORS.get(r["tokenizer"], "#8a8984")
        is_scratch = r["attention"] == "scratch"
        for ax, xkey in zip(axes, ["sec_per_epoch", "latency_ms"]):
            ax.scatter(r[xkey], r["test_acc"], color=color, marker="o" if is_scratch else "s", s=r["params"] / 1000)
            ax.annotate(r["model"], (r[xkey], r["test_acc"]), fontsize=6.5,
                        xytext=(4, 3 if is_scratch else -9), textcoords="offset points")
    for ax in axes:
        for m, ls in [("MLP", ":"), ("CNN", "--")]:
            if e1 and m in e1:
                ax.axhline(e1[m]["test_acc"], color="#8a8984", ls=ls, lw=1)
                ax.text(ax.get_xlim()[1], e1[m]["test_acc"], f" E1 {m}", va="center", fontsize=7)
        ax.grid(True, axis="y")
    axes[0].set(xlabel="Giây / epoch (train, MPS)", ylabel="Test accuracy", title="Accuracy vs. thời gian train")
    axes[1].set(xlabel="Latency batch=1 (ms, CPU)", title="Accuracy vs. latency suy luận")
    fig.tight_layout()
    return fig


@torch.no_grad()
def cls_attention_rollout(model, x, device):
    _, attn_maps = model(x.to(device), return_attn=True)
    T = attn_maps[0].shape[-1]
    eye = torch.eye(T, device=device)
    rollout = eye.expand(x.size(0), T, T)
    for A in attn_maps:
        A = A.mean(dim=1)                       # trung bình các head: [B, T, T]
        A = 0.5 * A + 0.5 * eye                 # tính cả residual
        A = A / A.sum(dim=-1, keepdim=True)
        rollout = A @ rollout
    return rollout[:, 0, 1:]                    # CLS -> các token ảnh: [B, N]


def plot_attention(model_by_tok, idx, x_norm: torch.Tensor, images: np.ndarray, labels: np.ndarray, device):
    """x_norm: ảnh đã chuẩn hóa để đưa vào model; images: ảnh uint8 gốc để hiển thị."""
    x = x_norm[idx]
    fig, axes = plt.subplots(len(model_by_tok) + 1, len(idx),
                             figsize=(len(idx) * 1.3, (len(model_by_tok) + 1) * 1.4))
    for j, i in enumerate(idx):
        axes[0, j].imshow(images[i], cmap="gray")
        axes[0, j].set_title(CLASS_NAMES[labels[i]], fontsize=7)
    for r, (tok, model) in enumerate(model_by_tok.items(), start=1):
        att = cls_attention_rollout(model, x, device).reshape(len(idx), 1, *TOKEN_GRID[tok])
        att = F.interpolate(att, size=(28, 28), mode="nearest").squeeze(1).cpu().numpy()
        for j, i in enumerate(idx):
            axes[r, j].imshow(images[i], cmap="gray")
            axes[r, j].imshow(att[j], cmap="jet", alpha=0.45)
        axes[r, 0].set_ylabel(tok, fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig
