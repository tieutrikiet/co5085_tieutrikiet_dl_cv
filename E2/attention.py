"""Multi-Head Self-Attention: bản tự viết (scratch) và bản dùng nn.MultiheadAttention (torch).

Input X ∈ R^{B×N×D}: B sample, N token, mỗi token D chiều, chia thành h head với d = D/h.

    Q = X W_Q,  K = X W_K,  V = X W_V
    head_i = softmax(Q_i K_i^T / sqrt(d)) V_i
    MSA(X) = Concat(head_1, ..., head_h) W_O

Q, K, V ∈ R^{B×h×N×d}  =>  Q K^T ∈ R^{B×h×N×N},  head_i ∈ R^{B×N×d}
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim                # D
        self.num_heads = num_heads                # h
        self.head_dim = embed_dim // num_heads    # D/h
        self.scale = 1 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)   # W_q, W_k, W_v combined
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Output projection
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, return_attn: bool = False):
        B, N, D = x.shape

        # 1. Tính Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, h, N, D/h)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, h, N, D/h)

        # 2. Tính attention scores bằng einsum
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale  # (B, h, N, N)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 3. Tính weighted sum của V
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_probs, v)  # (B, h, N, D/h)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, D)  # (B, N, D)

        # 4. Áp dụng output projection
        output = self.out_proj(attn_output)

        if return_attn:
            return output, attn_probs
        return output


class MSAPyTorch(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=bias,
            batch_first=True,  # Để đầu vào có dạng (B, N, D) thay vì (N, B, D)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_output, attn_weights = self.mha(x, x, x, need_weights=return_attn, average_attn_weights=False)
        if return_attn:
            return attn_output, attn_weights
        return attn_output


ATTENTIONS = {
    "scratch": MultiHeadSelfAttention,
    "torch": MSAPyTorch,
}

# Ánh xạ hậu tố key: scratch -> torch
_MSA_KEY_MAP = {
    "qkv.weight": "mha.in_proj_weight",
    "qkv.bias": "mha.in_proj_bias",
    "out_proj.weight": "mha.out_proj.weight",
    "out_proj.bias": "mha.out_proj.bias",
}


def scratch_to_torch_state_dict(state_dict: dict) -> dict:
    torch_state_dict = {}
    for key, value in state_dict.items():
        # Nếu key kết thúc bằng một hậu tố trong _MSA_KEY_MAP, ánh xạ sang key tương ứng của torch
        for suffix, torch_key in _MSA_KEY_MAP.items():
            if key.endswith(suffix):
                torch_state_dict[key[:-len(suffix)] + torch_key] = value
                break
        else:
            torch_state_dict[key] = value
    return torch_state_dict
