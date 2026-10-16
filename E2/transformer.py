"""Encoder Transformer nông + head phân loại."""
import torch
import torch.nn as nn

from E1.classifiers import MLPClassifier, SoftmaxClassifier

from .attention import ATTENTIONS
from .config import E2_CONFIG, NUM_CLASSES
from .tokenizers import TOKENIZERS


class FeedForward(nn.Module):
    """Áp dụng độc lập cho từng token: [B, N, D] -> [B, N, D]"""

    def __init__(self, embed_dim, mlp_ratio=2, dropout=0.1):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2, dropout=0.1, attention="scratch"):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = ATTENTIONS[attention](embed_dim=embed_dim, num_heads=num_heads, attn_dropout=dropout)
        self.drop = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, return_attn=False):
        # Pre-norm and attention
        if return_attn:
            attn_out, attn_weights = self.attention(self.norm1(x), return_attn=True)
        else:
            attn_out, attn_weights = self.attention(self.norm1(x)), None

        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))

        if return_attn:
            return x, attn_weights
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, tokenizer, attention="scratch", embed_dim=64, depth=2, num_heads=4,
                 mlp_ratio=2, dropout=0.1, pool="cls", head="softmax", use_pos=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.pool = pool
        num_tokens = tokenizer.num_tokens
        if pool == "cls":
            num_tokens += 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pool == "cls" else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim)) if use_pos else None
        self.pos_drop = nn.Dropout(p=dropout) if use_pos else None
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim=embed_dim, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, dropout=dropout, attention=attention)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        if head == "softmax":
            self.head = SoftmaxClassifier(input_dim=embed_dim, num_classes=NUM_CLASSES)
        else:
            self.head = MLPClassifier(input_dim=embed_dim, num_classes=NUM_CLASSES,
                                      hidden_dims=(embed_dim * mlp_ratio,), dropout=dropout)

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_attn: bool = False):
        tokens = self.tokenizer(x)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)

        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
            tokens = self.pos_drop(tokens)

        attn_maps = []
        for block in self.blocks:
            if return_attn:
                tokens, attn = block(tokens, return_attn=True)
                attn_maps.append(attn)
            else:
                tokens = block(tokens)

        tokens = self.norm(tokens)

        if self.pool == "cls":
            tokens = tokens[:, 0]
        else:
            tokens = tokens.mean(dim=1)

        logits = self.head(tokens)
        if return_attn:
            return logits, attn_maps
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.softmax(self(x), dim=-1)


def build_model(tokenizer_name, attention="scratch", **overrides):
    cfg = {**E2_CONFIG, **overrides}
    return TransformerClassifier(tokenizer=TOKENIZERS[tokenizer_name](cfg["embed_dim"]),
                                 attention=attention, **cfg)
