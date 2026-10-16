"""Các cách lấy token từ ảnh [B, C, H, W] -> [B, N, D]."""
import torch.nn as nn

from E1.classifiers import conv_block


class PatchEmbedding(nn.Module):
    """Tokenizer theo phương pháp Patching, default là patch_size=7, embed_dim=64"""
    def __init__(self, patch_size=7, embed_dim=64, img_size=28, in_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_tokens = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # [B, C, H/p, p, W/p, p] -> [B, H/p, W/p, C, p, p]
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, -1, C * p * p)
        return self.proj(x)


class RowEmbedding(nn.Module):
    """Tokenizer theo phương pháp Row Embedding, với token size đúng bằng 28pixel - tức là một dòng của mỗi ảnh"""
    def __init__(self, embed_dim=64, img_size=28, in_channels=1):
        super().__init__()
        self.num_tokens = img_size  # mỗi dòng là một token
        self.proj = nn.Linear(img_size * in_channels, embed_dim)  # mỗi dòng của ảnh được flatten và ánh xạ vào không gian embedding
        # tương đương L(C·W → D)

    def forward(self, x):
        B, C, H, W = x.shape
        # Đưa H lên trước C: [B, C, H, W] -> [B, H, C, W]
        # [B, H, C, W] -> [B, H, C*W]
        # rồi chiếu sang không gian embedding D
        x = x.permute(0, 2, 1, 3)  # [B, C, H, W] -> [B, H, C, W]
        x = x.reshape(B, H, C * W)  # [B, H, C, W] -> [B, H, C*W]
        return self.proj(x)


class CNNStemEmbedding(nn.Module):
    """Tokenizer theo phương pháp CNN Stem"""
    def __init__(self, embed_dim=64, img_size=28, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(input_channels, embed_dim // 2),
            conv_block(embed_dim // 2, embed_dim),
        )
        self.num_tokens = (img_size // 4) ** 2  # giả sử CNN stem giảm kích thước ảnh xuống 1/4
        self.proj = nn.Linear(embed_dim, embed_dim)  # chiếu các token CNN stem sang không gian embedding D

    def forward(self, x):
        f = self.features(x)  # [B, C', H', W']
        # flatten feature map: [B, C', H', W'] -> [B, H'*W', C']
        B, C, H, W = f.shape
        f = f.reshape(B, C, H * W)
        f = f.permute(0, 2, 1)  # [B, C, H*W] -> [B, H*W, C]
        return self.proj(f)


TOKENIZERS = {
    "patch7": lambda D: PatchEmbedding(patch_size=7, embed_dim=D),
    "row":    lambda D: RowEmbedding(embed_dim=D),
    "cnn":    lambda D: CNNStemEmbedding(embed_dim=D),
}
