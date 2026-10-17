"""Các cách đưa ảnh thành chuỗi: [B, C, H, W] -> [B, T, D].

T là số bước thời gian (RNN xử lý tuần tự t = 0, 1, ..., T-1),
D là số chiều của vector đưa vào ở mỗi bước (input_size của LSTM/GRU).
"""
import torch

from common.datasets import FASHION_MNIST_SHAPE


def create_sequence(x: torch.Tensor, mode: str = "row", patch_size: int = 7) -> torch.Tensor:
    """
    Biến ảnh [B, C, H, W] thành chuỗi [B, T, D]

    Args:
        mode (str): row | col | patch

    Note:
        B: batch size
        C: number of channels
        H: height of the image
        W: width of the image
        T: sequence length
        D: feature dimension
    """
    assert mode in ["row", "col", "patch"], f"Invalid mode: {mode}, must be one of ['row', 'col', 'patch']"
    B, C, H, W = x.shape

    if mode == "row":
        # mỗi hàng từ trên xuống dưới trở thành một bước trong chuỗi
        # T = H
        # D = C * W
        return x.permute(0, 2, 1, 3).reshape(B, H, C * W)

    if mode == "col":
        # mỗi cột từ trái sang phải trở thành một bước trong chuỗi
        # T = W
        # D = C * H
        return x.permute(0, 3, 1, 2).reshape(B, W, C * H)

    if mode == "patch":
        # chia ảnh thành các patch kích thước patch_size x patch_size
        # T = số lượng patch = (H // patch_size) * (W // patch_size)
        # D = C * patch_size * patch_size
        # ví dụ patch_size = 7, ảnh 28x28
        # T = (28 // 7) * (28 // 7) = 4 x 4 = 16
        # D = C * 7 * 7 = 49 (vì C = 1)
        p = patch_size
        assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // p) * (W // p), C * p * p)


def sequence_shape(spec, image_shape=FASHION_MNIST_SHAPE):
    T, D = create_sequence(torch.zeros(1, *image_shape), **spec).shape[1:]
    return T, D


SEQUENCES = {
    "row": dict(mode="row"),
    "col": dict(mode="col"),
    "patch": dict(mode="patch", patch_size=7),
}

SCAN_ORDER = {"row": "hàng: trên → dưới", "col": "cột: trái → phải", "patch": "patch: trái → phải, trên → dưới"}
