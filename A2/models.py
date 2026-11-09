"""YOLO5Face-n: backbone ShuffleNetV2, neck PAN, head Detect có thêm 10 giá trị landmark.

Kiến trúc theo `models/yolov5n.yaml` của yolov5-face, viết lại thành các module rõ ràng:
    ảnh (B, 3, S, S) -> Backbone -> P3 (S/8), P4 (S/16), P5 (S/32) -> Neck -> Head -> 3 tensor dự đoán.

Mỗi anchor dự đoán 16 giá trị: `[x, y, w, h, obj, 10 landmark, cls]`.
Hai biến thể (có / không landmark loss) dùng cùng một model, chỉ khác ở loss.
"""
import math
from typing import List

import torch
import torch.nn as nn

# Anchor (px) cho 3 tầng P3/8, P4/16, P5/32, lấy từ yolov5n.yaml
ANCHORS = (
    (4, 5, 8, 10, 13, 16),
    (23, 29, 43, 55, 73, 105),
    (146, 217, 231, 300, 335, 433),
)
STRIDES = (8, 16, 32)
N_LMK = 10


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU, padding giữ nguyên kích thước khi stride = 1."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StemBlock(nn.Module):
    """Khối đầu: giảm kích thước 4 lần bằng hai nhánh (conv stride 2 và max-pool) rồi gộp lại."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.stem_1 = Conv(c1, c2, 3, 2)
        self.stem_2a = Conv(c2, c2 // 2, 1, 1)
        self.stem_2b = Conv(c2 // 2, c2, 3, 2)
        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.stem_3 = Conv(c2 * 2, c2, 1, 1)

    def forward(self, x):
        x = self.stem_1(x)
        return self.stem_3(torch.cat((self.stem_2b(self.stem_2a(x)), self.stem_2p(x)), dim=1))


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Trộn kênh giữa các nhóm để thông tin của hai nhánh được trao đổi ở khối sau."""
    b, c, h, w = x.shape
    return x.view(b, groups, c // groups, h, w).transpose(1, 2).reshape(b, c, h, w)


class ShuffleV2Block(nn.Module):
    """Khối ShuffleNetV2.

    stride 1: tách đôi kênh, một nửa giữ nguyên, nửa kia qua conv 1x1 - depthwise 3x3 - conv 1x1.
    stride 2: cả hai nhánh đều xử lý toàn bộ đầu vào và giảm kích thước 2 lần, số kênh ra gấp đôi.
    """

    def __init__(self, c1: int, c2: int, stride: int):
        super().__init__()
        self.stride = stride
        mid = c2 // 2
        if stride == 1 and c1 != c2:
            raise ValueError("stride 1 cần c1 == c2")

        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(c1, c1, 3, stride, 1, groups=c1, bias=False), nn.BatchNorm2d(c1),
                nn.Conv2d(c1, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
            )
        else:
            self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            nn.Conv2d(c1 if stride > 1 else mid, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False), nn.BatchNorm2d(mid),
            nn.Conv2d(mid, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class Bottleneck(nn.Module):
    """Conv 1x1 -> conv 3x3, có cộng tắt (residual) khi `shortcut` và số kênh không đổi."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP bottleneck: một nhánh qua n Bottleneck, một nhánh chỉ conv 1x1, gộp lại bằng conv 1x1."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        mid = c2 // 2
        self.cv1 = Conv(c1, mid, 1, 1)
        self.cv2 = Conv(c1, mid, 1, 1)
        self.cv3 = Conv(2 * mid, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(mid, mid, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


def shuffle_stage(c1: int, c2: int, n_repeat: int) -> nn.Sequential:
    """Một tầng ShuffleNet: 1 khối stride 2 (giảm kích thước) + n khối stride 1."""
    return nn.Sequential(ShuffleV2Block(c1, c2, 2), *[ShuffleV2Block(c2, c2, 1) for _ in range(n_repeat)])


class Backbone(nn.Module):
    """Trích đặc trưng ở 3 độ phân giải: P3 (128 kênh, S/8), P4 (256, S/16), P5 (512, S/32)."""

    def __init__(self, widths=(32, 128, 256, 512), repeats=(3, 7, 3)):
        super().__init__()
        c0, c3, c4, c5 = widths
        self.stem = StemBlock(3, c0)                   # S/4
        self.stage3 = shuffle_stage(c0, c3, repeats[0])  # S/8
        self.stage4 = shuffle_stage(c3, c4, repeats[1])  # S/16
        self.stage5 = shuffle_stage(c4, c5, repeats[2])  # S/32
        self.out_channels = (c3, c4, c5)

    def forward(self, x):
        p3 = self.stage3(self.stem(x))
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


class Neck(nn.Module):
    """PAN: đường top-down đưa ngữ nghĩa từ P5 xuống P3, rồi đường bottom-up đưa chi tiết từ P3 lên P5."""

    def __init__(self, in_channels=(128, 256, 512), c: int = 128):
        super().__init__()
        c3, c4, c5 = in_channels
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # top-down
        self.reduce5 = Conv(c5, c, 1, 1)
        self.td4 = C3(c + c4, c, shortcut=False)
        self.reduce4 = Conv(c, c, 1, 1)
        self.td3 = C3(c + c3, c, shortcut=False)
        # bottom-up
        self.down3 = Conv(c, c, 3, 2)
        self.bu4 = C3(c + c, c, shortcut=False)
        self.down4 = Conv(c, c, 3, 2)
        self.bu5 = C3(c + c, c, shortcut=False)
        self.out_channels = (c, c, c)

    def forward(self, feats):
        p3, p4, p5 = feats
        r5 = self.reduce5(p5)                                  # S/32
        t4 = self.td4(torch.cat((self.up(r5), p4), dim=1))     # S/16
        r4 = self.reduce4(t4)
        n3 = self.td3(torch.cat((self.up(r4), p3), dim=1))     # S/8: đầu ra cho mặt nhỏ
        n4 = self.bu4(torch.cat((self.down3(n3), r4), dim=1))  # S/16: mặt vừa
        n5 = self.bu5(torch.cat((self.down4(n4), r5), dim=1))  # S/32: mặt lớn
        return n3, n4, n5


class DetectHead(nn.Module):
    """Mỗi tầng một conv 1x1 ra `n_anchor * 16` kênh.

    Khi train trả về đầu ra thô (B, A, H, W, 16) để loss tự giải mã.
    Khi eval giải mã về toạ độ pixel và trả về (B, tổng số anchor, 16).
    """

    def __init__(self, in_channels, n_classes: int = 1, anchors=ANCHORS, strides=STRIDES):
        super().__init__()
        self.nc = n_classes
        self.no = 5 + N_LMK + n_classes                 # x, y, w, h, obj, 10 landmark, cls
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.float32))
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32).view(self.nl, self.na, 2))
        self.m = nn.ModuleList(nn.Conv2d(c, self.na * self.no, 1) for c in in_channels)
        self._init_biases()

    def _init_biases(self):
        """Khởi tạo bias obj/cls theo yolov5: obj ban đầu ~ 8 vật thể trên ảnh 640, cls ~ 0.6."""
        for conv, s in zip(self.m, self.strides.tolist()):
            b = conv.bias.view(self.na, self.no)
            with torch.no_grad():
                b[:, 4] += math.log(8 / (640 / s) ** 2)
                b[:, 5 + N_LMK:] += math.log(0.6 / (self.nc - 0.99))

    def forward(self, feats, raw: bool = False) -> List[torch.Tensor]:
        outs = []
        for i, x in enumerate(feats):
            b, _, h, w = x.shape
            outs.append(self.m[i](x).view(b, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous())
        if self.training or raw:     # raw=True: lấy đầu ra thô ở chế độ eval để tính val loss
            return outs
        return torch.cat([self.decode(o, i) for i, o in enumerate(outs)], dim=1)

    def decode(self, out: torch.Tensor, i: int) -> torch.Tensor:
        """Đầu ra thô của tầng i -> box cxcywh pixel, obj/cls đã sigmoid, landmark pixel."""
        b, a, h, w, _ = out.shape
        stride, anchor = self.strides[i], self.anchors[i].view(1, a, 1, 1, 2)
        gy, gx = torch.meshgrid(torch.arange(h, device=out.device), torch.arange(w, device=out.device), indexing="ij")
        grid = torch.stack((gx, gy), dim=-1).view(1, 1, h, w, 2).float()

        y = out.clone()
        p = out.sigmoid()
        y[..., 0:2] = (p[..., 0:2] * 2 - 0.5 + grid) * stride          # tâm: lệch trong khoảng (-0.5, 1.5) ô
        y[..., 2:4] = (p[..., 2:4] * 2) ** 2 * anchor                  # kích thước: 0..4 lần anchor
        y[..., 4] = p[..., 4]
        lmk = out[..., 5:5 + N_LMK].view(b, a, h, w, 5, 2)            # landmark: không sigmoid, tính theo anchor
        y[..., 5:5 + N_LMK] = (lmk * anchor.unsqueeze(-2) + (grid * stride).unsqueeze(-2)).view(b, a, h, w, N_LMK)
        y[..., 5 + N_LMK:] = p[..., 5 + N_LMK:]
        return y.view(b, -1, self.no)


class YOLO5Face(nn.Module):
    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck(self.backbone.out_channels)
        self.head = DetectHead(self.neck.out_channels, n_classes)
        self._init_bn()

    def _init_bn(self):
        """BatchNorm eps/momentum theo yolov5 (momentum 0.03 ổn định hơn với batch nhỏ)."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03

    def forward(self, x, raw: bool = False):
        return self.head(self.neck(self.backbone(x)), raw=raw)


def build_model(name: str = "yolo5face_n") -> YOLO5Face:
    if name != "yolo5face_n":
        raise ValueError(f"Chưa hỗ trợ model {name}")
    return YOLO5Face()
