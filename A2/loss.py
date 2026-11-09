"""Loss của YOLO5Face: gán ground-truth cho anchor, rồi tính loss box, obj, cls và landmark.

Theo `utils/loss.py` của yolov5-face:
- box: 1 - CIoU giữa box dự đoán và box thật, chỉ trên các anchor được gán.
- obj: BCE trên mọi anchor; anchor được gán có nhãn = IoU (đã detach), còn lại = 0.
- cls: BCE, chỉ tính khi có hơn 1 lớp (WIDERFace có 1 lớp nên bằng 0).
- landmark: wing loss trên các điểm có nhãn, toạ độ tính theo đơn vị anchor.
"""
import math

import torch
import torch.nn as nn

# Trọng số obj cho từng tầng P3, P4, P5: tầng mịn có nhiều anchor âm nên được nhân lớn hơn
OBJ_BALANCE = (4.0, 1.0, 0.4)
# Dịch tâm sang ô lân cận: chính ô đó, phải, dưới, trái, trên
# (1, 0)          (0, 1)
#          (0, 0)
# (-1, 0)         (0, -1)
#
NEIGHBOR_OFFSETS = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """CIoU giữa hai tập box cxcywh cùng kích thước (N, 4).

    CIoU = IoU - khoảng cách tâm² / đường chéo khung bao² - phạt lệch tỉ lệ khung hình.
    """
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = box1[:, 2], box1[:, 3] + eps
    w2, h2 = box2[:, 2], box2[:, 3] + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)      # khung bao nhỏ nhất chứa cả hai box
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (box2[:, 0] - box1[:, 0]) ** 2 + (box2[:, 1] - box1[:, 1]) ** 2
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + v * alpha)


def wing_loss(pred: torch.Tensor, target: torch.Tensor, w: float = 10.0, e: float = 2.0) -> torch.Tensor:
    """Wing loss: dạng log khi sai số nhỏ (nhạy với lệch nhỏ), tuyến tính khi sai số lớn."""
    c = w - w * math.log(1 + w / e)
    diff = (pred - target).abs()
    return torch.where(diff < w, w * torch.log(1 + diff / e), diff - c)


class YoloFaceLoss(nn.Module):
    """Tính loss cho đầu ra thô của `DetectHead` (list 3 tensor (B, A, H, W, 16)).

    `hyp` gồm hệ số box/obj/cls/landmark và `anchor_t`. Biến thể không dùng landmark đặt
    `use_landmark=False`: landmark loss vẫn được tính để ghi log nhưng không cộng vào tổng.
    """

    def __init__(self, head, hyp: dict, use_landmark: bool = True):
        super().__init__()
        self.hyp = hyp
        self.use_landmark = use_landmark
        self.nc = head.nc
        self.na = head.na
        self.strides = head.strides
        # Anchor đổi về đơn vị ô lưới của từng tầng
        self.anchors = head.anchors / head.strides.view(-1, 1, 1)
        self.bce = nn.BCEWithLogitsLoss()

    def build_targets(self, preds, targets):
        """Gán mỗi mặt cho các (anchor, ô lưới) phù hợp ở từng tầng.

        Một anchor nhận mặt nếu tỉ lệ kích thước mặt/anchor nằm trong [1/anchor_t, anchor_t].
        Ngoài ô chứa tâm, mặt còn được gán cho 2 ô lân cận gần tâm nhất để có thêm mẫu dương.
        """
        device = targets.device
        nt = targets.shape[0]
        out = []
        offsets = torch.tensor(NEIGHBOR_OFFSETS, device=device, dtype=torch.float32) * 0.5
        # (A, nt, 17): thêm cột chỉ số anchor vào cuối mỗi nhãn
        ai = torch.arange(self.na, device=device).float().view(self.na, 1).repeat(1, nt)
        t_all = torch.cat((targets.repeat(self.na, 1, 1), ai[:, :, None]), dim=2)

        for i, p in enumerate(preds):
            h, w = p.shape[2], p.shape[3]
            gain = torch.ones(17, device=device)
            gain[2:6] = torch.tensor([w, h, w, h], device=device)
            gain[6:16] = torch.tensor([w, h] * 5, device=device)
            anchors = self.anchors[i].to(device)

            t = t_all * gain                       # đổi toạ độ chuẩn hoá -> đơn vị ô lưới
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                t = t[torch.max(r, 1 / r).max(dim=2)[0] < self.hyp["anchor_t"]]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < 0.5) & (gxy > 1)).T     # tâm gần cạnh trái/trên của ô
                l, m = ((gxi % 1 < 0.5) & (gxi > 1)).T     # tâm gần cạnh phải/dưới của ô
                mask = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[mask]
                off = (torch.zeros_like(gxy)[None] + offsets[:, None])[mask]
            else:
                t = t_all[0]
                off = torch.zeros((0, 2), device=device)

            b, c = t[:, 0].long(), t[:, 1].long()
            gxy, gwh = t[:, 2:4], t[:, 4:6]
            gij = (gxy - off).long()
            gi, gj = gij[:, 0].clamp(0, w - 1), gij[:, 1].clamp(0, h - 1)
            a = t[:, 16].long()

            # Landmark tính theo ô được gán; điểm thiếu (< 0) bị che bằng mặt nạ
            lmk = t[:, 6:16].clone()
            lmk_mask = (t[:, 6:16] >= 0).float()
            lmk -= gij.float().repeat(1, 5)

            out.append({
                "idx": (b, a, gj, gi),
                "box": torch.cat((gxy - gij, gwh), dim=1),   # tâm tương đối trong ô + kích thước
                "anchor": anchors[a],
                "cls": c,
                "lmk": lmk,
                "lmk_mask": lmk_mask,
            })
        return out

    def forward(self, preds, targets):
        """Trả về (loss tổng đã nhân batch size, tensor log [box, obj, cls, landmark, tổng])."""
        device = targets.device
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)
        lmark = torch.zeros(1, device=device)

        for i, (p, tgt) in enumerate(zip(preds, self.build_targets(preds, targets))):
            p = p.float()
            b, a, gj, gi = tgt["idx"]
            tobj = torch.zeros_like(p[..., 4])
            n = b.shape[0]
            if n:
                ps = p[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * tgt["anchor"]
                iou = bbox_ciou(torch.cat((pxy, pwh), dim=1), tgt["box"])
                lbox += (1 - iou).mean()
                tobj[b, a, gj, gi] = iou.detach().clamp(0).to(tobj.dtype)

                if self.nc > 1:
                    t = torch.zeros_like(ps[:, 15:])
                    t[range(n), tgt["cls"]] = 1.0
                    lcls += self.bce(ps[:, 15:], t)

                plmk = ps[:, 5:15] * tgt["anchor"].repeat(1, 5)
                mask = tgt["lmk_mask"]
                lmark += (wing_loss(plmk * mask, tgt["lmk"] * mask) * mask).sum() / (mask.sum() + 1e-14)

            lobj += self.bce(p[..., 4], tobj) * OBJ_BALANCE[i]

        s = 3 / len(preds)
        lbox *= self.hyp["box"] * s
        lobj *= self.hyp["obj"] * s
        lcls *= self.hyp["cls"] * s
        lmark *= self.hyp["landmark"] * s

        total = lbox + lobj + lcls + (lmark if self.use_landmark else 0)
        bs = preds[0].shape[0]
        return total * bs, torch.cat((lbox, lobj, lcls, lmark, total)).detach()
