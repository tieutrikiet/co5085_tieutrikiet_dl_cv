"""Cập nhật trọng số cho YOLO5Face: optimizer SGD 3 nhóm tham số, warmup + cosine, gom gradient, AMP, EMA.

Theo `train.py` của yolov5-face:
- 3 nhóm tham số: weight của BatchNorm (không weight decay), weight của conv (có weight decay), bias.
- Warmup theo iteration: LR bias đi từ 0.1 xuống lr, LR còn lại đi từ 0 lên lr, momentum 0.8 -> 0.937.
- Gom gradient cho đến khi đủ batch danh nghĩa 64 ảnh rồi mới cập nhật.
- EMA: bản sao trọng số trung bình trượt, dùng để đánh giá và lưu model.
"""
import math
import time
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_optimizer(model: nn.Module, cfg: dict, batch_size: int) -> torch.optim.Optimizer:
    """SGD Nesterov với 3 nhóm tham số; weight decay được nhân theo batch hiệu dụng / 64."""
    accumulate = max(round(cfg["nominal_batch"] / batch_size), 1)
    weight_decay = cfg["weight_decay"] * batch_size * accumulate / cfg["nominal_batch"]

    bn_weights, conv_weights, biases = [], [], []
    for module in model.modules():
        if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
            biases.append(module.bias)
        if isinstance(module, nn.BatchNorm2d):
            bn_weights.append(module.weight)
        elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            conv_weights.append(module.weight)

    optimizer = torch.optim.SGD(bn_weights, lr=cfg["lr0"], momentum=cfg["momentum"], nesterov=True)
    optimizer.add_param_group({"params": conv_weights, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": biases})
    return optimizer


def cosine_lambda(epochs: int, lrf: float):
    """Hệ số LR theo epoch: cosine từ 1 xuống lrf."""
    return lambda e: ((1 + math.cos(e * math.pi / epochs)) / 2) * (1 - lrf) + lrf


class ModelEMA:
    """Trung bình trượt của trọng số; decay tăng dần từ 0 lên `decay` để giai đoạn đầu bám sát model."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, updates: int = 0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


class Trainer:
    """Giữ optimizer, scheduler, scaler, EMA và bộ đếm iteration xuyên suốt các epoch."""

    def __init__(self, model: nn.Module, loss_fn: nn.Module, cfg: dict, batch_size: int, epochs: int,
                 iters_per_epoch: int, device: torch.device):
        self.model, self.loss_fn, self.cfg, self.device = model, loss_fn, cfg, device
        self.accumulate = max(round(cfg["nominal_batch"] / batch_size), 1)
        self.optimizer = build_optimizer(model, cfg, batch_size)
        self.lf = cosine_lambda(epochs, cfg["lrf"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.amp = cfg["amp"] and device.type in ("cuda", "mps")
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp)
        self.ema = ModelEMA(model, cfg["ema_decay"])
        self.n_warmup = max(round(cfg["warmup_epochs"] * iters_per_epoch), cfg["warmup_min_iters"])
        self.iters = 0           # tổng số batch đã chạy, dùng cho warmup
        self.last_step = -1      # iteration của lần optimizer.step gần nhất

    def warmup(self, epoch: int) -> int:
        """Trong warmup: nội suy LR, momentum và số batch gom gradient theo iteration."""
        ni = self.iters
        if ni > self.n_warmup:
            return self.accumulate
        xi = [0, self.n_warmup]
        accumulate = max(1, int(np.interp(ni, xi, [1, self.accumulate]).round()))
        for j, g in enumerate(self.optimizer.param_groups):
            start = self.cfg["warmup_bias_lr"] if j == 2 else 0.0
            g["lr"] = np.interp(ni, xi, [start, g["initial_lr"] * self.lf(epoch)])
            g["momentum"] = np.interp(ni, xi, [self.cfg["warmup_momentum"], self.cfg["momentum"]])
        return accumulate

    def optimizer_step(self) -> None:
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.model)
        self.last_step = self.iters

    def train_one_epoch(self, loader, epoch: int, log_every: int = 100, max_batches: Optional[int] = None) -> dict:
        """Một epoch; trả về loss trung bình [box, obj, cls, landmark, total], LR và thời gian."""
        self.model.train()
        n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
        mloss = torch.zeros(5, device=self.device)
        n_targets = 0
        self.optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for i, (imgs, targets, _) in enumerate(loader):
            if i >= n_batches:
                break
            accumulate = self.warmup(epoch)
            imgs = imgs.to(self.device, non_blocking=True).float() / 255     # uint8 -> [0, 1]
            targets = targets.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                preds = self.model(imgs)
            loss, items = self.loss_fn(preds, targets)       # loss tính ở FP32
            self.scaler.scale(loss).backward()
            self.iters += 1

            if self.iters - self.last_step >= accumulate:
                self.optimizer_step()

            mloss = (mloss * i + items) / (i + 1)
            n_targets += targets.shape[0]
            if log_every and ((i + 1) % log_every == 0 or i + 1 == n_batches):
                done = (i + 1) / n_batches
                el = time.time() - t0
                box, obj, cls, lmk, tot = mloss.tolist()
                log(f"  epoch {epoch} batch {i + 1}/{n_batches} | box {box:.4f} obj {obj:.4f} lmk {lmk:.4f} "
                    f"tổng {tot:.4f} | lr {self.optimizer.param_groups[1]['lr']:.5f} | "
                    f"{el / 60:.1f} phút, còn {el / done * (1 - done) / 60:.1f} phút")

        lrs = [g["lr"] for g in self.optimizer.param_groups]     # LR thực dùng ở cuối epoch, trước khi đổi epoch
        self.scheduler.step()
        box, obj, cls, lmk, tot = mloss.tolist()
        return {"box": box, "obj": obj, "cls": cls, "landmark": lmk, "total": tot, "lr": lrs, "targets": n_targets,
                "time": time.time() - t0}

    @torch.no_grad()
    def val_loss(self, loader, max_batches: Optional[int] = None) -> dict:
        """Loss trên val với model EMA (đầu ra thô, không giải mã) để vẽ learning curve."""
        model = self.ema.ema
        model.eval()
        mloss = torch.zeros(5, device=self.device)
        n = 0
        for i, (imgs, targets, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            imgs = imgs.to(self.device).float() / 255
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                preds = model(imgs, raw=True)
            _, items = self.loss_fn(preds, targets.to(self.device))
            mloss += items
            n += 1
        box, obj, cls, lmk, tot = (mloss / max(n, 1)).tolist()
        return {"box": box, "obj": obj, "cls": cls, "landmark": lmk, "total": tot}

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "ema": self.ema.ema.state_dict(), "ema_updates": self.ema.updates,
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(), "iters": self.iters}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.ema.ema.load_state_dict(state["ema"])
        self.ema.updates = state["ema_updates"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.iters = self.last_step = state["iters"]
