"""Cập nhật trọng số cho YOLO5Face: optimizer SGD 3 nhóm tham số, warmup + cosine, gom gradient, AMP, EMA.

Theo `train.py` của yolov5-face:
- 3 nhóm tham số: weight của BatchNorm (không weight decay), weight của conv (có weight decay), bias.
- Warmup theo iteration: LR bias đi từ 0.1 xuống lr, LR còn lại đi từ 0 lên lr, momentum 0.8 -> 0.937.
- Gom gradient cho đến khi đủ batch danh nghĩa 64 ảnh rồi mới cập nhật.
- EMA: bản sao trọng số trung bình trượt, dùng để đánh giá và lưu model.

`fit` chạy trọn một thí nghiệm: train + val loss mỗi epoch, ghi history, lưu checkpoint, resume khi bị ngắt.
"""
import gc
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


_LOG_FILE: Optional[Path] = None


def set_log_file(path: Optional[Path]) -> None:
    """Mọi lời gọi `log` sau đó được ghi thêm vào file này (theo dõi bằng `tail -f`)."""
    global _LOG_FILE
    _LOG_FILE = path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a") as f:
            f.write(line + "\n")


def sync(device: torch.device) -> None:
    """Chờ GPU tính xong trước khi đọc đồng hồ."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


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


def fit(trainer: Trainer, loaders: Dict, epochs: int, name: str, out_dir: Path, ckpt_dir: Path,
        meta: dict, max_batches: Optional[Dict[str, int]] = None) -> dict:
    """Train `epochs` epoch, resume được từ `{name}_last.pt`; ghi `{name}.json` vào out_dir.

    Mỗi epoch lưu `last.pt` (đủ để resume) và `best.pt` (trọng số EMA có val loss thấp nhất).
    `max_batches` giới hạn số batch train/val mỗi epoch, dùng cho thử nghiệm nhỏ.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    max_batches = max_batches or {}
    last_path, best_path = ckpt_dir / f"{name}_last.pt", ckpt_dir / f"{name}_best.pt"

    history = {k: [] for k in ("train", "val", "lr", "epoch_time", "val_time")}
    best = {"val_total": float("inf"), "epoch": None}
    start_epoch = 0
    if last_path.exists():
        state = torch.load(last_path, map_location=trainer.device, weights_only=False)
        trainer.load_state_dict(state["trainer"])
        history, best, start_epoch = state["history"], state["best"], state["epoch"]
        torch.set_rng_state(state["rng_cpu"].cpu())     # map_location đã đưa tensor sang GPU
        log(f"Resume {name} từ epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs):
        sync(trainer.device); t0 = time.time()
        tr = trainer.train_one_epoch(loaders["train"], epoch, log_every=meta.get("log_every", 100),
                                     max_batches=max_batches.get("train"))
        sync(trainer.device); t1 = time.time()
        va = trainer.val_loss(loaders["val"], max_batches=max_batches.get("val"))
        sync(trainer.device); t2 = time.time()

        history["train"].append({k: tr[k] for k in ("box", "obj", "cls", "landmark", "total")})
        history["val"].append(va)
        history["lr"].append(tr["lr"][1])
        history["epoch_time"].append(t1 - t0)
        history["val_time"].append(t2 - t1)

        if va["total"] < best["val_total"]:
            best = {"val_total": va["total"], "epoch": epoch + 1}
            torch.save({"model": trainer.ema.ema.state_dict(), "epoch": epoch + 1, "meta": meta}, best_path)
        torch.save({"trainer": trainer.state_dict(), "history": history, "best": best, "epoch": epoch + 1,
                    "rng_cpu": torch.get_rng_state()}, last_path)

        log(f"[{name}] epoch {epoch + 1}/{epochs} | train {tr['total']:.4f} (box {tr['box']:.4f} "
            f"obj {tr['obj']:.4f} lmk {tr['landmark']:.4f}) | val {va['total']:.4f} (box {va['box']:.4f} "
            f"obj {va['obj']:.4f}) | lr {tr['lr'][1]:.5f} | {t1 - t0:.0f}s + {t2 - t1:.0f}s")

    summary = {
        "run": name,
        "meta": meta,
        "best": best,
        "history": history,
        "total_params": sum(p.numel() for p in trainer.model.parameters()),
        "total_train_time_s": sum(history["epoch_time"]),
        "total_val_time_s": sum(history["val_time"]),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_experiment(model_name: str, variant: str, cfg: dict, optim: dict, loss_hyp: dict, datasets: Dict,
                   device: torch.device, out_dir: Path, ckpt_dir: Path,
                   max_batches: Optional[Dict[str, int]] = None) -> dict:
    """build_model -> loss theo biến thể -> DataLoader theo seed -> fit."""
    from common.utils import set_seed

    from .config import run_name
    from .dataset import make_loaders
    from .loss import YoloFaceLoss
    from .models import build_model

    set_seed(cfg["seed"])
    model = build_model(model_name).to(device)
    loss_fn = YoloFaceLoss(model.head, loss_hyp, use_landmark=(variant == "landmark"))
    loaders = make_loaders(datasets, cfg["batch_size"], cfg["seed"], cfg["num_workers"])
    n_train = len(loaders["train"]) if not max_batches else min(max_batches["train"], len(loaders["train"]))
    trainer = Trainer(model, loss_fn, optim, cfg["batch_size"], cfg["epochs"], n_train, device)
    name = run_name(model_name, variant, cfg)
    meta = {"model": model_name, "variant": variant, "config": cfg, "optim": optim, "loss": loss_hyp,
            "log_every": cfg["log_every"], "max_batches": max_batches}
    log(f"{name}: {sum(p.numel() for p in model.parameters()):,} tham số | {n_train} batch train/epoch | "
        f"gom {trainer.accumulate} batch | warmup {trainer.n_warmup} iter")
    try:
        return fit(trainer, loaders, cfg["epochs"], name, out_dir, ckpt_dir, meta, max_batches)
    finally:
        del model, trainer, loaders
        free_memory(device)
