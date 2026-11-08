"""Thử nghiệm nhỏ trước khi train thật (không dùng trong báo cáo):

pretrained : build_model nạp đúng trọng số ImageNet; mô hình gốc nhận ra ảnh pizza.
freeze     : sau một optimizer step, tham số + buffer BN của phần bị freeze không đổi.
overfit    : mỗi kiến trúc học thuộc 64 ảnh train (kiểm tra vòng train).
smoke      : cả 5 cấu hình chạy hết pipeline thật (fit, checkpoint, JSON, resume) trên tập con nhỏ.
estimate   : đo ms/batch trên DataLoader thật cho từng cấu hình -> ước tính tổng thời gian train.
"""
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from common.utils import set_seed

from .config import CHECK_DIR, CKPT_DIR, EXPERIMENTS
from .data import Food101Dataset, build_transforms
from .models import PRETRAINED_WEIGHTS, apply_freeze, build_model, count_params, param_groups, set_train_mode
from .training import fit, free_memory, log, run_experiment, sync

CHECK_CKPT_DIR = CKPT_DIR / "_checks"
CHECKS = ("pretrained", "freeze", "overfit", "smoke", "estimate")


def reset_check_outputs() -> None:
    """Xoá kết quả/checkpoint của lần kiểm tra trước để không resume nhầm."""
    for d in (CHECK_DIR, CHECK_CKPT_DIR):
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- pretrained

@torch.no_grad()
def check_pretrained(datasets, class_names, device) -> None:
    from torchvision.models import resnet50, vit_b_16
    refs = {"resnet50": (resnet50, "conv1.weight"), "vit_b_16": (vit_b_16, "conv_proj.weight")}
    idx = next(i for i, y in enumerate(datasets["val"].labels) if class_names[y] == "pizza")
    x, _ = datasets["val"][idx]

    for arch, (ctor, key) in refs.items():
        w = PRETRAINED_WEIGHTS[arch]
        ref = ctor(weights=w).eval()
        ours = build_model(arch, len(class_names), "pretrained")
        assert torch.equal(ours.state_dict()[key], ref.state_dict()[key]), f"{arch}: pretrained weights NOT loaded"
        scratch = build_model(arch, len(class_names), "scratch")
        assert not torch.equal(scratch.state_dict()[key], ref.state_dict()[key]), f"{arch}: scratch equals pretrained"

        probs = ref.to(device)(x[None].to(device)).softmax(1)[0]
        top = probs.topk(5)
        preds = [(w.meta["categories"][i], round(p, 2)) for p, i in zip(top.values.tolist(), top.indices.tolist())]
        log(f"[pretrained] {arch}: weights loaded via build_model OK | ImageNet top-5 on a pizza image: {preds}")
        del ref, ours, scratch
        free_memory(device)


# ---------------------------------------------------------------- freeze

def check_freeze(arch, freeze, train_loader, num_classes, device) -> None:
    set_seed(0)
    model = build_model(arch, num_classes, "pretrained", head_seed=0)
    apply_freeze(model, arch, freeze)
    model = model.to(device)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}   # gồm cả buffer BN

    opt = torch.optim.AdamW(param_groups(model, arch, 1e-3, 0.1, 0.01))
    set_train_mode(model)
    x, y = next(iter(train_loader))
    F.cross_entropy(model(x.to(device)), y.to(device)).backward()
    opt.step()

    after = model.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable <= changed, f"trainable params not updated: {sorted(trainable - changed)[:5]}"
    frozen_prefixes = tuple(m + "." for m in model.frozen_modules)
    leaked = {k for k in changed if k.startswith(frozen_prefixes)}
    assert not leaked, f"frozen state changed: {sorted(leaked)[:5]}"
    log(f"[freeze] {arch}/{freeze}: trainable {count_params(model, True):,} / {count_params(model):,} | "
        f"frozen {model.frozen_modules} | OK: frozen params + BN buffers unchanged")
    del model, opt
    free_memory(device)


# ---------------------------------------------------------------- overfit

def check_overfit(manifest, cfg, num_classes, device, epochs: int = 100, n_images: int = 64) -> None:
    rows = [r for r in manifest if r["split"] == "train"][:n_images]
    tiny = Food101Dataset(rows, cfg["data_dir"], build_transforms(train=False))     # không augment
    loader = DataLoader(tiny, batch_size=32, shuffle=True, num_workers=0)

    for arch in ("resnet50", "vit_b_16"):
        tiny_cfg = {**cfg, "epochs": epochs, "warmup_epochs": 0, "lr_head": 1e-3, "log_every": 0,
                    "arch": arch, "init": "scratch", "freeze": "full"}
        set_seed(cfg["seed"])
        model = build_model(arch, num_classes).to(device)
        s = fit(model, {"train": loader, "val": loader}, tiny_cfg, f"_check_overfit_{arch}", device,
                out_dir=CHECK_DIR, ckpt_dir=CHECK_CKPT_DIR)
        h = s["history"]
        status = "OK" if h["train_acc"][-1] >= 0.95 else "WARN: did not memorise"
        log(f"[overfit] {arch}: train loss {h['train_loss'][0]:.3f} -> {h['train_loss'][-1]:.3f}, "
            f"train acc {h['train_acc'][-1]:.3f} after {epochs} epochs | {status}")
        del model
        free_memory(device)


# ---------------------------------------------------------------- smoke

def tiny_datasets(datasets, n_train=256, n_val=128, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "train": Subset(datasets["train"], rng.choice(len(datasets["train"]), n_train, replace=False)),
        "val":   Subset(datasets["val"],   rng.choice(len(datasets["val"]),   n_val,   replace=False)),
    }


def check_smoke(datasets, cfg, num_classes, device, epochs: int = 2) -> None:
    smoke_cfg = {**cfg, "epochs": epochs, "warmup_epochs": 1, "log_every": 0}
    ds = tiny_datasets(datasets)
    for arch, init, freeze in EXPERIMENTS:
        s = run_experiment(arch, init, freeze, seed=0, cfg=smoke_cfg, datasets=ds, num_classes=num_classes,
                           device=device, out_dir=CHECK_DIR, ckpt_dir=CHECK_CKPT_DIR)
        log(f"[smoke] {arch:9s} {init:10s} {freeze:7s} trainable {s['trainable_params']:>11,} "
            f"| val top-1 {s['best']['val_top1']:.3f}")


# ---------------------------------------------------------------- estimate

def estimate_epoch_time(model, loaders, device, n_steps=20, warmup=3, amp=True) -> dict:
    """Median thời gian một batch (train ở chế độ freeze hiện tại, và eval) x số batch -> phút/epoch."""
    def timed(fn, loader):
        it, ts = iter(loader), []
        for i in range(warmup + n_steps):
            x, y = next(it)
            x, y = x.to(device), y.to(device)
            sync(device); t = time.time()
            fn(x, y)
            sync(device)
            if i >= warmup:
                ts.append(time.time() - t)
        return float(np.median(ts))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    def train_step(x, y):
        set_train_mode(model)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x).float(), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def eval_step(x, y):
        model.eval()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            model(x)

    train_ms = timed(train_step, loaders["train"])
    val_ms = timed(eval_step, loaders["val"])
    return {"train_ms": train_ms * 1000, "val_ms": val_ms * 1000,
            "epoch_min": (train_ms * len(loaders["train"]) + val_ms * len(loaders["val"])) / 60}


def check_estimate(loaders, cfg, num_classes, device, n_steps: int = 20) -> dict:
    estimates = {}
    for arch, init, freeze in EXPERIMENTS:
        set_seed(0)
        m = build_model(arch, num_classes, init)
        apply_freeze(m, arch, freeze)
        m = m.to(device)
        e = estimate_epoch_time(m, loaders, device, n_steps=n_steps, amp=cfg["amp"])
        estimates[(arch, init, freeze)] = e
        log(f"[estimate] {arch:9s} {init:10s} {freeze:7s} train {e['train_ms']:5.0f} ms/batch, "
            f"val {e['val_ms']:4.0f} ms/batch -> {e['epoch_min']:5.1f} min/epoch, "
            f"{e['epoch_min'] * cfg['epochs'] / 60:5.1f} h / {cfg['epochs']} epochs")
        del m
        free_memory(device)

    total_h = sum(e["epoch_min"] for e in estimates.values()) * cfg["epochs"] / 60
    log(f"[estimate] TOTAL {total_h:.1f} h for {len(EXPERIMENTS)} configs x {cfg['epochs']} epochs "
        f"(+10-20% margin -> {total_h * 1.1:.1f}-{total_h * 1.2:.1f} h)")
    return estimates
