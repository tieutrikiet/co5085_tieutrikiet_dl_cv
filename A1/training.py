"""Vòng train/eval cho A1: gradient accumulation, AMP, clipping, warmup + cosine, checkpoint/resume, log tiến độ."""
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from common.utils import set_seed

from .config import CKPT_DIR, run_name as make_run_name
from .data import make_loaders
from .models import PRETRAINED_WEIGHTS, apply_freeze, build_model, count_params, param_groups, set_train_mode

# ---------------------------------------------------------------- logging

_LOG_FILE: Optional[Path] = None


def set_log_file(path: Optional[Path]) -> None:
    """Mọi lời gọi `log` sau đó sẽ được ghi thêm vào file này (theo dõi bằng `tail -f`)."""
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


# ---------------------------------------------------------------- device helpers

def sync(device: torch.device) -> None:
    """Chờ GPU/MPS tính xong trước khi đọc đồng hồ (GPU chạy bất đồng bộ với Python)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_mb(device: torch.device) -> float:
    """CUDA: bộ nhớ đỉnh; MPS: bộ nhớ hiện tại của driver (MPS không có API đỉnh)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return float("nan")


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- schedule

def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Hệ số LR theo optimizer step: warmup tuyến tính 0.1x -> 1x, rồi cosine 1x -> 0.01x."""
    def lr_lambda(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def accumulation_group_sizes(n_samples: int, batch_size: int, accum_steps: int) -> list:
    """Số mẫu thật của từng nhóm accumulation. Train: 1,894 batch -> 474 nhóm, nhóm cuối 2 batch = 56 ảnh."""
    n_batches = -(-n_samples // batch_size)
    batch_sizes = [min(batch_size, n_samples - b * batch_size) for b in range(n_batches)]
    return [sum(batch_sizes[i:i + accum_steps]) for i in range(0, n_batches, accum_steps)]


# ---------------------------------------------------------------- train / eval

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, accum_steps: int, clip: float,
                    amp: bool = True, log_every: int = 200, tag: str = "") -> dict:
    """Một epoch: loss nhóm = tổng CE / số ảnh của cả nhóm; unscale -> clip -> step -> scheduler.step."""
    set_train_mode(model)
    group_sizes = accumulation_group_sizes(len(loader.dataset), loader.batch_size, accum_steps)
    n_batches = len(loader)
    trainable = [p for p in model.parameters() if p.requires_grad]

    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0
    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        n_group = group_sizes[i // accum_steps]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        batch_loss = F.cross_entropy(logits.float(), y, reduction="sum")
        scaler.scale(batch_loss / n_group).backward()

        loss_sum += batch_loss.detach()
        correct += (logits.argmax(dim=1) == y).sum()
        seen += y.size(0)

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if log_every and ((i + 1) % log_every == 0 or i + 1 == n_batches):
            done = (i + 1) / n_batches
            elapsed = time.time() - t_start
            log(f"  {tag} batch {i + 1:5d}/{n_batches} ({done:5.1%}) | loss {(loss_sum / seen).item():.3f} "
                f"| acc {(correct.float() / seen).item():.3f} | {elapsed / 60:5.1f} min, "
                f"ETA {elapsed / done * (1 - done) / 60:5.1f} min")

    return {"loss": (loss_sum / seen).item(), "acc": (correct.float() / seen).item()}


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = True, keep_outputs: bool = False) -> dict:
    """Loss / top-1 / top-5; keep_outputs giữ y_true và logits [N, 101] cho Phase 5."""
    model.eval()
    loss_sum = torch.zeros((), device=device)
    top1 = torch.zeros((), device=device, dtype=torch.long)
    top5 = torch.zeros((), device=device, dtype=torch.long)
    seen = 0
    all_y, all_logits = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        logits = logits.float()

        loss_sum += F.cross_entropy(logits, y, reduction="sum")
        top5_idx = logits.topk(5, dim=1).indices
        top1 += (top5_idx[:, 0] == y).sum()
        top5 += (top5_idx == y[:, None]).any(dim=1).sum()
        seen += y.size(0)
        if keep_outputs:
            all_y.append(y.cpu())
            all_logits.append(logits.cpu())

    out = {"loss": (loss_sum / seen).item(), "top1": (top1.float() / seen).item(),
           "top5": (top5.float() / seen).item()}
    if keep_outputs:
        out["y_true"] = torch.cat(all_y).numpy()
        out["logits"] = torch.cat(all_logits).numpy()
    return out


# ---------------------------------------------------------------- fit

def fit(model, loaders: Dict, cfg: dict, run_name: str, device, out_dir: Path, ckpt_dir: Path = CKPT_DIR) -> dict:
    """Một lần train hoàn chỉnh, resume được từ `{run_name}_last.pt`; ghi `{run_name}.json` + `.log` vào out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg["arch"], cfg["lr_head"], cfg["backbone_lr_mult"][cfg["init"]], cfg["weight_decay"]),
        betas=cfg["betas"], eps=cfg["eps"],
    )
    steps_per_epoch = math.ceil(len(loaders["train"]) / cfg["accum_steps"])
    scheduler = make_scheduler(optimizer, total_steps=cfg["epochs"] * steps_per_epoch,
                               warmup_steps=cfg["warmup_epochs"] * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"])

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_top1", "val_top5",
                               "lr", "epoch_time", "val_time", "memory_mb"]}
    best = {"val_top1": -1.0, "epoch": None}
    start_epoch = 0

    last_path = ckpt_dir / f"{run_name}_last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best, start_epoch = state["history"], state["best"], state["epoch"]
        torch.set_rng_state(state["rng_cpu"])
        log(f"Resumed {run_name} from epoch {start_epoch}/{cfg['epochs']}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, cfg["epochs"]):
        lr_now = next(g["lr"] for g in optimizer.param_groups if g["name"].startswith("head"))
        tag = f"[{run_name}] ep {epoch + 1}/{cfg['epochs']}"

        sync(device); t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], optimizer, scheduler, scaler, device,
                             cfg["accum_steps"], cfg["grad_clip"], cfg["amp"], cfg.get("log_every", 200), tag)
        sync(device); t1 = time.time()
        va = evaluate(model, loaders["val"], device, cfg["amp"])
        sync(device); t2 = time.time()

        for k, v in [("train_loss", tr["loss"]), ("train_acc", tr["acc"]),
                     ("val_loss", va["loss"]), ("val_top1", va["top1"]), ("val_top5", va["top5"]),
                     ("lr", lr_now), ("epoch_time", t1 - t0), ("val_time", t2 - t1),
                     ("memory_mb", memory_mb(device))]:
            history[k].append(v)

        if va["top1"] > best["val_top1"]:          # bằng nhau -> giữ epoch sớm hơn
            best = {"val_top1": va["top1"], "epoch": epoch + 1}
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "config": cfg},
                       ckpt_dir / f"{run_name}_best.pt")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best": best, "epoch": epoch + 1,
                    "rng_cpu": torch.get_rng_state()}, last_path)

        line = (f"[{run_name}] epoch {epoch + 1:2d}/{cfg['epochs']} | lr {lr_now:.2e} | "
                f"train {tr['loss']:.3f}/{tr['acc']:.3f} | val {va['loss']:.3f}/{va['top1']:.3f} "
                f"(top5 {va['top5']:.3f}) | {t1 - t0:.0f}s + {t2 - t1:.0f}s")
        log(line)
        with open(out_dir / f"{run_name}.log", "a") as f:
            f.write(line + "\n")

    summary = {
        "run": run_name,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
        "best": best,
        "history": history,
        "total_params": count_params(model),
        "trainable_params": count_params(model, trainable_only=True),
        "total_train_time_s": sum(history["epoch_time"]),
        "total_val_time_s": sum(history["val_time"]),
        "weights": PRETRAINED_WEIGHTS[cfg["arch"]].url if cfg["init"] == "pretrained" else None,
    }
    (out_dir / f"{run_name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_experiment(arch: str, init: str, freeze: str, seed: int, cfg: dict, datasets: Dict, num_classes: int,
                   device, out_dir: Path, ckpt_dir: Path = CKPT_DIR) -> dict:
    """build_model -> apply_freeze -> DataLoader theo seed -> fit."""
    cfg = {**cfg, "arch": arch, "init": init, "freeze": freeze, "seed": seed}
    set_seed(seed)
    model = build_model(arch, num_classes, init, cfg["head_init_std"], head_seed=seed)
    cfg["frozen_modules"] = apply_freeze(model, arch, freeze)
    model = model.to(device)
    log(f"{arch}/{init}/{freeze}: trainable {count_params(model, True):,} / {count_params(model):,} params, "
        f"frozen modules {cfg['frozen_modules'] or '-'}")

    loaders = make_loaders(datasets, batch_size=cfg["physical_batch"], seed=seed,
                           num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
    name = make_run_name(arch, init, freeze, cfg, seed)
    try:
        return fit(model, loaders, cfg, name, device, out_dir=out_dir, ckpt_dir=ckpt_dir)
    finally:
        del model, loaders
        free_memory(device)
