"""Thử nghiệm nhỏ trước khi train thật (không dùng trong báo cáo):

overfit  : model học thuộc một tập ảnh nhỏ cố định, không augment (kiểm tra loss + cập nhật trọng số).
smoke    : cả 2 biến thể chạy hết pipeline thật (fit, checkpoint, JSON, resume) với vài batch mỗi epoch.
estimate : đo ms/batch train và val trên DataLoader thật -> ước tính thời gian của thí nghiệm full.
"""
import json
import shutil
import time

import torch

from common.utils import set_seed

from .config import AUGMENT, CHECK_DIR, CKPT_DIR, EXPERIMENTS, LOSS, OPTIM, run_name
from .dataset import WiderFaceDataset, collate_fn, make_datasets, make_loaders
from .loss import YoloFaceLoss
from .models import build_model
from .training import Trainer, free_memory, log, run_experiment, sync

CHECK_CKPT_DIR = CKPT_DIR / "_checks"
CHECKS = ("overfit", "smoke", "estimate")


def reset_check_outputs() -> None:
    """Xoá kết quả và checkpoint của lần kiểm tra trước để không resume nhầm."""
    for d in (CHECK_DIR, CHECK_CKPT_DIR):
        shutil.rmtree(d, ignore_errors=True)


def check_overfit(rows, yolo_dir, cfg, device, n_images: int = 16, iters: int = 150) -> dict:
    """Mỗi biến thể học thuộc n ảnh train (letterbox, không augment); loss tổng phải giảm mạnh."""
    train_rows = [r for r in rows if r["split"] == "train" and r["n_faces"] > 0][:n_images]
    ds = WiderFaceDataset(train_rows, yolo_dir, cfg["img_size"], cfg["min_face_px"], augment=False)
    imgs, targets, _ = collate_fn([ds[i] for i in range(len(ds))])
    x, y = imgs.to(device).float() / 255, targets.to(device)
    log(f"[overfit] {len(ds)} ảnh, {len(targets)} mặt, {int((targets[:, 6] >= 0).sum())} mặt có landmark, {iters} bước")

    result = {}
    for model_name, variant in EXPERIMENTS:
        set_seed(cfg["seed"])
        model = build_model(model_name).to(device)
        loss_fn = YoloFaceLoss(model.head, LOSS, use_landmark=(variant == "landmark"))
        trainer = Trainer(model, loss_fn, {**OPTIM, "warmup_epochs": 0, "warmup_min_iters": 20},
                          len(ds), epochs=1, iters_per_epoch=1, device=device)
        trainer.accumulate = 1
        model.train()
        first = None
        for _ in range(iters):
            trainer.warmup(0)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=trainer.amp):
                preds = model(x)
            loss, items = loss_fn(preds, y)
            trainer.scaler.scale(loss).backward()
            trainer.iters += 1
            trainer.optimizer_step()
            first = items.tolist() if first is None else first
        last = items.tolist()
        drop = 1 - last[4] / first[4]
        key = f"{model_name}_{variant}"
        result[key] = {"first": first, "last": last, "total_drop": drop}
        log(f"[overfit] {key}: tổng {first[4]:.4f} -> {last[4]:.4f} (giảm {drop:.0%}) | "
            f"box {first[0]:.4f} -> {last[0]:.4f} | lmk {first[3]:.4f} -> {last[3]:.4f}")
        if drop < 0.3:
            log(f"[overfit] CẢNH BÁO: {key} giảm chưa tới 30%")
        del model, trainer
        free_memory(device)
    return result


def check_smoke(datasets, cfg, device, epochs: int = 2, train_batches: int = 10, val_batches: int = 5) -> dict:
    """Chạy fit thật với vài batch mỗi epoch, sau đó chạy tiếp thêm 1 epoch để kiểm tra resume."""
    small = {"train": train_batches, "val": val_batches}
    result = {}
    for model_name, variant in EXPERIMENTS:
        c = {**cfg, "epochs": epochs}
        run_experiment(model_name, variant, c, OPTIM, LOSS, datasets, device, CHECK_DIR, CHECK_CKPT_DIR, small)
        # Resume: tên run gắn với số epoch, nên đổi tên checkpoint sang tên của run dài hơn rồi chạy tiếp
        c2 = {**cfg, "epochs": epochs + 1}
        src = CHECK_CKPT_DIR / f"{run_name(model_name, variant, c)}_last.pt"
        shutil.copy(src, CHECK_CKPT_DIR / f"{run_name(model_name, variant, c2)}_last.pt")
        summary = run_experiment(model_name, variant, c2, OPTIM, LOSS, datasets, device, CHECK_DIR,
                                 CHECK_CKPT_DIR, small)
        n_epochs = len(summary["history"]["train"])
        ok = n_epochs == epochs + 1 and (CHECK_DIR / f"{summary['run']}.json").exists()
        result[f"{model_name}_{variant}"] = {"epochs_in_history": n_epochs, "resume_ok": ok}
        log(f"[smoke] {model_name}_{variant}: history {n_epochs} epoch, resume {'OK' if ok else 'LỖI'}")
    return result


def check_estimate(datasets, cfg, device, steps: int = 30, warmup_steps: int = 5) -> dict:
    """Đo ms/batch train (forward + loss + backward + step) và val trên DataLoader thật."""
    set_seed(cfg["seed"])
    model = build_model(EXPERIMENTS[0][0]).to(device)
    loss_fn = YoloFaceLoss(model.head, LOSS)
    loaders = make_loaders(datasets, cfg["batch_size"], cfg["seed"], cfg["num_workers"])
    trainer = Trainer(model, loss_fn, OPTIM, cfg["batch_size"], cfg["epochs"], len(loaders["train"]), device)

    def timed(loader, fn):
        it = iter(loader)
        for _ in range(warmup_steps):
            fn(*next(it))
        sync(device); t0 = time.time()
        for _ in range(steps):
            fn(*next(it))
        sync(device)
        return (time.time() - t0) / steps

    def train_step(imgs, targets, _):
        model.train()
        imgs = imgs.to(device).float() / 255
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=trainer.amp):
            preds = model(imgs)
        loss, _ = loss_fn(preds, targets.to(device))
        trainer.scaler.scale(loss).backward()
        trainer.iters += 1
        if trainer.iters % trainer.accumulate == 0:
            trainer.optimizer_step()

    @torch.no_grad()
    def val_step(imgs, targets, _):
        ema = trainer.ema.ema
        imgs = imgs.to(device).float() / 255
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=trainer.amp):
            preds = ema(imgs, raw=True)
        loss_fn(preds, targets.to(device))

    t_train = timed(loaders["train"], train_step)
    t_val = timed(loaders["val"], val_step)
    n_train, n_val = len(loaders["train"]), len(loaders["val"])
    epoch_min = (t_train * n_train + t_val * n_val) / 60
    projection = {e: round(epoch_min * e * len(EXPERIMENTS) / 60, 1) for e in (10, 20, 30, 50, 100)}
    log(f"[estimate] img {cfg['img_size']} batch {cfg['batch_size']}: train {t_train * 1000:.0f} ms/batch × {n_train} "
        f"+ val {t_val * 1000:.0f} ms/batch × {n_val} = {epoch_min:.1f} phút/epoch")
    log(f"[estimate] tổng giờ cho {len(EXPERIMENTS)} biến thể theo số epoch: {projection}")
    del model, trainer, loaders
    free_memory(device)
    return {"img_size": cfg["img_size"], "batch_size": cfg["batch_size"], "train_ms_per_batch": t_train * 1000,
            "val_ms_per_batch": t_val * 1000, "train_batches": n_train, "val_batches": n_val,
            "minutes_per_epoch": epoch_min, "hours_total_by_epochs": projection}


def run_checks(only, rows, yolo_dir, cfg, device) -> dict:
    datasets = make_datasets(rows, yolo_dir, cfg, AUGMENT)
    steps = {
        "overfit": lambda: check_overfit(rows, yolo_dir, cfg, device),
        "smoke": lambda: check_smoke(datasets, cfg, device),
        "estimate": lambda: check_estimate(datasets, cfg, device),
    }
    report = {}
    for name in CHECKS:
        if name in only:
            t0 = time.time()
            report[name] = steps[name]()
            log(f"[{name}] xong trong {(time.time() - t0) / 60:.1f} phút")
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    (CHECK_DIR / "checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
