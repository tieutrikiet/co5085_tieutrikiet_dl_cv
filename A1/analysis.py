"""Phase 5 — đánh giá trên tập test và phân tích kết quả.

collect_outputs        : nạp `best.pt`, chạy evaluate(keep_outputs=True) -> lưu y_true/logits (.npz, có cache).
compute_metrics        : top-1, top-5, macro-F1, accuracy từng lớp.
plot_confusion_matrix  : ma trận nhầm lẫn 101x101 (chuẩn hoá theo hàng) + bản thu nhỏ cho các lớp bị nhầm nhiều nhất.
plot_learning_curves   : loss / top-1 theo epoch của mọi cấu hình (từ JSON của fit).
error_analysis         : lớp tốt/kém nhất, cặp lớp bị nhầm nhiều nhất, ảnh sai với độ tự tin cao, so sánh CNN vs ViT.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from .data import image_path
from .models import build_model
from .training import evaluate, free_memory, log


# ---------------------------------------------------------------- runs

def discover_runs(result_dir: Path) -> list:
    """Mọi run đã có JSON trong result_dir (bỏ các run `_check_*`), theo thứ tự tên."""
    runs = []
    for p in sorted(result_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        s = json.loads(p.read_text())
        c = s["config"]
        runs.append({"name": s["run"], "arch": c["arch"], "init": c["init"], "freeze": c["freeze"],
                     "label": f"{c['arch']} {c['init']} {c['freeze']}", "summary": s})
    return runs


# ---------------------------------------------------------------- outputs

def collect_outputs(run: dict, dataset, num_classes: int, device, ckpt_dir: Path, out_dir: Path,
                    split: str = "test", batch_size: int = 64, num_workers: int = 8, amp: bool = True,
                    recompute: bool = False) -> dict:
    """y_true [N], logits [N, C] của checkpoint tốt nhất (theo val) trên `split`; cache ở `{run}_{split}.npz`."""
    cache = out_dir / f"{run['name']}_{split}.npz"
    if cache.exists() and not recompute:
        d = np.load(cache)
        if len(d["y_true"]) == len(dataset):
            return {"y_true": d["y_true"], "logits": d["logits"], "indices": d["indices"]}

    ckpt = ckpt_dir / f"{run['name']}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(run["arch"], num_classes, "scratch")       # kiến trúc rỗng, trọng số lấy từ checkpoint
    model.load_state_dict(state["model"])
    model = model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    log(f"[outputs] {run['name']}: best epoch {state['epoch']} -> evaluating {len(dataset):,} {split} images")
    out = evaluate(model, loader, device, amp=amp, keep_outputs=True)
    indices = np.asarray(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, y_true=out["y_true"], logits=out["logits"].astype(np.float16), indices=indices)
    del model
    free_memory(device)
    return {"y_true": out["y_true"], "logits": out["logits"], "indices": indices}


# ---------------------------------------------------------------- metrics

def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def compute_metrics(y_true, logits, num_classes: int) -> dict:
    top5 = np.argsort(-logits, axis=1)[:, :5]
    y_pred = top5[:, 0]
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    return {
        "top1": float((y_pred == y_true).mean()),
        "top5": float((top5 == y_true[:, None]).any(1).mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class_acc": recall, "per_class_f1": f1, "support": support,
        "cm": cm, "y_pred": y_pred,
    }


# ---------------------------------------------------------------- plots

def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str, path: Path, top_k: int = 20) -> None:
    """Hình trái: 101x101 chuẩn hoá theo hàng. Hình phải: top_k lớp có accuracy thấp nhất, kèm nhãn."""
    support = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, support, out=np.zeros(cm.shape), where=support > 0)
    worst = np.argsort(np.diag(cmn))[:top_k]

    fig, (a, b) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={"width_ratios": [1, 1.1]})
    im = a.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    a.set_title(f"{title} — all {len(class_names)} classes (row-normalised)")
    a.set_xlabel("predicted class index"); a.set_ylabel("true class index")
    fig.colorbar(im, ax=a, fraction=0.046)

    sub = cmn[np.ix_(worst, worst)]
    b.imshow(sub, cmap="Blues", vmin=0, vmax=max(sub.max(), 1e-6))
    names = [class_names[i] for i in worst]
    b.set_xticks(range(top_k), names, rotation=90, fontsize=8)
    b.set_yticks(range(top_k), names, fontsize=8)
    for i in range(top_k):
        for j in range(top_k):
            if sub[i, j] >= 0.05:
                b.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center", fontsize=6,
                       color="white" if sub[i, j] > 0.5 * sub.max() else "black")
    b.set_title(f"{top_k} lowest-accuracy classes")
    b.set_xlabel("predicted"); b.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(runs: list, path: Path) -> None:
    """3 panel: train/val loss, train acc / val top-1, LR — mỗi cấu hình một màu."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10.colors
    for k, run in enumerate(runs):
        h, c = run["summary"]["history"], colors[k % 10]
        ep = np.arange(1, len(h["val_top1"]) + 1)
        axes[0].plot(ep, h["train_loss"], "--", color=c)
        axes[0].plot(ep, h["val_loss"], "-", color=c, label=run["label"])
        axes[1].plot(ep, h["train_acc"], "--", color=c)
        axes[1].plot(ep, h["val_top1"], "-", color=c, label=run["label"])
        best = run["summary"]["best"]
        if best["epoch"]:
            axes[1].plot(best["epoch"], best["val_top1"], "o", color=c)
        axes[2].plot(ep, h["lr"], color=c, label=run["label"])
    axes[0].set(title="Loss (dashed = train, solid = val)", xlabel="epoch", ylabel="cross-entropy")
    axes[1].set(title="Accuracy (dashed = train, solid = val top-1, ● = best)", xlabel="epoch", ylabel="accuracy")
    axes[2].set(title="Head LR at epoch start", xlabel="epoch", ylabel="lr", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_comparison(per_class: dict, class_names: list, path: Path) -> None:
    """Accuracy từng lớp của mọi cấu hình, sắp xếp theo trung bình."""
    labels = list(per_class)
    mat = np.stack([per_class[l] for l in labels])
    order = np.argsort(mat.mean(0))
    fig, ax = plt.subplots(figsize=(10, 20))
    for k, l in enumerate(labels):
        ax.plot(mat[k, order], np.arange(len(order)), "o", ms=3, label=l)
    ax.set_yticks(range(len(order)), [class_names[i] for i in order], fontsize=6)
    ax.set_xlabel("test accuracy"); ax.set_xlim(0, 1); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Per-class test accuracy (sorted by mean over configurations)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_gallery(errors: list, base_ds, class_names: list, title: str, path: Path, n: int = 16) -> None:
    """Ảnh gốc (không transform) của các mẫu sai với độ tự tin cao nhất."""
    errors = errors[:n]
    if not errors:
        return
    cols = 4
    rows = int(np.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, e in zip(np.ravel(axes), errors):
        with Image.open(image_path(base_ds.data_dir, base_ds.image_ids[e["index"]])) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"true: {class_names[e['true']]}\npred: {class_names[e['pred']]} ({e['conf']:.2f})", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- error analysis

def softmax(logits):
    z = logits.astype(np.float64) - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def top_confusions(cm: np.ndarray, k: int = 15) -> list:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:k]
    return [(int(i), int(j), int(off[i, j]), float(off[i, j] / max(cm[i].sum(), 1)))
            for i, j in zip(*np.unravel_index(flat, off.shape)) if off[i, j] > 0]


def confident_errors(y_true, logits, indices) -> list:
    p = softmax(logits)
    pred, conf = p.argmax(1), p.max(1)
    wrong = np.where(pred != y_true)[0]
    wrong = wrong[np.argsort(-conf[wrong])]
    return [{"index": int(indices[i]), "true": int(y_true[i]), "pred": int(pred[i]), "conf": float(conf[i])}
            for i in wrong]


def compare_pair(a: dict, b: dict) -> dict:
    """Hai cấu hình trên cùng tập ảnh: cả hai đúng / chỉ A đúng / chỉ B đúng / cả hai sai."""
    ca, cb = a["y_pred"] == a["y_true"], b["y_pred"] == b["y_true"]
    return {"both_correct": int((ca & cb).sum()), "only_a": int((ca & ~cb).sum()),
            "only_b": int((~ca & cb).sum()), "both_wrong": int((~ca & ~cb).sum()),
            "same_wrong_pred": int((~ca & ~cb & (a["y_pred"] == b["y_pred"])).sum())}


# ---------------------------------------------------------------- driver

def run_analysis(runs: list, dataset, class_names: list, device, ckpt_dir: Path, out_dir: Path, fig_dir: Path,
                 split: str = "test", num_workers: int = 8, amp: bool = True, recompute: bool = False) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)
    base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset

    plot_learning_curves(runs, fig_dir / "learning_curves.png")
    log(f"[analysis] learning curves -> {fig_dir / 'learning_curves.png'}")

    results, table = {}, []
    for run in runs:
        try:
            o = collect_outputs(run, dataset, num_classes, device, ckpt_dir, out_dir / "outputs", split,
                                num_workers=num_workers, amp=amp, recompute=recompute)
        except FileNotFoundError as e:
            log(f"[analysis] SKIP {run['name']}: {e}")
            continue
        m = compute_metrics(o["y_true"], o["logits"], num_classes)
        m.update(y_true=o["y_true"], errors=confident_errors(o["y_true"], o["logits"], o["indices"]))
        results[run["label"]] = m
        s = run["summary"]
        table.append({"config": run["label"], "trainable_params": s["trainable_params"],
                      "best_epoch": s["best"]["epoch"], "val_top1": round(s["best"]["val_top1"], 4),
                      f"{split}_top1": round(m["top1"], 4), f"{split}_top5": round(m["top5"], 4),
                      f"{split}_macro_f1": round(m["macro_f1"], 4),
                      "train_time_h": round(s["total_train_time_s"] / 3600, 2),
                      "peak_memory_mb": round(max(s["history"]["memory_mb"]), 0)})
        log(f"[analysis] {run['label']:30s} {split} top-1 {m['top1']:.4f} top-5 {m['top5']:.4f} "
            f"macro-F1 {m['macro_f1']:.4f}")

        slug = run["name"]
        plot_confusion_matrix(m["cm"], class_names, run["label"], fig_dir / f"cm_{slug}.png")
        np.savetxt(out_dir / f"cm_{slug}.csv", m["cm"], fmt="%d", delimiter=",")
        plot_error_gallery(m["errors"], base_ds, class_names, f"{run['label']}: most confident errors",
                           fig_dir / f"errors_{slug}.png")

    if not results:
        log("[analysis] no checkpoints found — nothing to analyse")
        return {}

    with open(out_dir / f"summary_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)

    with open(out_dir / f"per_class_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + [f"{l} acc" for l in results] + [f"{l} f1" for l in results])
        for c in range(num_classes):
            w.writerow([class_names[c]] + [f"{m['per_class_acc'][c]:.4f}" for m in results.values()]
                       + [f"{m['per_class_f1'][c]:.4f}" for m in results.values()])
    plot_per_class_comparison({l: m["per_class_acc"] for l, m in results.items()}, class_names,
                              fig_dir / "per_class_accuracy.png")

    report = {"split": split, "n_images": len(dataset), "configs": {}}
    for label, m in results.items():
        acc = m["per_class_acc"]
        order = np.argsort(acc)
        report["configs"][label] = {
            "top1": m["top1"], "top5": m["top5"], "macro_f1": m["macro_f1"],
            "worst_classes": [(class_names[i], round(float(acc[i]), 3)) for i in order[:10]],
            "best_classes": [(class_names[i], round(float(acc[i]), 3)) for i in order[::-1][:10]],
            "top_confusions": [(class_names[i], class_names[j], n, round(r, 3))
                               for i, j, n, r in top_confusions(m["cm"])],
            "confident_errors": [{**e, "true": class_names[e["true"]], "pred": class_names[e["pred"]],
                                  "image_id": base_ds.image_ids[e["index"]], "conf": round(e["conf"], 3)}
                                 for e in m["errors"][:20]],
        }

    pairs = [("resnet50 pretrained full", "vit_b_16 pretrained full"),
             ("resnet50 scratch full", "resnet50 pretrained full"),
             ("resnet50 pretrained head", "resnet50 pretrained full"),
             ("resnet50 pretrained partial", "resnet50 pretrained full")]
    report["pairs"] = {}
    for a, b in pairs:
        if a in results and b in results:
            r = compare_pair(results[a], results[b])
            report["pairs"][f"{a} vs {b}"] = r
            log(f"[analysis] {a} vs {b}: {r}")

    (out_dir / f"error_analysis_{split}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"[analysis] tables -> {out_dir} | figures -> {fig_dir}")
    return report
