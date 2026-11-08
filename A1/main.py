"""A1 pipeline — ResNet-50 vs ViT-B/16 trên Food-101 (5 cấu hình, 1 seed).

Run from the repo root:
    python -m A1.main manifest                    # tạo (hoặc đọc lại) split manifest + kiểm tra dữ liệu
    python -m A1.main manifest --rebuild          # tạo lại manifest từ đầu (ghi đè file cũ)

    python -m A1.main check                       # thử nghiệm nhỏ: pretrained, freeze, overfit, smoke, estimate
    python -m A1.main check --only freeze smoke   # chỉ chạy một số kiểm tra

    python -m A1.main train                       # thí nghiệm thật: cả 5 cấu hình theo RUN_ORDER
    python -m A1.main train --only resnet50_pretrained_head
    python -m A1.main train --retrain             # bỏ qua kết quả đã có, train lại

    python -m A1.main analyze                     # Phase 5: test outputs, confusion matrix, learning curves, lỗi
    python -m A1.main analyze --source checks --limit 512   # thử nhanh trên kết quả smoke

Thí nghiệm thật chạy rất lâu (~20-25 h); chạy trong terminal và giữ máy không sleep:
    caffeinate -i python -m A1.main train
    tail -f A1/results/experiments/train.log      # theo dõi tiến độ từ terminal khác
Bị ngắt giữa chừng thì chạy lại đúng lệnh: run đã xong được bỏ qua, run dở được resume từ checkpoint.
"""
import argparse
import json
import sys
import time
import traceback

from common.utils import get_device, set_seed

from .config import (CHECK_DIR, CONFIG, EXPERIMENT_DIR, EXPERIMENTS, MANIFEST_PATH, RUN_ORDER, experiment_key,
                     run_name)
from .data import SPLITS, check_manifest, file_sha256, load_manifest, load_or_build_manifest, make_datasets, \
    make_loaders, read_classes
from .training import free_memory, log, run_experiment, set_log_file

EXPERIMENT_KEYS = [experiment_key(*e) for e in EXPERIMENTS]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="create or verify the locked split manifest")
    p.add_argument("--rebuild", action="store_true", help="rebuild the manifest even if it exists")

    from .checks import CHECKS
    p = sub.add_parser("check", help="small-scale checks + time estimate before the real run")
    p.add_argument("--only", nargs="+", choices=CHECKS, default=list(CHECKS))
    p.add_argument("--overfit-epochs", type=int, default=100)
    p.add_argument("--smoke-epochs", type=int, default=2)
    p.add_argument("--estimate-steps", type=int, default=20)

    p = sub.add_parser("train", help="real experiments (all 5 configurations)")
    p.add_argument("--only", nargs="+", choices=EXPERIMENT_KEYS, help="run only these configurations")
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="override CONFIG['epochs']")
    p.add_argument("--retrain", action="store_true", help="train again even if a finished result exists")

    p = sub.add_parser("analyze", help="evaluate best checkpoints on test + figures + error analysis")
    p.add_argument("--source", choices=("experiments", "checks"), default="experiments")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--limit", type=int, default=None, help="evaluate a random subset of N images (quick test)")
    p.add_argument("--recompute", action="store_true", help="ignore cached outputs (.npz)")
    return parser.parse_args()


# ---------------------------------------------------------------- shared setup

def load_data(cfg):
    if not MANIFEST_PATH.exists():
        sys.exit(f"Manifest not found at {MANIFEST_PATH}. Run `python -m A1.main manifest` first.")
    class_names = read_classes(cfg["data_dir"])
    manifest = load_manifest(MANIFEST_PATH)
    log(f"Manifest: {len(manifest):,} rows | SHA-256 {file_sha256(MANIFEST_PATH)[:16]}…")
    datasets = make_datasets(manifest, cfg["data_dir"], cfg["image_size"])
    return class_names, manifest, datasets


# ---------------------------------------------------------------- commands

def cmd_manifest(args, cfg):
    if not (cfg["data_dir"] / "images").is_dir():
        sys.exit(f"Food-101 not found at {cfg['data_dir']}")
    class_names = read_classes(cfg["data_dir"])
    if args.rebuild and MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    action = "Loading" if MANIFEST_PATH.exists() else "Building (hash + decode every image)"
    log(f"{action} manifest at {MANIFEST_PATH} (split_seed={cfg['split_seed']})")

    t0 = time.time()
    manifest = load_or_build_manifest(MANIFEST_PATH, cfg["data_dir"], class_names,
                                      val_per_class=cfg["per_class"]["val"], split_seed=cfg["split_seed"])
    log(f"Manifest rows: {len(manifest):,} ({time.time() - t0:.1f}s) | SHA-256 {file_sha256(MANIFEST_PATH)}")

    report = check_manifest(manifest, len(class_names), cfg["per_class"])
    log(f"Counts per split: {report['counts']}")
    log(f"Total counts match protocol: {report['counts_ok']} | per-class: {report['per_class_ok']} | "
        f"IDs disjoint: {report['disjoint_ids']}")
    log(f"Unreadable files: {len(report['unreadable'])} | non-RGB modes: {dict(report['non_rgb'])}")
    log(f"Exact-hash duplicate groups: {report['duplicate_groups']} | "
        f"cross-split: {len(report['cross_split_duplicates'])}")
    for group in report["cross_split_duplicates"]:
        log(f"  {group}")
    if not (report["counts_ok"] and report["per_class_ok"] and report["disjoint_ids"] and not report["unreadable"]):
        sys.exit("Manifest integrity check FAILED")


def cmd_check(args, cfg, device):
    from . import checks
    checks.reset_check_outputs()                     # xoá kết quả/checkpoint của lần kiểm tra trước
    set_log_file(CHECK_DIR / "check.log")
    log(f"Checks: {args.only} | device {device}")
    class_names, manifest, datasets = load_data(cfg)
    num_classes = len(class_names)
    loaders = make_loaders(datasets, batch_size=cfg["physical_batch"], seed=cfg["seed"],
                           num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))

    steps = {
        "pretrained": lambda: checks.check_pretrained(datasets, class_names, device),
        "freeze": lambda: [checks.check_freeze("resnet50", f, loaders["train"], num_classes, device)
                           for f in ("head", "partial")],
        "overfit": lambda: checks.check_overfit(manifest, cfg, num_classes, device, epochs=args.overfit_epochs),
        "smoke": lambda: checks.check_smoke(datasets, cfg, num_classes, device, epochs=args.smoke_epochs),
        "estimate": lambda: checks.check_estimate(loaders, cfg, num_classes, device, n_steps=args.estimate_steps),
    }
    failed = []
    for name in checks.CHECKS:
        if name not in args.only:
            continue
        log(f"===== check: {name} =====")
        t0 = time.time()
        try:
            steps[name]()
            log(f"===== {name} done in {(time.time() - t0) / 60:.1f} min =====")
        except Exception:
            failed.append(name)
            log(f"===== {name} FAILED =====\n{traceback.format_exc()}")
    log(f"Checks finished. Failed: {failed or 'none'}")
    if failed:
        sys.exit(1)


def is_finished(arch, init, freeze, cfg) -> bool:
    path = EXPERIMENT_DIR / f"{run_name(arch, init, freeze, cfg, cfg['seed'])}.json"
    if not path.exists():
        return False
    return len(json.loads(path.read_text())["history"]["val_top1"]) >= cfg["epochs"]


def cmd_train(args, cfg, device):
    cfg = {**cfg, "epochs": args.epochs}
    set_log_file(EXPERIMENT_DIR / "train.log")
    todo = [e for e in RUN_ORDER if not args.only or experiment_key(*e) in args.only]
    log(f"Training {len(todo)} configuration(s), {cfg['epochs']} epochs each, seed {cfg['seed']}, device {device}")
    class_names, _, datasets = load_data(cfg)
    for split in SPLITS:
        log(f"  {split:5s}: {len(datasets[split]):6,} images")

    run_start = time.time()
    results, failed = {}, []
    for k, (arch, init, freeze) in enumerate(todo, 1):
        key = experiment_key(arch, init, freeze)
        log("=" * 90)
        log(f"RUN {k}/{len(todo)}: {key} | elapsed {(time.time() - run_start) / 3600:.2f} h")
        log("=" * 90)
        if not args.retrain and is_finished(arch, init, freeze, cfg):
            log(f"SKIP {key}: finished result already in {EXPERIMENT_DIR}")
            continue
        if args.retrain:
            from .config import CKPT_DIR
            for suffix in ("_last.pt", "_best.pt"):
                (CKPT_DIR / f"{run_name(arch, init, freeze, cfg, cfg['seed'])}{suffix}").unlink(missing_ok=True)
        t0 = time.time()
        try:
            s = run_experiment(arch, init, freeze, seed=cfg["seed"], cfg=cfg, datasets=datasets,
                               num_classes=len(class_names), device=device, out_dir=EXPERIMENT_DIR)
            results[key] = s["best"]
            log(f"DONE {key}: best val top-1 {s['best']['val_top1']:.4f} @ epoch {s['best']['epoch']} "
                f"({(time.time() - t0) / 3600:.2f} h)")
        except Exception:
            failed.append(key)
            log(f"FAILED {key}:\n{traceback.format_exc()}")      # tiếp tục với cấu hình sau
        finally:
            free_memory(device)

    log("=" * 90)
    log(f"All runs finished in {(time.time() - run_start) / 3600:.2f} h | failed: {failed or 'none'}")
    for key, best in results.items():
        log(f"  {key:28s} best val top-1 {best['val_top1']:.4f} @ epoch {best['epoch']}")
    if failed:
        sys.exit(1)


def cmd_analyze(args, cfg, device):
    import numpy as np
    from torch.utils.data import Subset

    from . import analysis
    from .config import CKPT_DIR, FIG_DIR
    result_dir = EXPERIMENT_DIR if args.source == "experiments" else CHECK_DIR
    ckpt_dir = CKPT_DIR if args.source == "experiments" else CKPT_DIR / "_checks"
    out_dir = result_dir.parent / ("analysis" if args.source == "experiments" else "_checks_analysis")
    fig_dir = FIG_DIR / ("results" if args.source == "experiments" else "_checks")
    set_log_file(out_dir / "analyze.log")

    runs = analysis.discover_runs(result_dir)
    if not runs:
        sys.exit(f"No finished runs in {result_dir}")
    log(f"Analyzing {len(runs)} run(s) from {result_dir} on {args.split} | device {device}")
    class_names, _, datasets = load_data(cfg)
    ds = datasets[args.split]
    if args.limit:
        idx = np.sort(np.random.default_rng(0).choice(len(ds), min(args.limit, len(ds)), replace=False))
        ds = Subset(ds, idx)
    analysis.run_analysis(runs, ds, class_names, device, ckpt_dir, out_dir, fig_dir, split=args.split,
                          num_workers=cfg["num_workers"], amp=cfg["amp"], recompute=args.recompute)


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    set_seed(cfg["seed"])
    if args.command == "manifest":
        cmd_manifest(args, cfg)
        return
    device = get_device()
    if args.command == "check":
        cmd_check(args, cfg, device)
    elif args.command == "train":
        cmd_train(args, cfg, device)
    elif args.command == "analyze":
        cmd_analyze(args, cfg, device)


if __name__ == "__main__":
    main()
