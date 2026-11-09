"""A2 pipeline — YOLO5Face trên WIDERFace.

Chạy từ thư mục gốc của repo:
    python -m A2.main manifest              # tạo (hoặc đọc lại) manifest + in thống kê dữ liệu
    python -m A2.main manifest --rebuild    # tạo lại manifest từ đầu (ghi đè file cũ)
    python -m A2.main preview               # vẽ một batch train/val + đo tốc độ DataLoader
    python -m A2.main preview --batches 50 --workers 4
"""
import argparse
import json
import sys
import time

from .config import AUGMENT, CONFIG, FIG_DIR, MANIFEST_PATH, REPO_DIR, RESULTS_DIR, WIDER_DIR, YOLO_DIR
from .data import SPLITS, check_manifest, file_sha256, load_manifest, load_or_build_manifest

# Số ảnh chuẩn của WIDERFace, dùng để phát hiện dữ liệu thiếu
EXPECTED_IMAGES = {"train": 12880, "val": 3226}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="tạo hoặc kiểm tra manifest dữ liệu")
    p.add_argument("--rebuild", action="store_true", help="tạo lại manifest dù đã có")

    p = sub.add_parser("preview", help="vẽ batch đã augment và đo tốc độ DataLoader")
    p.add_argument("--batches", type=int, default=20, help="số batch train dùng để đo tốc độ")
    p.add_argument("--workers", type=int, default=CONFIG["num_workers"])
    return parser.parse_args()


def cmd_manifest(args, cfg):
    for split in SPLITS:
        if not (YOLO_DIR / split).is_dir():
            sys.exit(f"Không thấy {YOLO_DIR / split}. Chạy train2yolo.py / val2yolo.py trước.")
    if not (WIDER_DIR / "val" / "label.txt").is_file():
        sys.exit(f"Không thấy {WIDER_DIR / 'val' / 'label.txt'} (cần để lấy tên gốc ảnh val).")

    action = "Tạo lại" if args.rebuild or not MANIFEST_PATH.exists() else "Đọc"
    print(f"{action} manifest: {MANIFEST_PATH.relative_to(REPO_DIR)} (min_face_px={cfg['min_face_px']})")
    t0 = time.time()
    rows = load_or_build_manifest(MANIFEST_PATH, YOLO_DIR, WIDER_DIR, cfg["min_face_px"], rebuild=args.rebuild)
    print(f"{len(rows):,} dòng ({time.time() - t0:.1f}s) | SHA-256 {file_sha256(MANIFEST_PATH)[:16]}…")

    summary = check_manifest(rows, YOLO_DIR, cfg["min_face_px"])

    # Lưu thống kê để dùng khi viết báo cáo
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "data_summary.json"
    out.write_text(json.dumps({"min_face_px": cfg["min_face_px"], "splits": summary}, indent=2, ensure_ascii=False))
    print(f"\nĐã lưu thống kê: {out.relative_to(REPO_DIR)}")

    bad = {s: summary[s]["images"] for s in SPLITS if summary[s]["images"] != EXPECTED_IMAGES[s]}
    if bad:
        sys.exit(f"Số ảnh không khớp WIDERFace {EXPECTED_IMAGES}: {bad}")


def cmd_preview(args, cfg):
    from .dataset import make_datasets, make_loaders, plot_batch

    if not MANIFEST_PATH.exists():
        sys.exit("Chưa có manifest. Chạy `python -m A2.main manifest` trước.")
    datasets = make_datasets(load_manifest(MANIFEST_PATH), YOLO_DIR, cfg, AUGMENT)
    loaders = make_loaders(datasets, cfg["batch_size"], cfg["seed"], args.workers)
    print(f"train {len(datasets['train'])} ảnh / {len(loaders['train'])} batch | "
          f"val {len(datasets['val'])} ảnh / {len(loaders['val'])} batch | img_size {cfg['img_size']}")

    # Đo tốc độ trên các batch train; batch đầu tính riêng vì gồm thời gian khởi động worker
    t0 = time.time()
    it = iter(loaders["train"])
    imgs, labels, _ = next(it)
    t_first = time.time() - t0
    plot_batch(imgs, labels, FIG_DIR / "preview_train.png", "train (đã augment)")
    t0 = time.time()
    n_img = 0
    for _ in range(args.batches):
        imgs, labels, _ = next(it)
        n_img += len(imgs)
    dt = time.time() - t0
    print(f"batch đầu {t_first:.1f}s | {args.batches} batch tiếp: {n_img / dt:.0f} ảnh/s "
          f"({dt / args.batches * 1000:.0f} ms/batch, {args.workers} worker)")
    print(f"batch cuối: ảnh {tuple(imgs.shape)} {imgs.dtype} | nhãn {tuple(labels.shape)}")

    imgs, labels, metas = next(iter(loaders["val"]))
    plot_batch(imgs, labels, FIG_DIR / "preview_val.png", "val (letterbox, không augment)")
    print(f"val: ảnh {tuple(imgs.shape)} | meta[0] {metas[0]}")
    print(f"Đã lưu hình: {(FIG_DIR / 'preview_train.png').relative_to(REPO_DIR)}, "
          f"{(FIG_DIR / 'preview_val.png').relative_to(REPO_DIR)}")


def main():
    args = parse_args()
    if args.command == "manifest":
        cmd_manifest(args, CONFIG)
    elif args.command == "preview":
        cmd_preview(args, CONFIG)


if __name__ == "__main__":
    main()
