"""E2 pipeline — MSA tự viết vs. nn.MultiheadAttention, với 3 cách tokenize trên Fashion-MNIST.

Run from the repo root:
    python -m E2.main                       # train các model chưa có trong E2/results, rồi vẽ hình + report
    python -m E2.main --retrain             # bỏ qua kết quả đã lưu, train lại cả 6 model
    python -m E2.main --skip-checks         # bỏ qua các kiểm tra trước khi train (benchmark, overfit 1 batch)
    python -m E2.main --retrain --epochs 1  # quick smoke test (ghi đè E2/results)
    python -m E2.main --show                # also open each figure in a window
"""
import argparse
import json

import matplotlib
import numpy as np

from common.datasets import (load_fashion_mnist, load_or_create_split, make_eval_loaders, make_train_loader,
                             pixel_stats, to_tensor_dataset)
from common.evaluation import format_results_table
from common.plotting import plot_confusion_matrices, plot_misclassified, save_fig, setup_style
from common.utils import get_device, set_seed

from .checks import run_all_checks
from .config import CLASS_NAMES, CONFIG, E1_RESULTS_PATH, FIG_DIR, RESULTS_DIR, SPLIT_PATH
from .experiment import ResultStore, get_model, train_all
from .report import export_report, mcnemar_table, recall_table, summary_table
from .tokenizers import TOKENIZERS
from .visualize import plot_acc_vs_cost, plot_attention, plot_scratch_vs_torch_curves, plot_tokenizer_curves


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="override CONFIG['epochs']")
    parser.add_argument("--retrain", action="store_true", help="ignore saved results and train every model again")
    parser.add_argument("--skip-checks", action="store_true", help="skip the pre-training sanity checks")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    return parser.parse_args()


def main():
    args = parse_args()
    CONFIG["epochs"] = args.epochs
    if not args.show:
        matplotlib.use("Agg")  # headless: save figures only
    setup_style()

    set_seed(CONFIG["seed"])
    device = get_device()
    print(f"Device: {device}")

    # 1. Load data, dùng lại đúng split và chuẩn hóa của E1
    train_images, train_labels, test_images, test_labels = load_fashion_mnist(CONFIG["data_dir"])
    train_idx, val_idx = load_or_create_split(train_labels, SPLIT_PATH, CONFIG["seed"], CONFIG["val_ratio"])
    X_train, y_train = train_images[train_idx], train_labels[train_idx]
    X_val, y_val = train_images[val_idx], train_labels[val_idx]
    X_test, y_test = test_images, test_labels
    mean, std = pixel_stats(X_train)
    datasets = {
        "train": to_tensor_dataset(X_train, y_train, mean, std),
        "val": to_tensor_dataset(X_val, y_val, mean, std),
        "test": to_tensor_dataset(X_test, y_test, mean, std),
    }
    eval_loaders = make_eval_loaders(datasets, CONFIG["batch_size"])

    # 2. Kiểm tra MSA và mô hình trước khi train
    if not args.skip_checks:
        run_all_checks(make_train_loader(datasets["train"], CONFIG["batch_size"], CONFIG["seed"]), device)

    # 3. Training (model đã có trong results.json thì bỏ qua)
    store = ResultStore(RESULTS_DIR, fresh=args.retrain)
    train_all(store, datasets, eval_loaders, device)
    store.load_outputs()
    results, histories, outputs = store.results, store.histories, store.outputs

    # 4. Evaluation
    print("\n" + format_results_table(results))

    for tok in TOKENIZERS:
        save_fig(plot_scratch_vs_torch_curves(histories, tok), FIG_DIR, f"E2_curves_{tok}_scratch_vs_torch", args.show)
    save_fig(plot_tokenizer_curves(histories, TOKENIZERS), FIG_DIR, "E2_curves_tokenizers", args.show)

    print("\n" + mcnemar_table(outputs, TOKENIZERS))

    print("\n" + summary_table(results))
    e1 = ({r["model"]: r for r in json.loads(E1_RESULTS_PATH.read_text())}
          if E1_RESULTS_PATH.exists() else None)
    save_fig(plot_acc_vs_cost(results, e1), FIG_DIR, "E2_acc_vs_cost", args.show)

    scratch_outs = {t: outputs[f"{t}_scratch"] for t in TOKENIZERS}
    fig, cms = plot_confusion_matrices(scratch_outs, CLASS_NAMES)
    save_fig(fig, FIG_DIR, "E2_confusion_matrices", args.show)
    print("\n" + recall_table(cms))
    for t, (yt, yp, pr) in scratch_outs.items():
        save_fig(plot_misclassified(X_test, yt, yp, pr, CLASS_NAMES, f"{t}_scratch"),
                 FIG_DIR, f"E2_misclassified_{t}", args.show)

    models_attn = {t: get_model(t, RESULTS_DIR, datasets, eval_loaders, device) for t in TOKENIZERS}
    pick = [int(np.flatnonzero(y_test == k)[0]) for k in range(10)]  # ảnh test đầu tiên của mỗi lớp
    fig = plot_attention(models_attn, pick, datasets["test"].tensors[0], X_test, y_test, device)
    save_fig(fig, FIG_DIR, "E2_attention_rollout", args.show)

    # 5. Export "Evaluation" report
    report_path = export_report(RESULTS_DIR / "E2_report.txt", results, histories, outputs, cms, list(TOKENIZERS))
    print(f"\nFigures -> {FIG_DIR}\nReport  -> {report_path}")


if __name__ == "__main__":
    main()
