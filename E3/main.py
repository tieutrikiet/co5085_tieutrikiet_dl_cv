"""E3 pipeline — LSTM vs. GRU trên 3 cách đưa ảnh thành chuỗi, so với baseline MLP/CNN của E1.

Run from the repo root:
    python -m E3.main                       # train các model chưa có trong E3/results, rồi vẽ hình + report
    python -m E3.main --retrain             # bỏ qua kết quả đã lưu, train lại cả 8 model
    python -m E3.main --skip-checks         # bỏ qua các kiểm tra trước khi train
    python -m E3.main --retrain --epochs 1  # quick smoke test (ghi đè E3/results)
    python -m E3.main --show                # also open each figure in a window
"""
import argparse
import json

import matplotlib
import numpy as np

from common.datasets import (load_fashion_mnist, load_or_create_split, make_eval_loaders, make_train_loader,
                             pixel_stats, to_tensor_dataset)
from common.plotting import plot_confusion_matrices, plot_misclassified, save_fig, setup_style
from common.utils import get_device, set_seed
from E2.experiment import ResultStore

from .checks import run_all_checks
from .classifiers import RNNS
from .config import BASELINES, CLASS_NAMES, CONFIG, E1_RESULTS_PATH, FIG_DIR, RESULTS_DIR, SPLIT_PATH
from .experiment import train_all
from .report import export_report, mcnemar_table, recall_table, summary_table
from .sequences import SEQUENCES
from .visualize import plot_lstm_vs_gru_curves, plot_sequence_curves, plot_sequences


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

    # 2. Ảnh -> chuỗi: kiểm tra và minh họa 3 cách biểu diễn
    if not args.skip_checks:
        run_all_checks(make_train_loader(datasets["train"], CONFIG["batch_size"], CONFIG["seed"]))
    i = int(np.flatnonzero(y_test == 0)[0])  # ảnh T-shirt đầu tiên của tập test
    save_fig(plot_sequences(datasets["test"].tensors[0][i]), FIG_DIR, "E3_sequences", args.show)

    # 3. Training: 6 RNN + 2 baseline (model đã có trong results.json thì bỏ qua)
    store = ResultStore(RESULTS_DIR, fresh=args.retrain)
    train_all(store, datasets, eval_loaders, device)
    store.load_outputs()
    results, histories, outputs = store.results, store.histories, store.outputs

    if E1_RESULTS_PATH.exists():  # train lại có khớp với số đã báo cáo ở E1 không
        e1_reported = {r["model"]: r for r in json.loads(E1_RESULTS_PATH.read_text())}
        print(f"\n{'baseline':<10}{'E1 report':>11}{'E3 retrain':>12}")
        for name in BASELINES:
            r = next(r for r in results if r["model"] == name)
            print(f"{name:<10}{e1_reported[name]['test_acc']:>11.2%}{r['test_acc']:>12.2%}")

    # 4. Evaluation
    rnn_names = [f"{r}_{s}" for r in RNNS for s in SEQUENCES]
    names = rnn_names + BASELINES
    by_name = {r["model"]: r for r in results}
    print("\n" + summary_table(by_name, names))

    for seq_name in SEQUENCES:
        save_fig(plot_lstm_vs_gru_curves(histories, seq_name), FIG_DIR, f"E3_curves_{seq_name}_lstm_vs_gru", args.show)
    for rnn_type in RNNS:
        save_fig(plot_sequence_curves(histories, rnn_type), FIG_DIR, f"E3_curves_{rnn_type}_sequences", args.show)

    best_rnn = max(rnn_names, key=lambda n: by_name[n]["val_acc"])  # chọn theo VAL, không theo test
    pairs = [(f"lstm_{s}", f"gru_{s}") for s in SEQUENCES] + [(best_rnn, "MLP"), (best_rnn, "CNN")]
    print("\n" + mcnemar_table(outputs, by_name, best_rnn, pairs))

    cm_names = [best_rnn, "MLP", "CNN"]
    fig, cms = plot_confusion_matrices({n: outputs[n] for n in cm_names}, CLASS_NAMES)
    save_fig(fig, FIG_DIR, "E3_confusion_matrices", args.show)
    print("\n" + recall_table(cms))
    yt, yp, pr = outputs[best_rnn]
    save_fig(plot_misclassified(X_test, yt, yp, pr, CLASS_NAMES, best_rnn), FIG_DIR,
             f"E3_misclassified_{best_rnn}", args.show)

    # 5. Export "Evaluation" report
    report_path = export_report(RESULTS_DIR / "E3_report.txt", by_name, histories, outputs, cms, names,
                                list(RNNS), list(SEQUENCES), best_rnn, pairs)
    print(f"\nFigures -> {FIG_DIR}\nReport  -> {report_path}")


if __name__ == "__main__":
    main()
