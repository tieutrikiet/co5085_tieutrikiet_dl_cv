"""E1 pipeline — Softmax vs. MLP vs. CNN on Fashion-MNIST.

Run from the repo root:
    python -m E1.main                 # full run, same as the notebook
    python -m E1.main --epochs 1      # quick smoke test
    python -m E1.main --show          # also open each figure in a window
"""
import argparse
import json
import time

import matplotlib

from common.datasets import (load_fashion_mnist, load_or_create_split, make_eval_loaders,
                             make_train_loader, pixel_stats, to_tensor_dataset)
from common.evaluation import collect_outputs, evaluate_model, format_results_table
from common.exploration import describe_dataset, describe_pixels, describe_split, plot_samples_per_class
from common.plotting import (plot_confusion_matrices, plot_learning_curves, plot_misclassified,
                             save_fig, setup_style)
from common.training import fit
from common.utils import get_device, set_seed

from .classifiers import MODELS
from .config import (CLASS_NAMES, CONFIG, FIG_DIR, MODEL_COLORS, RESULTS_DIR, SAMPLES_PER_CLASS,
                     SPLIT_PATH)
from .report import export_report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="override CONFIG['epochs']")
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

    # 1. Load data
    train_images, train_labels, test_images, test_labels = load_fashion_mnist(CONFIG["data_dir"])
    describe_dataset(train_images, train_labels, test_images, test_labels)

    # 2. Stratified train/val split (saved and reused)
    train_idx, val_idx = load_or_create_split(train_labels, SPLIT_PATH, CONFIG["seed"], CONFIG["val_ratio"])
    X_train, y_train = train_images[train_idx], train_labels[train_idx]
    X_val, y_val = train_images[val_idx], train_labels[val_idx]
    X_test, y_test = test_images, test_labels
    describe_split(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    # 3. Data exploration
    fig = plot_samples_per_class(X_train, y_train, CLASS_NAMES, SAMPLES_PER_CLASS, CONFIG["seed"],
                                 title="Fashion-MNIST training samples")
    save_fig(fig, FIG_DIR, "E0_samples_per_class", args.show)
    describe_pixels(X_train)

    # 4. Normalize with train-split statistics, build loaders
    mean, std = pixel_stats(X_train)
    datasets = {
        "train": to_tensor_dataset(X_train, y_train, mean, std),
        "val": to_tensor_dataset(X_val, y_val, mean, std),
        "test": to_tensor_dataset(X_test, y_test, mean, std),
    }
    eval_loaders = make_eval_loaders(datasets, CONFIG["batch_size"])

    # 5. Train + evaluate every classifier
    results, histories, outputs = [], {}, {}
    for name, cls in MODELS.items():
        print(f"\nClassifier name: {name}")
        set_seed(CONFIG["seed"])
        model = cls()
        t0 = time.perf_counter()
        histories[name], best_epoch = fit(
            model,
            train_loader=make_train_loader(datasets["train"], CONFIG["batch_size"], CONFIG["seed"]),
            val_loader=eval_loaders["val"],
            epochs=CONFIG["epochs"],
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
            seed=CONFIG["seed"],
            patience=CONFIG["patience"],
            device=device,
        )
        histories[name]["train_time"] = time.perf_counter() - t0
        outputs[name] = collect_outputs(model, eval_loaders["test"], device=device)
        results.append(evaluate_model(model, name, best_epoch, histories[name]["train_time"],
                                      eval_loaders, device=device))

    print("\n" + format_results_table(results))

    # 6. Figures
    save_fig(plot_learning_curves(histories, MODEL_COLORS), FIG_DIR, "E1_learning_curves", args.show)
    fig, cms = plot_confusion_matrices(outputs, CLASS_NAMES)
    save_fig(fig, FIG_DIR, "E1_confusion_matrices", args.show)
    for name, (y_true, y_pred, probs) in outputs.items():
        fig = plot_misclassified(X_test, y_true, y_pred, probs, CLASS_NAMES, name)
        save_fig(fig, FIG_DIR, f"E1_misclassified_{name.lower()}", args.show)

    # 7. Export results + "Nhận xét" report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    report_path = export_report(RESULTS_DIR / "E1_report.txt", results, histories, outputs, cms)
    print(f"\nFigures -> {FIG_DIR}\nReport  -> {report_path}")


if __name__ == "__main__":
    main()
