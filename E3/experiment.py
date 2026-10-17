"""Train + evaluate một cấu hình RNN (hoặc một baseline của E1) và lưu tiến độ để chạy tiếp được."""
import time

import numpy as np

from common.datasets import FASHION_MNIST_SHAPE, make_train_loader
from common.evaluation import collect_outputs, evaluate_model, format_results_table
from common.training import fit
from common.utils import set_seed
from E1.classifiers import MODELS as E1_MODELS
from E2.experiment import ResultStore
from E2.timing import measure_latency

from .classifiers import RNNS, build_model
from .config import BASELINES, CONFIG
from .sequences import SEQUENCES, sequence_shape


def train_and_evaluate(name, make_model, datasets, eval_loaders, device, seed=CONFIG["seed"]):
    """Vòng chung cho mọi model: tạo model -> fit -> evaluate -> outputs trên test."""
    set_seed(seed)
    model = make_model()

    train_loader = make_train_loader(datasets["train"], CONFIG["batch_size"], seed)
    t0 = time.perf_counter()
    history, best_epoch = fit(
        model=model,
        train_loader=train_loader,
        val_loader=eval_loaders["val"],
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        seed=seed,
        patience=CONFIG["patience"],
        device=device,
    )

    t1 = time.perf_counter()
    history["train_time"] = t1 - t0

    row = evaluate_model(
        model=model,
        name=name,
        best_epoch=best_epoch,
        train_time=history["train_time"],
        eval_loaders=eval_loaders,
        device=device,
    )

    row.update(
        epochs_run=len(history["val_acc"]),
        sec_per_epoch=float(np.mean(history["epoch_time"])),
        latency_ms=measure_latency(model),
    )

    outputs = collect_outputs(
        model=model,
        dataloader=eval_loaders["test"],
        device=device,
    )

    return row, history, outputs, model


def run_experiment(rnn_type, seq_name, datasets, eval_loaders, device, seed=CONFIG["seed"]):
    name = f"{rnn_type}_{seq_name}"
    row, history, outputs, model = train_and_evaluate(name, lambda: build_model(rnn_type, seq_name),
                                                      datasets, eval_loaders, device, seed)
    T, D = sequence_shape(SEQUENCES[seq_name])
    row.update(
        rnn_type=rnn_type,
        sequence=seq_name,
        num_step=T,
        input_size=D,
    )
    return row, history, outputs, model


def run_baseline(name, datasets, eval_loaders, device, seed=CONFIG["seed"]):
    row, history, outputs, model = train_and_evaluate(name, lambda: E1_MODELS[name](),
                                                      datasets, eval_loaders, device, seed)
    row.update(
        rnn_type="baseline",
        sequence="image",
        num_step=1,
        input_size=int(np.prod(FASHION_MNIST_SHAPE)),
    )
    return row, history, outputs, model


def train_all(store: ResultStore, datasets, eval_loaders, device):
    """LSTM/GRU × row/col/patch, rồi baseline MLP/CNN; model đã có trong results.json thì bỏ qua."""
    for rnn_type in RNNS:
        for seq_name in SEQUENCES:
            name = f"{rnn_type}_{seq_name}"
            if name in store.done:
                print(f"Đã chạy {name}")
                continue

            print(f"Chạy {name}")
            row, history, outputs, model = run_experiment(rnn_type, seq_name, datasets, eval_loaders, device)
            store.record(name, row, history, outputs, model)

    for name in BASELINES:
        if name in store.done:
            print(f"Đã chạy baseline {name}")
            continue

        print(f"Chạy baseline {name}")
        row, history, outputs, model = run_baseline(name, datasets, eval_loaders, device)
        store.record(name, row, history, outputs, model)

    print(format_results_table(store.results))
