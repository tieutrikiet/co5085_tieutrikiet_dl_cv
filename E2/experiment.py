"""Train + evaluate một cấu hình (tokenizer, attention) và lưu tiến độ để chạy tiếp được."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.datasets import make_train_loader
from common.evaluation import collect_outputs, evaluate_model, format_results_table
from common.training import fit
from common.utils import set_seed

from .attention import scratch_to_torch_state_dict
from .config import CONFIG
from .timing import measure_latency
from .tokenizers import TOKENIZERS
from .transformer import build_model


def run_experiment(tokenizer_name, datasets, eval_loaders, device, attention="scratch",
                   init_state=None, seed=CONFIG['seed']):
    assert attention in ["scratch", "torch"], "Attention must be either 'scratch' or 'torch'"
    name = f"{tokenizer_name}_{attention}"
    set_seed(seed)
    model = build_model(tokenizer_name, attention)

    if init_state is not None:
        model.load_state_dict(init_state)

    init_snapshot = {
        k: v.detach().clone() for k, v in model.state_dict().items()
    }

    train_loader = make_train_loader(datasets["train"], CONFIG["batch_size"], CONFIG["seed"])
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
    history["train_time"] = time.perf_counter() - t0

    row = evaluate_model(
        model=model,
        name=name,
        best_epoch=best_epoch,
        train_time=history["train_time"],
        eval_loaders=eval_loaders,
        device=device,
    )
    row.update(
        tokenizer=tokenizer_name,
        attention=attention,
        num_tokens=model.tokenizer.num_tokens,
        sec_per_epoch=float(np.mean(history["epoch_time"])),
        latency_ms=measure_latency(model),
    )

    outputs = collect_outputs(
        model=model,
        dataloader=eval_loaders["test"],
        device=device,
    )

    return row, history, outputs, model, init_snapshot


class ResultStore:
    """results.json + histories.json + model_*.pt + outputs_*.npz; model đã có trong results thì bỏ qua."""

    def __init__(self, results_dir: Path, fresh: bool = False):
        self.dir = results_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results_path, self.hist_path = self.dir / "results.json", self.dir / "histories.json"
        load = not fresh
        self.results = json.loads(self.results_path.read_text()) if load and self.results_path.exists() else []
        self.histories = json.loads(self.hist_path.read_text()) if load and self.hist_path.exists() else {}
        self.outputs = {}
        self.done = {r["model"] for r in self.results}

    def save_progress(self):
        self.results_path.write_text(json.dumps(self.results, indent=2))
        self.hist_path.write_text(json.dumps(self.histories, indent=2))

    def record(self, name, row, hist, out, model):
        self.results.append(row)
        self.done.add(name)
        self.histories[name] = hist
        self.outputs[name] = out
        self.save_progress()
        torch.save(model.state_dict(), self.dir / f"model_{name}.pt")
        np.savez(self.dir / f"outputs_{name}.npz", y_true=out[0], y_pred=out[1], probs=out[2])

    def load_outputs(self):
        """Nạp lại outputs từ file cho các model đã train ở lần chạy trước"""
        for r in self.results:
            name = r["model"]
            if name not in self.outputs:
                d = np.load(self.dir / f"outputs_{name}.npz")
                self.outputs[name] = (d["y_true"], d["y_pred"], d["probs"])


def train_all(store: ResultStore, datasets, eval_loaders, device):
    for tokenizer_name in TOKENIZERS:
        scratch_name, torch_name = f"{tokenizer_name}_scratch", f"{tokenizer_name}_torch"
        init_path = store.dir / f"init_state_{scratch_name}.pt"

        # (1) Chạy toàn bộ tokenizer với cùng attention 'scratch' trước
        if scratch_name in store.done:
            print("Đã chạy ", scratch_name)
        else:
            print(f"\nModel: {scratch_name}")
            row, hist, out, model, init_st = run_experiment(tokenizer_name, datasets, eval_loaders, device,
                                                            attention="scratch")
            torch.save(init_st, init_path)
            store.record(scratch_name, row, hist, out, model)

        # (2) Chạy với attention 'torch', khởi tạo từ đúng trọng số ban đầu của bản scratch
        if torch_name in store.done:
            print("Đã chạy ", torch_name)
        else:
            print(f"\nModel: {torch_name}")
            init_st = torch.load(init_path)
            row, hist, out, model, _ = run_experiment(tokenizer_name, datasets, eval_loaders, device,
                                                      attention="torch",
                                                      init_state=scratch_to_torch_state_dict(init_st))
            store.record(torch_name, row, hist, out, model)

        # Dùng format_results_table ở E1
        print(format_results_table(store.results))


def get_model(tokenizer_name, results_dir: Path, datasets, eval_loaders, device, attention="scratch"):
    name = f"{tokenizer_name}_{attention}"
    ckpt = results_dir / f"model_{name}.pt"
    model = build_model(tokenizer_name, attention)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        print(f"[{name}] chưa có checkpoint -> train lại (số liệu có thể lệch nhẹ so với results.json)")
        _, _, _, model, _ = run_experiment(tokenizer_name, datasets, eval_loaders, device, attention)
        torch.save(model.state_dict(), ckpt)
    return model.to(device).eval()
