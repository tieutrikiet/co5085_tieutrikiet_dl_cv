from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .training import run_epoch
from .utils import count_parameters


@torch.no_grad()
def collect_outputs(model, dataloader: DataLoader, device="cpu"):
    """Return y_true, y_pred, probs in loader order (the loader must not shuffle)."""
    model.eval()
    all_outputs, all_labels = [], []
    for x, y in dataloader:
        logits = model(x.to(device))
        all_outputs.append(torch.softmax(logits, dim=1).cpu())
        all_labels.append(y)
    probs = torch.cat(all_outputs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return labels, probs.argmax(axis=1), probs


def evaluate_model(model, name: str, best_epoch: int, train_time: float,
                   eval_loaders: Dict[str, DataLoader], device="cpu") -> dict:
    """Evaluate on every split in eval_loaders and return one result row."""
    criterion = torch.nn.CrossEntropyLoss()
    row = {
        "model": name,
        "params": count_parameters(model),
        "best_epoch": best_epoch,
        "train_time": train_time,
    }
    for split, loader in eval_loaders.items():
        loss, acc = run_epoch(model, loader, criterion, None, device)
        row[f"{split}_loss"] = loss
        row[f"{split}_acc"] = acc
    return row


def format_results_table(results: List[dict]) -> str:
    header = f"{'Model':<9}{'Params':>10}{'Best ep':>9}{'Time(s)':>9}{'Train':>9}{'Val':>9}{'Test':>9}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(f"{r['model']:<9}{r['params']:>10,}{r['best_epoch']:>9}{r['train_time']:>9.1f}"
                     f"{r['train_acc']:>9.2%}{r['val_acc']:>9.2%}{r['test_acc']:>9.2%}")
    return "\n".join(lines)
