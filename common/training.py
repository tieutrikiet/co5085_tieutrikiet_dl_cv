import copy
import time

import torch
from torch.utils.data import DataLoader

from .utils import set_seed


def run_epoch(model, dataloader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                   # forward
            loss = criterion(logits, y)         # compute loss
            if is_train:
                optimizer.zero_grad()
                loss.backward()                 # backward pass
                optimizer.step()                # update weights

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float,
        weight_decay: float, seed: int, patience: int, device="cpu"):
    """Adam + early stopping on val accuracy; restores the best checkpoint."""
    set_seed(seed)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    best_acc, best_state, best_epoch, bad = -1.0, None, 0, 0

    for epoch in range(epochs):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        epoch_time = time.perf_counter() - start
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        for k, v in zip(history, (train_loss, train_acc, val_loss, val_acc, epoch_time)):
            history[k].append(v)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_epoch
