"""
Shared training loop and evaluation used by all experiments.
Returns structured dicts so results are easy to serialise.
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(model, loader: DataLoader, optimizer, criterion, device) -> dict:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    t0 = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        n          += len(y)
    return {
        "loss": total_loss / n,
        "acc":  correct / n,
        "time": time.perf_counter() - t0,
    }


def evaluate(model, loader: DataLoader, criterion, device) -> dict:
    # Note: @torch.no_grad() intentionally omitted — PennyLane's lightning.qubit
    # adjoint diff re-resolves its execution config under no_grad on Python 3.14
    # and crashes. Correctness is unaffected; we simply don't call .backward() here.
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x).detach()
        total_loss += criterion(out, y).item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        n          += len(y)
    return {"loss": total_loss / n, "acc": correct / n}


def save_state(model, path, meta: dict | None = None) -> None:
    """
    Persist a trained model's weights.

    Added 2026-08-24 for RQ4. Until then NO model weights were saved anywhere in this
    project — RQ1/RQ2/RQ3 kept only per-epoch metric histories, so every trained model
    was discarded the moment its run ended. That is why RQ4 has to retrain its own
    teacher from scratch rather than loading RQ1's. Anything trained from here on should
    write its weights, so a later question can be asked of a model instead of a number.

    `meta` is stored alongside the tensors (architecture, seed, dataset, accuracy) so the
    file is self-describing and can be reloaded without guessing how it was built.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_class": type(model).__name__,
        "meta": meta or {},
    }
    torch.save(payload, path)


def run_training(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device,
    seed: int | None = None,
    verbose: bool = True,
    save_state_path: str | Path | None = None,
    save_state_meta: dict | None = None,
) -> dict:
    """
    Full training run. Returns history dict with per-epoch train/test metrics.
    Pass seed for reproducibility across multi-seed experiments (RQ1).

    save_state_path (added 2026-08-24, RQ4): if given, the final weights are written
    there via save_state(). Defaults to None, so every existing RQ1/RQ2/RQ3 call site
    behaves exactly as before.
    """
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "epoch_time": []}

    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        te = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["test_loss"].append(te["loss"])
        history["test_acc"].append(te["acc"])
        history["epoch_time"].append(tr["time"])

        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={tr['loss']:.4f}  train_acc={tr['acc']:.3f}  "
                f"test_acc={te['acc']:.3f}  ({tr['time']:.1f}s)"
            )

    if save_state_path is not None:
        meta = {"epochs": epochs, "lr": lr, "seed": seed,
                "final_test_acc": history["test_acc"][-1],
                "final_train_acc": history["train_acc"][-1],
                **(save_state_meta or {})}
        save_state(model, save_state_path, meta)

    return history
