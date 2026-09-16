"""
Knowledge-distillation utilities for RQ4.

RQ4 asks whether a parameter-matched classical student can absorb the QCNN's learned
*function* — distillation used as a simulability probe rather than as a way to make the
student more accurate. See Appendix F of the dissertation.

Two hazards are handled here rather than left to call sites, because both fail silently:

  1. TEACHER-LOGIT ALIGNMENT. The train loader shuffles. If the logits do not travel
     with their own samples, the student trains against another sample's soft targets
     and still produces a plausible-looking curve. build_kd_dataset() puts (x, y, logit)
     in one TensorDataset so shuffling permutes the triple as a unit, and
     assert_logits_aligned() re-runs the teacher over the stored X to prove it.

  2. @torch.no_grad() CRASHES THIS STACK. PennyLane lightning's adjoint differentiation
     re-resolves its execution config under no_grad and dies — see the comment at
     utils/training.py:33 and memory feedback_qcnn_gotchas. Teacher inference here uses
     .detach() everywhere and never a no_grad context.
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.training import save_state


# ── Loss ──────────────────────────────────────────────────────────────────────

def kd_loss(student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            targets: torch.Tensor,
            T: float = 4.0,
            alpha: float = 0.9) -> torch.Tensor:
    """
    Hinton et al. (2015) knowledge distillation:

        alpha * T^2 * KL(softmax(t/T) || softmax(s/T))  +  (1 - alpha) * CE(s, y)

    The T^2 factor restores the gradient magnitude that dividing the logits by T removes,
    so alpha stays interpretable as a mixing weight rather than silently rescaling the
    learning rate. alpha=1.0 is pure distillation (no hard labels at all) — RQ4 runs it
    because hard labels actively pull the student away from the teacher, and fidelity,
    not accuracy, is the primary metric here.
    """
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.log_softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
        log_target=True,
    ) * (T ** 2)
    if alpha >= 1.0:
        return alpha * soft
    hard = F.cross_entropy(student_logits, targets)
    return alpha * soft + (1.0 - alpha) * hard


# ── Teacher inference ─────────────────────────────────────────────────────────

def collect_logits(model, X: torch.Tensor, device="cpu", batch_size: int = 32,
                   verbose: bool = False) -> torch.Tensor:
    """
    Forward `model` over X in its given order and return detached logits.

    Order in == order out. That property is the whole contract: everything downstream
    (build_kd_dataset, the alignment assertion, per-corruption fidelity) assumes row i of
    the result belongs to row i of X.

    NOT wrapped in torch.no_grad() — see the module docstring. .detach() gives the same
    memory behaviour for a model this small without touching PennyLane's grad plumbing.
    """
    model.eval()
    model.to(device)
    out = []
    t0 = time.perf_counter()
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        out.append(model(xb).detach().cpu())
        if verbose and (i // batch_size) % 10 == 0:
            print(f"      logits {i + len(xb)}/{len(X)}", flush=True)
    logits = torch.cat(out, dim=0).float()
    if verbose:
        print(f"      collected {tuple(logits.shape)} in {time.perf_counter() - t0:.1f}s", flush=True)
    return logits


def assert_logits_aligned(model, X: torch.Tensor, logits: torch.Tensor,
                          device="cpu", tol: float = 1e-5, n_check: int | None = None) -> float:
    """
    Verification gate 2 from the implementation plan. Re-runs the teacher over the stored
    X and asserts the cached logits reproduce to within `tol`.

    Raises rather than warns. A misalignment here invalidates every student run that
    consumes the cache, and it is invisible in the training curves — a student fitting
    shuffled soft targets converges to something, it is simply not the teacher.

    n_check limits the re-run to the first n rows (the quantum teacher is slow); None
    checks everything. Returns the max absolute difference.
    """
    sub = X if n_check is None else X[:n_check]
    ref = logits if n_check is None else logits[:n_check]
    fresh = collect_logits(model, sub, device=device)
    max_diff = float((fresh - ref).abs().max().item())
    if max_diff > tol:
        raise AssertionError(
            f"teacher logits do not reproduce: max|diff| = {max_diff:.3e} > {tol:.1e}. "
            "The cache is misaligned with its samples — do not train students on it."
        )
    return max_diff


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_kd_dataset(X: torch.Tensor, y: torch.Tensor,
                     teacher_logits: torch.Tensor) -> TensorDataset:
    """
    (x, y, teacher_logit) as one TensorDataset, so a shuffling DataLoader permutes the
    triple together. Mirrors the materialise-to-TensorDataset pattern get_pca_loaders
    already uses in utils/dataset.py.
    """
    if not (len(X) == len(y) == len(teacher_logits)):
        raise ValueError(
            f"length mismatch: X={len(X)}, y={len(y)}, logits={len(teacher_logits)}"
        )
    return TensorDataset(X.float(), y.long(), teacher_logits.float())


# ── Training loop ─────────────────────────────────────────────────────────────

def run_kd_training(
    student,
    kd_train_dataset: TensorDataset,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device,
    T: float = 4.0,
    alpha: float = 0.9,
    batch_size: int = 32,
    seed: int | None = None,
    verbose: bool = False,
    save_state_path: str | Path | None = None,
    save_state_meta: dict | None = None,
) -> dict:
    """
    Distillation counterpart to utils/training.py::run_training.

    Returns the SAME history keys (train_loss / train_acc / test_loss / test_acc /
    epoch_time) so summarise_seeds and the analysis scripts work on RQ4 output unchanged.
    alpha=0.0 reduces to plain cross-entropy, which is how the `hard` control arm runs
    through this same function rather than a second code path — it makes the control a
    genuine control.

    The DataLoader is built here, after the seed is set, so batch order is reproducible
    per seed exactly as it is in run_training.
    """
    if seed is not None:
        torch.manual_seed(seed)

    loader = DataLoader(kd_train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    student.to(device)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "epoch_time": []}

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss, correct, n = 0.0, 0, 0
        t0 = time.perf_counter()
        for x, y, tlog in loader:
            x, y, tlog = x.to(device), y.to(device), tlog.to(device)
            optimizer.zero_grad()
            out = student(x)
            loss = kd_loss(out, tlog, y, T=T, alpha=alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            n          += len(y)
        train_time = time.perf_counter() - t0

        # Test metrics stay plain CE/accuracy against the true labels, not the KD
        # objective — otherwise the `hard` and `qkd` arms would be scored on different
        # quantities and their test curves could not be compared at all.
        student.eval()
        te_loss, te_correct, te_n = 0.0, 0, 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = student(x).detach()
            te_loss    += criterion(out, y).item() * len(y)
            te_correct += (out.argmax(1) == y).sum().item()
            te_n       += len(y)

        history["train_loss"].append(total_loss / n)
        history["train_acc"].append(correct / n)
        history["test_loss"].append(te_loss / te_n)
        history["test_acc"].append(te_correct / te_n)
        history["epoch_time"].append(train_time)

        if verbose:
            print(f"      epoch {epoch:3d}/{epochs}  loss={total_loss / n:.4f}  "
                  f"test_acc={te_correct / te_n:.3f}  ({train_time:.1f}s)", flush=True)

    if save_state_path is not None:
        meta = {"epochs": epochs, "lr": lr, "seed": seed, "T": T, "alpha": alpha,
                "final_test_acc": history["test_acc"][-1],
                "final_train_acc": history["train_acc"][-1],
                **(save_state_meta or {})}
        save_state(student, save_state_path, meta)

    return history
