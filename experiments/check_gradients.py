"""
Gradient check: verifies PennyLane adjoint diff produces real gradients
through the QCNN circuit before running any RQ3 experiments.

Run with:
    python experiments/check_gradients.py

Expected: non-zero grad norm, no None grads, no nan values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from models.qcnn import QCNNModel
from utils.dataset import get_loaders


def check(ansatz: str, encoding: str = "angle", n_qubits: int = 4, n_layers: int = 1):
    print(f"\n{'='*55}")
    print(f"  ansatz={ansatz}  encoding={encoding}  qubits={n_qubits}  layers={n_layers}")
    print(f"{'='*55}")

    train_loader, _ = get_loaders("mnist", classes=[0, 1], n_train=32, batch_size=32)
    x, y = next(iter(train_loader))

    model = QCNNModel(n_qubits=n_qubits, n_layers=n_layers,
                      ansatz=ansatz, encoding=encoding, n_classes=2)
    print(f"  Circuit params : {model.n_circuit_params}")
    print(f"  Total params   : {model.n_params}")

    model.zero_grad()
    out  = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()

    grad = model.q_params.grad

    print(f"\n  Loss value     : {loss.item():.4f}")
    print(f"  Grad is None   : {grad is None}")

    if grad is None:
        print("  [FAIL] Gradient did not reach q_params.")
        print("         Check interface='torch' in build_qcnn_circuit().")
        return

    has_nan = grad.isnan().any().item()
    n_zero  = (grad == 0).sum().item()

    print(f"  Grad shape     : {list(grad.shape)}")
    print(f"  Grad norm      : {grad.norm().item():.6f}")
    print(f"  Grad max|val|  : {grad.abs().max().item():.6f}")
    print(f"  NaN entries    : {grad.isnan().sum().item()} / {grad.numel()}")
    print(f"  Zero entries   : {n_zero} / {grad.numel()}")
    print(f"\n  Per-param grads:\n  {grad.detach().numpy().round(4)}")

    if has_nan:
        print("\n  [FAIL] NaN in gradients — numerical blow-up.")
    elif grad.norm().item() < 1e-9:
        print("\n  [FAIL] Gradient norm is effectively zero — diff broken or plateau at init.")
    elif n_zero == grad.numel():
        print("\n  [FAIL] All gradients are exactly zero.")
    else:
        print("\n  [OK] Gradients look healthy.")


if __name__ == "__main__":
    # Check 1: HEA — should always have healthy gradients at depth 1
    check(ansatz="hardware_efficient", encoding="angle", n_qubits=4, n_layers=1)

    # Check 2: QCNN-native — the ansatz used in RQ1 and RQ2
    check(ansatz="qcnn_native", encoding="angle", n_qubits=4, n_layers=1)

    # Check 3: amplitude encoding — different preprocessing path
    check(ansatz="qcnn_native", encoding="amplitude", n_qubits=4, n_layers=1)

    print(f"\n{'='*55}")
    print("  Done. Fix any [FAIL] before running RQ3.")
    print(f"{'='*55}\n")
