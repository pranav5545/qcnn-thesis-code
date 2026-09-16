"""
LR diagnostic for RQ1: find a learning rate that lets the QCNN
break out of majority-class prediction before committing to the full run.

Both the QCNN and CNN use angle encoding, 6 qubits, 2 layers (74 vs 71 params).
Uses ONE seed and ONE device so there is no lightning.qubit tape-state issue.

Run:
    python experiments/check_lr.py

Expect: at some LR the QCNN test_acc rises above 0.55 within 30 epochs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from utils.dataset        import get_loaders, describe
from models.qcnn          import QCNNModel
from models.classical_cnn import make_matched_cnn
from utils.training       import run_training

SEED     = 42
N_QUBITS = 6
N_LAYERS = 2
EPOCHS   = 20
N_TRAIN  = 500
N_TEST   = 200
CLASSES  = [0, 1]

LRS_TO_TRY = [1e-3, 1e-2]

train_loader, test_loader = get_loaders(
    "mnist", classes=CLASSES, n_train=N_TRAIN, n_test=N_TEST
)
print(f"Data: {describe(train_loader)}\n")

# Create QCNN once — device/QNode shared across all LR trials
torch.manual_seed(SEED)
qcnn_ref = QCNNModel(n_qubits=N_QUBITS, n_layers=N_LAYERS,
                     ansatz="qcnn_native", encoding="amplitude", n_classes=2)
cnn_ref  = make_matched_cnn(qcnn_ref.n_params, image_size=8, n_classes=2)
print(f"QCNN params: {qcnn_ref.n_params}   CNN params: {cnn_ref.n_params}\n")
print(f"{'='*65}")

results = {}

for lr in LRS_TO_TRY:
    print(f"\n-- QCNN  LR={lr} --")

    # Re-init QCNN parameters; reuse the same circuit/device
    import math
    torch.manual_seed(SEED)
    with torch.no_grad():
        qcnn_ref.q_params.data.normal_(0, 0.1)
        nn.init.kaiming_uniform_(qcnn_ref.head.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(qcnn_ref.head.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(qcnn_ref.head.bias, -bound, bound)

    h = run_training(qcnn_ref, train_loader, test_loader,
                     epochs=EPOCHS, lr=lr, device="cpu", seed=SEED, verbose=True)

    final_acc  = h["test_acc"][-1]
    best_acc   = max(h["test_acc"])
    best_epoch = h["test_acc"].index(best_acc) + 1
    results[f"qcnn_lr{lr}"] = {"final": final_acc, "best": best_acc, "best_epoch": best_epoch}
    print(f"  -> final test_acc={final_acc:.4f}  best={best_acc:.4f} (epoch {best_epoch})")

print(f"\n-- CNN  LR=1e-3 (baseline) --")
torch.manual_seed(SEED)
cnn_fresh = make_matched_cnn(qcnn_ref.n_params, image_size=8, n_classes=2)
h_cnn = run_training(cnn_fresh, train_loader, test_loader,
                     epochs=EPOCHS, lr=1e-3, device="cpu", seed=SEED, verbose=True)
final_cnn = h_cnn["test_acc"][-1]
best_cnn  = max(h_cnn["test_acc"])
print(f"  -> final test_acc={final_cnn:.4f}  best={best_cnn:.4f}")

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"  {'Config':<25} {'Final acc':>10} {'Best acc':>10} {'Best epoch':>12}")
print(f"  {'-'*60}")
for cfg, r in results.items():
    flag = " <-- learns!" if r["best"] > 0.55 else ""
    print(f"  {cfg:<25} {r['final']:>10.4f} {r['best']:>10.4f} {r['best_epoch']:>12}{flag}")
print(f"  {'CNN lr=1e-3':<25} {final_cnn:>10.4f} {best_cnn:>10.4f}")
print(f"{'='*65}\n")
