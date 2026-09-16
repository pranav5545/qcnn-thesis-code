"""
Smoke test: verifies the full pipeline end-to-end on a tiny subset.
Run this first before any real experiment to catch import / shape errors.

Usage:
    python experiments/smoke_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.dataset  import get_loaders, describe
from models.qcnn    import QCNNModel
from models.classical_cnn import ClassicalCNN, make_matched_cnn
from utils.training import run_training

DEVICE   = "cpu"
N_TRAIN  = 64
N_TEST   = 32
EPOCHS   = 2
N_QUBITS = 6    # 2^6=64 = 8x8 image; canonical for amplitude encoding


def test_encoding(encoding: str):
    print(f"\n{'-'*60}")
    print(f"  Encoding: {encoding}")
    print(f"{'-'*60}")

    train_loader, test_loader = get_loaders(
        name="mnist", classes=[0, 1],
        n_train=N_TRAIN, n_test=N_TEST, batch_size=16
    )
    print("  Data:", describe(train_loader))

    qcnn = QCNNModel(n_qubits=N_QUBITS, n_layers=2, ansatz="qcnn_native",
                     encoding=encoding, n_classes=2)
    print(f"  QCNN params: {qcnn.n_params}  (circuit: {qcnn.n_circuit_params})")

    cnn = make_matched_cnn(n_qcnn_params=qcnn.n_params, image_size=8, n_classes=2)
    print(f"  CNN  params: {cnn.n_params}  (delta: {abs(cnn.n_params - qcnn.n_params)})")

    print("  Training QCNN...")
    run_training(qcnn, train_loader, test_loader,
                 epochs=EPOCHS, lr=1e-2, device=DEVICE, seed=42, verbose=True)

    print("  Training CNN...")
    run_training(cnn, train_loader, test_loader,
                 epochs=EPOCHS, lr=1e-3, device=DEVICE, seed=42, verbose=True)

    print(f"  [OK] {encoding} encoding passed")


if __name__ == "__main__":
    for enc in ["angle", "amplitude"]:   # skip IQP in smoke test (slow on CPU)
        test_encoding(enc)
    print("\n[OK] All smoke tests passed.")
