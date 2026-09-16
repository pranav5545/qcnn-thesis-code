"""FLAG 2: does the IQP path receive the same 6 PCA components the angle path receives?

Checks, per dataset:
  A. PCA loader feature dim and range.
  B. The tensor QCNNModel._prepare_input hands the circuit, angle vs iqp — must be identical.
  C. Sample ordering / labels identical between the two model instances.
  D. Diagnostic: distribution of the IQP second-order ZZ angles (pi-x_i)(pi-x_j),
     to see whether they exceed the 4*pi period of IsingZZ.
"""
import sys
import os
from pathlib import Path

# Run from anywhere: resolve the repository root from this file.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

from config import N_QUBITS, N_CLASSES, BATCH_SIZE
from models.qcnn import QCNNModel
from utils.dataset import get_loaders, get_pca_loaders, describe

CLASSES = [0, 1]
N_TRAIN, N_TEST = 500, 200

for ds in ["mnist", "fashion_mnist"]:
    print("=" * 70)
    print(f"  {ds.upper()}")
    print("=" * 70)

    raw_tr, raw_te = get_loaders(name=ds, classes=CLASSES, batch_size=BATCH_SIZE,
                                 n_train=N_TRAIN, n_test=N_TEST)
    pca_tr, pca_te = get_pca_loaders(name=ds, classes=CLASSES, batch_size=BATCH_SIZE,
                                     n_train=N_TRAIN, n_test=N_TEST, n_components=N_QUBITS)
    print(f"  raw train : {describe(raw_tr)}")
    print(f"  pca train : {describe(pca_tr)}")

    # Full PCA train tensor (unshuffled view straight off the dataset)
    Xp = pca_tr.dataset.tensors[0]
    yp = pca_tr.dataset.tensors[1]
    print(f"\n  A. PCA features: shape={tuple(Xp.shape)}  "
          f"min={Xp.min():.4f}  max={Xp.max():.4f}")
    print(f"     per-component min/max (train, should each span ~[-1,1]):")
    for c in range(Xp.shape[1]):
        print(f"       PC{c+1}: [{Xp[:, c].min():+.4f}, {Xp[:, c].max():+.4f}]  "
              f"std={Xp[:, c].std():.4f}")
    n_const = int((Xp.std(dim=0) < 1e-6).sum())
    print(f"     constant/degenerate components: {n_const}")

    # B. what each model actually feeds the circuit
    torch.manual_seed(42)
    m_angle = QCNNModel(n_qubits=N_QUBITS, n_layers=2, ansatz="qcnn_native",
                        encoding="angle", n_classes=N_CLASSES)
    torch.manual_seed(42)
    m_iqp = QCNNModel(n_qubits=N_QUBITS, n_layers=2, ansatz="qcnn_native",
                      encoding="iqp", n_classes=N_CLASSES)

    a_in = m_angle._prepare_input(Xp.clone())
    i_in = m_iqp._prepare_input(Xp.clone())
    identical = torch.equal(a_in, i_in)
    print(f"\n  B. _prepare_input(angle) vs _prepare_input(iqp): "
          f"identical={identical}  maxabsdiff={(a_in - i_in).abs().max():.3e}")
    print(f"     shape fed to circuit: {tuple(i_in.shape)}  "
          f"(n_qubits={N_QUBITS}) -> all components used: {i_in.shape[1] == N_QUBITS}")
    print(f"     no truncation vs source: {i_in.shape[1] == Xp.shape[1]}")
    print(f"     range after *pi: [{i_in.min():+.4f}, {i_in.max():+.4f}]  "
          f"(pi={np.pi:.4f})")
    # column-wise: is any component dropped or duplicated?
    dup = [(a, b) for a in range(i_in.shape[1]) for b in range(a + 1, i_in.shape[1])
           if torch.equal(i_in[:, a], i_in[:, b])]
    print(f"     duplicated columns (feature reuse): {dup if dup else 'none'}")

    # C. labels/order
    print(f"\n  C. label vector identical for both paths: True (same tensor) ; "
          f"class balance={torch.bincount(yp).tolist()}")

    # D. IQP second-order angle magnitudes
    x = i_in.numpy()
    zz = []
    for a in range(N_QUBITS):
        for b in range(a + 1, N_QUBITS):
            zz.append((np.pi - x[:, a]) * (np.pi - x[:, b]))
    zz = np.concatenate(zz)
    print(f"\n  D. IQP ZZ angles (pi-x_i)(pi-x_j): "
          f"min={zz.min():.3f} max={zz.max():.3f} mean={zz.mean():.3f}")
    print(f"     IsingZZ period = 4*pi = {4*np.pi:.3f}; "
          f"fraction of angles exceeding one period: {(np.abs(zz) > 4*np.pi).mean():.3f}")
    print(f"     max wraps around the period: {zz.max()/(4*np.pi):.2f}x")
    print()
