"""FLAG 2 follow-up: is IQP's weakness expressivity, or an artefact of feature scaling?

The shared angle/IQP pipeline scales PCA components to [-pi, pi]. That is right for
angle (RY spans the full Bloch rotation) but pushes the Havlicek second-order term
(pi - x_i)(pi - x_j) past IsingZZ's 4*pi period for 49.7% of feature pairs on
Fashion-MNIST, aliasing the interaction into a non-monotonic function of the features.

Probe: rerun the IQP core-grid cell (iqp / qcnn_native / depth 2) on Fashion-MNIST with
features mapped to [0, pi] instead, so (pi-x_i)(pi-x_j) in [0, pi^2=9.87] < 4*pi=12.57
and never wraps. Same seeds, same epochs, same data -> paired comparison against the
sweep's existing numbers.

Writes JSON so the result survives the session.
"""
import sys, os, json, time
from pathlib import Path

# Run from anywhere: resolve the repository root from this file.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

from config import N_QUBITS, N_CLASSES, BATCH_SIZE, EPOCHS, LR
from models.qcnn import QCNNModel
from utils.dataset import get_pca_loaders
from utils.training import run_training

OUT = ROOT / "results" / "rq2" / "probes" / "flag2_rescale_probe.json"
DATASET = "fashion_mnist"
SEEDS   = [42, 43]
EPOCHS_RUN = EPOCHS   # 50, matched to the sweep


class IQPRescaled(QCNNModel):
    """Identical to QCNNModel but maps [-1,1] PCA features to [0, pi], not [-pi, pi]."""
    def _prepare_input(self, x):
        return (x + 1.0) / 2.0 * np.pi


def main():
    tr, te = get_pca_loaders(name=DATASET, classes=[0, 1], batch_size=BATCH_SIZE,
                             n_train=500, n_test=200, n_components=N_QUBITS)
    results = {"dataset": DATASET, "epochs": EPOCHS_RUN, "seeds": SEEDS,
               "config": "iqp__qcnn_native__L2", "variant": "rescaled_0_pi",
               "histories": {}}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = IQPRescaled(n_qubits=N_QUBITS, n_layers=2, ansatz="qcnn_native",
                            encoding="iqp", n_classes=N_CLASSES)
        t0 = time.perf_counter()
        h = run_training(model, tr, te, epochs=EPOCHS_RUN, lr=LR, device="cpu",
                         seed=seed, verbose=False)
        dt = (time.perf_counter() - t0) / 60
        h["n_params"] = model.n_params
        results["histories"][str(seed)] = h
        print(f"seed {seed}: test_acc={h['test_acc'][-1]:.4f}  ({dt:.1f} min)", flush=True)
        OUT.write_text(json.dumps(results, indent=2))
    accs = [results["histories"][str(s)]["test_acc"][-1] for s in SEEDS]
    print(f"\nRESCALED [0,pi] mean over {len(accs)} seeds: {np.mean(accs):.4f}", flush=True)
    print("Compare against the sweep's iqp__qcnn_native__L2 on the SAME seeds.", flush=True)


if __name__ == "__main__":
    main()
