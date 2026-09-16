"""
RQ3: Barren plateau analysis — gradient variance and entanglement entropy
vs qubit count, following the canonical protocols in the literature.

Design (see writing/lit_review/rq3_bp_methodology_notes.md for sources):
  - Var[dC/dtheta_k] over BP_N_INITS random parameter draws, fixed input,
    fixed single parameter k = first param of the CENTER conv unitary in
    layer 1 (Pesah et al. 2021 convention). Full gradients are stored too,
    so mean per-parameter variance can be reported as well.
  - Cost: local <Z x Z> on the 2 surviving qubits after pooling to 2
    (Pesah's observable; our PauliZ readout is already local).
  - Depth: pool until exactly 2 active qubits (QCNN log-depth by design):
    n=4 -> 1 layer, n=6 -> 2, n=8 -> 2, n=10 -> 3.
  - Init axis: uniform [-pi, pi] (BP-inducing 2-design baseline) vs
    N(0, 0.1^2) (the pipeline's small-angle mitigation init).
  - Input axis: |0...0> (literature standard, tests Pesah's claim directly)
    vs a FIXED amplitude-encoded MNIST '0' (tests entanglement-induced BP;
    variance is over params only — data-driven BP studies keep input fixed).
    Feature prep per n: resize 4x4 / 8x8 / 16x16 for n=4/6/8; native 28x28
    zero-padded 784 -> 1024 for n=10 (literature precedents; never silent
    truncation).
  - Diagnostic: bipartite von Neumann entropy of the first n/2 qubits of the
    final state, S/S_max with S_max = n/2 bits; ratio -> 1 = 2-design regime.

Checkpoint note: results/rq3/checkpoint.json stores every finished cell
(qubits x ansatz x init x input); rerunning skips completed cells.

Run (cwd must be code/):
    python experiments/rq3_barren_plateau.py [--quick]
"""
import sys, argparse, time, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pennylane as qml
from torchvision import datasets
import torch.nn.functional as F

from config import SEED, BP_QUBIT_COUNTS, BP_N_INITS, DATA_DIR, RESULTS_DIR
from models.qcnn import (
    _conv_layer_qcnn, _conv_layer_hea, _pooling_layer,
    PARAMS_PER_CONV, PARAMS_PER_POOL,
)
from utils.metrics import save_results, load_results

ANSATZE = ["qcnn_native", "hardware_efficient"]
INITS   = ["uniform", "gauss0.1"]
INPUTS  = ["zero", "mnist"]

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="tiny grid (4,6 qubits, 10 inits), no checkpoint")
parser.add_argument("--inits", type=int, default=None)
args = parser.parse_args()

QUBIT_COUNTS = [4, 6] if args.quick else BP_QUBIT_COUNTS
N_INITS      = args.inits or (10 if args.quick else BP_N_INITS)


# ── Circuit structure helpers ─────────────────────────────────────────────────

def layers_to_two(n_qubits: int) -> int:
    """Number of conv+pool layers until exactly 2 active qubits remain."""
    active, layers = n_qubits, 0
    while active > 2:
        active = active // 2 + active % 2
        layers += 1
    return layers


def count_params(n_qubits: int, n_layers: int, ansatz: str) -> int:
    ppc, total, active = PARAMS_PER_CONV[ansatz], 0, n_qubits
    for _ in range(n_layers):
        if active < 2:
            break
        total  += (active // 2) * (ppc + PARAMS_PER_POOL)
        active  = active // 2 + active % 2
    return total


def center_param_index(n_qubits: int, ansatz: str) -> int:
    """First parameter of the center conv unitary in layer 1 (Pesah convention)."""
    n_pairs = n_qubits // 2
    return (n_pairs // 2) * PARAMS_PER_CONV[ansatz]


# ── Fixed MNIST input, prepared per qubit count ───────────────────────────────

def mnist_features(n_qubits: int) -> np.ndarray:
    """Fixed digit-0 image as a 2^n feature vector (resize; zero-pad at n=10)."""
    ds = datasets.MNIST(root=DATA_DIR, train=True, download=False)
    idx = int((ds.targets == 0).nonzero()[0])          # first '0' — deterministic
    img = ds.data[idx].float().unsqueeze(0).unsqueeze(0) / 255.0   # 1x1x28x28
    dim = 2 ** n_qubits
    if n_qubits == 10:
        vec = img.flatten()                            # 784, pad below
    else:
        side = int(math.isqrt(dim))                    # 4 / 8 / 16
        vec  = F.interpolate(img, size=(side, side),
                             mode="bilinear", align_corners=False).flatten()
    out = np.zeros(dim)
    out[: vec.numel()] = vec.numpy()
    return out / np.linalg.norm(out)


# ── QNodes ────────────────────────────────────────────────────────────────────

def build_qnodes(n_qubits: int, ansatz: str, input_kind: str):
    """Returns (grad_qnode(params)->cost, state_qnode(params)->state, n_params)."""
    dev      = qml.device("lightning.qubit", wires=n_qubits)
    conv_fn  = _conv_layer_qcnn if ansatz == "qcnn_native" else _conv_layer_hea
    ppc      = PARAMS_PER_CONV[ansatz]
    n_layers = layers_to_two(n_qubits)
    n_params = count_params(n_qubits, n_layers, ansatz)
    feats    = mnist_features(n_qubits) if input_kind == "mnist" else None

    def body(params):
        if feats is not None:
            qml.AmplitudeEmbedding(feats, wires=range(n_qubits), normalize=True)
        active, p = list(range(n_qubits)), 0
        for _ in range(n_layers):
            pairs = [(active[i], active[i + 1]) for i in range(0, len(active) - 1, 2)]
            for pair in pairs:
                conv_fn(params[p: p + ppc], wires=pair)
                p += ppc
            new_active = []
            for ctrl, tgt in pairs:
                _pooling_layer(params[p: p + PARAMS_PER_POOL], wires=[ctrl, tgt])
                p += PARAMS_PER_POOL
                new_active.append(tgt)
            if len(active) % 2 == 1:
                new_active.append(active[-1])
            active = new_active
        return active

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def grad_qnode(params):
        active = body(params)
        return qml.expval(qml.PauliZ(active[0]) @ qml.PauliZ(active[1]))

    @qml.qnode(dev, diff_method=None)
    def state_qnode(params):
        body(params)
        return qml.state()

    return grad_qnode, state_qnode, n_params


def half_cut_entropy_bits(state: np.ndarray, n_qubits: int) -> float:
    """Von Neumann entropy (bits) of the first floor(n/2) qubits."""
    n_a  = n_qubits // 2
    psi  = np.asarray(state).reshape(2 ** n_a, -1)
    rho  = psi @ psi.conj().T
    lam  = np.linalg.eigvalsh(rho).real
    lam  = lam[lam > 1e-12]
    return float(-(lam * np.log2(lam)).sum())


# ── Runner ────────────────────────────────────────────────────────────────────

def run_cell(n_qubits, ansatz, init, input_kind, seed):
    grad_qnode, state_qnode, n_params = build_qnodes(n_qubits, ansatz, input_kind)
    k   = center_param_index(n_qubits, ansatz)
    rng = np.random.default_rng(seed)

    grads, entropies = [], []
    for _ in range(N_INITS):
        if init == "uniform":
            sample = rng.uniform(-np.pi, np.pi, n_params)
        else:
            sample = rng.normal(0.0, 0.1, n_params)

        params = torch.tensor(sample, requires_grad=True)
        cost   = grad_qnode(params)
        cost.backward()
        grads.append(params.grad.numpy().copy())

        entropies.append(half_cut_entropy_bits(state_qnode(sample), n_qubits))

    grads  = np.stack(grads)                       # (N_INITS, n_params)
    s_max  = n_qubits // 2
    return {
        "n_params":         n_params,
        "n_layers":         layers_to_two(n_qubits),
        "target_param":     k,
        "seed":             seed,
        "var_target":       float(grads[:, k].var()),
        "var_per_param_mean": float(grads.var(axis=0).mean()),
        "entropy_ratio_mean": float(np.mean(entropies) / s_max),
        "entropy_ratio_std":  float(np.std(entropies) / s_max),
        "grads":            grads.tolist(),
        "entropies_bits":   entropies,
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    out_dir = Path(RESULTS_DIR) / "rq3"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.json"

    cells = {}
    if not args.quick and ckpt_path.exists():
        saved = load_results(ckpt_path)
        if saved.get("n_inits") == N_INITS:
            cells = saved["cells"]
            print(f"Resumed checkpoint: {len(cells)} cells done")

    grid = [(n, a, i, x) for n in QUBIT_COUNTS for a in ANSATZE
            for i in INITS for x in INPUTS]
    print(f"RQ3 grid: {len(grid)} cells x {N_INITS} inits "
          f"(qubits={QUBIT_COUNTS})\n")

    for ci, (n, ansatz, init, input_kind) in enumerate(grid):
        key = f"q{n}_{ansatz}_{init}_{input_kind}"
        if key in cells:
            print(f"[{ci+1:2d}/{len(grid)}] {key:44s} [skipped, in checkpoint]")
            continue
        t1 = time.perf_counter()
        cells[key] = run_cell(n, ansatz, init, input_kind, seed=SEED + ci)
        dt = time.perf_counter() - t1
        c  = cells[key]
        print(f"[{ci+1:2d}/{len(grid)}] {key:44s} "
              f"Var[dC/dth_{c['target_param']}]={c['var_target']:.3e}  "
              f"S/Smax={c['entropy_ratio_mean']:.3f}  ({dt:.1f}s)")
        if not args.quick:
            save_results({"experiment": "rq3", "n_inits": N_INITS,
                          "qubit_counts": QUBIT_COUNTS, "cells": cells}, ckpt_path)

    if not args.quick:
        save_results({"experiment": "rq3", "n_inits": N_INITS,
                      "qubit_counts": QUBIT_COUNTS, "cells": cells},
                     out_dir / "gradient_variance.json")

    print(f"\n{'='*72}")
    print("  RQ3 SUMMARY — Var[dC/dtheta_center] vs qubit count")
    print(f"{'='*72}")
    print(f"  {'ansatz':20s}{'init':10s}{'input':7s}" +
          "".join(f"{'q'+str(n):>12s}" for n in QUBIT_COUNTS))
    for a in ANSATZE:
        for i in INITS:
            for x in INPUTS:
                row = f"  {a:20s}{i:10s}{x:7s}"
                for n in QUBIT_COUNTS:
                    c = cells.get(f"q{n}_{a}_{i}_{x}")
                    row += f"{c['var_target']:>12.3e}" if c else f"{'—':>12s}"
                print(row)
    print(f"\nTotal: {(time.perf_counter()-t0)/60:.1f} min")
