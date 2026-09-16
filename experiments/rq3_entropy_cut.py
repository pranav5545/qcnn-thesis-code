"""
RQ3 post-hoc: recompute entanglement entropy on an INTERLEAVED cut.

Why: the half-cut (first n/2 qubits) used in rq3_barren_plateau.py is
structurally blind at n=4 and n=8 — with adjacent-pair conv/pooling no gate
ever crosses the middle bipartition, so S=0 for product inputs regardless of
parameters. The even/odd interleaved cut is crossed by every conv pair, giving
a 2-design diagnostic comparable across all qubit counts. Report BOTH cuts:
half-cut shows the causal-cone locality story, interleaved shows expressivity.

Reuses the exact circuits and parameter samples from the main run: each cell's
stored seed re-seeds default_rng, and samples are drawn in the same order, so
states here are identical to the ones behind the stored gradients.

Run (cwd must be code/, after rq3_barren_plateau.py has produced a checkpoint):
    python experiments/rq3_entropy_cut.py
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# rq3_barren_plateau parses args at import time — hide our argv from it
_argv, sys.argv = sys.argv, [sys.argv[0]]
from experiments.rq3_barren_plateau import build_qnodes
sys.argv = _argv

from config import RESULTS_DIR
from utils.metrics import save_results, load_results


def entropy_bits(state: np.ndarray, n_qubits: int, wires_a: list[int]) -> float:
    """Von Neumann entropy (bits) of subsystem wires_a (wire 0 = first axis)."""
    psi  = np.asarray(state).reshape((2,) * n_qubits)
    rest = [w for w in range(n_qubits) if w not in wires_a]
    psi  = psi.transpose(list(wires_a) + rest).reshape(2 ** len(wires_a), -1)
    rho  = psi @ psi.conj().T
    lam  = np.linalg.eigvalsh(rho).real
    lam  = lam[lam > 1e-12]
    return float(-(lam * np.log2(lam)).sum())


def load_cells() -> tuple[dict, int]:
    out_dir = Path(RESULTS_DIR) / "rq3"
    src = out_dir / "gradient_variance.json"
    if not src.exists():
        src = out_dir / "checkpoint.json"
    saved = load_results(src)
    return saved["cells"], saved["n_inits"]


if __name__ == "__main__":
    t0 = time.perf_counter()
    out_dir  = Path(RESULTS_DIR) / "rq3"
    out_path = out_dir / "entropy_interleaved.json"
    cells, n_inits = load_cells()

    results = load_results(out_path)["cells"] if out_path.exists() else {}

    print(f"Interleaved-cut entropy for {len(cells)} cells x {n_inits} inits\n")
    for ci, (key, cell) in enumerate(cells.items()):
        if key in results:
            print(f"[{ci+1:2d}/{len(cells)}] {key:44s} [skipped, done]")
            continue

        n_qubits   = int(key.split("_")[0][1:])            # "q6_..." -> 6
        ansatz     = "qcnn_native" if "qcnn_native" in key else "hardware_efficient"
        init       = "uniform" if "_uniform_" in key else "gauss0.1"
        input_kind = key.rsplit("_", 1)[-1]                # zero | mnist

        _, state_qnode, n_params = build_qnodes(n_qubits, ansatz, input_kind)
        assert n_params == cell["n_params"], f"param-count mismatch for {key}"

        rng     = np.random.default_rng(cell["seed"])      # same stream as main run
        wires_a = list(range(0, n_qubits, 2))              # even qubits
        s_max   = min(len(wires_a), n_qubits - len(wires_a))

        vals = []
        for _ in range(n_inits):
            if init == "uniform":
                sample = rng.uniform(-np.pi, np.pi, n_params)
            else:
                sample = rng.normal(0.0, 0.1, n_params)
            vals.append(entropy_bits(state_qnode(sample), n_qubits, wires_a))

        results[key] = {
            "cut":                 "interleaved_even",
            "s_max_bits":          s_max,
            "entropy_ratio_mean":  float(np.mean(vals) / s_max),
            "entropy_ratio_std":   float(np.std(vals) / s_max),
            "entropies_bits":      vals,
        }
        print(f"[{ci+1:2d}/{len(cells)}] {key:44s} "
              f"S/Smax={results[key]['entropy_ratio_mean']:.3f} "
              f"(half-cut was {cell['entropy_ratio_mean']:.3f})")
        save_results({"experiment": "rq3_entropy_interleaved",
                      "n_inits": n_inits, "cells": results}, out_path)

    print(f"\nTotal: {(time.perf_counter()-t0)/60:.1f} min")
