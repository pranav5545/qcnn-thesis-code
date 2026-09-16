"""
Quantum Convolutional Neural Network (QCNN) implemented in PennyLane + PyTorch.

Architecture follows Cong et al. (2019):
  - Alternating convolutional (2-qubit unitary) and pooling (measure+conditional)
    layers that halve the active qubit count each round.
  - Final measurement feeds a small classical linear head.

Supports three ansatze:
  - "qcnn_native"         – QCNN-specific 2-qubit unitaries (15 params per pair)
  - "hardware_efficient"  – general SU(4) approximation (intended as BP+ control)
"""
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.encodings import ENCODINGS


# ── Ansatz building blocks ────────────────────────────────────────────────────

def _conv_layer_qcnn(params, wires):
    """QCNN-native 2-qubit conv unitary (15 free params per pair)."""
    # General SU(4) decomposition without the full hardware-efficient structure
    # Uses the Reck decomposition approach: 3 CNOT + single-qubit gates
    assert len(params) == 15
    i, j = wires
    qml.U3(params[0], params[1], params[2], wires=i)
    qml.U3(params[3], params[4], params[5], wires=j)
    qml.CNOT(wires=[i, j])
    qml.RY(params[6], wires=i)
    qml.RZ(params[7], wires=j)
    qml.CNOT(wires=[j, i])
    qml.RY(params[8], wires=i)
    qml.CNOT(wires=[i, j])
    qml.U3(params[9],  params[10], params[11], wires=i)
    qml.U3(params[12], params[13], params[14], wires=j)


def _conv_layer_hea(params, wires):
    """Hardware-efficient 2-qubit block (6 params: 3 per qubit + CNOT)."""
    assert len(params) == 6
    i, j = wires
    qml.RX(params[0], wires=i)
    qml.RY(params[1], wires=i)
    qml.RZ(params[2], wires=i)
    qml.RX(params[3], wires=j)
    qml.RY(params[4], wires=j)
    qml.RZ(params[5], wires=j)
    qml.CNOT(wires=[i, j])


def _pooling_layer(params, wires):
    """
    Controlled pooling: measure ctrl qubit and apply conditional rotation on target.
    params: [theta, phi]
    """
    ctrl, target = wires
    qml.CRZ(params[0], wires=[ctrl, target])
    qml.PauliX(wires=ctrl)
    qml.CRX(params[1], wires=[ctrl, target])


# ── QCNN circuit ──────────────────────────────────────────────────────────────

PARAMS_PER_CONV = {"qcnn_native": 15, "hardware_efficient": 6}
PARAMS_PER_POOL = 2


def _count_params(n_qubits: int, n_layers: int, ansatz: str) -> int:
    ppc = PARAMS_PER_CONV[ansatz]
    total = 0
    active = n_qubits
    for _ in range(n_layers):
        if active < 2:
            break
        n_pairs  = active // 2
        leftover = active % 2
        total   += n_pairs * ppc               # conv
        total   += n_pairs * PARAMS_PER_POOL   # pool
        active   = n_pairs + leftover          # targets + unpaired qubit carried forward
    return total


def build_qcnn_circuit(n_qubits: int, n_layers: int, ansatz: str, encoding: str,
                        q_device: str = "lightning.qubit"):
    """
    Returns a PennyLane QNode that takes (x, params) and returns expectation values.

    q_device: PennyLane device name, e.g. "lightning.qubit" (CPU, default) or
    "lightning.gpu" (cuQuantum). NOTE — the forward pass below executes the QNode
    once per sample in a Python loop (not batched, see models/qcnn.py docstring /
    memory feedback_qcnn_gotchas Rule 7); for a 6-qubit circuit the per-call kernel-
    launch overhead on GPU has not been benchmarked against CPU and may dominate.
    Smoke-test before committing a full sweep to a GPU device.
    """
    dev = qml.device(q_device, wires=n_qubits)
    conv_fn = _conv_layer_qcnn if ansatz == "qcnn_native" else _conv_layer_hea
    ppc = PARAMS_PER_CONV[ansatz]
    encode_fn = ENCODINGS[encoding]

    # For amplitude encoding the input length must be 2^n_qubits
    # For angle/iqp it must be n_qubits — handled in QCNNModel.forward

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(x, params):
        encode_fn(x, wires=list(range(n_qubits)))
        active = list(range(n_qubits))
        p_idx = 0
        for _ in range(n_layers):
            if len(active) < 2:
                break
            # Conv: apply to every adjacent pair
            pairs = [(active[i], active[i + 1]) for i in range(0, len(active) - 1, 2)]
            for pair in pairs:
                conv_fn(params[p_idx: p_idx + ppc], wires=pair)
                p_idx += ppc
            # Pool: halve the active wires
            pool_pairs = [(active[i], active[i + 1]) for i in range(0, len(active) - 1, 2)]
            new_active = []
            for ctrl, tgt in pool_pairs:
                _pooling_layer(params[p_idx: p_idx + PARAMS_PER_POOL], wires=[ctrl, tgt])
                p_idx += PARAMS_PER_POOL
                new_active.append(tgt)
            if len(active) % 2 == 1:
                new_active.append(active[-1])
            active = new_active
        return [qml.expval(qml.PauliZ(w)) for w in active]

    return circuit


# ── PyTorch Module ────────────────────────────────────────────────────────────

class QCNNModel(nn.Module):
    """
    Full QCNN: quantum circuit + small classical linear head.

    Args:
        n_qubits:  number of qubits
        n_layers:  number of conv+pool block pairs
        ansatz:    "qcnn_native" or "hardware_efficient"
        encoding:  "angle", "amplitude", or "iqp"
        n_classes: output classes
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        ansatz: str = "qcnn_native",
        encoding: str = "angle",
        n_classes: int = 2,
        q_device: str = "lightning.qubit",
    ):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.ansatz    = ansatz
        self.encoding  = encoding
        self.q_device  = q_device

        n_circ_params = _count_params(n_qubits, n_layers, ansatz)
        self.q_params  = nn.Parameter(torch.randn(n_circ_params) * 0.1)
        self.circuit   = build_qcnn_circuit(n_qubits, n_layers, ansatz, encoding, q_device)

        # Mirror the circuit's active-qubit tracking to get the correct output size
        active = n_qubits
        for _ in range(n_layers):
            if active < 2:
                break
            active = active // 2 + active % 2  # pairs → targets, plus any unpaired qubit
        n_out = max(1, active)
        self.head = nn.Linear(n_out, n_classes)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess flat input to match encoding requirements."""
        if self.encoding == "amplitude":
            # Needs 2^n_qubits features; pad or truncate
            target = 2 ** self.n_qubits
            if x.shape[1] < target:
                x = torch.nn.functional.pad(x, (0, target - x.shape[1]))
            elif x.shape[1] > target:
                x = x[:, :target]
            # Normalise to unit norm (AmplitudeEmbedding does this too, but be explicit)
            norm = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
            x = x / norm
        else:
            # Angle / IQP: need exactly n_qubits features
            x = x[:, : self.n_qubits]
            # Scale to [-π, π]
            x = x * np.pi
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        # Process each sample individually (PennyLane QNode is not batch-native)
        outputs = torch.stack([
            torch.stack(self.circuit(x[i], self.q_params))
            for i in range(x.shape[0])
        ]).float()  # default.qubit returns float64; head expects float32
        return self.head(outputs)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def n_circuit_params(self) -> int:
        return self.q_params.numel()
