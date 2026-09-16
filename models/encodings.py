"""
Quantum data encoding strategies.

Each function takes a flat data vector x (length n_qubits or 2^n_qubits)
and applies gates to the wires of the active PennyLane device.
"""
import pennylane as qml
import numpy as np


def angle_encoding(x, wires):
    """
    Angle encoding: one feature per qubit via RY rotations.
    Requires len(x) == len(wires). Features should be in [-π, π].
    """
    for i, w in enumerate(wires):
        qml.RY(x[i], wires=w)


def amplitude_encoding(x, wires):
    """
    Amplitude encoding: encodes 2^n features into the amplitude vector.
    Requires len(x) == 2^len(wires). Input is L2-normalised internally.
    """
    qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)


def iqp_encoding(x, wires):
    """
    IQP (Instantaneous Quantum Polynomial) encoding.
    Layer of Hadamards, then ZZ interactions parameterised by feature products.
    Follows the structure from Havlíček et al. (2019).
    """
    n = len(wires)
    # First-order: RZ(x_i) after H
    for i, w in enumerate(wires):
        qml.Hadamard(wires=w)
        qml.RZ(x[i], wires=w)
    # Second-order: ZZ interactions
    for i in range(n):
        for j in range(i + 1, n):
            qml.IsingZZ((np.pi - x[i]) * (np.pi - x[j]), wires=[wires[i], wires[j]])


ENCODINGS = {
    "angle":     angle_encoding,
    "amplitude": amplitude_encoding,
    "iqp":       iqp_encoding,
}
