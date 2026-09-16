"""Draw the RQ1 QCNN circuit for ch3 §3.3.

ch3 carried a `[FIGURE: ... Generate with qml.draw_mpl]` placeholder that was never acted on,
so the methodology chapter described a circuit it never showed. This produces it.

Configuration is RQ1's exactly: 6 qubits, 2 layers, qcnn_native, amplitude encoding, the
74-parameter model. Output: results/figures/fig_circuit_qcnn.png (plus a text rendering,
which is the version worth pasting into a viva slide or an appendix).

Run:  python experiments/make_circuit_figure.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
import torch

from models.qcnn import QCNNModel

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "figures")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(42)

    model = QCNNModel(n_qubits=6, n_layers=2, ansatz="qcnn_native",
                      encoding="amplitude", n_classes=2)
    print("QCNN: %d circuit params, %d total" % (model.n_circuit_params, model.n_params))

    # One real input, so the encoding block draws with actual data rather than placeholders.
    x = torch.randn(1, 64)
    x = x / x.norm(dim=1, keepdim=True)

    qnode = model.circuit if hasattr(model, "circuit") else model.qnode

    # Text rendering first: it always works, and it is the one that survives being pasted
    # into a document without a figure pipeline.
    try:
        txt = qml.draw(qnode, max_length=200)(x[0], model.q_params)
        path_txt = os.path.join(OUT_DIR, "fig_circuit_qcnn.txt")
        with open(path_txt, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print("wrote %s" % path_txt)
        print(txt[:1500])
    except Exception as exc:
        print("text draw failed: %s" % exc)
        raise

    fig, ax = qml.draw_mpl(qnode, style="pennylane")(x[0], model.q_params)
    fig.suptitle("QCNN, $n=6$ qubits, 2 conv+pool layers, amplitude encoding "
                 "(74 trainable parameters)", fontsize=9)
    path_png = os.path.join(OUT_DIR, "fig_circuit_qcnn.png")
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % path_png)


if __name__ == "__main__":
    main()
