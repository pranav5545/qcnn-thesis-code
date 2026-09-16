"""
Central config for all experiments.
Edit here rather than hunting through individual experiment files.
"""

# ── Dataset ───────────────────────────────────────────────────────────────────

IMAGE_SIZE   = 8          # pixels per side after downsampling (8×8 = 64 features)

# N_QUBITS = 6 is the canonical choice for amplitude encoding:
#   2^6 = 64 amplitudes exactly maps the full 8×8 downsampled image with zero padding waste.
#   Angle and IQP encodings need exactly 6 features, sourced via PCA(64->6) from
#   utils.dataset.get_pca_loaders — NOT the first 6 of 64 pixels, which this comment
#   described until 2026-08-20. That truncation was an information-starvation bug
#   (the first 6 pixels are near-constant background), fixed under RQ2 design
#   decision Q4; see Appendix F of the dissertation (design-decision records).
#   Keeping qubit count fixed across encodings ensures circuit architecture
#   (depth, parameter count) is identical — required for a fair encoding ablation.
N_QUBITS     = 6
N_CLASSES    = 2          # binary task to start (extend to 4 later)
SEED         = 42
N_SEEDS      = 10         # multi-seed runs for variance reporting (RQ1) — bumped from 5
                           # on 2026-08-10 to firm up the underpowered Fashion-MNIST result

DATASETS = ["mnist", "fashion_mnist"]

# ── Training ──────────────────────────────────────────────────────────────────

BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-2

# ── QCNN design space (RQ2) ──────────────────────────────────────────────────

ENCODINGS     = ["angle", "amplitude", "iqp"]
# NOTE: this list is NOT what RQ2 swept. Pooling halves the active wires each block
# (6->3->2->1), so at N_QUBITS=6 every depth >= 3 is architecturally identical to
# depth 3 — depths 4/5/6 here would run three redundant circuits. The sweep in
# experiments/rq2_design_choices.py uses the non-degenerate range [1, 2, 3] instead.
CIRCUIT_DEPTHS = [2, 3, 4, 5, 6]             # number of conv+pool block pairs
ANSATZE       = ["qcnn_native", "hardware_efficient"]

# ── Barren plateau analysis (RQ3) ─────────────────────────────────────────────

BP_QUBIT_COUNTS  = [4, 6, 8, 10]
BP_N_INITS       = 200    # random initialisations to sample gradient variance

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR    = "data"
RESULTS_DIR = "results"
