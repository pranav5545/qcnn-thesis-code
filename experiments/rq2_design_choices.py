"""
RQ2: How do encoding strategy, circuit depth, and ansatz affect QCNN performance?

Internal QCNN design-choice ablation (no classical CNN — that comparison is RQ1).
Three axes, run as one config list so nothing is computed twice:

  Core grid  (fixed depth 2, 6 qubits):
      encoding {angle, amplitude, iqp}  x  ansatz {qcnn_native, hardware_efficient}
      -> 6 configs.  Tests the two headline design choices, including the
         qcnn_native (provably BP-free) vs hardware_efficient (2-design / BP-prone)
         contrast from the trainability literature.

  Depth sub-study  (amplitude + qcnn_native, the RQ1 baseline encoding):
      depth {1, 3}  (depth 2 already covered by the core grid)
      -> 2 configs.  [1,2,3] is the FULL non-degenerate depth range at 6 qubits:
         pooling halves active qubits each layer (6->3->2->1), so depths >=3 are
         architecturally identical (faithful to Cong et al. 2019). Sweeping
         [2,3,4,5,6] as config.py lists would silently run redundant circuits.

  Capacity-maximised hardware_efficient (Q5, added 2026-08-18):
      encoding {angle, amplitude, iqp}  x  ansatz hardware_efficient  x  depth 3
      -> 3 configs.  hardware_efficient@depth2 (38 params) is far under
         qcnn_native@depth2 (74 params) — any qcnn_native win in the core grid is
         confounded with ~2x capacity. depth=3 is hardware_efficient's own
         non-degenerate ceiling at 6 qubits (44 params — depth>=4 is identical to
         depth=3, same halving argument as above) and still does NOT reach 74; full
         depth-matching is architecturally impossible at this qubit count (verified
         2026-08-18: HE param count per block is fixed at 6 vs qcnn_native's 15, and
         both hit the same qubit-halving depth ceiling). Per the design decision
         (Appendix F of the dissertation, Q5), we report BOTH the raw
         depth-2 comparison AND this depth-3 "capacity-maximised" comparison,
         disclosing the residual 44-vs-74 gap rather than claiming a false match.

Total: 11 distinct configs x N_SEEDS seeds x 2 datasets.

Encoding input note (Q4, added 2026-08-18): angle/iqp configs now use
utils.dataset.get_pca_loaders (PCA 64->6, literature-standard per Hur et al. 2022)
instead of the original naive first-6-of-64-pixels truncation, per the Q4 design
decision. amplitude keeps the original get_loaders (needs the full 64 features for
AmplitudeEmbedding). The naive-truncation path is untouched in the code
(get_loaders is still what it was) and stays available to rerun later if the
truncation-as-is comparison is wanted — see the design-decision doc's "kept, not
deleted" note.

Device-reuse note: unlike RQ1, each config needs a different circuit (different
depth/ansatz/encoding => different parameter count), so a fresh QCNNModel is built
per (config, seed). A pre-flight probe confirmed sequential lightning.qubit device
creation is safe on the current Python 3.11 / PennyLane 0.44 stack (the Python-3.14
tape-state crash noted in PROGRESS.md Finding 5 no longer applies).

Checkpoint note: results/rq2/{dataset}_checkpoint.json stores every finished
(config, seed) history; reruns skip completed cells, so interrupted runs resume.

Usage:
    python experiments/rq2_design_choices.py --quick   # 2 seeds, 5 epochs, 200 train
    python experiments/rq2_design_choices.py           # N_SEEDS seeds, 50 epochs, 500 train
    python experiments/rq2_design_choices.py --qdevice lightning.gpu   # cuQuantum
"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import (
    N_QUBITS, N_CLASSES, SEED, N_SEEDS,
    EPOCHS, LR, BATCH_SIZE, RESULTS_DIR,
)
from models.qcnn    import QCNNModel
from utils.dataset  import get_loaders, get_pca_loaders, describe
from utils.training import run_training
from utils.metrics  import save_results, summarise_seeds, load_results

# ── Experiment settings ───────────────────────────────────────────────────────

CLASSES  = [0, 1]
DATASETS = ["mnist", "fashion_mnist"]
DEVICE   = "cpu"   # torch device for the classical head; independent of --qdevice

ENCODINGS = ["angle", "amplitude", "iqp"]
ANSATZE   = ["qcnn_native", "hardware_efficient"]
CORE_DEPTH = 2                       # fixed depth for the encoding x ansatz grid
DEPTH_STUDY = {"encoding": "amplitude", "ansatz": "qcnn_native", "depths": [1, 3]}
CAPACITY_STUDY = {"ansatz": "hardware_efficient", "depth": 3, "encodings": ENCODINGS}

PCA_ENCODINGS = {"angle", "iqp"}     # these use get_pca_loaders; amplitude uses get_loaders

N_TRAIN_FULL,  N_TEST_FULL  = 500, 200
N_TRAIN_QUICK, N_TEST_QUICK = 200, 100


def build_config_list() -> list[dict]:
    """The 11 distinct (encoding, ansatz, depth) cells, core grid first."""
    configs = []
    for enc in ENCODINGS:
        for ans in ANSATZE:
            configs.append({"encoding": enc, "ansatz": ans, "depth": CORE_DEPTH,
                            "group": "core"})
    for d in DEPTH_STUDY["depths"]:
        configs.append({"encoding": DEPTH_STUDY["encoding"],
                        "ansatz":   DEPTH_STUDY["ansatz"],
                        "depth":    d, "group": "depth"})
    for enc in CAPACITY_STUDY["encodings"]:
        configs.append({"encoding": enc, "ansatz": CAPACITY_STUDY["ansatz"],
                        "depth": CAPACITY_STUDY["depth"], "group": "capacity"})
    return configs


def config_key(cfg: dict) -> str:
    return f"{cfg['encoding']}__{cfg['ansatz']}__L{cfg['depth']}"


# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="2 seeds, 5 epochs, 200 train samples")
parser.add_argument("--datasets", nargs="+", default=None,
                    help="subset of datasets to run (default: both)")
parser.add_argument("--n_train", type=int, default=None)
parser.add_argument("--n_test",  type=int, default=None)
parser.add_argument("--seeds", type=int, default=None,
                    help="override N_SEEDS from config.py (config.py's N_SEEDS=10 "
                         "was bumped for RQ1; confirm before assuming it applies here)")
parser.add_argument("--qdevice", default="lightning.qubit",
                    help="PennyLane device, e.g. lightning.qubit (CPU) or "
                         "lightning.gpu (cuQuantum) — see models/qcnn.py docstring "
                         "on the non-batched-execution GPU caveat before using GPU")
args = parser.parse_args()

N_SEEDS_RUN = args.seeds or N_SEEDS

if args.quick:
    RUN_SEEDS  = [SEED + i for i in range(2)]
    RUN_EPOCHS = 5
    N_TRAIN    = args.n_train or N_TRAIN_QUICK
    N_TEST     = args.n_test  or N_TEST_QUICK
    print(f"[quick mode] seeds=2  epochs=5  n_train={N_TRAIN}  n_test={N_TEST}\n")
else:
    RUN_SEEDS  = [SEED + i for i in range(N_SEEDS_RUN)]
    RUN_EPOCHS = EPOCHS
    N_TRAIN    = args.n_train or N_TRAIN_FULL
    N_TEST     = args.n_test  or N_TEST_FULL
    print(f"[full mode] seeds={N_SEEDS_RUN}  epochs={EPOCHS}  n_train={N_TRAIN}  n_test={N_TEST}\n")

RUN_DATASETS = args.datasets or DATASETS
Q_DEVICE     = args.qdevice


# ── Per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(dataset_name: str, configs: list[dict]) -> dict:
    print(f"\n{'='*66}")
    print(f"  RQ2  Dataset: {dataset_name.upper()}")
    print(f"{'='*66}")

    # Two loader pairs: raw 64-dim for amplitude, PCA 6-dim for angle/iqp (Q4).
    # Same underlying sample selection (shared `seed` default in both functions),
    # so the two pairs cover the identical train/test images, just different features.
    train_loader, test_loader = get_loaders(
        name=dataset_name, classes=CLASSES, batch_size=BATCH_SIZE,
        n_train=N_TRAIN, n_test=N_TEST,
    )
    pca_train_loader, pca_test_loader = get_pca_loaders(
        name=dataset_name, classes=CLASSES, batch_size=BATCH_SIZE,
        n_train=N_TRAIN, n_test=N_TEST, n_components=N_QUBITS,
    )
    print(f"  Train (raw)  : {describe(train_loader)}")
    print(f"  Test  (raw)  : {describe(test_loader)}")
    print(f"  Train (pca)  : {describe(pca_train_loader)}")
    print(f"  Test  (pca)  : {describe(pca_test_loader)}")

    out_dir = Path(RESULTS_DIR) / "rq2"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{dataset_name}_checkpoint.json"

    # histories[config_key][str(seed)] = per-epoch history dict
    if checkpoint_path.exists():
        saved = load_results(checkpoint_path)
        if saved.get("epochs") == RUN_EPOCHS and saved.get("n_train") == N_TRAIN:
            histories = saved["histories"]
            print(f"\n  Resumed checkpoint: "
                  f"{sum(len(v) for v in histories.values())} (config,seed) cells done")
        else:
            print(f"\n  Checkpoint settings mismatch — starting fresh")
            histories = {}
    else:
        histories = {}

    for ci, cfg in enumerate(configs):
        key = config_key(cfg)
        histories.setdefault(key, {})
        print(f"\n  [{ci+1}/{len(configs)}] {key}  ({cfg['group']})")

        for seed in RUN_SEEDS:
            if str(seed) in histories[key]:
                print(f"      seed {seed}: [skipped, in checkpoint]")
                continue

            torch.manual_seed(seed)
            model = QCNNModel(
                n_qubits=N_QUBITS, n_layers=cfg["depth"],
                ansatz=cfg["ansatz"], encoding=cfg["encoding"], n_classes=N_CLASSES,
                q_device=Q_DEVICE,
            )
            cfg_train, cfg_test = (
                (pca_train_loader, pca_test_loader) if cfg["encoding"] in PCA_ENCODINGS
                else (train_loader, test_loader)
            )
            t0 = time.perf_counter()
            h = run_training(model, cfg_train, cfg_test,
                             epochs=RUN_EPOCHS, lr=LR, device=DEVICE,
                             seed=seed, verbose=False)
            dt = time.perf_counter() - t0
            h["n_params"] = model.n_params
            histories[key][str(seed)] = h
            print(f"      seed {seed}: test_acc={h['test_acc'][-1]:.3f}  "
                  f"n_params={model.n_params}  ({dt/60:.1f} min)")

            # Checkpoint after every (config, seed) so nothing is lost on interrupt
            save_results({
                "experiment": "rq2",
                "dataset":    dataset_name,
                "n_qubits":   N_QUBITS,
                "epochs":     RUN_EPOCHS,
                "n_train":    N_TRAIN,
                "n_test":     N_TEST,
                "seeds":      RUN_SEEDS,
                "core_depth": CORE_DEPTH,
                "histories":  histories,
            }, checkpoint_path)

        seeds_done = [s for s in RUN_SEEDS if str(s) in histories[key]]
        if seeds_done:
            summ = summarise_seeds([histories[key][str(s)] for s in seeds_done], "test_acc")
            print(f"      -> {key}: test_acc {summ['mean']:.4f} +- {summ['std']:.4f}  "
                  f"(n={len(seeds_done)})")

    # Final per-config summary table
    summary = {}
    for cfg in configs:
        key = config_key(cfg)
        seeds_done = [s for s in RUN_SEEDS if str(s) in histories.get(key, {})]
        if seeds_done:
            summary[key] = {
                **summarise_seeds([histories[key][str(s)] for s in seeds_done], "test_acc"),
                "group": cfg["group"], "encoding": cfg["encoding"],
                "ansatz": cfg["ansatz"], "depth": cfg["depth"],
            }

    result = {
        "experiment": "rq2", "dataset": dataset_name,
        "n_qubits": N_QUBITS, "epochs": RUN_EPOCHS,
        "n_train": N_TRAIN, "n_test": N_TEST, "seeds": RUN_SEEDS,
        "summary": summary, "histories": histories,
    }
    save_results(result, out_dir / f"{dataset_name}.json")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.perf_counter()
    configs = build_config_list()
    print(f"RQ2 config grid ({len(configs)} distinct cells):")
    for c in configs:
        print(f"   {config_key(c):40s} [{c['group']}]")

    all_results = {}
    for ds in RUN_DATASETS:
        all_results[ds] = run_dataset(ds, configs)

    print(f"\n{'='*66}")
    print("  RQ2 SUMMARY  (mean +- std test accuracy, final epoch)")
    print(f"{'='*66}")
    header = f"  {'Config':40s}" + "".join(f"{ds:>16s}" for ds in RUN_DATASETS)
    print(header)
    print(f"  {'-'*(38 + 16*len(RUN_DATASETS))}")
    for c in configs:
        key = config_key(c)
        row = f"  {key:40s}"
        for ds in RUN_DATASETS:
            s = all_results[ds]["summary"].get(key)
            row += f"{s['mean']:.3f}+-{s['std']:.3f} " .rjust(16) if s else " " * 16
        print(row)

    print(f"\n  Total wall time : {(time.perf_counter() - t0)/60:.1f} min")
    print(f"  Results saved to: {RESULTS_DIR}/rq2/")
    print(f"{'='*66}\n")
