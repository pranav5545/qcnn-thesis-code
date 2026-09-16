"""
RQ1: How does QCNN predictive performance compare to a parameter-matched
classical CNN under controlled training conditions on small-scale image tasks?

Runs both models across N_SEEDS seeds on MNIST and Fashion-MNIST (binary),
reports mean ± std test accuracy, and saves full per-epoch histories.

Device reuse note: lightning.qubit leaves global tape state after each circuit
execution. Re-creating the device inside the seed loop (one per seed) corrupts
that state on seed 2+, causing a crash in PennyLane's QueuingManager. Fix:
create the QCNNModel (and its device) ONCE per dataset, reset only parameters
between seeds. Pure-PyTorch CNN is safe to recreate each seed.

Checkpoint note: seeds already in results/rq1/{dataset}_checkpoint.json are
skipped automatically, so interrupted runs resume cleanly.

Usage:
    python experiments/rq1_performance.py --quick        # 3 seeds, 5 epochs, 200 train
    python experiments/rq1_performance.py                # 5 seeds, 30 epochs, 500 train
    python experiments/rq1_performance.py --n_train 800  # override sample count
"""
import sys, argparse, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from config import (
    N_QUBITS, N_CLASSES, SEED, N_SEEDS,
    EPOCHS, LR, BATCH_SIZE, RESULTS_DIR
)
from models.qcnn          import QCNNModel
from models.classical_cnn import ClassicalCNN, make_matched_cnn
from utils.dataset        import get_loaders, describe
from utils.training       import run_training
from utils.metrics        import save_results, summarise_seeds, load_results

# ── Experiment settings ───────────────────────────────────────────────────────

N_LAYERS  = 2
ANSATZ    = "qcnn_native"
ENCODING  = "amplitude"
CLASSES   = [0, 1]
DATASETS  = ["mnist", "fashion_mnist"]
DEVICE    = "cpu"

N_TRAIN_FULL  = 500
N_TEST_FULL   = 200
N_TRAIN_QUICK = 200
N_TEST_QUICK  = 100

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--quick",   action="store_true",
                    help="3 seeds, 5 epochs, 200 train samples")
parser.add_argument("--n_train", type=int, default=None)
parser.add_argument("--n_test",  type=int, default=None)
args = parser.parse_args()

if args.quick:
    RUN_SEEDS  = [SEED + i for i in range(3)]
    RUN_EPOCHS = 5
    N_TRAIN    = args.n_train or N_TRAIN_QUICK
    N_TEST     = args.n_test  or N_TEST_QUICK
    print(f"[quick mode] seeds=3  epochs=5  n_train={N_TRAIN}  n_test={N_TEST}\n")
else:
    RUN_SEEDS  = [SEED + i for i in range(N_SEEDS)]
    RUN_EPOCHS = EPOCHS
    N_TRAIN    = args.n_train or N_TRAIN_FULL
    N_TEST     = args.n_test  or N_TEST_FULL
    print(f"[full mode] seeds={N_SEEDS}  epochs={EPOCHS}  n_train={N_TRAIN}  n_test={N_TEST}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_qcnn_params(model: QCNNModel, seed: int) -> None:
    """Re-initialise QCNN parameters for a new seed without touching the device/QNode."""
    torch.manual_seed(seed)
    with torch.no_grad():
        model.q_params.data.normal_(0, 0.1)
        # Replicate nn.Linear's default kaiming_uniform_ init
        nn.init.kaiming_uniform_(model.head.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.head.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(model.head.bias, -bound, bound)


def run_dataset(dataset_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    train_loader, test_loader = get_loaders(
        name=dataset_name,
        classes=CLASSES,
        batch_size=BATCH_SIZE,
        n_train=N_TRAIN,
        n_test=N_TEST,
    )
    print(f"  Train: {describe(train_loader)}")
    print(f"  Test : {describe(test_loader)}")

    # Create QCNN ONCE — its lightning.qubit device and QNode are reused for all seeds.
    # Re-creating the device each seed corrupts PennyLane's global tape state after seed 1.
    torch.manual_seed(SEED)
    qcnn = QCNNModel(
        n_qubits=N_QUBITS, n_layers=N_LAYERS,
        ansatz=ANSATZ, encoding=ENCODING, n_classes=N_CLASSES,
    )

    # Find matched CNN architecture once (pure Python grid search, no quantum state)
    ref_cnn = make_matched_cnn(qcnn.n_params, image_size=8, n_classes=N_CLASSES)
    print(f"\n  QCNN params : {qcnn.n_params}  (circuit: {qcnn.n_circuit_params})")
    print(f"  CNN  params : {ref_cnn.n_params}  (delta: {abs(ref_cnn.n_params - qcnn.n_params)})")
    print(f"  CNN  arch   : n_filters={ref_cnn.n_filters}  hidden={ref_cnn.hidden}")

    out_dir = Path(RESULTS_DIR) / "rq1"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{dataset_name}_checkpoint.json"

    # Load checkpoint so interrupted runs resume without rerunning finished seeds
    if checkpoint_path.exists():
        saved      = load_results(checkpoint_path)
        done_seeds = set(saved["seeds_done"])
        # Only resume if the saved run used the same epoch count
        if saved.get("epochs") == RUN_EPOCHS:
            qcnn_histories = saved["qcnn_histories"]
            cnn_histories  = saved["cnn_histories"]
            print(f"\n  Resumed checkpoint: {sorted(done_seeds)} already done")
        else:
            print(f"\n  Checkpoint epoch count mismatch ({saved.get('epochs')} vs {RUN_EPOCHS}) — starting fresh")
            qcnn_histories, cnn_histories = [], []
            done_seeds = set()
    else:
        qcnn_histories, cnn_histories = [], []
        done_seeds = set()

    for run_idx, seed in enumerate(RUN_SEEDS):
        if seed in done_seeds:
            print(f"\n  -- Seed {seed}  ({run_idx + 1}/{len(RUN_SEEDS)}) -- [skipped, in checkpoint]")
            continue

        print(f"\n  -- Seed {seed}  ({run_idx + 1}/{len(RUN_SEEDS)}) --")

        # Reset QCNN parameters only — device/QNode are reused (avoids tape state bug)
        _reset_qcnn_params(qcnn, seed)

        # CNN is pure PyTorch: safe to recreate each seed
        torch.manual_seed(seed)
        cnn = ClassicalCNN(
            n_filters=ref_cnn.n_filters, hidden=ref_cnn.hidden,
            image_size=8, n_classes=N_CLASSES,
        )

        print(f"  [QCNN]")
        qh = run_training(qcnn, train_loader, test_loader,
                          epochs=RUN_EPOCHS, lr=LR, device=DEVICE,
                          seed=seed, verbose=True)
        qcnn_histories.append(qh)

        print(f"  [CNN]")
        ch = run_training(cnn, train_loader, test_loader,
                          epochs=RUN_EPOCHS, lr=LR, device=DEVICE,
                          seed=seed, verbose=True)
        cnn_histories.append(ch)

        # Save checkpoint after every seed
        checkpoint = {
            "dataset":        dataset_name,
            "n_qubits":       N_QUBITS,
            "n_layers":       N_LAYERS,
            "ansatz":         ANSATZ,
            "encoding":       ENCODING,
            "epochs":         RUN_EPOCHS,
            "n_train":        N_TRAIN,
            "n_test":         N_TEST,
            "seeds_done":     [s for s in RUN_SEEDS if s in done_seeds or s == seed],
            "qcnn_n_params":  qcnn.n_params,
            "cnn_n_params":   ref_cnn.n_params,
            "qcnn_histories": qcnn_histories,
            "cnn_histories":  cnn_histories,
        }
        done_seeds.add(seed)
        save_results(checkpoint, checkpoint_path)

        q_summary = summarise_seeds(qcnn_histories, "test_acc")
        c_summary = summarise_seeds(cnn_histories,  "test_acc")
        print(f"\n  After seed {seed}:")
        print(f"    QCNN test_acc so far: {q_summary['mean']:.4f} +- {q_summary['std']:.4f}")
        print(f"    CNN  test_acc so far: {c_summary['mean']:.4f} +- {c_summary['std']:.4f}")

    # Final result
    result = {
        **checkpoint,
        "seeds":             RUN_SEEDS,
        "qcnn_summary":      summarise_seeds(qcnn_histories, "test_acc"),
        "cnn_summary":       summarise_seeds(cnn_histories,  "test_acc"),
        "qcnn_time_summary": summarise_seeds(qcnn_histories, "epoch_time"),
        "cnn_time_summary":  summarise_seeds(cnn_histories,  "epoch_time"),
    }
    save_results(result, out_dir / f"{dataset_name}.json")
    return result

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.perf_counter()
    all_results = {}

    for ds in DATASETS:
        all_results[ds] = run_dataset(ds)

    print(f"\n{'='*60}")
    print("  RQ1 SUMMARY  (mean +- std test accuracy, final epoch)")
    print(f"{'='*60}")
    print(f"  {'Model':<32} {'MNIST':>14} {'FashionMNIST':>14}")
    print(f"  {'-'*60}")

    for label, key in [("QCNN (qcnn_native, angle)", "qcnn_summary"),
                        ("CNN  (param-matched)",       "cnn_summary")]:
        row = []
        for ds in DATASETS:
            s = all_results[ds][key]
            row.append(f"{s['mean']:.4f}+-{s['std']:.4f}")
        print(f"  {label:<32} {row[0]:>14} {row[1]:>14}")

    print(f"\n  Total wall time : {(time.perf_counter() - t0)/60:.1f} min")
    print(f"  Results saved to: {RESULTS_DIR}/rq1/")
    print(f"{'='*60}\n")
