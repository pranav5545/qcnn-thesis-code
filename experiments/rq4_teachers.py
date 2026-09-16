"""
RQ4 stage 1 — train the teachers and cache everything the student stage needs.

THIS IS THE ONLY QUANTUM COMPUTE IN RQ4. Stage 2 (rq4_distillation.py) is pure PyTorch
and consumes this stage's cached logits, so the expensive part runs exactly once.

Per (dataset, seed) it produces:
  - a trained QCNN teacher      — RQ1's baseline config (6 qubits, qcnn_native,
                                  amplitude, depth 2, 74 params)
  - a trained classical control teacher — capacity-IDENTICAL to the base student
                                  (design decision D1), so the qkd-vs-ckd contrast
                                  differs only in the teacher's model class
  - both teachers' WEIGHTS      — results/rq4/weights/
  - both teachers' logits on the train set, the clean test set, and all nine
                                  corrupted test sets — results/rq4/logits/

Weights: until 2026-08-24 this project saved no model weights anywhere. RQ1/RQ2/RQ3 kept
per-epoch metrics only, which is exactly why RQ4 cannot load RQ1's trained QCNN and has
to spend ~2 h retraining it. Everything trained here is written to disk so that never
costs a rerun again — see utils/training.py::save_state.

Device-reuse note (inherited from RQ1): lightning.qubit leaves global tape state after
execution, so the QCNNModel is built ONCE per dataset and its parameters reset between
seeds. Re-instantiating it inside the seed loop crashes PennyLane's QueuingManager on
seed 2.

Checkpointing: results/rq4/teachers_checkpoint.json after every (dataset, seed), guarded
on epochs/n_train exactly as rq2_design_choices.py does, so a reclaimed VM costs one cell.

Usage:
    python experiments/rq4_teachers.py --quick    # 2 seeds, 5 epochs, 100 train
    python experiments/rq4_teachers.py            # 5 seeds, 50 epochs, 500 train
"""
import sys, argparse, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import N_QUBITS, N_CLASSES, SEED, EPOCHS, LR, BATCH_SIZE, RESULTS_DIR
from models.qcnn          import QCNNModel
from models.classical_cnn import ClassicalCNN, make_matched_cnn
from utils.dataset        import get_tensors
from utils.training       import run_training
from utils.metrics        import save_results, load_results
from utils.distill        import collect_logits, assert_logits_aligned
from utils.corruptions    import build_corruption_suite, CORRUPTION_NAMES

# ── Teacher configuration — identical to RQ1's baseline cell ──────────────────

N_LAYERS  = 2
ANSATZ    = "qcnn_native"
ENCODING  = "amplitude"
CLASSES   = [0, 1]
DATASETS  = ["mnist", "fashion_mnist"]
DEVICE    = "cpu"

N_TRAIN_FULL,  N_TEST_FULL  = 500, 200
N_TRAIN_QUICK, N_TEST_QUICK = 100, 100

OUT_DIR     = Path(RESULTS_DIR) / "rq4"
WEIGHTS_DIR = OUT_DIR / "weights"
LOGITS_DIR  = OUT_DIR / "logits"
DATA_DIR_RQ4 = OUT_DIR / "data"      # exact tensors both stages are scored on

# ── Arguments ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="2 seeds, 5 epochs, 100 train")
parser.add_argument("--seeds", type=int, default=5,
                    help="number of seeds starting at config.SEED (default 5, matching RQ2)")
parser.add_argument("--qdevice", type=str, default="lightning.qubit",
                    help="PennyLane device. Do NOT set lightning.gpu — GPU is slower for "
                         "this unbatched 6-qubit circuit (verified, see handover).")
parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
args = parser.parse_args()

if args.quick:
    RUN_SEEDS  = [SEED + i for i in range(2)]
    RUN_EPOCHS = 5
    N_TRAIN, N_TEST = N_TRAIN_QUICK, N_TEST_QUICK
    print(f"[quick mode] seeds=2 epochs=5 n_train={N_TRAIN} n_test={N_TEST}\n", flush=True)
else:
    RUN_SEEDS  = [SEED + i for i in range(args.seeds)]
    RUN_EPOCHS = EPOCHS
    N_TRAIN, N_TEST = N_TRAIN_FULL, N_TEST_FULL
    print(f"[full mode] seeds={args.seeds} epochs={RUN_EPOCHS} "
          f"n_train={N_TRAIN} n_test={N_TEST}\n", flush=True)

RUN_DATASETS = [d.strip() for d in args.datasets.split(",") if d.strip()]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_qcnn_params(model: QCNNModel, seed: int) -> None:
    """
    Re-initialise QCNN parameters for a new seed without touching the device/QNode.
    Copied deliberately from experiments/rq1_performance.py rather than imported: RQ4's
    teacher must be initialised by exactly the procedure RQ1 used, and an import would
    let a later edit there silently change what RQ4 means by "the RQ1 teacher".
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        model.q_params.data.normal_(0, 0.1)
        nn.init.kaiming_uniform_(model.head.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.head.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(model.head.bias, -bound, bound)


def _cache_logits(model, label, X_train, X_test, corrupted, verbose_every=False) -> dict:
    """Teacher logits on every tensor the student stage will ever be scored against."""
    out = {"train": collect_logits(model, X_train, DEVICE, BATCH_SIZE),
           "test":  collect_logits(model, X_test,  DEVICE, BATCH_SIZE)}
    for name in CORRUPTION_NAMES:
        out[f"corrupt/{name}"] = collect_logits(model, corrupted[name], DEVICE, BATCH_SIZE)
        print(f"      [{label}] logits cached: corrupt/{name}", flush=True)
    return out


def run_dataset(dataset_name: str, checkpoint: dict) -> dict:
    print(f"\n{'='*66}\n  TEACHERS — {dataset_name.upper()}\n{'='*66}", flush=True)

    X_train, y_train, X_test, y_test = get_tensors(
        name=dataset_name, classes=CLASSES, n_train=N_TRAIN, n_test=N_TEST, seed=SEED,
    )
    print(f"  train {tuple(X_train.shape)}  test {tuple(X_test.shape)}  "
          f"classes {sorted(y_train.unique().tolist())}", flush=True)

    # Corruption suite is built once per dataset and SAVED. The student stage reloads
    # these exact tensors rather than regenerating them — teacher and student must be
    # scored on bit-identical inputs or their agreement is measured on different data.
    corrupted = build_corruption_suite(X_test)
    DATA_DIR_RQ4.mkdir(parents=True, exist_ok=True)
    torch.save({"X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test, "corrupted": corrupted},
               DATA_DIR_RQ4 / f"{dataset_name}.pt")
    print(f"  data + corruption suite ({len(corrupted)} sets) → "
          f"{DATA_DIR_RQ4 / (dataset_name + '.pt')}", flush=True)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # Build the QCNN ONCE — see the device-reuse note in the module docstring.
    torch.manual_seed(SEED)
    qcnn = QCNNModel(n_qubits=N_QUBITS, n_layers=N_LAYERS, ansatz=ANSATZ,
                     encoding=ENCODING, n_classes=N_CLASSES, q_device=args.qdevice)

    # The classical control teacher is derived, not hardcoded: it is whatever
    # make_matched_cnn picks for the QCNN's parameter count — i.e. the same 71-param
    # architecture as the base student (rung 1) and as RQ1's CNN.
    ref_cnn = make_matched_cnn(qcnn.n_params, image_size=8, n_classes=N_CLASSES)
    print(f"  QCNN  {qcnn.n_params} params (circuit {qcnn.n_circuit_params})", flush=True)
    print(f"  CNN   {ref_cnn.n_params} params  n_filters={ref_cnn.n_filters} "
          f"hidden={ref_cnn.hidden}", flush=True)

    done = set(checkpoint["seeds_done"].get(dataset_name, []))

    for seed in RUN_SEEDS:
        if seed in done:
            print(f"\n  -- seed {seed} -- [skipped, in checkpoint]", flush=True)
            continue
        print(f"\n  -- seed {seed} --", flush=True)
        t_seed = time.perf_counter()

        # ---- quantum teacher -------------------------------------------------
        _reset_qcnn_params(qcnn, seed)
        print("  [QCNN teacher]", flush=True)
        q_hist = run_training(
            qcnn, train_loader, test_loader, epochs=RUN_EPOCHS, lr=LR, device=DEVICE,
            seed=seed, verbose=True,
            save_state_path=WEIGHTS_DIR / dataset_name / f"qcnn_teacher_seed{seed}.pt",
            save_state_meta={"role": "teacher", "kind": "qcnn", "dataset": dataset_name,
                             "n_qubits": N_QUBITS, "n_layers": N_LAYERS,
                             "ansatz": ANSATZ, "encoding": ENCODING,
                             "n_params": qcnn.n_params, "n_train": N_TRAIN},
        )

        # ---- classical control teacher ---------------------------------------
        torch.manual_seed(seed)
        cnn_teacher = ClassicalCNN(n_filters=ref_cnn.n_filters, hidden=ref_cnn.hidden,
                                   image_size=8, n_classes=N_CLASSES)
        print("  [CNN control teacher]", flush=True)
        c_hist = run_training(
            cnn_teacher, train_loader, test_loader, epochs=RUN_EPOCHS, lr=LR,
            device=DEVICE, seed=seed, verbose=False,
            save_state_path=WEIGHTS_DIR / dataset_name / f"cnn_teacher_seed{seed}.pt",
            save_state_meta={"role": "teacher", "kind": "cnn", "dataset": dataset_name,
                             "n_filters": ref_cnn.n_filters, "hidden": ref_cnn.hidden,
                             "n_params": ref_cnn.n_params, "n_train": N_TRAIN},
        )
        print(f"    CNN teacher final test_acc={c_hist['test_acc'][-1]:.4f}", flush=True)

        # ---- logit cache + the alignment gate ---------------------------------
        print("  [caching logits]", flush=True)
        q_logits = _cache_logits(qcnn, "qcnn", X_train, X_test, corrupted)
        c_logits = _cache_logits(cnn_teacher, "cnn", X_train, X_test, corrupted)

        # VERIFICATION GATE 2 (implementation plan). Re-run each teacher over the stored
        # X and require the cached train logits to reproduce. A misalignment here is
        # invisible downstream — the student would fit shuffled soft targets and still
        # produce a plausible curve — so this raises rather than warns.
        q_diff = assert_logits_aligned(qcnn, X_train, q_logits["train"], DEVICE)
        c_diff = assert_logits_aligned(cnn_teacher, X_train, c_logits["train"], DEVICE)
        print(f"    alignment OK — max|diff| qcnn={q_diff:.2e} cnn={c_diff:.2e}", flush=True)

        LOGITS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"dataset": dataset_name, "seed": seed,
             "y_train": y_train, "y_test": y_test,
             "qcnn": q_logits, "cnn": c_logits,
             "n_train": N_TRAIN, "n_test": N_TEST, "epochs": RUN_EPOCHS},
            LOGITS_DIR / f"{dataset_name}_seed{seed}.pt",
        )

        # ---- checkpoint ------------------------------------------------------
        checkpoint["histories"].setdefault(dataset_name, {})[str(seed)] = {
            "qcnn": q_hist, "cnn": c_hist,
            "logit_max_diff": {"qcnn": q_diff, "cnn": c_diff},
        }
        done.add(seed)
        checkpoint["seeds_done"][dataset_name] = sorted(done)
        checkpoint["arch"] = {"qcnn_n_params": qcnn.n_params,
                              "cnn_n_params": ref_cnn.n_params,
                              "cnn_n_filters": ref_cnn.n_filters,
                              "cnn_hidden": ref_cnn.hidden}
        save_results(checkpoint, OUT_DIR / "teachers_checkpoint.json")

        print(f"    QCNN test_acc={q_hist['test_acc'][-1]:.4f}  "
              f"CNN test_acc={c_hist['test_acc'][-1]:.4f}  "
              f"({(time.perf_counter() - t_seed)/60:.1f} min)", flush=True)

    return checkpoint

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "teachers_checkpoint.json"

    # Resume guard: same shape as rq2_design_choices.py's. Only resume onto a checkpoint
    # written under the same epochs/n_train, or results from two different protocols get
    # silently mixed under one key.
    checkpoint = {"epochs": RUN_EPOCHS, "n_train": N_TRAIN, "n_test": N_TEST,
                  "seeds_done": {}, "histories": {}}
    if ckpt_path.exists():
        saved = load_results(ckpt_path)
        if saved.get("epochs") == RUN_EPOCHS and saved.get("n_train") == N_TRAIN:
            checkpoint = saved
            checkpoint.setdefault("seeds_done", {})
            checkpoint.setdefault("histories", {})
            n = sum(len(v) for v in checkpoint["seeds_done"].values())
            print(f"Resumed teachers checkpoint: {n} (dataset, seed) cells done\n", flush=True)
        else:
            print(f"Checkpoint protocol mismatch "
                  f"(epochs {saved.get('epochs')} vs {RUN_EPOCHS}, "
                  f"n_train {saved.get('n_train')} vs {N_TRAIN}) — starting fresh\n", flush=True)

    for ds in RUN_DATASETS:
        checkpoint = run_dataset(ds, checkpoint)

    save_results(checkpoint, ckpt_path)
    print(f"\n{'='*66}")
    print(f"  TEACHERS DONE — {(time.perf_counter() - t0)/60:.1f} min")
    print(f"  weights → {WEIGHTS_DIR}")
    print(f"  logits  → {LOGITS_DIR}")
    print(f"  data    → {DATA_DIR_RQ4}")
    print(f"{'='*66}\n", flush=True)
