"""
RQ4 stage 2 — the student grid. Pure PyTorch: no quantum compute happens here.

Every student trains against logits cached by experiments/rq4_teachers.py, so the QCNN is
simulated once for the whole study rather than once per student.

THE QUESTION. Can a parameter-matched classical student absorb the QCNN's learned
function? RQ1 gave parity of accuracy; this gives — or refuses — parity of function.

THE ARMS (5 variants):
    hard          alpha=0.0   plain cross-entropy. Also the RQ1 regression check: at
                              rung 1 this IS RQ1's 71-param CNN, so its Fashion-MNIST
                              accuracy must land near 0.947 or the harness differs from
                              RQ1 somewhere and nothing downstream is trustworthy.
    qkd  a=0.9    quantum teacher, Hinton's default mix
    qkd  a=1.0    quantum teacher, pure distillation (fidelity is the primary metric and
                  hard labels pull the student away from the teacher — D3)
    ckd  a=0.9    classical control teacher, capacity-identical to the base student
    ckd  a=1.0    so that qkd-vs-ckd differs ONLY in the teacher's model class (D1)

THE LADDER (D2 — grows filters AND head, never head alone). The naive best-fit rungs to
{71,150,350,750} all land on n_filters=1 with only `hidden` varying: if the bottleneck is
the conv, such a ladder reads flat for the wrong reason and destroys the very
risk-mitigation it exists for. Reading the ladder is how Stanton et al.'s optimisation
confound gets ruled out — flat agreement all the way up means "hard to optimise", NOT
"not simulable", and no simulability conclusion may be drawn from it.

FIDELITY IS MEASURED AGAINST BOTH TEACHERS FOR EVERY ARM, including `hard`. A
hard-trained student already agrees with the QCNN to some degree simply by solving the
same task; that baseline is what distillation has to beat, and without it an agreement
number means nothing.

Checkpointing: results/rq4/students_checkpoint.json after every cell.
Weights: results/rq4/weights/{dataset}/students/ — every student is saved.

Usage:
    python experiments/rq4_distillation.py --quick   # 2 seeds, 5 epochs, rung 1 only
    python experiments/rq4_distillation.py           # full grid
"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import N_CLASSES, SEED, EPOCHS, LR, BATCH_SIZE, RESULTS_DIR
from models.classical_cnn import ClassicalCNN, make_matched_cnn
from utils.metrics  import save_results, load_results, top1_agreement, predictive_kl, ece
from utils.distill  import build_kd_dataset, run_kd_training
from utils.corruptions import CORRUPTION_NAMES

DATASETS = ["mnist", "fashion_mnist"]
DEVICE   = "cpu"

OUT_DIR     = Path(RESULTS_DIR) / "rq4"
WEIGHTS_DIR = OUT_DIR / "weights"
LOGITS_DIR  = OUT_DIR / "logits"
DATA_DIR_RQ4 = OUT_DIR / "data"

# ── Grid ──────────────────────────────────────────────────────────────────────

# (rung, n_filters, hidden, expected params) — verified against ClassicalCNN.n_params.
RUNGS = [
    (1, 1,  7,   71),   # 1.0x  — RQ1-matched base student
    (2, 2,  8,  148),   # 2.1x
    (3, 4,  8,  342),   # 4.8x
    (4, 4, 32,  798),   # 11.2x
    (5, 4, 69, 1501),   # 21.1x
]

# (arm, teacher_key, alpha). teacher_key None = no soft targets.
ARMS = [
    ("hard", None,   0.0),
    ("qkd",  "qcnn", 0.9),
    ("qkd",  "qcnn", 1.0),
    ("ckd",  "cnn",  0.9),
    ("ckd",  "cnn",  1.0),
]

T_DEFAULT     = 4.0
T_SENSITIVITY = [2.0, 8.0]     # T=4 is already in the main grid
SAMPLE_EFFICIENCY_N = [100, 200]   # 500 is the main grid

# ── Arguments ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="2 seeds, 5 epochs, rung 1 only, no extra groups")
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
parser.add_argument("--no-t-sensitivity", action="store_true")
parser.add_argument("--no-sample-efficiency", action="store_true")
args = parser.parse_args()

if args.quick:
    RUN_SEEDS, RUN_EPOCHS = [SEED, SEED + 1], 5
    RUN_RUNGS = RUNGS[:1]
    DO_T, DO_SE = False, False
    print("[quick mode] seeds=2 epochs=5 rung 1 only\n", flush=True)
else:
    RUN_SEEDS, RUN_EPOCHS = [SEED + i for i in range(args.seeds)], EPOCHS
    RUN_RUNGS = RUNGS
    DO_T  = not args.no_t_sensitivity
    DO_SE = not args.no_sample_efficiency
    print(f"[full mode] seeds={args.seeds} epochs={RUN_EPOCHS} rungs={len(RUN_RUNGS)} "
          f"t_sensitivity={DO_T} sample_efficiency={DO_SE}\n", flush=True)

RUN_DATASETS = [d.strip() for d in args.datasets.split(",") if d.strip()]


def cell_key(dataset, rung, arm, alpha, T, n_train, seed) -> str:
    return f"{dataset}|r{rung}|{arm}|a{alpha}|T{T}|n{n_train}|s{seed}"


def build_cells():
    """
    Every (dataset, rung, arm, alpha, T, n_train, seed) the run will execute.

    Enumerated up front rather than nested in loops so the grid can be counted, printed
    and diffed against the plan before ~4 h of compute is spent on it.
    """
    cells = []
    for ds in RUN_DATASETS:
        for rung, nf, hid, _ in RUN_RUNGS:
            for arm, tkey, alpha in ARMS:
                for seed in RUN_SEEDS:
                    cells.append((ds, rung, nf, hid, arm, tkey, alpha, T_DEFAULT, None, seed))
        if DO_T:
            # Temperature sensitivity at the base rung, quantum teacher, pure distillation.
            rung, nf, hid, _ = RUNGS[0]
            for T in T_SENSITIVITY:
                for seed in RUN_SEEDS:
                    cells.append((ds, rung, nf, hid, "qkd", "qcnn", 1.0, T, None, seed))
        if DO_SE:
            # Sample efficiency at the base rung. n_train=100/200 are exact PREFIXES of the
            # 500-sample training set (utils/dataset.get_tensors docstring), so the cached
            # teacher logits are sliced, never recomputed — no extra quantum compute.
            rung, nf, hid, _ = RUNGS[0]
            for n_tr in SAMPLE_EFFICIENCY_N:
                for arm, tkey, alpha in ARMS:
                    for seed in RUN_SEEDS:
                        cells.append((ds, rung, nf, hid, arm, tkey, alpha, T_DEFAULT, n_tr, seed))
    return cells


def load_dataset_assets(dataset: str):
    data = torch.load(DATA_DIR_RQ4 / f"{dataset}.pt", weights_only=False)
    per_seed = {}
    for seed in RUN_SEEDS:
        p = LOGITS_DIR / f"{dataset}_seed{seed}.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"missing teacher logits: {p}. Run experiments/rq4_teachers.py first — "
                "the student stage never trains a teacher of its own."
            )
        per_seed[seed] = torch.load(p, weights_only=False)
    return data, per_seed


def evaluate_student(student, X_test, y_test, corrupted, t_logits) -> dict:
    """
    Everything measured about a finished student.

    Fidelity is computed against BOTH teachers regardless of which one (if any) trained
    this student — see the module docstring. predictive_kl is called teacher-first and
    kept that way, since KL is asymmetric.
    """
    from utils.distill import collect_logits
    s_test = collect_logits(student, X_test, DEVICE, BATCH_SIZE)
    out = {
        "test_acc":  float((s_test.argmax(1) == y_test).float().mean().item()),
        "ece":       ece(s_test, y_test),
        "agree_qcnn": top1_agreement(s_test, t_logits["qcnn"]["test"]),
        "agree_cnn":  top1_agreement(s_test, t_logits["cnn"]["test"]),
        "kl_qcnn":    predictive_kl(t_logits["qcnn"]["test"], s_test),
        "kl_cnn":     predictive_kl(t_logits["cnn"]["test"],  s_test),
        "corrupt": {},
    }
    for name in CORRUPTION_NAMES:
        s_c = collect_logits(student, corrupted[name], DEVICE, BATCH_SIZE)
        out["corrupt"][name] = {
            "acc":         float((s_c.argmax(1) == y_test).float().mean().item()),
            "agree_qcnn":  top1_agreement(s_c, t_logits["qcnn"][f"corrupt/{name}"]),
            "agree_cnn":   top1_agreement(s_c, t_logits["cnn"][f"corrupt/{name}"]),
        }
    return out

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "students_checkpoint.json"

    # Sanity: rung 1 must still be what make_matched_cnn picks for the 74-param QCNN.
    # If ClassicalCNN's architecture ever changes, this ladder stops being anchored to
    # RQ1's baseline and the regression check silently stops meaning anything.
    ref = make_matched_cnn(74, image_size=8, n_classes=N_CLASSES)
    assert (ref.n_filters, ref.hidden) == (RUNGS[0][1], RUNGS[0][2]), (
        f"rung 1 {(RUNGS[0][1], RUNGS[0][2])} != make_matched_cnn(74) "
        f"{(ref.n_filters, ref.hidden)} — the ladder is no longer anchored to RQ1")
    for rung, nf, hid, expect in RUNGS:
        got = ClassicalCNN(n_filters=nf, hidden=hid, image_size=8, n_classes=N_CLASSES).n_params
        assert got == expect, f"rung {rung}: {got} params, plan says {expect}"

    checkpoint = {"epochs": RUN_EPOCHS, "seeds": RUN_SEEDS, "cells": {}}
    if ckpt_path.exists():
        saved = load_results(ckpt_path)
        if saved.get("epochs") == RUN_EPOCHS:
            checkpoint = saved
            checkpoint.setdefault("cells", {})
            print(f"Resumed students checkpoint: {len(checkpoint['cells'])} cells done\n",
                  flush=True)
        else:
            print(f"Checkpoint epoch mismatch ({saved.get('epochs')} vs {RUN_EPOCHS}) "
                  "— starting fresh\n", flush=True)

    cells = build_cells()
    print(f"Grid: {len(cells)} student runs "
          f"({len(checkpoint['cells'])} already in checkpoint)\n", flush=True)

    assets = {}
    for idx, (ds, rung, nf, hid, arm, tkey, alpha, T, n_tr, seed) in enumerate(cells, 1):
        key = cell_key(ds, rung, arm, alpha, T, n_tr or "full", seed)
        if key in checkpoint["cells"]:
            continue

        if ds not in assets:
            assets[ds] = load_dataset_assets(ds)
            print(f"loaded teacher assets for {ds}", flush=True)
        data, per_seed = assets[ds]
        t_logits = per_seed[seed]

        X_train, y_train = data["X_train"], data["y_train"]
        X_test,  y_test  = data["X_test"],  data["y_test"]
        corrupted = data["corrupted"]

        # Soft targets: the training teacher's logits, or a zero placeholder for the
        # `hard` arm (alpha=0.0 means kd_loss never reads them — but they must still be
        # the right SHAPE, since they ride in the TensorDataset triple).
        if tkey is None:
            teacher_train = torch.zeros(len(X_train), N_CLASSES)
        else:
            teacher_train = t_logits[tkey]["train"]

        # Sample-efficiency slicing. Valid only because of the prefix property documented
        # in get_tensors: the n=100 subset is literally the first 100 rows of the n=500
        # subset, so slicing keeps every sample paired with its own logits.
        if n_tr is not None:
            X_tr_c, y_tr_c, tt_c = X_train[:n_tr], y_train[:n_tr], teacher_train[:n_tr]
        else:
            X_tr_c, y_tr_c, tt_c = X_train, y_train, teacher_train

        kd_ds = build_kd_dataset(X_tr_c, y_tr_c, tt_c)
        test_loader = DataLoader(TensorDataset(X_test, y_test),
                                 batch_size=BATCH_SIZE, shuffle=False)

        torch.manual_seed(seed)
        student = ClassicalCNN(n_filters=nf, hidden=hid, image_size=8, n_classes=N_CLASSES)

        wpath = (WEIGHTS_DIR / ds / "students" /
                 f"r{rung}_{arm}_a{alpha}_T{T}_n{n_tr or 'full'}_s{seed}.pt")
        hist = run_kd_training(
            student, kd_ds, test_loader, epochs=RUN_EPOCHS, lr=LR, device=DEVICE,
            T=T, alpha=alpha, batch_size=BATCH_SIZE, seed=seed, verbose=False,
            save_state_path=wpath,
            save_state_meta={"role": "student", "dataset": ds, "rung": rung,
                             "n_filters": nf, "hidden": hid, "arm": arm,
                             "teacher": tkey, "n_train": n_tr or len(X_train),
                             "n_params": student.n_params},
        )
        metrics = evaluate_student(student, X_test, y_test, corrupted, t_logits)

        checkpoint["cells"][key] = {
            "dataset": ds, "rung": rung, "n_filters": nf, "hidden": hid,
            "n_params": student.n_params, "arm": arm, "teacher": tkey,
            "alpha": alpha, "T": T, "n_train": n_tr or len(X_train), "seed": seed,
            "history": hist, "metrics": metrics,
            "weights": str(wpath),
        }
        save_results(checkpoint, ckpt_path)

        elapsed = (time.perf_counter() - t0) / 60
        print(f"[{idx}/{len(cells)}] {key}  acc={metrics['test_acc']:.4f}  "
              f"agree_q={metrics['agree_qcnn']:.4f}  agree_c={metrics['agree_cnn']:.4f}  "
              f"({elapsed:.1f} min elapsed)", flush=True)

    print(f"\n{'='*66}")
    print(f"  STUDENTS DONE — {len(checkpoint['cells'])} cells, "
          f"{(time.perf_counter() - t0)/60:.1f} min")
    print(f"  weights → {WEIGHTS_DIR}")
    print(f"{'='*66}\n", flush=True)
