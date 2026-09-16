"""
RQ4 re-evaluation on an enlarged test set.

WHY THIS EXISTS
---------------
RQ4's headline metric — top-1 agreement between a student and its teacher — turned out to
be unresolvable at the test-set size the rest of the thesis uses. With n_test = 200, one
image is worth 0.005 of agreement and one seed of five moves a cell mean by 0.001. The
interval between the no-teacher control and the inter-teacher reference level is 0.0030 on
MNIST and 0.0260 on Fashion-MNIST — i.e. the entire informative range of the metric is
0.6 and 5.2 test images wide. The capacity-ladder verdict flipped on a single image.

This script removes that limitation the cheap way. Decision D5 means RQ4 saved every trained
model, so the fix is forward passes over more data, NOT retraining: MNIST has 2115 binary
test images available against the 200 used, Fashion-MNIST 2000. That is a ~10x improvement
in granularity for ~27 min of QCNN inference and zero minutes of training.

THE PREFIX PROPERTY IS LOAD-BEARING (verified 2026-08-24)
--------------------------------------------------------
`_subsample_idx` draws one permutation seeded with `seed` and returns perm[:n]. Passing the
FULL count explicitly therefore yields a superset whose first 200 rows are exactly the
published test set — checked directly, X and y both identical. Passing n_test=None does NOT
have this property: it returns the indices unpermuted, so its first 200 rows are a different
set of images. Always pass the integer.

That property is what makes this a strict improvement rather than a different experiment:
every published 200-image number must be reproducible from the first-200 slice of the
enlarged predictions, and `--verify` checks exactly that.

WHAT IT SAVES
-------------
Per-image LOGITS for every one of the 390 models, not summary metrics. Everything
downstream — agreement, KL, ECE, bootstrap confidence intervals over test images, and the
teacher/student disagreement-set overlap — is computed from these. Saving summaries instead
would force another 27-minute pass every time a new question is asked, which is the mistake
that made RQ4 retrain RQ1's teachers in the first place.

USAGE
-----
    python experiments/rq4_reeval.py                # run both datasets
    python experiments/rq4_reeval.py --dataset mnist
    python experiments/rq4_reeval.py --verify       # check against published numbers only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import N_CLASSES, RESULTS_DIR  # noqa: E402
from models.classical_cnn import ClassicalCNN  # noqa: E402
from models.qcnn import QCNNModel  # noqa: E402
from utils.dataset import get_tensors  # noqa: E402
from utils.distill import collect_logits  # noqa: E402

RQ4_DIR = Path(RESULTS_DIR) / "rq4"
WEIGHTS_DIR = RQ4_DIR / "weights"
OUT_DIR = RQ4_DIR / "reeval"

# The full binary test-set size per dataset. Hardcoded deliberately rather than derived at
# runtime: these are the numbers the write-up quotes, and a silent change to the dataset
# filter that altered them should break this script loudly rather than shift the results.
FULL_TEST = {"mnist": 2115, "fashion_mnist": 2000}

PUBLISHED_N_TEST = 200  # what RQ1-RQ4 reported; the prefix slice must reproduce it


def rebuild(payload: dict):
    """Reconstruct a model from a save_state payload. Meta is self-describing by design."""
    m = payload["meta"]
    cls = payload["model_class"]
    if cls == "QCNNModel":
        model = QCNNModel(
            n_qubits=m["n_qubits"],
            n_layers=m["n_layers"],
            ansatz=m["ansatz"],
            encoding=m["encoding"],
            n_classes=N_CLASSES,
        )
    elif cls == "ClassicalCNN":
        model = ClassicalCNN(
            n_filters=m["n_filters"], hidden=m["hidden"], image_size=8, n_classes=N_CLASSES
        )
    else:
        raise ValueError(f"unknown model_class {cls!r}")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def weight_files(dataset: str) -> list[Path]:
    d = WEIGHTS_DIR / dataset
    teachers = sorted(d.glob("*.pt"))
    students = sorted((d / "students").glob("*.pt"))
    return teachers + students


def run_dataset(dataset: str, resume: bool = True) -> Path:
    n_test = FULL_TEST[dataset]
    _, _, X, y = get_tensors(dataset, n_train=500, n_test=n_test)
    assert len(y) == n_test, f"expected {n_test} test rows, got {len(y)}"

    out_path = OUT_DIR / f"{dataset}_logits.pt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    store: dict = {"dataset": dataset, "n_test": n_test, "y": y, "logits": {}, "meta": {}}
    if resume and out_path.exists():
        store = torch.load(out_path, map_location="cpu", weights_only=False)
        print(f"  resuming — {len(store['logits'])} models already done")

    files = weight_files(dataset)
    print(f"  {dataset}: {len(files)} models, {n_test} test images")

    t_start = time.time()
    for i, f in enumerate(files, 1):
        key = f.relative_to(WEIGHTS_DIR / dataset).as_posix()
        if key in store["logits"]:
            continue
        payload = torch.load(f, map_location="cpu", weights_only=False)
        model = rebuild(payload)
        t0 = time.time()
        logits = collect_logits(model, X)
        store["logits"][key] = logits
        store["meta"][key] = payload["meta"]
        dt = time.time() - t0
        if dt > 5 or i % 50 == 0:
            acc = (logits.argmax(1) == y).sum().item() / len(y)
            print(f"    [{i}/{len(files)}] {key}  {dt:5.1f}s  acc(full)={acc:.4f}", flush=True)
            torch.save(store, out_path)  # checkpoint after anything slow

    torch.save(store, out_path)
    print(f"  {dataset} done in {(time.time() - t_start) / 60:.1f} min -> {out_path}")
    return out_path


def verify(dataset: str) -> bool:
    """
    Gate: the first PUBLISHED_N_TEST rows must reproduce the accuracies already recorded in
    the checkpoints. If this fails, the enlarged set is not a superset of the published one
    and nothing downstream can be compared against the existing chapters.
    """
    out_path = OUT_DIR / f"{dataset}_logits.pt"
    store = torch.load(out_path, map_location="cpu", weights_only=False)
    y200 = store["y"][:PUBLISHED_N_TEST]

    # independent check that the prefix really is the published test set
    _, _, X200, y200_ref = get_tensors(dataset, n_train=500, n_test=PUBLISHED_N_TEST)
    assert torch.equal(y200, y200_ref), "prefix labels differ from the published test set"

    bad = []
    for key, logits in store["logits"].items():
        recorded = store["meta"][key].get("final_test_acc")
        if recorded is None:
            continue
        # Integer ratio in float64, NOT .float().mean(): a float32 mean of 194/200 gives
        # 0.9700000286, which differs from the JSON-recorded 0.97 by ~3e-8 and would fail an
        # exact check for a reason that has nothing to do with the model.
        got = (logits[:PUBLISHED_N_TEST].argmax(1) == y200).sum().item() / PUBLISHED_N_TEST
        if abs(got - recorded) > 1e-9:
            bad.append((key, recorded, got))

    n = len(store["logits"])
    if bad:
        print(f"  {dataset}: {len(bad)}/{n} MISMATCH against recorded accuracies")
        for k, r, g in bad[:10]:
            print(f"     {k}: recorded {r:.4f}  prefix {g:.4f}")
        return False
    print(f"  {dataset}: all {n} models reproduce their recorded 200-image accuracy exactly")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(FULL_TEST), default=None)
    ap.add_argument("--verify", action="store_true", help="verify only, do not re-evaluate")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else list(FULL_TEST)

    if not args.verify:
        print("=== re-evaluating on the enlarged test set ===")
        for ds in datasets:
            run_dataset(ds, resume=not args.no_resume)

    print("\n=== verification gate: prefix must reproduce published numbers ===")
    ok = all(verify(ds) for ds in datasets)
    print("\nGATE PASSED" if ok else "\nGATE FAILED — do not use these results")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
