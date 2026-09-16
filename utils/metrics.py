"""
Result serialisation and summary statistics across seeds.
"""
import json
import numpy as np
from pathlib import Path


def save_results(results: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {path}")


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def summarise_seeds(seed_histories: list[dict], key: str = "test_acc") -> dict:
    """
    Given a list of per-seed history dicts, return mean ± std of the final-epoch metric.
    """
    finals = [h[key][-1] for h in seed_histories]
    return {
        "mean": float(np.mean(finals)),
        "std":  float(np.std(finals)),
        "runs": finals,
    }


def print_summary(label: str, summary: dict):
    print(f"{label:40s}  {summary['mean']:.4f} ± {summary['std']:.4f}  "
          f"(runs: {[f'{v:.4f}' for v in summary['runs']]})")


# ── RQ4: fidelity and calibration ────────────────────────────────────────────
# Appended 2026-08-24. Deliberately additive — RQ1/RQ2/RQ3 import the functions
# above and none of them are altered here.

import torch
import torch.nn.functional as F


def top1_agreement(a_logits: torch.Tensor, b_logits: torch.Tensor) -> float:
    """
    Fraction of samples on which two models predict the same class.

    RQ4's PRIMARY metric. Note what it is not: it says nothing about either model being
    right. A student that reproduces its teacher's mistakes scores 1.0, which is the
    point — the question is whether the student absorbed the teacher's *function*, not
    whether that function is good.
    """
    return float((a_logits.argmax(1) == b_logits.argmax(1)).float().mean().item())


def predictive_kl(a_logits: torch.Tensor, b_logits: torch.Tensor) -> float:
    """
    Mean KL(P_a ‖ P_b) over the predictive distributions, in nats.

    Complements top1_agreement, which is blind to confidence: two models can agree on
    every argmax while one is at 0.51 and the other at 0.99. Asymmetric — call it as
    predictive_kl(teacher, student) and keep the order fixed across a comparison.
    """
    log_p = F.log_softmax(a_logits, dim=1)
    log_q = F.log_softmax(b_logits, dim=1)
    return float((log_p.exp() * (log_p - log_q)).sum(1).mean().item())


def ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Expected calibration error with equal-width confidence bins.

    Follows Guo et al. 2017: |accuracy - confidence| per bin, weighted by bin
    population. Included because Abnar et al. 2020 (RQ4's methodological template)
    measure inductive-bias transfer by calibration and shift robustness rather than
    clean accuracy — this is the citable escape from the weak-teacher problem.
    """
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    correct = pred.eq(labels).float()

    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    n = labels.numel()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Lower-exclusive/upper-inclusive so the top bin captures conf == 1.0 exactly.
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        total += (count / n) * abs(correct[in_bin].mean().item() - conf[in_bin].mean().item())
    return float(total)
