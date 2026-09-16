"""
RQ4 analysis on the enlarged test set: KL-primary, bootstrap CIs, disagreement overlap.

WHAT CHANGED AND WHY
--------------------
The first RQ4 analysis (rq4_analysis.py) reported top-1 agreement as the primary fidelity
metric and binned each capacity ladder into a verdict. That could not be sustained:

1. On a 200-image test set one image is worth 0.005 agreement, and the whole interval
   between the no-teacher control and the inter-teacher reference level was 0.0030 (MNIST)
   and 0.0260 (Fashion) -- 0.6 and 5.2 images. The Fashion/QCNN verdict missed one threshold
   by 0.0010 and cleared the other by 0.0000.
2. Reporting agreement against the inter-teacher level alone omitted the control that makes
   it interpretable. A student that never sees the QCNN already agrees with it 0.9790 of the
   time on MNIST, so a raw 0.9863 is +0.0073, not a strong positive.

This script fixes all three problems at once:

* **KL is primary.** Predictive KL has no granularity floor and separated the arms by 1-3
  orders of magnitude where agreement separated them by fractions of a percent.
* **Bootstrap CIs over test images.** Seed SD alone describes only seed-to-seed variation and
  says nothing about how much of a difference is test-set sampling. Both are now reported.
* **Disagreement-set overlap.** Asks whether a student's residual disagreements with its
  teacher fall on the images where two independent teachers ALSO disagree. If they do, the
  residual is irreducible task ambiguity rather than a failure to absorb the function --
  which is the question the ladder was being asked to answer and could not.

NO VERDICT BINNING. rq4_analysis.py's ladder_verdict is deliberately not reproduced here.
Applying a threshold rule to a non-monotonic curve at a margin finer than the measurement
granularity is what produced the fragile result; the fix is to report the interval, not to
pick a rule that lands on the nicer side of it.

BOOTSTRAP DESIGN
----------------
Agreement and KL are both means over images, so a per-image vector can be resampled directly
and the result is exactly equal to recomputing the metric on the resampled set. For a cell
with 5 seeds, per-image values are first averaged across seeds, then image indices are
resampled; that yields a CI on the cell mean reflecting TEST-SET sampling with seeds held
fixed. Seed variation is reported separately as an SD -- the two uncertainties answer
different questions and are not pooled.

ECE is a binned statistic, not a mean, so it is recomputed per replicate rather than
resampled from a per-image vector.

USAGE
-----
    python experiments/rq4_reeval_analysis.py
    python experiments/rq4_reeval_analysis.py --bootstrap 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR  # noqa: E402
from utils.metrics import ece  # noqa: E402

RQ4_DIR = Path(RESULTS_DIR) / "rq4"
REEVAL_DIR = RQ4_DIR / "reeval"
DATASETS = ["mnist", "fashion_mnist"]
PUBLISHED_N = 200
SEEDS = [42, 43, 44, 45, 46]
PRETTY = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST"}


# ------------------------------------------------------------------ per-image primitives

def agree_vec(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Per-image agreement indicator. Mean of this is exactly top1_agreement(a, b)."""
    return (a.argmax(1) == b.argmax(1)).numpy().astype(np.float64)


def kl_vec(t: torch.Tensor, s: torch.Tensor) -> np.ndarray:
    """
    Per-image KL(P_teacher || P_student) in nats. Mean of this is exactly
    predictive_kl(t, s). Teacher-first, matching rq4_distillation.py.
    """
    log_p = F.log_softmax(t, dim=1)
    log_q = F.log_softmax(s, dim=1)
    return (log_p.exp() * (log_p - log_q)).sum(1).numpy().astype(np.float64)


def boot_ci(vals: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI on the mean of a per-image vector, resampling images with replacement."""
    n = len(vals)
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = vals[idx].mean(axis=1)
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


# ------------------------------------------------------------------ store access

def load_store(dataset: str) -> dict:
    p = REEVAL_DIR / f"{dataset}_logits.pt"
    if not p.exists():
        raise SystemExit(f"missing {p} — run experiments/rq4_reeval.py first")
    return torch.load(p, map_location="cpu", weights_only=False)


def teacher_key(kind: str, seed: int) -> str:
    return f"{kind}_teacher_seed{seed}.pt"


def student_index(store: dict) -> dict:
    """(rung, arm, n_train, alpha, T) -> {seed: key}. Keyed off meta, never the filename."""
    out: dict = defaultdict(dict)
    for key, m in store["meta"].items():
        if m.get("role") != "student":
            continue
        cfg = (m["rung"], m["arm"], m["n_train"], m["alpha"], m["T"])
        out[cfg][m["seed"]] = key
    return out


# ------------------------------------------------------------------ analyses

def teacher_block(store: dict, sl: slice, n_boot: int, rng) -> dict:
    """Teacher accuracy and the inter-teacher reference level, with image-bootstrap CIs."""
    y = store["y"][sl]
    out: dict = {}
    for kind in ("qcnn", "cnn"):
        accs, pair_vecs = [], []
        for s in SEEDS:
            lg = store["logits"][teacher_key(kind, s)][sl]
            accs.append((lg.argmax(1) == y).sum().item() / len(y))
        for i, a in enumerate(SEEDS):
            for b in SEEDS[i + 1:]:
                pair_vecs.append(
                    agree_vec(store["logits"][teacher_key(kind, a)][sl],
                              store["logits"][teacher_key(kind, b)][sl])
                )
        pair_mean = np.mean(pair_vecs, axis=0)
        lo, hi = boot_ci(pair_mean, n_boot, rng)
        out[kind] = {
            "acc_mean": float(np.mean(accs)),
            "acc_sd": float(np.std(accs)),
            "reference_level": float(pair_mean.mean()),
            "reference_ci": [lo, hi],
            "n_pairs": len(pair_vecs),
        }
    return out


def ladder_block(store: dict, sl: slice, n_boot: int, rng) -> dict:
    """
    Agreement and KL against the QCNN teacher, per capacity rung, for every arm.
    Reported with an image-bootstrap CI and a seed SD, and against the hard-label control.
    """
    y = store["y"][sl]
    sidx = student_index(store)
    rows: dict = defaultdict(dict)

    for (rung, arm, n_train, alpha, T), by_seed in sidx.items():
        if n_train != 500 or T != 4.0:
            continue
        if arm != "hard" and alpha != 1.0:
            continue
        # T == 4.0 is not cosmetic. Rung 1 carries a temperature sweep (T = 2, 4, 8) that
        # rungs 2-5 do not, so averaging over T silently makes rung 1 a mean of three
        # configurations while every other rung is one -- the rung-1 point then moves for a
        # reason unrelated to capacity. rq4_analysis.py's `ladder` pins T=4.0 for exactly
        # this reason; omitting it here shifted rung 1 by +0.0020 (MNIST) and +0.0040
        # (Fashion) against the published numbers, and nothing else.
        ag, kl, accs, eces = [], [], [], []
        for s in SEEDS:
            if s not in by_seed:
                continue
            st = store["logits"][by_seed[s]][sl]
            tq = store["logits"][teacher_key("qcnn", s)][sl]
            ag.append(agree_vec(st, tq))
            kl.append(kl_vec(tq, st))
            accs.append((st.argmax(1) == y).sum().item() / len(y))
            eces.append(ece(st, y))
        if not ag:
            continue
        ag_m = np.mean(ag, axis=0)
        kl_m = np.mean(kl, axis=0)
        ag_lo, ag_hi = boot_ci(ag_m, n_boot, rng)
        kl_lo, kl_hi = boot_ci(kl_m, n_boot, rng)
        rows[rung][arm] = {
            "n_params": store["meta"][by_seed[SEEDS[0]]]["n_params"],
            "agree": float(ag_m.mean()), "agree_ci": [ag_lo, ag_hi],
            "agree_seed_sd": float(np.std([v.mean() for v in ag])),
            "kl": float(kl_m.mean()), "kl_ci": [kl_lo, kl_hi],
            "kl_seed_sd": float(np.std([v.mean() for v in kl])),
            "acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
            "ece": float(np.mean(eces)), "ece_sd": float(np.std(eces)),
            "n_seeds": len(ag),
        }
    return {str(k): v for k, v in sorted(rows.items())}


def disagreement_block(store: dict, sl: slice) -> dict:
    """
    Do a student's residual disagreements with its teacher land where teachers disagree
    with each other?

    For student seed s against teacher seed s, D_st is the set of test images where they
    differ. The reference set D_tt is where teacher s differs from another teacher s'.
    `lift` = P(i in D_tt | i in D_st) / P(i in D_tt). Lift near 1 means the student's
    residual is spread like any random subset -- no evidence the leftovers are special.
    Lift well above 1 means the residual concentrates on images the teachers themselves
    find contested, i.e. it is irreducible ambiguity rather than failure to absorb.

    Also reported: what fraction of the student's disagreements sit on images the teacher
    itself gets wrong, since copying a teacher's mistakes and failing to copy its successes
    are different failures.
    """
    y = store["y"][sl]
    sidx = student_index(store)
    out: dict = {}

    for (rung, arm, n_train, alpha, T), by_seed in sidx.items():
        if n_train != 500 or arm != "qkd" or alpha != 1.0 or T != 4.0:
            continue  # pin T for the same reason as ladder_block
        lifts, in_dtt, base, on_wrong = [], [], [], []
        for s in SEEDS:
            if s not in by_seed:
                continue
            st = store["logits"][by_seed[s]][sl].argmax(1)
            tq = store["logits"][teacher_key("qcnn", s)][sl].argmax(1)
            d_st = (st != tq).numpy()
            if d_st.sum() == 0:
                continue
            others = [store["logits"][teacher_key("qcnn", o)][sl].argmax(1)
                      for o in SEEDS if o != s]
            d_tt = np.mean([(tq != o).numpy().astype(np.float64) for o in others], axis=0)
            p_cond = float(d_tt[d_st].mean())
            p_base = float(d_tt.mean())
            if p_base > 0:
                lifts.append(p_cond / p_base)
            in_dtt.append(p_cond)
            base.append(p_base)
            t_wrong = (tq != y).numpy()
            on_wrong.append(float(t_wrong[d_st].mean()))
        if not in_dtt:
            continue
        out[str(rung)] = {
            "lift_mean": float(np.mean(lifts)) if lifts else None,
            "lift_sd": float(np.std(lifts)) if lifts else None,
            "p_contested_given_disagree": float(np.mean(in_dtt)),
            "p_contested_base": float(np.mean(base)),
            "frac_on_teacher_errors": float(np.mean(on_wrong)),
            "n_seeds": len(in_dtt),
        }
    return out


# ------------------------------------------------------------------ report

def report(res: dict) -> str:
    L = []
    w = L.append
    w("=" * 78)
    w("RQ4 — ENLARGED TEST SET: KL-primary, bootstrap CIs, disagreement overlap")
    w("=" * 78)
    w("")
    for ds in DATASETS:
        if ds not in res:
            continue
        r = res[ds]
        w(f"### {PRETTY[ds]}  (n_test {r['n_published']} -> {r['n_full']}, "
          f"{r['n_full'] / r['n_published']:.1f}x)")
        w("")
        t = r["full"]["teachers"]
        for kind in ("qcnn", "cnn"):
            b = t[kind]
            w(f"  {kind:5s} accuracy {b['acc_mean']:.4f} ± {b['acc_sd']:.4f}   "
              f"inter-teacher reference {b['reference_level']:.4f} "
              f"[{b['reference_ci'][0]:.4f}, {b['reference_ci'][1]:.4f}]")
        w("")
        w("  CAPACITY LADDER — agreement with the QCNN teacher, and KL")
        w(f"  {'rung':>4} {'params':>7} {'agree(QKD)':>22} {'ctrl':>8} {'excess':>8} "
          f"{'KL(QKD)':>20} {'KL ctrl':>9}")
        lad = r["full"]["ladder"]
        for rung in sorted(lad, key=int):
            q = lad[rung].get("qkd")
            h = lad[rung].get("hard")
            if not q or not h:
                continue
            w(f"  {rung:>4} {q['n_params']:>7} "
              f"{q['agree']:.4f} [{q['agree_ci'][0]:.4f},{q['agree_ci'][1]:.4f}] "
              f"{h['agree']:>8.4f} {q['agree'] - h['agree']:>+8.4f} "
              f"{q['kl']:.4f} [{q['kl_ci'][0]:.4f},{q['kl_ci'][1]:.4f}] "
              f"{h['kl']:>9.4f}")
        w("")
        ref = t["qcnn"]["reference_level"]
        ctrl = lad[sorted(lad, key=int)[0]]["hard"]["agree"]
        span = ref - ctrl
        # The n=200 comparison must use the span AS MEASURED at n=200, not the full-set
        # span rescaled. They are different numbers (0.0030 vs 0.0061 on MNIST) and the
        # rescaled version silently overstates how much room the old test set had.
        pt = r["published"]["teachers"]["qcnn"]["reference_level"]
        pc = r["published"]["ladder"][sorted(r["published"]["ladder"], key=int)[0]]["hard"]["agree"]
        pspan = pt - pc
        w(f"  Usable range of top-1 agreement: reference {ref:.4f} − control {ctrl:.4f} "
          f"= {span:.4f}")
        w(f"    = {span * r['n_full']:.1f} test images at n={r['n_full']}; "
          f"as measured at n={r['n_published']} it was {pspan:.4f} "
          f"= {pspan * r['n_published']:.1f} images")
        w("")
        w("  DISAGREEMENT OVERLAP — do the student's residual disagreements land where")
        w("  two independent teachers also disagree?")
        w(f"  {'rung':>4} {'lift':>14} {'P(contested|disagree)':>22} {'base':>8} "
          f"{'on teacher errors':>18}")
        for rung, d in sorted(r["full"]["disagreement"].items(), key=lambda kv: int(kv[0])):
            lift = f"{d['lift_mean']:.2f} ± {d['lift_sd']:.2f}" if d["lift_mean"] else "n/a"
            w(f"  {rung:>4} {lift:>14} {d['p_contested_given_disagree']:>22.4f} "
              f"{d['p_contested_base']:>8.4f} {d['frac_on_teacher_errors']:>18.4f}")
        w("")
    w("-" * 78)
    w("No verdict binning. A threshold rule on a non-monotonic curve at a margin finer than")
    w("the measurement granularity is what made the first analysis fragile; the interval is")
    w("reported instead. CIs are over TEST IMAGES with seeds fixed; seed SD is separate.")
    return "\n".join(L)


# ------------------------------------------------------------------ figures

# Palette copied from rq4_analysis.py rather than imported: that module runs argparse and
# reads results at import time, so importing it here would execute a second analysis.
C_BLUE, C_ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ARM_COLOR = {"hard": MUTED, "qkd": C_BLUE, "ckd": C_ORANGE}
ARM_LABEL = {"hard": "Hard labels (no teacher)", "qkd": "QCNN teacher", "ckd": "CNN teacher"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 200,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def make_figures(res: dict) -> list[str]:
    plt = _style()
    fig_dir = RQ4_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # ---- fig5: ladder at the enlarged test size -------------------------------------
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(11.5, 7.2))
    for col, ds in enumerate(DATASETS):
        lad = res[ds]["full"]["ladder"]
        rungs = sorted(lad, key=int)
        x = [lad[r]["qkd"]["n_params"] for r in rungs]

        ax = axes[0, col]
        for arm in ("qkd", "ckd", "hard"):
            y = [lad[r][arm]["agree"] for r in rungs if arm in lad[r]]
            lo = [lad[r][arm]["agree_ci"][0] for r in rungs if arm in lad[r]]
            hi = [lad[r][arm]["agree_ci"][1] for r in rungs if arm in lad[r]]
            if not y:
                continue
            ax.plot(x, y, color=ARM_COLOR[arm], marker="o", lw=1.8, ms=5,
                    label=ARM_LABEL[arm])
            ax.fill_between(x, lo, hi, color=ARM_COLOR[arm], alpha=0.15, lw=0)
        t = res[ds]["full"]["teachers"]["qcnn"]
        ax.axhline(t["reference_level"], color=C_BLUE, ls=":", lw=1.4)
        ax.fill_between([min(x) * 0.8, max(x) * 1.25], t["reference_ci"][0],
                        t["reference_ci"][1], color=C_BLUE, alpha=0.08, lw=0)
        ax.text(max(x) * 1.2, t["reference_level"], "QCNN inter-teacher level",
                ha="right", va="bottom", fontsize=8, color=C_BLUE)
        ax.set_xscale("log")
        ax.set_xlim(min(x) * 0.8, max(x) * 1.25)   # fig2's defect: never leave this to auto
        ax.set_title(f"{PRETTY[ds]}  (n_test = {res[ds]['n_full']})", fontsize=11, color=INK)
        if col == 0:
            ax.set_ylabel("top-1 agreement with QCNN teacher")
            ax.legend(frameon=False, fontsize=8, loc="lower right")

        ax = axes[1, col]
        for arm in ("qkd", "ckd", "hard"):
            y = [lad[r][arm]["kl"] for r in rungs if arm in lad[r]]
            lo = [lad[r][arm]["kl_ci"][0] for r in rungs if arm in lad[r]]
            hi = [lad[r][arm]["kl_ci"][1] for r in rungs if arm in lad[r]]
            if not y:
                continue
            ax.plot(x, y, color=ARM_COLOR[arm], marker="o", lw=1.8, ms=5)
            ax.fill_between(x, lo, hi, color=ARM_COLOR[arm], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(x) * 0.8, max(x) * 1.25)
        ax.set_xlabel("student trainable parameters")
        if col == 0:
            ax.set_ylabel("KL(teacher $\\Vert$ student), nats")
    fig.suptitle("Agreement saturates; KL separates the arms by orders of magnitude",
                 fontsize=12, color=INK, y=0.98)
    p = fig_dir / "fig5_ladder_enlarged.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p.name)

    # ---- fig6: disagreement lift ------------------------------------------------------
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.2), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        dis = res[ds]["full"]["disagreement"]
        lad = res[ds]["full"]["ladder"]
        rungs = sorted(dis, key=int)
        x = [lad[r]["qkd"]["n_params"] for r in rungs]
        y = [dis[r]["lift_mean"] for r in rungs]
        e = [dis[r]["lift_sd"] for r in rungs]
        ax.errorbar(x, y, yerr=e, color=C_BLUE, marker="o", lw=1.8, ms=5, capsize=3)
        ax.axhline(1.0, color=INK2, ls="--", lw=1.2)
        ax.text(max(x) * 1.2, 1.0, "no concentration", ha="right", va="bottom",
                fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(min(x) * 0.8, max(x) * 1.25)
        ax.set_xlabel("student trainable parameters")
        ax.set_title(PRETTY[ds], fontsize=11, color=INK)
    axes[0].set_ylabel("lift: student\u2013teacher disagreements\nfalling on contested images")
    fig.suptitle("Residual disagreement concentrates where teachers disagree with each other",
                 fontsize=12, color=INK, y=1.02)
    p = fig_dir / "fig6_disagreement_lift.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p.name)

    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    res: dict = {}
    for ds in DATASETS:
        store = load_store(ds)
        n_full = store["n_test"]
        res[ds] = {"n_full": n_full, "n_published": PUBLISHED_N}
        for label, sl in (("published", slice(0, PUBLISHED_N)), ("full", slice(0, n_full))):
            res[ds][label] = {
                "teachers": teacher_block(store, sl, args.bootstrap, rng),
                "ladder": ladder_block(store, sl, args.bootstrap, rng),
                "disagreement": disagreement_block(store, sl),
            }
        print(f"  {ds}: analysed at n={PUBLISHED_N} and n={n_full}")

    out_json = RQ4_DIR / "reeval_analysis.json"
    out_txt = RQ4_DIR / "reeval_analysis_report.txt"
    out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    txt = report(res)
    out_txt.write_text(txt, encoding="utf-8")
    print()
    print(txt)
    figs = make_figures(res)
    print(f"\nsaved -> {out_json}, {out_txt}")
    print("figures:", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
