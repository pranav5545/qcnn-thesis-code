"""
RQ4 analysis — knowledge distillation as a simulability probe.

Reads what experiments/rq4_teachers.py and experiments/rq4_distillation.py produce and
emits `analysis.json`, `analysis_report.txt` and four figures, following the structure of
experiments/rq2_analysis.py.

THE QUESTION AND HOW THIS SCRIPT ANSWERS IT. RQ1 gave parity of *accuracy*; RQ4 asks for
parity of *function*. The primary metric is top-1 agreement between a student and its
teacher, and the number that makes it interpretable is the **teacher-teacher agreement
ceiling**: two QCNN teachers trained on the same data with different seeds do not agree
perfectly with each other, so a student cannot be expected to beat that. Every fidelity
number is read against it.

THE READING RULE THAT DECIDES WHAT RQ4 MAY CLAIM (verification gate 4 of the
implementation plan). If agreement is flat across the whole capacity ladder, that is
Stanton et al.'s optimisation regime — the student may be able to represent the teacher and
simply fail to find it — and **no simulability conclusion may be drawn**. The report prints
this verdict explicitly rather than leaving it to the reader, because the tempting
misreading (flat = not simulable) is the exact error the ladder exists to prevent.

PARTIAL DATA IS SUPPORTED ON PURPOSE. The teacher-side quantities — including the ceiling —
depend only on stage 1, so they can be computed while stage 2 is still running. Sections
with no data are reported as pending rather than crashing.

Usage:
    python experiments/rq4_analysis.py
    python experiments/rq4_analysis.py --results-dir results/rq4
"""
import sys, json, argparse, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.metrics import top1_agreement, predictive_kl
from utils.corruptions import CORRUPTION_NAMES

DATASETS = ["mnist", "fashion_mnist"]
DS_LABEL = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST"}

# ---------------------------------------------------------------- palette
# Colour means TEACHER here, and nothing else. Blue has meant "the quantum model" since
# RQ1/RQ3 (blue->qcnn_native), so blue stays with the QCNN teacher; orange is the classical
# control teacher, carried over from RQ2's slot-2; the hard-label control is deliberately
# neutral grey because it has no teacher at all and colouring it would imply one.
C_BLUE = "#2a78d6"     # quantum teacher (qkd)
C_ORANGE = "#eb6834"   # classical control teacher (ckd)
C_AQUA = "#1baf7a"     # reserved: alpha/T variants where a third series is needed
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"      # hard-label control
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

ARM_COLOR = {"hard": MUTED, "qkd": C_BLUE, "ckd": C_ORANGE}
ARM_LABEL = {"hard": "Hard labels (no teacher)", "qkd": "QCNN teacher", "ckd": "CNN teacher"}
ALPHA_STYLE = {0.9: "--", 1.0: "-", 0.0: "-"}
ALPHA_MARKER = {0.9: "s", 1.0: "o", 0.0: "^"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---------------------------------------------------------------- args
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default="results/rq4")
args = parser.parse_args()

RQ4 = Path(args.results_dir)
FIG_DIR = RQ4 / "figures"


# ---------------------------------------------------------------- loading

def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_teacher_logits():
    """{(dataset, seed): payload} for every logit cache present."""
    out = {}
    for p in sorted((RQ4 / "logits").glob("*_seed*.pt")):
        try:
            d = torch.load(p, weights_only=False)
        except Exception as e:
            print(f"  ! could not read {p.name}: {type(e).__name__}")
            continue
        out[(d["dataset"], int(d["seed"]))] = d
    return out


# ---------------------------------------------------------------- statistics

def paired_contrast(a, b, label_a, label_b):
    """Paired comparison of two seed-matched arrays. Mirrors rq2_analysis.py's treatment.

    Wilcoxon is reported but cannot reach p<0.05 at n=5 (its exact two-sided floor is
    2/2^5 = 0.0625), so the t-test and Cohen's d_z carry the quantitative story — the same
    documented stance as RQ2.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    out = {"label_a": label_a, "label_b": label_b, "n": n,
           "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": float(d.mean())}
    if n < 2 or np.allclose(d, 0):
        out.update({"t": None, "p_t": None, "d_z": 0.0 if np.allclose(d, 0) else None,
                    "note": "zero variance in differences" if np.allclose(d, 0) else "n<2"})
        return out
    sd = d.std(ddof=1)
    out["t"] = float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else None
    out["d_z"] = float(d.mean() / sd) if sd > 0 else None
    try:
        from scipy import stats
        out["p_t"] = float(stats.ttest_rel(a, b).pvalue)
        out["p_wilcoxon"] = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        # scipy is not a declared dependency of this project's env; the t statistic and
        # d_z above are computed directly and remain valid without it.
        out["p_t"] = None
        out["p_wilcoxon"] = None
        out["note"] = "scipy unavailable — p-values omitted, t and d_z are exact"
    return out


def teacher_ceiling(logits, dataset, kind):
    """
    Agreement between two independently-seeded teachers of the same kind.

    THIS IS THE REFERENCE LINE FOR EVERY FIDELITY NUMBER. A student cannot reasonably be
    asked to agree with its teacher more than two teachers agree with each other, so
    "student agreement = 0.93" means nothing until this is known. Computed over all
    unordered seed pairs on the clean test set.
    """
    seeds = sorted(s for (ds, s) in logits if ds == dataset)
    pairs = []
    for i, j in itertools.combinations(seeds, 2):
        a = logits[(dataset, i)][kind]["test"]
        b = logits[(dataset, j)][kind]["test"]
        pairs.append({"seeds": [i, j], "agreement": top1_agreement(a, b),
                      "kl": predictive_kl(a, b)})
    if not pairs:
        return None
    vals = [p["agreement"] for p in pairs]
    return {"kind": kind, "dataset": dataset, "n_pairs": len(pairs),
            "seeds": seeds,
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals)),
            "pairs": pairs}


def teacher_accuracy(teachers, dataset, kind):
    hist = (teachers.get("histories") or {}).get(dataset, {})
    accs = [h[kind]["test_acc"][-1] for h in hist.values() if kind in h]
    if not accs:
        return None
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "n": len(accs), "runs": accs}


# ---------------------------------------------------------------- student aggregation

def cells_frame(students):
    """Flatten the checkpoint's cells into plain dicts, one per (config, seed)."""
    if not students:
        return []
    return list((students.get("cells") or {}).values())


def group(cells, **where):
    """Cells matching every key=value in `where`."""
    out = []
    for c in cells:
        if all(c.get(k) == v for k, v in where.items()):
            out.append(c)
    return out


def by_seed(cells, metric_path):
    """Seed-ordered values of a metric, so contrasts stay paired by seed."""
    got = {}
    for c in cells:
        v = c["metrics"]
        for part in metric_path:
            v = v[part]
        got[c["seed"]] = v
    return [got[s] for s in sorted(got)]


def ladder(cells, dataset, arm, alpha, metric_path, rungs):
    """Mean/std of a metric per rung, for one arm at the default T and full n_train."""
    rows = []
    for rung, n_params in rungs:
        sel = group(cells, dataset=dataset, arm=arm, alpha=alpha, rung=rung,
                    T=4.0, n_train=500)
        if not sel:
            continue
        vals = by_seed(sel, metric_path)
        rows.append({"rung": rung, "n_params": n_params, "n_seeds": len(vals),
                     "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                     "runs": vals})
    return rows


def rung_params(cells):
    """[(rung, n_params)] present in the data, ascending."""
    seen = {}
    for c in cells:
        seen[c["rung"]] = c["n_params"]
    return sorted(seen.items())


def ladder_verdict(rows, ceiling):
    """
    Verification gate 4. Decides, and says out loud, whether a simulability reading is
    permitted by the shape of the ladder.
    """
    if len(rows) < 3:
        return {"verdict": "PENDING", "reason": "fewer than three rungs present"}
    lo, hi = rows[0]["mean"], rows[-1]["mean"]
    rise = hi - lo
    spread = max(r["mean"] for r in rows) - min(r["mean"] for r in rows)
    v = {"lowest_rung_mean": lo, "highest_rung_mean": hi, "rise": float(rise),
         "spread": float(spread), "ceiling": ceiling}
    if ceiling is not None and hi >= ceiling - 0.01:
        v["verdict"] = "REACHES CEILING"
        v["reason"] = ("the top rung's agreement is within 1 point of the teacher-teacher "
                       "ceiling: the student absorbs the teacher's function about as well "
                       "as another teacher does. Supports the simulability reading.")
    elif rise >= 0.02:
        v["verdict"] = "RISES"
        v["reason"] = ("agreement increases with student capacity without reaching the "
                       "ceiling: capacity is a binding constraint, and the gap at the top "
                       "rung is the quantity to report.")
    else:
        v["verdict"] = "FLAT — INCONCLUSIVE"
        v["reason"] = ("agreement does not improve with capacity. Per Stanton et al. this "
                       "is the OPTIMISATION regime: the student may be able to represent "
                       "the teacher and simply fail to find it. NO SIMULABILITY CONCLUSION "
                       "MAY BE DRAWN — report as inconclusive, not as evidence of "
                       "non-simulability.")
    return v


# ---------------------------------------------------------------- figures

def fig1_capacity_ladder(analysis, cells, rungs):
    """THE HEADLINE FIGURE: fidelity vs student capacity, against the ceiling."""
    if not cells:
        return False
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, DATASETS):
        drew = False
        for arm, tkey, alpha in [("qkd", "qcnn", 1.0), ("ckd", "cnn", 1.0),
                                 ("hard", "qcnn", 0.0)]:
            rows = ladder(cells, ds, arm, alpha, ["agree_" + tkey], rungs)
            if not rows:
                continue
            drew = True
            x = [r["n_params"] for r in rows]
            y = [r["mean"] for r in rows]
            e = [r["std"] for r in rows]
            ax.errorbar(x, y, yerr=e, color=ARM_COLOR[arm], marker=ALPHA_MARKER[alpha],
                        ls=ALPHA_STYLE[alpha], lw=1.8, ms=6, capsize=3,
                        label=f"{ARM_LABEL[arm]}" + (f" (α={alpha})" if arm != "hard" else ""))
        ceil = (analysis["teacher_ceiling"].get(ds, {}) or {}).get("qcnn")
        if ceil:
            ax.axhline(ceil["mean"], color=C_BLUE, ls=":", lw=1.4, alpha=0.85)
            ax.text(0.99, ceil["mean"], " QCNN teacher↔teacher ceiling",
                    transform=matplotlib.transforms.blended_transform_factory(
                        ax.transAxes, ax.transData),
                    ha="right", va="bottom", fontsize=8, color=C_BLUE)
        ax.set_xscale("log")
        ax.set_xlabel("student trainable parameters")
        ax.set_title(DS_LABEL[ds], fontsize=11, color=INK)
        if not drew:
            ax.text(0.5, 0.5, "no student data yet", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
    axes[0].set_ylabel("top-1 agreement with teacher")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Can a parameter-matched classical student absorb the QCNN's function?",
                 fontsize=12, color=INK, y=1.02)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig1_capacity_ladder.png", bbox_inches="tight")
    plt.close(fig)
    return True


def fig2_accuracy_ece(cells, rungs):
    if not cells:
        return False
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(11, 7), sharex=True)
    for col, ds in enumerate(DATASETS):
        for row, (metric, label) in enumerate([(["test_acc"], "test accuracy"),
                                               (["ece"], "ECE (15 bins)")]):
            ax = axes[row, col]
            for arm, alpha in [("hard", 0.0), ("qkd", 1.0), ("ckd", 1.0)]:
                rows = ladder(cells, ds, arm, alpha, metric, rungs)
                if not rows:
                    continue
                ax.errorbar([r["n_params"] for r in rows], [r["mean"] for r in rows],
                            yerr=[r["std"] for r in rows], color=ARM_COLOR[arm],
                            marker=ALPHA_MARKER[alpha], ls=ALPHA_STYLE[alpha],
                            lw=1.6, ms=5, capsize=3, label=ARM_LABEL[arm])
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(DS_LABEL[ds], fontsize=11, color=INK)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("student trainable parameters")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "fig2_accuracy_ece.png", bbox_inches="tight")
    plt.close(fig)
    return True


def fig3_robustness(cells, rungs):
    """Accuracy under each corruption, base rung only, by arm."""
    if not cells:
        return False
    base = rungs[0][0] if rungs else 1
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    xs = np.arange(len(CORRUPTION_NAMES))
    for ax, ds in zip(axes, DATASETS):
        for off, (arm, alpha) in enumerate([("hard", 0.0), ("qkd", 1.0), ("ckd", 1.0)]):
            sel = group(cells, dataset=ds, arm=arm, alpha=alpha, rung=base,
                        T=4.0, n_train=500)
            if not sel:
                continue
            means, stds = [], []
            for name in CORRUPTION_NAMES:
                vals = by_seed(sel, ["corrupt", name, "acc"])
                means.append(np.mean(vals)); stds.append(np.std(vals))
            ax.errorbar(xs + (off - 1) * 0.12, means, yerr=stds, color=ARM_COLOR[arm],
                        marker=ALPHA_MARKER[alpha], ls="none", ms=5, capsize=3,
                        label=ARM_LABEL[arm])
        ax.set_xticks(xs)
        ax.set_xticklabels(CORRUPTION_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_title(DS_LABEL[ds], fontsize=11, color=INK)
    axes[0].set_ylabel("accuracy under corruption")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "fig3_robustness.png", bbox_inches="tight")
    plt.close(fig)
    return True


def fig4_fidelity_vs_accuracy(cells):
    """Stanton decoupling: fidelity and accuracy need not move together."""
    if not cells:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for arm, tkey in [("qkd", "qcnn"), ("ckd", "cnn"), ("hard", "qcnn")]:
        sel = [c for c in cells if c["arm"] == arm and c["T"] == 4.0 and c["n_train"] == 500]
        if not sel:
            continue
        ax.scatter([c["metrics"]["agree_" + tkey] for c in sel],
                   [c["metrics"]["test_acc"] for c in sel],
                   c=ARM_COLOR[arm], s=26, alpha=0.75, edgecolors="none",
                   label=ARM_LABEL[arm])
    ax.set_xlabel("top-1 agreement with teacher (fidelity)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Fidelity and accuracy are different quantities", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIG_DIR / "fig4_fidelity_vs_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- report

def build_report(analysis):
    L = []
    A = L.append
    A("=" * 78)
    A("RQ4 — KNOWLEDGE DISTILLATION AS A SIMULABILITY PROBE")
    A("=" * 78)
    A("")
    A(f"teacher cells present : {analysis['coverage']['teacher_cells']}/10")
    A(f"student cells present : {analysis['coverage']['student_cells']}/370")
    if analysis["coverage"]["student_cells"] == 0:
        A("")
        A("  STAGE 2 HAS NOT PRODUCED DATA YET — teacher-side results only.")
    A("")

    A("-" * 78)
    A("TEACHERS")
    A("-" * 78)
    for ds in DATASETS:
        A(f"  {DS_LABEL[ds]}")
        for kind in ("qcnn", "cnn"):
            acc = (analysis["teacher_accuracy"].get(ds) or {}).get(kind)
            if acc:
                A(f"    {kind:5s} accuracy   {acc['mean']:.4f} ± {acc['std']:.4f}  "
                  f"(n={acc['n']})")
        for kind in ("qcnn", "cnn"):
            c = (analysis["teacher_ceiling"].get(ds) or {}).get(kind)
            if c:
                A(f"    {kind:5s} ceiling    {c['mean']:.4f} ± {c['std']:.4f}  "
                  f"[{c['min']:.4f}, {c['max']:.4f}] over {c['n_pairs']} seed pairs")
    A("")
    A("  The ceiling is agreement between two independently-seeded teachers of the same")
    A("  kind. Every student fidelity number below must be read against it: a student")
    A("  cannot reasonably be asked to match its teacher more closely than a second")
    A("  teacher does.")
    A("")

    if analysis["coverage"]["student_cells"] == 0:
        A("-" * 78)
        A("STUDENTS — pending")
        A("-" * 78)
        A("  Re-run this script once experiments/rq4_distillation.py has produced cells.")
        return "\n".join(L)

    A("-" * 78)
    A("CAPACITY LADDER — the headline")
    A("-" * 78)
    for ds in DATASETS:
        for arm in ("qkd", "ckd"):
            rows = (analysis["ladder"].get(ds) or {}).get(arm)
            if not rows:
                continue
            A(f"  {DS_LABEL[ds]} / {ARM_LABEL[arm]} (α=1.0)")
            for r in rows:
                A(f"    rung {r['rung']}  {r['n_params']:5d} params   "
                  f"agreement {r['mean']:.4f} ± {r['std']:.4f}")
            v = (analysis["verdict"].get(ds) or {}).get(arm)
            if v:
                A(f"    VERDICT: {v['verdict']}")
                A(f"      {v['reason']}")
            A("")

    A("-" * 78)
    A("CAVEATS")
    A("-" * 78)
    A("  n=5 seeds per cell. The exact two-sided Wilcoxon floor at n=5 is 2/2^5 = 0.0625,")
    A("  so the rank test cannot reach p<0.05 at any effect size; paired t-tests and")
    A("  Cohen's d_z carry the quantitative story, as in RQ2 §4.2.")
    A("  Accuracies are final-epoch point estimates, as in §4.1 and §4.2.")
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main():
    teachers = load_json(RQ4 / "teachers_checkpoint.json") or {}
    students = load_json(RQ4 / "students_checkpoint.json")
    logits = load_teacher_logits()
    cells = cells_frame(students)

    n_teacher_cells = sum(len(v) for v in (teachers.get("seeds_done") or {}).values())
    print(f"teacher cells: {n_teacher_cells}/10   logit caches: {len(logits)}   "
          f"student cells: {len(cells)}/370")

    analysis = {
        "coverage": {"teacher_cells": n_teacher_cells,
                     "student_cells": len(cells),
                     "logit_caches": len(logits)},
        "teacher_accuracy": {}, "teacher_ceiling": {},
        "ladder": {}, "verdict": {},
    }

    for ds in DATASETS:
        analysis["teacher_accuracy"][ds] = {
            k: teacher_accuracy(teachers, ds, k) for k in ("qcnn", "cnn")}
        analysis["teacher_ceiling"][ds] = {
            k: teacher_ceiling(logits, ds, k) for k in ("qcnn", "cnn")}

    rungs = rung_params(cells)
    if cells:
        for ds in DATASETS:
            analysis["ladder"][ds] = {}
            analysis["verdict"][ds] = {}
            for arm, tkey in (("qkd", "qcnn"), ("ckd", "cnn")):
                rows = ladder(cells, ds, arm, 1.0, ["agree_" + tkey], rungs)
                analysis["ladder"][ds][arm] = rows
                ceil = (analysis["teacher_ceiling"][ds].get(tkey) or {}).get("mean")
                analysis["verdict"][ds][arm] = ladder_verdict(rows, ceil)

    RQ4.mkdir(parents=True, exist_ok=True)
    with open(RQ4 / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    report = build_report(analysis)
    (RQ4 / "analysis_report.txt").write_text(report, encoding="utf-8")
    print(report)

    made = []
    if fig1_capacity_ladder(analysis, cells, rungs): made.append("fig1_capacity_ladder")
    if fig2_accuracy_ece(cells, rungs): made.append("fig2_accuracy_ece")
    if fig3_robustness(cells, rungs): made.append("fig3_robustness")
    if fig4_fidelity_vs_accuracy(cells): made.append("fig4_fidelity_vs_accuracy")
    print(f"\nfigures written: {made or 'none (no student data yet)'}")
    print(f"saved → {RQ4/'analysis.json'}, {RQ4/'analysis_report.txt'}")
    print("\nOPEN AND EYEBALL EVERY FIGURE before citing it — three of RQ2's four figures "
          "had label collisions on first render.")


if __name__ == "__main__":
    main()
