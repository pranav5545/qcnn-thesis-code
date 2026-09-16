"""RQ1 statistical analysis + learning-curve figures.

Consumes the stored per-epoch histories in results/rq1/{mnist,fashion_mnist}.json
(QCNN vs parameter-matched CNN, 5 seeds x 50 epochs each) and produces:

  results/rq1/analysis.json        - paired stats, cold-start metrics, per-seed diffs
  results/rq1/analysis_report.txt  - console dump
  results/rq1/figures/fig1_learning_curves.png  - test_acc vs epoch, mean +/- std band
  results/rq1/figures/fig2_paired_seeds.png     - per-seed final-acc paired slope plot

Design notes:
- The comparison is PAIRED by seed: under the training protocol (ch3 s3.5) a given seed
  re-initialises BOTH models and fixes the (shared) data subset, so QCNN seed i and CNN
  seed i are matched. Hence paired t-test + Wilcoxon signed-rank, not two-sample tests.
- n = 5 seeds/cell is LOW POWER. In particular the two-sided Wilcoxon signed-rank floor
  at n=5 (5 non-zero diffs) is 2/2^5 = 0.0625, so it CANNOT reach p<0.05 regardless of
  effect. This is reported explicitly rather than hidden; the paired t-test and the
  effect size (Cohen's dz) carry the quantitative story.

Run from the repository root:  python experiments/rq1_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results" / "rq1"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATASETS = [("mnist", "MNIST"), ("fashion_mnist", "Fashion-MNIST")]
COLD_START_THRESH = 0.80  # test_acc level used to quantify "slow cold start"

# ---------------------------------------------------------------- palette (matches
# rq3_analysis.py; blue = QCNN kept identical, amber for CNN so it does NOT read as the
# green HEA series from the RQ3 figures)
C_BLUE = "#2a78d6"   # qcnn
C_AMBER = "#d9782a"  # classical cnn
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
MODEL_COLOR = {"qcnn": C_BLUE, "cnn": C_AMBER}
MODEL_LABEL = {"qcnn": "QCNN", "cnn": "CNN"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 9, "axes.labelsize": 10,
})


def load(key, exclude_seeds: set[int] = frozenset()):
    d = json.load(open(RESULTS / f"{key}.json"))
    seeds = d["seeds"]
    keep = [i for i, s in enumerate(seeds) if s not in exclude_seeds]
    seeds = [seeds[i] for i in keep]
    # final-epoch test_acc per seed, recomputed from histories (not the pre-baked summary)
    q_final = np.array([d["qcnn_histories"][i]["test_acc"][-1] for i in keep])
    c_final = np.array([d["cnn_histories"][i]["test_acc"][-1] for i in keep])
    # full test_acc trajectories, shape (n_seeds, n_epochs)
    q_traj = np.array([d["qcnn_histories"][i]["test_acc"] for i in keep])
    c_traj = np.array([d["cnn_histories"][i]["test_acc"] for i in keep])
    return seeds, q_final, c_final, q_traj, c_traj


def epochs_to_thresh(traj, thresh):
    """First epoch index (1-based) at which test_acc >= thresh; n_epochs+1 if never."""
    out = []
    for row in traj:
        hit = np.argmax(row >= thresh)
        out.append(int(hit) + 1 if row[hit] >= thresh else len(row) + 1)
    return np.array(out)


def paired_stats(q, c):
    """Paired comparison of QCNN vs CNN final accuracies (diff = QCNN - CNN)."""
    d = q - c
    n = len(d)
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    se = sd_d / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    ci = (mean_d - tcrit * se, mean_d + tcrit * se)
    # paired t-test
    t_res = stats.ttest_rel(q, c)
    # Wilcoxon signed-rank (two-sided, exact); guard the all-zero-diff degenerate case
    try:
        w_res = stats.wilcoxon(q, c)
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    dz = mean_d / sd_d if sd_d > 0 else float("nan")  # Cohen's dz (paired)
    # exact two-sided Wilcoxon floor for this many non-zero diffs
    n_nonzero = int(np.count_nonzero(d))
    wilcoxon_floor = 2.0 / (2 ** n_nonzero) if n_nonzero > 0 else float("nan")
    return {
        "diffs_qcnn_minus_cnn": d.round(6).tolist(),
        "mean_diff": mean_d, "sd_diff": sd_d, "diff_ci95": [ci[0], ci[1]],
        "paired_t": float(t_res.statistic), "paired_t_p": float(t_res.pvalue),
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "wilcoxon_twosided_floor": wilcoxon_floor,
        "cohen_dz": float(dz),
        "n_seeds": n,
    }


def analyse(exclude_seeds: set[int] = frozenset(), out_prefix: str = "analysis",
            title_suffix: str = ""):
    excl_note = (f"Excludes seed(s) {sorted(exclude_seeds)} as a documented sensitivity "
                 f"check — see results/rq1/SENSITIVITY_seed47.md. This is NOT the primary "
                 f"analysis." if exclude_seeds else
                 "Primary analysis — all collected seeds, no exclusions.")
    report = ["RQ1 STATISTICAL ANALYSIS - QCNN vs parameter-matched CNN" + title_suffix,
              "=" * 62,
              excl_note,
              "Paired by seed. diff = QCNN - CNN final-epoch test_acc.",
              "NOTE: the two-sided Wilcoxon signed-rank test has an exact minimum",
              "      p-value of 2/2^(nonzero diffs) at small n — reported per-dataset",
              "      below rather than assumed; it may not reach p<0.05 regardless of effect.",
              ""]
    out = {"_meta": {"threshold_cold_start": COLD_START_THRESH,
                     "exclude_seeds": sorted(exclude_seeds),
                     "wilcoxon_floor_note": "two-sided exact min p at n nonzero diffs = 2/2^n"}}

    traj_store = {}
    for key, title in DATASETS:
        seeds, q, c, q_traj, c_traj = load(key, exclude_seeds)
        traj_store[key] = (seeds, q_traj, c_traj)
        st = paired_stats(q, c)
        q_cold = epochs_to_thresh(q_traj, COLD_START_THRESH)
        c_cold = epochs_to_thresh(c_traj, COLD_START_THRESH)
        st.update({
            "qcnn_final": q.round(6).tolist(), "cnn_final": c.round(6).tolist(),
            "qcnn_final_mean": float(q.mean()), "qcnn_final_std": float(q.std(ddof=1)),
            "cnn_final_mean": float(c.mean()), "cnn_final_std": float(c.std(ddof=1)),
            f"qcnn_epochs_to_{int(COLD_START_THRESH*100)}": q_cold.tolist(),
            f"cnn_epochs_to_{int(COLD_START_THRESH*100)}": c_cold.tolist(),
            "qcnn_coldstart_mean": float(q_cold.mean()),
            "cnn_coldstart_mean": float(c_cold.mean()),
        })
        out[key] = st

        winner = "QCNN" if st["mean_diff"] > 0 else "CNN"
        report += [
            f"## {title}  (n={st['n_seeds']}, seeds={seeds})",
            f"  QCNN final: {q.mean():.4f} +/- {q.std(ddof=1):.4f}   runs {q.tolist()}",
            f"  CNN  final: {c.mean():.4f} +/- {c.std(ddof=1):.4f}   runs {c.tolist()}",
            f"  per-seed diff (QCNN-CNN): {st['diffs_qcnn_minus_cnn']}",
            f"  mean diff {st['mean_diff']:+.4f}  95% CI [{st['diff_ci95'][0]:+.4f}, {st['diff_ci95'][1]:+.4f}]  -> favours {winner}",
            f"  paired t = {st['paired_t']:.3f},  p = {st['paired_t_p']:.4f}",
            f"  Wilcoxon W = {st['wilcoxon_stat']:.1f}, p = {st['wilcoxon_p']:.4f}  (floor {st['wilcoxon_twosided_floor']:.4f})",
            f"  Cohen's dz = {st['cohen_dz']:+.3f}",
            f"  cold-start (epochs to test_acc>={COLD_START_THRESH:.2f}): "
            f"QCNN {q_cold.tolist()} (mean {q_cold.mean():.1f}) | "
            f"CNN {c_cold.tolist()} (mean {c_cold.mean():.1f})",
            "",
        ]

    (RESULTS / f"{out_prefix}.json").write_text(json.dumps(out, indent=2))
    (RESULTS / f"{out_prefix}_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"wrote {RESULTS/f'{out_prefix}.json'}")
    print(f"wrote {RESULTS/f'{out_prefix}_report.txt'}")
    return traj_store


def fig_learning_curves(traj_store):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, DATASETS):
        _, q_traj, c_traj = traj_store[key]
        epochs = np.arange(1, q_traj.shape[1] + 1)
        for traj, tag in [(q_traj, "qcnn"), (c_traj, "cnn")]:
            m, s = traj.mean(0), traj.std(0)
            col = MODEL_COLOR[tag]
            ax.fill_between(epochs, m - s, m + s, color=col, alpha=0.12, lw=0)
            ax.plot(epochs, m, color=col, lw=2, label=MODEL_LABEL[tag])
        ax.axhline(0.5, color=AXIS, lw=1, ls=":")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_xlim(1, q_traj.shape[1])
    n = next(iter(traj_store.values()))[1].shape[0]
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"RQ1 learning curves - mean +/- std over {n} seeds", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig1_learning_curves.png")
    plt.close(fig)
    print(f"wrote {FIG_DIR/'fig1_learning_curves.png'}")


def fig_paired_seeds(traj_store):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4), sharey=True)
    for ax, (key, title) in zip(axes, DATASETS):
        seeds, q_traj, c_traj = traj_store[key]
        q, c = q_traj[:, -1], c_traj[:, -1]
        for i, sd in enumerate(seeds):
            ax.plot([0, 1], [q[i], c[i]], "-", color=MUTED, lw=1, alpha=0.6, zorder=1)
        ax.scatter(np.zeros_like(q), q, color=C_BLUE, s=55, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2, label="QCNN")
        ax.scatter(np.ones_like(c), c, color=C_AMBER, s=55, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2, label="CNN")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["QCNN", "CNN"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(title)
    n = next(iter(traj_store.values()))[1].shape[0]
    axes[0].set_ylabel("final-epoch test accuracy")
    fig.suptitle(f"RQ1 paired per-seed final accuracy ({n} seeds)", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig2_paired_seeds.png")
    plt.close(fig)
    print(f"wrote {FIG_DIR/'fig2_paired_seeds.png'}")


if __name__ == "__main__":
    traj_store = analyse()  # primary: all seeds, no exclusions
    fig_learning_curves(traj_store)
    fig_paired_seeds(traj_store)

    # Documented sensitivity check — see results/rq1/SENSITIVITY_seed47.md for why.
    # Deliberately does NOT touch the primary analysis/figures above.
    print("\n" + "=" * 62)
    print("SENSITIVITY CHECK: excluding seed 47 (CNN dying-ReLU failure)")
    print("=" * 62 + "\n")
    analyse(exclude_seeds={47}, out_prefix="analysis_excl_seed47",
            title_suffix=" [SENSITIVITY: excl. seed 47]")
