"""RQ2 statistical analysis + figures for the design-choice ablation.

Consumes results/rq2/{mnist,fashion_mnist}.json (11 configs x 5 seeds x 50 epochs,
sweep completed 2026-08-19) and produces:

  results/rq2/analysis.json         - every table below, machine-readable
  results/rq2/analysis_report.txt   - console dump, the source for ch4 s4.2
  results/rq2/figures/fig1_config_grid.png       - all 11 configs, both datasets
  results/rq2/figures/fig2_encoding_matched.png  - encoding, within matched cells
  results/rq2/figures/fig3_learning_curves.png   - test_acc vs epoch by encoding
  results/rq2/figures/fig4_depth_capacity.png    - depth + capacity sub-studies

Design notes
------------
- CONFOUNDED ROLL-UP. The obvious "mean accuracy per encoding" averages every config
  carrying that encoding, but the 11 cells are not balanced across encodings: amplitude
  appears in 5 configs (it owns both depth-study cells, L1 and L3) while angle and iqp
  appear in 3 each. Amplitude's roll-up therefore includes an L3 qcnn_native cell that
  the other two encodings have no counterpart for. The roll-up is still computed (it is
  what the handover quoted) but is reported as `confounded: true`, and the headline
  encoding claim is taken from the MATCHED comparison instead: the three (ansatz, depth)
  strata in which all three encodings were actually run --
      (qcnn_native, L2)  (hardware_efficient, L2)  (hardware_efficient, L3)
  Those are like-for-like; the roll-up is not.

- PAIRED BY SEED. Every config was run on the same seed list [42..46] and, within a
  dataset, on the same train/test subset (utils.dataset seeds the subsample from
  config.SEED, not from the run seed). Config A seed i and config B seed i are therefore
  matched, so contrasts use paired tests.

- n = 5 seeds is LOW POWER, exactly as in RQ1. The two-sided exact Wilcoxon floor at 5
  non-zero differences is 2/2^5 = 0.0625, so it CANNOT reach p<0.05 whatever the effect.
  Reported explicitly; the paired t-test and Cohen's dz carry the quantitative story.

- ZERO-VARIANCE CELLS. Four MNIST configs have std exactly 0. `zero_variance_audit`
  records, for each, whether the shared value sits at a saturation level and how many
  distinct final accuracies the 5 seeds produced, so s4.2 can attribute it to the
  ceiling effect (established 2026-08-19: Fashion-MNIST has 0/11 such cells under the
  identical pipeline) rather than printing a bare "+- 0.0000".

Run from the repository root:  python experiments/rq2_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results" / "rq2"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [("mnist", "MNIST"), ("fashion_mnist", "Fashion-MNIST")]
ENCODINGS = ["amplitude", "angle", "iqp"]
ANSATZE = ["qcnn_native", "hardware_efficient"]

# ---------------------------------------------------------------- palette
# Encoding series take categorical slots 1-3 (dataviz skill reference palette);
# validated 2026-08-20 with --pairs all --mode light: all checks PASS, worst CVD
# dE 9.2, worst normal-vision dE 24.0, contrast WARN on aqua (2.74:1) -> the relief
# rule is satisfied by direct labels on every encoding series.
# Ansatz is NOT colour-coded here: RQ1/RQ3 already bind blue->qcnn_native and
# green->hardware_efficient, so within RQ2 figures ansatz is carried by marker
# shape and hatch instead, leaving colour to mean encoding and nothing else.
C_BLUE = "#2a78d6"    # slot 1 -> amplitude (continuity: RQ1's QCNN baseline was amplitude)
C_ORANGE = "#eb6834"  # slot 2 -> angle
C_AQUA = "#1baf7a"    # slot 3 -> iqp
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

ENC_COLOR = {"amplitude": C_BLUE, "angle": C_ORANGE, "iqp": C_AQUA}
ENC_LABEL = {"amplitude": "Amplitude", "angle": "Angle", "iqp": "IQP"}
ANSATZ_LABEL = {"qcnn_native": "QCNN (native)", "hardware_efficient": "HEA"}
ANSATZ_MARKER = {"qcnn_native": "o", "hardware_efficient": "s"}
ANSATZ_HATCH = {"qcnn_native": "", "hardware_efficient": "///"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "axes.labelsize": 10,
})


# ---------------------------------------------------------------- loading

def parse_key(key: str) -> dict:
    """'angle__qcnn_native__L2' -> {'encoding':..., 'ansatz':..., 'depth':2}."""
    enc, ansatz, ldepth = key.split("__")
    return {"encoding": enc, "ansatz": ansatz, "depth": int(ldepth[1:])}


def load(ds_key: str) -> dict:
    d = json.load(open(RESULTS / f"{ds_key}.json"))
    seeds = [str(s) for s in d["seeds"]]
    cells = {}
    for key, hist_by_seed in d["histories"].items():
        meta = parse_key(key)
        final = np.array([hist_by_seed[s]["test_acc"][-1] for s in seeds])
        traj = np.array([hist_by_seed[s]["test_acc"] for s in seeds])
        train_traj = np.array([hist_by_seed[s]["train_acc"] for s in seeds])
        cells[key] = {
            **meta,
            "group": d["summary"][key]["group"],
            "n_params": hist_by_seed[seeds[0]]["n_params"],
            "final": final,
            "traj": traj,
            "train_traj": train_traj,
        }
    return {
        "seeds": [int(s) for s in seeds],
        "epochs": d["epochs"],
        "n_train": d["n_train"],
        "n_test": d["n_test"],
        "cells": cells,
    }


def key_of(encoding: str, ansatz: str, depth: int) -> str:
    return f"{encoding}__{ansatz}__L{depth}"


# ---------------------------------------------------------------- statistics

def paired_contrast(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    """Paired-by-seed contrast, diff = a - b. Degenerate cases guarded.

    Both zero-variance cells and identical cells make sd(diff)=0, which sends the
    paired t statistic to +-inf and makes Wilcoxon raise. Those are reported as such
    rather than as NaN, because a constant non-zero difference across all 5 seeds is
    a meaningful (if untestable) result, not a missing one.
    """
    d = a - b
    n = len(d)
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    n_nonzero = int(np.count_nonzero(d))

    if sd_d > 0:
        se = sd_d / np.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        ci = [mean_d - tcrit * se, mean_d + tcrit * se]
        t_res = stats.ttest_rel(a, b)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
        dz = mean_d / sd_d
    else:
        ci = [mean_d, mean_d]
        t_stat = float("inf") * np.sign(mean_d) if mean_d != 0 else 0.0
        t_p = 0.0 if mean_d != 0 else 1.0
        dz = float("inf") * np.sign(mean_d) if mean_d != 0 else 0.0

    try:
        w_res = stats.wilcoxon(a, b)
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "a": label_a, "b": label_b,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diffs": d.round(6).tolist(),
        "mean_diff": mean_d, "sd_diff": sd_d, "diff_ci95": ci,
        "paired_t": t_stat, "paired_t_p": t_p,
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "wilcoxon_twosided_floor": 2.0 / (2 ** n_nonzero) if n_nonzero else float("nan"),
        "cohen_dz": float(dz),
        "n_seeds": n,
        "constant_difference": sd_d == 0.0,
    }


def per_config_table(data: dict) -> dict:
    out = {}
    for key, c in data["cells"].items():
        f = c["final"]
        out[key] = {
            "encoding": c["encoding"], "ansatz": c["ansatz"], "depth": c["depth"],
            "group": c["group"], "n_params": c["n_params"],
            "mean": float(f.mean()), "std": float(f.std(ddof=0)),
            "sd_sample": float(f.std(ddof=1)),
            "min": float(f.min()), "max": float(f.max()),
            "runs": f.round(6).tolist(),
            "n_distinct": int(len(np.unique(f))),
        }
    return out


def encoding_rollup(data: dict) -> dict:
    """Mean over every config carrying each encoding. CONFOUNDED - see module docstring."""
    out = {}
    for enc in ENCODINGS:
        keys = [k for k, c in data["cells"].items() if c["encoding"] == enc]
        means = np.array([data["cells"][k]["final"].mean() for k in keys])
        out[enc] = {
            "mean_of_config_means": float(means.mean()),
            "n_configs": len(keys),
            "configs": sorted(keys),
            "confounded": True,
            "why": ("config counts differ across encodings (amplitude owns both "
                    "depth-study cells); use matched_encoding instead"),
        }
    return out


def matched_encoding(data: dict) -> dict:
    """Encoding comparison within each (ansatz, depth) stratum holding all three."""
    strata = [("qcnn_native", 2), ("hardware_efficient", 2), ("hardware_efficient", 3)]
    out = {}
    for ansatz, depth in strata:
        keys = {e: key_of(e, ansatz, depth) for e in ENCODINGS}
        if not all(k in data["cells"] for k in keys.values()):
            continue
        finals = {e: data["cells"][k]["final"] for e, k in keys.items()}
        order = sorted(ENCODINGS, key=lambda e: -finals[e].mean())
        out[f"{ansatz}__L{depth}"] = {
            "ansatz": ansatz, "depth": depth,
            "means": {e: float(finals[e].mean()) for e in ENCODINGS},
            "ranking": order,
            "contrasts": {
                "amplitude_vs_angle": paired_contrast(
                    finals["amplitude"], finals["angle"], "amplitude", "angle"),
                "amplitude_vs_iqp": paired_contrast(
                    finals["amplitude"], finals["iqp"], "amplitude", "iqp"),
                "angle_vs_iqp": paired_contrast(
                    finals["angle"], finals["iqp"], "angle", "iqp"),
            },
        }
    # Does one ordering hold across every stratum?
    rankings = [v["ranking"] for v in out.values()]
    out["_consistency"] = {
        "rankings": rankings,
        "identical_across_strata": len({tuple(r) for r in rankings}) == 1,
        "iqp_last_everywhere": all(r[-1] == "iqp" for r in rankings),
        "amplitude_first_everywhere": all(r[0] == "amplitude" for r in rankings),
    }
    return out


def ansatz_comparison(data: dict) -> dict:
    """qcnn_native vs hardware_efficient, matched on encoding.

    Two readings, per the Q5 design decision (rq2_design_decisions.md):
      raw       - both at depth 2. qcnn_native has 74 params vs HEA's 38, so any
                  qcnn_native win is confounded with ~2x capacity.
      capacity  - qcnn_native L2 (74) vs HEA L3 (44), HEA's own non-degenerate
                  ceiling at 6 qubits. Still not matched (44 vs 74) and said so.
    """
    out = {"raw_depth2": {}, "capacity_maximised": {}}
    for enc in ENCODINGS:
        q2 = data["cells"].get(key_of(enc, "qcnn_native", 2))
        h2 = data["cells"].get(key_of(enc, "hardware_efficient", 2))
        h3 = data["cells"].get(key_of(enc, "hardware_efficient", 3))
        if q2 and h2:
            c = paired_contrast(q2["final"], h2["final"], "qcnn_native_L2", "hea_L2")
            c["n_params"] = {"qcnn_native_L2": q2["n_params"], "hea_L2": h2["n_params"]}
            out["raw_depth2"][enc] = c
        if q2 and h3:
            c = paired_contrast(q2["final"], h3["final"], "qcnn_native_L2", "hea_L3")
            c["n_params"] = {"qcnn_native_L2": q2["n_params"], "hea_L3": h3["n_params"]}
            c["residual_param_gap"] = q2["n_params"] - h3["n_params"]
            out["capacity_maximised"][enc] = c
    out["_note"] = ("HEA at its own depth ceiling (44 params) still does not reach "
                    "qcnn_native@L2 (74). Depth-matching is architecturally impossible "
                    "at 6 qubits; the residual gap is disclosed, not claimed away.")
    return out


def depth_study(data: dict) -> dict:
    """amplitude + qcnn_native at depths 1, 2, 3 (the full non-degenerate range)."""
    out = {"depths": {}, "contrasts": {}}
    finals = {}
    for d in (1, 2, 3):
        c = data["cells"].get(key_of("amplitude", "qcnn_native", d))
        if c is None:
            continue
        finals[d] = c["final"]
        out["depths"][d] = {"mean": float(c["final"].mean()),
                            "std": float(c["final"].std(ddof=0)),
                            "n_params": c["n_params"]}
    for lo, hi in ((1, 2), (2, 3), (1, 3)):
        if lo in finals and hi in finals:
            out["contrasts"][f"L{hi}_vs_L{lo}"] = paired_contrast(
                finals[hi], finals[lo], f"L{hi}", f"L{lo}")
    out["_note"] = ("Pooling halves the active qubits each layer (6->3->2->1), so "
                    "depth>=3 is architecturally identical at 6 qubits; [1,2,3] is the "
                    "complete non-degenerate range, not a truncated sweep.")
    return out


def zero_variance_audit(data: dict) -> dict:
    """Which cells have exactly zero seed-variance, and is saturation the reason?"""
    cells = []
    for key, c in data["cells"].items():
        f = c["final"]
        if f.std(ddof=0) == 0.0:
            val = float(f[0])
            n_test = data["n_test"]
            cells.append({
                "config": key,
                "value": val,
                "at_perfect_accuracy": val == 1.0,
                "n_test_items_missed": int(round((1.0 - val) * n_test)),
                "note": (f"all 5 seeds land on {val:.4f} = "
                         f"{int(round(val * n_test))}/{n_test} test items"),
            })
    return {
        "n_zero_variance": len(cells),
        "n_configs": len(data["cells"]),
        "cells": sorted(cells, key=lambda c: -c["value"]),
        "spread_of_all_cells": {
            k: float(c["final"].std(ddof=0)) for k, c in data["cells"].items()
        },
    }


# ---------------------------------------------------------------- figures

def fig1_config_grid(all_data: dict):
    """All 11 configs, mean +- std, one panel per dataset, grouped by encoding."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    for ax, (ds_key, ds_label) in zip(axes, DATASETS):
        data = all_data[ds_key]
        ordered = []
        for enc in ENCODINGS:
            ks = [k for k, c in data["cells"].items() if c["encoding"] == enc]
            ks.sort(key=lambda k: (data["cells"][k]["ansatz"], data["cells"][k]["depth"]))
            ordered += ks
        ys = np.arange(len(ordered))[::-1]
        for y, k in zip(ys, ordered):
            c = data["cells"][k]
            m, s = c["final"].mean(), c["final"].std(ddof=0)
            col = ENC_COLOR[c["encoding"]]
            ax.errorbar(m, y, xerr=s, fmt=ANSATZ_MARKER[c["ansatz"]], color=col,
                        markersize=9, capsize=4, elinewidth=1.8,
                        markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
            # per-seed points, so zero-variance cells read as a single stack
            ax.scatter(c["final"], np.full_like(c["final"], y), s=14, color=col,
                       alpha=0.35, zorder=2, linewidths=0)
            ax.text(m + s + 0.012, y, f"{m:.3f}", va="center", ha="left",
                    fontsize=8.5, color=INK2)
        labels = [f"{ENC_LABEL[data['cells'][k]['encoding']]} · "
                  f"{ANSATZ_LABEL[data['cells'][k]['ansatz']]} · L{data['cells'][k]['depth']}"
                  for k in ordered]
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(0.55, 1.06)
        ax.set_xlabel("Final test accuracy (mean ± std over 5 seeds)")
        ax.set_title(ds_label)
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker=ANSATZ_MARKER[a], color=MUTED, linestyle="",
                          markersize=8, label=ANSATZ_LABEL[a]) for a in ANSATZE]
    axes[1].legend(handles=handles, loc="lower right", title="Ansatz (marker)",
                   title_fontsize=9)
    fig.suptitle("RQ2 — all 11 design configurations, colour = encoding",
                 fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig1_config_grid.png", bbox_inches="tight")
    plt.close(fig)


def fig2_encoding_matched(all_data: dict, analysis: dict):
    """Encoding effect within each matched (ansatz, depth) stratum."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)
    for ax, (ds_key, ds_label) in zip(axes, DATASETS):
        data = all_data[ds_key]
        strata = [k for k in analysis[ds_key]["matched_encoding"] if not k.startswith("_")]
        width = 0.26
        xs = np.arange(len(strata))
        for i, enc in enumerate(ENCODINGS):
            means, sds = [], []
            for st in strata:
                ansatz, depth = st.split("__L")
                c = data["cells"][key_of(enc, ansatz, int(depth))]
                means.append(c["final"].mean())
                sds.append(c["final"].std(ddof=0))
            off = (i - 1) * (width + 0.02)
            bars = ax.bar(xs + off, means, width, yerr=sds, color=ENC_COLOR[enc],
                          capsize=3, error_kw={"elinewidth": 1.4, "ecolor": INK2},
                          edgecolor=SURFACE, linewidth=2.0, zorder=3)
            for b, m, s in zip(bars, means, sds):
                ax.text(b.get_x() + b.get_width() / 2, m + s + 0.010, f"{m:.3f}",
                        ha="center", va="bottom", fontsize=8, color=INK2)
            # Direct label inside the first group's bars — the relief rule for the
            # aqua slot (2.74:1 on this surface), and it keeps identity off colour
            # alone without a legend that would overlap the depth-3 bars.
            b0 = bars[0]
            ax.text(b0.get_x() + b0.get_width() / 2, 0.575, ENC_LABEL[enc],
                    ha="center", va="bottom", fontsize=8.5, color=SURFACE,
                    fontweight="bold", rotation=90, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{ANSATZ_LABEL[s.split('__L')[0]]}\ndepth {s.split('__L')[1]}"
                            for s in strata], fontsize=9)
        ax.set_ylim(0.55, 1.09)
        ax.set_title(ds_label)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Final test accuracy")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ENC_COLOR[e], label=ENC_LABEL[e])
               for e in ENCODINGS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.935),
               title="Encoding", title_fontsize=9)
    fig.suptitle("RQ2 — encoding compared only within matched (ansatz, depth) cells",
                 fontsize=12.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIG_DIR / "fig2_encoding_matched.png", bbox_inches="tight")
    plt.close(fig)


def fig3_learning_curves(all_data: dict):
    """test_acc vs epoch, mean +- std band, core grid only (depth 2), split by ansatz."""
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True, sharey=True)
    for col, (ds_key, ds_label) in enumerate(DATASETS):
        data = all_data[ds_key]
        for row, ansatz in enumerate(ANSATZE):
            ax = axes[row][col]
            ends = []
            for enc in ENCODINGS:
                c = data["cells"][key_of(enc, ansatz, 2)]
                traj = c["traj"]
                ep = np.arange(1, traj.shape[1] + 1)
                m, s = traj.mean(0), traj.std(0, ddof=0)
                ax.plot(ep, m, color=ENC_COLOR[enc], linewidth=2.0, zorder=3)
                ax.fill_between(ep, m - s, m + s, color=ENC_COLOR[enc], alpha=0.16,
                                linewidth=0, zorder=2)
                ends.append([float(m[-1]), enc])
            # Amplitude and angle converge to within <0.01 in three of the four
            # panels, so end labels are nudged apart before drawing rather than
            # written on top of each other.
            ends.sort()
            min_gap = 0.030
            for i in range(1, len(ends)):
                if ends[i][0] - ends[i - 1][0] < min_gap:
                    ends[i][0] = ends[i - 1][0] + min_gap
            for y, enc in ends:
                ax.text(ep[-1] + 0.7, y, ENC_LABEL[enc], color=ENC_COLOR[enc],
                        fontsize=9, va="center", ha="left", fontweight="bold")
            ax.set_title(f"{ds_label} · {ANSATZ_LABEL[ansatz]}", fontsize=10)
            ax.set_xlim(1, 58)
            ax.set_ylim(0.42, 1.03)
            if row == 1:
                ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel("Test accuracy")
    fig.suptitle("RQ2 — learning curves, core grid (depth 2), mean ± std over 5 seeds",
                 fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig3_learning_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig4_depth_capacity(all_data: dict):
    """Left: depth sub-study. Right: capacity-maximised HEA vs qcnn_native."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    ax = axes[0]
    for ds_key, ds_label in DATASETS:
        data = all_data[ds_key]
        ds_style = "-" if ds_key == "mnist" else "--"
        depths, means, sds = [], [], []
        for d in (1, 2, 3):
            c = data["cells"][key_of("amplitude", "qcnn_native", d)]
            depths.append(d)
            means.append(c["final"].mean())
            sds.append(c["final"].std(ddof=0))
        ax.errorbar(depths, means, yerr=sds, fmt="o" + ds_style, color=C_BLUE,
                    markersize=8, capsize=4, linewidth=2.0,
                    markeredgecolor=SURFACE, markeredgewidth=1.6)
        ax.text(depths[-1] + 0.06, means[-1], ds_label, color=INK2, fontsize=9,
                va="center", ha="left")
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.8, 3.9)
    ax.set_xlabel("Circuit depth (conv+pool blocks)")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Depth sub-study — amplitude · QCNN (native)\n"
                 "59 / 74 / 89 parameters at L1 / L2 / L3", fontsize=10)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    labels, xs = [], []
    width = 0.36
    for i, (ds_key, ds_label) in enumerate(DATASETS):
        data = all_data[ds_key]
        for j, enc in enumerate(ENCODINGS):
            q2 = data["cells"][key_of(enc, "qcnn_native", 2)]
            h3 = data["cells"][key_of(enc, "hardware_efficient", 3)]
            x = i * 3.6 + j
            ax.bar(x - width / 2, q2["final"].mean(), width,
                   yerr=q2["final"].std(ddof=0), color=ENC_COLOR[enc],
                   edgecolor=SURFACE, linewidth=2.0, capsize=3,
                   error_kw={"elinewidth": 1.3, "ecolor": INK2}, zorder=3)
            ax.bar(x + width / 2, h3["final"].mean(), width,
                   yerr=h3["final"].std(ddof=0), color=ENC_COLOR[enc],
                   edgecolor=SURFACE, linewidth=2.0, capsize=3, alpha=0.55,
                   hatch=ANSATZ_HATCH["hardware_efficient"],
                   error_kw={"elinewidth": 1.3, "ecolor": INK2}, zorder=3)
            xs.append(x)
            labels.append(ENC_LABEL[enc])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.55, 1.06)
    ax.set_title("Capacity-maximised comparison\n"
                 "solid = QCNN native L2 (74 params) · hatched = HEA L3 (44)",
                 fontsize=10)
    ax.set_ylabel("Final test accuracy")
    ax.grid(axis="x", visible=False)
    # Dataset names go below the encoding tick labels, in axes coordinates, so they
    # never land on top of a bar.
    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (_, ds_label) in enumerate(DATASETS):
        ax.text(i * 3.6 + 1, -0.135, ds_label, ha="center", va="top",
                fontsize=10, color=INK, transform=trans)

    fig.suptitle("RQ2 — depth and capacity sub-studies", fontsize=12.5, color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig4_depth_capacity.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- report

def fmt_contrast(c: dict, indent: str = "      ") -> list[str]:
    lines = [f"{indent}{c['a']} {c['mean_a']:.4f}  vs  {c['b']} {c['mean_b']:.4f}"
             f"   diff {c['mean_diff']:+.4f}"]
    if c["constant_difference"]:
        lines.append(f"{indent}  difference is CONSTANT across all {c['n_seeds']} seeds "
                     f"(sd=0) — t/dz undefined in the usual sense")
    else:
        lines.append(f"{indent}  95% CI [{c['diff_ci95'][0]:+.4f}, {c['diff_ci95'][1]:+.4f}]"
                     f"   t={c['paired_t']:+.2f}  p={c['paired_t_p']:.4f}"
                     f"   dz={c['cohen_dz']:+.2f}")
    lines.append(f"{indent}  Wilcoxon p={c['wilcoxon_p']:.4f} "
                 f"(exact two-sided floor at n={c['n_seeds']}: "
                 f"{c['wilcoxon_twosided_floor']:.4f})")
    return lines


def build_report(all_data: dict, analysis: dict) -> str:
    L = []
    add = L.append
    add("=" * 78)
    add("  RQ2 ANALYSIS — QCNN design choices (encoding × ansatz × depth)")
    add("=" * 78)
    d0 = all_data["mnist"]
    add(f"  11 configs × {len(d0['seeds'])} seeds × {d0['epochs']} epochs × 2 datasets")
    add(f"  seeds {d0['seeds']}   n_train={d0['n_train']}  n_test={d0['n_test']}")
    add("")
    add("  POWER: n=5 seeds. The exact two-sided Wilcoxon floor at 5 non-zero diffs is")
    add("  2/2^5 = 0.0625, so Wilcoxon CANNOT reach p<0.05 here regardless of effect")
    add("  size. Paired t and Cohen's dz carry the quantitative story, as in RQ1.")
    add("")

    for ds_key, ds_label in DATASETS:
        a = analysis[ds_key]
        add("=" * 78)
        add(f"  {ds_label}")
        add("=" * 78)

        add("")
        add("  PER-CONFIG (final-epoch test accuracy)")
        add(f"  {'config':38s} {'mean':>7s} {'sd':>7s} {'params':>7s}  {'group':10s} distinct")
        add(f"  {'-' * 76}")
        for key, r in sorted(a["per_config"].items(),
                             key=lambda kv: (kv[1]["encoding"], kv[1]["ansatz"],
                                             kv[1]["depth"])):
            add(f"  {key:38s} {r['mean']:7.4f} {r['std']:7.4f} {r['n_params']:7d}  "
                f"{r['group']:10s} {r['n_distinct']}/5")

        add("")
        add("  ENCODING ROLL-UP  ** CONFOUNDED — do not use as the headline **")
        for enc in ENCODINGS:
            r = a["encoding_rollup"][enc]
            add(f"    {ENC_LABEL[enc]:10s} {r['mean_of_config_means']:.4f}  "
                f"(over {r['n_configs']} configs)")
        add("    Config counts are unbalanced: amplitude owns both depth-study cells,")
        add("    so its roll-up includes an L3 qcnn_native cell that angle and IQP have")
        add("    no counterpart for. Use the matched comparison below instead.")

        add("")
        add("  ENCODING, MATCHED WITHIN (ansatz, depth) — the defensible comparison")
        for st, r in a["matched_encoding"].items():
            if st.startswith("_"):
                continue
            add(f"    [{ANSATZ_LABEL[r['ansatz']]}, depth {r['depth']}]  "
                f"ranking: {' > '.join(ENC_LABEL[e] for e in r['ranking'])}")
            for name, c in r["contrasts"].items():
                add(f"      {name}:")
                L.extend(fmt_contrast(c, indent="        "))
        cons = a["matched_encoding"]["_consistency"]
        add(f"    Same ordering in every stratum: {cons['identical_across_strata']}")
        add(f"    IQP last in every stratum:      {cons['iqp_last_everywhere']}")
        add(f"    Amplitude first in every stratum: {cons['amplitude_first_everywhere']}")

        add("")
        add("  ANSATZ — raw (both depth 2; qcnn_native 74 params vs HEA 38)")
        for enc, c in a["ansatz"]["raw_depth2"].items():
            add(f"    {ENC_LABEL[enc]}:")
            L.extend(fmt_contrast(c, indent="      "))
        add("")
        add("  ANSATZ — capacity-maximised (qcnn_native L2 = 74 vs HEA L3 = 44)")
        for enc, c in a["ansatz"]["capacity_maximised"].items():
            add(f"    {ENC_LABEL[enc]}  (residual param gap {c['residual_param_gap']}):")
            L.extend(fmt_contrast(c, indent="      "))

        add("")
        add("  DEPTH SUB-STUDY (amplitude · qcnn_native)")
        for d, r in a["depth"]["depths"].items():
            add(f"    depth {d}: {r['mean']:.4f} ± {r['std']:.4f}  ({r['n_params']} params)")
        for name, c in a["depth"]["contrasts"].items():
            add(f"    {name}:")
            L.extend(fmt_contrast(c, indent="      "))

        add("")
        zv = a["zero_variance"]
        add(f"  ZERO SEED-VARIANCE AUDIT — {zv['n_zero_variance']}/{zv['n_configs']} configs")
        if zv["cells"]:
            for c in zv["cells"]:
                add(f"    {c['config']:38s} {c['note']}"
                    f"{'  [perfect]' if c['at_perfect_accuracy'] else ''}")
        else:
            add("    none — every config varies across seeds")
        add("")

    add("=" * 78)
    add("  CROSS-DATASET")
    add("=" * 78)
    x = analysis["cross_dataset"]
    add("")
    add("  Amplitude's margin over angle, within matched cells:")
    for st, r in x["amplitude_over_angle_by_stratum"].items():
        add(f"    [{st}]  MNIST {r['mnist']:+.4f}   Fashion-MNIST {r['fashion_mnist']:+.4f}")
    add("")
    add("  " + x["ceiling_note"])
    add("")
    add("  Zero-variance cells: "
        f"MNIST {x['zero_variance_counts']['mnist']}/11, "
        f"Fashion-MNIST {x['zero_variance_counts']['fashion_mnist']}/11 "
        "— same code, same seeds, same pipeline, only the dataset differs, which is "
        "what rules out a non-propagating seed and leaves the ceiling effect.")
    add("")
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main():
    all_data = {ds: load(ds) for ds, _ in DATASETS}

    analysis = {}
    for ds_key, _ in DATASETS:
        data = all_data[ds_key]
        analysis[ds_key] = {
            "seeds": data["seeds"], "epochs": data["epochs"],
            "n_train": data["n_train"], "n_test": data["n_test"],
            "per_config": per_config_table(data),
            "encoding_rollup": encoding_rollup(data),
            "matched_encoding": matched_encoding(data),
            "ansatz": ansatz_comparison(data),
            "depth": depth_study(data),
            "zero_variance": zero_variance_audit(data),
        }

    # cross-dataset summary
    strata = [k for k in analysis["mnist"]["matched_encoding"] if not k.startswith("_")]
    amp_over_angle = {}
    for st in strata:
        amp_over_angle[st] = {
            ds: analysis[ds]["matched_encoding"][st]["contrasts"]
                ["amplitude_vs_angle"]["mean_diff"]
            for ds, _ in DATASETS
        }
    analysis["cross_dataset"] = {
        "amplitude_over_angle_by_stratum": amp_over_angle,
        "zero_variance_counts": {
            ds: analysis[ds]["zero_variance"]["n_zero_variance"] for ds, _ in DATASETS
        },
        "ceiling_note": (
            "MNIST 0-vs-1 saturates (two configs reach 1.0000), which flatters whichever "
            "encoding is ahead and compresses every contrast at the top of the range. "
            "Fashion-MNIST is the more informative dataset and should carry the argument."
        ),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=float)

    report = build_report(all_data, analysis)
    (RESULTS / "analysis_report.txt").write_text(report, encoding="utf-8")
    print(report)

    fig1_config_grid(all_data)
    fig2_encoding_matched(all_data, analysis)
    fig3_learning_curves(all_data)
    fig4_depth_capacity(all_data)
    print(f"\nWrote {RESULTS/'analysis.json'}")
    print(f"Wrote {RESULTS/'analysis_report.txt'}")
    print(f"Wrote 4 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
