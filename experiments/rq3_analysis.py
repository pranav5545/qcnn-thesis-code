"""RQ3 analysis pass: scaling fits, corrected gauss-arm statistics, entropy tables, figures.

Consumes results/rq3/{gradient_variance,entropy_interleaved}.json (data collection
2026-07-08, 32 cells x 200 inits; no rerun needed — full per-init gradients stored).
Produces results/rq3/analysis.json + results/rq3/figures/*.png.

Statistics
----------
- Per cell: Var[g] (ddof=1), E[g^2], mean|g|, mean g. Per the D5 caveat
  (rq3_design_decisions.md) the gauss arm's init mean is not 0, so Var understates
  signal; E[g^2] (raw second moment, = Var when mean=0) is the cross-init metric
  used for scaling fits on every arm, with Var reported alongside.
- Per arm (ansatz x init x input, 4 qubit counts): least-squares fits of
    polynomial  ln y = ln a - k ln n   (Pesah 2021: at worst poly decay for QCNN)
    exponential ln y = ln a - b n      (McClean 2018 barren-plateau signature)
  Both have 2 params, so AIC comparison reduces to RSS comparison on ln y.
- Bootstrap (B=1000, resampling the 200 per-init gradients within each cell):
  CI on the poly exponent k and exp rate b, and P(poly) = fraction of resamples
  where the polynomial model has lower RSS. Only 4 support points per fit — the
  bootstrap quantifies sampling noise in y, not model identifiability; treated as
  descriptive, stated as such in SUMMARY.md.
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter

RESULTS_DIR = Path("results/rq3")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

QUBITS = [4, 6, 8, 10]
ANSATZE = ["qcnn_native", "hardware_efficient"]
INITS = ["uniform", "gauss0.1"]
INPUTS = ["zero", "mnist"]
B_BOOT = 1000
RNG = np.random.default_rng(0)

# ---------------------------------------------------------------- palette (dataviz skill,
# validated 2026-07-15: PASS light mode, contrast WARN on magenta/yellow -> direct labels)
C_BLUE = "#2a78d6"    # slot 1 -> qcnn_native
C_GREEN = "#008300"   # slot 2 -> hardware_efficient
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

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

ANSATZ_COLOR = {"qcnn_native": C_BLUE, "hardware_efficient": C_GREEN}
ANSATZ_LABEL = {"qcnn_native": "QCNN (native)", "hardware_efficient": "HEA"}
INPUT_STYLE = {"zero": "-", "mnist": "--"}
INPUT_LABEL = {"zero": r"$|0\rangle$ input", "mnist": "MNIST input"}
INIT_LABEL = {"uniform": r"uniform $[-\pi,\pi]$", "gauss0.1": r"$\mathcal{N}(0,0.1^2)$"}


def cell_key(n, ansatz, init, inp):
    return f"q{n}_{ansatz}_{init}_{inp}"


def load():
    gv = json.loads((RESULTS_DIR / "gradient_variance.json").read_text())
    en = json.loads((RESULTS_DIR / "entropy_interleaved.json").read_text())
    return gv["cells"], en["cells"]


def cell_stats(grads, target_param):
    # grads is (n_inits, n_params); analysis follows D2: the single fixed
    # center-conv parameter. ddof=0 matches the stored var_target convention.
    g = np.asarray(grads)[:, target_param]
    # percentile-bootstrap CI on E[g^2]: g^2 is heavy-tailed, normal approx is poor
    bm = np.mean(g[RNG.integers(0, g.size, (2000, g.size))] ** 2, axis=1)
    return {
        "var": float(np.var(g)),
        "e_g2": float(np.mean(g**2)),
        "e_g2_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
        "mean_abs_g": float(np.mean(np.abs(g))),
        "mean_g": float(np.mean(g)),
        "n": int(g.size),
    }


def fit_models(n_arr, y_arr):
    """Fit ln y = a - k ln n (poly) and ln y = a - b n (exp); R^2 on ln y."""
    ly = np.log(y_arr)

    def lsq(x):
        A = np.vstack([np.ones_like(x), x]).T
        coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
        pred = A @ coef
        rss = float(np.sum((ly - pred) ** 2))
        tss = float(np.sum((ly - ly.mean()) ** 2))
        return coef, rss, 1 - rss / tss if tss > 0 else float("nan")

    (a_p, slope_p), rss_p, r2_p = lsq(np.log(n_arr))
    (a_e, slope_e), rss_e, r2_e = lsq(np.asarray(n_arr, float))
    n_pts = len(n_arr)
    # equal param count -> ΔAIC = n * ln(RSS_exp / RSS_poly); >0 favors poly
    daic = n_pts * (np.log(rss_e) - np.log(rss_p)) if min(rss_p, rss_e) > 0 else 0.0
    return {
        "poly_exponent_k": float(-slope_p), "poly_r2": float(r2_p), "poly_rss": rss_p,
        "exp_rate_b": float(-slope_e), "exp_r2": float(r2_e), "exp_rss": rss_e,
        "delta_aic_exp_minus_poly": float(daic),
        "preferred": "polynomial" if rss_p < rss_e else "exponential",
    }


def bootstrap_arm(grads_by_n, metric):
    """Resample per-init gradients within each cell; refit both models B times."""
    ks, bs, poly_wins = [], [], 0
    cells = {n: np.asarray(g) for n, g in grads_by_n.items()}
    for _ in range(B_BOOT):
        ys = []
        for n in QUBITS:
            g = cells[n]
            gb = g[RNG.integers(0, g.size, g.size)]
            ys.append(np.mean(gb**2) if metric == "e_g2" else np.var(gb))
        f = fit_models(QUBITS, np.asarray(ys))
        ks.append(f["poly_exponent_k"])
        bs.append(f["exp_rate_b"])
        poly_wins += f["preferred"] == "polynomial"
    ks, bs = np.asarray(ks), np.asarray(bs)
    return {
        "k_ci95": [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))],
        "b_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "p_poly_preferred": poly_wins / B_BOOT,
        "n_boot": B_BOOT,
    }


def main():
    gv_cells, en_cells = load()

    # ---------------- per-cell stats
    cells_out = {}
    for n in QUBITS:
        for a in ANSATZE:
            for i in INITS:
                for x in INPUTS:
                    k = cell_key(n, a, i, x)
                    c = gv_cells[k]
                    s = cell_stats(c["grads"], c["target_param"])
                    s["var_target_stored"] = c["var_target"]
                    s["halfcut_ratio_mean"] = c["entropy_ratio_mean"]
                    s["halfcut_ratio_std"] = c["entropy_ratio_std"]
                    e = en_cells[k]
                    s["interleaved_ratio_mean"] = e["entropy_ratio_mean"]
                    s["interleaved_ratio_std"] = e["entropy_ratio_std"]
                    s["n_layers"] = c["n_layers"]
                    s["n_params"] = c["n_params"]
                    cells_out[k] = s

    # ---------------- per-arm fits (metric: e_g2 everywhere; var reported alongside)
    arms_out = {}
    for a in ANSATZE:
        for i in INITS:
            for x in INPUTS:
                arm = f"{a}_{i}_{x}"
                y_e2 = np.array([cells_out[cell_key(n, a, i, x)]["e_g2"] for n in QUBITS])
                y_var = np.array([cells_out[cell_key(n, a, i, x)]["var"] for n in QUBITS])
                grads_by_n = {
                    n: np.asarray(gv_cells[cell_key(n, a, i, x)]["grads"])
                    [:, gv_cells[cell_key(n, a, i, x)]["target_param"]]
                    for n in QUBITS
                }
                arms_out[arm] = {
                    "e_g2_by_n": dict(zip(map(str, QUBITS), y_e2.tolist())),
                    "var_by_n": dict(zip(map(str, QUBITS), y_var.tolist())),
                    "decay_factor_n4_to_n10_e_g2": float(y_e2[0] / y_e2[-1]),
                    "fit_e_g2": fit_models(QUBITS, y_e2),
                    "fit_var": fit_models(QUBITS, y_var),
                    "bootstrap_e_g2": bootstrap_arm(grads_by_n, "e_g2"),
                }

    # ---------------- derived comparisons
    comparisons = {
        "mnist_over_zero_e_g2": {   # <1 = data input suppresses gradients
            f"{a}_{i}": {
                str(n): cells_out[cell_key(n, a, i, "mnist")]["e_g2"]
                / cells_out[cell_key(n, a, i, "zero")]["e_g2"]
                for n in QUBITS
            }
            for a in ANSATZE for i in INITS
        },
        "gauss_over_uniform_e_g2": {  # >1 = small-angle init preserves gradient signal
            f"{a}_{x}": {
                str(n): cells_out[cell_key(n, a, "gauss0.1", x)]["e_g2"]
                / cells_out[cell_key(n, a, "uniform", x)]["e_g2"]
                for n in QUBITS
            }
            for a in ANSATZE for x in INPUTS
        },
    }

    out = {
        "experiment": "rq3_analysis",
        "date": "2026-07-15",
        "metric_note": "fits on E[g^2] (raw 2nd moment; = Var at mean-0 inits); "
                       "D5 caveat: gauss-arm Var understates signal (mean != 0)",
        "cells": cells_out, "arms": arms_out, "comparisons": comparisons,
    }
    (RESULTS_DIR / "analysis.json").write_text(json.dumps(out, indent=1))

    # ---------------- console report
    print("=" * 100)
    print(f"{'arm':<38}{'E[g2] n=4':>11}{'n=10':>11}{'decay':>8}"
          f"{'k (poly)':>10}{'R2p':>6}{'b (exp)':>9}{'R2e':>6}{'pref':>6}{'P(poly)':>9}")
    for arm, d in arms_out.items():
        f, bs = d["fit_e_g2"], d["bootstrap_e_g2"]
        print(f"{arm:<38}{d['e_g2_by_n']['4']:>11.2e}{d['e_g2_by_n']['10']:>11.2e}"
              f"{d['decay_factor_n4_to_n10_e_g2']:>8.1f}{f['poly_exponent_k']:>10.2f}"
              f"{f['poly_r2']:>6.2f}{f['exp_rate_b']:>9.3f}{f['exp_r2']:>6.2f}"
              f"{f['preferred'][:4]:>6}{bs['p_poly_preferred']:>9.2f}")
    print("=" * 100)
    print("\nMNIST/zero E[g2] ratio (suppression <1):")
    for arm, r in comparisons["mnist_over_zero_e_g2"].items():
        print(f"  {arm:<34}" + "".join(f"  n={n}: {v:6.2f}" for n, v in r.items()))
    print("\ngauss/uniform E[g2] ratio (mitigation >1):")
    for arm, r in comparisons["gauss_over_uniform_e_g2"].items():
        print(f"  {arm:<34}" + "".join(f"  n={n}: {v:6.1f}" for n, v in r.items()))
    print("\nInterleaved S/Smax (mean +- std over 200 inits):")
    for a in ANSATZE:
        for i in INITS:
            for x in INPUTS:
                row = [cells_out[cell_key(n, a, i, x)] for n in QUBITS]
                print(f"  {a}/{i}/{x:<6}" + "".join(
                    f"  n={n}: {c['interleaved_ratio_mean']:.3f}±{c['interleaved_ratio_std']:.3f}"
                    for n, c in zip(QUBITS, row)))
    print("\nHalf-cut S/Smax means (locality exhibit; structurally 0 at n=4,8 for product inputs):")
    for a in ANSATZE:
        for x in INPUTS:
            row = [cells_out[cell_key(n, a, "uniform", x)] for n in QUBITS]
            print(f"  {a}/uniform/{x:<6}" + "".join(
                f"  n={n}: {c['halfcut_ratio_mean']:.3f}" for n, c in zip(QUBITS, row)))

    make_figures(cells_out, arms_out)
    print(f"\nWrote {RESULTS_DIR/'analysis.json'} and 4 figures to {FIG_DIR}/")


# ------------------------------------------------------------------ figures

def _loglog_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(QUBITS)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(LogLocator(subs=()))
    ax.set_xlabel("qubits $n$")


def _series(ax, y, color, label, ls="-", yerr=None):
    ax.plot(QUBITS, y, ls, color=color, lw=2, marker="o", ms=7,
            markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=label, zorder=3)
    if yerr is not None:
        ci = np.asarray(yerr)  # (4, 2) [lo, hi] absolute bounds
        ax.errorbar(QUBITS, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="none",
                    ecolor=color, elinewidth=1.2, capsize=3, alpha=0.6, zorder=2)


def fit_curve(n_arr, y_arr, fit):
    xs = np.linspace(min(n_arr), max(n_arr), 50)
    ly = np.log(y_arr)
    if fit["preferred"] == "polynomial":
        k = fit["poly_exponent_k"]
        a = np.mean(ly + k * np.log(n_arr))
        return xs, np.exp(a - k * np.log(xs))
    b = fit["exp_rate_b"]
    a = np.mean(ly + b * np.asarray(n_arr))
    return xs, np.exp(a - b * xs)


def make_figures(cells, arms):
    get = lambda a, i, x, f: np.array([cells[cell_key(n, a, i, x)][f] for n in QUBITS])

    # --- Fig 1: the Pesah test — uniform init, |0> input, Var vs n, both ansatze + fits
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for a in ANSATZE:
        y = get(a, "uniform", "zero", "var")
        f = arms[f"{a}_uniform_zero"]["fit_var"]
        _series(ax, y, ANSATZ_COLOR[a], ANSATZ_LABEL[a])
        xs, yf = fit_curve(QUBITS, y, f)
        ax.plot(xs, yf, ":", color=ANSATZ_COLOR[a], lw=1.5, zorder=2)
        if f["preferred"] == "polynomial":
            tag = f"$\\propto n^{{-{f['poly_exponent_k']:.1f}}}$  ($R^2$={f['poly_r2']:.2f})"
        else:
            tag = f"$\\propto e^{{-{f['exp_rate_b']:.2f}n}}$  ($R^2$={f['exp_r2']:.2f})"
        ax.annotate(f"{ANSATZ_LABEL[a]}: {tag}", xy=(QUBITS[-1], y[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=ANSATZ_COLOR[a], fontsize=9, va="center")
    _loglog_axes(ax)
    ax.set_ylabel(r"Var$[\partial C/\partial\theta_{\mathrm{center}}]$")
    ax.set_title("Gradient-variance scaling, uniform init, $|0\\rangle$ input (Pesah test)")
    ax.legend(loc="lower left")
    ax.set_xlim(3.6, 14.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_pesah_scaling.png")
    plt.close(fig)

    # --- Fig 2: encoding effect — per ansatz, uniform init, |0> vs MNIST
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, a in zip(axes, ANSATZE):
        for x in INPUTS:
            y = get(a, "uniform", x, "e_g2")
            _series(ax, y, ANSATZ_COLOR[a], INPUT_LABEL[x], ls=INPUT_STYLE[x],
                    yerr=get(a, "uniform", x, "e_g2_ci95"))
            ax.annotate(INPUT_LABEL[x], xy=(QUBITS[-1], y[-1]), xytext=(6, 0),
                        textcoords="offset points", color=ANSATZ_COLOR[a],
                        fontsize=9, va="center")
        _loglog_axes(ax)
        ax.set_title(ANSATZ_LABEL[a])
        ax.set_xlim(3.6, 13.5)
        ax.legend(loc="lower left")
    axes[0].set_ylabel(r"$\mathbb{E}[g^2]$")
    fig.suptitle("Amplitude-encoded MNIST input vs $|0\\rangle$ (uniform init)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_input_effect.png")
    plt.close(fig)

    # --- Fig 3: init mitigation — per ansatz, |0> input, uniform vs gauss (E[g2])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, a in zip(axes, ANSATZE):
        for i, ls in [("uniform", "-"), ("gauss0.1", "--")]:
            y = get(a, i, "zero", "e_g2")
            _series(ax, y, ANSATZ_COLOR[a], INIT_LABEL[i], ls=ls,
                    yerr=get(a, i, "zero", "e_g2_ci95"))
            ax.annotate(INIT_LABEL[i], xy=(QUBITS[-1], y[-1]), xytext=(6, 0),
                        textcoords="offset points", color=ANSATZ_COLOR[a],
                        fontsize=9, va="center")
        _loglog_axes(ax)
        ax.set_title(ANSATZ_LABEL[a])
        ax.set_xlim(3.6, 14)
        ax.legend(loc="lower left")
    axes[0].set_ylabel(r"$\mathbb{E}[g^2]$")
    fig.suptitle("Small-angle init as BP mitigation ($|0\\rangle$ input)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_init_mitigation.png")
    plt.close(fig)

    # --- Fig 4: interleaved entropy — uniform init, 4 series
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for a in ANSATZE:
        for x in INPUTS:
            y = get(a, "uniform", x, "interleaved_ratio_mean")
            s = get(a, "uniform", x, "interleaved_ratio_std")
            ax.fill_between(QUBITS, y - s, y + s, color=ANSATZ_COLOR[a], alpha=0.12, lw=0)
            _series(ax, y, ANSATZ_COLOR[a], f"{ANSATZ_LABEL[a]}, {INPUT_LABEL[x]}",
                    ls=INPUT_STYLE[x])
    ax.axhline(1.0, color=AXIS, lw=1, ls="-")
    ax.annotate("maximal (2-design regime)", xy=(4.1, 1.0), xytext=(0, 4),
                textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_xticks(QUBITS)
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"interleaved-cut $S/S_{\max}$")
    ax.set_ylim(0, 1.08)
    ax.set_title("Output-state entanglement (uniform init, mean ± std over 200 inits)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_entropy_interleaved.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
