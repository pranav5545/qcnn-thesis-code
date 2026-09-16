# RQ3 Results Summary — Barren Plateaus: QCNN vs HEA

**Data**: collected 2026-07-08 (`rq3_barren_plateau.py` + `rq3_entropy_cut.py`), analysed
2026-07-15 (`rq3_analysis.py`). 32 cells: qubits {4,6,8,10} × ansatz {qcnn_native, HEA} ×
init {uniform[-π,π], N(0,0.1²)} × input {|0⟩, amplitude-encoded MNIST '0'} × **200 inits**,
no training. Protocol per Pesah 2021 (D1–D8 in `writing/thesis/rq3_design_decisions.md`):
Var[∂C/∂θ] of one fixed center-conv parameter, local ⟨Z⊗Z⟩ observable, pool-to-2 depth.

**Metric**: scaling fits use **E[g²]** (raw second moment; = Var at mean-zero inits). Per the
D5 caveat the gauss arm's init mean ≠ 0, so Var understates its signal. Var is reported in
`analysis.json` alongside; for every uniform arm the two agree to 3 significant figures.

**Fits**: polynomial `ln y = a − k·ln n` vs exponential `ln y = a − b·n`, least squares on
4 support points; equal parameter count, so ΔAIC = 4·ln(RSS_exp/RSS_poly) (> 0 favors
polynomial). CIs and P(poly) from a B=1000 bootstrap resampling the 200 per-init gradients
within each cell. **With 4 qubit counts the model comparison has low power** — treat
"preferred" as descriptive except where ΔAIC is large; the bootstrap quantifies gradient-
sampling noise, not model identifiability.

---

## Headline table (E[g²], all 8 arms)

| arm (ansatz / init / input) | n=4 | n=10 | decay | poly k [CI95] | R²p | exp b [CI95] | R²e | ΔAIC | P(poly) |
|---|---|---|---|---|---|---|---|---|---|
| QCNN / uniform / \|0⟩ | 2.75e-2 | 1.93e-3 | 14× | **2.74 [2.31, 3.22]** | 0.97 | 0.420 | 0.97 | +0.4 | 0.55 |
| QCNN / uniform / MNIST | 3.31e-2 | 3.31e-4 | 100× | 4.90 | 0.98 | **0.759 [0.70, 0.82]** | 1.00 | **−11.2** | 0.02 |
| QCNN / gauss / \|0⟩ | 1.82e-2 | 4.27e-4 | 43× | 2.21 | 0.13 | 0.332 | 0.12 | +0.0 | 0.95 |
| QCNN / gauss / MNIST | 1.63e-2 | **3.25e-6** | 5018× | 7.19 | 0.54 | **1.177 [1.11, 1.25]** | 0.61 | −0.7 | 0.00 |
| HEA / uniform / \|0⟩ | 3.07e-3 | 2.43e-4 | 13× | 2.90 | 0.59 | 0.486 [0.36, 0.62] | 0.70 | −1.3 | 0.00 |
| HEA / uniform / MNIST | 8.73e-3 | 4.08e-4 | 21× | 3.54 | 0.97 | 0.532 | 0.94 | +3.6 | 0.94 |
| HEA / gauss / \|0⟩ | 9.20e-3 | 6.65e-3 | **1.4×** | 0.31 | 0.77 | **0.045 [−0.001, 0.094]** | 0.73 | +0.7 | 0.65 |
| HEA / gauss / MNIST | 9.78e-3 | 1.52e-4 | 64× | 4.79 | 0.94 | 0.703 | 0.86 | +3.2 | 1.00 |

## Verdicts on the three hypotheses

### H1 — Pesah's polynomial-scaling claim (QCNN, uniform init, |0⟩ input)

**Consistent with polynomial decay; formally indistinguishable from exponential at n ≤ 10.**
The canonical arm decays only 14× over n = 4→10 with a clean power-law read
(k = 2.74 [2.31, 3.22], R² = 0.97) — but the exponential fit is equally good
(ΔAIC = +0.4, bootstrap P(poly) = 0.55). Four support points cannot separate n^−2.7 from
e^−0.42n over this range; what CAN be said is quantitative: at n = 10 the QCNN's gradient
signal (1.9e-3) sits ~8× above the matched-depth HEA (2.4e-4) and ~600× above the frozen
regime (3e-6, see H2), and the decay *rate* is modest. The HEA contrast arm prefers
exponential in every bootstrap resample (P(poly) = 0.00) despite noisy fits (R² ≤ 0.70,
non-monotonic n=4→6). Honest thesis claim: *"consistent with Pesah's polynomial guarantee
and clearly separated from the HEA contrast, but discrimination of the functional form
would require larger n"* — extending past n = 10 is a statevector-cost decision for ch5
future work, not a gap in this protocol.

### H2 — Entanglement-induced BP from amplitude-encoded data (the strongest result)

**Confirmed, and it is the sharpest effect in the grid.** Feeding a fixed amplitude-encoded
MNIST digit instead of |0⟩ flips the QCNN/uniform arm from ambiguous-polynomial to
*decisively exponential*: b = 0.76 [0.70, 0.82], R² = 1.00, ΔAIC = −11.2, P(poly) = 0.02,
decay 100× (vs 14×). Mechanism check agrees: interleaved-cut S/S_max jumps from a flat
~0.44–0.47 (|0⟩, any n) to 0.70–0.77 at n ≥ 6, i.e. the data input pushes the circuit
toward the high-entanglement regime Pesah's input-state assumption excludes. Combined with
small-angle init (the training pipeline's default!) the effect compounds:
QCNN/gauss/MNIST at n = 10 is E[g²] = 3.3e-6 [2.1e-6, 4.8e-6], mean|g| = 1.1e-3 —
**near-frozen**. The architecture's BP protection is real but *conditional on the input
state*, and our own encoding choice erodes it.

**Anchor for RQ1**: RQ1's exact training configuration (n = 6, native, gauss, MNIST) sits
at E[g²] = 1.10e-3, mean|g| = 2.5e-2 — small but workable gradients, consistent with RQ1's
observed slow-cold-start-then-recover behavior. The same arm at n = 10 is 340× smaller:
RQ1's setup would NOT scale to 10 qubits without changing encoding or init.

### H3 — Small-angle init N(0, 0.1²) as mitigation

**Rescues the HEA almost completely; does nothing reliable for the QCNN.** HEA/gauss/|0⟩ is
statistically flat: decay 1.4× over the whole range, exponential rate b = 0.045 with a CI
that includes zero [−0.001, 0.094] — at matched depth, small-angle init removes the HEA's
decay entirely (gauss/uniform ratio at n = 10: **27×**). For the QCNN the picture inverts:
gauss/uniform ≤ 1 at n = 6, 10 (0.04×, 0.22×), i.e. the QCNN gains nothing (it has no
plateau to mitigate) and mostly just loses gradient magnitude. With MNIST input, gauss init
makes things strictly worse for both ansätze (ratios 0.01–0.02 at n = 10) — mitigation and
encoding-induced BP do not compose.

**Parity structure, not noise**: QCNN/gauss/|0⟩ oscillates with n — high at n = 4, 8
(1.8e-2, 3.3e-2), low at n = 6, 10 (3.3e-4, 4.3e-4) — with tight bootstrap CIs, and R²
collapses (0.13). This is the same n = 4, 8 vs n = 6, 10 split as the half-cut-entropy
wiring parity (D8): near the identity point, whether the target parameter's causal cone
reaches the measured qubits depends on the pool-wiring parity at each n. Flag for ch4
discussion; it means the gauss-arm scaling fit is not meaningful for the QCNN.

## Entanglement diagnostics (mechanism layer)

Interleaved-cut S/S_max, mean ± std over 200 inits (uniform init):

| arm | n=4 | n=6 | n=8 | n=10 |
|---|---|---|---|---|
| QCNN / \|0⟩ | 0.468 ± 0.183 | 0.450 ± 0.152 | 0.473 ± 0.133 | 0.459 ± 0.113 |
| QCNN / MNIST | 0.443 ± 0.200 | 0.731 ± 0.038 | 0.705 ± 0.036 | 0.766 ± 0.028 |
| HEA / \|0⟩ | 0.445 ± 0.193 | 0.444 ± 0.153 | 0.444 ± 0.146 | 0.444 ± 0.126 |
| HEA / MNIST | 0.462 ± 0.184 | 0.740 ± 0.038 | 0.696 ± 0.039 | 0.753 ± 0.027 |

- **Flat ~0.44–0.47 for |0⟩ input at every n and both ansätze** — neither circuit reaches
  the 2-design regime (S/S_max → 1) at pool-to-2 depth. This is why the HEA doesn't show a
  textbook-catastrophic plateau here: at matched (log) depth the HEA arm is a *shallow
  unstructured* ansatz, not the deep 2-design of McClean 2018. The exponential preference
  it shows anyway (P(poly)=0.00) is the onset, not the asymptote.
- **MNIST input lifts S/S_max to 0.70–0.77 for n ≥ 6** regardless of ansatz — the
  entanglement is imported by the encoded data, not generated by the circuit; n = 4 stays
  low because the 4×4-resized digit is nearly a product state.
- Gauss init + |0⟩: S/S_max ≈ 0.04 (near-product, as designed); gauss + MNIST ≈ 0.62–0.71 —
  input entanglement dominates init entanglement.
- Half-cut entropy retained separately as the causal-cone locality exhibit (structurally 0
  at n = 4, 8 for product inputs — D8).

## Caveats

- **4 qubit counts** is the binding constraint on every functional-form claim; n = 12–16
  native-arm extension is the single highest-value follow-up (statevector sim cost rises
  16× per +2 qubits; feasible to ~14 on this machine, untested — and MemTest86 first).
- Single fixed MNIST digit (a '0'), per the fixed-input convention (D6); different digits
  → different input entanglement; a small robustness check (2–3 digits at n = 8) would be
  cheap.
- HEA arm is depth-matched to the QCNN (D4), so it tests "unstructured ansatz at equal
  depth", not the deep-HEA worst case.
- Bootstrap CIs capture init-sampling noise only; the 4-point fits share no error model.
- Values quoted from `analysis.json` (regenerate: `python experiments/rq3_analysis.py`
  from `code/`, conda env `thesis`).

## Files

- `analysis.json` — all per-cell stats (Var, E[g²]+CI95, mean|g|, entropies), per-arm fits
  + bootstrap, suppression/mitigation ratio tables. `analysis_report.txt` — console dump.
- `figures/fig1_pesah_scaling.png` — H1 headline (uniform/|0⟩, both ansätze, fits annotated)
- `figures/fig2_input_effect.png` — H2 (|0⟩ vs MNIST per ansatz, CI whiskers)
- `figures/fig3_init_mitigation.png` — H3 (uniform vs gauss per ansatz; parity zig-zag visible)
- `figures/fig4_entropy_interleaved.png` — mechanism layer (S/S_max vs n)
- Raw data: `gradient_variance.json` (per-init full gradient vectors — any statistic
  recomputable), `entropy_interleaved.json`, `checkpoint.json`, `run.log`.
