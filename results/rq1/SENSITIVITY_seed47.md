# Sensitivity note: seed 47's CNN dying-ReLU failure

**Status**: documented robustness check, not a data exclusion. The primary n=10 analysis
(`analysis.json`/`analysis_report.txt`, all 10 seeds, no exclusions) remains the dataset of
record. This note and `analysis_excl_seed47.{json,_report.txt}` exist alongside it as a
transparent secondary analysis, per the standard practice below.

## What happened

Seed 47's classical CNN comparator got stuck at chance-level accuracy on **both** datasets:
MNIST flat at test_acc=0.52 from epoch ~2 onward, Fashion-MNIST flat at 0.45 from epoch ~3
onward (confirmed via the full per-epoch `test_acc` arrays, not just the final number — see
`changes_made.md`'s 2026-08-17 entry). Seed 47 is new to the 10-seed extension; the original
5-seed run (42–46) never encountered it.

## Root cause — confirmed, not assumed

Diagnosed by reconstructing seed 47's exact CNN (same `torch.manual_seed(47)` init, same
architecture: `n_filters=1, hidden=7`, from `make_matched_cnn`'s parameter-matching search)
and instrumenting the first conv layer:

- Initial `conv1` weights/bias push most of the image below zero pre-ReLU.
- Fraction of dead (zero) post-ReLU activations climbs from 46.5% (epoch 1) to 67.4% by
  epoch 3, then plateaus.
- `conv1.weight.grad.norm()` is **exactly 0.000000** from epoch 1 onward — a true dead
  unit, not slow convergence. ReLU's zero-gradient region gives no path back once the
  sole channel dies.

This is the textbook **dying ReLU** failure mode, and it is architecturally specific to
this experiment's fairness protocol: parameter-matching the CNN to the QCNN's 74-parameter
budget forces `n_filters=1` — a single first-layer channel with **no redundancy**. With
2+ filters, one dying channel wouldn't necessarily kill the network; with exactly 1, it
does. The QCNN has no equivalent failure mode (no ReLU, unitary gates only) — this is a
classical-side-only pathology, not a symmetric one.

**Framing for ch5**: this is a genuine, citable methodological finding, not noise to
explain away. Parameter-matching two architectures for a "fair" capacity comparison can
still leave one side more fragile to its *own* class of trainability failure — a real
asymmetry worth discussing alongside the QCNN's own barren-plateau-adjacent "slow
cold-start" story (`STUDY_GUIDE.md` §D.3), as a parallel but architecturally distinct
trainability risk on the classical side.

## Why a sensitivity check, not silent exclusion

Verified 2026-08-17 (WebSearch against current methodology sources, not assumed): the
standard, recommended practice for an influential/outlier observation is to **report both
the primary analysis (all data) and a sensitivity analysis excluding the outlier**,
explicitly, rather than silently dropping the point. A result that is "significant
including the outlier, non-significant excluding it" (or vice versa, as here) is flagged
in the methodology literature as something to disclose directly, precisely because it
shows how fragile the primary conclusion is to one observation. Silent exclusion —
reporting only the cleaner number — is the practice explicitly warned against.

## Primary vs. sensitivity, side by side

| | Primary (n=10, all seeds) | Sensitivity (n=9, excl. seed 47) | Original (n=5, pre-extension) |
|---|---|---|---|
| **MNIST** mean diff | +0.029 (favours QCNN) | **−0.019** (favours CNN) | −0.019 (favours CNN) |
| MNIST paired t | p=0.567 | **p=0.0002** | p=0.020 |
| MNIST Cohen's dz | +0.19 | **−2.12** | −1.67 |
| **Fashion-MNIST** mean diff | +0.054 (favours QCNN) | +0.004 (favours QCNN) | ~0 (favours neither) |
| Fashion-MNIST paired t | p=0.313 | p=0.616 | p=0.61 |

Reading this: Fashion-MNIST's conclusion ("no real difference") is **stable** across all
three views — the 10-seed extension did what it was meant to do there. MNIST's conclusion
is **not** stable at n=10-with-seed-47 (flips to non-significant, even reverses direction),
but excluding the one seed with a confirmed, understood training failure **restores and
strengthens** the original CNN-advantage finding (p=0.0002 vs. the original p=0.020) — nine
of ten seeds tell a highly consistent story; one doesn't, and now we know precisely why.

## Recommendation for the ch4 §4.1 rewrite

Report the primary n=10 numbers as the headline (per lit-review/data-integrity discipline
— no seed gets silently dropped from the dataset of record), but present the sensitivity
result and this diagnosis prominently alongside it, not buried in a footnote. The honest
summary sentence: *"At n=10, MNIST's result is unstable to a single seed on which the CNN
comparator suffered a diagnosed dying-ReLU failure; excluding that seed, the original
CNN-advantage finding is confirmed and its significance strengthens (p=0.0002, dz=−2.12)."*
That is a stronger, more defensible claim than either number alone.
