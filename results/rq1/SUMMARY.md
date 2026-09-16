# RQ1 Results Summary — QCNN vs. Classical CNN

**Config**: 6 qubits, `qcnn_native` ansatz, amplitude encoding, 2 conv+pool layers, 50 epochs,
binary classification (classes 0 vs 1), `adjoint` diff method, 500 train / 200 test samples per seed.
QCNN: 74 params (68 circuit + 6 head). CNN: 71 params (parameter-matched, delta=3).

**RQ1 is now fully complete at n=10** — both datasets, all 10 seeds each (42–51). Seeds 42–46
ran locally; seeds 47–51 were added 2026-08-17 on a remote CPU session (Kaggle) after the
local run was paused on thermal grounds — a different CPU/BLAS and an unpinned `torch`
version (PennyLane/Lightning were pinned to match). See `../../../writing/thesis/
ch3_methodology.md` §3.5 for the full disclosure. **One seed (47) surfaced a genuine CNN
training failure** — see the dedicated section below before reading the per-seed tables.

---

## The seed-47 anomaly — read this first

Seed 47's CNN failed to train on **both** datasets: MNIST test_acc flat at 0.520 (train_acc
0.564), Fashion-MNIST flat at 0.450 (train_acc 0.508) — both essentially chance level for a
binary task, confirmed via the full per-epoch curves (flat from epoch ~2–3 onward), not a
one-off low reading. The seed's QCNN trained completely normally on both datasets in the
same run.

**Root cause, confirmed by instrumentation**: reconstructing the exact CNN
(`torch.manual_seed(47)`, `n_filters=1, hidden=7` — the architecture `make_matched_cnn`
selects to match the QCNN's 74-parameter budget) and instrumenting the first conv layer
shows a textbook **dying ReLU**. The dead-activation fraction climbs from 46.5% at epoch 1
to 67.4% by epoch 3 and plateaus; `conv1.weight.grad.norm()` is exactly `0.000000` from
epoch 1 onward — a true dead unit, not slow convergence, since ReLU's zero-gradient region
gives no path back. This is architecturally specific: matching a 74-parameter QCNN forces
`n_filters=1`, a single first-layer channel with **no redundancy** — with 2+ filters a
single dead channel would not necessarily kill the network. The QCNN has no equivalent
failure mode (unitary gates, no ReLU).

**How this is handled statistically**: per standard practice for an influential outlier
(verified against current methodology sources, not assumed), both a **primary analysis**
(all 10 seeds, no exclusions — the tables below) and a **documented sensitivity analysis**
(excluding seed 47) are reported. Full stats: `analysis.json` (primary),
`analysis_excl_seed47.json` (sensitivity), reasoning and the side-by-side numbers table:
`SENSITIVITY_seed47.md`. Headline: including seed 47, MNIST's previously-significant CNN
advantage flips to non-significant and reverses sign (paired t $p=0.567$, $d_z=+0.19$);
excluding it, the same nine seeds give a result that's not just restored but *stronger*
than the original 5-seed estimate ($p=0.0002$, $d_z=-2.12$). Fashion-MNIST's "no real
difference" conclusion is stable either way ($p=0.313$ vs. $p=0.615$).

---

## MNIST — COMPLETE (10/10 seeds)

| Seed | QCNN test_acc (final epoch) | QCNN train_acc | CNN test_acc (final epoch) | CNN train_acc |
|------|------------------------------|-----------------|------------------------------|-----------------|
| 42   | 0.970                        | 0.974           | 0.990                        | 0.992           |
| 43   | 0.965                        | 0.972           | 0.995                        | 0.988           |
| 44   | 0.970                        | 0.960           | 1.000                        | 1.000           |
| 45   | 0.985                        | 0.974           | 0.990                        | 0.988           |
| 46   | 0.985                        | 0.968           | 0.995                        | 0.988           |
| 47†  | 0.980                        | 0.974           | **0.520**                    | **0.564**       |
| 48   | 0.980                        | 0.972           | 1.000                        | 0.998           |
| 49   | 0.990                        | 0.972           | 1.000                        | 1.000           |
| 50   | 0.975                        | 0.972           | 1.000                        | 0.994           |
| 51   | 0.970                        | 0.974           | 0.995                        | 1.000           |

†CNN dying-ReLU failure — see above. Included in the mean/std below (primary analysis);
excluded in the sensitivity analysis.

**Final mean ± std (10 seeds, primary)**: QCNN **0.9770 ± 0.0082** · CNN **0.9485 ± 0.1506**
**Final mean ± std (9 seeds, excl. seed 47)**: QCNN **0.9767 ± 0.0087**$^*$ · CNN **0.9961 ± 0.0042**

*All standard deviations in this file are sample (ddof=1) estimators, matching
`analysis.json`, `analysis_excl_seed47.json` and the thesis (§3.5). Until 2026-08-25 the
primary rows used the population (ddof=0) estimator while the sensitivity rows used the
sample one, so a single block mixed the two; the convention was unified.*

### Overall MNIST takeaway

- **Nine of the ten seeds repeat the original pattern exactly**: the CNN beats the QCNN by
  a consistent 0.5–3.0 points. Seed 47 is the sole exception, and a diagnosed one (above) —
  not a fluke to shrug off, but not representative of the other nine either.
- The QCNN's own seed-to-seed spread is low and stable (std 0.0082 across all ten,
  unaffected by seed 47 since its own QCNN run trained normally): the ten values cluster
  tightly in 0.965–0.990, without the bimodal-looking split the original 5-seed sample
  suggested — more seeds diluted what had looked like two local-optima clusters into a
  simple tight spread.
- The QCNN's slow cold-start (chance-level accuracy for the first few epochs before a
  sharp jump) appears on every seed, taking 4–6 epochs to clear the 0.80 threshold (mean
  4.9); the worst case is seed 47 at 6 epochs — the same seed whose CNN never escaped at
  all, suggesting that seed's fixed data split is mildly harder for both architectures,
  though only the CNN's narrow capacity turned that into a permanent failure. The CNN,
  when it trains, typically clears the threshold within 2–4 epochs.
- Net conclusion: **excluding the diagnosed seed-47 failure, the classical CNN is both
  more accurate and more consistent** at this parameter budget (~70–75 params) and qubit
  count (6) — more so than the original 5-seed estimate suggested (paired t $p=0.0002$,
  $d_z=-2.12$ vs. the original $p=0.020$, $d_z=-1.67$) — but the QCNN is not far behind
  (within ~2 points on the seeds that both models handle normally) and does solve the task
  well above chance.

### Seed 45 note

QCNN's second-best seed here (0.985 test_acc); its train_acc (0.974) is close to test_acc,
not a case of overfitting. **Correction (2026-08-17): earlier drafts of this note claimed
seed 45 had the QCNN's slowest cold-start ("flat for 7 epochs, jump at epoch 8") — checked
against the actual per-epoch data and that is wrong.** Seed 45's QCNN crosses the 0.80
threshold at epoch 5; the "epoch 8" figure was seed 45's **CNN** cold-start (8 epochs, the
CNN's slowest of any successfully-trained seed on MNIST), mislabeled as the QCNN's in the
original note. The genuinely slowest QCNN cold-start on MNIST is seed 47, at 6 epochs (see
the anomaly section above).

### Seed 44 note

The CNN hit perfect train/test accuracy (1.0/1.0) — likely a case where seed 44's data
split happens to be linearly/near-linearly separable for the CNN's feature space, not
evidence the CNN is "better" at this seed specifically. The QCNN did not show the same
jump to perfect separation (capped at 0.97), consistent with a smoother, less "spiky"
decision boundary given its more constrained circuit structure.

### What this indicates

- **Both models solve the task well above chance on every seed that trained** — MNIST
  0-vs-1 is an easy binary task, so near-ceiling accuracy from both architectures isn't
  surprising by itself.
- **CNN is consistently ahead of the QCNN at convergence** on the nine seeds that trained
  normally, and this gap strengthens rather than weakens with more data (§ above) — with
  only 74 vs 71 parameters, the CNN's inductive bias (local spatial convolution +
  translation-equivariant filters) appears to be a better fit for image data than the
  QCNN's amplitude-encoded, pairwise-unitary structure at this scale.
- **QCNN trains more slowly early on**: epoch-1 test_acc is ~0.48–0.53 (near chance) on
  every seed, taking 4–6 epochs to climb past 0.80. This is the same barren-plateau-adjacent
  optimization signature noted in earlier design notes — amplitude encoding avoids the
  worst of it (unlike angle encoding), but the QCNN still has a slower effective learning
  rate in the first few epochs even with LR=1e-2.
- **The CNN's failure mode is binary, not continuous**: when its single-filter first layer
  survives initialisation, it converges fast and often to near-perfect accuracy; when that
  filter dies (seed 47), it never recovers. This is a materially different statistical
  object than the QCNN's smooth, low seed-to-seed spread — part of why the naive combined
  std comparison is misleading without the sensitivity treatment above.
- **A "fair" comparison on parameter count is not automatically a fair comparison on
  trainability risk.** Constraining the CNN to match the QCNN's small parameter budget
  forced a single-filter first layer with no redundancy against dying ReLU — a purely
  classical pathology the QCNN structurally cannot exhibit. Worth carrying into the
  thesis's discussion chapter as a genuine methodological finding.

---

## Fashion-MNIST — COMPLETE (10/10 seeds)

| Seed | QCNN test_acc (final epoch) | QCNN train_acc | CNN test_acc (final epoch) | CNN train_acc |
|------|------------------------------|-----------------|------------------------------|-----------------|
| 42   | 0.950                        | 0.944           | 0.925                        | 0.950           |
| 43   | 0.950                        | 0.936           | 0.935                        | 0.958           |
| 44   | 0.950                        | 0.950           | 0.955                        | 0.966           |
| 45   | 0.945                        | 0.942           | 0.960                        | 0.960           |
| 46   | 0.960                        | 0.952           | 0.960                        | 0.952           |
| 47†  | 0.950                        | 0.950           | **0.450**                    | **0.508**       |
| 48   | 0.950                        | 0.942           | 0.970                        | 0.962           |
| 49   | 0.940                        | 0.946           | 0.945                        | 0.972           |
| 50   | 0.950                        | 0.942           | 0.960                        | 0.964           |
| 51   | 0.950                        | 0.936           | 0.900                        | 0.942           |

†Same CNN dying-ReLU failure as MNIST — confirms it is a property of the seed's weight
initialisation interacting with the CNN's architecture, not a dataset-specific fluke.

**Final mean ± std (10 seeds, primary)**: QCNN **0.9495 ± 0.0050** · CNN **0.8960 ± 0.1581**
**Final mean ± std (9 seeds, excl. seed 47)**: QCNN **0.9494 ± 0.0053**$^*$ · CNN **0.9456 ± 0.0221**$^*$

*$^*$Sample (ddof=1) std, matching `analysis_excl_seed47.json`.*

### Overall Fashion-MNIST takeaway

- **The seed-47 CNN failure recurs here too** (0.450, again roughly chance) — the clearest
  evidence that this is a seed/architecture interaction, not an MNIST-specific artefact.
- Across the other nine seeds the picture matches the original five-seed finding closely:
  QCNN and CNN stay close, with per-seed ranking flipping back and forth (QCNN ahead on
  42/43/49, CNN ahead on 44/45/46/48/50/51) rather than one model dominating — "close, no
  clear winner" is the right read, and this conclusion is **stable at n=10** regardless of
  whether seed 47 is included (paired t $p=0.313$ primary vs. $p=0.615$ sensitivity — both
  clearly non-significant, small effect sizes both ways).
- **QCNN variance stays low and stable** (std 0.0050, materially unaffected by seed 47
  since its own run trained normally). The CNN's descriptive spread is dominated by the
  same single outlier that distorts MNIST; excluding it, the CNN's std (0.0221) is closer
  to but still somewhat above the QCNN's — a milder version of the recurring overfitting
  signature noted below, not another dying-ReLU case.
- **Contrast with MNIST**: on MNIST, once seed 47 is accounted for, the CNN has a real,
  strengthening advantage; on Fashion-MNIST the two are genuinely statistically
  indistinguishable at n=10, not merely underpowered to detect a difference as the n=5
  estimate left open. This suggests the CNN's inductive-bias advantage seen on MNIST
  doesn't transfer as cleanly to a harder, more textured task — plausibly because
  Fashion-MNIST's higher intra-class variation gives the CNN's larger effective capacity
  more surface area to overfit on just 500 training samples, while the QCNN's more
  constrained circuit is less prone to that failure mode.

---

## Overall RQ1 Verdict (both datasets, all 20 seed-runs)

1. **No universal winner, and the margin of that verdict is now better understood.** On
   MNIST, once the diagnosed seed-47 training failure is accounted for, the CNN has a real,
   *strengthening* advantage (not merely a five-seed suggestion). On Fashion-MNIST, ten
   seeds confirm genuine statistical parity, not just insufficient power. A single-dataset
   comparison would still have pointed in opposite directions depending on which dataset
   was chosen — the two-dataset design remains the main empirical payoff of this RQ.
2. **Variance is not a fixed property of either architecture — and part of what looked
   like variance was a discrete failure, not spread.** The QCNN's seed-to-seed spread is
   low and stable on both datasets. The CNN's descriptive spread is dominated by a single
   binary outcome (trains normally, or doesn't) rather than continuous variation — these
   are not comparable statistical objects without the sensitivity treatment above.
3. **QCNN's barren-plateau-adjacent slow cold-start** is present on both datasets at a
   similar magnitude (mean 4.8–4.9 epochs to threshold) — worth investigating further
   whether encoding/task interaction affects when it bites hardest (RQ2/RQ3 territory).
4. **CNN overfitting is a bigger factor on Fashion-MNIST than MNIST** — small-to-negligible
   train/test gap on MNIST (excluding seed 47) but a persistent one on Fashion-MNIST,
   suggesting the classical model's capacity advantage is a double-edged sword depending on
   data complexity relative to the fixed ~500-sample training set size.
5. **Parameter-matching two architectures for capacity fairness does not guarantee
   fairness of failure mode.** The 71-parameter CNN's single-filter first layer has no
   redundancy against dying ReLU — a purely classical pathology the QCNN structurally
   cannot exhibit. A genuine methodological finding for the discussion chapter, not noise
   to explain away.
6. **Both models solve both tasks well above chance on every seed that trained** (all such
   final accuracies ≥ 0.90) — the story is about the shape, consistency, and failure modes
   of the comparison, not a pass/fail result for either architecture.

Full per-epoch trajectories are preserved in `mnist_checkpoint.json` and
`fashion_mnist_checkpoint.json` (and the final `mnist.json` / `fashion_mnist.json` outputs)
for plotting learning curves — see `figures/fig1_learning_curves.png` and
`figures/fig2_paired_seeds.png`.

---

## Notes / caveats

- Values above are single point estimates from the final training epoch, not best-epoch or
  early-stopped values — worth keeping in mind since a couple of seeds show late-epoch
  noise in both models.
- Seeds 42–46 ran locally; seeds 47–51 ran on Kaggle (different CPU/BLAS, unpinned `torch`
  version) — disclosed in `ch3_methodology.md` §3.5. Seed-level statistics are unaffected
  (they characterise stochastic training variation, not deterministic reproducibility).
- The MNIST paired tests are sensitive to the single diagnosed seed-47 outlier — handled by
  reporting primary and sensitivity analyses together (`SENSITIVITY_seed47.md`) rather than
  a single point estimate.
- This summary reflects the completed n=10 extension (2026-08-17). The thesis text itself
  is at `writing/thesis/ch4_results.md` §4.1, which carries the same framing in full prose.
