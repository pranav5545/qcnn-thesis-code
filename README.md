# Quantum Convolutional Neural Networks versus Parameter-Matched Classical CNNs

Code and results for an MSc Data Science dissertation — Middlesex University, CST4990,
September 2026.

**Author:** Pranav Sharma

The question is narrow on purpose: given the **same number of trainable parameters**, does a
quantum model's inductive bias buy anything on small image tasks? A QCNN with 74 parameters
is compared against a classical CNN matched to it at 71, on binary MNIST and Fashion-MNIST.

Everything runs on **CPU**. No quantum hardware is needed, and no GPU — a GPU was measured
to be *slower* here, because these circuits are simulated one sample at a time.

---

## 1. Install

Python 3.11 is required (3.12+ will not resolve the pinned PennyLane build).

```bash
git clone <this repository>
cd qcnn-thesis-code

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Conda works equally well:

```bash
conda create -n qcnn python=3.11 -y && conda activate qcnn
pip install -r requirements.txt
```

## 2. Check the install works

```bash
python experiments/smoke_test.py
```

Takes about a minute. It builds a QCNN and its matched CNN, trains both for two epochs on
64 examples, and runs angle and amplitude encodings. Expect to see:

```
  QCNN params: 74  (circuit: 68)
  CNN  params: 71  (delta: 3)
  ...
[OK] All smoke tests passed.
```

The parameter counts are the point — `74` and `71` are the matched pair the whole
dissertation rests on. If those two numbers appear, the environment is correct.

Datasets (~150 MB) download automatically on first use. To fetch them deliberately first:

```bash
python scripts/download_data.py
```

**Run everything from the repository root**, not from inside `experiments/`.

---

## 3. Verifying the published results *without* re-training

This is the fast path, and the one worth taking first.

`results/` already contains the output of every run. The `*_analysis.py` scripts read those
files and regenerate the statistics — every table in the dissertation's results chapter comes
out of them. None of them trains anything, so each finishes in seconds:

```bash
python experiments/rq1_analysis.py          # QCNN vs matched CNN, per seed
python experiments/rq2_analysis.py          # encoding / ansatz / depth ablation
python experiments/rq3_analysis.py          # gradient variance vs qubit count
python experiments/rq4_analysis.py          # distillation ladder
python experiments/rq4_reeval_analysis.py   # full-test-split re-evaluation
```

Each prints its tables and rewrites `results/<rq>/analysis.json` and
`analysis_report.txt`. Re-running them is safe: the inputs are the raw run files, which the
analysis never modifies. Comparing a regenerated `analysis_report.txt` against the copy
shipped here is a direct check that the reported numbers follow from the recorded data.

Four of the five regenerate **byte-identically**. The exception is
`rq4_reeval_analysis.py`, whose bootstrap resampling is not seeded: its point estimates
(accuracy, agreement, KL) reproduce exactly, while the confidence-interval bounds move in
the fourth decimal place between runs — `[0.9812, 0.9875]` becoming `[0.9811, 0.9875]`, for
example. That is the bootstrap behaving normally, not a discrepancy, and it is too small to
touch any conclusion. The files shipped here hold the values the dissertation quotes.

Figure PNGs also differ byte-for-byte when re-rendered, while being visually identical.

`results/rq1/SUMMARY.md` and `results/rq3/SUMMARY.md` are readable write-ups of what those
two experiments found, including the seed-47 CNN training failure and how it was handled.

---

## 4. Re-running the experiments from scratch

Only necessary to reproduce the raw data itself. **Measured wall-clock on the CPU that
produced these results** (totalled from the `epoch_time` fields in the checkpoints):

| | Question | Command | Measured cost |
|---|---|---|---|
| **RQ3** | Do gradients vanish as qubits grow? | `python experiments/rq3_barren_plateau.py` | **5.5 min** |
| **RQ1** | Does a QCNN beat a matched CNN? | `python experiments/rq1_performance.py` | **2h 31m** |
| **RQ4** | Can a classical student absorb the QCNN's function? | see below | **1h 41m** |
| **RQ2** | Which design choices actually move accuracy? | `python experiments/rq2_design_choices.py` | **12h 57m** |

**Start with RQ3.** It trains nothing — it samples 200 random initialisations at 4, 6, 8 and
10 qubits and measures gradient variance — so it reproduces a headline result of the
dissertation end to end in about five minutes.

RQ4 has an order, because each stage consumes the previous one's output:

```bash
python experiments/rq4_teachers.py        # 1h 29m — train the 20 QCNN teachers
python experiments/rq4_distillation.py    # 12m    — distil classical students
python experiments/rq4_reeval.py          #        — re-evaluate on the full test split
python experiments/rq4_analysis.py
python experiments/rq4_reeval_analysis.py
```

**RQ1 and RQ2 are resumable.** They checkpoint after every (configuration, seed) cell into
`results/<rq>/<dataset>_checkpoint.json` and skip completed cells on restart, so they can be
interrupted with Ctrl-C and continued later. That also means **they will skip everything if
the shipped checkpoints are present** — to force a genuine re-run, move the existing
`*_checkpoint.json` aside first.

Supporting scripts: `check_gradients.py` and `check_lr.py` are diagnostics,
`rq3_entropy_cut.py` recomputes entanglement entropy on the interleaved cut,
`make_circuit_figure.py` draws the circuit diagram, and `rq2_probe_iqp_rescale.py` /
`rq2_check_iqp_encoding_path.py` are the two IQP probes discussed in the dissertation.

---

## 5. Layout

```
config.py                 every shared constant: qubits, seeds, epochs, learning rate
models/
  qcnn.py                 the QCNN: convolution and pooling blocks
  classical_cnn.py        the CNN, and make_matched_cnn() which does the parameter match
  encodings.py            angle, amplitude and IQP state preparation
utils/
  dataset.py              loaders, 8x8 downsampling, hand-rolled PCA
  training.py             the single training loop every experiment shares
  distill.py              knowledge distillation (RQ4)
  metrics.py              accuracy, agreement, KL
  corruptions.py          rotation, noise and occlusion (RQ4 robustness)
experiments/              one entry point per research question, plus its analysis
results/                  every number in the dissertation, as produced
scripts/download_data.py  pre-fetch the datasets
data/                     the datasets (not in git; see below)
```

---

## 6. Why `results/` ships, and `data/` does not

`results/` is **tracked deliberately**, ~46 MB. Re-running everything costs over seventeen
hours of CPU, and RQ2 alone is thirteen. Shipping code without results would mean the
dissertation's numbers could not be checked without spending that, which is not a reasonable
thing to ask of a reader.

`data/` is **not tracked**. MNIST and Fashion-MNIST are public, byte-identical for everyone,
and re-fetched in about a minute. Excluding them is what holds the repository at ~46 MB
instead of ~200 MB. *(The archive submitted for assessment does include them, so it is
self-contained offline.)*

`results/rq2/archive_pre_redesign_2026-08-18/` is kept on purpose. It holds superseded runs
from before a design fault was found: angle and IQP encodings were receiving the first six of
sixty-four pixels — near-constant background — rather than six PCA components. Those numbers
are wrong and are retained so the correction is visible rather than silent.

---

## 7. Reproducibility

Seeds are fixed in `config.py` (`SEED = 42`, ten seeds for RQ1). The data subset is fixed
across seeds *independently of the run seed*, so changing a seed changes initialisation and
shuffling but never which examples are used.

Bit-exact reproduction across machines is **not** promised, and the dissertation says so. The
twenty RQ4 teacher models reproduce RQ1's final-epoch accuracies exactly while their test
losses differ by at most 2.05e-07; `experiments/rq4_cross_machine_stability.py` records that
measurement. Seeds 47–51 of RQ1 were run on different hardware, which is disclosed in §3.5
and Appendix G of the dissertation.

The pinned environment is in `requirements.txt`: Python 3.11, PennyLane 0.44.1 with
`pennylane-lightning` 0.44.0, PyTorch 2.12.1. PennyLane 0.45 is deliberately *not* used — it
crashed natively on this workload. `scikit-learn` is deliberately absent; PCA is hand-rolled
in `utils/dataset.py` to keep the dependency list short.

Code comments cite the dissertation's appendices for design decisions — Appendix E is the
extended methodology, Appendix F the design-decision records.

---

## 8. Licence

MIT, see `LICENSE`. The datasets are MNIST (LeCun et al., 1998) and Fashion-MNIST (Xiao et
al., 2017), distributed under their own terms and not redistributed in this repository.
