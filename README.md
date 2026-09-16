# Quantum Convolutional Neural Networks versus Parameter-Matched Classical CNNs

Code and results for an MSc Data Science dissertation (Middlesex University, CST4990,
September 2026) comparing quantum convolutional neural networks against classical CNNs
matched to them **parameter for parameter**, on small binary image tasks.

The question is narrow on purpose: given the same number of trainable parameters, does a
quantum model's inductive bias buy anything at a scale that fits on a simulator?

Everything here runs on CPU. No quantum hardware and no GPU is required — a GPU was
measured to be *slower* for these unbatched circuits.

---

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python experiments/smoke_test.py                 # ~1 minute, verifies the pipeline
```

The smoke test builds a QCNN and its matched CNN, trains both for two epochs on 64
examples, and prints parameter counts and accuracies. If it completes, the environment is
good. Datasets download automatically on first use (~150 MB); `python
scripts/download_data.py` fetches them ahead of time instead.

Run everything from the repository root, not from inside `experiments/`.

---

## Layout

```
config.py                 every shared constant: qubits, seeds, epochs, learning rate
models/
  qcnn.py                 the QCNN: encodings, convolution and pooling blocks
  classical_cnn.py        the CNN, plus make_matched_cnn() which does the parameter match
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
```

---

## The four experiments

| | Question | Entry point | Analysis | Cost |
|---|---|---|---|---|
| **RQ1** | Does a QCNN beat a parameter-matched CNN? | `rq1_performance.py` | `rq1_analysis.py` | hours |
| **RQ2** | Which design choices actually move accuracy? | `rq2_design_choices.py` | `rq2_analysis.py` | ~9 h |
| **RQ3** | Do gradients vanish as qubits grow? | `rq3_barren_plateau.py` | `rq3_analysis.py` | ~6 min |
| **RQ4** | Can a classical student absorb the QCNN's function? | `rq4_teachers.py` → `rq4_distillation.py` → `rq4_reeval.py` | `rq4_analysis.py`, `rq4_reeval_analysis.py` | hours |

```bash
python experiments/rq3_barren_plateau.py      # the cheap one: no training at all
python experiments/rq3_analysis.py
```

**RQ3 is the one to run if you only run one.** It trains nothing — it samples 200 random
initialisations at 4, 6, 8 and 10 qubits and measures gradient variance — so it finishes in
minutes and reproduces a headline result of the dissertation end to end.

RQ1 and RQ2 checkpoint after every (configuration, seed) cell into
`results/<rq>/<dataset>_checkpoint.json` and skip completed cells on restart, so they can be
interrupted and resumed. RQ2 is the expensive one: eleven configurations across two datasets
at five seeds and fifty epochs.

---

## Results are included, and that is deliberate

`results/` holds the actual output of the runs above: raw JSON, the analysis JSON derived
from it, a readable `analysis_report.txt`, and the figures. It is ~46 MB and it is tracked.

The reason is that RQ2 costs about nine hours of CPU. Shipping only code would mean the
numbers in the dissertation could not be checked without spending that, which is not a
reasonable thing to ask of a reader. Every table in the results chapter is regenerated from
these files by the `*_analysis.py` scripts, so the chain from raw output to printed number
is inspectable without re-running anything.

`results/rq2/archive_pre_redesign_2026-08-18/` is kept on purpose. It holds superseded runs
from before a design fault was found — angle and IQP encodings were receiving the first six
of sixty-four pixels, which are near-constant background, rather than six PCA components.
The old numbers are wrong and are retained so the correction is visible rather than silent.

---

## Reproducibility

Seeds are fixed in `config.py` (`SEED = 42`, ten seeds for RQ1). The data subset is fixed
across seeds independently of the run seed, so a seed changes initialisation and shuffling
but never which examples are used.

Bit-exact reproduction is not promised across machines. The twenty RQ4 teacher models
reproduce RQ1's final-epoch accuracies exactly while their test losses differ by up to
2.05e-07; `experiments/rq4_cross_machine_stability.py` records that measurement.

The pinned environment is in `requirements.txt` — Python 3.11, PennyLane 0.44.1 with
`pennylane-lightning` 0.44.0, PyTorch 2.12.1. PennyLane 0.45 is deliberately *not* used: it
crashed natively on this workload. `scikit-learn` is deliberately absent; PCA is hand-rolled
in `utils/dataset.py` to keep the dependency list short.

---

## Licence

MIT, see `LICENSE`. The datasets are MNIST (LeCun et al.) and Fashion-MNIST (Xiao et al.),
distributed under their own terms and not redistributed here.
