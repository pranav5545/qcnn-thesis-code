"""Record the cross-machine stability comparison as a results artefact.

Appendix G now states that the twenty RQ4 teacher models reproduce RQ1's final-epoch
accuracies exactly while their test losses differ by at most 2.05e-07. That number was
measured, not read from an existing results file, so it is written here to give it the same
provenance every other number in the thesis has.
"""
import json, os

from pathlib import Path

# Run from anywhere: resolve the repository root from this file.
R = str(Path(__file__).resolve().parent.parent / "results")
h = json.load(open(os.path.join(R, "rq4", "teachers_checkpoint.json")))["histories"]

rows, worst = [], 0.0
acc_mismatches = 0
for ds in ("mnist", "fashion_mnist"):
    rq1 = json.load(open(os.path.join(R, "rq1", ds + ".json")))
    for i, s in enumerate(rq1["seeds"]):
        if str(s) not in h[ds]:
            continue
        for model, key in (("qcnn", "qcnn_histories"), ("cnn", "cnn_histories")):
            a_acc = rq1[key][i]["test_acc"][-1]
            b_acc = h[ds][str(s)][model]["test_acc"][-1]
            a_loss = rq1[key][i]["test_loss"][-1]
            b_loss = h[ds][str(s)][model]["test_loss"][-1]
            d = abs(a_loss - b_loss)
            worst = max(worst, d)
            if a_acc != b_acc:
                acc_mismatches += 1
            rows.append({
                "dataset": ds, "seed": s, "model": model,
                "rq1_test_acc": a_acc, "rq4_test_acc": b_acc,
                "acc_identical": a_acc == b_acc,
                "rq1_test_loss": a_loss, "rq4_test_loss": b_loss,
                "abs_loss_diff": d,
            })

out = {
    "_meta": {
        "what": "RQ1 (original) vs RQ4 (retrained on a remote CPU session) final-epoch "
                "results for the same seeds, models and configuration.",
        "why": "Quantifies the local/remote hardware split disclosed in Appendix G, which "
               "was previously asserted to be harmless without measurement.",
        "generated": "2026-09-02",
    },
    "n_models": len(rows),
    "accuracy_mismatches": acc_mismatches,
    "max_abs_test_loss_diff": worst,
    "comparisons": rows,
}
p = os.path.join(R, "rq4", "cross_machine_stability.json")
json.dump(out, open(p, "w"), indent=2)
print("wrote %s" % p)
print("  models compared          : %d" % len(rows))
print("  accuracy mismatches      : %d" % acc_mismatches)
print("  max |test loss diff|     : %.6e" % worst)
