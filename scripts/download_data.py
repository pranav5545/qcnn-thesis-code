"""Pre-fetch MNIST and Fashion-MNIST into data/.

Not strictly required -- utils/dataset.py passes download=True, so the first
experiment you run will fetch them itself. This exists so the download can be done
once, deliberately, on a machine that has network access, before running anything
else.

    python scripts/download_data.py

Downloads roughly 150 MB into data/, which is deliberately NOT included in this
repository: it is public data, byte-identical for everyone, and re-fetchable in a
minute. Nothing in the results depends on a local copy.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchvision import datasets  # noqa: E402

from config import DATA_DIR  # noqa: E402


def main():
    root = ROOT / DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    for name, cls in (("MNIST", datasets.MNIST),
                      ("Fashion-MNIST", datasets.FashionMNIST)):
        for split, train in (("train", True), ("test", False)):
            ds = cls(str(root), train=train, download=True)
            print("  %-14s %-5s %6d examples" % (name, split, len(ds)))
    print("\ndata is in %s" % root)


if __name__ == "__main__":
    main()
