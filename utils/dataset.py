"""
Dataset loading and preprocessing.

Returns flattened, normalised tensors ready for both classical and quantum models.
Supports binary and 4-class subsets of MNIST / Fashion-MNIST.
"""
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMAGE_SIZE, BATCH_SIZE, DATA_DIR, SEED


def _class_filter(dataset, classes: list[int]):
    """Return indices whose label is in `classes`."""
    targets = dataset.targets.clone().detach() if isinstance(dataset.targets, torch.Tensor) else torch.tensor(dataset.targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for c in classes:
        mask |= (targets == c)
    return mask.nonzero(as_tuple=True)[0]


def _prepare_datasets(name: str, classes: list[int], image_size: int):
    """Shared load + filter + relabel step used by every loader variant below."""
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),     # → [-1, 1]
        transforms.Lambda(lambda x: x.view(-1)),  # flatten
    ])

    root = Path(DATA_DIR)
    cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST

    train_ds = cls(root, train=True,  download=True, transform=tf)
    test_ds  = cls(root, train=False, download=True, transform=tf)

    label_map = {c: i for i, c in enumerate(classes)}

    train_idx = _class_filter(train_ds, classes)
    test_idx  = _class_filter(test_ds,  classes)

    # Remap only after filtering so unknown labels are never touched
    for ds in (train_ds, test_ds):
        new_targets = ds.targets.clone()
        for c, new_label in label_map.items():
            new_targets[ds.targets == c] = new_label
        ds.targets = new_targets

    return train_ds, test_ds, train_idx, test_idx


def _subsample_idx(idx: torch.Tensor, n: int | None, seed: int) -> torch.Tensor:
    if n is None:
        return idx
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(idx), generator=g)
    return idx[perm[:n]]


def get_loaders(
    name: str = "mnist",
    classes: list[int] = [0, 1],
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    n_train: int | None = None,
    n_test:  int | None = None,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader).
    Images are resized to image_size×image_size, flattened, and normalised to [-1, 1].
    Labels are remapped to 0…len(classes)-1.
    """
    train_ds, test_ds, train_idx, test_idx = _prepare_datasets(name, classes, image_size)

    train_idx = _subsample_idx(train_idx, n_train, seed)
    test_idx  = _subsample_idx(test_idx,  n_test,  seed)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(Subset(test_ds,  test_idx),  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def _pca_fit_transform(X_train: np.ndarray, X_test: np.ndarray, n_components: int):
    """
    Hand-rolled PCA (no sklearn dependency — not installed in the `thesis` env and
    kept that way deliberately, see RQ2 design decision Q4). Fit on train only,
    then min-max scale each retained component to [-1, 1] using the train range so
    downstream angle/IQP encoding (which does `x * pi`) sees the same input range
    as the naive-truncation path did.
    """
    mean = X_train.mean(axis=0, keepdims=True)
    Xc_train = X_train - mean
    Xc_test  = X_test  - mean

    # Economy SVD on the centred training data; top `n_components` right-singular
    # vectors are the principal axes (numpy convention: X = U @ diag(S) @ Vt).
    _, _, Vt = np.linalg.svd(Xc_train, full_matrices=False)
    components = Vt[:n_components]  # (n_components, n_features)

    train_proj = Xc_train @ components.T
    test_proj  = Xc_test  @ components.T

    lo = train_proj.min(axis=0, keepdims=True)
    hi = train_proj.max(axis=0, keepdims=True)
    span = np.clip(hi - lo, 1e-8, None)

    train_scaled = 2 * (train_proj - lo) / span - 1
    test_scaled  = 2 * (test_proj  - lo) / span - 1
    test_scaled  = np.clip(test_scaled, -1.0, 1.0)  # test may fall slightly outside train range

    return train_scaled.astype(np.float32), test_scaled.astype(np.float32)


def get_pca_loaders(
    name: str = "mnist",
    classes: list[int] = [0, 1],
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    n_train: int | None = None,
    n_test:  int | None = None,
    n_components: int = 6,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader]:
    """
    Same sample selection as get_loaders (same seed => identical train/test subset),
    but returns PCA(n_components)-reduced features scaled to [-1, 1] instead of the
    raw flattened image. Used for angle/IQP encoding under RQ2's Q4 design decision
    — see writing/thesis/rq2_design_decisions.md — replacing the naive
    first-n_components-of-64-pixels truncation.

    PCA is fit once on the selected training subset (never on test) and reused —
    fitting per-seed would be data leakage across the seed axis for no benefit,
    since the sample selection itself doesn't vary by seed here (see docstring
    note in rq2_design_choices.py on the `seed` param).
    """
    train_ds, test_ds, train_idx, test_idx = _prepare_datasets(name, classes, image_size)

    train_idx = _subsample_idx(train_idx, n_train, seed)
    test_idx  = _subsample_idx(test_idx,  n_test,  seed)

    X_train = torch.stack([train_ds[i][0] for i in train_idx]).numpy()
    y_train = torch.tensor([train_ds.targets[i].item() for i in train_idx])
    X_test  = torch.stack([test_ds[i][0]  for i in test_idx]).numpy()
    y_test  = torch.tensor([test_ds.targets[i].item()  for i in test_idx])

    X_train_pca, X_test_pca = _pca_fit_transform(X_train, X_test, n_components)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_pca), y_train),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test_pca), y_test),
        batch_size=batch_size, shuffle=False,
    )
    return train_loader, test_loader


def get_tensors(
    name: str = "mnist",
    classes: list[int] = [0, 1],
    image_size: int = IMAGE_SIZE,
    n_train: int | None = None,
    n_test:  int | None = None,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same selection as get_loaders, returned as fixed-order tensors instead of loaders.

    Added for RQ4 (2026-08-24). Knowledge distillation needs the teacher's logits to
    travel with their samples through a shuffling loader, which means the samples must
    first exist in a known, stable order — a DataLoader cannot give that. Returns
    (X_train, y_train, X_test, y_test) stacked in exactly the order get_loaders' Subset
    would iterate them, so a TensorDataset built from these is sample-for-sample
    identical to RQ1's loaders (this is what makes RQ4's `hard` arm a valid RQ1
    regression check).

    Note the prefix property that RQ4's sample-efficiency arm relies on: _subsample_idx
    draws one permutation from a generator seeded with `seed` and takes perm[:n], so the
    n_train=100 subset is exactly the first 100 rows of the n_train=500 subset. Cached
    teacher logits can therefore be sliced rather than recomputed.
    """
    train_ds, test_ds, train_idx, test_idx = _prepare_datasets(name, classes, image_size)

    train_idx = _subsample_idx(train_idx, n_train, seed)
    test_idx  = _subsample_idx(test_idx,  n_test,  seed)

    X_train = torch.stack([train_ds[i][0] for i in train_idx])
    y_train = torch.tensor([train_ds.targets[i].item() for i in train_idx])
    X_test  = torch.stack([test_ds[i][0]  for i in test_idx])
    y_test  = torch.tensor([test_ds.targets[i].item()  for i in test_idx])

    return X_train, y_train, X_test, y_test


def describe(loader: DataLoader) -> str:
    x, y = next(iter(loader))
    n = len(loader.dataset)
    return f"n={n}, input_dim={x.shape[1]}, classes={sorted(y.unique().tolist())}"
