"""
Distribution-shift corruptions for RQ4's robustness arm.

Why corruptions at all: Abnar, Dehghani & Zuidema (2020) — RQ4's methodological template
— measure inductive-bias transfer under *shift*, not clean in-distribution accuracy.
Clean accuracy cannot separate "the student absorbed the teacher's biases" from "both
models fit an easy task"; §4.1 already shows this task saturates.

Everything operates on flat 64-dim tensors in [-1, 1] (the format get_loaders and
get_tensors return): reshape to 8x8 -> corrupt -> reflatten. Background in this range is
-1.0, not 0.0 — the dataset pipeline applies Normalize((0.5,), (0.5,)) — so rotation and
occlusion fill with -1.0. Filling with 0.0 would paint mid-grey squares onto black
background and measure the fill colour rather than the corruption.

WHAT IS DELIBERATELY ABSENT (design decision D4): translation and scaling. At 8x8 a
one-pixel shift is 12.5% of image width, which is a different task rather than a
perturbation of this one. Raising IMAGE_SIZE is not an option — amplitude encoding needs
exactly 2^6 = 64 features, a load-bearing choice in config.py.

Corrupted sets are generated ONCE by build_corruption_suite(), saved to disk by
experiments/rq4_teachers.py, and reloaded by the student stage. Teacher and student must
be scored on bit-identical tensors or their agreement is measured on different data;
regenerating per stage would make that depend on RNG state matching across processes.
"""
import torch
import torchvision.transforms.functional as TF

BACKGROUND = -1.0          # [-1, 1] normalised black
CORRUPTION_SEED = 1234     # fixed: the suite must be reproducible across machines


def _to_images(X: torch.Tensor, image_size: int = 8) -> torch.Tensor:
    return X.view(-1, 1, image_size, image_size)


def _to_flat(imgs: torch.Tensor) -> torch.Tensor:
    return imgs.reshape(imgs.shape[0], -1)


def rotate(X: torch.Tensor, degrees: float, image_size: int = 8) -> torch.Tensor:
    """Rotate every image by a fixed angle, bilinear, background fill."""
    imgs = _to_images(X, image_size)
    out = TF.rotate(
        imgs, angle=float(degrees),
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=[BACKGROUND],
    )
    return _to_flat(out).clamp(-1.0, 1.0)


def gaussian_noise(X: torch.Tensor, sigma: float, seed: int = CORRUPTION_SEED) -> torch.Tensor:
    """Additive N(0, sigma^2) noise, clamped back into the valid [-1, 1] range."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(X.shape, generator=g) * sigma
    return (X + noise).clamp(-1.0, 1.0)


def occlude(X: torch.Tensor, patch: int, image_size: int = 8,
            seed: int = CORRUPTION_SEED) -> torch.Tensor:
    """
    Blank a patch x patch square to background, at a per-sample random position.

    Position is drawn from a seeded generator, so the suite is reproducible: the same
    call produces the same occlusions on any machine. A fixed centre position instead
    would be reproducible too, but at 8x8 it would delete the digit's centre of mass on
    every sample and make the corruption a near-constant transformation.
    """
    g = torch.Generator().manual_seed(seed)
    imgs = _to_images(X, image_size).clone()
    span = image_size - patch + 1
    rows = torch.randint(0, span, (imgs.shape[0],), generator=g)
    cols = torch.randint(0, span, (imgs.shape[0],), generator=g)
    for i in range(imgs.shape[0]):
        r, c = int(rows[i]), int(cols[i])
        imgs[i, :, r:r + patch, c:c + patch] = BACKGROUND
    return _to_flat(imgs)


def build_corruption_suite(X: torch.Tensor, image_size: int = 8,
                           seed: int = CORRUPTION_SEED) -> dict[str, torch.Tensor]:
    """
    The nine corrupted copies of a test set, keyed by name.

    Keys are stable strings — they become column names in the analysis and figure
    labels, so renaming one silently orphans results already collected under the old
    name.
    """
    return {
        "rot_-20":    rotate(X, -20, image_size),
        "rot_-10":    rotate(X, -10, image_size),
        "rot_+10":    rotate(X, +10, image_size),
        "rot_+20":    rotate(X, +20, image_size),
        "noise_0.10": gaussian_noise(X, 0.10, seed),
        "noise_0.25": gaussian_noise(X, 0.25, seed),
        "noise_0.50": gaussian_noise(X, 0.50, seed),
        "occl_2x2":   occlude(X, 2, image_size, seed),
        "occl_3x3":   occlude(X, 3, image_size, seed),
    }


CORRUPTION_NAMES = [
    "rot_-20", "rot_-10", "rot_+10", "rot_+20",
    "noise_0.10", "noise_0.25", "noise_0.50",
    "occl_2x2", "occl_3x3",
]
