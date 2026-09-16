"""
Parameter-matched classical CNN baseline.

The network is intentionally tiny to match the QCNN parameter budget.
Input: flat vector of length input_dim (e.g. 64 for 8×8 images).
"""
import torch
import torch.nn as nn


class ClassicalCNN(nn.Module):
    """
    Small CNN that reshapes the flat input back to (1, H, W) and applies
    two conv layers followed by a linear head.

    n_params is exposed so callers can verify parity with the QCNN.
    """

    def __init__(
        self,
        input_dim: int = 64,
        image_size: int = 8,
        n_classes: int = 2,
        n_filters: int = 2,
        hidden: int = 8,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_filters  = n_filters
        self.hidden     = hidden

        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (n_filters, H/2, W/2)
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (n_filters, H/4, W/4)
        )
        flat = n_filters * (image_size // 4) ** 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)  →  (B, 1, H, W)
        b = x.shape[0]
        x = x.view(b, 1, self.image_size, self.image_size)
        return self.head(self.conv(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_matched_cnn(n_qcnn_params: int, image_size: int = 8, n_classes: int = 2) -> "ClassicalCNN":
    """
    Find the (n_filters, hidden) combination whose total param count is closest
    to n_qcnn_params. Minimises |cnn.n_params - n_qcnn_params| so the comparison
    is as fair as possible; ties are broken in favour of the smaller model.
    """
    best, best_delta = None, float("inf")
    for n_filters in range(1, 12):
        for hidden in range(2, 128):
            m = ClassicalCNN(image_size=image_size, n_classes=n_classes,
                             n_filters=n_filters, hidden=hidden)
            delta = abs(m.n_params - n_qcnn_params)
            if delta < best_delta:
                best, best_delta = m, delta
    if best is None:
        raise ValueError(f"Could not match {n_qcnn_params} params.")
    return best
