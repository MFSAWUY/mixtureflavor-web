"""Loss utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_pos_weights(targets: torch.Tensor) -> torch.Tensor:
    """Compute per-class positive weights: sqrt(neg/pos)."""
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    return torch.sqrt((negatives + 1.0) / (positives + 1.0))


def smooth_positive_labels(targets: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
    """Apply smoothing to positive labels only."""
    if epsilon <= 0:
        return targets
    return torch.where(targets > 0.5, 1.0 - epsilon, targets)


def multilabel_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    smoothed = smooth_positive_labels(targets, epsilon=label_smoothing)
    if pos_weight is None:
        return F.binary_cross_entropy_with_logits(logits, smoothed)
    return F.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=pos_weight)

