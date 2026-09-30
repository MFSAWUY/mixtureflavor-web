"""Loss utilities for mixture training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flavorgnn.training.losses import compute_pos_weights, smooth_positive_labels


def _masked_component_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(mask):
        return logits.new_tensor(0.0)
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets)


def _delta_supervision_loss(
    interaction_logits: torch.Tensor,
    interaction_targets: torch.Tensor,
    interaction_mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(interaction_mask):
        return interaction_logits.new_tensor(0.0)
    predicted = torch.tanh(interaction_logits[interaction_mask])
    target = torch.tanh(interaction_targets[interaction_mask])
    return F.smooth_l1_loss(predicted, target)


def compute_mixture_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.05,
    baseline_loss_weight: float = 0.5,
    sparse_loss_weight: float = 0.01,
    delta_supervision_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = smooth_positive_labels(batch["y"], epsilon=label_smoothing)
    logits = outputs["logits"]
    if pos_weight is None:
        mixture_loss = F.binary_cross_entropy_with_logits(logits, targets)
    else:
        mixture_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    baseline_loss = _masked_component_loss(
        logits=outputs["component_baseline_logits"],
        targets=batch["single_mol_labels"],
        mask=batch["molecule_mask"],
    )
    sparse_loss = outputs["interaction_logit"].abs().mean()
    delta_loss = _delta_supervision_loss(
        interaction_logits=outputs["interaction_logit"],
        interaction_targets=batch["interaction_targets"],
        interaction_mask=batch["interaction_mask"],
    )
    total_loss = (
        mixture_loss
        + baseline_loss_weight * baseline_loss
        + sparse_loss_weight * sparse_loss
        + delta_supervision_weight * delta_loss
    )
    return total_loss, {
        "loss": total_loss.detach(),
        "mixture_loss": mixture_loss.detach(),
        "baseline_loss": baseline_loss.detach(),
        "sparse_loss": sparse_loss.detach(),
        "delta_supervision_loss": delta_loss.detach(),
    }
