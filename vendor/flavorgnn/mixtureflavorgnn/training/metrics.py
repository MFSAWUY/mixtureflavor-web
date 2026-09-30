"""Evaluation metrics for mixture flavor prediction."""

from __future__ import annotations

import numpy as np
import torch

from flavorgnn.training.metrics import compute_multilabel_metrics


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 10,
) -> float:
    flat_probs = probabilities.reshape(-1)
    flat_targets = targets.reshape(-1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(flat_probs)
    ece = 0.0
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right == 1.0:
            mask = (flat_probs >= left) & (flat_probs <= right)
        else:
            mask = (flat_probs >= left) & (flat_probs < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(flat_probs[mask]))
        accuracy = float(np.mean(flat_targets[mask]))
        ece += (float(np.sum(mask)) / max(total, 1)) * abs(confidence - accuracy)
    return ece


def compute_counterfactual_accuracy(
    before_probabilities: np.ndarray,
    after_probabilities: np.ndarray,
    flavor_indices: np.ndarray,
    expected_changes: np.ndarray,
) -> float:
    valid = flavor_indices >= 0
    if not np.any(valid):
        return 0.0
    deltas = after_probabilities[valid, flavor_indices[valid]] - before_probabilities[valid, flavor_indices[valid]]
    directions = expected_changes[valid]
    success = np.where(directions > 0, deltas > 0.0, deltas < 0.0)
    return float(np.mean(success.astype(np.float32)))


def compute_mixture_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    interaction_logits: torch.Tensor,
    interaction_targets: torch.Tensor,
    interaction_mask: torch.Tensor,
    max_single_component_probabilities: torch.Tensor,
    counterfactual_before_probabilities: torch.Tensor | None = None,
    counterfactual_after_probabilities: torch.Tensor | None = None,
    counterfactual_flavor_indices: torch.Tensor | None = None,
    counterfactual_expected_changes: torch.Tensor | None = None,
) -> dict[str, float]:
    metrics = compute_multilabel_metrics(logits=logits, targets=targets)

    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(np.int32)
    interaction_logits_np = interaction_logits.detach().cpu().numpy()
    interaction_targets_np = interaction_targets.detach().cpu().numpy()
    interaction_mask_np = interaction_mask.detach().cpu().numpy().astype(bool)
    max_single_np = max_single_component_probabilities.detach().cpu().numpy()

    synergy_mask = interaction_mask_np & (interaction_targets_np > 0)
    antagonism_mask = interaction_mask_np & (interaction_targets_np < 0)
    synergy_detection_rate = float(np.mean(interaction_logits_np[synergy_mask] > 0.0)) if np.any(synergy_mask) else 0.0
    antagonism_detection_rate = float(np.mean(interaction_logits_np[antagonism_mask] < 0.0)) if np.any(antagonism_mask) else 0.0

    interacting_delta_score = (
        float(np.mean(np.abs(interaction_logits_np[interaction_mask_np]))) if np.any(interaction_mask_np) else 0.0
    )
    non_interacting_mask = ~interaction_mask_np
    non_interacting_delta_score = (
        float(np.mean(np.abs(interaction_logits_np[non_interacting_mask]))) if np.any(non_interacting_mask) else 0.0
    )

    lift_base = np.clip(max_single_np, 1e-6, None)
    lift = probabilities / lift_base
    lift_mask = synergy_mask | (y_true > 0)
    mean_lift = float(np.mean(lift[lift_mask])) if np.any(lift_mask) else float(np.mean(lift))

    metrics.update(
        {
            "expected_calibration_error": expected_calibration_error(probabilities=probabilities, targets=y_true),
            "synergy_detection_rate": synergy_detection_rate,
            "antagonism_detection_rate": antagonism_detection_rate,
            "interaction_delta_score": float(np.mean(np.abs(interaction_logits_np))),
            "interacting_delta_score": interacting_delta_score,
            "non_interacting_delta_score": non_interacting_delta_score,
            "mean_lift_score": mean_lift,
        }
    )

    if (
        counterfactual_before_probabilities is not None
        and counterfactual_after_probabilities is not None
        and counterfactual_flavor_indices is not None
        and counterfactual_expected_changes is not None
    ):
        metrics["counterfactual_accuracy"] = compute_counterfactual_accuracy(
            before_probabilities=counterfactual_before_probabilities.detach().cpu().numpy(),
            after_probabilities=counterfactual_after_probabilities.detach().cpu().numpy(),
            flavor_indices=counterfactual_flavor_indices.detach().cpu().numpy(),
            expected_changes=counterfactual_expected_changes.detach().cpu().numpy(),
        )
    else:
        metrics["counterfactual_accuracy"] = 0.0

    return metrics
