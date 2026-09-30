"""Evaluation metrics for multilabel flavor prediction."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)


def _best_macro_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.1, 0.9, 17)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.int32)
        score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def compute_multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = targets.detach().cpu().numpy().astype(np.int32)
    y_prob = torch.sigmoid(logits).detach().cpu().numpy()

    # AUROC per class can fail when a class has only one target value.
    per_class_auc = []
    for idx in range(y_true.shape[1]):
        unique_values = np.unique(y_true[:, idx])
        if unique_values.shape[0] < 2:
            continue
        try:
            per_class_auc.append(float(roc_auc_score(y_true[:, idx], y_prob[:, idx])))
        except ValueError:
            continue
    macro_auroc = float(np.mean(per_class_auc)) if per_class_auc else 0.0

    per_class_ap = []
    for idx in range(y_true.shape[1]):
        if int(y_true[:, idx].sum()) == 0:
            continue
        try:
            per_class_ap.append(float(average_precision_score(y_true[:, idx], y_prob[:, idx])))
        except ValueError:
            continue
    macro_ap = float(np.mean(per_class_ap)) if per_class_ap else 0.0

    best_threshold, best_macro_f1 = _best_macro_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(np.int32)
    exact_match = float(np.all(y_pred == y_true, axis=1).mean())
    hamming = float(hamming_loss(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "macro_auroc": macro_auroc,
        "macro_average_precision": macro_ap,
        "macro_f1": macro_f1,
        "best_macro_f1": best_macro_f1,
        "best_f1_threshold": best_threshold,
        "exact_match_ratio": exact_match,
        "hamming_loss": hamming,
    }
