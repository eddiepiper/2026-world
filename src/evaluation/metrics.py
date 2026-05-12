"""Core prediction quality metric functions.

All functions accept:
    y_true: pd.Series of string labels ('H', 'D', 'A')
    proba:  np.ndarray of shape (n, 3) with columns ordered [A, D, H]
            matching XGBoost's classes_ alphabetical sort order.

Lower is better for: log_loss, brier_score, rps, ece.
Higher is better for: accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Column ordering in proba arrays — must match XGBoost classes_ (alphabetical)
CLASSES: list[str] = ["A", "D", "H"]

# Map each class label to its column index
_CLASS_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


def _onehot(y_true: pd.Series) -> np.ndarray:
    """Convert string labels to one-hot matrix (n × 3) aligned with CLASSES."""
    n = len(y_true)
    ohe = np.zeros((n, len(CLASSES)), dtype=float)
    for i, label in enumerate(y_true):
        ohe[i, _CLASS_IDX[label]] = 1.0
    return ohe


def accuracy(y_true: pd.Series, proba: np.ndarray) -> float:
    """Fraction of matches where argmax prediction is correct."""
    pred_indices = np.argmax(proba, axis=1)
    pred_labels = [CLASSES[i] for i in pred_indices]
    correct = sum(p == t for p, t in zip(pred_labels, y_true))
    return correct / len(y_true)


def log_loss_score(y_true: pd.Series, proba: np.ndarray) -> float:
    """Multiclass log loss (lower is better). Uses sklearn.metrics.log_loss."""
    return float(log_loss(y_true, proba, labels=CLASSES))


def brier_score(y_true: pd.Series, proba: np.ndarray) -> float:
    """Multiclass Brier score: mean(sum((p - onehot)^2)) over all rows."""
    ohe = _onehot(y_true)
    return float(np.mean(np.sum((proba - ohe) ** 2, axis=1)))


def ranked_probability_score(y_true: pd.Series, proba: np.ndarray) -> float:
    """Ranked Probability Score (RPS) for ordered outcomes: A < D < H.

    RPS = mean over matches of sum over k of (CDF_pred_k - CDF_true_k)^2
    where k steps through the ordered outcome thresholds.

    Outcome ordering: A=0, D=1, H=2 (alphabetical, matches CLASSES).
    """
    ohe = _onehot(y_true)  # shape (n, 3)

    # Cumulative sums along outcome axis
    cdf_pred = np.cumsum(proba, axis=1)  # (n, 3)
    cdf_true = np.cumsum(ohe, axis=1)   # (n, 3)

    # RPS per match = sum of squared CDF differences (excluding last threshold,
    # which always equals 1 for both — but the standard formula sums all K-1
    # thresholds where K=3, so we sum over the first 2 columns).
    rps_per_match = np.sum((cdf_pred[:, :-1] - cdf_true[:, :-1]) ** 2, axis=1)
    return float(np.mean(rps_per_match))


def expected_calibration_error(
    y_true: pd.Series,
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE averaged across all 3 classes.

    For each class, bin predicted probabilities into n_bins equal-width buckets
    and compute the weighted mean absolute difference between mean predicted
    probability and fraction of true positives in each bin.
    """
    ohe = _onehot(y_true)  # (n, 3)
    n_samples = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_total = 0.0

    for col_idx in range(len(CLASSES)):
        pred_col = proba[:, col_idx]
        true_col = ohe[:, col_idx]

        class_ece = 0.0
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            # Last bin is inclusive on the right
            if b < n_bins - 1:
                mask = (pred_col >= lo) & (pred_col < hi)
            else:
                mask = (pred_col >= lo) & (pred_col <= hi)

            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue

            mean_pred = pred_col[mask].mean()
            mean_true = true_col[mask].mean()
            class_ece += (n_in_bin / n_samples) * abs(mean_pred - mean_true)

        ece_total += class_ece

    return float(ece_total / len(CLASSES))


def compute_all_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    """Return all 5 metrics in one dict.

    Keys: accuracy, log_loss, brier_score, rps, ece.
    """
    return {
        "accuracy": accuracy(y_true, proba),
        "log_loss": log_loss_score(y_true, proba),
        "brier_score": brier_score(y_true, proba),
        "rps": ranked_probability_score(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
    }
