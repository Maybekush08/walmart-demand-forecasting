"""Evaluation metrics for demand forecasting."""

from __future__ import annotations

import numpy as np


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error — primary comparison metric."""
    return float(np.abs(predictions - targets).mean())


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(((predictions - targets) ** 2).mean()))


def compute_mape(
    predictions: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error."""
    return float(
        np.abs((targets - predictions) / (targets + epsilon)).mean() * 100
    )


def compute_rmsse(
    predictions: np.ndarray,
    targets: np.ndarray,
    historical: np.ndarray,
) -> float:
    """Root Mean Squared Scaled Error (M5 competition metric).

    Normalizes forecast errors by the variance of the naive
    (random-walk) forecast on the training data.

    Args:
        predictions: ``(n_series, horizon)``.
        targets: ``(n_series, horizon)``.
        historical: ``(n_series, history_len)`` training data for scaling.
    """
    # Numerator: mean squared error per series
    numerator = ((predictions - targets) ** 2).mean(axis=1)

    # Denominator: MSE of 1-step naive forecast on history
    diffs = historical[:, 1:] - historical[:, :-1]
    denominator = (diffs**2).mean(axis=1)
    denominator = np.maximum(denominator, 1e-8)

    rmsse_per_series = np.sqrt(numerator / denominator)
    return float(rmsse_per_series.mean())


def compute_wrmsse(
    predictions: np.ndarray,
    targets: np.ndarray,
    historical: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted RMSSE — the official M5 competition metric.

    Args:
        weights: Per-series importance weights ``(n_series,)``.
            If None, uniform weights are used.
    """
    n_series = predictions.shape[0]

    if weights is None:
        weights = np.ones(n_series) / n_series

    numerator = ((predictions - targets) ** 2).mean(axis=1)
    diffs = historical[:, 1:] - historical[:, :-1]
    denominator = np.maximum((diffs**2).mean(axis=1), 1e-8)

    rmsse = np.sqrt(numerator / denominator)
    return float((weights * rmsse).sum())


def evaluate_model(
    predictions: np.ndarray,
    targets: np.ndarray,
    historical: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all metrics and return as a dictionary."""
    results = {
        "MAE": compute_mae(predictions, targets),
        "RMSE": compute_rmse(predictions, targets),
    }

    if historical is not None:
        results["RMSSE"] = compute_rmsse(predictions, targets, historical)
        results["WRMSSE"] = compute_wrmsse(predictions, targets, historical)

    return results
