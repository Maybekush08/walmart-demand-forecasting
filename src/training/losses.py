"""Loss functions for demand forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mae_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error loss.

    Robust to outliers and directly interpretable as average unit error.
    Primary metric for model comparison on M5 dataset.
    """
    return F.l1_loss(predictions, targets)


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error loss.

    Penalizes large errors more heavily than MAE.
    """
    return F.mse_loss(predictions, targets)


def rmsse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    historical: torch.Tensor,
) -> torch.Tensor:
    """Root Mean Squared Scaled Error.

    The official M5 competition metric.  Normalizes squared errors by the
    historical naive forecast variance, making the metric comparable
    across series with different scales.

    Args:
        predictions: Forecast values ``(batch, horizon)``.
        targets: Actual values ``(batch, horizon)``.
        historical: Historical sales used for scaling ``(batch, history_len)``.
    """
    # Numerator: MSE of predictions
    numerator = ((predictions - targets) ** 2).mean(dim=1)

    # Denominator: variance of naive forecast (1-step differences)
    diffs = historical[:, 1:] - historical[:, :-1]
    denominator = (diffs**2).mean(dim=1).clamp(min=1e-8)

    return torch.sqrt(numerator / denominator).mean()
