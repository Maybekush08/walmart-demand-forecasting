"""M5 PyTorch Dataset for time series demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class M5Dataset(Dataset):
    """Sliding-window dataset for M5 sales forecasting.

    Generates (input_window, forecast_horizon) pairs from the full
    time series matrix.  Each sample consists of lagged sales values
    plus engineered features, and the target is the next
    ``forecast_horizon`` days of sales.

    Args:
        sales: Sales matrix of shape ``(n_series, n_days)``.
        features: Feature tensor of shape ``(n_series, n_days, n_features)``.
        input_window: Number of historical days used as input.
        forecast_horizon: Number of future days to predict.
    """

    def __init__(
        self,
        sales: np.ndarray,
        features: np.ndarray,
        input_window: int = 28,
        forecast_horizon: int = 28,
    ) -> None:
        self.sales = torch.tensor(sales, dtype=torch.float32)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.n_series, self.n_days = sales.shape

        if features.shape[:2] != sales.shape:
            raise ValueError(
                "features must match the first two sales dimensions; "
                f"got sales={sales.shape}, features={features.shape}"
            )
        if input_window <= 0 or forecast_horizon <= 0:
            raise ValueError("input_window and forecast_horizon must be positive")

        # Valid start indices for sliding window
        max_start = self.n_days - input_window - forecast_horizon + 1
        self.indices: list[tuple[int, int]] = []
        for series_idx in range(self.n_series):
            for t in range(max_start):
                self.indices.append((series_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        series_idx, t = self.indices[idx]
        end_input = t + self.input_window
        end_target = end_input + self.forecast_horizon

        x_sales = self.sales[series_idx, t:end_input]
        x_features = self.features[series_idx, t:end_input]
        y = self.sales[series_idx, end_input:end_target]

        return {
            "x_sales": x_sales,
            "x_features": x_features,
            "y": y,
        }


def create_data_splits(
    sales: np.ndarray,
    features: np.ndarray,
    input_window: int = 28,
    forecast_horizon: int = 28,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[M5Dataset, M5Dataset, M5Dataset]:
    """Split time series temporally into train / val / test datasets.

    The split is performed along the time axis to prevent data leakage.
    """
    n_days = sales.shape[1]
    train_end = int(n_days * train_ratio)
    val_end = int(n_days * (train_ratio + val_ratio))
    split_lengths = (train_end, val_end - train_end, n_days - val_end)
    minimums = (
        input_window + forecast_horizon,
        forecast_horizon,
        forecast_horizon,
    )
    if any(length < minimum for length, minimum in zip(split_lengths, minimums)):
        raise ValueError(
            "time splits are too short for the requested windows: "
            f"split_lengths={split_lengths}, input_window={input_window}, "
            f"forecast_horizon={forecast_horizon}"
        )

    context = input_window
    train_ds = M5Dataset(
        sales[:, :train_end],
        features[:, :train_end],
        input_window,
        forecast_horizon,
    )
    val_ds = M5Dataset(
        sales[:, train_end - context : val_end],
        features[:, train_end - context : val_end],
        input_window,
        forecast_horizon,
    )
    test_ds = M5Dataset(
        sales[:, val_end - context :],
        features[:, val_end - context :],
        input_window,
        forecast_horizon,
    )
    return train_ds, val_ds, test_ds
