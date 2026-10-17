"""DeepAR-style probabilistic recurrent demand forecaster."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepAR(nn.Module):
    """Autoregressive LSTM that models demand with a Gaussian likelihood.

    The model consumes historical sales and covariates. During training it
    uses teacher forcing; inference recursively feeds back the predicted mean.
    ``distribution`` exposes both the mean and strictly-positive scale.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        input_window: int = 28,
        forecast_horizon: int = 28,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.encoder = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            1,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.parameters_head = nn.Linear(hidden_size, 2)

    def _history(
        self, x_sales: torch.Tensor, x_features: torch.Tensor | None
    ) -> torch.Tensor:
        if x_features is None:
            if self.n_features != 1:
                raise ValueError("x_features are required for this model")
            return x_sales.unsqueeze(-1)
        return torch.cat([x_sales.unsqueeze(-1), x_features], dim=-1)

    def distribution(
        self,
        x_sales: torch.Tensor,
        x_features: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, state = self.encoder(self._history(x_sales, x_features))
        current = x_sales[:, -1:].unsqueeze(-1)
        means: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []

        for step in range(self.forecast_horizon):
            decoded, state = self.decoder(current, state)
            params = self.parameters_head(decoded[:, -1])
            mean = params[:, 0]
            scale = F.softplus(params[:, 1]) + 1e-4
            means.append(mean)
            scales.append(scale)
            next_value = y[:, step] if y is not None else mean
            current = next_value[:, None, None]

        return torch.stack(means, dim=1), torch.stack(scales, dim=1)

    def forward(
        self,
        x_sales: torch.Tensor,
        x_features: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean, _ = self.distribution(x_sales, x_features, y)
        return mean

    def training_loss(
        self,
        x_sales: torch.Tensor,
        x_features: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        mean, scale = self.distribution(x_sales, x_features, y)
        return -torch.distributions.Normal(mean, scale).log_prob(y).mean()
