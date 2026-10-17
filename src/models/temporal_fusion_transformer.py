"""Compact Temporal Fusion Transformer for multi-horizon forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    """Nonlinear transformation with gating, residuals, and layer norm."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.skip = (
            nn.Identity()
            if input_size == hidden_size
            else nn.Linear(input_size, hidden_size)
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(torch.nn.functional.elu(self.fc1(x)))
        hidden = self.dropout(hidden)
        gated = torch.sigmoid(self.gate(hidden)) * hidden
        return self.norm(self.skip(x) + gated)


class TemporalFusionTransformer(nn.Module):
    """LSTM local processing plus interpretable temporal self-attention.

    This implementation follows TFT's core variable selection, gated residual,
    recurrent, and attention stages while targeting the continuous M5 inputs.
    """

    def __init__(
        self,
        n_features: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_lstm_layers: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
        input_window: int = 28,
        forecast_horizon: int = 28,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_features = n_features
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

        self.feature_embeddings = nn.ModuleList(
            nn.Linear(1, d_model) for _ in range(n_features)
        )
        self.selection = nn.Linear(n_features, n_features)
        self.variable_grn = GatedResidualNetwork(d_model, d_model, dropout)
        self.lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.post_lstm_gate = GatedResidualNetwork(d_model, d_model, dropout)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.post_attention_gate = GatedResidualNetwork(d_model, d_model, dropout)
        self.positionwise = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.output = nn.Linear(d_model * input_window, forecast_horizon)

    def forward(
        self,
        x_sales: torch.Tensor,
        x_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_features is None:
            values = x_sales.unsqueeze(-1)
        else:
            values = torch.cat([x_sales.unsqueeze(-1), x_features], dim=-1)
        if values.size(-1) != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, got {values.size(-1)}"
            )

        weights = torch.softmax(self.selection(values), dim=-1)
        embedded = torch.stack(
            [layer(values[..., i : i + 1]) for i, layer in enumerate(self.feature_embeddings)],
            dim=-2,
        )
        selected = (embedded * weights.unsqueeze(-1)).sum(dim=-2)
        selected = self.variable_grn(selected)

        local, _ = self.lstm(selected)
        local = self.post_lstm_gate(local + selected)
        attended, _ = self.attention(local, local, local, need_weights=False)
        fused = self.post_attention_gate(local + attended)
        fused = fused + self.positionwise(fused)
        return self.output(fused.flatten(start_dim=1))
