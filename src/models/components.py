"""Shared transformer components for time series forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Injects position information into the embedding via fixed sinusoidal
    signals, allowing the transformer to reason about temporal order.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(
        self, d_model: int, max_len: int = 5000, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input of shape ``(batch, seq_len, d_model)``.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TimeSeriesEmbedding(nn.Module):
    """Project raw time series features into d_model dimensions.

    Combines a linear projection of continuous features with the
    sinusoidal positional encoding.

    Args:
        n_features: Number of input features (sales + engineered).
        d_model: Target embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self, n_features: int, d_model: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input features.

        Args:
            x: Input of shape ``(batch, seq_len, n_features)``.

        Returns:
            Embedded tensor of shape ``(batch, seq_len, d_model)``.
        """
        x = self.projection(x)
        x = self.pos_encoding(x)
        return self.norm(x)


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate a causal (upper-triangular) attention mask.

    Returns a boolean mask of shape ``(seq_len, seq_len)`` where
    ``True`` indicates positions that should be masked (future tokens).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()
