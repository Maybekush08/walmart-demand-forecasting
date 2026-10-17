"""Encoder-Decoder Transformer for time series forecasting.

This is the best-performing model, achieving MAE 122.01 on the M5 dataset.
The encoder processes historical context while the decoder autoregressively
generates the forecast using cross-attention to attend to the encoded
history.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import PositionalEncoding, TimeSeriesEmbedding, generate_causal_mask


class EncoderDecoderTransformer(nn.Module):
    """Full Transformer with encoder-decoder architecture for forecasting.

    The encoder ingests the historical input window and builds contextual
    representations.  The decoder uses these representations via
    cross-attention to produce the forecast autoregressively during
    training (teacher forcing) and step-by-step during inference.

    Key design choices:
      - Pre-norm architecture (norm_first=True) for training stability
      - Residual connections in both encoder and decoder
      - Causal masking in decoder to prevent future information leakage
      - Separate embedding layers for encoder and decoder inputs

    Args:
        n_features: Number of input features per time step.
        d_model: Transformer embedding dimension.
        n_heads: Number of attention heads.
        n_encoder_layers: Number of encoder layers.
        n_decoder_layers: Number of decoder layers.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
        input_window: Length of the historical input sequence.
        forecast_horizon: Number of future steps to predict.
    """

    def __init__(
        self,
        n_features: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        input_window: int = 28,
        forecast_horizon: int = 28,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

        # Encoder embedding
        self.encoder_embedding = TimeSeriesEmbedding(
            n_features, d_model, dropout
        )

        # Decoder embedding (single feature: sales value)
        self.decoder_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            PositionalEncoding(d_model, dropout=dropout),
            nn.LayerNorm(d_model),
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_decoder_layers
        )

        # Output projection
        self.output_layer = nn.Linear(d_model, 1)

    def _encode(
        self,
        x_sales: torch.Tensor,
        x_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode the historical context.

        Args:
            x_sales: ``(batch, input_window)``.
            x_features: ``(batch, input_window, n_feat)`` or None.

        Returns:
            Encoder memory of shape ``(batch, input_window, d_model)``.
        """
        if x_features is not None:
            src = torch.cat([x_sales.unsqueeze(-1), x_features], dim=-1)
        else:
            src = x_sales.unsqueeze(-1)

        src = self.encoder_embedding(src)
        return self.encoder(src)

    def forward(
        self,
        x_sales: torch.Tensor,
        x_features: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional teacher forcing.

        During training, pass ``y`` (ground truth) for teacher forcing.
        During inference, omit ``y`` to use autoregressive generation.

        Args:
            x_sales: Historical sales ``(batch, input_window)``.
            x_features: Historical features ``(batch, input_window, n_feat)``.
            y: Target sales ``(batch, forecast_horizon)`` for teacher forcing.

        Returns:
            Predictions of shape ``(batch, forecast_horizon)``.
        """
        memory = self._encode(x_sales, x_features)

        if y is not None:
            # Teacher forcing: shift target right, prepend last known value
            decoder_input = torch.cat(
                [x_sales[:, -1:], y[:, :-1]], dim=1
            ).unsqueeze(-1)  # (B, H, 1)

            tgt = self.decoder_embedding(decoder_input)
            tgt_mask = generate_causal_mask(tgt.size(1), tgt.device)
            decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            output = self.output_layer(decoded).squeeze(-1)  # (B, H)
        else:
            # Autoregressive inference
            output = self._autoregressive_decode(memory, x_sales[:, -1])

        return output

    def _autoregressive_decode(
        self,
        memory: torch.Tensor,
        last_value: torch.Tensor,
    ) -> torch.Tensor:
        """Generate forecast step-by-step without teacher forcing.

        Args:
            memory: Encoder output ``(batch, input_window, d_model)``.
            last_value: Last known sales value ``(batch,)``.

        Returns:
            Predictions of shape ``(batch, forecast_horizon)``.
        """
        batch_size = memory.size(0)
        device = memory.device
        predictions = []

        current = last_value.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        for step in range(self.forecast_horizon):
            tgt = self.decoder_embedding(current)
            tgt_mask = generate_causal_mask(tgt.size(1), device)
            decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)

            pred = self.output_layer(decoded[:, -1:])  # (B, 1, 1)
            predictions.append(pred.squeeze(-1))

            current = torch.cat([current, pred], dim=1)

        return torch.cat(predictions, dim=1)  # (B, H)
