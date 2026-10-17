"""Model construction from project configuration."""

from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from . import MODEL_REGISTRY


def build_model(config: ModelConfig, n_features: int) -> nn.Module:
    """Build a registered forecasting model from a ``ModelConfig``."""
    if config.name not in MODEL_REGISTRY:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model {config.name!r}; choose from: {choices}")

    common = {
        "n_features": n_features,
        "input_window": config.input_window,
        "forecast_horizon": config.forecast_horizon,
    }
    if config.name == "linear":
        return MODEL_REGISTRY[config.name](**common)
    if config.name == "encoder_only":
        return MODEL_REGISTRY[config.name](
            **common,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
    if config.name == "encoder_decoder":
        return MODEL_REGISTRY[config.name](
            **common,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
    if config.name == "deepar":
        return MODEL_REGISTRY[config.name](
            **common,
            hidden_size=config.hidden_size,
            n_layers=config.n_lstm_layers,
            dropout=config.dropout,
        )
    return MODEL_REGISTRY[config.name](
        **common,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_lstm_layers=config.tft_lstm_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    )
