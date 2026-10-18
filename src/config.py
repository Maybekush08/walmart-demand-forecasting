"""Frozen dataclass configuration system with YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Data loading and feature engineering settings."""

    data_dir: str = "data/m5"
    store_id: str | None = "CA_1"
    history_days: int | None = 365
    target_col: str = "sales"
    features: tuple[str, ...] = (
        "sales", "day_of_week", "month", "snap", "sell_price",
    )
    lag_features: tuple[int, ...] = (7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 28)
    train_ratio: float = 0.8
    val_ratio: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture settings."""

    name: str = "linear"
    input_window: int = 28
    forecast_horizon: int = 28
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    hidden_size: int = 128
    n_lstm_layers: int = 2
    tft_lstm_layers: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop settings."""

    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    patience: int = 5
    num_workers: int = 4
    device: str = "auto"
    warmup_steps: int = 0
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100


@dataclass(frozen=True)
class Config:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _to_tuple(val: Any) -> Any:
    """Convert lists to tuples for frozen dataclasses."""
    if isinstance(val, list):
        return tuple(val)
    return val


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    data_cfg = DataConfig(
        **{k: _to_tuple(v) for k, v in raw.get("data", {}).items()}
    )
    model_cfg = ModelConfig(
        **{k: _to_tuple(v) for k, v in raw.get("model", {}).items()}
    )
    training_cfg = TrainingConfig(
        **{k: _to_tuple(v) for k, v in raw.get("training", {}).items()}
    )
    return Config(data=data_cfg, model=model_cfg, training=training_cfg)
