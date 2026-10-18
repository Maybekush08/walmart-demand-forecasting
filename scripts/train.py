"""Training script for demand forecasting models.

Usage:
    python scripts/train.py --config configs/encoder_decoder.yaml
    python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.dataset import create_data_splits
from src.data.preprocessing import FeatureEngineer
from src.models.factory import build_model
from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a forecasting model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Model: {config.model.name}")
    print(f"Device: {config.training.device}")

    # Feature engineering
    print("\nBuilding features...")
    engineer = FeatureEngineer(
        config.data.data_dir,
        store_id=config.data.store_id,
        history_days=config.data.history_days,
    )
    print(
        f"  Scope: store={config.data.store_id}, "
        f"latest_days={config.data.history_days}"
    )
    sales_matrix, features = engineer.build_all_features()
    print(f"  Sales matrix: {sales_matrix.shape}")
    print(f"  Features: {features.shape}")

    n_features = features.shape[2] + 1  # +1 for sales itself

    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_data_splits(
        sales_matrix, features,
        input_window=config.model.input_window,
        forecast_horizon=config.model.forecast_horizon,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    model = build_model(config.model, n_features)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Train
    trainer = Trainer(model, config.training)
    history = trainer.fit(train_loader, val_loader)

    print(f"\nFinal train MAE: {history['train_loss'][-1]:.2f}")
    print(f"Best val MAE: {trainer.best_val_loss:.2f}")


if __name__ == "__main__":
    main()
