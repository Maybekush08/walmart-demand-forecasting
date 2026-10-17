"""Evaluation script for trained forecasting models.

Usage:
    python scripts/evaluate.py --config configs/encoder_decoder.yaml --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.dataset import create_data_splits
from src.data.preprocessing import FeatureEngineer
from src.evaluation.metrics import evaluate_model
from src.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # Load data
    engineer = FeatureEngineer(config.data.data_dir)
    sales_matrix, features = engineer.build_all_features()
    n_features = features.shape[2] + 1

    _, _, test_ds = create_data_splits(
        sales_matrix, features,
        input_window=config.model.input_window,
        forecast_horizon=config.model.forecast_horizon,
    )

    test_loader = DataLoader(test_ds, batch_size=256)

    # Load model
    model = build_model(config.model, n_features)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run inference
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            x_sales = batch["x_sales"].to(device)
            x_features = batch["x_features"].to(device)
            y = batch["y"]

            preds = model(x_sales, x_features)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Compute metrics
    n_days = sales_matrix.shape[1]
    train_end = int(n_days * config.data.train_ratio)
    historical = sales_matrix[:, :train_end]

    results = evaluate_model(predictions, targets)
    print(f"\nModel: {config.model.name}")
    print(f"{'Metric':<10} {'Value':>10}")
    print("-" * 22)
    for metric, value in results.items():
        print(f"{metric:<10} {value:>10.2f}")


if __name__ == "__main__":
    main()
