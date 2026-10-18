"""Generate forecasts and save visualizations.

Usage:
    python scripts/predict.py --config configs/encoder_decoder.yaml --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.preprocessing import FeatureEngineer
from src.evaluation.visualization import plot_forecast
from src.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate forecasts")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for output visualizations",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5,
        help="Number of sample series to visualize",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)

    # Load data
    engineer = FeatureEngineer(
        config.data.data_dir,
        store_id=config.data.store_id,
        history_days=config.data.history_days,
    )
    sales_matrix, features = engineer.build_all_features()
    n_features = features.shape[2] + 1

    # Load model
    model = build_model(config.model, n_features)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # Generate sample forecasts
    np.random.seed(42)
    sample_indices = np.random.choice(
        sales_matrix.shape[0], size=args.n_samples, replace=False
    )

    for idx in sample_indices:
        series = sales_matrix[idx]
        feat = features[idx]

        # Get the last input_window for prediction
        t = len(series) - config.model.forecast_horizon
        x_sales = torch.tensor(
            series[t - config.model.input_window : t], dtype=torch.float32
        ).unsqueeze(0)
        x_feat = torch.tensor(
            feat[t - config.model.input_window : t], dtype=torch.float32
        ).unsqueeze(0)
        actual = series[t : t + config.model.forecast_horizon]

        with torch.no_grad():
            pred = model(x_sales, x_feat).squeeze(0).numpy()

        plot_forecast(
            actual, {config.model.name: pred},
            series_id=f"Series {idx}",
            save_path=output_dir / f"forecast_series_{idx}.png",
        )

    print(f"Saved {args.n_samples} forecast plots to {output_dir}")


if __name__ == "__main__":
    main()
