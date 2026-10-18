"""Train and evaluate every model and produce one self-contained HTML report.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --report experiment_report.html --samples 3
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config
from src.data.dataset import create_data_splits
from src.data.preprocessing import FeatureEngineer
from src.evaluation.metrics import evaluate_model
from src.evaluation.report import write_experiment_report
from src.models.factory import build_model
from src.training.trainer import Trainer


DEFAULT_CONFIGS = [
    "configs/default.yaml",
    "configs/encoder_only.yaml",
    "configs/encoder_decoder.yaml",
    "configs/deepar.yaml",
    "configs/tft.yaml",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_shared_data(configs: list[Config]) -> None:
    first = configs[0]
    reference = (
        asdict(first.data),
        first.model.input_window,
        first.model.forecast_horizon,
    )
    for config in configs[1:]:
        candidate = (
            asdict(config.data),
            config.model.input_window,
            config.model.forecast_horizon,
        )
        if candidate != reference:
            raise ValueError(
                "all-model reporting requires identical data, window, and "
                "forecast settings across configurations"
            )


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    for batch in loader:
        x_sales = batch["x_sales"].to(device)
        x_features = batch["x_features"].to(device)
        prediction = model(x_sales, x_features)
        predictions.append(prediction.cpu().numpy())
        targets.append(batch["y"].numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all forecasting models and create one HTML report"
    )
    parser.add_argument(
        "--configs", nargs="+", default=DEFAULT_CONFIGS,
        help="Model YAML files to run in order",
    )
    parser.add_argument(
        "--report", default="experiment_report.html",
        help="Self-contained HTML report path",
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of held-out forecasts embedded in the report",
    )
    parser.add_argument(
        "--epochs", type=int,
        help="Override the epoch count in every selected configuration",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")

    _set_seed(args.seed)
    configs = [load_config(path) for path in args.configs]
    _validate_shared_data(configs)
    first = configs[0]

    print("Building the shared CA_1 feature dataset once...")
    engineer = FeatureEngineer(
        first.data.data_dir,
        store_id=first.data.store_id,
        history_days=first.data.history_days,
    )
    sales, features = engineer.build_all_features()
    train_ds, val_ds, test_ds = create_data_splits(
        sales,
        features,
        input_window=first.model.input_window,
        forecast_horizon=first.model.forecast_horizon,
        train_ratio=first.data.train_ratio,
        val_ratio=first.data.val_ratio,
    )
    n_features = features.shape[2] + 1
    sample_indices = np.linspace(
        0, len(test_ds) - 1, min(args.samples, len(test_ds)), dtype=int
    )

    metadata = {
        "run_status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "store_id": first.data.store_id,
        "history_days": first.data.history_days,
        "series": int(sales.shape[0]),
        "days": int(sales.shape[1]),
        "engineered_features": int(features.shape[2]),
        "training_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "test_samples": len(test_ds),
        "seed": args.seed,
        "epoch_override": args.epochs,
        "config_files": args.configs,
    }
    runs: dict[str, dict] = {
        config.model.name: {
            "status": "pending",
            "config": asdict(config),
        }
        for config in configs
    }
    report_path = write_experiment_report(args.report, runs, metadata)
    print(f"Report initialized: {report_path.resolve()}")

    for config in configs:
        name = config.model.name
        run = runs[name]
        run["status"] = "running"
        write_experiment_report(report_path, runs, metadata)
        print(f"\n{'=' * 72}\nTraining {name}\n{'=' * 72}")
        started = time.time()
        model = None
        trainer = None
        try:
            _set_seed(args.seed)
            checkpoint_dir = Path("checkpoints") / name
            configured_training = config.training
            if args.epochs is not None:
                configured_training = replace(
                    configured_training, epochs=args.epochs
                )
            training_config = replace(
                configured_training, checkpoint_dir=str(checkpoint_dir)
            )
            run["config"]["training"] = asdict(training_config)
            train_loader = DataLoader(
                train_ds,
                batch_size=training_config.batch_size,
                shuffle=True,
                num_workers=training_config.num_workers,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=training_config.batch_size,
                num_workers=training_config.num_workers,
            )
            test_loader = DataLoader(test_ds, batch_size=256)

            model = build_model(config.model, n_features)
            run["parameter_count"] = sum(
                parameter.numel() for parameter in model.parameters()
            )
            trainer = Trainer(model, training_config)
            history = trainer.fit(train_loader, val_loader)
            checkpoint = checkpoint_dir / "best_model.pt"
            model.load_state_dict(
                torch.load(checkpoint, map_location=trainer.device)
            )
            predictions, targets = _predict(model, test_loader, trainer.device)

            run.update(
                {
                    "status": "completed",
                    "metrics": evaluate_model(predictions, targets),
                    "best_val_mae": trainer.best_val_loss,
                    "history": history,
                    "sample_predictions": predictions[sample_indices].tolist(),
                    "sample_targets": targets[sample_indices].tolist(),
                }
            )
        except Exception:
            run["status"] = "failed"
            run["error"] = traceback.format_exc()
            print(run["error"], file=sys.stderr)
        finally:
            run["duration_seconds"] = time.time() - started
            write_experiment_report(report_path, runs, metadata)
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    completed = sum(run["status"] == "completed" for run in runs.values())
    metadata["run_status"] = "completed"
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    write_experiment_report(report_path, runs, metadata)
    print(
        f"\nFinished {completed}/{len(runs)} models. "
        f"Open {report_path.resolve()} in a web browser."
    )


if __name__ == "__main__":
    main()
