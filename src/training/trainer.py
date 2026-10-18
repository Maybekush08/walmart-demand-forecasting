"""Training loop with metrics logging and checkpointing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config, TrainingConfig
from .losses import mae_loss


class Trainer:
    """Configurable training loop for forecasting models.

    Supports Adam optimizer with CosineAnnealing or ReduceLROnPlateau
    scheduling, best-model checkpointing, and per-epoch metric logging.

    Args:
        model: PyTorch model to train.
        config: Training configuration.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig) -> None:
        self.model = model
        self.config = config

        # Resolve device
        if config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)

        # Optimizer
        self.optimizer = Adam(
            model.parameters(), lr=config.learning_rate
        )

        # Scheduler
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=config.patience,
                factor=0.5,
            )

        self.best_val_loss = float("inf")
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x_sales = batch["x_sales"].to(self.device)
            x_features = batch["x_features"].to(self.device)
            y = batch["y"].to(self.device)

            self.optimizer.zero_grad()

            if hasattr(self.model, "training_loss"):
                loss = self.model.training_loss(x_sales, x_features, y)
                preds = None
            elif hasattr(self.model, "_encode"):
                preds = self.model(x_sales, x_features, y=y)
            else:
                preds = self.model(x_sales, x_features)
            if preds is not None:
                loss = mae_loss(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set and return average MAE."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x_sales = batch["x_sales"].to(self.device)
            x_features = batch["x_features"].to(self.device)
            y = batch["y"].to(self.device)

            preds = self.model(x_sales, x_features)
            loss = mae_loss(preds, y)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            Training history dict with 'train_loss', 'val_loss', and
            'learning_rate' per epoch.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            # Scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Checkpoint best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    checkpoint_dir / "best_model.pt",
                )

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"Train loss: {train_loss:.2f} | "
                f"Val MAE: {val_loss:.2f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed / 60:.1f}m"
            )

        # Save final model
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "final_model.pt",
        )

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time / 60:.1f} minutes")
        print(f"Best validation MAE: {self.best_val_loss:.2f}")

        return self.history
