"""Visualization utilities for forecast analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Dark theme matching portfolio design
plt.rcParams.update({
    "figure.facecolor": "#09090b",
    "axes.facecolor": "#131316",
    "axes.edgecolor": "#1e1e24",
    "text.color": "#e4e4e7",
    "axes.labelcolor": "#e4e4e7",
    "xtick.color": "#9295a0",
    "ytick.color": "#9295a0",
    "grid.color": "#1e1e24",
    "legend.facecolor": "#131316",
    "legend.edgecolor": "#1e1e24",
})

ACCENT = "#818cf8"
COLORS = ["#818cf8", "#f472b6", "#34d399", "#fbbf24", "#60a5fa", "#a78bfa"]


def plot_forecast(
    actual: np.ndarray,
    predicted: dict[str, np.ndarray],
    series_id: str = "Sample Series",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot actual vs predicted values for a single time series.

    Args:
        actual: Ground truth values.
        predicted: Dict mapping model name to prediction arrays.
        series_id: Label for the title.
        save_path: Optional file path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    days = np.arange(len(actual))

    ax.plot(days, actual, color="#e4e4e7", linewidth=2, label="Actual", zorder=3)

    for i, (name, preds) in enumerate(predicted.items()):
        ax.plot(
            days, preds,
            color=COLORS[i % len(COLORS)],
            linewidth=1.5,
            alpha=0.8,
            label=name,
        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Sales")
    ax.set_title(f"Forecast Comparison — {series_id}", fontsize=14)
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "MAE",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing models on a given metric.

    Args:
        results: Dict mapping model name to metric dict.
        metric: Which metric to plot.
        save_path: Optional file path to save.
    """
    models = list(results.keys())
    values = [results[m][metric] for m in models]
    best_idx = int(np.argmin(values))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [ACCENT if i == best_idx else "#55576a" for i in range(len(models))]

    bars = ax.barh(models, values, color=colors, height=0.6)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=11, color="#e4e4e7",
        )

    ax.set_xlabel(metric)
    ax.set_title(f"Model Comparison — {metric}", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss curves for multiple models.

    Args:
        histories: Dict mapping model name to training history dict
            (keys: 'train_loss', 'val_loss').
        save_path: Optional file path to save.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, hist) in enumerate(histories.items()):
        color = COLORS[i % len(COLORS)]
        epochs = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], color=color, label=name)
        axes[1].plot(epochs, hist["val_loss"], color=color, label=name)

    axes[0].set_title("Training Loss (MAE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss (MAE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: list[str],
    top_k: int = 15,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of feature importance.

    Args:
        importance: Feature importance scores.
        feature_names: Corresponding feature names.
        top_k: Number of top features to display.
        save_path: Optional file path to save.
    """
    idx = np.argsort(importance)[-top_k:]
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(
        [feature_names[i] for i in idx],
        importance[idx],
        color=ACCENT,
        height=0.6,
    )
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Top 15)", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
