"""Tests for the self-contained experiment report."""

from __future__ import annotations

from src.evaluation.report import write_experiment_report


def test_report_embeds_results_and_charts(tmp_path):
    output = tmp_path / "experiment_report.html"
    runs = {
        "linear": {
            "status": "completed",
            "metrics": {"MAE": 1.25, "RMSE": 1.75},
            "best_val_mae": 1.1,
            "duration_seconds": 30.0,
            "parameter_count": 100,
            "history": {
                "train_loss": [2.0, 1.0],
                "val_loss": [1.5, 1.1],
                "learning_rate": [0.001, 0.0005],
            },
            "sample_predictions": [[1.0, 2.0, 3.0]],
            "sample_targets": [[1.0, 2.5, 2.5]],
            "config": {"model": {"name": "linear"}},
        },
        "deepar": {
            "status": "failed",
            "error": "example failure",
            "config": {"model": {"name": "deepar"}},
        },
    }

    result = write_experiment_report(
        output, runs, {"store_id": "CA_1", "history_days": 365}
    )
    document = result.read_text(encoding="utf-8")

    assert result == output
    assert "Walmart Demand Forecasting" in document
    assert "linear" in document
    assert "1.2500" in document
    assert "example failure" in document
    assert "data:image/png;base64," in document
    assert not output.with_suffix(".html.tmp").exists()
