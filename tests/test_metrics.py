"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_mae,
    compute_mape,
    compute_rmse,
    compute_rmsse,
    evaluate_model,
)


class TestMAE:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        pred = np.array([1.0, 2.0, 3.0])
        actual = np.array([2.0, 3.0, 4.0])
        assert compute_mae(pred, actual) == pytest.approx(1.0)

    def test_symmetric(self):
        pred = np.array([1.0, 5.0])
        actual = np.array([3.0, 3.0])
        assert compute_mae(pred, actual) == compute_mae(actual, pred)


class TestRMSE:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        pred = np.array([1.0, 2.0, 3.0])
        actual = np.array([2.0, 3.0, 4.0])
        assert compute_rmse(pred, actual) == pytest.approx(1.0)


class TestRMSSE:
    def test_basic(self):
        pred = np.array([[5.0, 6.0]])
        actual = np.array([[4.0, 5.0]])
        historical = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = compute_rmsse(pred, actual, historical)
        assert result > 0

    def test_perfect_predictions(self):
        pred = np.array([[4.0, 5.0]])
        actual = np.array([[4.0, 5.0]])
        historical = np.array([[1.0, 2.0, 3.0, 4.0]])
        assert compute_rmsse(pred, actual, historical) == pytest.approx(0.0)


class TestEvaluateModel:
    def test_returns_all_metrics(self):
        pred = np.random.randn(10, 28)
        actual = np.random.randn(10, 28)
        historical = np.random.randn(10, 100)

        results = evaluate_model(pred, actual, historical)
        assert "MAE" in results
        assert "RMSE" in results
        assert "RMSSE" in results
