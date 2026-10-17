"""Unit tests for the M5 dataset loader."""

import numpy as np
import pytest

from src.data.dataset import M5Dataset, create_data_splits


@pytest.fixture
def synthetic_data():
    """Generate synthetic sales and features for testing."""
    np.random.seed(42)
    n_series = 10
    n_days = 600
    n_features = 5

    sales = np.random.poisson(lam=5, size=(n_series, n_days)).astype(np.float32)
    features = np.random.randn(n_series, n_days, n_features).astype(np.float32)
    return sales, features


class TestM5Dataset:
    def test_rejects_misaligned_features(self, synthetic_data):
        sales, features = synthetic_data
        with pytest.raises(ValueError, match="features must match"):
            M5Dataset(sales, features[:, :-1])
    def test_dataset_length(self, synthetic_data):
        sales, features = synthetic_data
        ds = M5Dataset(sales, features, input_window=28, forecast_horizon=28)
        expected = sales.shape[0] * (sales.shape[1] - 28 - 28 + 1)
        assert len(ds) == expected

    def test_item_shapes(self, synthetic_data):
        sales, features = synthetic_data
        ds = M5Dataset(sales, features, input_window=28, forecast_horizon=28)
        item = ds[0]

        assert item["x_sales"].shape == (28,)
        assert item["x_features"].shape == (28, 5)
        assert item["y"].shape == (28,)

    def test_no_data_leakage(self, synthetic_data):
        """Verify that target values come after input values in time."""
        sales, features = synthetic_data
        ds = M5Dataset(sales, features, input_window=28, forecast_horizon=28)

        for i in range(min(50, len(ds))):
            series_idx, t = ds.indices[i]
            item = ds[i]
            end_input = t + 28
            end_target = end_input + 28

            np.testing.assert_array_equal(
                item["x_sales"].numpy(), sales[series_idx, t:end_input]
            )
            np.testing.assert_array_equal(
                item["y"].numpy(), sales[series_idx, end_input:end_target]
            )


class TestDataSplits:
    def test_split_sizes(self, synthetic_data):
        sales, features = synthetic_data
        train, val, test = create_data_splits(
            sales, features,
            input_window=28, forecast_horizon=28,
            train_ratio=0.8, val_ratio=0.1,
        )
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        train_end = int(sales.shape[1] * 0.8)
        val_end = int(sales.shape[1] * 0.9)
        assert train.sales.shape[1] == train_end
        assert val.sales.shape[1] == val_end - train_end + 28
        assert test.sales.shape[1] == sales.shape[1] - val_end + 28

    def test_rejects_splits_without_a_full_horizon(self):
        sales = np.zeros((2, 100), dtype=np.float32)
        features = np.zeros((2, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="time splits are too short"):
            create_data_splits(sales, features, input_window=28, forecast_horizon=28)
