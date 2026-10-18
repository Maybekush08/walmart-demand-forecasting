"""Tests for scoped M5 loading and feature alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessing import FeatureEngineer


def _write_m5_fixture(tmp_path, n_days: int = 370) -> None:
    day_cols = [f"d_{day}" for day in range(1, n_days + 1)]
    sales_rows = []
    for store_id, offset in (("CA_1", 0), ("TX_1", 1000)):
        row = {
            "id": f"ITEM_1_{store_id}",
            "item_id": "ITEM_1",
            "dept_id": "DEPT_1",
            "cat_id": "CAT_1",
            "store_id": store_id,
            "state_id": store_id[:2],
        }
        row.update(
            {column: offset + day for day, column in enumerate(day_cols, start=1)}
        )
        sales_rows.append(row)

    calendar = pd.DataFrame(
        {
            "d": day_cols,
            "wm_yr_wk": [(day - 1) // 7 + 1 for day in range(1, n_days + 1)],
            "wday": [(day - 1) % 7 + 1 for day in range(1, n_days + 1)],
            "month": [1 + (day - 1) // 31 % 12 for day in range(1, n_days + 1)],
            "snap_CA": [0] * n_days,
            "snap_TX": [0] * n_days,
            "snap_WI": [0] * n_days,
            "event_name_1": [None] * n_days,
            "event_name_2": [None] * n_days,
        }
    )
    weeks = sorted(calendar["wm_yr_wk"].unique())
    prices = pd.DataFrame(
        [
            {
                "store_id": store_id,
                "item_id": "ITEM_1",
                "wm_yr_wk": week,
                "sell_price": float(week),
            }
            for store_id in ("CA_1", "TX_1")
            for week in weeks
        ]
    )

    pd.DataFrame(sales_rows).to_csv(
        tmp_path / "sales_train_evaluation.csv", index=False
    )
    calendar.to_csv(tmp_path / "calendar.csv", index=False)
    prices.to_csv(tmp_path / "sell_prices.csv", index=False)


def test_filters_store_and_latest_history_before_building_features(tmp_path):
    _write_m5_fixture(tmp_path)
    engineer = FeatureEngineer(
        str(tmp_path), store_id="CA_1", history_days=365
    )

    sales, calendar, prices = engineer.load_raw()
    assert sales["store_id"].unique().tolist() == ["CA_1"]
    assert calendar["d"].iloc[0] == "d_6"
    assert calendar["d"].iloc[-1] == "d_370"
    assert set(prices["store_id"]) == {"CA_1"}

    sales_matrix, features = engineer.build_all_features()
    assert sales_matrix.shape == (1, 365)
    assert features.shape == (1, 365, 18)
    assert sales_matrix[0, 0] == pytest.approx(6.0)
    assert features[0, 0, 0] == pytest.approx(calendar["wday"].iloc[0])


def test_rejects_unknown_store(tmp_path):
    _write_m5_fixture(tmp_path)
    engineer = FeatureEngineer(
        str(tmp_path), store_id="WI_9", history_days=365
    )
    with pytest.raises(ValueError, match="was not found"):
        engineer.load_raw()


def test_rejects_history_longer_than_available_data(tmp_path):
    _write_m5_fixture(tmp_path, n_days=100)
    engineer = FeatureEngineer(
        str(tmp_path), store_id="CA_1", history_days=365
    )
    with pytest.raises(ValueError, match="only 100 are available"):
        engineer.load_raw()
