"""Feature engineering pipeline for M5 demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Build temporal, lag, and rolling features from raw M5 data.

    The M5 dataset has three core files:
      - ``sales_train_evaluation.csv``: daily unit sales per product-store
      - ``calendar.csv``: date metadata (weekday, events, SNAP flags)
      - ``sell_prices.csv``: weekly sell prices per product-store

    This class merges them and produces a feature tensor suitable for
    the forecasting models.
    """

    LAG_DAYS = (7, 14, 28)
    ROLLING_WINDOWS = (7, 14, 28)

    def __init__(
        self,
        data_dir: str = "data/m5",
        store_id: str | None = "CA_1",
        history_days: int | None = 365,
    ) -> None:
        self.data_dir = data_dir
        self.store_id = store_id
        self.history_days = history_days
        if history_days is not None and history_days <= 0:
            raise ValueError("history_days must be positive or None")

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load M5 files and restrict them before feature construction."""
        sales = pd.read_csv(f"{self.data_dir}/sales_train_evaluation.csv")
        calendar = pd.read_csv(f"{self.data_dir}/calendar.csv")
        prices = pd.read_csv(f"{self.data_dir}/sell_prices.csv")

        if self.store_id is not None:
            sales = sales.loc[sales["store_id"] == self.store_id].copy()
            prices = prices.loc[prices["store_id"] == self.store_id].copy()
            if sales.empty:
                raise ValueError(f"store_id {self.store_id!r} was not found")

        day_cols = sorted(
            (column for column in sales.columns if column.startswith("d_")),
            key=lambda column: int(column.removeprefix("d_")),
        )
        if self.history_days is not None:
            if len(day_cols) < self.history_days:
                raise ValueError(
                    f"requested {self.history_days} history days, "
                    f"but only {len(day_cols)} are available"
                )
            day_cols = day_cols[-self.history_days :]

        metadata_cols = [
            column for column in sales.columns if not column.startswith("d_")
        ]
        sales = sales.loc[:, metadata_cols + day_cols].reset_index(drop=True)

        # Align calendar positions to the retained sales columns. This is
        # essential when keeping the latest N days rather than d_1 onward.
        day_order = {day: position for position, day in enumerate(day_cols)}
        calendar = calendar.loc[calendar["d"].isin(day_order)].copy()
        if len(calendar) != len(day_cols):
            raise ValueError("calendar does not contain every retained sales day")
        calendar["_day_order"] = calendar["d"].map(day_order)
        calendar = (
            calendar.sort_values("_day_order")
            .drop(columns="_day_order")
            .reset_index(drop=True)
        )

        # Retain only price weeks needed by the selected store and date range.
        retained_weeks = calendar["wm_yr_wk"].unique()
        prices = prices.loc[prices["wm_yr_wk"].isin(retained_weeks)].reset_index(
            drop=True
        )
        return sales, calendar, prices

    def extract_sales_matrix(self, sales_df: pd.DataFrame) -> np.ndarray:
        """Extract the pure sales matrix (n_series x n_days)."""
        day_cols = [c for c in sales_df.columns if c.startswith("d_")]
        return sales_df[day_cols].values.astype(np.float32)

    def build_calendar_features(self, calendar: pd.DataFrame) -> np.ndarray:
        """Extract day-level calendar features.

        Returns array of shape ``(n_days, n_calendar_features)`` with:
          - day_of_week (0-6)
          - month (1-12)
          - snap_CA, snap_TX, snap_WI (binary)
          - is_weekend (binary)
          - has_event (binary)
        """
        cal = calendar.copy()
        cal["is_weekend"] = cal["wday"].isin([1, 2]).astype(int)
        cal["has_event"] = (
            cal["event_name_1"].notna() | cal["event_name_2"].notna()
        ).astype(int)

        feature_cols = [
            "wday", "month", "snap_CA", "snap_TX", "snap_WI",
            "is_weekend", "has_event",
        ]
        return cal[feature_cols].values.astype(np.float32)

    def build_lag_features(
        self, sales_matrix: np.ndarray
    ) -> np.ndarray:
        """Create lag features for each time series.

        Returns array of shape ``(n_series, n_days, n_lags)``.
        """
        n_series, n_days = sales_matrix.shape
        lags = np.zeros((n_series, n_days, len(self.LAG_DAYS)), dtype=np.float32)

        for i, lag in enumerate(self.LAG_DAYS):
            if lag < n_days:
                lags[:, lag:, i] = sales_matrix[:, :-lag]

        return lags

    def build_rolling_features(
        self, sales_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute rolling statistics (mean, std) for each window size.

        Returns array of shape ``(n_series, n_days, n_windows * 2)``.
        """
        n_series, n_days = sales_matrix.shape
        n_feats = len(self.ROLLING_WINDOWS) * 2  # mean + std per window
        rolling = np.zeros((n_series, n_days, n_feats), dtype=np.float32)

        for w_idx, window in enumerate(self.ROLLING_WINDOWS):
            for t in range(window, n_days):
                segment = sales_matrix[:, t - window : t]
                rolling[:, t, w_idx * 2] = segment.mean(axis=1)
                rolling[:, t, w_idx * 2 + 1] = segment.std(axis=1)

        return rolling

    def build_price_features(
        self,
        sales_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        calendar: pd.DataFrame,
        n_days: int,
    ) -> np.ndarray:
        """Merge sell prices and compute price-based features.

        Returns array of shape ``(n_series, n_days, 2)`` with:
          - normalized_price (price / mean_price_per_item)
          - price_momentum (current_price - price_lag_7)
        """
        cal_wm = calendar[["d", "wm_yr_wk"]].copy()
        cal_wm["day_idx"] = np.arange(len(cal_wm))

        n_series = len(sales_df)
        price_features = np.zeros((n_series, n_days, 2), dtype=np.float32)

        for s_idx in range(n_series):
            item_id = sales_df.iloc[s_idx]["item_id"]
            store_id = sales_df.iloc[s_idx]["store_id"]

            item_prices = prices_df[
                (prices_df["item_id"] == item_id)
                & (prices_df["store_id"] == store_id)
            ].merge(cal_wm, on="wm_yr_wk", how="left")

            if len(item_prices) == 0:
                continue

            mean_price = item_prices["sell_price"].mean()
            if mean_price > 0:
                for _, row in item_prices.iterrows():
                    d = int(row["day_idx"])
                    if 0 <= d < n_days:
                        price_features[s_idx, d, 0] = row["sell_price"] / mean_price

        # Price momentum (lag-7 difference)
        if n_days > 7:
            price_features[:, 7:, 1] = (
                price_features[:, 7:, 0] - price_features[:, :-7, 0]
            )

        return price_features

    def build_all_features(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the full feature engineering pipeline.

        Returns:
            sales_matrix: Shape ``(n_series, n_days)``.
            features: Shape ``(n_series, n_days, total_features)``.
        """
        sales_df, calendar, prices = self.load_raw()
        sales_matrix = self.extract_sales_matrix(sales_df)
        n_series, n_days = sales_matrix.shape
        if len(calendar) != n_days:
            raise ValueError(
                "calendar and sales history are misaligned: "
                f"calendar_days={len(calendar)}, sales_days={n_days}"
            )

        # Calendar features → broadcast to (n_series, n_days, n_cal)
        cal_feats = self.build_calendar_features(calendar)[:n_days]
        cal_broadcast = np.tile(cal_feats, (n_series, 1, 1))

        # Lag features → (n_series, n_days, n_lags)
        lag_feats = self.build_lag_features(sales_matrix)

        # Rolling features → (n_series, n_days, n_rolling)
        roll_feats = self.build_rolling_features(sales_matrix)

        # Price features → (n_series, n_days, 2)
        price_feats = self.build_price_features(
            sales_df, prices, calendar, n_days
        )

        # Concatenate all features along feature axis
        features = np.concatenate(
            [cal_broadcast, lag_feats, roll_feats, price_feats], axis=2
        )

        return sales_matrix, features
