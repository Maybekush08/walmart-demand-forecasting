# Changelog

## Unreleased

- Added an all-model runner that consolidates metrics, training histories, and
  embedded charts into one self-contained `experiment_report.html` file.
- Isolated each model's checkpoints and set data-loader workers to zero for
  reliable laptop and Windows execution.
- Restricted default experiments to store CA_1 and its latest 365 days, with
  calendar and price filtering performed before feature construction.
- Added Linear, Encoder-Only Transformer, and Encoder-Decoder Transformer models.
- Added DeepAR-style probabilistic recurrent forecasting.
- Added a Temporal Fusion Transformer for continuous M5 features.
- Added chronological dataset splitting, feature engineering, metrics, and CLIs.
- Added configuration and tests for all five neural models.
- Excluded XGBoost and LightGBM models and dependencies by design.
