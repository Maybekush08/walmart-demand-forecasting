# Walmart Demand Forecasting

An end-to-end PyTorch project for multi-horizon retail demand forecasting on
the M5 Walmart dataset. It provides a shared feature pipeline, temporal data
splits, training and evaluation CLIs, and five neural forecasting models.

## Models

| Config | Architecture | Training objective |
|---|---|---|
| `configs/default.yaml` | Linear baseline | MAE |
| `configs/encoder_only.yaml` | Causal encoder-only Transformer | MAE |
| `configs/encoder_decoder.yaml` | Autoregressive encoder-decoder Transformer | MAE |
| `configs/deepar.yaml` | DeepAR-style probabilistic LSTM | Gaussian negative log-likelihood |
| `configs/tft.yaml` | Temporal Fusion Transformer | MAE |

XGBoost and LightGBM are intentionally not included.

## Setup

Python 3.9 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
```

## Data

The project expects these M5 competition files in `data/m5/`:

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

To download them with the Kaggle CLI:

```bash
python -m pip install kaggle
python src/data/download.py
```

Kaggle credentials and acceptance of the M5 competition rules are required.
The dataset is deliberately excluded from Git.

By default, every model configuration filters the raw files to store `CA_1`
and retains its latest 365 sales days before constructing features. This keeps
laptop experiments tractable and ensures the full ten-store feature tensor is
never materialized. Change `data.store_id` or `data.history_days` in a YAML file
to select another scope; set either value to `null` to disable that filter.

## Train, evaluate, and predict

### Run every model and create one report

#### One-click Windows setup and background run

On Windows, double-click `START_ALL_MODELS.cmd` in the project folder. The
launcher performs the complete workflow automatically:

1. verifies the three M5 CSV files under `data/m5/`;
2. installs Python 3.12 with Windows Package Manager when Python is missing;
3. creates `.venv` and installs the project dependencies;
4. runs the automated test suite;
5. prevents duplicate training processes; and
6. starts all five models sequentially as a hidden background process.

The launcher can be closed after it reports the background process ID. Running
the launcher again while training is active opens the current report instead of
starting a duplicate process. Once a complete report exists, it opens that report
without retraining. Progress and diagnostic logs are written to
`experiment_run.log` and `experiment_error.log`.

#### Command-line execution

From the repository root, run:

```bash
python scripts/run_all.py
```

For a short first pass that verifies the complete workflow, use:

```bash
python scripts/run_all.py --epochs 2
```

This trains all five models in sequence and creates one self-contained file at
`experiment_report.html`. Open that file in any browser to review:

- validation and test metrics for every successful model;
- parameter counts and training durations;
- model-comparison charts;
- validation-loss and learning-rate curves;
- held-out forecast plots; and
- complete per-epoch histories and model configurations.

Charts are embedded directly in the HTML, so the report can be copied or shared
as one file. It is rewritten after each model finishes, preserving partial
results if a later model fails. Binary weights are stored separately under
`checkpoints/<model-name>/` so models can still be evaluated and reused.

On Windows PowerShell:

```powershell
python scripts\run_all.py
Start-Process .\experiment_report.html
```

Use `--report` to change the filename, `--samples` to change the number of
embedded forecast examples, `--epochs` to override every model's epoch count,
or `--configs` to run a selected subset:

```bash
python scripts/run_all.py --samples 5 --configs configs/default.yaml configs/deepar.yaml
```

Data-loader workers default to zero for reliable laptop and Windows execution.

### Run one model manually

Choose any model config:

```bash
python scripts/train.py --config configs/encoder_decoder.yaml
python scripts/evaluate.py \
  --config configs/encoder_decoder.yaml \
  --checkpoint checkpoints/best_model.pt
python scripts/predict.py \
  --config configs/encoder_decoder.yaml \
  --checkpoint checkpoints/best_model.pt
```

Replace the config path with `default.yaml`, `encoder_only.yaml`,
`deepar.yaml`, or `tft.yaml` to run another architecture. Checkpoints are
ignored by Git and written to `checkpoints/` by default.

## Feature pipeline

The feature engineer combines:

- calendar variables, SNAP indicators, weekends, and events;
- 7-, 14-, and 28-day sales lags;
- rolling means and standard deviations;
- normalized prices and 7-day price momentum.

Train, validation, and test windows are split chronologically. Validation and
test sets retain only the historical context needed for their first forecast,
preventing target leakage across split boundaries.

## Project layout

```text
configs/              model and training configurations
scripts/              training, all-model reporting, evaluation, and prediction
src/data/             M5 loading, feature engineering, temporal datasets
src/models/           five model implementations and model factory
src/training/         losses, optimization, checkpointing
src/evaluation/       metrics and visualizations
tests/                model, dataset, and metric tests
```

## Attribution

This project is adapted from
[Time-Series Demand Forecasting for Smart Retail](https://github.com/allureking/Time-Series-Demand-Forecasting-for-Smart-Retail)
by Honghuai Ke. The original MIT license and copyright notice are retained in
[`LICENSE`](LICENSE). DeepAR and Temporal Fusion Transformer implementations,
integration, and associated tests were added here because the reference
repository described those models but did not contain their source code.
