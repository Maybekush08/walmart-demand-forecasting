"""Download M5 dataset from Kaggle.

Requires the Kaggle CLI to be installed and authenticated:
    pip install kaggle
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


COMPETITION = "m5-forecasting-accuracy"
REQUIRED_FILES = [
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv",
]


def download_m5(data_dir: str = "data/m5") -> None:
    """Download M5 competition data to ``data_dir``."""
    dest = Path(data_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if all((dest / f).exists() for f in REQUIRED_FILES):
        print(f"M5 data already exists in {dest}")
        return

    print(f"Downloading M5 data to {dest} ...")
    try:
        subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", COMPETITION,
                "-p", str(dest),
            ],
            check=True,
        )
    except FileNotFoundError:
        print(
            "Error: 'kaggle' CLI not found. Install with: pip install kaggle",
            file=sys.stderr,
        )
        sys.exit(1)

    # Unzip
    zip_path = dest / f"{COMPETITION}.zip"
    if zip_path.exists():
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        zip_path.unlink()
        print("Download and extraction complete.")


if __name__ == "__main__":
    download_m5()
