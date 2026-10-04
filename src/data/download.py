"""Download Lending Club loan data from Kaggle."""

import shutil
from pathlib import Path

import kagglehub

from config.settings import RAW_DIR


def download_lending_club_data() -> Path:
    """Download the Lending Club dataset using kagglehub.

    Returns the path to the downloaded CSV file in data/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Lending Club dataset from Kaggle...")
    downloaded_path = kagglehub.dataset_download("wordsforthewise/lending-club")
    downloaded_path = Path(downloaded_path)
    print(f"Downloaded to: {downloaded_path}")

    csv_files = [f for f in downloaded_path.rglob("*.csv.gz") if f.is_file()]
    if not csv_files:
        csv_files = [f for f in downloaded_path.rglob("*.csv") if f.is_file()]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {downloaded_path}")

    for f in csv_files:
        dest = RAW_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            print(f"Copied {f.name} to {dest}")
        else:
            print(f"{f.name} already exists in {RAW_DIR}")

    accepted = RAW_DIR / "accepted_2007_to_2018Q4.csv.gz"
    if accepted.exists():
        return accepted

    csv_in_raw = [f for f in RAW_DIR.iterdir() if f.suffix in (".csv", ".gz")]
    if not csv_in_raw:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
    return csv_in_raw[0]


if __name__ == "__main__":
    path = download_lending_club_data()
    print(f"\nData ready at: {path}")
