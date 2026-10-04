"""Load raw CSV data into Google BigQuery."""

import pandas as pd
from google.cloud import bigquery

from config.settings import (
    BQ_DATASET,
    BQ_PROJECT,
    BQ_RAW_TABLE,
    RAW_DIR,
    SELECTED_COLUMNS,
)


def load_to_bigquery(csv_path: str | None = None) -> None:
    """Upload the Lending Club CSV to BigQuery.

    Args:
        csv_path: Path to the CSV file. If None, auto-detects from data/raw/.
    """
    if csv_path is None:
        candidates = list(RAW_DIR.glob("accepted_*.csv*"))
        if not candidates:
            candidates = list(RAW_DIR.glob("*.csv*"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files in {RAW_DIR}")
        csv_path = str(candidates[0])

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    available = [c for c in SELECTED_COLUMNS if c in df.columns]
    missing = set(SELECTED_COLUMNS) - set(available)
    if missing:
        print(f"Warning: columns not found in data: {missing}")
    df = df[available]
    print(f"Filtered to {len(available)} columns")

    client = bigquery.Client(project=BQ_PROJECT)

    dataset_ref = f"{BQ_PROJECT}.{BQ_DATASET}"
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset, exists_ok=True)
        print(f"Created dataset {dataset_ref}")

    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_RAW_TABLE}"
    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    print(f"Uploading to {table_id}...")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows:,} rows to {table_id}")


if __name__ == "__main__":
    load_to_bigquery()
