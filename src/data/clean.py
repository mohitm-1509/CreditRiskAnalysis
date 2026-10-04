"""Data cleaning pipeline — mirrors the SQL logic for local development."""

import pandas as pd
import numpy as np

from config.settings import (
    PROCESSED_DIR,
    RAW_DIR,
    SELECTED_COLUMNS,
    TERMINAL_STATUSES,
    DEFAULT_STATUSES,
)


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw Lending Club CSV."""
    if path is None:
        candidates = list(RAW_DIR.glob("accepted_*.csv*"))
        if not candidates:
            candidates = list(RAW_DIR.glob("*.csv*"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files in {RAW_DIR}")
        path = str(candidates[0])

    print(f"Loading raw data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns we need."""
    available = [c for c in SELECTED_COLUMNS if c in df.columns]
    missing = set(SELECTED_COLUMNS) - set(available)
    if missing:
        print(f"Warning: columns not in data: {missing}")
    return df[available].copy()


def filter_terminal_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only loans with terminal status (Fully Paid, Charged Off, Default)."""
    before = len(df)
    df = df[df["loan_status"].isin(TERMINAL_STATUSES)].copy()
    print(f"Filtered to terminal statuses: {before:,} → {len(df):,} rows")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary default flag: 1 = Charged Off / Default, 0 = Fully Paid."""
    df["default_flag"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    default_rate = df["default_flag"].mean()
    print(f"Default rate: {default_rate:.2%}")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns and extract year/month."""
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="mixed", errors="coerce")
        df["issue_year"] = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month

    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(
            df["earliest_cr_line"], format="mixed", errors="coerce"
        )
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean numeric columns that may have formatting issues."""
    if "term" in df.columns:
        df["term"] = (
            df["term"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )

    if "int_rate" in df.columns:
        df["int_rate"] = (
            df["int_rate"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )

    if "revol_util" in df.columns:
        df["revol_util"] = (
            df["revol_util"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )

    if "emp_length" in df.columns:
        emp_map = {"< 1 year": 0, "1 year": 1, "10+ years": 10}
        for i in range(2, 10):
            emp_map[f"{i} years"] = i
        df["emp_length_num"] = df["emp_length"].map(emp_map)

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numerics, mode for categoricals."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            pct = n_missing / len(df) * 100
            if pct > 50:
                print(f"  Dropping {col}: {pct:.1f}% missing")
                df = df.drop(columns=[col])
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    for col in categorical_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            pct = n_missing / len(df) * 100
            if pct > 50:
                print(f"  Dropping {col}: {pct:.1f}% missing")
                df = df.drop(columns=[col])
            else:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)

    return df


def run_cleaning_pipeline(raw_path: str | None = None) -> pd.DataFrame:
    """Execute the full cleaning pipeline and save to parquet."""
    df = load_raw_data(raw_path)
    df = select_columns(df)
    df = filter_terminal_loans(df)
    df = create_target(df)
    df = parse_dates(df)
    df = clean_numeric_columns(df)
    df = handle_missing_values(df)

    df = df.drop(columns=["loan_status"], errors="ignore")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "clean_loans.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved cleaned data: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    df = run_cleaning_pipeline()
    print("\n--- Summary ---")
    print(df.info())
    print(df.describe())
