"""Feature engineering pipeline for credit risk modelling."""

import numpy as np
import pandas as pd

from config.settings import PROCESSED_DIR, TRAIN_END_YEAR, TEST_START_YEAR


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned parquet file."""
    path = PROCESSED_DIR / "clean_loans.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Clean data not found at {path}. Run clean.py first.")
    return pd.read_parquet(path)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for modelling."""
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income_ratio"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
        median_lti = df["loan_to_income_ratio"].median()
        df["loan_to_income_ratio"] = df["loan_to_income_ratio"].fillna(median_lti)

    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        df["credit_history_years"] = (
            (pd.to_datetime(df["issue_d"]) - pd.to_datetime(df["earliest_cr_line"])).dt.days / 365.25
        )
        df["credit_history_years"] = df["credit_history_years"].clip(lower=0)
        median_ch = df["credit_history_years"].median()
        df["credit_history_years"] = df["credit_history_years"].fillna(median_ch)

    if "revol_util" in df.columns:
        df["credit_util_bin"] = pd.cut(
            df["revol_util"],
            bins=[0, 20, 40, 60, 80, 150],
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            include_lowest=True,
        )

    if "fico_avg" in df.columns:
        df["fico_band"] = pd.cut(
            df["fico_avg"],
            bins=[0, 600, 650, 700, 750, 900],
            labels=["Very Poor", "Poor", "Fair", "Good", "Excellent"],
            include_lowest=True,
        )

    if "dti" in df.columns:
        df["dti"] = df["dti"].clip(lower=0)
        df["dti_bin"] = pd.cut(
            df["dti"],
            bins=[0, 10, 20, 30, 100],
            labels=["Low", "Medium", "High", "Very High"],
            include_lowest=True,
        )

    if "annual_inc" in df.columns:
        df["income_bracket"] = pd.cut(
            df["annual_inc"],
            bins=[0, 30000, 60000, 100000, float("inf")],
            labels=["Low", "Medium", "High", "Very High"],
            include_lowest=True,
        )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features for modelling."""
    cat_columns = ["purpose", "home_ownership", "verification_status", "initial_list_status", "application_type"]
    cat_columns = [c for c in cat_columns if c in df.columns]

    df = pd.get_dummies(df, columns=cat_columns, drop_first=True, dtype=int)
    return df


def compute_woe_iv(df: pd.DataFrame, feature: str, target: str = "default_flag") -> pd.DataFrame:
    """Compute Weight of Evidence and Information Value for a feature."""
    grouped = df.groupby(feature)[target].agg(["sum", "count"])
    grouped.columns = ["events", "total"]
    grouped["non_events"] = grouped["total"] - grouped["events"]

    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    grouped["pct_events"] = grouped["events"] / total_events
    grouped["pct_non_events"] = grouped["non_events"] / total_non_events

    grouped["pct_events"] = grouped["pct_events"].replace(0, 0.0001)
    grouped["pct_non_events"] = grouped["pct_non_events"].replace(0, 0.0001)

    grouped["woe"] = np.log(grouped["pct_non_events"] / grouped["pct_events"])
    grouped["iv"] = (grouped["pct_non_events"] - grouped["pct_events"]) * grouped["woe"]

    return grouped


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time: train on 2007-2016, test on 2017-2018."""
    if "issue_year" not in df.columns:
        raise ValueError("issue_year column required. Run parse_dates first.")

    train = df[df["issue_year"] <= TRAIN_END_YEAR].copy()
    test = df[df["issue_year"] >= TEST_START_YEAR].copy()

    print(f"Train: {len(train):,} rows ({train['issue_year'].min()}-{train['issue_year'].max()})")
    print(f"Test:  {len(test):,} rows ({test['issue_year'].min()}-{test['issue_year'].max()})")
    print(f"Train default rate: {train['default_flag'].mean():.2%}")
    print(f"Test default rate:  {test['default_flag'].mean():.2%}")

    return train, test


def get_model_features(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (exclude target, IDs, dates, raw categoricals)."""
    exclude = [
        "default_flag", "loan_status", "issue_d", "earliest_cr_line",
        "issue_year", "issue_month", "emp_length", "emp_title", "title",
        "grade", "sub_grade",
        "fico_band", "dti_bin", "credit_util_bin", "income_bracket",
        "total_pymnt", "total_rec_prncp", "total_rec_int",
        "total_rec_late_fee", "recoveries", "collection_recovery_fee",
        "last_pymnt_amnt", "last_fico_range_high", "last_fico_range_low",
    ]
    features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return features


def run_feature_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full feature engineering pipeline: load → derive → encode → split → save."""
    df = load_clean_data()
    df = create_derived_features(df)
    df = encode_categoricals(df)

    train, test = time_based_split(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "features_train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "features_test.parquet", index=False)
    print(f"\nSaved to {PROCESSED_DIR / 'features_train.parquet'}")
    print(f"Saved to {PROCESSED_DIR / 'features_test.parquet'}")

    features = get_model_features(train)
    print(f"\nModel features ({len(features)}): {features[:10]}...")

    return train, test


if __name__ == "__main__":
    train, test = run_feature_pipeline()
