"""Export data for Power BI dashboard consumption."""

import numpy as np
import pandas as pd
import joblib

from config.settings import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR

DASHBOARD_DIR = OUTPUTS_DIR / "dashboard"


def export_portfolio_overview() -> None:
    """Export portfolio-level data for Tab 1."""
    df = pd.read_parquet(PROCESSED_DIR / "clean_loans.parquet")
    if "fico_avg" not in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    monthly = df.groupby(["issue_year", "issue_month"]).agg(
        loan_count=("loan_amnt", "count"),
        total_exposure=("loan_amnt", "sum"),
        avg_loan=("loan_amnt", "mean"),
        default_rate=("default_flag", "mean"),
        avg_int_rate=("int_rate", "mean"),
        avg_fico=("fico_avg", "mean"),
        avg_dti=("dti", "mean"),
    ).reset_index()
    monthly["date"] = pd.to_datetime(monthly[["issue_year", "issue_month"]].assign(day=1)
                                      .rename(columns={"issue_year": "year", "issue_month": "month"}))
    monthly.to_csv(DASHBOARD_DIR / "monthly_portfolio.csv", index=False)
    print(f"  Saved: monthly_portfolio.csv ({len(monthly)} rows)")


def export_borrower_risk() -> None:
    """Export borrower-level risk data for Tab 2."""
    el_path = OUTPUTS_DIR / "expected_loss" / "expected_loss_results.parquet"
    if not el_path.exists():
        print("  SKIP: expected_loss_results.parquet not found")
        return

    el = pd.read_parquet(el_path)
    test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")

    if "fico_avg" not in test.columns:
        test["fico_avg"] = (test["fico_range_low"] + test["fico_range_high"]) / 2

    export_cols = ["grade", "int_rate", "fico_avg", "dti", "loan_amnt", "annual_inc",
                   "term", "purpose", "home_ownership"]
    export_cols = [c for c in export_cols if c in test.columns]

    borrower = test[export_cols].copy()
    borrower["pd"] = el["pd"].values
    borrower["lgd"] = el["lgd"].values
    borrower["expected_loss"] = el["expected_loss_dollar"].values
    borrower["risk_tier"] = el["risk_tier"].values
    borrower["actual_default"] = el["actual_default"].values

    borrower.to_csv(DASHBOARD_DIR / "borrower_risk.csv", index=False)
    print(f"  Saved: borrower_risk.csv ({len(borrower):,} rows)")


def export_vintage_analysis() -> None:
    """Export vintage analysis data for Tab 3."""
    df = pd.read_parquet(PROCESSED_DIR / "clean_loans.parquet")

    vintage = df.groupby(["issue_year", "grade"]).agg(
        loan_count=("loan_amnt", "count"),
        default_rate=("default_flag", "mean"),
        total_exposure=("loan_amnt", "sum"),
        avg_loan=("loan_amnt", "mean"),
    ).reset_index()
    vintage.to_csv(DASHBOARD_DIR / "vintage_by_grade.csv", index=False)
    print(f"  Saved: vintage_by_grade.csv ({len(vintage)} rows)")

    heatmap = df.groupby(["issue_year", "issue_month"])["default_flag"].mean().reset_index()
    heatmap.columns = ["year", "month", "default_rate"]
    heatmap.to_csv(DASHBOARD_DIR / "monthly_default_heatmap.csv", index=False)
    print(f"  Saved: monthly_default_heatmap.csv ({len(heatmap)} rows)")


def export_did_data() -> None:
    """Export DiD data for Tab 4."""
    df = pd.read_parquet(PROCESSED_DIR / "clean_loans.parquet")
    df = df[df["issue_year"] >= 2010].copy()

    did_trend = df.groupby(["issue_year", "term"])["default_flag"].agg(["mean", "count"]).reset_index()
    did_trend.columns = ["year", "term", "default_rate", "count"]
    did_trend["term_label"] = did_trend["term"].astype(int).astype(str) + " months"
    did_trend.to_csv(DASHBOARD_DIR / "did_trend.csv", index=False)
    print(f"  Saved: did_trend.csv ({len(did_trend)} rows)")

    did_results_path = OUTPUTS_DIR / "causal" / "did_results.csv"
    if did_results_path.exists():
        did_results = pd.read_csv(did_results_path)
        did_results.to_csv(DASHBOARD_DIR / "did_results.csv", index=False)
        print(f"  Saved: did_results.csv")


def run_export() -> None:
    """Export all data for Power BI."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting data for Power BI...\n")

    export_portfolio_overview()
    export_borrower_risk()
    export_vintage_analysis()
    export_did_data()

    print(f"\nAll exports saved to {DASHBOARD_DIR}/")
    print("\nPower BI Setup Instructions:")
    print("  1. Open Power BI Desktop")
    print("  2. Get Data → Text/CSV → select each CSV")
    print("  3. Tab 1 (Portfolio Overview): monthly_portfolio.csv")
    print("  4. Tab 2 (Borrower Risk): borrower_risk.csv")
    print("  5. Tab 3 (Default Trends): vintage_by_grade.csv + monthly_default_heatmap.csv")
    print("  6. Tab 4 (Policy Impact): did_trend.csv + did_results.csv")


if __name__ == "__main__":
    run_export()
