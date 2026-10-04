"""Bivariate EDA — correlations, cross-tabulations, and time series."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import PROCESSED_DIR
from src.eda.plots import setup_style, save_plot, pct_formatter


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "clean_loans.parquet"
    df = pd.read_parquet(path)
    if "fico_avg" not in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = [
        "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
        "fico_avg", "open_acc", "pub_rec", "revol_bal", "revol_util",
        "total_acc", "mort_acc", "pub_rec_bankruptcies", "emp_length_num",
        "default_flag",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Matrix — Key Numeric Features")
    fig.tight_layout()
    save_plot(fig, "correlation_heatmap")


def plot_default_rate_heatmap_grade_term(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(
        values="default_flag", index="grade",
        columns="term", aggfunc="mean",
    ) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Default Rate (%)"})
    ax.set_title("Default Rate (%) by Grade × Loan Term")
    ax.set_xlabel("Term (months)")
    ax.set_ylabel("Grade")
    fig.tight_layout()
    save_plot(fig, "default_rate_heatmap_grade_term")


def plot_monthly_default_rate_trend(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby(["issue_year", "issue_month"])["default_flag"]
        .agg(["mean", "count"])
        .reset_index()
    )
    monthly.columns = ["year", "month", "default_rate", "count"]
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    monthly = monthly.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(monthly["date"], monthly["default_rate"] * 100, color="#ef4444", linewidth=1.5)
    ax1.fill_between(monthly["date"], monthly["default_rate"] * 100, alpha=0.15, color="#ef4444")
    ax1.set_xlabel("Issue Date")
    ax1.set_ylabel("Default Rate (%)", color="#ef4444")
    ax1.set_title("Monthly Default Rate & Loan Volume Over Time")
    ax1.tick_params(axis="y", labelcolor="#ef4444")

    ax2 = ax1.twinx()
    ax2.bar(monthly["date"], monthly["count"], width=25, alpha=0.3, color="#3b82f6")
    ax2.set_ylabel("Number of Loans Issued", color="#3b82f6")
    ax2.tick_params(axis="y", labelcolor="#3b82f6")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    fig.tight_layout()
    save_plot(fig, "monthly_default_rate_trend")


def plot_default_rate_by_grade_over_time(df: pd.DataFrame) -> None:
    yearly = (
        df.groupby(["issue_year", "grade"])["default_flag"]
        .mean()
        .reset_index()
    )
    yearly.columns = ["year", "grade", "default_rate"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for grade in sorted(yearly["grade"].unique()):
        subset = yearly[yearly["grade"] == grade]
        ax.plot(subset["year"], subset["default_rate"] * 100, "o-", label=grade, markersize=4)

    ax.set_title("Default Rate by Grade Over Time")
    ax.set_xlabel("Issue Year")
    ax.set_ylabel("Default Rate (%)")
    ax.legend(title="Grade", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticks(sorted(df["issue_year"].unique()))

    fig.tight_layout()
    save_plot(fig, "default_rate_by_grade_over_time")


def plot_fico_vs_interest_rate(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(20_000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        sample["fico_avg"], sample["int_rate"],
        c=sample["default_flag"], cmap="RdYlGn_r",
        alpha=0.3, s=8, edgecolors="none",
    )
    ax.set_title("FICO Score vs Interest Rate (coloured by default)")
    ax.set_xlabel("FICO Score")
    ax.set_ylabel("Interest Rate (%)")
    cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(["Fully Paid", "Defaulted"])

    fig.tight_layout()
    save_plot(fig, "fico_vs_interest_rate")


def plot_loan_amount_vs_income(df: pd.DataFrame) -> None:
    sample = df[df["annual_inc"] <= 300_000].sample(n=min(20_000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        sample["annual_inc"], sample["loan_amnt"],
        c=sample["default_flag"], cmap="RdYlGn_r",
        alpha=0.3, s=8, edgecolors="none",
    )
    ax.set_title("Annual Income vs Loan Amount (coloured by default)")
    ax.set_xlabel("Annual Income ($)")
    ax.set_ylabel("Loan Amount ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(["Fully Paid", "Defaulted"])

    fig.tight_layout()
    save_plot(fig, "loan_amount_vs_income")


def plot_vintage_default_curves(df: pd.DataFrame) -> None:
    vintage = (
        df.groupby("issue_year")["default_flag"]
        .agg(["mean", "count"])
        .reset_index()
    )
    vintage.columns = ["year", "default_rate", "count"]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", n_colors=len(vintage))
    bars = ax.bar(vintage["year"].astype(str), vintage["default_rate"] * 100,
                  color=colors, edgecolor="white")
    ax.set_title("Default Rate by Loan Vintage Year")
    ax.set_xlabel("Issue Year")
    ax.set_ylabel("Default Rate (%)")

    for bar, (_, row) in zip(bars, vintage.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{row['default_rate']*100:.1f}%", ha="center", fontsize=8)

    fig.tight_layout()
    save_plot(fig, "vintage_default_curves")


def plot_dti_vs_fico_default_heatmap(df: pd.DataFrame) -> None:
    df_temp = df.copy()
    df_temp["fico_bin"] = pd.cut(df_temp["fico_avg"],
                                  bins=[0, 650, 680, 700, 720, 750, 900],
                                  labels=["<650", "650-679", "680-699", "700-719", "720-749", "750+"])
    df_temp["dti_bin"] = pd.cut(df_temp["dti"],
                                 bins=[0, 10, 15, 20, 25, 30, 100],
                                 labels=["0-10", "10-15", "15-20", "20-25", "25-30", "30+"])

    pivot = df_temp.pivot_table(values="default_flag", index="dti_bin",
                                 columns="fico_bin", aggfunc="mean", observed=True) * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Default Rate (%)"})
    ax.set_title("Default Rate (%) by DTI × FICO Band")
    ax.set_xlabel("FICO Band")
    ax.set_ylabel("DTI Range")

    fig.tight_layout()
    save_plot(fig, "dti_vs_fico_default_heatmap")


def generate_summary_stats(df: pd.DataFrame) -> None:
    """Print key summary statistics."""
    print("\n=== PORTFOLIO SUMMARY ===")
    print(f"Total loans:        {len(df):,}")
    print(f"Total exposure:     ${df['loan_amnt'].sum():,.0f}")
    print(f"Avg loan amount:    ${df['loan_amnt'].mean():,.0f}")
    print(f"Default rate:       {df['default_flag'].mean():.2%}")
    print(f"Avg interest rate:  {df['int_rate'].mean():.2f}%")
    print(f"Avg DTI:            {df['dti'].mean():.2f}")
    print(f"Avg FICO:           {df['fico_avg'].mean():.0f}")
    print(f"Date range:         {df['issue_year'].min()} - {df['issue_year'].max()}")

    print("\n=== DEFAULT RATE BY GRADE ===")
    grade_rates = df.groupby("grade")["default_flag"].agg(["mean", "count"])
    grade_rates.columns = ["default_rate", "count"]
    grade_rates["default_rate"] = grade_rates["default_rate"].map("{:.2%}".format)
    grade_rates["count"] = grade_rates["count"].map("{:,}".format)
    print(grade_rates.to_string())

    print("\n=== DEFAULT RATE BY TERM ===")
    term_rates = df.groupby("term")["default_flag"].agg(["mean", "count"])
    term_rates.columns = ["default_rate", "count"]
    term_rates["default_rate"] = term_rates["default_rate"].map("{:.2%}".format)
    term_rates["count"] = term_rates["count"].map("{:,}".format)
    print(term_rates.to_string())


def run_bivariate_eda() -> None:
    """Run all bivariate EDA plots."""
    setup_style()
    print("Loading data...")
    df = load_data()
    print(f"Data shape: {df.shape}\n")

    generate_summary_stats(df)

    print("\nGenerating bivariate plots...")
    plot_correlation_heatmap(df)
    plot_default_rate_heatmap_grade_term(df)
    plot_monthly_default_rate_trend(df)
    plot_default_rate_by_grade_over_time(df)
    plot_fico_vs_interest_rate(df)
    plot_loan_amount_vs_income(df)
    plot_vintage_default_curves(df)
    plot_dti_vs_fico_default_heatmap(df)

    print("\nBivariate EDA complete.")


if __name__ == "__main__":
    run_bivariate_eda()
