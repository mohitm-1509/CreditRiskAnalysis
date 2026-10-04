"""Univariate EDA — distributions and default rates by segment."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import PROCESSED_DIR
from src.eda.plots import setup_style, save_plot, add_bar_labels, pct_formatter, dollar_formatter


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "clean_loans.parquet"
    df = pd.read_parquet(path)
    if "fico_avg" not in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df


def plot_loan_amount_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["loan_amnt"], bins=40, kde=True, ax=axes[0], color="#2563eb")
    axes[0].set_title("Loan Amount Distribution")
    axes[0].set_xlabel("Loan Amount ($)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(dollar_formatter))

    df_plot = df.assign(status=df["default_flag"].map({0: "Fully Paid", 1: "Defaulted"}))
    sns.boxplot(x="status", y="loan_amnt", data=df_plot, hue="status", ax=axes[1],
                palette={"Fully Paid": "#22c55e", "Defaulted": "#ef4444"}, legend=False)
    axes[1].set_title("Loan Amount by Default Status")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Loan Amount ($)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(dollar_formatter))

    fig.tight_layout()
    save_plot(fig, "loan_amount_distribution")


def plot_interest_rate_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["int_rate"], bins=40, kde=True, ax=axes[0], color="#7c3aed")
    axes[0].set_title("Interest Rate Distribution")
    axes[0].set_xlabel("Interest Rate (%)")

    sns.kdeplot(data=df, x="int_rate", hue="default_flag", ax=axes[1],
                fill=True, alpha=0.4, palette={0: "#22c55e", 1: "#ef4444"})
    axes[1].set_title("Interest Rate by Default Status")
    axes[1].set_xlabel("Interest Rate (%)")
    axes[1].legend(["Fully Paid", "Defaulted"])

    fig.tight_layout()
    save_plot(fig, "interest_rate_distribution")


def plot_fico_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["fico_avg"], bins=40, kde=True, ax=axes[0], color="#0891b2")
    axes[0].set_title("FICO Score Distribution")
    axes[0].set_xlabel("FICO Score")

    sns.kdeplot(data=df, x="fico_avg", hue="default_flag", ax=axes[1],
                fill=True, alpha=0.4, palette={0: "#22c55e", 1: "#ef4444"})
    axes[1].set_title("FICO Score by Default Status")
    axes[1].set_xlabel("FICO Score")
    axes[1].legend(["Fully Paid", "Defaulted"])

    fig.tight_layout()
    save_plot(fig, "fico_distribution")


def plot_dti_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    filtered = df[df["dti"] <= 60]
    sns.histplot(filtered["dti"], bins=40, kde=True, ax=axes[0], color="#ea580c")
    axes[0].set_title("DTI Distribution (capped at 60)")
    axes[0].set_xlabel("Debt-to-Income Ratio")

    sns.kdeplot(data=filtered, x="dti", hue="default_flag", ax=axes[1],
                fill=True, alpha=0.4, palette={0: "#22c55e", 1: "#ef4444"})
    axes[1].set_title("DTI by Default Status")
    axes[1].set_xlabel("Debt-to-Income Ratio")
    axes[1].legend(["Fully Paid", "Defaulted"])

    fig.tight_layout()
    save_plot(fig, "dti_distribution")


def plot_default_rate_by_grade(df: pd.DataFrame) -> None:
    rates = df.groupby("grade")["default_flag"].agg(["mean", "count"]).reset_index()
    rates.columns = ["grade", "default_rate", "count"]
    rates = rates.sort_values("grade")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(rates["grade"], rates["default_rate"] * 100,
                   color=sns.color_palette("YlOrRd", n_colors=len(rates)), edgecolor="white")
    ax1.set_xlabel("Loan Grade")
    ax1.set_ylabel("Default Rate (%)")
    ax1.set_title("Default Rate by Loan Grade")
    add_bar_labels(ax1)

    ax2 = ax1.twinx()
    ax2.plot(rates["grade"], rates["count"], "ko-", markersize=6, linewidth=1.5)
    ax2.set_ylabel("Number of Loans")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    fig.tight_layout()
    save_plot(fig, "default_rate_by_grade")


def plot_default_rate_by_fico_band(df: pd.DataFrame) -> None:
    bands = pd.cut(df["fico_avg"],
                   bins=[0, 650, 670, 690, 710, 730, 750, 900],
                   labels=["<650", "650-669", "670-689", "690-709", "710-729", "730-749", "750+"])
    rates = df.assign(fico_band=bands).groupby("fico_band", observed=True)["default_flag"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    rates.plot(kind="bar", ax=ax, color=sns.color_palette("YlOrRd_r", n_colors=len(rates)), edgecolor="white")
    ax.set_title("Default Rate by FICO Band")
    ax.set_xlabel("FICO Score Band")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    add_bar_labels(ax)

    fig.tight_layout()
    save_plot(fig, "default_rate_by_fico_band")


def plot_default_rate_by_term(df: pd.DataFrame) -> None:
    rates = df.groupby("term")["default_flag"].agg(["mean", "count"]).reset_index()
    rates.columns = ["term", "default_rate", "count"]
    rates["term_label"] = rates["term"].astype(int).astype(str) + " months"

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(rates["term_label"], rates["default_rate"] * 100,
                  color=["#3b82f6", "#ef4444"], edgecolor="white", width=0.5)
    ax.set_title("Default Rate by Loan Term")
    ax.set_xlabel("Loan Term")
    ax.set_ylabel("Default Rate (%)")
    add_bar_labels(ax)

    for i, row in rates.iterrows():
        ax.text(i, -1.5, f"n={row['count']:,.0f}", ha="center", fontsize=9, color="gray")

    fig.tight_layout()
    save_plot(fig, "default_rate_by_term")


def plot_default_rate_by_purpose(df: pd.DataFrame) -> None:
    rates = df.groupby("purpose")["default_flag"].agg(["mean", "count"]).reset_index()
    rates.columns = ["purpose", "default_rate", "count"]
    rates = rates[rates["count"] >= 1000].sort_values("default_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("YlOrRd", n_colors=len(rates))
    ax.barh(rates["purpose"], rates["default_rate"] * 100, color=colors, edgecolor="white")
    ax.set_title("Default Rate by Loan Purpose (n >= 1,000)")
    ax.set_xlabel("Default Rate (%)")

    for i, (_, row) in enumerate(rates.iterrows()):
        ax.text(row["default_rate"] * 100 + 0.3, i, f"{row['default_rate']*100:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    save_plot(fig, "default_rate_by_purpose")


def plot_default_rate_by_income_bracket(df: pd.DataFrame) -> None:
    brackets = pd.cut(df["annual_inc"],
                      bins=[0, 30000, 50000, 75000, 100000, 150000, float("inf")],
                      labels=["<30K", "30-50K", "50-75K", "75-100K", "100-150K", "150K+"])
    rates = df.assign(income=brackets).groupby("income", observed=True)["default_flag"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    rates.plot(kind="bar", ax=ax, color=sns.color_palette("YlOrRd_r", n_colors=len(rates)), edgecolor="white")
    ax.set_title("Default Rate by Income Bracket")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    add_bar_labels(ax)

    fig.tight_layout()
    save_plot(fig, "default_rate_by_income_bracket")


def plot_default_rate_by_home_ownership(df: pd.DataFrame) -> None:
    top = df["home_ownership"].value_counts().head(4).index
    subset = df[df["home_ownership"].isin(top)]
    rates = subset.groupby("home_ownership")["default_flag"].agg(["mean", "count"]).reset_index()
    rates.columns = ["home_ownership", "default_rate", "count"]
    rates = rates.sort_values("default_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rates["home_ownership"], rates["default_rate"] * 100,
           color=sns.color_palette("Set2", n_colors=len(rates)), edgecolor="white")
    ax.set_title("Default Rate by Home Ownership")
    ax.set_xlabel("Home Ownership")
    ax.set_ylabel("Default Rate (%)")
    add_bar_labels(ax)

    fig.tight_layout()
    save_plot(fig, "default_rate_by_home_ownership")


def run_univariate_eda() -> None:
    """Run all univariate EDA plots."""
    setup_style()
    print("Loading data...")
    df = load_data()
    print(f"Data shape: {df.shape}")
    print(f"Default rate: {df['default_flag'].mean():.2%}\n")

    print("Generating univariate plots...")
    plot_loan_amount_distribution(df)
    plot_interest_rate_distribution(df)
    plot_fico_distribution(df)
    plot_dti_distribution(df)
    plot_default_rate_by_grade(df)
    plot_default_rate_by_fico_band(df)
    plot_default_rate_by_term(df)
    plot_default_rate_by_purpose(df)
    plot_default_rate_by_income_bracket(df)
    plot_default_rate_by_home_ownership(df)

    print("\nUnivariate EDA complete.")


if __name__ == "__main__":
    run_univariate_eda()
