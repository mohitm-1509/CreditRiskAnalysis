"""Expected Loss calculation — EL = PD × LGD × EAD."""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR
from src.data.features import get_model_features

EL_OUTPUT_DIR = OUTPUTS_DIR / "expected_loss"


def assign_risk_tier(el_pct: pd.Series) -> pd.Series:
    """Assign risk tiers based on Expected Loss quantiles."""
    return pd.qcut(
        el_pct,
        q=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )


def compute_expected_loss() -> pd.DataFrame:
    """Compute Expected Loss for all test set borrowers."""
    print("Loading models and test data...")
    pd_model = joblib.load(MODELS_DIR / "pd_xgboost.joblib")
    lgd_model = joblib.load(MODELS_DIR / "lgd_xgboost.joblib")
    pd_features = joblib.load(MODELS_DIR / "pd_features.joblib")
    lgd_features = joblib.load(MODELS_DIR / "lgd_features.joblib")

    test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")
    print(f"Test set: {len(test):,} rows")

    X_pd = test[pd_features].values
    X_lgd = test[lgd_features].values

    pd_probs = pd_model.predict_proba(X_pd)[:, 1]
    lgd_preds = lgd_model.predict(X_lgd).clip(0, 1)
    ead = test["loan_amnt"].values

    el_dollar = pd_probs * lgd_preds * ead
    el_pct = pd_probs * lgd_preds

    result = pd.DataFrame({
        "loan_amnt": ead,
        "grade": test["grade"].values if "grade" in test.columns else "N/A",
        "int_rate": test["int_rate"].values if "int_rate" in test.columns else np.nan,
        "pd": pd_probs,
        "lgd": lgd_preds,
        "ead": ead,
        "expected_loss_dollar": el_dollar,
        "expected_loss_pct": el_pct,
        "actual_default": test["default_flag"].values,
    })
    result["risk_tier"] = assign_risk_tier(result["expected_loss_pct"])

    print(f"\nExpected Loss Summary:")
    print(f"  Mean PD:              {pd_probs.mean():.4f}")
    print(f"  Mean LGD:             {lgd_preds.mean():.4f}")
    print(f"  Mean EAD:             ${ead.mean():,.0f}")
    print(f"  Mean EL ($):          ${el_dollar.mean():,.2f}")
    print(f"  Mean EL (%):          {el_pct.mean():.4f}")
    print(f"  Total portfolio EL:   ${el_dollar.sum():,.0f}")
    print(f"  Total portfolio EAD:  ${ead.sum():,.0f}")
    print(f"  Portfolio loss rate:   {el_dollar.sum() / ead.sum():.4f}")

    print(f"\nRisk Tier Distribution:")
    tier_counts = result["risk_tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(result) * 100
        avg_el = result[result["risk_tier"] == tier]["expected_loss_dollar"].mean()
        print(f"  {tier:12s}: {count:>8,} ({pct:5.1f}%)  avg EL = ${avg_el:,.2f}")

    return result


def plot_expected_loss(result: pd.DataFrame) -> None:
    """Generate Expected Loss plots."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold", "savefig.bbox": "tight"})
    EL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. EL distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(result["expected_loss_dollar"], bins=50, alpha=0.7, color="#ef4444",
                 edgecolor="white", density=True)
    axes[0].set_title("Expected Loss Distribution ($)")
    axes[0].set_xlabel("Expected Loss ($)")
    axes[0].set_ylabel("Density")

    axes[1].hist(result["expected_loss_pct"], bins=50, alpha=0.7, color="#8b5cf6",
                 edgecolor="white", density=True)
    axes[1].set_title("Expected Loss Distribution (%)")
    axes[1].set_xlabel("Expected Loss (%)")
    axes[1].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(EL_OUTPUT_DIR / "el_distribution.png")
    plt.close(fig)
    print(f"  Saved: {EL_OUTPUT_DIR / 'el_distribution.png'}")

    # 2. Risk tier distribution
    tier_order = ["Low", "Medium", "High", "Very High"]
    tier_colors = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444", "Very High": "#7c2d12"}
    tier_counts = result["risk_tier"].value_counts().reindex(tier_order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(tier_counts.index, tier_counts.values,
                color=[tier_colors[t] for t in tier_counts.index], edgecolor="white")
    axes[0].set_title("Number of Loans by Risk Tier")
    axes[0].set_xlabel("Risk Tier")
    axes[0].set_ylabel("Count")
    for i, (tier, count) in enumerate(tier_counts.items()):
        axes[0].text(i, count + len(result) * 0.005, f"{count:,}", ha="center", fontsize=9)

    tier_el = result.groupby("risk_tier", observed=True)["expected_loss_dollar"].sum().reindex(tier_order)
    axes[1].bar(tier_el.index, tier_el.values / 1e6,
                color=[tier_colors[t] for t in tier_el.index], edgecolor="white")
    axes[1].set_title("Total Expected Loss by Risk Tier ($M)")
    axes[1].set_xlabel("Risk Tier")
    axes[1].set_ylabel("Expected Loss ($M)")
    for i, (tier, el) in enumerate(tier_el.items()):
        axes[1].text(i, el / 1e6 + tier_el.max() / 1e6 * 0.01, f"${el/1e6:.1f}M", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(EL_OUTPUT_DIR / "risk_tier_distribution.png")
    plt.close(fig)
    print(f"  Saved: {EL_OUTPUT_DIR / 'risk_tier_distribution.png'}")

    # 3. EL by grade
    if "grade" in result.columns:
        grade_stats = result.groupby("grade").agg(
            mean_pd=("pd", "mean"),
            mean_lgd=("lgd", "mean"),
            mean_el=("expected_loss_dollar", "mean"),
            total_el=("expected_loss_dollar", "sum"),
            count=("pd", "count"),
            actual_default_rate=("actual_default", "mean"),
        ).reset_index().sort_values("grade")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(grade_stats))
        width = 0.25
        axes[0].bar(x - width, grade_stats["mean_pd"], width, label="PD", color="#3b82f6", edgecolor="white")
        axes[0].bar(x, grade_stats["mean_lgd"], width, label="LGD", color="#ef4444", edgecolor="white")
        axes[0].bar(x + width, grade_stats["actual_default_rate"], width, label="Actual Default",
                    color="#22c55e", edgecolor="white")
        axes[0].set_xlabel("Grade")
        axes[0].set_ylabel("Rate")
        axes[0].set_title("PD, LGD, and Actual Default Rate by Grade")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(grade_stats["grade"])
        axes[0].legend()

        axes[1].bar(grade_stats["grade"], grade_stats["total_el"] / 1e6,
                    color=sns.color_palette("YlOrRd", n_colors=len(grade_stats)), edgecolor="white")
        axes[1].set_xlabel("Grade")
        axes[1].set_ylabel("Total Expected Loss ($M)")
        axes[1].set_title("Total Expected Loss by Grade")
        for i, (_, row) in enumerate(grade_stats.iterrows()):
            axes[1].text(i, row["total_el"] / 1e6 + grade_stats["total_el"].max() / 1e6 * 0.01,
                        f"${row['total_el']/1e6:.1f}M", ha="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(EL_OUTPUT_DIR / "el_by_grade.png")
        plt.close(fig)
        print(f"  Saved: {EL_OUTPUT_DIR / 'el_by_grade.png'}")

    # 4. PD vs LGD scatter
    sample_idx = np.random.RandomState(42).choice(len(result), size=min(15_000, len(result)), replace=False)
    sample = result.iloc[sample_idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        sample["pd"], sample["lgd"],
        c=sample["expected_loss_dollar"], cmap="YlOrRd",
        alpha=0.3, s=10, edgecolors="none",
    )
    ax.set_xlabel("Probability of Default (PD)")
    ax.set_ylabel("Loss Given Default (LGD)")
    ax.set_title("PD vs LGD (coloured by Expected Loss $)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Expected Loss ($)")
    fig.tight_layout()
    fig.savefig(EL_OUTPUT_DIR / "pd_vs_lgd_scatter.png")
    plt.close(fig)
    print(f"  Saved: {EL_OUTPUT_DIR / 'pd_vs_lgd_scatter.png'}")

    # Save summary table
    grade_stats_out = result.groupby("grade").agg(
        count=("pd", "count"),
        mean_pd=("pd", "mean"),
        mean_lgd=("lgd", "mean"),
        mean_el_dollar=("expected_loss_dollar", "mean"),
        total_el_dollar=("expected_loss_dollar", "sum"),
        mean_ead=("ead", "mean"),
        actual_default_rate=("actual_default", "mean"),
    ).reset_index()
    grade_stats_out.to_csv(EL_OUTPUT_DIR / "el_summary_by_grade.csv", index=False)
    print(f"  Saved: {EL_OUTPUT_DIR / 'el_summary_by_grade.csv'}")


def run_expected_loss_pipeline() -> pd.DataFrame:
    """Full Expected Loss pipeline."""
    result = compute_expected_loss()
    plot_expected_loss(result)

    EL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(EL_OUTPUT_DIR / "expected_loss_results.parquet", index=False)
    print(f"\nResults saved to {EL_OUTPUT_DIR / 'expected_loss_results.parquet'}")

    return result


if __name__ == "__main__":
    run_expected_loss_pipeline()
