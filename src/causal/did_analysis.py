"""Diff-in-Diff analysis — effect of ~2016 credit tightening on default rates."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config.settings import PROCESSED_DIR, OUTPUTS_DIR

CAUSAL_OUTPUT_DIR = OUTPUTS_DIR / "causal"
TREATMENT_YEAR = 2016


def load_did_data() -> pd.DataFrame:
    """Load and prepare data for DiD analysis."""
    df = pd.read_parquet(PROCESSED_DIR / "clean_loans.parquet")

    if "fico_avg" not in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    df = df[df["issue_year"] >= 2010].copy()
    df["treated"] = (df["term"] == 60).astype(int)
    df["post"] = (df["issue_year"] >= TREATMENT_YEAR).astype(int)
    df["treated_post"] = df["treated"] * df["post"]

    print(f"DiD dataset: {len(df):,} rows")
    print(f"Year range: {df['issue_year'].min()}-{df['issue_year'].max()}")
    print(f"Treatment group (60m): {df['treated'].sum():,}")
    print(f"Control group (36m):   {(df['treated'] == 0).sum():,}")
    print(f"Pre-treatment:  {(df['post'] == 0).sum():,}")
    print(f"Post-treatment: {(df['post'] == 1).sum():,}")

    return df


def plot_parallel_trends(df: pd.DataFrame) -> None:
    """Plot default rate trends for treatment and control groups."""
    CAUSAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold", "savefig.bbox": "tight"})

    yearly = df.groupby(["issue_year", "treated"])["default_flag"].agg(["mean", "count"]).reset_index()
    yearly.columns = ["year", "treated", "default_rate", "count"]
    yearly["group"] = yearly["treated"].map({0: "36-month (Control)", 1: "60-month (Treatment)"})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for group_name, group_data in yearly.groupby("group"):
        color = "#3b82f6" if "36" in group_name else "#ef4444"
        axes[0].plot(group_data["year"], group_data["default_rate"] * 100, "o-",
                     label=group_name, color=color, linewidth=2, markersize=6)

    axes[0].axvline(x=TREATMENT_YEAR - 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    axes[0].text(TREATMENT_YEAR - 0.4, axes[0].get_ylim()[1] * 0.95, "Treatment\n(2016)",
                 fontsize=9, color="gray", ha="left", va="top")
    axes[0].set_xlabel("Issue Year")
    axes[0].set_ylabel("Default Rate (%)")
    axes[0].set_title("Parallel Trends Check")
    axes[0].legend()
    axes[0].set_xticks(sorted(df["issue_year"].unique()))

    # Difference plot
    pivot = yearly.pivot(index="year", columns="treated", values="default_rate")
    if 0 in pivot.columns and 1 in pivot.columns:
        pivot["diff"] = (pivot[1] - pivot[0]) * 100
        axes[1].plot(pivot.index, pivot["diff"], "o-", color="#8b5cf6", linewidth=2, markersize=6)
        axes[1].axvline(x=TREATMENT_YEAR - 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
        pre_diff = pivot[pivot.index < TREATMENT_YEAR]["diff"].mean()
        axes[1].axhline(y=pre_diff, color="#22c55e", linestyle=":", alpha=0.7,
                        label=f"Pre-treatment avg diff: {pre_diff:.1f}pp")
        axes[1].set_xlabel("Issue Year")
        axes[1].set_ylabel("Difference in Default Rate (pp)")
        axes[1].set_title("60m − 36m Default Rate Gap")
        axes[1].legend()
        axes[1].set_xticks(sorted(df["issue_year"].unique()))

    fig.tight_layout()
    fig.savefig(CAUSAL_OUTPUT_DIR / "parallel_trends.png")
    plt.close(fig)
    print(f"  Saved: {CAUSAL_OUTPUT_DIR / 'parallel_trends.png'}")


def run_did_regression(df: pd.DataFrame) -> dict:
    """Run the Diff-in-Diff regression with controls."""
    CAUSAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("DIFF-IN-DIFF REGRESSION")
    print("=" * 60)

    # Simple DiD (no controls)
    print("\n--- Model 1: Simple DiD ---")
    model_simple = smf.ols("default_flag ~ treated + post + treated_post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["issue_year"]},
    )
    print(model_simple.summary().tables[1])

    # DiD with controls
    print("\n--- Model 2: DiD with Controls ---")
    controls = ["fico_avg", "dti", "loan_amnt", "annual_inc", "int_rate"]
    control_vars = " + ".join(controls)
    formula = f"default_flag ~ treated + post + treated_post + {control_vars}"

    model_controls = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["issue_year"]},
    )
    print(model_controls.summary().tables[1])

    # Extract key results
    did_coef_simple = model_simple.params["treated_post"]
    did_se_simple = model_simple.bse["treated_post"]
    did_pval_simple = model_simple.pvalues["treated_post"]

    did_coef_controls = model_controls.params["treated_post"]
    did_se_controls = model_controls.bse["treated_post"]
    did_pval_controls = model_controls.pvalues["treated_post"]

    print("\n" + "=" * 60)
    print("DiD TREATMENT EFFECT SUMMARY")
    print("=" * 60)
    print(f"\n  Simple DiD:")
    print(f"    Treatment effect: {did_coef_simple:.4f} ({did_coef_simple*100:.2f} pp)")
    print(f"    SE: {did_se_simple:.4f}, p-value: {did_pval_simple:.4f}")
    print(f"    95% CI: [{did_coef_simple - 1.96*did_se_simple:.4f}, {did_coef_simple + 1.96*did_se_simple:.4f}]")

    print(f"\n  DiD with Controls:")
    print(f"    Treatment effect: {did_coef_controls:.4f} ({did_coef_controls*100:.2f} pp)")
    print(f"    SE: {did_se_controls:.4f}, p-value: {did_pval_controls:.4f}")
    print(f"    95% CI: [{did_coef_controls - 1.96*did_se_controls:.4f}, {did_coef_controls + 1.96*did_se_controls:.4f}]")
    print(f"    R²: {model_controls.rsquared:.4f}")

    results = {
        "simple_coef": did_coef_simple, "simple_se": did_se_simple, "simple_pval": did_pval_simple,
        "controls_coef": did_coef_controls, "controls_se": did_se_controls, "controls_pval": did_pval_controls,
        "controls_r2": model_controls.rsquared,
        "n_obs": len(df),
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(CAUSAL_OUTPUT_DIR / "did_results.csv", index=False)
    print(f"\n  Saved: {CAUSAL_OUTPUT_DIR / 'did_results.csv'}")

    return results


def plot_did_visualization(df: pd.DataFrame, results: dict) -> None:
    """Create DiD effect visualization."""
    CAUSAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    means = df.groupby(["treated", "post"])["default_flag"].mean().reset_index()
    means["group"] = means["treated"].map({0: "36-month\n(Control)", 1: "60-month\n(Treatment)"})
    means["period"] = means["post"].map({0: "Pre-2016", 1: "Post-2016"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2x2 DiD table as grouped bar
    x = np.arange(2)
    width = 0.3
    pre = means[means["post"] == 0].sort_values("treated")
    post = means[means["post"] == 1].sort_values("treated")

    axes[0].bar(x - width / 2, pre["default_flag"] * 100, width, label="Pre-2016",
                color="#3b82f6", edgecolor="white")
    axes[0].bar(x + width / 2, post["default_flag"] * 100, width, label="Post-2016",
                color="#ef4444", edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["36-month\n(Control)", "60-month\n(Treatment)"])
    axes[0].set_ylabel("Default Rate (%)")
    axes[0].set_title("Default Rates: Before vs After Treatment")
    axes[0].legend()

    for i, (_, row) in enumerate(pre.iterrows()):
        axes[0].text(i - width / 2, row["default_flag"] * 100 + 0.5,
                     f"{row['default_flag']*100:.1f}%", ha="center", fontsize=9)
    for i, (_, row) in enumerate(post.iterrows()):
        axes[0].text(i + width / 2, row["default_flag"] * 100 + 0.5,
                     f"{row['default_flag']*100:.1f}%", ha="center", fontsize=9)

    # Effect size with CI
    models = ["Simple DiD", "DiD + Controls"]
    coefs = [results["simple_coef"] * 100, results["controls_coef"] * 100]
    ses = [results["simple_se"] * 100, results["controls_se"] * 100]
    ci_low = [c - 1.96 * s for c, s in zip(coefs, ses)]
    ci_high = [c + 1.96 * s for c, s in zip(coefs, ses)]

    y_pos = np.arange(len(models))
    axes[1].barh(y_pos, coefs, xerr=[np.array(coefs) - np.array(ci_low),
                                      np.array(ci_high) - np.array(coefs)],
                 color=["#3b82f6", "#8b5cf6"], edgecolor="white", height=0.4,
                 capsize=5, error_kw={"linewidth": 1.5})
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(models)
    axes[1].set_xlabel("Treatment Effect (percentage points)")
    axes[1].set_title("DiD Estimated Treatment Effect ± 95% CI")

    for i, (c, lo, hi) in enumerate(zip(coefs, ci_low, ci_high)):
        axes[1].text(max(c, hi) + 0.3, i, f"{c:.2f}pp\n[{lo:.2f}, {hi:.2f}]",
                     va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(CAUSAL_OUTPUT_DIR / "did_effect.png")
    plt.close(fig)
    print(f"  Saved: {CAUSAL_OUTPUT_DIR / 'did_effect.png'}")


def run_placebo_test(df: pd.DataFrame) -> None:
    """Run placebo test with fake treatment date (2013)."""
    CAUSAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pre_data = df[df["issue_year"] < TREATMENT_YEAR].copy()
    placebo_year = 2013
    pre_data["post_placebo"] = (pre_data["issue_year"] >= placebo_year).astype(int)
    pre_data["treated_post_placebo"] = pre_data["treated"] * pre_data["post_placebo"]

    model = smf.ols(
        "default_flag ~ treated + post_placebo + treated_post_placebo + fico_avg + dti + loan_amnt + annual_inc + int_rate",
        data=pre_data,
    ).fit(cov_type="cluster", cov_kwds={"groups": pre_data["issue_year"]})

    coef = model.params["treated_post_placebo"]
    pval = model.pvalues["treated_post_placebo"]

    print(f"\n  Placebo Test (fake treatment at {placebo_year}):")
    print(f"    Coefficient: {coef:.4f} ({coef*100:.2f} pp)")
    print(f"    p-value: {pval:.4f}")
    print(f"    {'PASS — not significant (good!)' if pval > 0.05 else 'FAIL — significant (parallel trends violated)'}")


def run_bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000) -> tuple[float, float]:
    """Bootstrap confidence interval for the DiD estimator."""
    CAUSAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Running {n_boot} bootstrap replications...")

    rng = np.random.RandomState(42)
    boot_coefs = []

    for i in range(n_boot):
        boot_idx = rng.choice(len(df), size=len(df), replace=True)
        boot_df = df.iloc[boot_idx]
        try:
            model = smf.ols("default_flag ~ treated + post + treated_post", data=boot_df).fit()
            boot_coefs.append(model.params["treated_post"])
        except Exception:
            continue

    boot_coefs = np.array(boot_coefs)
    ci_low = np.percentile(boot_coefs, 2.5)
    ci_high = np.percentile(boot_coefs, 97.5)
    boot_mean = boot_coefs.mean()

    print(f"  Bootstrap Results ({len(boot_coefs)} successful replications):")
    print(f"    Mean estimate: {boot_mean:.4f} ({boot_mean*100:.2f} pp)")
    print(f"    95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"    SE (bootstrap): {boot_coefs.std():.4f}")

    # Plot bootstrap distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(boot_coefs * 100, bins=40, alpha=0.7, color="#8b5cf6", edgecolor="white", density=True)
    ax.axvline(x=boot_mean * 100, color="#ef4444", linewidth=2, label=f"Mean: {boot_mean*100:.2f}pp")
    ax.axvline(x=ci_low * 100, color="gray", linestyle="--", linewidth=1.5, label=f"95% CI: [{ci_low*100:.2f}, {ci_high*100:.2f}]")
    ax.axvline(x=ci_high * 100, color="gray", linestyle="--", linewidth=1.5)
    ax.axvline(x=0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("DiD Treatment Effect (percentage points)")
    ax.set_ylabel("Density")
    ax.set_title(f"Bootstrap Distribution of DiD Estimator ({n_boot} replications)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CAUSAL_OUTPUT_DIR / "bootstrap_distribution.png")
    plt.close(fig)
    print(f"  Saved: {CAUSAL_OUTPUT_DIR / 'bootstrap_distribution.png'}")

    return ci_low, ci_high


def run_did_pipeline() -> dict:
    """Full Diff-in-Diff pipeline."""
    df = load_did_data()
    plot_parallel_trends(df)
    results = run_did_regression(df)
    plot_did_visualization(df, results)
    run_placebo_test(df)
    ci_low, ci_high = run_bootstrap_ci(df, n_boot=1000)
    results["boot_ci_low"] = ci_low
    results["boot_ci_high"] = ci_high

    print("\n" + "=" * 60)
    print("Diff-in-Diff analysis complete.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_did_pipeline()
