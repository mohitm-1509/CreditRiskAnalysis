"""Local SHAP analysis — individual borrower explanations."""

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from config.settings import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED

SHAP_OUTPUT_DIR = OUTPUTS_DIR / "shap"


def select_interesting_cases(X_test: pd.DataFrame, y_test: np.ndarray,
                              pd_probs: np.ndarray) -> list[dict]:
    """Select interesting borrowers for local explanation."""
    cases = []

    # Case 1: High PD, actually defaulted — model got it right
    high_pd_default = np.where((pd_probs > np.percentile(pd_probs, 95)) & (y_test == 1))[0]
    if len(high_pd_default) > 0:
        idx = high_pd_default[0]
        cases.append({"idx": idx, "label": "High PD — True Default",
                       "description": "Model correctly flagged this as high risk"})

    # Case 2: Low PD, actually defaulted — model missed it
    low_pd_default = np.where((pd_probs < np.percentile(pd_probs, 25)) & (y_test == 1))[0]
    if len(low_pd_default) > 0:
        idx = low_pd_default[0]
        cases.append({"idx": idx, "label": "Low PD — Surprise Default",
                       "description": "Model missed this default — appeared low risk"})

    # Case 3: High PD, fully paid — false alarm
    high_pd_paid = np.where((pd_probs > np.percentile(pd_probs, 90)) & (y_test == 0))[0]
    if len(high_pd_paid) > 0:
        idx = high_pd_paid[0]
        cases.append({"idx": idx, "label": "High PD — False Alarm",
                       "description": "Model flagged as risky but borrower paid in full"})

    # Case 4: Median PD, defaulted — borderline case
    median_pd = np.percentile(pd_probs, 50)
    near_median = np.where((np.abs(pd_probs - median_pd) < 0.02) & (y_test == 1))[0]
    if len(near_median) > 0:
        idx = near_median[0]
        cases.append({"idx": idx, "label": "Borderline PD — Default",
                       "description": "Borderline case that ended up defaulting"})

    # Case 5: Very low PD, fully paid — model confidently correct
    very_low_pd_paid = np.where((pd_probs < np.percentile(pd_probs, 5)) & (y_test == 0))[0]
    if len(very_low_pd_paid) > 0:
        idx = very_low_pd_paid[0]
        cases.append({"idx": idx, "label": "Very Low PD — Fully Paid",
                       "description": "Model confidently predicted low risk, confirmed"})

    return cases


def plot_waterfall_cases(shap_values: shap.Explanation, cases: list[dict],
                         pd_probs: np.ndarray) -> None:
    """Plot SHAP waterfall for each interesting case."""
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, case in enumerate(cases):
        idx = case["idx"]
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
        plt.title(f"{case['label']} (PD={pd_probs[idx]:.3f})\n{case['description']}",
                  fontsize=11, fontweight="bold", pad=15)
        plt.tight_layout()
        fname = f"waterfall_case_{i+1}.png"
        plt.savefig(SHAP_OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {SHAP_OUTPUT_DIR / fname}")


def plot_force_cases(shap_values: shap.Explanation, cases: list[dict],
                     pd_probs: np.ndarray, feature_names: list[str]) -> None:
    """Plot SHAP force plots for interesting cases as a combined figure."""
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(cases), 1, figsize=(16, 3 * len(cases)))
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        idx = case["idx"]
        sv = shap_values[idx]
        vals = sv.values
        base = sv.base_values

        sorted_idx = np.argsort(np.abs(vals))[::-1][:10]
        top_features = [(feature_names[j], vals[j]) for j in sorted_idx]

        pos_feats = [(f, v) for f, v in top_features if v > 0]
        neg_feats = [(f, v) for f, v in top_features if v <= 0]

        y_pos = 0
        for f, v in pos_feats:
            ax.barh(y_pos, v, height=0.6, color="#ef4444", alpha=0.8)
            ax.text(v + 0.002, y_pos, f"{f}: +{v:.3f}", va="center", fontsize=8)
            y_pos += 1

        for f, v in neg_feats:
            ax.barh(y_pos, v, height=0.6, color="#3b82f6", alpha=0.8)
            ax.text(v - 0.002, y_pos, f"{f}: {v:.3f}", va="center", fontsize=8, ha="right")
            y_pos += 1

        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.set_title(f"{case['label']} — PD={pd_probs[idx]:.3f}", fontsize=10, fontweight="bold")
        ax.set_yticks([])
        ax.set_xlabel("SHAP value")

    fig.suptitle("SHAP Force Decomposition — Interesting Cases", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SHAP_OUTPUT_DIR / "force_plots_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {SHAP_OUTPUT_DIR / 'force_plots_combined.png'}")


def create_case_summary(cases: list[dict], X_sample: pd.DataFrame,
                         pd_probs: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Create a summary table of interesting cases."""
    rows = []
    key_cols = ["int_rate", "fico_avg", "dti", "loan_amnt", "annual_inc",
                "term", "revol_util", "credit_history_years"]
    key_cols = [c for c in key_cols if c in X_sample.columns]

    for case in cases:
        idx = case["idx"]
        row = {"case": case["label"], "pd": pd_probs[idx],
               "actual_default": int(y_test[idx])}
        for col in key_cols:
            row[col] = X_sample.iloc[idx][col]
        rows.append(row)

    df = pd.DataFrame(rows)
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SHAP_OUTPUT_DIR / "interesting_cases.csv", index=False)
    print(f"  Saved: {SHAP_OUTPUT_DIR / 'interesting_cases.csv'}")

    print("\n  Interesting Cases Summary:")
    print(df.to_string(index=False))
    return df


def run_local_shap() -> None:
    """Run local SHAP analysis on interesting borrower cases."""
    print("Loading models and data...")
    xgb_model = joblib.load(MODELS_DIR / "pd_xgboost.joblib")
    pd_features = joblib.load(MODELS_DIR / "pd_features.joblib")
    test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")

    X_test = test[pd_features]
    y_test = test["default_flag"].values

    print("Computing PD predictions...")
    pd_probs = xgb_model.predict_proba(X_test.values)[:, 1]

    rng = np.random.RandomState(RANDOM_SEED)
    sample_size = min(5_000, len(X_test))
    idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test.iloc[idx].copy()
    y_sample = y_test[idx]
    pd_sample = pd_probs[idx]

    print("Computing SHAP values on sample...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample)

    cases = select_interesting_cases(X_sample, y_sample, pd_sample)
    print(f"\nSelected {len(cases)} interesting cases:")
    for c in cases:
        print(f"  - {c['label']}: {c['description']}")

    plot_waterfall_cases(shap_values, cases, pd_sample)
    plot_force_cases(shap_values, cases, pd_sample, pd_features)
    create_case_summary(cases, X_sample, pd_sample, y_sample)

    print("\nLocal SHAP analysis complete.")


if __name__ == "__main__":
    run_local_shap()
