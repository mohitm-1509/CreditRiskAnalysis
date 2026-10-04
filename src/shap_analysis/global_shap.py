"""Global SHAP analysis — feature importance and summary plots."""

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED

SHAP_OUTPUT_DIR = OUTPUTS_DIR / "shap"
SHAP_SAMPLE_SIZE = 5_000


def load_models_and_data() -> tuple:
    """Load PD models, features, and test data."""
    xgb_model = joblib.load(MODELS_DIR / "pd_xgboost.joblib")
    lr_pipeline = joblib.load(MODELS_DIR / "pd_logreg.joblib")
    pd_features = joblib.load(MODELS_DIR / "pd_features.joblib")

    test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")
    X_test = test[pd_features]

    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X_test), size=min(SHAP_SAMPLE_SIZE, len(X_test)), replace=False)
    X_sample = X_test.iloc[idx].copy()

    print(f"Test set: {len(X_test):,} rows")
    print(f"SHAP sample: {len(X_sample):,} rows")

    return xgb_model, lr_pipeline, pd_features, X_test, X_sample


def compute_xgb_shap(xgb_model, X_sample: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for XGBoost using TreeExplainer."""
    print("\nComputing XGBoost SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample)
    print(f"  SHAP values shape: {shap_values.values.shape}")
    return shap_values


def plot_beeswarm(shap_values: shap.Explanation) -> None:
    """SHAP beeswarm summary plot."""
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Beeswarm — XGBoost PD Model", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_OUTPUT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {SHAP_OUTPUT_DIR / 'shap_beeswarm.png'}")


def plot_bar(shap_values: shap.Explanation) -> None:
    """SHAP mean |SHAP| bar plot."""
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("Mean |SHAP| — XGBoost PD Model", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_OUTPUT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {SHAP_OUTPUT_DIR / 'shap_bar.png'}")


def plot_lr_vs_xgb_importance(lr_pipeline, xgb_shap_values: shap.Explanation,
                               feature_names: list[str]) -> None:
    """Compare LR coefficients vs XGBoost SHAP importance side by side."""
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lr_coefs = lr_pipeline.named_steps["model"].coef_[0]
    scaler = lr_pipeline.named_steps["scaler"]
    scaled_coefs = lr_coefs * scaler.scale_
    lr_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(scaled_coefs),
    }).sort_values("importance", ascending=False)
    lr_importance["rank"] = range(1, len(lr_importance) + 1)

    xgb_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(xgb_shap_values.values).mean(axis=0),
    }).sort_values("importance", ascending=False)
    xgb_importance["rank"] = range(1, len(xgb_importance) + 1)

    top_n = 15
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    lr_top = lr_importance.head(top_n).sort_values("importance")
    axes[0].barh(lr_top["feature"], lr_top["importance"], color="#3b82f6", edgecolor="white")
    axes[0].set_title("Logistic Regression\n(|scaled coefficient|)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Importance")

    xgb_top = xgb_importance.head(top_n).sort_values("importance")
    axes[1].barh(xgb_top["feature"], xgb_top["importance"], color="#ef4444", edgecolor="white")
    axes[1].set_title("XGBoost\n(mean |SHAP|)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Importance")

    fig.suptitle("Feature Importance: Logistic Regression vs XGBoost", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SHAP_OUTPUT_DIR / "lr_vs_xgb_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {SHAP_OUTPUT_DIR / 'lr_vs_xgb_importance.png'}")

    merged = lr_importance[["feature", "rank"]].merge(
        xgb_importance[["feature", "rank"]], on="feature", suffixes=("_lr", "_xgb"),
    )
    merged["rank_diff"] = abs(merged["rank_lr"] - merged["rank_xgb"])
    merged = merged.sort_values("rank_xgb")

    print("\n  Feature Rank Comparison (top 15):")
    print(f"  {'Feature':<35s} {'LR Rank':>8s} {'XGB Rank':>9s} {'Diff':>5s}")
    print("  " + "-" * 60)
    for _, row in merged.head(15).iterrows():
        print(f"  {row['feature']:<35s} {row['rank_lr']:>8d} {row['rank_xgb']:>9d} {row['rank_diff']:>5d}")

    merged.to_csv(SHAP_OUTPUT_DIR / "feature_rank_comparison.csv", index=False)
    print(f"\n  Saved: {SHAP_OUTPUT_DIR / 'feature_rank_comparison.csv'}")


def plot_dependence(shap_values: shap.Explanation, X_sample: pd.DataFrame) -> None:
    """SHAP dependence plots for top features."""
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_features = pd.Series(mean_abs, index=X_sample.columns).nlargest(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, feat in zip(axes.ravel(), top_features):
        feat_idx = list(X_sample.columns).index(feat)
        shap.plots.scatter(shap_values[:, feat_idx], ax=ax, show=False)
        ax.set_title(feat, fontsize=11, fontweight="bold")

    fig.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SHAP_OUTPUT_DIR / "shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {SHAP_OUTPUT_DIR / 'shap_dependence.png'}")


def run_global_shap() -> shap.Explanation:
    """Run the full global SHAP analysis."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold", "savefig.bbox": "tight"})

    xgb_model, lr_pipeline, pd_features, X_test, X_sample = load_models_and_data()
    shap_values = compute_xgb_shap(xgb_model, X_sample)

    plot_beeswarm(shap_values)
    plot_bar(shap_values)
    plot_lr_vs_xgb_importance(lr_pipeline, shap_values, pd_features)
    plot_dependence(shap_values, X_sample)

    print("\nGlobal SHAP analysis complete.")
    return shap_values


if __name__ == "__main__":
    run_global_shap()
