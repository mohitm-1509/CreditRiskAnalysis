"""Model evaluation utilities — metrics, calibration, and comparison plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from config.settings import OUTPUTS_DIR

MODELS_OUTPUT_DIR = OUTPUTS_DIR / "models"


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.titleweight": "bold",
        "savefig.bbox": "tight",
    })


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "gini": 2 * roc_auc_score(y_true, y_prob) - 1,
        "threshold": threshold,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, method: str = "youden") -> float:
    """Find optimal classification threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if method == "youden":
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
    else:
        raise ValueError(f"Unknown method: {method}")
    return thresholds[best_idx]


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(tpr - fpr)


def plot_roc_curves(results: dict, y_true: np.ndarray) -> None:
    """Plot ROC curves for multiple models."""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {"Logistic Regression": "#3b82f6", "XGBoost": "#ef4444"}
    for name, y_prob in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = colors.get(name, "#666666")
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(MODELS_OUTPUT_DIR / "roc_curves.png")
    plt.close(fig)
    print(f"  Saved: {MODELS_OUTPUT_DIR / 'roc_curves.png'}")


def plot_pr_curves(results: dict, y_true: np.ndarray) -> None:
    """Plot Precision-Recall curves for multiple models."""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    baseline = y_true.mean()
    colors = {"Logistic Regression": "#3b82f6", "XGBoost": "#ef4444"}
    for name, y_prob in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        color = colors.get(name, "#666666")
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.4f})", color=color, linewidth=2)

    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.2%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(MODELS_OUTPUT_DIR / "pr_curves.png")
    plt.close(fig)
    print(f"  Saved: {MODELS_OUTPUT_DIR / 'pr_curves.png'}")


def plot_calibration_curves(results: dict, y_true: np.ndarray) -> None:
    """Plot calibration curves for multiple models."""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {"Logistic Regression": "#3b82f6", "XGBoost": "#ef4444"}
    for name, y_prob in results.items():
        fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        color = colors.get(name, "#666666")
        ax.plot(mean_predicted, fraction_pos, "o-", label=name, color=color, linewidth=2, markersize=6)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Default Rate")
    ax.set_title("Calibration Curves")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(MODELS_OUTPUT_DIR / "calibration_curves.png")
    plt.close(fig)
    print(f"  Saved: {MODELS_OUTPUT_DIR / 'calibration_curves.png'}")


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray,
                          threshold: float, model_name: str) -> None:
    """Plot confusion matrix heatmap."""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Fully Paid", "Defaulted"],
                yticklabels=["Fully Paid", "Defaulted"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name} (threshold={threshold:.3f})")
    fig.tight_layout()
    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(MODELS_OUTPUT_DIR / fname)
    plt.close(fig)
    print(f"  Saved: {MODELS_OUTPUT_DIR / fname}")


def plot_score_distribution(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    """Plot predicted probability distributions by actual class."""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Fully Paid",
            color="#22c55e", density=True, edgecolor="white")
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Defaulted",
            color="#ef4444", density=True, edgecolor="white")
    ax.set_xlabel("Predicted Probability of Default")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {model_name}")
    ax.legend()
    fig.tight_layout()
    fname = f"score_distribution_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(MODELS_OUTPUT_DIR / fname)
    plt.close(fig)
    print(f"  Saved: {MODELS_OUTPUT_DIR / fname}")


def plot_lift_chart(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, n_bins: int = 10) -> None:
    """Plot lift and cumulative gains chart."""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    n = len(y_sorted)
    bin_size = n // n_bins
    baseline_rate = y_true.mean()

    deciles = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else n
        bin_rate = y_sorted[start:end].mean()
        deciles.append({
            "decile": i + 1,
            "default_rate": bin_rate,
            "lift": bin_rate / baseline_rate,
            "cumulative_capture": y_sorted[:end].sum() / y_true.sum(),
            "pct_population": end / n,
        })

    df_lift = pd.DataFrame(deciles)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(df_lift["decile"], df_lift["lift"], color=sns.color_palette("YlOrRd", n_bins), edgecolor="white")
    axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Decile (highest risk → lowest)")
    axes[0].set_ylabel("Lift")
    axes[0].set_title(f"Lift Chart — {model_name}")
    axes[0].set_xticks(range(1, n_bins + 1))

    axes[1].plot(df_lift["pct_population"] * 100, df_lift["cumulative_capture"] * 100,
                 "o-", color="#3b82f6", linewidth=2, markersize=6, label="Model")
    axes[1].plot([0, 100], [0, 100], "k--", alpha=0.4, label="Random")
    axes[1].set_xlabel("% Population")
    axes[1].set_ylabel("% Defaults Captured")
    axes[1].set_title(f"Cumulative Gains — {model_name}")
    axes[1].legend()

    fig.tight_layout()
    fname = f"lift_chart_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(MODELS_OUTPUT_DIR / fname)
    plt.close(fig)
    print(f"  Saved: {MODELS_OUTPUT_DIR / fname}")


def print_comparison_table(all_metrics: dict) -> None:
    """Print a side-by-side comparison table."""
    display_keys = [
        "auc_roc", "auc_pr", "gini", "brier_score",
        "accuracy", "precision", "recall", "f1", "specificity", "threshold",
    ]
    rows = []
    for key in display_keys:
        row = {"Metric": key}
        for model_name, metrics in all_metrics.items():
            row[model_name] = f"{metrics[key]:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Metric")
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)

    df.to_csv(MODELS_OUTPUT_DIR / "model_comparison.csv")
    print(f"\nSaved: {MODELS_OUTPUT_DIR / 'model_comparison.csv'}")
