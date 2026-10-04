"""Shared plotting utilities for EDA."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config.settings import OUTPUTS_DIR

EDA_DIR = OUTPUTS_DIR / "eda"


def setup_style() -> None:
    """Set consistent plot styling."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def save_plot(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    """Save a figure to outputs/eda/ and close it."""
    out_dir = EDA_DIR / subdir if subdir else EDA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def pct_formatter(x, _):
    """Format axis tick as percentage."""
    return f"{x:.0f}%"


def dollar_formatter(x, _):
    """Format axis tick as dollar amount."""
    if x >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:.0f}"


def add_bar_labels(ax, fmt="{:.1f}%", fontsize=9):
    """Add value labels on top of bars."""
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                fmt.format(height),
                ha="center", va="bottom", fontsize=fontsize,
            )
