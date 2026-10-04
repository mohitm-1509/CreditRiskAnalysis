"""Analyze all 38 features after filter_terminal_loans(), before handle_missing_values().

Produces:
  - Console output: missing values, duplicates, statistical summaries, distributions
  - PDF report: outputs/eda/feature_analysis_report.pdf
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config.settings import (
    RAW_DIR,
    SELECTED_COLUMNS,
    TERMINAL_STATUSES,
    DEFAULT_STATUSES,
    OUTPUTS_DIR,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)

REPORT_DIR = OUTPUTS_DIR / "eda"


# ── Data Loading (pipeline up to filter_terminal_loans) ─────────────────────

def load_filtered_data() -> pd.DataFrame:
    """Load raw data and apply steps up to filter_terminal_loans + create_target."""
    path = str(list(RAW_DIR.glob("accepted_*.csv*"))[0])
    print(f"Loading raw data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Raw: {len(df):,} rows, {len(df.columns)} columns")

    available = [c for c in SELECTED_COLUMNS if c in df.columns]
    df = df[available].copy()
    print(f"After column selection: {len(df.columns)} columns")

    before = len(df)
    df = df[df["loan_status"].isin(TERMINAL_STATUSES)].copy()
    print(f"After filter_terminal_loans: {before:,} → {len(df):,} rows "
          f"(dropped {before - len(df):,} = {(before - len(df)) / before * 100:.1f}%)")

    df["default_flag"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    print(f"Default rate: {df['default_flag'].mean():.2%}\n")
    return df


def classify_columns(df: pd.DataFrame):
    """Split columns into numeric, categorical, and date lists."""
    date_cols = [c for c in ["issue_d", "earliest_cr_line"] if c in df.columns]
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        if col in date_cols:
            continue
        if df[col].dtype in ["float64", "int64"]:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols, date_cols


# ── Console Analysis ────────────────────────────────────────────────────────

def print_missing_values(df: pd.DataFrame):
    print("=" * 80)
    print("1. MISSING VALUES SUMMARY")
    print("=" * 80)
    rows = []
    for col in df.columns:
        n_miss = df[col].isna().sum()
        rows.append({
            "Feature": col,
            "Type": str(df[col].dtype),
            "Missing": f"{n_miss:,}",
            "Missing %": f"{n_miss / len(df) * 100:.2f}%",
            "Non-null": f"{df[col].notna().sum():,}",
        })
    mdf = pd.DataFrame(rows)
    mdf["_sort"] = [float(x.replace("%", "")) for x in mdf["Missing %"]]
    mdf = mdf.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    print(mdf.to_string(index=False))
    n_complete = sum(1 for _, r in mdf.iterrows() if r["Missing"] == "0")
    print(f"\nColumns with 0 missing: {n_complete}/{len(df.columns)}")


def print_duplicates(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("2. DUPLICATE ANALYSIS")
    print("=" * 80)
    n_full = df.duplicated().sum()
    print(f"Fully duplicate rows: {n_full:,} ({n_full / len(df) * 100:.4f}%)")
    print(f"\n{'Feature':<35} {'Unique':>10} {'Duplicated':>12} {'Dup %':>8}")
    print("-" * 70)
    for col in df.columns:
        n_uniq = df[col].nunique(dropna=False)
        n_dup = len(df) - n_uniq
        print(f"{col:<35} {n_uniq:>10,} {n_dup:>12,} {n_dup / len(df) * 100:>7.2f}%")


def print_numeric_stats(df: pd.DataFrame, numeric_cols: list):
    print("\n" + "=" * 80)
    print("3. STATISTICAL SUMMARY — NUMERIC FEATURES")
    print("=" * 80)
    for col in numeric_cols:
        s = df[col].dropna()
        print(f"\n--- {col} ---")
        print(f"  Count:    {len(s):,}")
        print(f"  Missing:  {df[col].isna().sum():,} ({df[col].isna().sum() / len(df) * 100:.2f}%)")
        print(f"  Mean:     {s.mean():,.4f}")
        print(f"  Std:      {s.std():,.4f}")
        print(f"  Min:      {s.min():,.4f}")
        print(f"  25%:      {s.quantile(0.25):,.4f}")
        print(f"  Median:   {s.median():,.4f}")
        print(f"  75%:      {s.quantile(0.75):,.4f}")
        print(f"  Max:      {s.max():,.4f}")
        print(f"  Skewness: {s.skew():,.4f}")
        print(f"  Kurtosis: {s.kurtosis():,.4f}")
        lower = s.mean() - 3 * s.std()
        upper = s.mean() + 3 * s.std()
        n_out = ((s < lower) | (s > upper)).sum()
        print(f"  Outliers (>3σ): {n_out:,} ({n_out / len(s) * 100:.2f}%)")
        n_zeros = (s == 0).sum()
        if n_zeros > 0:
            print(f"  Zeros:    {n_zeros:,} ({n_zeros / len(s) * 100:.2f}%)")


def print_categorical_stats(df: pd.DataFrame, categorical_cols: list):
    print("\n" + "=" * 80)
    print("4. STATISTICAL SUMMARY — CATEGORICAL FEATURES")
    print("=" * 80)
    for col in categorical_cols:
        s = df[col].dropna()
        print(f"\n--- {col} ---")
        print(f"  Count:      {len(s):,}")
        print(f"  Missing:    {df[col].isna().sum():,} ({df[col].isna().sum() / len(df) * 100:.2f}%)")
        print(f"  Unique:     {s.nunique():,}")
        if len(s.mode()) > 0:
            mode_val = s.mode().iloc[0]
            print(f"  Mode:       {mode_val}")
            print(f"  Mode freq:  {(s == mode_val).sum():,} ({(s == mode_val).sum() / len(s) * 100:.2f}%)")
        vc = s.value_counts()
        limit = 15 if len(vc) <= 15 else 10
        label = "Distribution:" if len(vc) <= 15 else "Top 10 values:"
        print(f"  {label}")
        for val, cnt in vc.head(limit).items():
            bar = "█" * int(cnt / len(s) * 40)
            print(f"    {str(val):<30} {cnt:>10,} ({cnt / len(s) * 100:>6.2f}%) {bar}")
        if len(vc) > 15:
            print(f"    ... and {len(vc) - 10:,} more unique values")


def print_date_stats(df: pd.DataFrame, date_cols: list):
    print("\n" + "=" * 80)
    print("5. DATE FEATURES")
    print("=" * 80)
    for col in date_cols:
        if col not in df.columns:
            continue
        s = pd.to_datetime(df[col], format="mixed", errors="coerce")
        print(f"\n--- {col} ---")
        print(f"  Count:    {s.notna().sum():,}")
        print(f"  Missing:  {s.isna().sum():,} ({s.isna().sum() / len(df) * 100:.2f}%)")
        print(f"  Min:      {s.min()}")
        print(f"  Max:      {s.max()}")
        print(f"  Range:    {(s.max() - s.min()).days:,} days")
        if col == "issue_d":
            years = s.dt.year.value_counts().sort_index()
            print("  By year:")
            for yr, cnt in years.items():
                bar = "█" * int(cnt / len(s) * 40)
                print(f"    {int(yr)}: {cnt:>10,} ({cnt / len(s) * 100:>5.1f}%) {bar}")


# ── PDF Report ──────────────────────────────────────────────────────────────

COLOR_HIST = "#3b6ea5"
COLOR_BOX = "#3b6ea5"
COLOR_BOX_MEDIAN = "#c0392b"
COLOR_BAR = "#3b6ea5"
COLOR_BAR_ALT = "#e8f0f8"


def _add_title_page(pdf: PdfPages, df: pd.DataFrame, numeric_cols, categorical_cols, date_cols):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.85, "Feature Analysis Report", fontsize=28, fontweight="bold",
            ha="center", va="top", color="#1a1d23")
    ax.text(0.5, 0.78, "Credit Risk Analysis — Lending Club Dataset",
            fontsize=14, ha="center", va="top", color="#5a5e68")
    ax.text(0.5, 0.72, "After filter_terminal_loans(), before handle_missing_values()",
            fontsize=11, ha="center", va="top", color="#8a7e72", style="italic")

    summary = (
        f"Total rows: {len(df):,}\n"
        f"Total features: {len(df.columns)}\n"
        f"  Numeric: {len(numeric_cols)}\n"
        f"  Categorical: {len(categorical_cols)}\n"
        f"  Date: {len(date_cols)}\n\n"
        f"Default rate: {df['default_flag'].mean():.2%}\n"
        f"Fully duplicate rows: {df.duplicated().sum():,}\n\n"
        f"Features with missing values: "
        f"{sum(1 for c in df.columns if df[c].isna().sum() > 0)}/{len(df.columns)}"
    )
    ax.text(0.5, 0.55, summary, fontsize=12, ha="center", va="top",
            color="#1a1d23", family="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f6f7f9", edgecolor="#d8dbe2"))
    pdf.savefig(fig)
    plt.close(fig)


def _add_missing_values_page(pdf: PdfPages, df: pd.DataFrame):
    missing = []
    for col in df.columns:
        n = df[col].isna().sum()
        if n > 0:
            missing.append((col, n, n / len(df) * 100))
    missing.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    if missing:
        names = [m[0] for m in missing]
        pcts = [m[2] for m in missing]
        y = range(len(names))
        ax.barh(y, pcts, color=COLOR_HIST, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Missing %", fontsize=11)
        for i, (name, count, pct) in enumerate(missing):
            ax.text(pct + 0.1, i, f"{count:,} ({pct:.2f}%)", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        ax.axis("off")
    ax.set_title("Missing Values Summary", fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_numeric_page(pdf: PdfPages, df: pd.DataFrame, col: str):
    """One page per numeric feature: histogram + box plot + stats table."""
    s = df[col].dropna()
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(col, fontsize=20, fontweight="bold", y=0.97, color="#1a1d23")

    # Histogram (top-left)
    ax1 = fig.add_axes([0.08, 0.52, 0.55, 0.38])
    q01, q99 = s.quantile(0.01), s.quantile(0.99)
    s_clipped = s[(s >= q01) & (s <= q99)]
    ax1.hist(s_clipped, bins=50, color=COLOR_HIST, edgecolor="white", linewidth=0.3, alpha=0.85)
    ax1.set_title("Distribution (1st–99th percentile)", fontsize=11, pad=8)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.1f}" if abs(x) < 100 else f"{x:,.0f}"))

    # Box plot (bottom-left)
    ax2 = fig.add_axes([0.08, 0.08, 0.55, 0.32])
    bp = ax2.boxplot(s_clipped, vert=False, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=COLOR_BAR_ALT, edgecolor=COLOR_BOX, linewidth=1.2),
                     medianprops=dict(color=COLOR_BOX_MEDIAN, linewidth=2),
                     whiskerprops=dict(color=COLOR_BOX, linewidth=1),
                     capprops=dict(color=COLOR_BOX, linewidth=1),
                     flierprops=dict(marker="o", markerfacecolor=COLOR_BOX, markersize=2, alpha=0.4))
    ax2.set_title("Box Plot (1st–99th percentile)", fontsize=11, pad=8)
    ax2.set_yticklabels([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.1f}" if abs(x) < 100 else f"{x:,.0f}"))

    # Stats table (right side)
    ax3 = fig.add_axes([0.68, 0.08, 0.30, 0.82])
    ax3.axis("off")

    lower = s.mean() - 3 * s.std()
    upper = s.mean() + 3 * s.std()
    n_out = ((s < lower) | (s > upper)).sum()
    n_zeros = (s == 0).sum()

    stats = [
        ("Count", f"{len(s):,}"),
        ("Missing", f"{df[col].isna().sum():,} ({df[col].isna().sum() / len(df) * 100:.2f}%)"),
        ("", ""),
        ("Mean", f"{s.mean():,.4f}"),
        ("Std", f"{s.std():,.4f}"),
        ("Min", f"{s.min():,.4f}"),
        ("25%", f"{s.quantile(0.25):,.4f}"),
        ("Median", f"{s.median():,.4f}"),
        ("75%", f"{s.quantile(0.75):,.4f}"),
        ("Max", f"{s.max():,.4f}"),
        ("", ""),
        ("Skewness", f"{s.skew():,.4f}"),
        ("Kurtosis", f"{s.kurtosis():,.4f}"),
        ("Outliers (>3σ)", f"{n_out:,} ({n_out / len(s) * 100:.2f}%)"),
    ]
    if n_zeros > 0:
        stats.append(("Zeros", f"{n_zeros:,} ({n_zeros / len(s) * 100:.2f}%)"))

    y_pos = 0.95
    for label, value in stats:
        if label == "" and value == "":
            y_pos -= 0.02
            continue
        ax3.text(0.0, y_pos, label, fontsize=9, fontweight="bold", color="#5a5e68",
                 transform=ax3.transAxes)
        ax3.text(1.0, y_pos, value, fontsize=9, ha="right", color="#1a1d23",
                 family="monospace", transform=ax3.transAxes)
        y_pos -= 0.045

    pdf.savefig(fig)
    plt.close(fig)


def _add_categorical_page(pdf: PdfPages, df: pd.DataFrame, col: str):
    """One page per categorical feature: bar chart + stats."""
    s = df[col].dropna()
    vc = s.value_counts()

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(col, fontsize=20, fontweight="bold", y=0.97, color="#1a1d23")

    show_top = min(20, len(vc))
    vc_top = vc.head(show_top)

    # Bar chart
    ax1 = fig.add_axes([0.08, 0.15, 0.55, 0.72])
    y = range(len(vc_top))
    ax1.barh(y, vc_top.values, color=COLOR_HIST, height=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels([str(v)[:30] for v in vc_top.index], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Count", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for i, (val, cnt) in enumerate(vc_top.items()):
        ax1.text(cnt + len(s) * 0.005, i, f"{cnt / len(s) * 100:.1f}%",
                 va="center", fontsize=8, color="#5a5e68")
    if len(vc) > show_top:
        ax1.set_title(f"Top {show_top} of {len(vc):,} values", fontsize=11, pad=8)
    else:
        ax1.set_title("Value Distribution", fontsize=11, pad=8)

    # Stats (right side)
    ax2 = fig.add_axes([0.68, 0.15, 0.30, 0.72])
    ax2.axis("off")

    mode_val = s.mode().iloc[0] if len(s.mode()) > 0 else "N/A"
    mode_freq = (s == mode_val).sum() if len(s.mode()) > 0 else 0

    stats = [
        ("Count", f"{len(s):,}"),
        ("Missing", f"{df[col].isna().sum():,} ({df[col].isna().sum() / len(df) * 100:.2f}%)"),
        ("Unique", f"{s.nunique():,}"),
        ("", ""),
        ("Mode", str(mode_val)[:25]),
        ("Mode freq", f"{mode_freq:,}"),
        ("Mode %", f"{mode_freq / len(s) * 100:.2f}%"),
    ]

    y_pos = 0.95
    for label, value in stats:
        if label == "" and value == "":
            y_pos -= 0.03
            continue
        ax2.text(0.0, y_pos, label, fontsize=10, fontweight="bold", color="#5a5e68",
                 transform=ax2.transAxes)
        ax2.text(1.0, y_pos, value, fontsize=10, ha="right", color="#1a1d23",
                 family="monospace", transform=ax2.transAxes)
        y_pos -= 0.06

    pdf.savefig(fig)
    plt.close(fig)


def _add_date_page(pdf: PdfPages, df: pd.DataFrame, col: str):
    """One page per date feature: yearly distribution bar chart."""
    s = pd.to_datetime(df[col], format="mixed", errors="coerce")

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(col, fontsize=20, fontweight="bold", y=0.97, color="#1a1d23")

    ax1 = fig.add_axes([0.08, 0.15, 0.55, 0.72])
    years = s.dt.year.dropna().astype(int)
    vc = years.value_counts().sort_index()

    if col == "issue_d":
        ax1.bar(vc.index, vc.values, color=COLOR_HIST, width=0.7)
        ax1.set_xlabel("Year", fontsize=10)
        ax1.set_ylabel("Count", fontsize=10)
        ax1.set_title("Loans by Issue Year", fontsize=11, pad=8)
    else:
        decade_bins = list(range(int(years.min() // 10 * 10), int(years.max()) + 10, 10))
        ax1.hist(years, bins=decade_bins, color=COLOR_HIST, edgecolor="white", linewidth=0.5)
        ax1.set_xlabel("Year", fontsize=10)
        ax1.set_ylabel("Count", fontsize=10)
        ax1.set_title("Distribution by Decade", fontsize=11, pad=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Stats
    ax2 = fig.add_axes([0.68, 0.15, 0.30, 0.72])
    ax2.axis("off")
    stats = [
        ("Count", f"{s.notna().sum():,}"),
        ("Missing", f"{s.isna().sum():,}"),
        ("Min", str(s.min())[:10]),
        ("Max", str(s.max())[:10]),
        ("Range", f"{(s.max() - s.min()).days:,} days"),
    ]
    y_pos = 0.95
    for label, value in stats:
        ax2.text(0.0, y_pos, label, fontsize=10, fontweight="bold", color="#5a5e68",
                 transform=ax2.transAxes)
        ax2.text(1.0, y_pos, value, fontsize=10, ha="right", color="#1a1d23",
                 family="monospace", transform=ax2.transAxes)
        y_pos -= 0.06

    pdf.savefig(fig)
    plt.close(fig)


def _add_findings_page(pdf: PdfPages):
    """Summary page of key findings and red flags."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.95, "Key Findings & Red Flags", fontsize=22, fontweight="bold",
            ha="center", va="top", color="#1a1d23")

    findings = [
        "MISSING VALUES",
        "  • emp_title (6.38%) and emp_length (5.84%) — missing = likely unemployed/self-employed",
        "  • mort_acc (3.51%) — missing = no mortgage history, median impute is appropriate",
        "  • revol_util, pub_rec_bankruptcies, dti — negligible missing (<0.1%)",
        "  • No columns exceed 50% missing — none will be auto-dropped",
        "",
        "DATA QUALITY ISSUES",
        "  • dti: min = -1.0 (negative DTI is impossible), max = 999 (placeholder/error)",
        "  • revol_util: max = 892.3% (utilisation over 100% is a data error)",
        "  • annual_inc: 361 rows have $0 income — suspicious for approved loans",
        "  • last_fico_range_low: 35,878 rows have value 0 (valid FICO range is 300–850)",
        "  • last_fico_range_high: 209 rows have value 0",
        "  • pub_rec: max = 86 derogatory records — extreme but technically possible",
        "",
        "DISTRIBUTION CHARACTERISTICS",
        "  • annual_inc: extreme right skew (46.3) — median $65K vs mean $76K",
        "  • recoveries: 86.3% zeros — only defaulted loans have recovery amounts",
        "  • total_rec_late_fee: 95.6% zeros — most loans never incur late fees",
        "  • pub_rec: 83.1% zeros — most borrowers have clean records",
        "  • pub_rec_bankruptcies: 87.5% zeros",
        "",
        "CLASS & CATEGORY BALANCE",
        "  • default_flag: 80% / 20% — moderate imbalance, handled by scale_pos_weight",
        "  • term: 75.9% are 36-month — 60-month is the minority",
        "  • application_type: 98.1% Individual — Joint App is effectively negligible",
        "  • purpose: debt_consolidation dominates at 58%",
        "  • emp_title: 378K unique values — too many categories for direct encoding",
    ]

    y = 0.88
    for line in findings:
        if line == "":
            y -= 0.015
            continue
        is_header = not line.startswith("  ")
        ax.text(0.05, y, line, fontsize=9.5 if not is_header else 11,
                fontweight="bold" if is_header else "normal",
                color="#3b6ea5" if is_header else "#1a1d23",
                family="monospace" if line.startswith("  ") else "sans-serif",
                transform=ax.transAxes)
        y -= 0.03

    pdf.savefig(fig)
    plt.close(fig)


def _render_text_page(pdf: PdfPages, title: str, lines: list, font_size: float = 9.0):
    """Render a text-heavy page. Lines are (text, style) tuples or plain strings."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.96, title, fontsize=20, fontweight="bold",
            ha="center", va="top", color="#1a1d23")

    y = 0.90
    for line in lines:
        if isinstance(line, tuple):
            text, style = line
        else:
            text, style = line, "body"

        if text == "":
            y -= 0.012
            continue

        if style == "h2":
            y -= 0.008
            ax.text(0.04, y, text, fontsize=12, fontweight="bold",
                    color="#3b6ea5", transform=ax.transAxes)
            y -= 0.032
        elif style == "h3":
            ax.text(0.04, y, text, fontsize=10, fontweight="bold",
                    color="#1a1d23", transform=ax.transAxes)
            y -= 0.028
        elif style == "table_header":
            ax.text(0.06, y, text, fontsize=8.5, fontweight="bold",
                    color="#5a5e68", family="monospace", transform=ax.transAxes)
            y -= 0.022
        elif style == "table_row":
            ax.text(0.06, y, text, fontsize=8, color="#1a1d23",
                    family="monospace", transform=ax.transAxes)
            y -= 0.021
        elif style == "bullet":
            ax.text(0.06, y, text, fontsize=font_size, color="#1a1d23",
                    transform=ax.transAxes)
            y -= 0.025
        else:
            ax.text(0.06, y, text, fontsize=font_size, color="#1a1d23",
                    transform=ax.transAxes)
            y -= 0.025

        if y < 0.03:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.96, f"{title} (cont.)", fontsize=20, fontweight="bold",
                    ha="center", va="top", color="#1a1d23")
            y = 0.90

    pdf.savefig(fig)
    plt.close(fig)


def _add_column_selection_pages(pdf: PdfPages):
    """Why these 38 columns were chosen and why 113 were dropped."""

    # Page 1: Kept columns — origination features
    lines_p1 = [
        ("CATEGORY A: LOAN & BORROWER FEATURES AT ORIGINATION (29 columns)", "h2"),
        ("These are known at the time the borrower applies — usable for PD prediction.", "body"),
        ("", ""),
        ("Col                         Reason for keeping", "table_header"),
        ("─" * 75, "table_row"),
        ("loan_amnt                   Loan amount requested — directly impacts exposure", "table_row"),
        ("funded_amnt                 What Lending Club actually funded — can differ from requested", "table_row"),
        ("term                        36 or 60 months — strongest default predictor (60m defaults 2x)", "table_row"),
        ("int_rate                    Interest rate — encodes LC's own risk assessment", "table_row"),
        ("installment                 Monthly payment — affects borrower's ability to repay", "table_row"),
        ("grade                       LC's letter grade (A–G) — internal risk tier", "table_row"),
        ("sub_grade                   Finer grade (A1–G5) — more granular risk tier", "table_row"),
        ("emp_title                   Job title — kept for EDA, not model feature (378K unique)", "table_row"),
        ("emp_length                  Years employed — proxy for income stability", "table_row"),
        ("home_ownership              RENT/OWN/MORTGAGE — financial stability indicator", "table_row"),
        ("annual_inc                  Borrower income — core ability-to-pay indicator", "table_row"),
        ("verification_status         Whether income was verified — unverified = riskier", "table_row"),
        ("issue_d                     Loan issue date — needed for time-based train/test split", "table_row"),
        ("loan_status                 Fully Paid / Charged Off / Default — our TARGET variable", "table_row"),
        ("purpose                     Why borrower wants the loan (debt consolidation, medical...)", "table_row"),
        ("title                       Borrower-written loan title — kept for EDA, not model feature", "table_row"),
        ("dti                         Debt-to-income ratio — key credit risk metric", "table_row"),
        ("earliest_cr_line            Date of first credit line → derives credit_history_years", "table_row"),
        ("open_acc                    Number of open credit accounts", "table_row"),
        ("pub_rec                     Number of derogatory public records", "table_row"),
        ("revol_bal                   Total revolving credit balance", "table_row"),
        ("revol_util                  Revolving utilisation rate — high = higher risk", "table_row"),
        ("total_acc                   Total number of credit lines ever", "table_row"),
        ("initial_list_status         Whole (w) vs fractional (f) listing", "table_row"),
        ("application_type            Individual vs Joint application", "table_row"),
        ("mort_acc                    Number of mortgage accounts — proxy for asset ownership", "table_row"),
        ("pub_rec_bankruptcies        Bankruptcy count — severe credit event", "table_row"),
        ("fico_range_low              FICO score lower bound", "table_row"),
        ("fico_range_high             FICO score upper bound — averaged to get fico_avg", "table_row"),
    ]
    _render_text_page(pdf, "Why These 38 Columns? (1/4)", lines_p1, font_size=8.5)

    # Page 2: Kept columns — post-origination (for LGD)
    lines_p2 = [
        ("CATEGORY B: POST-ORIGINATION COLUMNS — KEPT FOR LGD MODEL (9 columns)", "h2"),
        ("Known only after the loan plays out. Cannot be used for PD (that would be leakage).", "body"),
        ("Essential for the LGD model which needs recovery data after default.", "body"),
        ("These are explicitly excluded from PD features via get_model_features().", "body"),
        ("", ""),
        ("Col                         Reason for keeping", "table_header"),
        ("─" * 75, "table_row"),
        ("last_pymnt_amnt             Last payment amount — useful for LGD context", "table_row"),
        ("total_pymnt                 Total payments received — recovery calculations", "table_row"),
        ("total_rec_prncp             Principal received — component of total recovery", "table_row"),
        ("total_rec_int               Interest received", "table_row"),
        ("total_rec_late_fee          Late fees received", "table_row"),
        ("recoveries                  Post-charge-off collections — THE LGD TARGET", "table_row"),
        ("                            LGD = 1 − (recoveries / loan_amnt)", "table_row"),
        ("collection_recovery_fee     Fees paid for recovery — net recovery calculation", "table_row"),
        ("last_fico_range_high        Most recent FICO — shows credit deterioration", "table_row"),
        ("last_fico_range_low         Most recent FICO lower bound", "table_row"),
    ]
    _render_text_page(pdf, "Why These 38 Columns? (2/4)", lines_p2)

    # Page 3: Dropped columns — categories C through F
    lines_p3 = [
        ("DROPPED — THE REMAINING 113 COLUMNS AND WHY", "h2"),
        ("", ""),
        ("Category C: Identifiers & Metadata (5 cols) — useless for modelling", "h3"),
        ("  id, member_id              Internal IDs — no predictive value", "table_row"),
        ("  url                        LC URL for listing — just a link", "table_row"),
        ("  pymnt_plan                 Nearly all 'n' — zero variance", "table_row"),
        ("  policy_code                Always 1 in public data — zero information", "table_row"),
        ("", ""),
        ("Category D: Geographic (2 cols) — privacy risk, low signal", "h3"),
        ("  zip_code                   900+ categories, sparse, can introduce geographic bias", "table_row"),
        ("  addr_state                 50 categories, low signal after controlling for income/FICO", "table_row"),
        ("", ""),
        ("Category E: Redundant Credit Bureau Features (47 cols)", "h3"),
        ("  Highly correlated with what we already kept (open_acc, pub_rec, revol_util, FICO).", "body"),
        ("  delinq_2yrs, acc_now_delinq, delinq_amnt     — correlated with pub_rec + FICO", "table_row"),
        ("  inq_last_6mths, inq_fi, inq_last_12m (4)     — inquiries, correlated with each other", "table_row"),
        ("  mths_since_last_delinq/record/derog (3)       — >50% missing, high-missingness problem", "table_row"),
        ("  mths_since_recent_bc/bc_dlq/revol_delinq (4)  — same high-missingness issue", "table_row"),
        ("  collections_12m, chargeoff_12m, tax_liens (3)  — near-zero variance (vast majority = 0)", "table_row"),
        ("  tot_coll_amt, tot_cur_bal, avg_cur_bal (3)     — redundant with revol_bal + total_acc", "table_row"),
        ("  total_rev_hi_lim, tot_hi_cred_lim (5)          — credit limits, correlated with income", "table_row"),
        ("  bc_open_to_buy, bc_util, all_util (6)          — granular util breakdowns, revol_util covers", "table_row"),
        ("  open_acc_6m through acc_open_past_24m (8)       — overlapping windows, correlated w/ open_acc", "table_row"),
        ("  num_accts_ever_120_pd, num_tl_30dpd (4)        — delinquency counts, correlated w/ pub_rec", "table_row"),
        ("  num_actv_bc_tl through num_sats (9)            — granular trade-line counts, open/total_acc cover", "table_row"),
        ("  mo_sin_old_il/rev_tl (4)                       — months since oldest acct, credit_history covers", "table_row"),
        ("  pct_tl_nvr_dlq, total_bal_il                   — inverse of delinquency / niche balance", "table_row"),
        ("", ""),
        ("Category F: Joint Application Fields (10 cols) — 98% missing", "h3"),
        ("  annual_inc_joint, dti_joint, verification_status_joint", "table_row"),
        ("  revol_bal_joint, sec_app_* (10 columns total)  — only 1.9% are Joint Apps", "table_row"),
        ("  98% null — imputation would fabricate data for nearly every row", "body"),
    ]
    _render_text_page(pdf, "Why These 38 Columns? (3/4)", lines_p3, font_size=8.5)

    # Page 4: Dropped — categories G and H
    lines_p4 = [
        ("Category G: Hardship & Settlement Fields (21 cols)", "h3"),
        ("  Post-default administrative data — cannot be used for prediction.", "body"),
        ("  hardship_flag through hardship_last_payment_amount (14 cols)", "table_row"),
        ("    → Describes hardship programs entered AFTER borrower struggled", "table_row"),
        ("  debt_settlement_flag through settlement_term (6 cols)", "table_row"),
        ("    → Settlement details, only populated after charge-off", "table_row"),
        ("  disbursement_method", "table_row"),
        ("    → How funds were sent (Cash vs DirectPay) — no risk signal", "table_row"),
        ("", ""),
        ("Category H: Investor/Payment Tracking (5 cols)", "h3"),
        ("  Operational accounting, not predictive.", "body"),
        ("  funded_amnt_inv             Nearly identical to funded_amnt", "table_row"),
        ("  out_prncp, out_prncp_inv    Outstanding principal — post-origination", "table_row"),
        ("  total_pymnt_inv             Investor's share of payments", "table_row"),
        ("  next_pymnt_d, last_pymnt_d, last_credit_pull_d — post-origination dates", "table_row"),
        ("", ""),
        ("SUMMARY", "h2"),
        ("  Category A (origination):    29 cols → KEPT (PD model features)", "bullet"),
        ("  Category B (post-origination): 9 cols → KEPT (LGD model + EL only)", "bullet"),
        ("  Category C (identifiers):     5 cols → DROPPED (no predictive value)", "bullet"),
        ("  Category D (geographic):      2 cols → DROPPED (sparse, bias risk)", "bullet"),
        ("  Category E (redundant bureau):47 cols → DROPPED (correlated with kept features)", "bullet"),
        ("  Category F (joint app):      10 cols → DROPPED (98% null)", "bullet"),
        ("  Category G (hardship):       21 cols → DROPPED (post-outcome, can't predict with)", "bullet"),
        ("  Category H (investor ops):    5 cols → DROPPED (operational, not predictive)", "bullet"),
        ("", ""),
        ("  Total kept:    38 columns", "bullet"),
        ("  Total dropped: 113 columns", "bullet"),
    ]
    _render_text_page(pdf, "Why These 38 Columns? (4/4)", lines_p4)


def _add_cache_explanation_page(pdf: PdfPages):
    """Explain what a cache directory is and why we copy to data/raw/."""
    lines = [
        ("WHAT IS A CACHE DIRECTORY?", "h2"),
        ("When you run kagglehub.dataset_download('wordsforthewise/lending-club'),", "body"),
        ("kagglehub downloads the dataset to its own cache directory — a hidden folder", "body"),
        ("that kagglehub manages internally, typically at:", "body"),
        ("", ""),
        ("  ~/.cache/kagglehub/datasets/wordsforthewise/lending-club/versions/3/", "table_row"),
        ("", ""),
        ("WHY NOT READ DIRECTLY FROM THE CACHE?", "h2"),
        ("", ""),
        ("Problem 1: kagglehub can delete it", "h3"),
        ("  If you run kagglehub.cache.clear() or if kagglehub updates its cache", "bullet"),
        ("  management, that directory is wiped. Your pipeline breaks.", "bullet"),
        ("", ""),
        ("Problem 2: The path changes across machines", "h3"),
        ("  On your Mac it's ~/.cache/kagglehub/..., on Linux it might be elsewhere,", "bullet"),
        ("  on a teammate's machine it's under their home directory. If you hardcode", "bullet"),
        ("  the cache path, it only works on your machine.", "bullet"),
        ("", ""),
        ("Problem 3: Version changes move the files", "h3"),
        ("  If kagglehub downloads version 4 of the dataset, the path becomes", "bullet"),
        ("  .../versions/4/ and the old path stops existing.", "bullet"),
        ("", ""),
        ("THE SOLUTION: COPY TO data/raw/", "h2"),
        ("By copying to a project-local data/raw/ directory, the file lives in a", "body"),
        ("stable, predictable location that:", "body"),
        ("", ""),
        ("  • Doesn't change when kagglehub updates", "bullet"),
        ("  • Is the same relative path on every machine", "bullet"),
        ("  • Survives cache clears", "bullet"),
        ("  • Is tracked by .gitignore (not committed, but developers know to look here)", "bullet"),
        ("", ""),
        ("This is what 'reproducible regardless of cache state' means — your pipeline", "body"),
        ("works whether the cache exists, was cleared, or points to a different version.", "body"),
    ]
    _render_text_page(pdf, "Cache Directory Explained", lines)


def _add_module1_files_pages(pdf: PdfPages):
    """Explain each file in Module 1 and the reasoning behind method choices."""

    lines_p1 = [
        ("config/settings.py — Central Configuration", "h2"),
        ("Single source of truth — every script imports from here instead of hardcoding.", "body"),
        ("If you move the project or rename a directory, you change one file, not twenty.", "body"),
        ("", ""),
        ("Key design decisions:", "h3"),
        ("  • SELECTED_COLUMNS (38 from 151): Only features available at loan origination", "bullet"),
        ("    plus post-origination columns needed for LGD. Avoids data leakage.", "bullet"),
        ("  • TERMINAL_STATUSES: Only 'Fully Paid', 'Charged Off', 'Default'. Excludes", "bullet"),
        ("    'Current' and 'In Grace Period' — unknown outcomes would need survival analysis.", "bullet"),
        ("  • TRAIN_END_YEAR=2016, TEST_START_YEAR=2017: Time-based split config.", "bullet"),
        ("    In production, you always train on past data and predict future.", "bullet"),
        ("    Random split would let the model 'see' 2018 patterns during training.", "bullet"),
        ("  • RANDOM_SEED=42: Reproducibility across all random operations.", "bullet"),
        ("", ""),
        ("src/data/download.py — Data Download", "h2"),
        ("Downloads Lending Club data from Kaggle, copies CSVs to data/raw/.", "body"),
        ("", ""),
        ("Key design decisions:", "h3"),
        ("  • kagglehub over kaggle CLI: Newer Python API, handles auth/caching/versioning", "bullet"),
        ("    natively without shell commands or manual zip extraction.", "bullet"),
        ("  • Copies to data/raw/ instead of reading from cache: Makes pipeline reproducible", "bullet"),
        ("    regardless of cache state (see 'Cache Directory Explained' section).", "bullet"),
        ("  • Looks for .csv.gz first, then .csv: Dataset ships compressed. pandas reads", "bullet"),
        ("    .csv.gz directly, avoiding unnecessary decompression step.", "bullet"),
        ("  • Idempotent: 'if not dest.exists()' means re-running doesn't re-download.", "bullet"),
    ]
    _render_text_page(pdf, "Module 1 Files Explained (1/2)", lines_p1, font_size=8.5)

    lines_p2 = [
        ("src/data/clean.py — Cleaning Pipeline", "h2"),
        ("Full pipeline in 7 steps, mirrors the SQL logic for local development.", "body"),
        ("", ""),
        ("Step-by-step reasoning:", "h3"),
        ("", ""),
        ("  1. load_raw_data(): reads CSV with low_memory=False because Lending Club file", "bullet"),
        ("     has mixed types (e.g. term = ' 36 months' as string). Without this flag,", "bullet"),
        ("     pandas infers types per chunk and may get it wrong.", "bullet"),
        ("", ""),
        ("  2. select_columns(): reduces 151 → 38 immediately. Working with 151 columns", "bullet"),
        ("     on 2.2M rows wastes memory and slows every operation.", "bullet"),
        ("", ""),
        ("  3. filter_terminal_loans(): drops ~40% of data ('Current', 'In Grace Period').", "bullet"),
        ("     These loans haven't finished — labelling them 'not default' would be wrong.", "bullet"),
        ("", ""),
        ("  4. create_target(): binary default_flag (1/0). 'Charged Off' and 'Default' are", "bullet"),
        ("     both losses — the distinction is operational, not analytically meaningful.", "bullet"),
        ("", ""),
        ("  5. parse_dates(): extracts issue_year (for time split) and issue_month (for", "bullet"),
        ("     seasonality). format='mixed' handles inconsistent date formats.", "bullet"),
        ("", ""),
        ("  6. clean_numeric_columns(): parses term (' 36 months' → 36), int_rate ('13.5%'", "bullet"),
        ("     → 13.5), revol_util (strip %), emp_length ('10+ years' → 10). emp_length", "bullet"),
        ("     is mapped to integers to preserve ordinality (more experience = ordered).", "bullet"),
        ("", ""),
        ("  7. handle_missing_values(): median for numerics (robust to outliers like $10.9M", "bullet"),
        ("     income), mode for categoricals. >50% missing → drop column entirely.", "bullet"),
        ("", ""),
        ("  Output: Parquet (not CSV) — preserves dtypes, ~5x smaller, ~10x faster to read.", "bullet"),
        ("", ""),
        ("src/data/load_bq.py — BigQuery Upload (Optional)", "h2"),
        ("  • Portfolio projects need a SQL story — most DS/DA roles use SQL daily.", "body"),
        ("  • WRITE_TRUNCATE: replaces table on each run, making upload idempotent.", "body"),
        ("  • autodetect=True: lets BigQuery infer schema — no boilerplate for one-time upload.", "body"),
        ("  • Creates dataset if missing (try/except) — works on fresh GCP projects.", "body"),
        ("  • Optional: rest of pipeline reads from Parquet. This script exists purely to", "body"),
        ("    demonstrate the SQL workflow. Without GCP credentials, everything else works.", "body"),
    ]
    _render_text_page(pdf, "Module 1 Files Explained (2/2)", lines_p2, font_size=8.5)


def _add_missing_values_design_pages(pdf: PdfPages):
    """Design decisions for missing value imputation — per feature."""

    lines_p1 = [
        ("OUR APPROACH: MEDIAN (NUMERIC) / MODE (CATEGORICAL)", "h2"),
        ("handle_missing_values() in clean.py applies a simple, deliberate strategy:", "body"),
        ("  • Numeric columns → fill with median", "bullet"),
        ("  • Categorical columns → fill with mode (most frequent value)", "bullet"),
        ("  • Any column >50% missing → drop entirely", "bullet"),
        ("", ""),
        ("WHY NOT KNN IMPUTATION?", "h2"),
        ("KNN imputer replaces each missing value with the mean of its K nearest", "body"),
        ("neighbours in feature space. Rejected for three reasons:", "body"),
        ("", ""),
        ("  1. Computational cost: KNN imputation is O(n² × d). On 1.3M rows × 38 features,", "bullet"),
        ("     that's ~65 billion distance calculations. Runs for hours on a single machine.", "bullet"),
        ("  2. Curse of dimensionality: With 25+ numeric features, Euclidean distance", "bullet"),
        ("     becomes unreliable — all points appear equidistant in high dimensions.", "bullet"),
        ("     KNN imputation works well with <10 features, not 25+.", "bullet"),
        ("  3. Marginal benefit: Our highest-missing numeric column is mort_acc at 3.51%.", "bullet"),
        ("     With <4% missing, KNN and median produce nearly identical results — the", "bullet"),
        ("     98% of present data dominates model training regardless.", "bullet"),
        ("", ""),
        ("WHY NOT MICE (Multiple Imputation by Chained Equations)?", "h2"),
        ("MICE builds a regression model for each feature with missing values,", "body"),
        ("iterating multiple times to stabilise. Rejected because:", "body"),
        ("", ""),
        ("  1. Overkill for low missingness: MICE shines when 15–40% is missing and", "bullet"),
        ("     missingness depends on other features (MAR pattern). Our max is 6.38%.", "bullet"),
        ("  2. Multiple imputations needed: Proper MICE creates M datasets (typically 5–10),", "bullet"),
        ("     trains M models, pools results. That multiplies training time by M.", "bullet"),
        ("  3. Assumption of linearity: Default MICE uses linear regression for numeric", "bullet"),
        ("     and logistic for categorical. Our data has extreme skew (annual_inc", "bullet"),
        ("     skewness=46.3) — linear assumptions don't hold without transforms.", "bullet"),
        ("  4. Information leak risk: MICE uses target-adjacent features to impute.", "bullet"),
        ("     With post-origination columns in the dataset, MICE could implicitly", "bullet"),
        ("     leak outcome information into PD features.", "bullet"),
    ]
    _render_text_page(pdf, "Missing Value Design Decisions (1/3)", lines_p1, font_size=8.5)

    lines_p2 = [
        ("WHY NOT ITERATIVE IMPUTER (sklearn)?", "h2"),
        ("sklearn's IterativeImputer is a MICE variant with a single imputation.", "body"),
        ("Same problems: linear assumptions, computational cost, leak risk.", "body"),
        ("And without the multiple-imputation pooling, it loses MICE's main advantage.", "body"),
        ("", ""),
        ("WHY NOT INDICATOR / MISSING-AS-CATEGORY?", "h2"),
        ("Adding a binary 'is_missing' indicator column is sometimes useful when", "body"),
        ("missingness itself is informative (MNAR — Missing Not At Random).", "body"),
        ("", ""),
        ("  • emp_title missing = unemployed/self-employed → missingness IS informative,", "bullet"),
        ("    but emp_title is excluded from model features (378K categories).", "bullet"),
        ("  • emp_length missing = same population → already captured by emp_length_num", "bullet"),
        ("    which maps missing to NaN then gets median-filled.", "bullet"),
        ("  • For mort_acc, revol_util, dti: missingness is <4% and appears random", "bullet"),
        ("    (MCAR) — an indicator column would be 96%+ zeros, near-zero variance.", "bullet"),
        ("", ""),
        ("WHY MEDIAN OVER MEAN? (for numeric)", "h2"),
        ("  • annual_inc: mean = $76K, median = $65K. The mean is pulled up by", "bullet"),
        ("    extreme outliers ($10.9M max). Filling with $76K would overstate income", "bullet"),
        ("    for ~75% of borrowers. Median is robust to the right tail.", "bullet"),
        ("  • revol_bal: mean = $16K, median = $11K. Same skew problem (max $2.9M).", "bullet"),
        ("  • pub_rec, pub_rec_bankruptcies: 83–87% zeros. Mean = 0.21, median = 0.", "bullet"),
        ("    Median correctly fills with the dominant value (zero records).", "bullet"),
        ("", ""),
        ("WHY MODE FOR CATEGORICALS?", "h2"),
        ("  • Mode = most frequent category. With low missingness (<7%), filling with", "bullet"),
        ("    the dominant class barely shifts the distribution.", "bullet"),
        ("  • Alternative: 'Unknown' category. We avoided this because XGBoost would", "bullet"),
        ("    treat 'Unknown' as its own split point, potentially overfitting to the", "bullet"),
        ("    small missing-value population rather than learning from the majority.", "bullet"),
    ]
    _render_text_page(pdf, "Missing Value Design Decisions (2/3)", lines_p2, font_size=8.5)

    lines_p3 = [
        ("PER-FEATURE IMPUTATION DECISIONS", "h2"),
        ("", ""),
        ("Feature              Missing%  Type         Method       Rationale", "table_header"),
        ("─" * 80, "table_row"),
        ("", ""),
        ("NUMERIC FEATURES WITH MISSING VALUES:", "h3"),
        ("", ""),
        ("mort_acc              3.51%   float64      Median (1.0)  83% zero-inflated, median", "table_row"),
        ("                                                         captures 'typical borrower has", "table_row"),
        ("                                                         0–1 mortgages'. Mean (1.67)", "table_row"),
        ("                                                         overestimates for most.", "table_row"),
        ("", ""),
        ("revol_util            0.06%   float64      Median (52.2) Nearly symmetric distribution", "table_row"),
        ("                                                         (skew = −0.04). Median ≈ mean", "table_row"),
        ("                                                         here, so either would work.", "table_row"),
        ("                                                         857 rows — negligible impact.", "table_row"),
        ("", ""),
        ("pub_rec_bankruptcies  0.05%   float64      Median (0.0)  87.5% zeros. Median correctly", "table_row"),
        ("                                                         fills with 0 (most have none).", "table_row"),
        ("                                                         697 rows — negligible impact.", "table_row"),
        ("", ""),
        ("dti                   0.03%   float64      Median (17.6) Moderate skew but 374 rows.", "table_row"),
        ("                                                         Median is between 25th and", "table_row"),
        ("                                                         75th percentile — safe fill.", "table_row"),
        ("", ""),
        ("CATEGORICAL FEATURES WITH MISSING VALUES:", "h3"),
        ("", ""),
        ("emp_title             6.38%   object       Mode          NOT used in model (378K unique,", "table_row"),
        ("                                           ('Teacher')   too many categories). Filled", "table_row"),
        ("                                                         for EDA completeness only.", "table_row"),
        ("", ""),
        ("emp_length            5.84%   object       Mode          Mapped to emp_length_num (0–10)", "table_row"),
        ("                                           ('10+ yrs')   via clean_numeric_columns().", "table_row"),
        ("                                                         Missing likely = unemployed/", "table_row"),
        ("                                                         self-employed. Mode fill is", "table_row"),
        ("                                                         conservative (10+ is 35%).", "table_row"),
        ("", ""),
        ("title                 1.24%   object       Mode          NOT used in model (61K unique).", "table_row"),
        ("                                           ('Debt cons') Filled for EDA only.", "table_row"),
    ]
    _render_text_page(pdf, "Missing Value Design Decisions (3/3)", lines_p3, font_size=8.0)


def _add_outlier_design_pages(pdf: PdfPages):
    """Design decisions for outlier handling — per feature."""

    lines_p1 = [
        ("OUR APPROACH: MINIMAL INTERVENTION — LET THE MODEL HANDLE IT", "h2"),
        ("", ""),
        ("We deliberately chose NOT to remove or cap most outliers. Here's why:", "body"),
        ("", ""),
        ("XGBoost is our primary model. It is a tree-based algorithm:", "h3"),
        ("  • Trees split on RANK, not MAGNITUDE. A borrower with $10.9M income is", "bullet"),
        ("    simply placed in the 'above threshold X' bucket — the exact dollar", "bullet"),
        ("    amount doesn't matter. Whether the max is $10.9M or $500K, the split", "bullet"),
        ("    point stays the same.", "bullet"),
        ("  • No distance calculations: Unlike KNN or SVM, trees don't compute", "bullet"),
        ("    Euclidean distance. An outlier doesn't 'pull' the model toward it.", "bullet"),
        ("  • Histogram-based splits (tree_method='hist'): XGBoost bins values into", "bullet"),
        ("    ~256 buckets. Extreme outliers sit in the last bucket — they can't", "bullet"),
        ("    dominate the split search.", "bullet"),
        ("", ""),
        ("Logistic Regression uses StandardScaler:", "h3"),
        ("  • StandardScaler transforms each feature to mean=0, std=1.", "bullet"),
        ("  • This neutralises scale differences (income in 10Ks vs FICO in 100s).", "bullet"),
        ("  • Outliers still affect the mean/std computation, but with 1.3M rows,", "bullet"),
        ("    a few hundred extreme values barely move the mean or std.", "bullet"),
        ("  • class_weight='balanced' adjusts for class imbalance — the model focuses", "bullet"),
        ("    on classification boundary, not on fitting extreme income values.", "bullet"),
        ("", ""),
        ("WHY NOT WINSORISATION (CAPPING AT 1st/99th PERCENTILE)?", "h2"),
        ("  • Information loss: A $10.9M income borrower IS genuinely different from a", "bullet"),
        ("    $200K borrower. Capping both to $200K destroys real signal. In credit risk,", "bullet"),
        ("    very high-income borrowers default less — flattening this hurts the model.", "bullet"),
        ("  • Our tree model doesn't need it: Winsorisation helps linear models and", "bullet"),
        ("    distance-based models. XGBoost gains nothing from it.", "bullet"),
        ("  • Risk of masking data errors: dti=999 and revol_util=892% are likely", "bullet"),
        ("    errors, not true outliers. Capping them to 99th percentile would quietly", "bullet"),
        ("    accept bad data. Instead, dti is clipped to ≥0 in features.py (removes", "bullet"),
        ("    the impossible negative value), and the erroneous 999 is handled by", "bullet"),
        ("    XGBoost's binning — it won't get its own split.", "bullet"),
    ]
    _render_text_page(pdf, "Outlier Design Decisions (1/3)", lines_p1, font_size=8.5)

    lines_p2 = [
        ("WHY NOT Z-SCORE REMOVAL (DROP ROWS WHERE |z| > 3)?", "h2"),
        ("  • Data loss: annual_inc has 10,021 outliers beyond 3σ (0.74%). pub_rec has", "bullet"),
        ("    12,662 (0.94%). Across all features, thousands of rows would be removed.", "bullet"),
        ("  • Multivariate problem: A row outlying in income might be normal in FICO.", "bullet"),
        ("    Removing it loses a valid FICO data point. Feature-wise removal compounds.", "bullet"),
        ("  • Assumes normality: Z-score outlier detection assumes Gaussian distribution.", "bullet"),
        ("    Most of our features are heavily skewed (annual_inc skew=46, pub_rec", "bullet"),
        ("    skew=11.6). Z-score flags are meaningless on non-normal distributions.", "bullet"),
        ("", ""),
        ("WHY NOT IQR METHOD (DROP ROWS OUTSIDE Q1 − 1.5×IQR, Q3 + 1.5×IQR)?", "h2"),
        ("  • Same data loss problem as z-score, often worse. For zero-inflated features", "bullet"),
        ("    like pub_rec (Q1=Q2=Q3=0, IQR=0), any nonzero value would be flagged.", "bullet"),
        ("  • Same multivariate problem — per-feature removal is too aggressive.", "bullet"),
        ("", ""),
        ("WHY NOT ISOLATION FOREST / LOF (MULTIVARIATE OUTLIER DETECTION)?", "h2"),
        ("  • Computationally expensive on 1.3M rows with 25 numeric features.", "bullet"),
        ("  • Produces a contamination fraction — requires subjective tuning.", "bullet"),
        ("  • Removes entire rows, losing all features. In credit risk, 'unusual'", "bullet"),
        ("    borrowers (very high income + very high DTI) are exactly who the model", "bullet"),
        ("    needs to learn about — they are edge cases, not noise.", "bullet"),
        ("", ""),
        ("EXPLICIT OUTLIER HANDLING WE DID APPLY:", "h2"),
        ("", ""),
        ("These are targeted corrections for known data quality issues, not blanket rules:", "body"),
        ("", ""),
        ("  Feature                Action              Reason", "table_header"),
        ("  ─────────────────────  ──────────────────── ─────────────────────────────────", "table_row"),
        ("  dti                   clip(lower=0)        Negative DTI is impossible", "table_row"),
        ("                                              (dti min = −1.0 is a data error)", "table_row"),
        ("  credit_history_years  clip(lower=0)        Can't have negative credit history", "table_row"),
        ("                                              (would mean credit line after loan)", "table_row"),
        ("  recovery_rate (LGD)   clip(0, 1)           Recovery rate must be 0–100%", "table_row"),
        ("                                              (bounded by definition)", "table_row"),
        ("  LGD predictions       clip(0, 1)           Model output must stay in [0,1]", "table_row"),
    ]
    _render_text_page(pdf, "Outlier Design Decisions (2/3)", lines_p2, font_size=8.5)

    lines_p3 = [
        ("PER-FEATURE OUTLIER ANALYSIS", "h2"),
        ("", ""),
        ("Feature            Outliers(3σ)  Action        Why", "table_header"),
        ("─" * 80, "table_row"),
        ("loan_amnt           0 (0.0%)     None          Bounded [500, 40000] by LC policy", "table_row"),
        ("funded_amnt         0 (0.0%)     None          Same — LC caps at $40K", "table_row"),
        ("int_rate            9,765 (0.7%) None          High rates are real (Grade G)", "table_row"),
        ("installment         13,408 (1%)  None          Follows from loan_amnt × rate", "table_row"),
        ("annual_inc          10,021 (0.7%) None         High incomes are real; tree handles", "table_row"),
        ("dti                 2,536 (0.2%) clip(≥0)      Only fix: remove negative DTI", "table_row"),
        ("                                               999 is data error but rare (1 row)", "table_row"),
        ("open_acc            15,154 (1.1%) None         90 accounts is extreme but possible", "table_row"),
        ("pub_rec             12,662 (0.9%) None         86 derog records = extreme but real", "table_row"),
        ("revol_bal           17,084 (1.3%) None         $2.9M balance = real for HNW clients", "table_row"),
        ("revol_util          80 (0.01%)   None          892% is likely error; 80 rows — too", "table_row"),
        ("                                               few to affect 1.3M row training", "table_row"),
        ("total_acc           14,021 (1%)  None          176 total accounts = heavy user, real", "table_row"),
        ("mort_acc            17,794 (1.4%) None         51 mortgages = real estate investor", "table_row"),
        ("pub_rec_bankruptcies 9,964 (0.7%) None        12 bankruptcies = extreme but legal", "table_row"),
        ("fico_range_low      19,806 (1.5%) None        FICO bounded by scoring model, real", "table_row"),
        ("fico_range_high     19,806 (1.5%) None        Same", "table_row"),
        ("", ""),
        ("POST-ORIGINATION (LGD FEATURES):", "h3"),
        ("last_pymnt_amnt     28,283 (2.1%) None        Large final payments = payoff", "table_row"),
        ("total_pymnt         9,235 (0.7%) None         Full payment history, real", "table_row"),
        ("total_rec_prncp     5,537 (0.4%) None         Bounded by loan_amnt", "table_row"),
        ("total_rec_int       29,542 (2.2%) None        Long-term loans accumulate interest", "table_row"),
        ("total_rec_late_fee  19,451 (1.5%) None        95.6% zeros; nonzero = real late fees", "table_row"),
        ("recoveries          27,927 (2.1%) None        Post-charge-off collections, real", "table_row"),
        ("collection_rec_fee  27,073 (2%)  None         Fees for recovery, proportional", "table_row"),
        ("last_fico_low/high  209–35,878   None         Zeros are data errors (FICO 300–850)", "table_row"),
        ("                                               but only affect LGD model, not PD", "table_row"),
        ("", ""),
        ("CATEGORICAL FEATURES:", "h3"),
        ("Categorical features don't have 'outliers' in the statistical sense.", "body"),
        ("Rare categories (home_ownership='NONE' = 48 rows) are legitimate.", "body"),
        ("One-hot encoding handles them — the column gets near-zero values but", "body"),
        ("doesn't distort other features. No action needed.", "body"),
    ]
    _render_text_page(pdf, "Outlier Design Decisions (3/3)", lines_p3, font_size=7.8)


def _add_skewness_design_pages(pdf: PdfPages):
    """Design decisions for skewness handling and standardisation."""

    lines_p1 = [
        ("OUR APPROACH: StandardScaler FOR LOGISTIC REGRESSION, NOTHING FOR XGBoost", "h2"),
        ("", ""),
        ("The pipeline applies StandardScaler (z-score normalisation) inside the", "body"),
        ("Logistic Regression pipeline only. XGBoost receives raw, untransformed data.", "body"),
        ("No log transforms, Box-Cox, or Yeo-Johnson are applied to any feature.", "body"),
        ("", ""),
        ("WHY StandardScaler FOR LOGISTIC REGRESSION?", "h2"),
        ("  • Logistic Regression is a linear model — it computes a weighted sum of", "bullet"),
        ("    features. If annual_inc is in [0, 10.9M] and revol_util is in [0, 892],", "bullet"),
        ("    the raw coefficient for annual_inc would be ~1000× smaller, making", "bullet"),
        ("    regularisation (C=0.1) penalise features unequally.", "bullet"),
        ("  • StandardScaler puts all features on the same scale (mean=0, std=1).", "bullet"),
        ("    Now the L2 penalty treats each feature fairly.", "bullet"),
        ("  • This is standard practice for any regularised linear model.", "bullet"),
        ("", ""),
        ("WHY NO SCALING FOR XGBoost?", "h2"),
        ("  • Tree-based models split on thresholds: 'is annual_inc > 75000?'", "bullet"),
        ("    Standardising to 'is z_annual_inc > 0.52?' produces identical splits.", "bullet"),
        ("  • XGBoost is scale-invariant and order-invariant. Scaling adds computation", "bullet"),
        ("    with zero benefit. It's actively harmful for interpretability — SHAP values", "bullet"),
        ("    would be in z-score units instead of dollars/percentages.", "bullet"),
        ("", ""),
        ("WHY NOT LOG TRANSFORM FOR SKEWED FEATURES?", "h2"),
        ("  • log(x) requires x > 0. Several features have zeros:", "bullet"),
        ("    - pub_rec: 83% zeros, pub_rec_bankruptcies: 87.5% zeros", "bullet"),
        ("    - recoveries: 86.3% zeros, total_rec_late_fee: 95.6% zeros", "bullet"),
        ("    - annual_inc: 361 zeros, revol_bal: 6,564 zeros", "bullet"),
        ("    log(0) is undefined — you'd need log(x+1) or log(x + ε), which", "bullet"),
        ("    introduces arbitrary constants and distorts the zero-inflated structure.", "bullet"),
        ("  • For XGBoost: no benefit (tree-based, scale-invariant).", "bullet"),
        ("  • For Logistic Regression: StandardScaler already handles scale. Log would", "bullet"),
        ("    help normality, but LR doesn't assume normally distributed features —", "bullet"),
        ("    it assumes a linear relationship between features and log-odds. Log", "bullet"),
        ("    transform changes that relationship and may make it worse.", "bullet"),
    ]
    _render_text_page(pdf, "Skewness & Standardisation (1/3)", lines_p1, font_size=8.5)

    lines_p2 = [
        ("WHY NOT BOX-COX TRANSFORM?", "h2"),
        ("  • Box-Cox requires strictly positive values (x > 0). Same zero problem", "bullet"),
        ("    as log — even worse, it fails entirely on zeros and negatives.", "bullet"),
        ("  • It finds the optimal power λ to make data most Gaussian. But:", "bullet"),
        ("    - XGBoost doesn't benefit from Gaussian features.", "bullet"),
        ("    - LR doesn't require Gaussian features.", "bullet"),
        ("    - The λ must be fitted on training data and stored for inference.", "bullet"),
        ("      This adds pipeline complexity for no model performance gain.", "bullet"),
        ("", ""),
        ("WHY NOT YEO-JOHNSON TRANSFORM?", "h2"),
        ("  • Yeo-Johnson extends Box-Cox to handle zeros and negatives — it's the", "bullet"),
        ("    'fixed' version. It would technically work on our data.", "bullet"),
        ("  • But the same argument applies: neither XGBoost nor regularised LR", "bullet"),
        ("    requires Gaussian input features.", "bullet"),
        ("  • Added complexity: Yeo-Johnson parameters (one λ per feature) must be", "bullet"),
        ("    fitted on training data, serialised, and applied at inference time.", "bullet"),
        ("    For a pipeline already using StandardScaler (which handles the scale", "bullet"),
        ("    problem), Yeo-Johnson adds a second transform for near-zero benefit.", "bullet"),
        ("", ""),
        ("WHY NOT QUANTILE TRANSFORM (RANK-BASED)?", "h2"),
        ("  • Maps values to a uniform or Gaussian distribution by rank.", "bullet"),
        ("  • Destroys the original scale — SHAP values become uninterpretable.", "bullet"),
        ("  • For XGBoost: pointless, since XGBoost already splits on ranks internally", "bullet"),
        ("    (histogram-based splits are essentially a quantile transform).", "bullet"),
        ("  • For LR: could help with extreme skew, but at the cost of losing", "bullet"),
        ("    monotonic relationships. If annual_inc has a roughly monotonic effect", "bullet"),
        ("    on default (higher income → lower default), quantile transform preserves", "bullet"),
        ("    this. But it complicates feature engineering and explainability.", "bullet"),
        ("", ""),
        ("WHY NOT ROBUST SCALER?", "h2"),
        ("  • RobustScaler uses median and IQR instead of mean and std.", "bullet"),
        ("  • Better than StandardScaler when outliers dominate. But:", "bullet"),
        ("    - With 1.3M rows, outliers barely affect mean/std (10K outliers", "bullet"),
        ("      in a 1.3M sample shift the mean by <1%).", "bullet"),
        ("    - StandardScaler is the sklearn default and sufficient here.", "bullet"),
    ]
    _render_text_page(pdf, "Skewness & Standardisation (2/3)", lines_p2, font_size=8.5)

    lines_p3 = [
        ("PER-FEATURE SKEWNESS ANALYSIS", "h2"),
        ("", ""),
        ("Feature              Skewness  Transform Applied   Why", "table_header"),
        ("─" * 80, "table_row"),
        ("", ""),
        ("NEAR-SYMMETRIC (|skew| < 1) — no transform needed:", "h3"),
        ("loan_amnt             0.78     StandardScaler (LR)  Mild right skew, well-behaved", "table_row"),
        ("funded_amnt           0.78     StandardScaler (LR)  Nearly identical to loan_amnt", "table_row"),
        ("int_rate              0.71     StandardScaler (LR)  Mild skew, bounded [5.3, 31]", "table_row"),
        ("revol_util           −0.04     StandardScaler (LR)  Nearly perfectly symmetric", "table_row"),
        ("fico_range_low        1.29     StandardScaler (LR)  Moderate skew, bounded", "table_row"),
        ("fico_range_high       1.29     StandardScaler (LR)  Same as fico_range_low", "table_row"),
        ("total_acc             0.96     StandardScaler (LR)  Mild skew", "table_row"),
        ("", ""),
        ("MODERATELY SKEWED (1 < |skew| < 5):", "h3"),
        ("installment           1.01     StandardScaler (LR)  Derived from loan_amnt × rate", "table_row"),
        ("open_acc              1.30     StandardScaler (LR)  Right tail but no zeros", "table_row"),
        ("mort_acc              1.64     StandardScaler (LR)  40% zeros — log would distort", "table_row"),
        ("last_pymnt_amnt       1.79     None (LGD only)      Post-origination, not in PD", "table_row"),
        ("total_pymnt           1.04     None (LGD only)      Post-origination, not in PD", "table_row"),
        ("total_rec_int         2.64     None (LGD only)      Post-origination, not in PD", "table_row"),
        ("pub_rec_bankruptcies  3.44     StandardScaler (LR)  87.5% zeros; zero-inflated dist", "table_row"),
        ("                                                     can't be normalised by any", "table_row"),
        ("                                                     continuous transform", "table_row"),
        ("", ""),
        ("EXTREMELY SKEWED (|skew| > 5):", "h3"),
        ("annual_inc           46.32     StandardScaler (LR)  Extreme right tail ($10.9M max).", "table_row"),
        ("                                                     Log would help symmetry but 361", "table_row"),
        ("                                                     zeros block it. XGB handles raw.", "table_row"),
        ("dti                  27.11     StandardScaler (LR)  dti=999 is a data error causing", "table_row"),
        ("                                                     artificial skew. Clip(≥0) in", "table_row"),
        ("                                                     features.py. XGB bins it away.", "table_row"),
        ("pub_rec              11.56     StandardScaler (LR)  83% zeros; zero-inflated dist", "table_row"),
        ("revol_bal            13.75     StandardScaler (LR)  Long tail ($2.9M), has zeros.", "table_row"),
        ("                                                     XGB is primary; LR gets scaled.", "table_row"),
        ("recoveries            8.19     None (LGD only)      86% zeros — post-origination", "table_row"),
        ("total_rec_late_fee   18.90     None (LGD only)      95.6% zeros — post-origination", "table_row"),
        ("collection_rec_fee    8.74     None (LGD only)      87% zeros — post-origination", "table_row"),
        ("last_fico_low/high  −0.6/−3.2  None (LGD only)     Left-skewed from deteriorated", "table_row"),
        ("                                                     credit; post-origination", "table_row"),
        ("", ""),
        ("CATEGORICAL FEATURES:", "h3"),
        ("Skewness doesn't apply to categorical features — they have no numeric", "body"),
        ("distribution to skew. Category imbalance (e.g. debt_consolidation = 58%)", "body"),
        ("is handled by one-hot encoding + model regularisation, not transforms.", "body"),
    ]
    _render_text_page(pdf, "Skewness & Standardisation (3/3)", lines_p3, font_size=7.8)


def _add_categorical_encoding_pages(pdf: PdfPages):
    """Design decisions for categorical feature encoding and class imbalance."""

    lines_p1 = [
        ("OUR APPROACH: ONE-HOT ENCODING WITH drop_first=True", "h2"),
        ("", ""),
        ("encode_categoricals() in features.py applies pd.get_dummies() to 5 features:", "body"),
        ("  purpose, home_ownership, verification_status, initial_list_status, application_type", "body"),
        ("With drop_first=True and dtype=int.", "body"),
        ("", ""),
        ("HOW IT WORKS:", "h3"),
        ("  Each category becomes a binary column (0/1). drop_first=True removes one", "bullet"),
        ("  category per feature to avoid the 'dummy variable trap' — perfect multi-", "bullet"),
        ("  collinearity where columns sum to 1, making the design matrix singular.", "bullet"),
        ("  The dropped category becomes the reference (baseline) class.", "bullet"),
        ("", ""),
        ("  Example — home_ownership has 6 values:", "bullet"),
        ("    MORTGAGE, RENT, OWN, ANY, OTHER, NONE", "bullet"),
        ("    → 5 binary columns: home_ownership_RENT, _OWN, _ANY, _OTHER, _NONE", "bullet"),
        ("    MORTGAGE is the dropped reference — if all 5 columns are 0, it's MORTGAGE", "bullet"),
        ("", ""),
        ("TOTAL COLUMNS PRODUCED:", "h3"),
        ("  purpose:              14 categories → 13 binary columns", "bullet"),
        ("  home_ownership:        6 categories →  5 binary columns", "bullet"),
        ("  verification_status:   3 categories →  2 binary columns", "bullet"),
        ("  initial_list_status:   2 categories →  1 binary column", "bullet"),
        ("  application_type:      2 categories →  1 binary column", "bullet"),
        ("  Total:                27 categories → 22 binary columns added", "bullet"),
        ("", ""),
        ("WHY ONE-HOT OVER LABEL ENCODING?", "h2"),
        ("  Label encoding assigns integers: RENT=0, MORTGAGE=1, OWN=2.", "body"),
        ("  This implies an ordering (MORTGAGE > RENT) that doesn't exist.", "body"),
        ("", ""),
        ("  • XGBoost CAN handle label encoding (it just splits: ≤1 vs >1), but it", "bullet"),
        ("    creates artificial split boundaries. With one-hot, each category gets", "bullet"),
        ("    its own independent split — more interpretable SHAP values.", "bullet"),
        ("  • Logistic Regression CANNOT handle label encoding properly — it treats the", "bullet"),
        ("    integer as a continuous feature, so OWN=2 would have 2× the coefficient", "bullet"),
        ("    effect of MORTGAGE=1. One-hot gives each category its own coefficient.", "bullet"),
    ]
    _render_text_page(pdf, "Categorical Encoding Decisions (1/3)", lines_p1, font_size=8.5)

    lines_p2 = [
        ("WHY NOT TARGET ENCODING (MEAN ENCODING)?", "h2"),
        ("  Target encoding replaces each category with its mean default rate:", "body"),
        ("  e.g. purpose='small_business' → 0.27 (27% default rate).", "body"),
        ("", ""),
        ("  Rejected because:", "body"),
        ("  1. Target leakage: The encoding uses the target variable (default_flag).", "bullet"),
        ("     Even with leave-one-out or k-fold target encoding, information from the", "bullet"),
        ("     test set can leak through shared categories. With time-based splitting", "bullet"),
        ("     (train: 2007–2016, test: 2017–2018), encoding must be computed on train", "bullet"),
        ("     only — but if a new category appears in test, it gets no encoding.", "bullet"),
        ("  2. Overfitting on rare categories: purpose='educational' has 326 rows.", "bullet"),
        ("     Its mean default rate is noisy (high variance). Target encoding would", "bullet"),
        ("     memorise this noise rather than learning a general pattern.", "bullet"),
        ("  3. Loses interpretability: SHAP values for a target-encoded feature show", "bullet"),
        ("     'purpose_encoded = 0.27 → increases default probability' — circular.", "bullet"),
        ("     One-hot SHAP shows 'purpose_small_business = 1 → +0.05 default' — direct.", "bullet"),
        ("", ""),
        ("WHY NOT WoE (WEIGHT OF EVIDENCE) ENCODING?", "h2"),
        ("  We have compute_woe_iv() in features.py but use it for EDA analysis only.", "body"),
        ("", ""),
        ("  WoE replaces each category with ln(% non-events / % events). It's the", "body"),
        ("  standard in traditional credit scoring (logistic regression scorecards).", "body"),
        ("", ""),
        ("  Why we didn't use it for model features:", "body"),
        ("  1. Same leakage problem as target encoding — uses the target variable.", "bullet"),
        ("  2. XGBoost doesn't benefit: WoE linearises the relationship between feature", "bullet"),
        ("     and log-odds, which helps logistic regression but is meaningless for", "bullet"),
        ("     tree-based models that find non-linear splits automatically.", "bullet"),
        ("  3. Monotonicity assumption: WoE forces a monotonic relationship. If", "bullet"),
        ("     purpose='vacation' has LOWER default than 'credit_card' in one period", "bullet"),
        ("     but HIGHER in another, WoE averages this out. One-hot lets the model", "bullet"),
        ("     learn the relationship independently for each category.", "bullet"),
        ("  4. We use it for EDA: Information Value (IV) from WoE tells us which", "bullet"),
        ("     features have predictive power. IV > 0.3 = strong, IV < 0.02 = useless.", "bullet"),
        ("     This informs feature selection, not feature encoding.", "bullet"),
    ]
    _render_text_page(pdf, "Categorical Encoding Decisions (2/3)", lines_p2, font_size=8.5)

    lines_p3 = [
        ("WHY NOT ORDINAL ENCODING FOR GRADE/SUB_GRADE?", "h2"),
        ("  grade (A–G) and sub_grade (A1–G5) are ordinal — A < B < ... < G.", "body"),
        ("  We could encode grade as integers 1–7. We chose to EXCLUDE both instead:", "body"),
        ("", ""),
        ("  • int_rate already captures the same information: Lending Club sets the", "bullet"),
        ("    interest rate based on grade. grade='B' → int_rate ≈ 10–12%.", "bullet"),
        ("  • Including both grade AND int_rate creates multicollinearity. The model", "bullet"),
        ("    can't tell which variable is driving the prediction.", "bullet"),
        ("  • int_rate is continuous (654 unique values vs 7 for grade) — more", "bullet"),
        ("    information-rich. We keep the richer representation.", "bullet"),
        ("", ""),
        ("WHY NOT FREQUENCY ENCODING?", "h2"),
        ("  Replaces each category with its frequency: purpose='debt_consolidation' → 780342.", "body"),
        ("", ""),
        ("  • Assumes popularity = signal. But 'debt_consolidation' is 58% of loans and", "bullet"),
        ("    defaults at ~19%, while 'small_business' is 1.15% and defaults at ~27%.", "bullet"),
        ("    Frequency encoding would say they're 50× apart when the default-rate", "bullet"),
        ("    difference is only 1.4×. Misleading for both models.", "bullet"),
        ("", ""),
        ("HOW CATEGORY IMBALANCE IS HANDLED:", "h2"),
        ("", ""),
        ("  Feature             Dominant Category         Rare Categories", "table_header"),
        ("  ───────────────── ─────────────────────────── ────────────────────────", "table_row"),
        ("  purpose            debt_consolidation (58%)   renewable_energy (0.07%)", "table_row"),
        ("                                                educational (0.02%)", "table_row"),
        ("  home_ownership     MORTGAGE (49.5%)           ANY (0.02%), OTHER (0.01%)", "table_row"),
        ("                                                NONE (0.004%)", "table_row"),
        ("  verification       Source Verified (38.8%)    Roughly balanced (3 classes)", "table_row"),
        ("  initial_list       w (58.3%)                  Mildly imbalanced (binary)", "table_row"),
        ("  application_type   Individual (98.1%)         Joint App (1.9%)", "table_row"),
        ("", ""),
        ("  How each imbalance is handled:", "h3"),
        ("  • One-hot columns for rare categories (e.g. purpose_educational) are", "bullet"),
        ("    mostly zeros. XGBoost simply ignores them if they don't improve splits.", "bullet"),
        ("    No data is lost — the model just doesn't use what isn't useful.", "bullet"),
        ("  • LR regularisation (C=0.1): L2 penalty shrinks coefficients for rare-", "bullet"),
        ("    category columns toward zero, preventing overfitting to 326 rows.", "bullet"),
        ("  • class_weight='balanced' (LR) and scale_pos_weight (XGBoost) handle the", "bullet"),
        ("    TARGET imbalance (80/20 Fully Paid vs Default), not feature imbalance.", "bullet"),
        ("    Feature category imbalance doesn't need special treatment — the model", "bullet"),
        ("    simply learns that MORTGAGE is the baseline and rare categories are", "bullet"),
        ("    deviations from it.", "bullet"),
        ("", ""),
        ("FEATURES EXCLUDED FROM MODEL (NOT ENCODED):", "h3"),
        ("  emp_title:   378K unique — one-hot would create 378K sparse columns.", "bullet"),
        ("  title:       61K unique — same problem. Kept for EDA text analysis only.", "bullet"),
        ("  emp_length:  Converted to numeric (emp_length_num, 0–10) — ordinal.", "bullet"),
        ("  term:        Converted to numeric (36 or 60) — already a number.", "bullet"),
        ("  grade/sub_grade: Excluded — redundant with int_rate (see above).", "bullet"),
    ]
    _render_text_page(pdf, "Categorical Encoding Decisions (3/3)", lines_p3, font_size=8.0)


def generate_pdf(df: pd.DataFrame, numeric_cols, categorical_cols, date_cols):
    """Generate the full PDF report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "feature_analysis_report.pdf"
    print(f"Generating PDF report at {out_path}...")

    with PdfPages(str(out_path)) as pdf:
        _add_title_page(pdf, df, numeric_cols, categorical_cols, date_cols)
        _add_missing_values_page(pdf, df)
        _add_findings_page(pdf)

        print("  Adding column selection reasoning...")
        _add_column_selection_pages(pdf)

        print("  Adding cache directory explanation...")
        _add_cache_explanation_page(pdf)

        print("  Adding Module 1 files explanation...")
        _add_module1_files_pages(pdf)

        print("  Adding missing value design decisions...")
        _add_missing_values_design_pages(pdf)

        print("  Adding outlier design decisions...")
        _add_outlier_design_pages(pdf)

        print("  Adding skewness & standardisation decisions...")
        _add_skewness_design_pages(pdf)

        print("  Adding categorical encoding decisions...")
        _add_categorical_encoding_pages(pdf)

        print("  Plotting numeric features...")
        for i, col in enumerate(numeric_cols):
            _add_numeric_page(pdf, df, col)
            if (i + 1) % 5 == 0:
                print(f"    {i + 1}/{len(numeric_cols)} done")

        print("  Plotting categorical features...")
        for col in categorical_cols:
            _add_categorical_page(pdf, df, col)

        print("  Plotting date features...")
        for col in date_cols:
            _add_date_page(pdf, df, col)

    print(f"PDF saved: {out_path}")
    return out_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    df = load_filtered_data()
    numeric_cols, categorical_cols, date_cols = classify_columns(df)

    print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(date_cols)} date\n")

    print_missing_values(df)
    print_duplicates(df)
    print_numeric_stats(df, numeric_cols)
    print_categorical_stats(df, categorical_cols)
    print_date_stats(df, date_cols)

    pdf_path = generate_pdf(df, numeric_cols, categorical_cols, date_cols)
    print(f"\nDone. PDF report: {pdf_path}")


if __name__ == "__main__":
    main()
