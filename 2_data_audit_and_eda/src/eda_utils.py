"""EDA helpers for text, category, temporal, and market diagnostics."""

from __future__ import annotations

import re
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HTML_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s{2,}")


def text_quality_summary(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Summarize text completeness, length, and common noise patterns."""

    rows = []

    for column in columns:
        if column not in df.columns:
            continue

        raw = df[column]
        text = raw.fillna("").astype(str)
        stripped = text.str.strip()
        non_empty = stripped.ne("")

        char_length = stripped.str.len()
        word_length = stripped.str.split().str.len()

        rows.append(
            {
                "column": column,
                "rows": len(df),
                "missing_count": int(raw.isna().sum()),
                "empty_after_strip_count": int((~non_empty).sum()),
                "median_chars_non_empty": float(char_length[non_empty].median()) if non_empty.any() else np.nan,
                "mean_chars_non_empty": float(char_length[non_empty].mean()) if non_empty.any() else np.nan,
                "median_words_non_empty": float(word_length[non_empty].median()) if non_empty.any() else np.nan,
                "html_like_pct": float(stripped.str.contains(HTML_PATTERN, regex=True).mean() * 100),
                "url_like_pct": float(stripped.str.contains(URL_PATTERN, regex=True).mean() * 100),
                "repeated_whitespace_pct": float(
                    stripped.str.contains(WHITESPACE_PATTERN, regex=True).mean() * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def normalized_title_duplicate_summary(df: pd.DataFrame, column: str = "Title") -> dict[str, float | int]:
    """Estimate duplicate-title prevalence after conservative normalization."""

    if column not in df.columns:
        return {
            "rows": len(df),
            "non_missing_titles": 0,
            "duplicate_title_rows": 0,
            "duplicate_title_pct": np.nan,
        }

    normalized = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    duplicates = int(normalized.duplicated(keep="first").sum())

    return {
        "rows": len(df),
        "non_missing_titles": len(normalized),
        "duplicate_title_rows": duplicates,
        "duplicate_title_pct": (duplicates / len(normalized) * 100) if len(normalized) else np.nan,
    }


def numeric_summary(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return descriptive statistics for available numeric columns in a sample."""

    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.DataFrame()

    numeric = df[available].apply(pd.to_numeric, errors="coerce")
    summary = numeric.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    summary["missing"] = numeric.isna().sum()
    summary["missing_pct"] = numeric.isna().mean() * 100
    return summary


def plot_missingness(missingness: pd.DataFrame, top_n: int | None = None) -> None:
    """Plot missing-value percentages by column."""

    data = missingness.copy()
    if top_n is not None:
        data = data.head(top_n)

    data = data.sort_values("missing_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(data) * 0.35)))
    ax.barh(data["column"], data["missing_pct"])
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Column")
    ax.set_title("Missingness by Column")
    plt.tight_layout()
    plt.show()


def plot_top_counts(
    counts: pd.DataFrame,
    category_column: str,
    top_n: int = 15,
    title: str | None = None,
) -> None:
    """Plot the most frequent values from a two-column count table."""

    data = counts.head(top_n).sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(data) * 0.4)))
    ax.barh(data[category_column].astype(str), data["count"])
    ax.set_xlabel("Number of records")
    ax.set_ylabel(category_column)
    ax.set_title(title or f"Top {top_n} Values: {category_column}")
    plt.tight_layout()
    plt.show()


def plot_records_by_year(by_year: pd.DataFrame) -> None:
    """Plot the number of records per publication year."""

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(by_year["year"], by_year["count"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of records")
    ax.set_title("News Records by Year")
    ax.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()
    plt.show()


def plot_text_length_histograms(
    df: pd.DataFrame,
    columns: Sequence[str],
    max_quantile: float = 0.99,
) -> None:
    """Plot one character-length histogram per text field."""

    for column in columns:
        if column not in df.columns:
            continue

        lengths = df[column].fillna("").astype(str).str.len()
        upper = lengths.quantile(max_quantile)
        clipped = lengths[lengths <= upper]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(clipped, bins=50)
        ax.set_xlabel("Characters")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Text Length Distribution: {column} (<= {max_quantile:.0%} quantile)")
        plt.tight_layout()
        plt.show()
