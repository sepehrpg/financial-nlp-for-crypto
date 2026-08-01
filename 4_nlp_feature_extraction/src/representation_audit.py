"""Representation-level text coverage and length diagnostics.

All calculations are descriptive. They do not mutate input text and do not learn
parameters from the corpus.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


WORD_PATTERN = r"[A-Za-z0-9]+(?:[\-'’][A-Za-z0-9]+)*"
COMPARISON_COLUMNS = [
    "representation",
    "row_count",
    "null_count",
    "null_pct",
    "empty_count",
    "empty_pct",
    "nonempty_count",
    "nonempty_pct",
    "median_word_count",
    "p95_word_count",
    "max_word_count",
    "median_character_count",
    "p95_character_count",
    "max_character_count",
    "very_short_word_count",
    "very_short_word_pct_nonempty",
    "very_short_character_count",
    "very_short_character_pct_nonempty",
]


def normalize_text_series(series: pd.Series) -> pd.Series:
    """Return a nullable-string series with missing values represented as empty."""

    return series.astype("string").fillna("")


def row_length_metrics(series: pd.Series) -> pd.DataFrame:
    """Calculate null, empty, character, and word metrics for every row."""

    text = normalize_text_series(series)
    stripped = text.str.strip()
    return pd.DataFrame(
        {
            "is_null": series.isna().astype("int8"),
            "is_empty": stripped.eq("").astype("int8"),
            "character_count": stripped.str.len().fillna(0).astype("int32"),
            "word_count": stripped.str.count(WORD_PATTERN).fillna(0).astype("int32"),
        },
        index=series.index,
    )


def _safe_pct(count: int, denominator: int) -> float:
    return float(count / denominator * 100) if denominator else 0.0


def build_representation_comparison(
    df: pd.DataFrame,
    representations: Iterable[str],
    *,
    very_short_max_words: int = 3,
    very_short_max_characters: int = 20,
    percentile: float = 0.95,
) -> pd.DataFrame:
    """Summarize text coverage and length for each available representation."""

    rows: list[dict[str, Any]] = []
    for representation in representations:
        metrics = row_length_metrics(df[representation])
        row_count = len(metrics)
        null_count = int(metrics["is_null"].sum())
        empty_count = int(metrics["is_empty"].sum())
        nonempty_mask = metrics["is_empty"].eq(0)
        nonempty_count = int(nonempty_mask.sum())

        word_counts = metrics["word_count"]
        character_counts = metrics["character_count"]
        short_word_count = int(
            (nonempty_mask & word_counts.le(very_short_max_words)).sum()
        )
        short_character_count = int(
            (
                nonempty_mask
                & character_counts.le(very_short_max_characters)
            ).sum()
        )

        rows.append(
            {
                "representation": representation,
                "row_count": row_count,
                "null_count": null_count,
                "null_pct": _safe_pct(null_count, row_count),
                "empty_count": empty_count,
                "empty_pct": _safe_pct(empty_count, row_count),
                "nonempty_count": nonempty_count,
                "nonempty_pct": _safe_pct(nonempty_count, row_count),
                "median_word_count": float(word_counts.median()) if row_count else 0.0,
                "p95_word_count": (
                    float(word_counts.quantile(percentile)) if row_count else 0.0
                ),
                "max_word_count": int(word_counts.max()) if row_count else 0,
                "median_character_count": (
                    float(character_counts.median()) if row_count else 0.0
                ),
                "p95_character_count": (
                    float(character_counts.quantile(percentile)) if row_count else 0.0
                ),
                "max_character_count": (
                    int(character_counts.max()) if row_count else 0
                ),
                "very_short_word_count": short_word_count,
                "very_short_word_pct_nonempty": _safe_pct(
                    short_word_count, nonempty_count
                ),
                "very_short_character_count": short_character_count,
                "very_short_character_pct_nonempty": _safe_pct(
                    short_character_count, nonempty_count
                ),
            }
        )

    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
