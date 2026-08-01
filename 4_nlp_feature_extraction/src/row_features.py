"""Corpus-independent, explainable row-level text features."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import re

import pandas as pd

from .representation_audit import normalize_text_series, row_length_metrics


NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z0-9_])"
)
PERCENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%"
)
ALPHABETIC_RE = re.compile(r"[A-Za-z]")
UPPERCASE_RE = re.compile(r"[A-Z]")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not slug:
        raise ValueError(f"Could not create a feature name from: {value!r}")
    return slug


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword.strip()).replace(r"\ ", r"\s+")
    return re.compile(
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )


def _count_regex(text: pd.Series, pattern: re.Pattern[str] | str) -> pd.Series:
    return text.str.count(pattern).fillna(0).astype("int32")


def extract_representation_features(
    series: pd.Series,
    *,
    prefix: str,
    currency_symbols: Sequence[str],
    keyword_groups: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    """Extract deterministic numeric features from one text representation."""

    text = normalize_text_series(series).str.strip()
    lengths = row_length_metrics(series)
    features = pd.DataFrame(index=series.index)

    features[f"{prefix}__is_empty"] = lengths["is_empty"].astype("int8")
    features[f"{prefix}__character_count"] = lengths["character_count"]
    features[f"{prefix}__word_count"] = lengths["word_count"]
    features[f"{prefix}__number_count"] = _count_regex(text, NUMBER_RE)
    features[f"{prefix}__percentage_count"] = _count_regex(text, PERCENT_RE)

    currency_pattern = "[" + re.escape("".join(currency_symbols)) + "]"
    features[f"{prefix}__currency_symbol_count"] = _count_regex(
        text, currency_pattern
    )

    uppercase_count = _count_regex(text, UPPERCASE_RE)
    alphabetic_count = _count_regex(text, ALPHABETIC_RE)
    features[f"{prefix}__uppercase_character_count"] = uppercase_count
    features[f"{prefix}__alphabetic_character_count"] = alphabetic_count

    denominator = alphabetic_count.where(alphabetic_count.ne(0), 1)
    uppercase_ratio = uppercase_count.div(denominator).where(
        alphabetic_count.ne(0), 0.0
    )
    features[f"{prefix}__uppercase_ratio"] = uppercase_ratio.astype("float32")

    features[f"{prefix}__question_mark_count"] = _count_regex(text, r"\?")
    features[f"{prefix}__exclamation_mark_count"] = _count_regex(text, r"!")

    total_keyword_count = pd.Series(0, index=series.index, dtype="int32")
    for group_name, keywords in keyword_groups.items():
        group_count = pd.Series(0, index=series.index, dtype="int32")
        for keyword in keywords:
            group_count = group_count.add(
                _count_regex(text, _keyword_pattern(keyword)),
                fill_value=0,
            ).astype("int32")

        group_slug = _slug(group_name)
        features[f"{prefix}__keyword_{group_slug}_count"] = group_count
        features[f"{prefix}__keyword_{group_slug}_present"] = (
            group_count.gt(0).astype("int8")
        )
        total_keyword_count = total_keyword_count.add(
            group_count, fill_value=0
        ).astype("int32")

    features[f"{prefix}__financial_keyword_count"] = total_keyword_count
    return features.reset_index(drop=True)


def build_row_level_features(
    df: pd.DataFrame,
    *,
    id_column: str,
    representations: Sequence[str],
    representation_prefixes: Mapping[str, str],
    currency_symbols: Sequence[str],
    keyword_groups: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    """Create one feature row per input row while preserving identifier order."""

    output = pd.DataFrame({id_column: df[id_column].reset_index(drop=True).copy()})
    frames = [output]

    for representation in representations:
        prefix = _slug(representation_prefixes[representation])
        frames.append(
            extract_representation_features(
                df[representation],
                prefix=prefix,
                currency_symbols=currency_symbols,
                keyword_groups=keyword_groups,
            )
        )

    return pd.concat(frames, axis=1)
