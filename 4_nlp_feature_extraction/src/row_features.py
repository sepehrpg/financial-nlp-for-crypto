"""Validation, representation auditing, and explainable row-level features.

This stage intentionally uses text and stable row identifiers only. It does not
read sentiment, price, volume, market-move, or outcome fields as features.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import string
from typing import Final, Sequence

import pandas as pd


REPRESENTATION_COLUMNS: Final[tuple[str, ...]] = (
    "text_title_description",
    "Filtered_Text",
)
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("source_row_id", *REPRESENTATION_COLUMNS)
FINANCIAL_KEYWORDS: Final[tuple[str, ...]] = (
    "bitcoin",
    "crypto",
    "etf",
    "sec",
    "inflation",
    "interest rate",
    "market",
)

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\d")
_CURRENCY_RE = re.compile(r"[$€£¥₿]")
_PERCENT_RE = re.compile(r"%|\bpercent\b", flags=re.IGNORECASE)


def default_input_path(project_root: Path) -> Path:
    """Return the canonical Phase 3 handoff path."""

    return (
        project_root
        / "3_text_preprocessing"
        / "data"
        / "processed"
        / "cryptovision_v1_preprocessed.parquet"
    )


def validate_phase3_input(
    frame: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> dict[str, int]:
    """Validate the Phase 3/4 contract and return compact row diagnostics."""

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Phase 3 input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Phase 3 input is empty; no row-level features can be built.")

    missing_ids = int(frame["source_row_id"].isna().sum())
    duplicated_ids = int(frame["source_row_id"].duplicated(keep=False).sum())
    if missing_ids:
        raise ValueError(f"source_row_id contains {missing_ids} missing value(s).")
    if duplicated_ids:
        raise ValueError(
            "source_row_id must be unique; "
            f"{duplicated_ids} row(s) participate in duplicated IDs."
        )

    return {
        "row_count": int(len(frame)),
        "unique_source_row_ids": int(frame["source_row_id"].nunique()),
        "duplicated_source_row_id_rows": duplicated_ids,
    }


def load_phase3_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the Phase 3 Parquet handoff without modifying it."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 3 Parquet file not found at {path}. Run Phase 3 or place "
            "cryptovision_v1_preprocessed.parquet at that path, then rerun."
        )
    frame = pd.read_parquet(path)
    validate_phase3_input(frame)
    return frame


def _text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a missing-safe string series while treating whitespace as empty."""

    return frame[column].fillna("").astype(str)


def audit_representations(
    frame: pd.DataFrame,
    representations: Sequence[str] = REPRESENTATION_COLUMNS,
    *,
    very_short_word_threshold: int = 5,
) -> dict[str, object]:
    """Compare candidate text representations with compact JSON-safe metrics."""

    contract = validate_phase3_input(frame, ("source_row_id", *representations))
    report: dict[str, object] = {
        "contract": contract,
        "very_short_definition": (
            f"non-empty text with fewer than {very_short_word_threshold} words"
        ),
        "representations": {},
    }
    representation_report = report["representations"]
    assert isinstance(representation_report, dict)

    for column in representations:
        text = _text_series(frame, column)
        stripped = text.str.strip()
        characters = stripped.str.len()
        words = stripped.str.findall(_WORD_RE).str.len()
        non_empty = stripped.ne("")
        representation_report[column] = {
            "empty_count": int((~non_empty).sum()),
            "character_count_total": int(characters.sum()),
            "word_count_total": int(words.sum()),
            "character_count_median": float(characters.median()),
            "character_count_p95": float(characters.quantile(0.95)),
            "character_count_max": int(characters.max()),
            "word_count_median": float(words.median()),
            "word_count_p95": float(words.quantile(0.95)),
            "word_count_max": int(words.max()),
            "very_short_text_count": int((non_empty & words.lt(very_short_word_threshold)).sum()),
        }
    return report


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide counts and return zero when the denominator is zero."""

    return numerator.div(denominator.where(denominator.ne(0))).fillna(0.0)


def _keyword_pattern(keyword: str) -> str:
    return rf"(?<!\w){re.escape(keyword)}(?!\w)"


def build_row_level_features(
    frame: pd.DataFrame,
    *,
    text_column: str = "text_title_description",
    keywords: Sequence[str] = FINANCIAL_KEYWORDS,
) -> pd.DataFrame:
    """Create deterministic numeric text features, keyed by ``source_row_id``.

    ``text_column`` must be an approved Phase 3 representation. No other input
    columns are examined, which makes the leakage boundary explicit.
    """

    if text_column not in REPRESENTATION_COLUMNS:
        raise ValueError(
            f"Unsupported text representation {text_column!r}; choose one of "
            f"{REPRESENTATION_COLUMNS}."
        )
    validate_phase3_input(frame)
    text = _text_series(frame, text_column)
    character_count = text.str.len().astype("int64")
    word_count = text.str.findall(_WORD_RE).str.len().astype("int64")
    digit_count = text.str.count(_DIGIT_RE).astype("int64")
    letter_count = text.str.count(r"[A-Za-z]").astype("int64")
    uppercase_count = text.str.count(r"[A-Z]").astype("int64")

    features = pd.DataFrame({"source_row_id": frame["source_row_id"].to_numpy()})
    features["text_character_count"] = character_count.to_numpy()
    features["text_word_count"] = word_count.to_numpy()
    features["digit_count"] = digit_count.to_numpy()
    features["digit_ratio"] = _safe_ratio(digit_count, character_count).to_numpy()
    features["percentage_mention_count"] = text.str.count(_PERCENT_RE).to_numpy()
    features["currency_symbol_count"] = text.str.count(_CURRENCY_RE).to_numpy()
    features["uppercase_letter_count"] = uppercase_count.to_numpy()
    features["uppercase_ratio"] = _safe_ratio(uppercase_count, letter_count).to_numpy()
    features["punctuation_count"] = text.map(
        lambda value: sum(character in string.punctuation for character in value)
    ).to_numpy()
    features["question_mark_count"] = text.str.count(r"\?").to_numpy()
    features["exclamation_mark_count"] = text.str.count("!").to_numpy()

    for keyword in keywords:
        safe_name = re.sub(r"[^a-z0-9]+", "_", keyword.casefold()).strip("_")
        features[f"has_keyword_{safe_name}"] = (
            text.str.contains(_keyword_pattern(keyword), case=False, regex=True).astype("int8")
        ).to_numpy()
    return features


def run_row_feature_pipeline(
    input_path: Path,
    feature_output_path: Path,
    audit_output_path: Path,
    *,
    text_column: str = "text_title_description",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the reusable Stage 1 pipeline and persist only derived artifacts."""

    frame = load_phase3_dataset(input_path)
    audit = audit_representations(frame)
    features = build_row_level_features(frame, text_column=text_column)
    feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_output_path, index=False)
    audit_output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return features, audit
