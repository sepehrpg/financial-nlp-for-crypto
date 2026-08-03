"""Validation and quality reporting for scraped SEC records."""

from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "source_row_id",
    "release_number",
    "title",
    "full_text",
    "published_at",
    "canonical_url",
    "content_hash",
    "is_bitcoin_related",
]


def validate_frame(frame: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required output columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Scraper output contains no Bitcoin-related records.")
    if frame["source_row_id"].isna().any():
        raise ValueError("source_row_id contains missing values.")
    if frame["source_row_id"].duplicated().any():
        raise ValueError("source_row_id must be unique.")
    if frame["canonical_url"].duplicated().any():
        raise ValueError("canonical_url must be unique.")
    if frame["content_hash"].duplicated().any():
        raise ValueError("Duplicate full-text content was detected.")
    if not frame["is_bitcoin_related"].fillna(False).all():
        raise ValueError("Non-Bitcoin rows were included in the final output.")
    if frame["title"].fillna("").str.strip().eq("").any():
        raise ValueError("At least one title is empty.")
    if frame["full_text"].fillna("").str.strip().eq("").any():
        raise ValueError("At least one full_text value is empty.")


def quality_report(frame: pd.DataFrame) -> pd.DataFrame:
    published = pd.to_datetime(frame["published_at"], errors="coerce")
    rows: list[dict[str, Any]] = [
        {"metric": "row_count", "value": int(len(frame))},
        {"metric": "unique_release_numbers", "value": int(frame["release_number"].nunique())},
        {"metric": "duplicate_source_row_ids", "value": int(frame["source_row_id"].duplicated().sum())},
        {"metric": "duplicate_canonical_urls", "value": int(frame["canonical_url"].duplicated().sum())},
        {"metric": "duplicate_content_hashes", "value": int(frame["content_hash"].duplicated().sum())},
        {"metric": "missing_titles", "value": int(frame["title"].isna().sum() + frame["title"].fillna("").str.strip().eq("").sum())},
        {"metric": "missing_full_text", "value": int(frame["full_text"].isna().sum() + frame["full_text"].fillna("").str.strip().eq("").sum())},
        {"metric": "invalid_published_dates", "value": int(published.isna().sum())},
        {"metric": "median_word_count", "value": float(frame["word_count"].median())},
        {"metric": "min_word_count", "value": int(frame["word_count"].min())},
        {"metric": "max_word_count", "value": int(frame["word_count"].max())},
        {"metric": "oldest_publication_date", "value": published.min().date().isoformat() if published.notna().any() else None},
        {"metric": "newest_publication_date", "value": published.max().date().isoformat() if published.notna().any() else None},
    ]
    return pd.DataFrame(rows)


def summary_payload(frame: pd.DataFrame, *, failures: int, collection_mode: str) -> dict[str, Any]:
    published = pd.to_datetime(frame["published_at"], errors="coerce")
    return {
        "status": "completed",
        "collection_mode": collection_mode,
        "records": int(len(frame)),
        "failures": int(failures),
        "oldest_publication_date": published.min().date().isoformat() if published.notna().any() else None,
        "newest_publication_date": published.max().date().isoformat() if published.notna().any() else None,
        "median_word_count": float(frame["word_count"].median()),
        "all_bitcoin_related": bool(frame["is_bitcoin_related"].all()),
        "source_row_id_unique": bool(not frame["source_row_id"].duplicated().any()),
        "canonical_url_unique": bool(not frame["canonical_url"].duplicated().any()),
        "content_hash_unique": bool(not frame["content_hash"].duplicated().any()),
    }
