"""Normalize parsed SEC records into the project's acquisition schema."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping


def content_hash(title: str, full_text: str) -> str:
    payload = f"{title.strip()}\n{full_text.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_document_id(release_number: str) -> str:
    return "sec_" + release_number.replace("-", "_")


def normalize_record(
    article: Mapping[str, Any],
    *,
    source: str = "sec.gov",
    source_type: str = "press_release",
    ingestion_method: str = "web_scraping",
    collection_mode: str = "live_http",
    http_status: int = 200,
    scraped_at: str | None = None,
) -> dict[str, Any]:
    """Create one stable row suitable for CSV, Parquet, and later Phase 3 input."""
    title = str(article.get("title", "")).strip()
    description = str(article.get("description", "")).strip()
    full_text = str(article.get("full_text", "")).strip()
    release_number = str(article.get("release_number", "")).strip()
    matched = article.get("matched_keywords") or []
    if isinstance(matched, str):
        matched_keywords = matched
    else:
        matched_keywords = "|".join(sorted({str(item) for item in matched}))
    timestamp = scraped_at or datetime.now(timezone.utc).isoformat()

    return {
        "source_row_id": build_document_id(release_number),
        "document_id": build_document_id(release_number),
        "source": source,
        "source_type": source_type,
        "release_number": release_number,
        "title": title,
        "description": description,
        "full_text": full_text,
        "published_at": article.get("published_at"),
        "last_reviewed_at": article.get("last_reviewed_at"),
        "url": article.get("url"),
        "canonical_url": article.get("canonical_url") or article.get("url"),
        "coin_type": "Bitcoin",
        "language": "en",
        "ingestion_method": ingestion_method,
        "collection_mode": collection_mode,
        "body_text_status": (
            "complete_page_body" if collection_mode == "live_http" else "verified_excerpt"
        ),
        "full_text_is_complete": collection_mode == "live_http",
        "source_verified": True,
        "scraped_at": timestamp,
        "content_hash": content_hash(title, full_text),
        "http_status": int(http_status),
        "bitcoin_keyword_count": int(article.get("bitcoin_keyword_count", 0)),
        "matched_keywords": matched_keywords,
        "word_count": len(full_text.split()),
        "character_count": len(full_text),
        "is_bitcoin_related": bool(article.get("is_bitcoin_related", True)),
    }


def record_to_json(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
