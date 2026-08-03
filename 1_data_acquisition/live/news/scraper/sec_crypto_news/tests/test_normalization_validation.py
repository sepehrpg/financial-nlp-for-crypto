import pandas as pd
import pytest

from src.normalization import normalize_record
from src.storage import records_to_frame
from src.validation import quality_report, validate_frame


def _record(release_number: str = "2025-101") -> dict:
    return normalize_record(
        {
            "title": "Bitcoin ETF update",
            "description": "Official update",
            "full_text": "Bitcoin market information and investor protection details.",
            "release_number": release_number,
            "published_at": "2025-07-29",
            "last_reviewed_at": "2025-07-30",
            "url": f"https://www.sec.gov/newsroom/press-releases/{release_number}",
            "canonical_url": f"https://www.sec.gov/newsroom/press-releases/{release_number}",
            "matched_keywords": ["bitcoin"],
            "bitcoin_keyword_count": 2,
            "is_bitcoin_related": True,
        },
        collection_mode="live_http",
    )


def test_normalization_and_quality_report() -> None:
    frame = records_to_frame([_record()])
    validate_frame(frame)
    assert frame.loc[0, "source_row_id"] == "sec_2025_101"
    assert bool(frame.loc[0, "full_text_is_complete"]) is True
    report = quality_report(frame).set_index("metric")["value"]
    assert int(report["row_count"]) == 1
    assert int(report["duplicate_content_hashes"]) == 0


def test_duplicate_ids_are_rejected() -> None:
    frame = records_to_frame([_record(), _record()])
    with pytest.raises(ValueError, match="source_row_id"):
        validate_frame(frame)
