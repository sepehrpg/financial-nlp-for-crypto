"""Atomic storage helpers for raw HTML, processed datasets, and reports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .validation import quality_report, summary_payload, validate_frame


OUTPUT_COLUMNS = [
    "source_row_id",
    "document_id",
    "source",
    "source_type",
    "release_number",
    "title",
    "description",
    "full_text",
    "published_at",
    "last_reviewed_at",
    "url",
    "canonical_url",
    "coin_type",
    "language",
    "ingestion_method",
    "collection_mode",
    "body_text_status",
    "full_text_is_complete",
    "source_verified",
    "scraped_at",
    "content_hash",
    "http_status",
    "bitcoin_keyword_count",
    "matched_keywords",
    "word_count",
    "character_count",
    "is_bitcoin_related",
]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_raw_html(path: Path, html: str) -> None:
    _atomic_text(Path(path), html)


def records_to_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame([dict(record) for record in records])
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[OUTPUT_COLUMNS]
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce").dt.date.astype("string")
    frame["last_reviewed_at"] = pd.to_datetime(frame["last_reviewed_at"], errors="coerce").dt.date.astype("string")
    frame = frame.sort_values(["published_at", "release_number"], ascending=[False, False]).reset_index(drop=True)
    return frame


def merge_incremental(existing: pd.DataFrame | None, new_frame: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = new_frame.copy()
    else:
        merged = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["release_number"], keep="last")
    merged = merged.drop_duplicates(subset=["canonical_url"], keep="last")
    merged = merged.drop_duplicates(subset=["content_hash"], keep="last")
    return merged[OUTPUT_COLUMNS].sort_values(
        ["published_at", "release_number"], ascending=[False, False]
    ).reset_index(drop=True)


def read_existing(csv_path: Path) -> pd.DataFrame | None:
    path = Path(csv_path)
    if not path.is_file():
        return None
    return pd.read_csv(path, low_memory=False)


def save_outputs(
    frame: pd.DataFrame,
    *,
    csv_path: Path,
    parquet_path: Path,
    jsonl_path: Path,
    quality_path: Path,
    summary_path: Path,
    failures: list[Mapping[str, Any]],
    failed_urls_path: Path,
    collection_mode: str,
) -> dict[str, str]:
    validate_frame(frame)
    for path in (csv_path, parquet_path, jsonl_path, quality_path, summary_path, failed_urls_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    csv_temp = Path(csv_path).with_suffix(Path(csv_path).suffix + ".tmp")
    frame.to_csv(csv_temp, index=False, encoding="utf-8-sig")
    csv_temp.replace(csv_path)

    parquet_written = False
    parquet_error: str | None = None
    parquet_temp = Path(parquet_path).with_suffix(Path(parquet_path).suffix + ".tmp")
    try:
        frame.to_parquet(parquet_temp, index=False)
        parquet_temp.replace(parquet_path)
        parquet_written = True
    except ImportError as error:
        if parquet_temp.exists():
            parquet_temp.unlink()
        parquet_error = str(error)
        unavailable_path = Path(parquet_path).with_suffix(Path(parquet_path).suffix + ".unavailable.json")
        _atomic_text(
            unavailable_path,
            json.dumps(
                {
                    "status": "not_generated",
                    "reason": "No Parquet engine is installed in this runtime.",
                    "install": "pip install pyarrow",
                    "error": parquet_error,
                },
                indent=2,
            ),
        )

    jsonl = "\n".join(json.dumps(record, ensure_ascii=False) for record in frame.to_dict(orient="records")) + "\n"
    _atomic_text(Path(jsonl_path), jsonl)

    report = quality_report(frame)
    report.to_csv(quality_path, index=False)
    pd.DataFrame(failures, columns=["url", "stage", "error"]).to_csv(failed_urls_path, index=False)

    summary = summary_payload(frame, failures=len(failures), collection_mode=collection_mode)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    scraper_root = Path(summary_path).resolve().parents[2]

    def display_path(value: Path) -> str:
        resolved = Path(value).resolve()
        try:
            return resolved.relative_to(scraper_root).as_posix()
        except ValueError:
            return str(resolved)

    summary["outputs"] = {
        "csv": display_path(Path(csv_path)),
        "parquet": display_path(Path(parquet_path)) if parquet_written else None,
        "parquet_note": None if parquet_written else "Install pyarrow and rerun to create Parquet.",
        "jsonl": display_path(Path(jsonl_path)),
        "quality_report": display_path(Path(quality_path)),
        "failed_urls": display_path(Path(failed_urls_path)),
    }
    _atomic_text(Path(summary_path), json.dumps(summary, indent=2, ensure_ascii=False))
    return summary["outputs"]
