"""End-to-end SEC Bitcoin press-release collection pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin

import pandas as pd

from .article_parser import parse_press_release
from .config import find_scraper_root, load_config, resolve_config_paths
from .discovery import build_listing_params, deduplicate_candidates, parse_listing_page
from .http_client import PoliteHttpClient
from .normalization import normalize_record
from .storage import merge_incremental, read_existing, records_to_frame, save_outputs, save_raw_html


def _client_from_config(config: Mapping[str, Any]) -> PoliteHttpClient:
    http = config["http"]
    return PoliteHttpClient(
        user_agent=http["user_agent"],
        allowed_hosts=config["source"]["allowed_hosts"],
        accept=http["accept"],
        accept_encoding=http["accept_encoding"],
        timeout_seconds=http["timeout_seconds"],
        delay_seconds=http["delay_seconds"],
        max_retries=http["max_retries"],
        backoff_factor=http["backoff_factor"],
        verify_ssl=http["verify_ssl"],
    )


def _storage_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in config["storage"].items()}


def run_live_scrape(
    config: Mapping[str, Any],
    *,
    max_pages: int | None = None,
    max_articles: int | None = None,
    incremental: bool | None = None,
) -> pd.DataFrame:
    """Discover crypto press releases, parse them, and retain Bitcoin matches."""
    source = config["source"]
    filter_cfg = config["bitcoin_filter"]
    run_cfg = config["run"]
    paths = _storage_paths(config)
    base_url = source["base_url"]
    listing_url = urljoin(base_url, source["listing_path"])
    page_limit = int(max_pages or source["max_pages"])
    article_limit = max_articles if max_articles is not None else run_cfg.get("max_articles")
    use_incremental = run_cfg["incremental"] if incremental is None else incremental

    candidates = []
    failures: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    scraped_at = datetime.now(timezone.utc).isoformat()

    with _client_from_config(config) as client:
        for page in range(int(source["start_page"]), int(source["start_page"]) + page_limit):
            try:
                listing_result = client.get(
                    listing_url,
                    params=build_listing_params(query=source["search_query"], page=page),
                )
                page_candidates = parse_listing_page(
                    listing_result.text,
                    base_url=base_url,
                    page=page,
                )
            except Exception as error:  # network failures belong in a report
                failures.append({"url": listing_url, "stage": f"listing_page_{page}", "error": repr(error)})
                break
            if not page_candidates and run_cfg["stop_on_empty_listing"]:
                break
            candidates.extend(page_candidates)

        candidates = deduplicate_candidates(candidates)
        if article_limit is not None:
            candidates = candidates[: int(article_limit)]

        existing = read_existing(paths["processed_csv"]) if use_incremental else None
        known_urls = set(existing["canonical_url"].dropna().astype(str)) if existing is not None else set()

        for candidate in candidates:
            if use_incremental and candidate.url in known_urls:
                continue
            try:
                result = client.get(candidate.url)
                if run_cfg["save_raw_html"]:
                    release_token = (candidate.release_number or "unknown").replace("-", "_")
                    save_raw_html(paths["raw_html_directory"] / f"sec_{release_token}.html", result.text)
                parsed = parse_press_release(
                    result.text,
                    url=result.url,
                    keywords=filter_cfg["keywords"],
                    required_matches=filter_cfg["required_matches"],
                    case_sensitive=filter_cfg["case_sensitive"],
                )
                if not parsed.is_bitcoin_related:
                    continue
                records.append(
                    normalize_record(
                        parsed.to_dict(),
                        source=source["name"],
                        source_type=source["source_type"],
                        ingestion_method="web_scraping",
                        collection_mode="live_http",
                        http_status=result.status_code,
                        scraped_at=scraped_at,
                    )
                )
            except Exception as error:
                failures.append({"url": candidate.url, "stage": "article", "error": repr(error)})

    new_frame = records_to_frame(records)
    final_frame = merge_incremental(existing, new_frame) if use_incremental else new_frame
    save_outputs(
        final_frame,
        csv_path=paths["processed_csv"],
        parquet_path=paths["processed_parquet"],
        jsonl_path=paths["processed_jsonl"],
        quality_path=paths["quality_report_csv"],
        summary_path=paths["summary_json"],
        failures=failures,
        failed_urls_path=paths["failed_urls_csv"],
        collection_mode="live_http",
    )
    return final_frame


def import_verified_snapshot(config: Mapping[str, Any], snapshot_path: Path) -> pd.DataFrame:
    """Import verified official-page records when live network access is unavailable."""
    paths = _storage_paths(config)
    records: list[dict[str, Any]] = []
    with Path(snapshot_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    frame = records_to_frame(records)
    save_outputs(
        frame,
        csv_path=paths["processed_csv"],
        parquet_path=paths["processed_parquet"],
        jsonl_path=paths["processed_jsonl"],
        quality_path=paths["quality_report_csv"],
        summary_path=paths["summary_json"],
        failures=[],
        failed_urls_path=paths["failed_urls_csv"],
        collection_mode="verified_official_page_snapshot",
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Bitcoin-related SEC press releases.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scraper-root", type=Path, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--no-incremental", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=None, help="Import a verified JSONL snapshot instead of making HTTP requests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scraper_root = (args.scraper_root or find_scraper_root()).resolve()
    config_path = (args.config or scraper_root / "configs" / "scraper_config.yaml").resolve()
    config = resolve_config_paths(load_config(config_path), scraper_root)
    if args.snapshot is not None:
        frame = import_verified_snapshot(config, args.snapshot.resolve())
    else:
        frame = run_live_scrape(
            config,
            max_pages=args.max_pages,
            max_articles=args.max_articles,
            incremental=not args.no_incremental,
        )
    print(f"Saved {len(frame):,} Bitcoin-related SEC press releases.")
    print(config["storage"]["processed_csv"])


if __name__ == "__main__":
    main()
