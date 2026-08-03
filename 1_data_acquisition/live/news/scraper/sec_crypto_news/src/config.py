"""Configuration loading and path resolution for the SEC scraper."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def find_scraper_root(start: Path | None = None) -> Path:
    """Find the scraper root by locating ``configs/scraper_config.yaml``."""
    start_path = (start or Path.cwd()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "configs" / "scraper_config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate sec_crypto_news scraper root. Run inside the scraper folder "
        "or pass --scraper-root."
    )


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the scraper YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Scraper config must contain a mapping at its root.")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate fields required by discovery, HTTP, filtering, and storage."""
    for section in ("source", "bitcoin_filter", "http", "run", "storage"):
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"Missing or invalid config section: {section}")

    source = config["source"]
    for key in ("name", "source_type", "base_url", "listing_path", "search_query"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"source.{key} must be a non-empty string.")
    if not isinstance(source.get("start_page"), int) or source["start_page"] < 0:
        raise ValueError("source.start_page must be a non-negative integer.")
    if not isinstance(source.get("max_pages"), int) or source["max_pages"] <= 0:
        raise ValueError("source.max_pages must be a positive integer.")
    hosts = source.get("allowed_hosts")
    if not isinstance(hosts, list) or not hosts or not all(isinstance(x, str) for x in hosts):
        raise ValueError("source.allowed_hosts must contain at least one hostname.")

    filter_cfg = config["bitcoin_filter"]
    keywords = filter_cfg.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        raise ValueError("bitcoin_filter.keywords must be a non-empty list.")
    if not all(isinstance(keyword, str) and keyword.strip() for keyword in keywords):
        raise ValueError("Every bitcoin_filter keyword must be a non-empty string.")
    if not isinstance(filter_cfg.get("required_matches"), int) or filter_cfg["required_matches"] <= 0:
        raise ValueError("bitcoin_filter.required_matches must be positive.")

    http_cfg = config["http"]
    for key in ("user_agent", "accept", "accept_encoding"):
        if not isinstance(http_cfg.get(key), str) or not http_cfg[key].strip():
            raise ValueError(f"http.{key} must be a non-empty string.")
    for key in ("timeout_seconds", "delay_seconds", "backoff_factor"):
        if not isinstance(http_cfg.get(key), (int, float)) or http_cfg[key] < 0:
            raise ValueError(f"http.{key} must be a non-negative number.")
    if not isinstance(http_cfg.get("max_retries"), int) or http_cfg["max_retries"] < 0:
        raise ValueError("http.max_retries must be a non-negative integer.")

    storage = config["storage"]
    for key in (
        "raw_html_directory",
        "processed_csv",
        "processed_parquet",
        "processed_jsonl",
        "failed_urls_csv",
        "quality_report_csv",
        "summary_json",
        "state_json",
    ):
        if not isinstance(storage.get(key), str) or not storage[key].strip():
            raise ValueError(f"storage.{key} must be a non-empty string.")


def resolve_config_paths(config: Mapping[str, Any], scraper_root: Path) -> dict[str, Any]:
    """Return a deep copy with storage paths resolved against the scraper root."""
    resolved = deepcopy(dict(config))
    for key, value in resolved["storage"].items():
        path = Path(value)
        if not path.is_absolute():
            path = scraper_root / path
        resolved["storage"][key] = str(path.resolve())
    return resolved
