"""Configuration loading and validation for Phase 4 stage 1.

The defaults are intentionally tolerant of normal dataset evolution. Row counts
are observed at runtime. An optional reference row count can be supplied for
reporting, but it never blocks execution by itself.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "relative_path": (
            "3_text_preprocessing/data/processed/"
            "cryptovision_v1_preprocessed.parquet"
        ),
        "id_column": "source_row_id",
        "representations": ["text_title_description", "Filtered_Text"],
        # Optional documentation-only baseline. Normal row-count drift is reported,
        # never treated as a contract failure.
        "reference_row_count": None,
    },
    "validation": {
        # Only checks marked as critical block the pipeline by default.
        "fail_on_severity": ["critical"],
        # If one requested representation is unavailable, use the others and report it.
        "continue_with_available_representations": True,
        # A large drift can be labeled warning instead of info, but remains non-blocking.
        "row_count_warning_ratio": 0.25,
    },
    "audit": {
        "very_short_max_words": 3,
        "very_short_max_characters": 20,
        "percentile": 0.95,
    },
    "features": {
        "representation_prefixes": {
            "text_title_description": "title_description",
            "Filtered_Text": "filtered_text",
        },
        "currency_symbols": ["$", "€", "£", "¥", "₹", "₽", "₩", "₿"],
        "keyword_groups": {
            "crypto_assets": [
                "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency"
            ],
            "market_direction": [
                "rally", "surge", "jump", "gain", "rise", "bullish",
                "drop", "fall", "plunge", "decline", "bearish"
            ],
            "regulation": [
                "sec", "regulator", "regulation", "approval", "ban", "lawsuit"
            ],
            "instruments": ["etf", "futures", "options", "spot"],
            "macro": [
                "inflation", "interest rate", "rate cut", "rate hike",
                "federal reserve", "fed"
            ],
            "risk_events": [
                "hack", "exploit", "liquidation", "bankruptcy", "fraud", "default"
            ],
        },
    },
    "output": {
        "features_relative_path": (
            "4_nlp_feature_extraction/data/row_features/"
            "row_level_text_features.parquet"
        ),
        "comparison_csv_relative_path": (
            "4_nlp_feature_extraction/data/reports/representation_comparison.csv"
        ),
        "comparison_json_relative_path": (
            "4_nlp_feature_extraction/data/reports/representation_comparison.json"
        ),
        "contract_validation_relative_path": (
            "4_nlp_feature_extraction/data/reports/phase3_contract_validation.csv"
        ),
        "feature_validation_relative_path": (
            "4_nlp_feature_extraction/data/reports/row_feature_validation.csv"
        ),
        "metadata_relative_path": (
            "4_nlp_feature_extraction/data/reports/stage1_run_metadata.json"
        ),
    },
}

_ALLOWED_SEVERITIES = {"info", "warning", "error", "critical"}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and merge it with tolerant defaults."""

    config = deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as file:
            user_config = yaml.safe_load(file) or {}
        if not isinstance(user_config, Mapping):
            raise TypeError("The YAML root must be a mapping.")
        config = _deep_merge(config, user_config)

    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate only settings required to interpret the recipe safely."""

    input_cfg = config.get("input", {})
    id_column = input_cfg.get("id_column")
    if not isinstance(id_column, str) or not id_column.strip():
        raise ValueError("input.id_column must be a non-empty string.")

    representations = input_cfg.get("representations", [])
    if not representations or not all(
        isinstance(item, str) and item.strip() for item in representations
    ):
        raise ValueError("input.representations must contain at least one column name.")
    if len(set(representations)) != len(representations):
        raise ValueError("input.representations must not contain duplicates.")

    reference_rows = input_cfg.get("reference_row_count")
    if reference_rows is not None and (
        not isinstance(reference_rows, int) or reference_rows <= 0
    ):
        raise ValueError("input.reference_row_count must be a positive integer or null.")

    validation_cfg = config.get("validation", {})
    fail_on = validation_cfg.get("fail_on_severity", ["critical"])
    if not isinstance(fail_on, list) or not fail_on:
        raise ValueError("validation.fail_on_severity must be a non-empty list.")
    unknown = sorted(set(fail_on) - _ALLOWED_SEVERITIES)
    if unknown:
        raise ValueError(f"Unknown validation severities: {unknown}")

    warning_ratio = float(validation_cfg.get("row_count_warning_ratio", 0.25))
    if warning_ratio < 0:
        raise ValueError("validation.row_count_warning_ratio must be non-negative.")

    audit_cfg = config.get("audit", {})
    percentile = float(audit_cfg.get("percentile", 0.95))
    if not 0 < percentile <= 1:
        raise ValueError("audit.percentile must be in the interval (0, 1].")

    for name in ("very_short_max_words", "very_short_max_characters"):
        value = audit_cfg.get(name)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"audit.{name} must be a non-negative integer.")

    prefixes = config.get("features", {}).get("representation_prefixes", {})
    if not isinstance(prefixes, Mapping):
        raise ValueError("features.representation_prefixes must be a mapping.")
    missing_prefixes = [name for name in representations if name not in prefixes]
    if missing_prefixes:
        raise ValueError(
            "Missing feature prefixes for representations: " f"{missing_prefixes}"
        )
    selected_prefixes = [str(prefixes[name]).strip() for name in representations]
    if any(not prefix for prefix in selected_prefixes):
        raise ValueError("Every requested representation needs a non-empty prefix.")
    if len(set(selected_prefixes)) != len(selected_prefixes):
        raise ValueError("Feature prefixes for requested representations must be unique.")

    currency_symbols = config.get("features", {}).get("currency_symbols", [])
    if not isinstance(currency_symbols, list) or not currency_symbols:
        raise ValueError("features.currency_symbols must be a non-empty list.")
    if not all(isinstance(symbol, str) and symbol for symbol in currency_symbols):
        raise ValueError("Every currency symbol must be a non-empty string.")

    keyword_groups = config.get("features", {}).get("keyword_groups", {})
    if not isinstance(keyword_groups, Mapping) or not keyword_groups:
        raise ValueError("features.keyword_groups must be a non-empty mapping.")
    for group, keywords in keyword_groups.items():
        if not isinstance(group, str) or not group.strip() or not keywords:
            raise ValueError("Every keyword group needs a name and keywords.")
        if not all(isinstance(keyword, str) and keyword.strip() for keyword in keywords):
            raise ValueError(f"Invalid keyword in group: {group}")
