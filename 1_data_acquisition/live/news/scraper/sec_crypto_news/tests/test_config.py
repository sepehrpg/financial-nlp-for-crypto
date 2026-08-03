from copy import deepcopy
from pathlib import Path

import pytest

from src.config import load_config, validate_config


def test_real_config_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "scraper_config.yaml"
    config = load_config(path)
    assert config["source"]["name"] == "sec.gov"
    assert "bitcoin" in config["bitcoin_filter"]["keywords"]
    assert config["http"]["delay_seconds"] >= 1.0


def test_invalid_keyword_config_is_rejected() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "scraper_config.yaml"
    config = load_config(path)
    invalid = deepcopy(config)
    invalid["bitcoin_filter"]["keywords"] = []
    with pytest.raises(ValueError, match="keywords"):
        validate_config(invalid)
