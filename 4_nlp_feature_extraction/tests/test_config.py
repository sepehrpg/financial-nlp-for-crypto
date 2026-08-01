from pathlib import Path

import pytest

from src.config import load_config


def test_default_config_is_row_count_tolerant() -> None:
    config = load_config()
    assert config["input"]["representations"] == [
        "text_title_description",
        "Filtered_Text",
    ]
    assert config["input"]["reference_row_count"] is None
    assert config["validation"]["fail_on_severity"] == ["critical"]
    assert config["validation"]["continue_with_available_representations"] is True


def test_optional_reference_row_count_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "input:\n  reference_row_count: 100\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["input"]["reference_row_count"] == 100


def test_invalid_yaml_root_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(TypeError, match="YAML root"):
        load_config(path)


def test_missing_representation_prefix_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "input:\n  representations: [new_representation]\n"
        "features:\n  representation_prefixes:\n    Filtered_Text: filtered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing feature prefixes"):
        load_config(path)
