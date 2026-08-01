from pathlib import Path

import pytest
import yaml

from src.tfidf_features import load_feature_recipes


def test_default_feature_recipes_load(phase4_dir: Path) -> None:
    config = load_feature_recipes(phase4_dir / "configs" / "feature_recipes.yaml")
    assert config["input"]["representations"] == [
        "text_title_description",
        "Filtered_Text",
    ]
    assert set(config["recipes"]) == {
        "word_tfidf",
        "character_tfidf",
        "word_character_tfidf",
    }
    assert config["recipes"]["word_tfidf"]["vectorizer"]["ngram_range"] == [1, 2]


def test_invalid_feature_recipe_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump({
            "input": {"id_column": "source_row_id", "representations": ["text"]},
            "smoke_test": {"sample_size": 10, "random_state": 42, "include_all_empty_rows": True},
            "recipes": {"bad": {"kind": "single", "vectorizer": {"analyzer": "word", "ngram_range": [2, 1]}}},
            "output": {},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ngram_range"):
        load_feature_recipes(path)
