from copy import deepcopy
from pathlib import Path

import pytest

from src.transformer_features import (
    load_transformer_recipes,
    validate_transformer_recipes,
)


def test_transformer_recipe_loads(phase4_dir: Path) -> None:
    config = load_transformer_recipes(
        phase4_dir / "configs" / "transformer_recipes.yaml"
    )
    assert list(config["models"]) == ["bert", "finbert"]
    assert config["models"]["bert"]["pooling"] == "masked_mean"
    assert config["models"]["finbert"]["sentiment_output_order"] == [
        "positive",
        "neutral",
        "negative",
    ]
    assert config["reduction"]["pca"]["n_components"] > 0
    assert config["reduction"]["truncated_svd"]["n_components"] > 0


def test_transformer_recipe_rejects_invalid_pooling(phase4_dir: Path) -> None:
    config = load_transformer_recipes(
        phase4_dir / "configs" / "transformer_recipes.yaml"
    )
    invalid = deepcopy(config)
    invalid["models"]["bert"]["pooling"] = "average_everything"
    with pytest.raises(ValueError, match="pooling"):
        validate_transformer_recipes(invalid)


def test_transformer_recipe_rejects_invalid_sentiment_order(phase4_dir: Path) -> None:
    config = load_transformer_recipes(
        phase4_dir / "configs" / "transformer_recipes.yaml"
    )
    invalid = deepcopy(config)
    invalid["models"]["finbert"]["sentiment_output_order"] = [
        "negative",
        "neutral",
        "positive",
    ]
    with pytest.raises(ValueError, match="sentiment_output_order"):
        validate_transformer_recipes(invalid)


def test_transformer_recipe_rejects_max_length_above_bert_limit(phase4_dir: Path) -> None:
    config = load_transformer_recipes(
        phase4_dir / "configs" / "transformer_recipes.yaml"
    )
    invalid = deepcopy(config)
    invalid["models"]["bert"]["max_length"] = 1024
    with pytest.raises(ValueError, match="max_length"):
        validate_transformer_recipes(invalid)
