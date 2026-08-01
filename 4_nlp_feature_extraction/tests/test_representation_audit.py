import pandas as pd

from src.representation_audit import (
    build_representation_comparison,
    row_length_metrics,
)


def test_row_length_metrics_handles_null_and_empty() -> None:
    metrics = row_length_metrics(pd.Series([None, "  ", "BTC rises 5%"]))
    assert metrics["is_null"].tolist() == [1, 0, 0]
    assert metrics["is_empty"].tolist() == [1, 1, 0]
    assert metrics["word_count"].tolist() == [0, 0, 3]


def test_comparison_contains_requested_statistics(sample_frame: pd.DataFrame) -> None:
    result = build_representation_comparison(
        sample_frame,
        ["text_title_description", "Filtered_Text"],
        very_short_max_words=3,
        very_short_max_characters=20,
        percentile=0.95,
    )
    assert result["representation"].tolist() == [
        "text_title_description",
        "Filtered_Text",
    ]
    assert {
        "empty_count",
        "median_word_count",
        "p95_word_count",
        "max_word_count",
        "median_character_count",
        "p95_character_count",
        "max_character_count",
        "very_short_word_count",
    }.issubset(result.columns)
    assert result.loc[0, "empty_count"] == 1
    assert result.loc[1, "empty_count"] == 1
