from pathlib import Path
import sys

import pandas as pd
import pytest


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from src.row_features import (  # noqa: E402
    audit_representations,
    build_row_level_features,
    load_phase3_dataset,
    validate_phase3_input,
)


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_id": [10, 11, 12],
            "text_title_description": [
                "BITCOIN ETF rises 12.5%! Price is $42,000.",
                "Interest rate falls by 2 percent?",
                None,
            ],
            "Filtered_Text": ["bitcoin etf rises", "rate falls", "   "],
            # These leakage-prone columns must have no effect on features.
            "sentiment_score": [0.9, -0.8, 1.0],
            "Close": [1, 999, -2],
            "future_return": [0.2, -0.5, 8.0],
        }
    )


def test_validation_reports_contract(sample_frame: pd.DataFrame) -> None:
    assert validate_phase3_input(sample_frame) == {
        "row_count": 3,
        "unique_source_row_ids": 3,
        "duplicated_source_row_id_rows": 0,
    }


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(), "missing required columns"),
        (
            pd.DataFrame(
                columns=["source_row_id", "text_title_description", "Filtered_Text"]
            ),
            "input is empty",
        ),
        (
            pd.DataFrame(
                {
                    "source_row_id": [1, 1],
                    "text_title_description": ["a", "b"],
                    "Filtered_Text": ["a", "b"],
                }
            ),
            "must be unique",
        ),
    ],
)
def test_validation_rejects_invalid_input(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_phase3_input(frame)


def test_representation_audit_handles_missing_and_whitespace(
    sample_frame: pd.DataFrame,
) -> None:
    audit = audit_representations(sample_frame)
    title = audit["representations"]["text_title_description"]
    filtered = audit["representations"]["Filtered_Text"]
    assert title["empty_count"] == 1
    assert title["character_count_max"] == 42
    assert title["word_count_max"] == 9
    assert filtered["empty_count"] == 1
    assert filtered["very_short_text_count"] == 2


def test_row_features_are_explainable_and_missing_safe(sample_frame: pd.DataFrame) -> None:
    features = build_row_level_features(sample_frame)
    first = features.iloc[0]
    assert list(features["source_row_id"]) == [10, 11, 12]
    assert first["digit_count"] == 8
    assert first["percentage_mention_count"] == 1
    assert first["currency_symbol_count"] == 1
    assert first["has_keyword_bitcoin"] == 1
    assert first["has_keyword_etf"] == 1
    assert first["exclamation_mark_count"] == 1
    assert features.iloc[2].drop(labels="source_row_id").sum() == 0


def test_unapproved_columns_cannot_be_feature_inputs(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unsupported text representation"):
        build_row_level_features(sample_frame, text_column="future_return")


def test_feature_values_ignore_non_text_columns(sample_frame: pd.DataFrame) -> None:
    baseline = build_row_level_features(sample_frame)
    changed = sample_frame.copy()
    changed[["sentiment_score", "Close", "future_return"]] = 123456
    pd.testing.assert_frame_equal(baseline, build_row_level_features(changed))


def test_missing_parquet_has_actionable_message(tmp_path: Path) -> None:
    missing = tmp_path / "cryptovision_v1_preprocessed.parquet"
    with pytest.raises(FileNotFoundError, match="Run Phase 3 or place"):
        load_phase3_dataset(missing)
