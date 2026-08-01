import pandas as pd
import pytest

from src.validation import (
    available_representations,
    validate_numeric_feature_output,
    validate_phase3_contract,
)


def test_changed_row_count_is_reported_but_not_blocking(
    sample_frame: pd.DataFrame,
) -> None:
    result = validate_phase3_contract(
        sample_frame,
        id_column="source_row_id",
        requested_representations=["text_title_description", "Filtered_Text"],
        reference_row_count=100,
    )
    row = result.report.set_index("check").loc["row_count_vs_optional_reference"]
    assert row["status"] == "issue"
    assert row["blocks_pipeline"] == False  # noqa: E712
    assert not result.has_blockers


def test_no_reference_count_means_runtime_recording_only(
    sample_frame: pd.DataFrame,
) -> None:
    result = validate_phase3_contract(
        sample_frame,
        id_column="source_row_id",
        requested_representations=["text_title_description", "Filtered_Text"],
    )
    checks = result.report["check"].tolist()
    assert "row_count_observed" in checks
    assert "row_count_vs_optional_reference" not in checks
    assert not result.has_blockers


def test_duplicate_identifier_is_warning_not_failure(
    sample_frame: pd.DataFrame,
) -> None:
    broken = sample_frame.copy()
    broken.loc[1, "source_row_id"] = broken.loc[0, "source_row_id"]
    result = validate_phase3_contract(
        broken,
        id_column="source_row_id",
        requested_representations=["text_title_description", "Filtered_Text"],
    )
    row = result.report.set_index("check").loc["id_duplicate_rows"]
    assert row["severity"] == "warning"
    assert row["blocks_pipeline"] == False  # noqa: E712
    result.raise_for_blockers()


def test_one_missing_representation_is_skipped_not_blocked(
    sample_frame: pd.DataFrame,
) -> None:
    reduced = sample_frame.drop(columns="Filtered_Text")
    result = validate_phase3_contract(
        reduced,
        id_column="source_row_id",
        requested_representations=["text_title_description", "Filtered_Text"],
    )
    assert not result.has_blockers
    assert available_representations(
        reduced, ["text_title_description", "Filtered_Text"]
    ) == ["text_title_description"]


def test_missing_all_representations_is_critical(
    sample_frame: pd.DataFrame,
) -> None:
    reduced = sample_frame.drop(
        columns=["text_title_description", "Filtered_Text"]
    )
    result = validate_phase3_contract(
        reduced,
        id_column="source_row_id",
        requested_representations=["text_title_description", "Filtered_Text"],
    )
    assert result.has_blockers
    with pytest.raises(RuntimeError, match="critical input requirements"):
        result.raise_for_blockers()


def test_missing_id_column_is_critical(sample_frame: pd.DataFrame) -> None:
    result = validate_phase3_contract(
        sample_frame.drop(columns="source_row_id"),
        id_column="source_row_id",
        requested_representations=["text_title_description", "Filtered_Text"],
    )
    assert result.has_blockers


def test_empty_dataset_is_critical(sample_frame: pd.DataFrame) -> None:
    result = validate_phase3_contract(
        sample_frame.iloc[0:0],
        id_column="source_row_id",
        requested_representations=["text_title_description", "Filtered_Text"],
    )
    assert result.has_blockers


def test_numeric_output_reports_duplicate_ids_without_blocking(
    sample_frame: pd.DataFrame,
) -> None:
    expected = sample_frame.copy()
    expected.loc[1, "source_row_id"] = expected.loc[0, "source_row_id"]
    features = pd.DataFrame(
        {
            "source_row_id": expected["source_row_id"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = validate_numeric_feature_output(
        features,
        id_column="source_row_id",
        expected_ids=expected["source_row_id"],
    )
    assert not result.has_blockers
    row = result.report.set_index("check").loc["feature_id_duplicate_rows"]
    assert row["severity"] == "warning"


def test_numeric_output_changed_order_is_critical(
    sample_frame: pd.DataFrame,
) -> None:
    features = pd.DataFrame(
        {
            "source_row_id": [5, 2, 9, 14],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = validate_numeric_feature_output(
        features,
        id_column="source_row_id",
        expected_ids=sample_frame["source_row_id"],
    )
    assert result.has_blockers
