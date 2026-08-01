from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import run_stage1_pipeline


def _install_pickle_parquet_stand_in(monkeypatch) -> None:
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.read_pickle(path))
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False: self.to_pickle(path),
    )


def _write_config(project_root: Path, body: str = "") -> Path:
    path = project_root / "sample_config.yaml"
    path.write_text(
        "input:\n"
        "  reference_row_count: 100\n"
        "features:\n"
        "  keyword_groups:\n"
        "    crypto_assets: [bitcoin, btc, crypto]\n"
        "    regulation: [sec, approval]\n"
        "output:\n"
        "  features_relative_path: out/features.parquet\n"
        "  comparison_csv_relative_path: out/comparison.csv\n"
        "  comparison_json_relative_path: out/comparison.json\n"
        "  contract_validation_relative_path: out/contract_validation.csv\n"
        "  feature_validation_relative_path: out/feature_validation.csv\n"
        "  metadata_relative_path: out/metadata.json\n"
        + body,
        encoding="utf-8",
    )
    return path


def _write_input(project_root: Path, frame: pd.DataFrame) -> Path:
    input_path = (
        project_root
        / "3_text_preprocessing"
        / "data"
        / "processed"
        / "cryptovision_v1_preprocessed.parquet"
    )
    input_path.parent.mkdir(parents=True)
    frame.to_pickle(input_path)
    (project_root / "1_data_acquisition").mkdir()
    return input_path


def test_pipeline_completes_when_row_count_differs(
    tmp_path: Path,
    sample_frame: pd.DataFrame,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    _write_input(project_root, sample_frame)
    config_path = _write_config(project_root)
    _install_pickle_parquet_stand_in(monkeypatch)

    result = run_stage1_pipeline(
        project_root=project_root,
        config_path=config_path,
    )

    assert result["metadata"]["input_rows"] == 4
    assert result["metadata"]["status"] == "completed_with_warnings"
    assert result["paths"]["features"].is_file()
    assert result["paths"]["contract_validation"].is_file()
    assert result["paths"]["feature_validation"].is_file()

    contract = pd.read_csv(result["paths"]["contract_validation"])
    count_check = contract.loc[
        contract["check"].eq("row_count_vs_optional_reference")
    ].iloc[0]
    assert bool(count_check["blocks_pipeline"]) is False

    metadata = json.loads(result["paths"]["metadata"].read_text(encoding="utf-8"))
    assert metadata["input_rows"] == 4
    assert metadata["feature_rows"] == 4


def test_pipeline_uses_available_representation_when_one_is_missing(
    tmp_path: Path,
    sample_frame: pd.DataFrame,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    reduced = sample_frame.drop(columns="Filtered_Text")
    _write_input(project_root, reduced)
    config_path = _write_config(project_root)
    _install_pickle_parquet_stand_in(monkeypatch)

    result = run_stage1_pipeline(
        project_root=project_root,
        config_path=config_path,
    )

    assert result["metadata"]["used_representations"] == [
        "text_title_description"
    ]
    assert result["metadata"]["skipped_representations"] == ["Filtered_Text"]
    assert result["comparison"]["representation"].tolist() == [
        "text_title_description"
    ]


def test_pipeline_writes_report_before_blocking_on_missing_id(
    tmp_path: Path,
    sample_frame: pd.DataFrame,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    _write_input(project_root, sample_frame.drop(columns="source_row_id"))
    config_path = _write_config(project_root)
    _install_pickle_parquet_stand_in(monkeypatch)

    with pytest.raises(RuntimeError, match="critical input requirements"):
        run_stage1_pipeline(
            project_root=project_root,
            config_path=config_path,
        )

    assert (project_root / "out" / "contract_validation.csv").is_file()
    metadata_path = project_root / "out" / "metadata.json"
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "blocked"
