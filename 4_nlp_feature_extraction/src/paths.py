"""Repository and artifact path helpers for Phase 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from the root, phase folder, or notebook folder."""

    start_path = (start or Path.cwd()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (
            (candidate / "1_data_acquisition").exists()
            and (candidate / "3_text_preprocessing").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root. Expected both "
        "'1_data_acquisition' and '3_text_preprocessing'."
    )


def resolve_project_path(project_root: Path, relative_path: str) -> Path:
    """Resolve a configured repository-relative path without allowing path escape."""

    root = project_root.resolve()
    resolved = (root / relative_path).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"Configured path escapes the repository: {relative_path}")
    return resolved


def resolve_pipeline_paths(
    project_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Path]:
    """Resolve all configured stage 1 input and output paths."""

    output_cfg = config["output"]
    return {
        "input": resolve_project_path(project_root, config["input"]["relative_path"]),
        "features": resolve_project_path(
            project_root, output_cfg["features_relative_path"]
        ),
        "comparison_csv": resolve_project_path(
            project_root, output_cfg["comparison_csv_relative_path"]
        ),
        "comparison_json": resolve_project_path(
            project_root, output_cfg["comparison_json_relative_path"]
        ),
        "contract_validation": resolve_project_path(
            project_root, output_cfg["contract_validation_relative_path"]
        ),
        "feature_validation": resolve_project_path(
            project_root, output_cfg["feature_validation_relative_path"]
        ),
        "metadata": resolve_project_path(
            project_root, output_cfg["metadata_relative_path"]
        ),
    }
