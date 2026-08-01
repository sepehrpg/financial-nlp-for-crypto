"""Executable Stage 1 pipeline for representation audit and row features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .config import load_config
from .paths import find_project_root, resolve_pipeline_paths
from .representation_audit import build_representation_comparison
from .row_features import build_row_level_features
from .validation import (
    ValidationReport,
    available_representations,
    validate_numeric_feature_output,
    validate_phase3_contract,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _prepare_output_directories(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        if key != "input":
            path.parent.mkdir(parents=True, exist_ok=True)


def _validation_summary(report: ValidationReport) -> dict[str, int | bool]:
    return {
        "issue_count": report.issue_count,
        "warning_count": report.warning_count,
        "critical_count": report.critical_count,
        "has_blockers": report.has_blockers,
    }


def _write_blocked_metadata(
    *,
    path: Path,
    started: float,
    input_path: Path,
    input_rows: int | None,
    input_columns: int | None,
    reason: str,
    contract_validation: ValidationReport | None = None,
    feature_validation: ValidationReport | None = None,
) -> None:
    payload: dict[str, Any] = {
        "phase": "4_nlp_feature_extraction",
        "stage": "04_01_representation_audit_and_row_features",
        "status": "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_rows": input_rows,
        "input_columns": input_columns,
        "reason": reason,
        "elapsed_seconds": round(perf_counter() - started, 4),
    }
    if contract_validation is not None:
        payload["contract_validation"] = _validation_summary(contract_validation)
    if feature_validation is not None:
        payload["feature_validation"] = _validation_summary(feature_validation)
    _write_json(path, payload)


def run_stage1_pipeline(
    *,
    project_root: Path | None = None,
    config_path: Path | None = None,
    input_path: Path | None = None,
) -> dict[str, Any]:
    """Run Stage 1 with report-first validation and tolerant dataset handling."""

    started = perf_counter()
    root = (project_root or find_project_root()).resolve()
    config = load_config(config_path)
    paths = resolve_pipeline_paths(root, config)
    if input_path is not None:
        paths["input"] = Path(input_path).resolve()
    _prepare_output_directories(paths)

    if not paths["input"].is_file():
        _write_blocked_metadata(
            path=paths["metadata"],
            started=started,
            input_path=paths["input"],
            input_rows=None,
            input_columns=None,
            reason="Phase 3 Parquet output was not found.",
        )
        raise FileNotFoundError(
            "Phase 3 Parquet output was not found: "
            f"{paths['input']}\nRun Phase 3 first or pass input_path explicitly."
        )

    try:
        df = pd.read_parquet(paths["input"])
    except Exception as exc:  # Preserve the original exception as the root cause.
        _write_blocked_metadata(
            path=paths["metadata"],
            started=started,
            input_path=paths["input"],
            input_rows=None,
            input_columns=None,
            reason=f"Phase 3 Parquet could not be read: {type(exc).__name__}: {exc}",
        )
        raise RuntimeError(
            "Phase 3 Parquet could not be read. Install a Parquet engine and verify "
            f"the file is valid. Details: {type(exc).__name__}: {exc}"
        ) from exc

    input_cfg = config["input"]
    validation_cfg = config["validation"]
    fail_on = tuple(validation_cfg["fail_on_severity"])
    requested = list(input_cfg["representations"])

    contract_validation = validate_phase3_contract(
        df,
        id_column=input_cfg["id_column"],
        requested_representations=requested,
        reference_row_count=input_cfg.get("reference_row_count"),
        row_count_warning_ratio=float(validation_cfg["row_count_warning_ratio"]),
        fail_on_severity=fail_on,
    )
    contract_validation.report.to_csv(paths["contract_validation"], index=False)

    if contract_validation.has_blockers:
        _write_blocked_metadata(
            path=paths["metadata"],
            started=started,
            input_path=paths["input"],
            input_rows=len(df),
            input_columns=len(df.columns),
            reason="Critical Phase 3 input requirements failed.",
            contract_validation=contract_validation,
        )
        contract_validation.raise_for_blockers()

    usable_representations = available_representations(df, requested)
    if not validation_cfg.get("continue_with_available_representations", True):
        missing = [name for name in requested if name not in usable_representations]
        if missing:
            _write_blocked_metadata(
                path=paths["metadata"],
                started=started,
                input_path=paths["input"],
                input_rows=len(df),
                input_columns=len(df.columns),
                reason=(
                    "Requested representations are missing and tolerant fallback "
                    f"is disabled: {missing}"
                ),
                contract_validation=contract_validation,
            )
            raise RuntimeError(
                "Requested representations are missing and tolerant fallback is disabled: "
                f"{missing}"
            )

    audit_cfg = config["audit"]
    comparison = build_representation_comparison(
        df,
        usable_representations,
        very_short_max_words=int(audit_cfg["very_short_max_words"]),
        very_short_max_characters=int(audit_cfg["very_short_max_characters"]),
        percentile=float(audit_cfg["percentile"]),
    )

    feature_cfg = config["features"]
    row_features = build_row_level_features(
        df,
        id_column=input_cfg["id_column"],
        representations=usable_representations,
        representation_prefixes=feature_cfg["representation_prefixes"],
        currency_symbols=feature_cfg["currency_symbols"],
        keyword_groups=feature_cfg["keyword_groups"],
    )
    feature_validation = validate_numeric_feature_output(
        row_features,
        id_column=input_cfg["id_column"],
        expected_ids=df[input_cfg["id_column"]],
        fail_on_severity=fail_on,
    )
    feature_validation.report.to_csv(paths["feature_validation"], index=False)

    if feature_validation.has_blockers:
        _write_blocked_metadata(
            path=paths["metadata"],
            started=started,
            input_path=paths["input"],
            input_rows=len(df),
            input_columns=len(df.columns),
            reason="Generated feature artifact failed critical structural checks.",
            contract_validation=contract_validation,
            feature_validation=feature_validation,
        )
        feature_validation.raise_for_blockers()

    try:
        row_features.to_parquet(paths["features"], index=False)
    except Exception as exc:
        _write_blocked_metadata(
            path=paths["metadata"],
            started=started,
            input_path=paths["input"],
            input_rows=len(df),
            input_columns=len(df.columns),
            reason=f"Feature Parquet could not be written: {type(exc).__name__}: {exc}",
            contract_validation=contract_validation,
            feature_validation=feature_validation,
        )
        raise RuntimeError(
            "Row-level features could not be written as Parquet. Install pyarrow "
            f"and verify the output path. Details: {type(exc).__name__}: {exc}"
        ) from exc

    comparison.to_csv(paths["comparison_csv"], index=False)
    _write_json(paths["comparison_json"], comparison.to_dict(orient="records"))

    total_issues = contract_validation.issue_count + feature_validation.issue_count
    metadata = {
        "phase": "4_nlp_feature_extraction",
        "stage": "04_01_representation_audit_and_row_features",
        "status": "completed_with_warnings" if total_issues else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(paths["input"]),
        "input_rows": len(df),
        "input_columns": len(df.columns),
        "requested_representations": requested,
        "used_representations": usable_representations,
        "skipped_representations": [
            name for name in requested if name not in usable_representations
        ],
        "feature_rows": len(row_features),
        "feature_columns": len(row_features.columns),
        "contract_validation": _validation_summary(contract_validation),
        "feature_validation": _validation_summary(feature_validation),
        "elapsed_seconds": round(perf_counter() - started, 4),
        "outputs": {key: str(value) for key, value in paths.items() if key != "input"},
    }
    _write_json(paths["metadata"], metadata)

    return {
        "contract_validation": contract_validation.report,
        "feature_validation": feature_validation.report,
        "comparison": comparison,
        "features": row_features,
        "metadata": metadata,
        "paths": paths,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 Stage 1 representation audit and row features."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_stage1_pipeline(
        project_root=args.project_root,
        config_path=args.config,
        input_path=args.input,
    )
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
