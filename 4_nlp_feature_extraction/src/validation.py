"""Report-first validation for the Phase 3 to Phase 4 handoff.

Normal dataset evolution should be visible, not fatal. This module classifies
checks by severity and only blocks when continuing would make the requested
artifact impossible or structurally unsafe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


_SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


@dataclass(frozen=True)
class ValidationReport:
    """A tabular validation result with explicit blocking semantics."""

    report: pd.DataFrame
    fail_on_severity: tuple[str, ...] = ("critical",)

    @property
    def passed(self) -> bool:
        return bool(self.report["status"].eq("pass").all())

    @property
    def has_blockers(self) -> bool:
        return bool(self.report["blocks_pipeline"].fillna(False).any())

    @property
    def issue_count(self) -> int:
        return int(self.report["status"].ne("pass").sum())

    @property
    def warning_count(self) -> int:
        return int(self.report["severity"].eq("warning").sum())

    @property
    def critical_count(self) -> int:
        return int(self.report["severity"].eq("critical").sum())

    def raise_for_blockers(self) -> None:
        """Raise only when checks explicitly classified as blocking fail."""

        if not self.has_blockers:
            return
        failed = self.report.loc[
            self.report["blocks_pipeline"],
            ["check", "severity", "observed", "expected", "message"],
        ]
        raise RuntimeError(
            "Phase 4 cannot continue because critical input requirements failed. "
            "See the saved validation report.\n" + failed.to_string(index=False)
        )


def _row(
    *,
    check: str,
    passed: bool,
    severity_if_failed: str,
    observed: object,
    expected: object,
    message: str,
    fail_on_severity: Iterable[str],
) -> dict[str, object]:
    severity = "info" if passed else severity_if_failed
    return {
        "check": check,
        "status": "pass" if passed else "issue",
        "severity": severity,
        "blocks_pipeline": bool(not passed and severity in set(fail_on_severity)),
        "observed": observed,
        "expected": expected,
        "message": message,
    }


def validate_phase3_contract(
    df: pd.DataFrame,
    *,
    id_column: str,
    requested_representations: Sequence[str],
    reference_row_count: int | None = None,
    row_count_warning_ratio: float = 0.25,
    fail_on_severity: Sequence[str] = ("critical",),
) -> ValidationReport:
    """Inspect Phase 3 output without treating normal row-count drift as failure."""

    fail_on = tuple(fail_on_severity)
    rows: list[dict[str, object]] = []
    row_count = len(df)

    rows.append(
        _row(
            check="dataset_has_rows",
            passed=row_count > 0,
            severity_if_failed="critical",
            observed=row_count,
            expected="> 0",
            message=(
                "The dataset contains rows."
                if row_count > 0
                else "An empty dataset cannot produce representation statistics or features."
            ),
            fail_on_severity=fail_on,
        )
    )

    rows.append(
        _row(
            check="row_count_observed",
            passed=True,
            severity_if_failed="info",
            observed=row_count,
            expected="recorded at runtime",
            message="Row count is descriptive metadata, not a fixed contract.",
            fail_on_severity=fail_on,
        )
    )

    if reference_row_count is not None:
        delta = row_count - reference_row_count
        ratio = abs(delta) / reference_row_count
        severity = "warning" if ratio > row_count_warning_ratio else "info"
        rows.append(
            _row(
                check="row_count_vs_optional_reference",
                passed=delta == 0,
                severity_if_failed=severity,
                observed=f"{row_count} (delta {delta:+d}, {ratio:.2%})",
                expected=reference_row_count,
                message=(
                    "The current dataset differs from the optional reference. "
                    "This is reported for traceability and does not block execution."
                ),
                fail_on_severity=fail_on,
            )
        )

    id_present = id_column in df.columns
    rows.append(
        _row(
            check="id_column_present",
            passed=id_present,
            severity_if_failed="critical",
            observed="present" if id_present else "missing",
            expected=id_column,
            message=(
                "The identifier column is available."
                if id_present
                else "The feature output cannot preserve Phase 3 row provenance without the identifier."
            ),
            fail_on_severity=fail_on,
        )
    )

    available_representations = [
        name for name in requested_representations if name in df.columns
    ]
    for representation in requested_representations:
        present = representation in df.columns
        rows.append(
            _row(
                check=f"representation_present::{representation}",
                passed=present,
                severity_if_failed="warning",
                observed="present" if present else "missing",
                expected="present",
                message=(
                    "Representation is available."
                    if present
                    else "This representation will be skipped; other available representations can still run."
                ),
                fail_on_severity=fail_on,
            )
        )

    rows.append(
        _row(
            check="at_least_one_representation_available",
            passed=bool(available_representations),
            severity_if_failed="critical",
            observed=available_representations or "none",
            expected="at least one requested representation",
            message=(
                "At least one requested representation can be processed."
                if available_representations
                else "No requested text representation exists, so Phase 4 has no text input."
            ),
            fail_on_severity=fail_on,
        )
    )

    if id_present:
        id_series = df[id_column]
        missing_count = int(id_series.isna().sum())
        duplicate_rows = int(id_series.duplicated(keep=False).sum())
        integer_dtype = bool(pd.api.types.is_integer_dtype(id_series.dtype))
        monotonic = bool(id_series.is_monotonic_increasing)

        rows.extend(
            [
                _row(
                    check="id_missing_values",
                    passed=missing_count == 0,
                    severity_if_failed="warning",
                    observed=missing_count,
                    expected=0,
                    message=(
                        "No identifier values are missing."
                        if missing_count == 0
                        else "Missing identifiers are preserved and reported; review before downstream joins."
                    ),
                    fail_on_severity=fail_on,
                ),
                _row(
                    check="id_duplicate_rows",
                    passed=duplicate_rows == 0,
                    severity_if_failed="warning",
                    observed=duplicate_rows,
                    expected=0,
                    message=(
                        "Identifiers are unique."
                        if duplicate_rows == 0
                        else "Duplicate identifiers are preserved and reported; row-level features can still be generated."
                    ),
                    fail_on_severity=fail_on,
                ),
                _row(
                    check="id_integer_dtype",
                    passed=integer_dtype,
                    severity_if_failed="warning",
                    observed=str(id_series.dtype),
                    expected="integer-like identifier",
                    message=(
                        "Identifier dtype is integer."
                        if integer_dtype
                        else "The identifier dtype changed. It will be preserved rather than silently cast."
                    ),
                    fail_on_severity=fail_on,
                ),
                _row(
                    check="id_monotonic_order",
                    passed=monotonic,
                    severity_if_failed="info",
                    observed=monotonic,
                    expected=True,
                    message=(
                        "Rows are ordered by identifier."
                        if monotonic
                        else "Non-monotonic order is recorded; the current row order will be preserved."
                    ),
                    fail_on_severity=fail_on,
                ),
            ]
        )

    duplicate_subset = [
        column
        for column in [id_column, *available_representations]
        if column in df.columns
    ]
    try:
        duplicate_contract_rows = int(
            df.duplicated(subset=duplicate_subset, keep=False).sum()
        ) if row_count and duplicate_subset else 0
        duplicate_message = (
            "No duplicate contract rows were detected."
            if duplicate_contract_rows == 0
            else "Duplicate identifier/text rows are reported but not removed in Phase 4."
        )
    except TypeError as exc:
        duplicate_contract_rows = -1
        duplicate_message = (
            "Duplicate-row inspection could not be calculated for the current dtypes; "
            f"the pipeline continues. Details: {type(exc).__name__}: {exc}"
        )

    rows.append(
        _row(
            check="duplicate_contract_rows",
            passed=duplicate_contract_rows == 0,
            severity_if_failed=("info" if duplicate_contract_rows < 0 else "warning"),
            observed=(
                "not calculated" if duplicate_contract_rows < 0 else duplicate_contract_rows
            ),
            expected=0,
            message=duplicate_message,
            fail_on_severity=fail_on,
        )
    )

    return ValidationReport(pd.DataFrame(rows), fail_on)


def validate_numeric_feature_output(
    features: pd.DataFrame,
    *,
    id_column: str,
    expected_ids: pd.Series,
    fail_on_severity: Sequence[str] = ("critical",),
) -> ValidationReport:
    """Validate generated features and report ordinary identifier issues."""

    fail_on = tuple(fail_on_severity)
    rows: list[dict[str, object]] = []
    id_present = id_column in features.columns

    rows.append(
        _row(
            check="feature_id_column_present",
            passed=id_present,
            severity_if_failed="critical",
            observed="present" if id_present else "missing",
            expected=id_column,
            message="The generated artifact must retain the Phase 3 identifier.",
            fail_on_severity=fail_on,
        )
    )

    row_count_matches = len(features) == len(expected_ids)
    rows.append(
        _row(
            check="feature_row_count_matches_input",
            passed=row_count_matches,
            severity_if_failed="critical",
            observed=len(features),
            expected=len(expected_ids),
            message="Feature extraction must produce exactly one output row per input row.",
            fail_on_severity=fail_on,
        )
    )

    if id_present:
        actual_ids = features[id_column].reset_index(drop=True)
        expected = expected_ids.reset_index(drop=True)
        order_matches = actual_ids.equals(expected)
        rows.append(
            _row(
                check="feature_id_order_preserved",
                passed=order_matches,
                severity_if_failed="critical",
                observed=order_matches,
                expected=True,
                message="A changed identifier order would corrupt later joins.",
                fail_on_severity=fail_on,
            )
        )
        duplicate_rows = int(actual_ids.duplicated(keep=False).sum())
        rows.append(
            _row(
                check="feature_id_duplicate_rows",
                passed=duplicate_rows == 0,
                severity_if_failed="warning",
                observed=duplicate_rows,
                expected=0,
                message="Existing duplicate identifiers are reported but do not alter row-local feature computation.",
                fail_on_severity=fail_on,
            )
        )

    feature_columns = [column for column in features.columns if column != id_column]
    rows.append(
        _row(
            check="feature_columns_exist",
            passed=bool(feature_columns),
            severity_if_failed="critical",
            observed=len(feature_columns),
            expected="> 0",
            message="At least one numeric feature column is required.",
            fail_on_severity=fail_on,
        )
    )

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    rows.append(
        _row(
            check="feature_columns_numeric",
            passed=not non_numeric,
            severity_if_failed="critical",
            observed=non_numeric or "all numeric",
            expected="all feature columns numeric",
            message="Only the identifier may be non-feature metadata.",
            fail_on_severity=fail_on,
        )
    )

    if feature_columns:
        nan_count = int(features[feature_columns].isna().sum().sum())
        rows.append(
            _row(
                check="feature_nan_values",
                passed=nan_count == 0,
                severity_if_failed="critical",
                observed=nan_count,
                expected=0,
                message="NaN values would make the saved numeric artifact unreliable.",
                fail_on_severity=fail_on,
            )
        )

        finite = True
        if not non_numeric:
            values = features[feature_columns].to_numpy(dtype="float64", copy=False)
            finite = bool(np.isfinite(values).all())
        rows.append(
            _row(
                check="feature_values_finite",
                passed=finite,
                severity_if_failed="critical",
                observed=finite,
                expected=True,
                message="Infinite values are not valid row-level features.",
                fail_on_severity=fail_on,
            )
        )

    return ValidationReport(pd.DataFrame(rows), fail_on)


def available_representations(
    df: pd.DataFrame,
    requested_representations: Sequence[str],
) -> list[str]:
    """Return requested representations that actually exist, preserving order."""

    return [name for name in requested_representations if name in df.columns]
