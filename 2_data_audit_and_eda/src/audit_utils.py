"""Data-audit helpers for the Financial NLP crypto project.

The functions in this module are intentionally small and notebook-friendly.
They do not write reports or modify source data. The notebook remains the
primary interface for Phase 2.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd


V1_EXPECTED_COLUMNS = [
    "URL",
    "Title",
    "Description",
    "Full Text",
    "Date Time",
    "Coin Type",
    "Filtered_Text",
    "sentiment_label",
    "sentiment_score",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Movement_OpenClose_%",
    "Movement_HighLow_%",
    "Market_Move",
]

# These columns are outcomes, annotations, or market variables and therefore
# should not be used as text features in the initial text-only model.
POTENTIAL_LEAKAGE_COLUMNS = [
    "sentiment_label",
    "sentiment_score",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Movement_OpenClose_%",
    "Movement_HighLow_%",
    "Market_Move",
]


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by locating ``1_data_acquisition``.

    Parameters
    ----------
    start:
        Directory from which the upward search begins. If omitted, the current
        working directory is used.

    Returns
    -------
    pathlib.Path
        The detected repository root.

    Raises
    ------
    FileNotFoundError
        If no parent directory contains ``1_data_acquisition``.
    """

    start_path = (start or Path.cwd()).resolve()
    candidates = [start_path, *start_path.parents]

    for candidate in candidates:
        if (candidate / "1_data_acquisition").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. Expected a parent directory "
        "containing '1_data_acquisition'."
    )


def resolve_historical_dataset(
    project_root: Path,
    filename: str,
) -> Path:
    """Resolve a historical CSV path using the project's current layout.

    The user's existing acquisition phase stores files in a folder named
    ``row``. A ``raw`` fallback is also supported in case the folder is renamed
    later to the more conventional data-engineering term.
    """

    historical_dir = project_root / "1_data_acquisition" / "historical"

    candidates = [
        historical_dir / "row" / filename,
        historical_dir / "raw" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the current-layout path so error messages remain actionable.
    return candidates[0]


def file_metadata(path: Path) -> dict[str, object]:
    """Return simple file-level metadata without reading the CSV contents."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    size_bytes = path.stat().st_size
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": int(size_bytes),
        "size_mb": size_bytes / (1024**2),
        "size_gb": size_bytes / (1024**3),
    }


def read_header(path: Path) -> list[str]:
    """Read only the CSV header and return column names."""

    return pd.read_csv(path, nrows=0).columns.tolist()


def load_sample(
    path: Path,
    nrows: int = 1_000,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a small sample for fast structural and visual inspection."""

    return pd.read_csv(
        path,
        nrows=nrows,
        usecols=usecols,
        low_memory=False,
    )


def schema_audit(
    actual_columns: Sequence[str],
    expected_columns: Sequence[str] = V1_EXPECTED_COLUMNS,
) -> pd.DataFrame:
    """Compare the observed schema with the documented V1 schema."""

    actual = list(actual_columns)
    expected = list(expected_columns)
    all_columns = list(dict.fromkeys([*expected, *actual]))

    rows = []
    for column in all_columns:
        rows.append(
            {
                "column": column,
                "expected": column in expected,
                "present": column in actual,
                "position_expected": expected.index(column) + 1 if column in expected else np.nan,
                "position_actual": actual.index(column) + 1 if column in actual else np.nan,
            }
        )

    return pd.DataFrame(rows)


def missingness_streaming(
    path: Path,
    chunksize: int = 5_000,
) -> pd.DataFrame:
    """Compute full-dataset missing-value counts using bounded memory."""

    columns = read_header(path)
    missing = pd.Series(0, index=columns, dtype="int64")
    total_rows = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        missing = missing.add(chunk.isna().sum(), fill_value=0).astype("int64")

    result = pd.DataFrame(
        {
            "column": columns,
            "missing_count": [int(missing[column]) for column in columns],
        }
    )
    result["missing_pct"] = (
        result["missing_count"] / total_rows * 100 if total_rows else 0.0
    )
    result["total_rows"] = total_rows

    return result.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def row_count_streaming(
    path: Path,
    chunksize: int = 50_000,
) -> int:
    """Count CSV rows without loading the entire dataset into memory."""

    first_column = read_header(path)[0]
    return sum(
        len(chunk)
        for chunk in pd.read_csv(
            path,
            usecols=[first_column],
            chunksize=chunksize,
            low_memory=False,
        )
    )


def value_counts_streaming(
    path: Path,
    column: str,
    chunksize: int = 20_000,
    dropna: bool = False,
) -> pd.DataFrame:
    """Compute exact value counts for one column across the full CSV."""

    counter: Counter[object] = Counter()

    for chunk in pd.read_csv(
        path,
        usecols=[column],
        chunksize=chunksize,
        low_memory=False,
    ):
        series = chunk[column]
        if dropna:
            series = series.dropna()
        else:
            series = series.fillna("<MISSING>")
        counter.update(series.tolist())

    return pd.DataFrame(
        counter.most_common(),
        columns=[column, "count"],
    )


def duplicate_key_audit(
    path: Path,
    columns: Sequence[str],
    chunksize: int = 20_000,
) -> dict[str, float | int | list[str]]:
    """Count duplicate records based on one or more identity columns.

    For news datasets, URL duplication is usually more informative than full-row
    duplication because market annotations or text-processing columns may differ
    even when two rows refer to the same underlying article.
    """

    seen: set[object] = set()
    duplicate_count = 0
    total_rows = 0

    for chunk in pd.read_csv(
        path,
        usecols=list(columns),
        chunksize=chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)

        if len(columns) == 1:
            keys: Iterable[object] = chunk[columns[0]].fillna("<MISSING>").astype(str)
        else:
            normalized = chunk[list(columns)].fillna("<MISSING>").astype(str)
            keys = map(tuple, normalized.itertuples(index=False, name=None))

        for key in keys:
            if key in seen:
                duplicate_count += 1
            else:
                seen.add(key)

    return {
        "columns": list(columns),
        "total_rows": total_rows,
        "unique_keys": len(seen),
        "duplicate_rows": duplicate_count,
        "duplicate_pct": (duplicate_count / total_rows * 100) if total_rows else 0.0,
    }


def datetime_audit(
    path: Path,
    column: str = "Date Time",
    chunksize: int = 20_000,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Audit timestamp parsing and compute year/month distributions."""

    total = 0
    missing = 0
    invalid = 0
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    year_counter: Counter[int] = Counter()
    month_counter: Counter[str] = Counter()

    for chunk in pd.read_csv(
        path,
        usecols=[column],
        chunksize=chunksize,
        low_memory=False,
    ):
        raw = chunk[column]
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)

        total += len(chunk)
        missing += int(raw.isna().sum())
        invalid += int((raw.notna() & parsed.isna()).sum())

        valid = parsed.dropna()
        if valid.empty:
            continue

        current_min = valid.min()
        current_max = valid.max()
        min_date = current_min if min_date is None else min(min_date, current_min)
        max_date = current_max if max_date is None else max(max_date, current_max)

        year_counter.update(valid.dt.year.astype(int).tolist())
        month_counter.update(valid.dt.strftime("%Y-%m").tolist())

    summary = {
        "column": column,
        "total_rows": total,
        "missing_count": missing,
        "invalid_non_missing_count": invalid,
        "parse_success_pct_non_missing": (
            (total - missing - invalid) / (total - missing) * 100
            if total > missing
            else np.nan
        ),
        "min_datetime_utc": min_date,
        "max_datetime_utc": max_date,
    }

    by_year = pd.DataFrame(
        sorted(year_counter.items()),
        columns=["year", "count"],
    )
    by_month = pd.DataFrame(
        sorted(month_counter.items()),
        columns=["month", "count"],
    )

    return summary, by_year, by_month


def source_domain_counts(
    path: Path,
    url_column: str = "URL",
    chunksize: int = 20_000,
) -> pd.DataFrame:
    """Count publisher/source domains extracted from article URLs."""

    counter: Counter[str] = Counter()

    for chunk in pd.read_csv(
        path,
        usecols=[url_column],
        chunksize=chunksize,
        low_memory=False,
    ):
        for value in chunk[url_column].dropna().astype(str):
            try:
                domain = urlparse(value).netloc.lower()
            except ValueError:
                domain = ""

            if domain.startswith("www."):
                domain = domain[4:]
            counter[domain or "<INVALID_OR_MISSING>"] += 1

    return pd.DataFrame(
        counter.most_common(),
        columns=["domain", "count"],
    )


def reservoir_sample_csv(
    path: Path,
    n: int = 3_000,
    columns: Sequence[str] | None = None,
    chunksize: int = 5_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a uniform random sample while streaming through the CSV.

    A random priority is generated for every row. At each step, only the ``n``
    rows with the smallest priorities are retained. This avoids loading the
    entire dataset while preventing the temporal/source bias of simply taking
    the first ``n`` rows.
    """

    rng = np.random.default_rng(random_state)
    retained: pd.DataFrame | None = None

    for chunk in pd.read_csv(
        path,
        usecols=list(columns) if columns is not None else None,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = chunk.copy()
        chunk["__sample_priority"] = rng.random(len(chunk))

        if retained is None:
            retained = chunk.nsmallest(min(n, len(chunk)), "__sample_priority")
        else:
            retained = pd.concat([retained, chunk], ignore_index=True)
            retained = retained.nsmallest(min(n, len(retained)), "__sample_priority")

    if retained is None:
        return pd.DataFrame(columns=columns)

    return retained.drop(columns="__sample_priority").reset_index(drop=True)


def financial_sanity_checks(
    path: Path,
    chunksize: int = 20_000,
) -> pd.DataFrame:
    """Run full-dataset sanity checks on OHLCV and one documented movement field.

    These checks diagnose data quality only. They do not redefine the project's
    future 1h/4h/24h target labels.
    """

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Movement_OpenClose_%",
    ]
    available = set(read_header(path))
    missing_required = [column for column in required if column not in available]

    if missing_required:
        return pd.DataFrame(
            [
                {
                    "check": "required_market_columns_present",
                    "violations": len(missing_required),
                    "rows_checked": 0,
                    "violation_pct": np.nan,
                    "notes": f"Missing columns: {missing_required}",
                }
            ]
        )

    counters = Counter()
    total = 0

    for chunk in pd.read_csv(
        path,
        usecols=required,
        chunksize=chunksize,
        low_memory=False,
    ):
        numeric = chunk.apply(pd.to_numeric, errors="coerce")
        total += len(numeric)

        valid_prices = numeric[["Open", "High", "Low", "Close"]].notna().all(axis=1)
        valid_ohlcv = valid_prices & numeric["Volume"].notna()

        counters["non_positive_price"] += int(
            (valid_prices & (numeric[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)).sum()
        )
        counters["negative_volume"] += int(
            (numeric["Volume"].notna() & (numeric["Volume"] < 0)).sum()
        )
        counters["high_below_ohlc"] += int(
            (
                valid_prices
                & (
                    numeric["High"]
                    < numeric[["Open", "Low", "Close"]].max(axis=1)
                )
            ).sum()
        )
        counters["low_above_ohlc"] += int(
            (
                valid_prices
                & (
                    numeric["Low"]
                    > numeric[["Open", "High", "Close"]].min(axis=1)
                )
            ).sum()
        )

        computed_move = (numeric["Close"] - numeric["Open"]) / numeric["Open"] * 100
        comparable = (
            numeric["Movement_OpenClose_%"].notna()
            & computed_move.notna()
            & np.isfinite(computed_move)
        )
        counters["open_close_movement_mismatch"] += int(
            (
                comparable
                & ~np.isclose(
                    numeric["Movement_OpenClose_%"],
                    computed_move,
                    rtol=1e-3,
                    atol=1e-3,
                )
            ).sum()
        )
        counters["valid_ohlcv_rows"] += int(valid_ohlcv.sum())
        counters["movement_rows_compared"] += int(comparable.sum())

    checks = [
        (
            "non_positive_price",
            counters["non_positive_price"],
            total,
            "Open/High/Low/Close should normally be positive.",
        ),
        (
            "negative_volume",
            counters["negative_volume"],
            total,
            "Volume should normally be non-negative.",
        ),
        (
            "high_below_ohlc",
            counters["high_below_ohlc"],
            counters["valid_ohlcv_rows"],
            "High should be >= Open, Low, and Close within a valid candle.",
        ),
        (
            "low_above_ohlc",
            counters["low_above_ohlc"],
            counters["valid_ohlcv_rows"],
            "Low should be <= Open, High, and Close within a valid candle.",
        ),
        (
            "open_close_movement_mismatch",
            counters["open_close_movement_mismatch"],
            counters["movement_rows_compared"],
            "Stored Movement_OpenClose_% is compared with (Close-Open)/Open*100.",
        ),
    ]

    rows = []
    for name, violations, denominator, notes in checks:
        rows.append(
            {
                "check": name,
                "violations": int(violations),
                "rows_checked": int(denominator),
                "violation_pct": (violations / denominator * 100) if denominator else np.nan,
                "notes": notes,
            }
        )

    return pd.DataFrame(rows)


def leakage_review(columns: Sequence[str]) -> pd.DataFrame:
    """Flag columns that must not enter the initial text-only feature matrix."""

    actual = set(columns)
    rows = []
    for column in POTENTIAL_LEAKAGE_COLUMNS:
        rows.append(
            {
                "column": column,
                "present": column in actual,
                "initial_text_only_feature": False,
                "reason": (
                    "Publisher-provided annotation or market/outcome information; "
                    "keep for audit/analysis only until the project's own labeling "
                    "strategy is defined."
                ),
            }
        )
    return pd.DataFrame(rows)
