"""
Generic raw CSV dataset inspector.

Add dataset names and paths to the DATASETS list.
The script performs lightweight inspection without modifying raw files.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path("row")

SAMPLE_ROWS = 500


DATASETS = [
    {
        "name": "CryptoVision V1",
        "path": BASE_DIR / "cryptovision_v1.csv",
    },
    {
        "name": "CryptoVision V2",
        "path": BASE_DIR / "cryptovision_v2.csv",
    },

    # Future datasets:
    # {
    #     "name": "Another Dataset",
    #     "path": BASE_DIR / "another_dataset.csv",
    # },
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable file size."""

    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def inspect_dataset(
    name: str,
    path: Path,
) -> None:
    """
    Perform lightweight inspection of a CSV dataset.

    Parameters
    ----------
    name:
        Human-readable dataset name.

    path:
        Path to the CSV file.
    """

    print()
    print("=" * 80)
    print(f"Dataset: {name}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # File checks
    # -----------------------------------------------------------------

    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return

    if not path.is_file():
        print(f"[ERROR] Path is not a file: {path}")
        return

    file_size = path.stat().st_size

    if file_size == 0:
        print(f"[ERROR] File is empty: {path}")
        return

    print(f"Path:      {path}")
    print(f"File size: {format_size(file_size)}")

    # -----------------------------------------------------------------
    # Read sample
    # -----------------------------------------------------------------

    try:
        df = pd.read_csv(
            path,
            nrows=SAMPLE_ROWS,
            low_memory=False,
        )

    except pd.errors.EmptyDataError:
        print("[ERROR] CSV file contains no readable data.")
        return

    except pd.errors.ParserError as error:
        print("[ERROR] Failed to parse CSV file.")
        print(f"Reason: {error}")
        return

    except UnicodeDecodeError as error:
        print("[ERROR] Failed to decode CSV file.")
        print(f"Reason: {error}")
        return

    except Exception as error:
        print("[ERROR] Unexpected error while reading dataset.")
        print(f"Reason: {error}")
        return

    # -----------------------------------------------------------------
    # Basic information
    # -----------------------------------------------------------------

    print()
    print("Sample shape:")
    print(df.shape)

    print()
    print(f"Number of columns: {len(df.columns)}")

    print()
    print("Columns:")

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        print(f"{index:>3}. {column}")

    # -----------------------------------------------------------------
    # Data types
    # -----------------------------------------------------------------

    print()
    print("Data types:")

    print(df.dtypes)

    # -----------------------------------------------------------------
    # Missing values
    # -----------------------------------------------------------------

    print()
    print("Missing values in sample:")

    missing_values = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing_values)

    # -----------------------------------------------------------------
    # Unique values
    # -----------------------------------------------------------------

    print()
    print("Unique values in sample:")

    unique_values = (
        df.nunique(
            dropna=False
        )
        .sort_values()
    )

    print(unique_values)

    # -----------------------------------------------------------------
    # Duplicate rows
    # -----------------------------------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    duplicate_percentage = (
        duplicate_count
        / len(df)
        * 100
        if len(df) > 0
        else 0
    )

    print()
    print("Duplicate rows in sample:")

    print(
        f"{duplicate_count} "
        f"({duplicate_percentage:.2f}%)"
    )

    # -----------------------------------------------------------------
    # First rows
    # -----------------------------------------------------------------

    print()
    print("First 5 rows:")

    print(
        df.head()
        .to_string()
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print()
    print("-" * 80)
    print("Inspection summary")
    print("-" * 80)

    print(
        f"Sample rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print(
        f"Duplicate rows: {duplicate_count:,}"
    )

    print(
        f"Columns with nulls: "
        f"{int((missing_values > 0).sum()):,}"
    )

    print(
        f"File size: {format_size(file_size)}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Inspect all configured datasets."""

    if not DATASETS:
        print("No datasets configured.")
        return

    print(
        f"Found {len(DATASETS)} "
        f"dataset(s) to inspect."
    )

    for dataset in DATASETS:

        inspect_dataset(
            name=dataset["name"],
            path=dataset["path"],
        )

    print()
    print("=" * 80)
    print("All dataset inspections completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()