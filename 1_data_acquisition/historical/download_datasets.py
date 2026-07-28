"""
Generic historical dataset downloader.

Add dataset download URLs and desired filenames to the DATASETS list.
All files are downloaded into: Current Directory


"""

from pathlib import Path

import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("row")

CHUNK_SIZE = 1024 * 1024  # 1 MB


DATASETS = [
    {
        "url": "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/df2217b2-1632-4cf3-8170-d278dd5c1f56",
        "filename": "cryptovision_v2.csv",
    },
    {
        "url": "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/b92a410d-9ef5-44af-b167-a5225487135b",
        "filename": "cryptovision_v1.csv",
    },

    # Future datasets:
    # {
    #     "url": "https://example.com/dataset.csv",
    #     "filename": "another_dataset.csv",
    # },
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    """
    Convert bytes to a human-readable file size.

    Examples:
        1024 -> 1.00 KB
        1048576 -> 1.00 MB
    """

    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def download_file(
    url: str,
    filename: str,
    output_dir: Path,
) -> Path:
    """
    Download a single file from a direct URL.

    The file is downloaded in chunks to avoid loading
    the entire dataset into memory.

    Parameters
    ----------
    url:
        Direct download URL.

    filename:
        Desired output filename.

    output_dir:
        Directory where the file will be saved.

    Returns
    -------
    Path
        Path to the downloaded file.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / filename

    # Skip files that have already been downloaded.
    if output_path.exists():
        print(f"[SKIP] File already exists: {output_path}")
        return output_path

    print("-" * 70)
    print(f"Downloading: {filename}")
    print(f"Source:      {url}")
    print(f"Destination: {output_path}")

    try:
        with requests.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=(10, 120),
        ) as response:

            response.raise_for_status()

            total_size = int(
                response.headers.get(
                    "content-length",
                    0,
                )
            )

            downloaded_size = 0

            if total_size:
                print(
                    f"File size:   "
                    f"{format_size(total_size)}"
                )

            print()

            with output_path.open("wb") as file:

                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):
                    if not chunk:
                        continue

                    file.write(chunk)

                    downloaded_size += len(chunk)

                    if total_size:
                        progress = (
                            downloaded_size
                            / total_size
                            * 100
                        )

                        print(
                            f"\rProgress: "
                            f"{progress:6.2f}% "
                            f"({format_size(downloaded_size)} "
                            f"/ {format_size(total_size)})",
                            end="",
                            flush=True,
                        )

                    else:
                        print(
                            f"\rDownloaded: "
                            f"{format_size(downloaded_size)}",
                            end="",
                            flush=True,
                        )

    except requests.RequestException as error:

        # Remove incomplete files if the download fails.
        if output_path.exists():
            output_path.unlink()

        print()
        print(f"[ERROR] Failed to download {filename}")
        print(f"Reason: {error}")

        raise

    print()
    print(f"[DONE] Saved: {output_path}")

    return output_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Download all datasets defined in the DATASETS list."""

    if not DATASETS:
        print("No datasets configured.")
        return

    print(
        f"Found {len(DATASETS)} dataset(s) "
        f"to download."
    )

    for index, dataset in enumerate(
        DATASETS,
        start=1,
    ):
        print()
        print(
            f"Dataset {index}/{len(DATASETS)}"
        )

        download_file(
            url=dataset["url"],
            filename=dataset["filename"],
            output_dir=OUTPUT_DIR,
        )

    print()
    print("=" * 70)
    print("All dataset downloads completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()