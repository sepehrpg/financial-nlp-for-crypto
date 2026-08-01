"""Text-preprocessing helpers for Phase 3 of the Financial NLP crypto project.

The helpers in this module are intentionally conservative. They prepare a clean,
reproducible dataset without applying model-specific transformations such as
stemming, stopword removal, aggressive lowercasing, TF-IDF, or embeddings.
"""

from __future__ import annotations

from collections import Counter
from html import unescape
from pathlib import Path
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Iterable, Mapping, Sequence
from langdetect import DetectorFactory, LangDetectException, detect
import pandas as pd

# Make language detection reproducible.
DetectorFactory.seed = 0

# -----------------------------------------------------------------------------
# Project and schema configuration
# -----------------------------------------------------------------------------

PRIMARY_FILENAME = "cryptovision_v1.csv"
DEFAULT_OUTPUT_FILENAME = "cryptovision_v1_preprocessed.parquet"

# CryptoVision V1 uses spaces in these column names. Underscore aliases are kept
# only to make the helper resilient to earlier notebook assumptions or a future
# local rename. The raw file itself is never renamed or overwritten.
COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "url": ("URL",),
    "title": ("Title",),
    "description": ("Description",),
    "full_text": ("Full Text", "Full_Text"),
    "date_time": ("Date Time", "Date_Time"),
    "coin_type": ("Coin Type", "Coin_Type"),
    "filtered_text": ("Filtered_Text",),
    "sentiment_label": ("sentiment_label", "Sentiment_Label"),
    "sentiment_score": ("sentiment_score", "Sentiment_Score"),
}

CORE_SEMANTIC_COLUMNS = (
    "url",
    "title",
    "description",
    "full_text",
    "date_time",
    "coin_type",
    "filtered_text",
)

GENERATED_TEXT_COLUMNS = (
    "title_clean",
    "description_clean",
    "full_text_clean",
    "text_title",
    "text_title_description",
)

# | -------- | --------------------- |
# | `gclid`  | Google Ads            |
# | `fbclid` | Facebook/Meta         |
# | `mc_cid` | Mailchimp Campaign ID |
# | `mc_eid` | Mailchimp Email ID    |
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL) # To remove HTML comments.
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>") # To remove HTML tags.
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\[\]{}\"']+", flags=re.IGNORECASE) # To find a URL within the text.
WHITESPACE_RE = re.compile(r"\s+") # To equalize the spacing.
MULTI_LABEL_SPLIT_RE = re.compile(r"\s*(?:,|;|\||/|\s+&\s+)\s*") # To separate multiple labels in Coin Type.


# -----------------------------------------------------------------------------
# Repository and dataset paths
# -----------------------------------------------------------------------------

def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by locating ``1_data_acquisition``.

    The function works when Jupyter is launched from the repository root, the
    Phase 3 folder, or the notebook folder.
    """

    start_path = (start or Path.cwd()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "1_data_acquisition").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the repository root. Expected a parent directory "
        "containing '1_data_acquisition'."
    )


def resolve_historical_dataset(
    project_root: Path,
    filename: str = PRIMARY_FILENAME,
) -> Path:
    """Resolve the Phase 1 historical dataset without modifying it."""

    historical_dir = project_root / "1_data_acquisition" / "historical"
    candidates = [
        historical_dir / "row" / filename,
        historical_dir / "raw" / filename,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Searched:\n{searched}\n"
        "Run Phase 1 acquisition first."
    )


def default_processed_output_path(project_root: Path) -> Path:
    """Return the standard Phase 3 Parquet output path."""

    return (
        project_root
        / "3_text_preprocessing"
        / "data"
        / "processed"
        / DEFAULT_OUTPUT_FILENAME
    )


# -----------------------------------------------------------------------------
# Schema helpers
# -----------------------------------------------------------------------------

def resolve_schema(columns: Sequence[str]) -> dict[str, str | None]:
    """Map semantic field names to the actual columns present in a dataset."""

    available = set(columns)
    resolved: dict[str, str | None] = {}

    for semantic_name, candidates in COLUMN_ALIASES.items():
        resolved[semantic_name] = next(
            (candidate for candidate in candidates if candidate in available),
            None,
        )

    return resolved


def validate_core_schema(columns: Sequence[str]) -> dict[str, str]:
    """Validate the columns required by the current Phase 3 pipeline."""

    resolved = resolve_schema(columns)
    missing = [name for name in CORE_SEMANTIC_COLUMNS if resolved[name] is None]

    if missing:
        details = {
            name: COLUMN_ALIASES[name]
            for name in missing
        }
        raise ValueError(
            "The dataset does not contain the required Phase 3 fields. "
            f"Missing semantic fields and accepted aliases: {details}"
        )

    return {name: resolved[name] for name in CORE_SEMANTIC_COLUMNS}  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Conservative text cleaning
# -----------------------------------------------------------------------------

def clean_financial_text(value: object, *, replace_urls: bool = True) -> str:
    """Apply conservative, model-agnostic cleaning to one text value.

    The function intentionally preserves case, numbers, percentages, currency
    symbols, tickers, punctuation, and domain terminology such as BTC, ETF, SEC,
    and DeFi.

    Operations:
    - convert missing values to an empty string,
    - decode HTML entities,
    - remove HTML comments/tags,
    - normalize Unicode with NFKC,
    - remove zero-width formatting characters,
    - replace embedded URLs with ``<URL>`` by default,
    - normalize whitespace.
    """

    if value is None or pd.isna(value):
        return ""

    text = str(value)
    text = unescape(text)
    text = HTML_COMMENT_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\ufeff", "")

    if replace_urls:
        text = URL_RE.sub(_replace_embedded_url, text)

    text = WHITESPACE_RE.sub(" ", text).strip()
    return text

def _replace_embedded_url(match: re.Match[str]) -> str:
    """Replace a URL while keeping sentence punctuation outside the token."""

    raw = match.group(0)
    trailing = ""
    while raw and raw[-1] in ".,;:!?":
        trailing = raw[-1] + trailing
        raw = raw[:-1]
    return f"<URL>{trailing}"

def normalize_text_key(value: object) -> str:
    """Normalize text only for duplicate matching, not for model input."""

    cleaned = clean_financial_text(value, replace_urls=False)
    return cleaned.casefold()


def combine_text_fields(*values: object) -> str:
    """Join non-empty text fields with one space without changing content."""

    parts = [str(value).strip() for value in values if str(value).strip()]
    return " ".join(parts)


# -----------------------------------------------------------------------------
# URL normalization and duplicate identity
# -----------------------------------------------------------------------------

def canonicalize_article_url(value: object) -> str:
    """Create a conservative URL key for duplicate detection.

    Raw URLs are preserved in the output dataset. This key only normalizes the
    scheme/host case, removes URL fragments and common tracking parameters, and
    removes a non-root trailing slash.
    """

    if value is None or pd.isna(value):
        return ""

    raw = str(value).strip()
    if not raw:
        return ""

    candidate = raw if "://" in raw else f"https://{raw}"

    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return raw

    if not parsed.netloc:
        return raw

    filtered_query = []
    for key, val in parse_qsl(parsed.query, keep_blank_values=True):
        lower_key = key.casefold()
        if lower_key.startswith("utm_") or lower_key in TRACKING_QUERY_KEYS:
            continue
        filtered_query.append((key, val))

    path = parsed.path
    if path != "/":
        path = path.rstrip("/")

    normalized = urlunsplit(
        (
            parsed.scheme.casefold() or "https",
            parsed.netloc.casefold(),
            path,
            urlencode(filtered_query, doseq=True),
            "",
        )
    )
    return normalized


# -----------------------------------------------------------------------------
# Bitcoin selection based on observed Coin Type values
# -----------------------------------------------------------------------------

def audit_coin_type_values(
    csv_path: Path,
    coin_column: str,
    *,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Count actual Coin Type values in the raw CSV.

    By default ``chunksize=None`` reads the requested column in one pass with
    ``low_memory=False`` so pandas infers one consistent dtype for the full
    column. A positive ``chunksize`` can be supplied later when memory-bounded
    streaming is desired.
    """

    counter: Counter[str] = Counter()
    missing_count = 0

    if chunksize is None:
        frames = [
            pd.read_csv(
                csv_path,
                usecols=[coin_column],
                low_memory=False,
            )
        ]
    else:
        if chunksize <= 0:
            raise ValueError("chunksize must be a positive integer or None.")
        frames = pd.read_csv(
            csv_path,
            usecols=[coin_column],
            chunksize=chunksize,
        )

    for frame in frames:
        series = frame[coin_column]
        missing_count += int(series.isna().sum())
        for value, count in series.dropna().astype(str).value_counts().items():
            counter[str(value)] += int(count)

    rows = [
        {
            coin_column: value,
            "count": count,
            "is_explicit_bitcoin_label": coin_label_mentions_bitcoin(value),
        }
        for value, count in counter.most_common()
    ]

    if missing_count:
        rows.append(
            {
                coin_column: pd.NA,
                "count": missing_count,
                "is_explicit_bitcoin_label": False,
            }
        )

    return pd.DataFrame(rows)

def explicit_bitcoin_labels(
    coin_counts: pd.DataFrame,
    coin_column: str,
) -> list[str]:
    """Return the observed full Coin Type labels accepted by the strict rule."""

    if coin_counts.empty:
        return []

    mask = coin_counts["is_explicit_bitcoin_label"].fillna(False)
    return coin_counts.loc[mask, coin_column].dropna().astype(str).tolist()



def bitcoin_mask_from_labels(series: pd.Series, labels: Iterable[str]) -> pd.Series:
    """Select rows whose full Coin Type value is one of the audited BTC labels."""

    accepted = {str(label).strip().casefold() for label in labels}
    normalized = series.astype("string").str.strip().str.casefold()
    return normalized.isin(accepted).fillna(False)


def coin_label_mentions_bitcoin(value: object) -> bool:
    """Return True only for explicit Bitcoin/BTC labels.

    The rule avoids substring matching that would accidentally treat labels such
    as ``Bitcoin Cash`` or ``Bitcoin SV`` as native Bitcoin records. Multi-label
    values such as ``Bitcoin, Ethereum`` are accepted because one component is an
    explicit Bitcoin label.
    """

    if value is None or pd.isna(value):
        return False

    raw = str(value).strip()
    if not raw:
        return False

    pieces = MULTI_LABEL_SPLIT_RE.split(raw)
    accepted = {
        "bitcoin",
        "btc",
        "bitcoin (btc)",
        "btc (bitcoin)",
    }

    normalized_whole = WHITESPACE_RE.sub(" ", raw).strip().casefold()
    if normalized_whole in accepted:
        return True

    accepted_piece_re = re.compile(
        r"(?:bitcoin(?:\s*\(btc\))?|btc(?:\s*\(bitcoin\))?)",
        flags=re.IGNORECASE,
    )
    return any(
        accepted_piece_re.fullmatch(_normalize_label_piece(piece)) is not None
        for piece in pieces
    )


def _normalize_label_piece(value: str) -> str:
    value = value.strip().strip("[]{}\"'")
    value = WHITESPACE_RE.sub(" ", value)
    return value.casefold()



# -----------------------------------------------------------------------------
# Preprocessing and validation
# -----------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    schema: Mapping[str, str],
) -> pd.DataFrame:
    """Create Phase 3 text representations for an already selected dataframe."""

    result = df.copy()

    title_col = schema["title"]
    description_col = schema["description"]
    full_text_col = schema["full_text"]
    filtered_col = schema["filtered_text"]
    url_col = schema["url"]

    # Create cleaned text columns without modifying the original columns.
    result["title_clean"] = result[title_col].map(clean_financial_text)
    result["description_clean"] = result[description_col].map(clean_financial_text)
    result["full_text_clean"] = result[full_text_col].map(clean_financial_text)

    # General-purpose text representations.
    result["text_title"] = result["title_clean"]
    result["text_title_description"] = [
        combine_text_fields(title, description)
        for title, description in zip(
            result["title_clean"],
            result["description_clean"],
            strict=False,
        )
    ]

    # Internal keys used only during preprocessing/deduplication.
    result["_url_key"] = result[url_col].map(canonicalize_article_url)
    result["_title_key"] = result[title_col].map(normalize_text_key)
    result["_description_key"] = result[description_col].map(normalize_text_key)
    result["_filtered_nonempty"] = _nonempty_string_mask(result[filtered_col])

    return result


def _nonempty_string_mask(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().ne("")


def _duplicate_summary(df: pd.DataFrame) -> dict[str, int | float]:
    url_available = df["_url_key"].ne("")
    title_available = df["_title_key"].ne("")

    url_duplicate_rows = int(
        df.loc[url_available, "_url_key"].duplicated(keep="first").sum()
    )
    url_title_duplicate_rows = int(
        df.loc[url_available & title_available, ["_url_key", "_title_key"]]
        .duplicated(keep="first")
        .sum()
    )

    total = len(df)
    return {
        "url_duplicate_rows": url_duplicate_rows,
        "url_duplicate_pct": (url_duplicate_rows / total * 100) if total else 0.0,
        "url_title_duplicate_rows": url_title_duplicate_rows,
        "url_title_duplicate_pct": (
            url_title_duplicate_rows / total * 100 if total else 0.0
        ),
    }



def _dtype_signature(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, str]:
    """Return a readable dtype signature without modifying the dataframe."""

    return {column: str(df[column].dtype) for column in columns}


def _validate_original_dtypes_unchanged(
    processed: pd.DataFrame,
    expected_dtypes: Mapping[str, str],
) -> None:
    """Fail if preprocessing changed the dtype of any original column.

    This validator never casts data. It only reports a mismatch so dtype
    decisions remain explicit and outside Phase 3 text preprocessing.
    """

    actual = _dtype_signature(processed, list(expected_dtypes))
    mismatches = {
        column: {"before": expected_dtypes[column], "after": actual[column]}
        for column in expected_dtypes
        if actual[column] != expected_dtypes[column]
    }

    if mismatches:
        raise TypeError(
            "Original column dtypes changed during preprocessing. "
            f"No automatic dtype repair was applied. Mismatches: {mismatches}"
        )


def preprocess_csv_to_parquet(
    csv_path: Path,
    output_path: Path,
    *,
    bitcoin_labels: Sequence[str],
    chunksize: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the canonical Phase 3 processed dataset and write one Parquet file.

    Parameters
    ----------
    chunksize:
        ``None`` (default) reads the complete CSV once with ``low_memory=False``.
        This is the current recommended mode because pandas infers one dtype per
        column from the full V1 file before preprocessing. Supplying a positive
        integer enables the earlier chunked path for future memory-constrained
        experiments. No original column is explicitly cast in either mode.
    """

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    schema = validate_core_schema(header)
    coin_col = schema["coin_type"]

    if not bitcoin_labels:
        raise ValueError(
            "No explicit Bitcoin labels were provided. Audit Coin Type values first."
        )

    selected_frames: list[pd.DataFrame] = []
    raw_rows = 0
    bitcoin_rows = 0
    source_offset = 0
    original_columns = header
    expected_dtypes: dict[str, str] | None = None

    if chunksize is None:
        frames = [pd.read_csv(csv_path, low_memory=False)]
    else:
        if chunksize <= 0:
            raise ValueError("chunksize must be a positive integer or None.")
        frames = pd.read_csv(csv_path, chunksize=chunksize)

    for frame in frames:
        frame_dtype_signature = _dtype_signature(frame, original_columns)
        if expected_dtypes is None:
            expected_dtypes = frame_dtype_signature
        elif frame_dtype_signature != expected_dtypes:
            mismatches = {
                column: {
                    "first_chunk": expected_dtypes[column],
                    "current_chunk": frame_dtype_signature[column],
                }
                for column in original_columns
                if frame_dtype_signature[column] != expected_dtypes[column]
            }
            raise TypeError(
                "Chunked CSV reading inferred inconsistent dtypes across chunks. "
                "No dtype conversion was applied. Use chunksize=None for the "
                f"current stable path. Mismatches: {mismatches}"
            )

        frame_rows = len(frame)
        frame.insert(
            0,
            "source_row_id",
            range(source_offset, source_offset + frame_rows),
        )
        source_offset += frame_rows
        raw_rows += frame_rows

        selected = frame.loc[
            bitcoin_mask_from_labels(frame[coin_col], bitcoin_labels)
        ].copy()
        bitcoin_rows += len(selected)

        if selected.empty:
            continue

        selected = preprocess_dataframe(selected, schema)
        selected_frames.append(selected)

    if not selected_frames:
        raise ValueError(
            "The Bitcoin selection produced zero rows. Review the observed Coin Type values."
        )

    if len(selected_frames) == 1:
        processed = selected_frames[0].reset_index(drop=True)
    else:
        processed = pd.concat(selected_frames, ignore_index=True)

    title_or_description_available = _nonempty_string_mask(
        processed["text_title_description"]
    )
    full_text_available = _nonempty_string_mask(
        processed["full_text_clean"]
    )
    filtered_available = processed["_filtered_nonempty"].fillna(False)
    usable_text_mask = (
            title_or_description_available
            | full_text_available
            | filtered_available
    )

    missing_text_removed = int((~usable_text_mask).sum())
    processed = processed.loc[usable_text_mask].copy()

    # -------------------------------------------------------------------------
    # English-language filtering
    # -------------------------------------------------------------------------

    language_rows_audited = len(processed)

    processed, language_distribution = filter_english_text_rows(
        processed,
        primary_text_column="text_title_description",
        fallback_text_column="full_text_clean",
        progress_every=5_000,
    )

    english_rows_selected = len(processed)
    non_english_or_unknown_rows_removed = (
            language_rows_audited - english_rows_selected
    )

    # Deduplication must run after language filtering.
    duplicate_before = _duplicate_summary(processed)

    # Prefer the row with the richest available text when the same article URL
    # appears more than once. Source order is the deterministic tie-breaker.
    processed["_text_field_count"] = (
            _nonempty_string_mask(processed["title_clean"]).astype(int)
            + _nonempty_string_mask(processed["description_clean"]).astype(int)
            + _nonempty_string_mask(processed["full_text_clean"]).astype(int)
            + processed["_filtered_nonempty"].astype(int)
    )
    processed["_text_length_score"] = (
            processed["title_clean"].astype("string").fillna("").str.len()
            + processed["description_clean"].astype("string").fillna("").str.len()
            + processed["full_text_clean"].astype("string").fillna("").str.len()
            + processed[schema["filtered_text"]].astype("string").fillna("").str.len()
    )

    with_url = processed[processed["_url_key"].ne("")].copy()
    without_url = processed[processed["_url_key"].eq("")].copy()

    with_url = with_url.sort_values(
        ["_text_field_count", "_text_length_score", "source_row_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    before_url = len(with_url)
    with_url = with_url.drop_duplicates(subset=["_url_key"], keep="first")
    removed_duplicate_url = before_url - len(with_url)

    # A strict fallback is used only when URL is missing. We intentionally do not
    # deduplicate same-title stories that have different URLs because they may be
    # legitimate syndicated or follow-up reporting.
    fallback_key_available = (
        without_url["_title_key"].ne("")
        & without_url["_description_key"].ne("")
    )
    fallback_candidates = without_url.loc[fallback_key_available].copy()
    fallback_other = without_url.loc[~fallback_key_available].copy()

    fallback_candidates = fallback_candidates.sort_values(
        ["_text_field_count", "_text_length_score", "source_row_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    before_fallback = len(fallback_candidates)
    fallback_candidates = fallback_candidates.drop_duplicates(
        subset=["_title_key", "_description_key"],
        keep="first",
    )
    removed_missing_url_text_duplicates = before_fallback - len(fallback_candidates)

    processed = pd.concat(
        [with_url, fallback_candidates, fallback_other],
        ignore_index=True,
    ).sort_values("source_row_id", kind="stable")

    internal_columns = [
        "_url_key",
        "_title_key",
        "_description_key",
        "_filtered_nonempty",
        "_text_field_count",
        "_text_length_score",
    ]
    processed = processed.drop(columns=internal_columns).reset_index(drop=True)

    if expected_dtypes is None:
        raise RuntimeError("Could not determine the raw V1 dtype signature.")
    _validate_original_dtypes_unchanged(processed, expected_dtypes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        processed.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Writing Parquet requires pyarrow. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    summary: dict[str, object] = {
        "raw_rows": raw_rows,
        "bitcoin_rows_selected": bitcoin_rows,
        "non_bitcoin_rows_removed": raw_rows - bitcoin_rows,
        "missing_all_candidate_text_rows_removed": missing_text_removed,

        "language_rows_audited": language_rows_audited,
        "language_distribution_before_filter": language_distribution,
        "english_rows_selected": english_rows_selected,
        "non_english_or_unknown_rows_removed": non_english_or_unknown_rows_removed,

        "url_duplicate_rows_before_dedup": int(
            duplicate_before["url_duplicate_rows"]
        ),
        "url_duplicate_pct_before_dedup": float(
            duplicate_before["url_duplicate_pct"]
        ),
        "url_title_duplicate_rows_before_dedup": int(
            duplicate_before["url_title_duplicate_rows"]
        ),
        "url_title_duplicate_pct_before_dedup": float(
            duplicate_before["url_title_duplicate_pct"]
        ),
        "duplicate_url_rows_removed": int(removed_duplicate_url),
        "missing_url_exact_text_duplicates_removed": int(
            removed_missing_url_text_duplicates
        ),
        "output_rows": len(processed),
        "bitcoin_labels_used": list(bitcoin_labels),
    }

    return processed, summary


def validation_summary(
    processed: pd.DataFrame,
    *,
    url_column: str = "URL",
    filtered_text_column: str = "Filtered_Text",
) -> pd.DataFrame:
    """Return compact post-processing sanity checks."""

    url_keys = processed[url_column].map(canonicalize_article_url)
    nonempty_url = url_keys.ne("")

    checks = [
        ("rows", len(processed)),
        (
            "duplicate_normalized_url_rows",
            int(url_keys.loc[nonempty_url].duplicated(keep="first").sum()),
        ),
        (
            "empty_title_clean",
            int((~_nonempty_string_mask(processed["title_clean"])).sum()),
        ),
        (
            "empty_description_clean",
            int((~_nonempty_string_mask(processed["description_clean"])).sum()),
        ),
        (
            "empty_full_text_clean",
            int((~_nonempty_string_mask(processed["full_text_clean"])).sum()),
        ),
        (
            "empty_text_title",
            int((~_nonempty_string_mask(processed["text_title"])).sum()),
        ),
        (
            "empty_text_title_description",
            int((~_nonempty_string_mask(processed["text_title_description"])).sum()),
        ),
        (
            "empty_publisher_filtered_text",
            int((~_nonempty_string_mask(processed[filtered_text_column])).sum()),
        ),
        (
            "empty_all_text_candidates",
            int(
                (
                        ~_nonempty_string_mask(processed["text_title_description"])
                        & ~_nonempty_string_mask(processed["full_text_clean"])
                        & ~_nonempty_string_mask(processed[filtered_text_column])
                ).sum()
            ),
        ),
    ]

    return pd.DataFrame(checks, columns=["check", "value"])


def financial_text_smoke_test() -> pd.DataFrame:
    """Show that important financial information survives canonical cleaning."""

    examples = [
        "BTC jumps 5.2% to $67,500 after SEC ETF update.",
        "<p>ETH/BTC ratio rises; read more at https://example.com/a?utm_source=x</p>",
        "Fed says inflation is 3.1% — crypto markets remain volatile.",
    ]

    return pd.DataFrame(
        {
            "before": examples,
            "after": [clean_financial_text(value) for value in examples],
        }
    )


def detect_language_safe(value: object) -> str:
    """Return the detected ISO language code or 'unknown'."""

    if value is None or pd.isna(value):
        return "unknown"

    text = str(value).strip()

    if not text:
        return "unknown"

    # Limit the input before performing the character check.
    text = text[:500]

    alphabetic_count = sum(
        character.isalpha()
        for character in text
    )

    if alphabetic_count < 20:
        return "unknown"

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"




def filter_english_text_rows(
    df: pd.DataFrame,
    *,
    primary_text_column: str = "text_title_description",
    fallback_text_column: str = "full_text_clean",
    progress_every: int = 5_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep English rows without adding a language column."""

    required_columns = {
        primary_text_column,
        fallback_text_column,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Language-filtering columns are missing: "
            f"{sorted(missing_columns)}"
        )

    primary_text = (
        df[primary_text_column]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    fallback_text = (
        df[fallback_text_column]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    # Use the shorter Title + Description representation.
    # Full Text is used only for rows where it is unavailable.
    detection_text = primary_text.where(
        primary_text.ne(""),
        fallback_text,
    )

    detected_languages: list[str] = []
    total_rows = len(detection_text)

    for position, text in enumerate(detection_text, start=1):
        detected_languages.append(
            detect_language_safe(text)
        )

        if (
            progress_every > 0
            and (
                position % progress_every == 0
                or position == total_rows
            )
        ):
            print(
                "Language detection: "
                f"{position:,}/{total_rows:,} rows"
            )

    language_codes = pd.Series(
        detected_languages,
        index=df.index,
        dtype="string",
    )

    language_counts = {
        str(language): int(count)
        for language, count in language_codes.value_counts(
            dropna=False
        ).items()
    }

    english_mask = language_codes.eq("en")

    filtered = df.loc[english_mask].copy()

    return filtered, language_counts