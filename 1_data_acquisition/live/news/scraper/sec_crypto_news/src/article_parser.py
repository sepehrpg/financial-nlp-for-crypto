"""Parse SEC press-release pages and identify Bitcoin-related documents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import re
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from dateutil import parser as date_parser


RELEASE_NUMBER_RE = re.compile(r"\b(?:19|20)\d{2}-\d{1,4}\b")
LAST_REVIEWED_RE = re.compile(r"^Last Reviewed or Updated:\s*(.+)$", re.IGNORECASE)
DATELINE_RE = re.compile(
    r"^(?:Washington\s*,?\s*D\.?C\.?,?\s*)?(.+?)(?:\s*[—–-]\s*)?$",
    re.IGNORECASE,
)


@dataclass
class ParsedArticle:
    title: str
    description: str
    full_text: str
    release_number: str
    published_at: str
    last_reviewed_at: str | None
    url: str
    canonical_url: str
    matched_keywords: list[str]
    bitcoin_keyword_count: int
    is_bitcoin_related: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def clean_text(value: str | None) -> str:
    """Normalize Unicode whitespace without removing financial punctuation."""
    if value is None:
        return ""
    return " ".join(value.replace("\xa0", " ").split())


def _main_container(soup: BeautifulSoup):  # type: ignore[no-untyped-def]
    return (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one("[role='main']")
        or soup.select_one(".main-content")
        or soup.body
        or soup
    )


def _line_sequence(container) -> list[str]:  # type: ignore[no-untyped-def]
    raw = [clean_text(text) for text in container.stripped_strings]
    lines: list[str] = []
    for text in raw:
        if not text:
            continue
        if not lines or text != lines[-1]:
            lines.append(text)
    return lines


def _parse_iso_date(value: str) -> str:
    parsed = date_parser.parse(value, fuzzy=True)
    return parsed.date().isoformat()


def _extract_published_at(lines: list[str], release_index: int) -> tuple[str, int]:
    iso_date_re = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
    for index in range(release_index + 1, min(len(lines), release_index + 8)):
        candidate = lines[index]
        if "Last Reviewed" in candidate:
            continue
        lowered = candidate.casefold()
        has_named_month = any(month in lowered for month in (
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ))
        has_iso_date = bool(iso_date_re.search(candidate))
        is_dateline = lowered.startswith("washington")
        if has_named_month or has_iso_date or is_dateline:
            cleaned = candidate.split(",", 1)[-1].strip() if is_dateline else candidate
            cleaned = cleaned.rstrip("—–- ")
            return _parse_iso_date(cleaned), index
    raise ValueError("Could not locate the press release publication date.")


def _keyword_hits(text: str, keywords: Iterable[str], *, case_sensitive: bool) -> tuple[list[str], int]:
    flags = 0 if case_sensitive else re.IGNORECASE
    hits: list[str] = []
    total = 0
    for keyword in keywords:
        escaped = re.escape(keyword.strip()).replace(r"\ ", r"\s+")
        if keyword.casefold() == "btc":
            pattern = re.compile(rf"\b{escaped}\b", flags=flags)
        else:
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", flags=flags)
        count = len(pattern.findall(text))
        if count:
            hits.append(keyword)
            total += count
    return hits, total


def parse_press_release(
    html: str,
    *,
    url: str,
    keywords: Iterable[str],
    required_matches: int = 1,
    case_sensitive: bool = False,
) -> ParsedArticle:
    """Parse one SEC page using semantic elements plus line-based fallbacks."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    container = _main_container(soup)
    title_node = container.find("h1") or soup.find("h1")
    title = clean_text(title_node.get_text(" ", strip=True) if title_node else "")
    if not title:
        raise ValueError("SEC article title was not found.")

    canonical_node = soup.find("link", rel=lambda value: value and "canonical" in value)
    canonical_url = urljoin(url, canonical_node.get("href")) if canonical_node else url

    lines = _line_sequence(container)
    try:
        title_index = lines.index(title)
    except ValueError:
        title_index = 0

    release_number = ""
    release_index = -1
    for index in range(title_index, min(len(lines), title_index + 15)):
        match = RELEASE_NUMBER_RE.fullmatch(lines[index]) or RELEASE_NUMBER_RE.search(lines[index])
        if match:
            release_number = match.group(0)
            release_index = index
            break
    if not release_number:
        url_match = RELEASE_NUMBER_RE.search(url)
        if not url_match:
            raise ValueError("SEC release number was not found.")
        release_number = url_match.group(0)
        release_index = title_index

    published_at, date_index = _extract_published_at(lines, release_index)

    description_parts: list[str] = []
    for value in lines[title_index + 1 : release_index]:
        if value.casefold() in {"press release", "for immediate release"}:
            continue
        if len(value) > 4:
            description_parts.append(value)
    description = clean_text(" ".join(description_parts))

    last_reviewed_at: str | None = None
    for value in lines:
        last_match = LAST_REVIEWED_RE.match(value)
        if last_match:
            last_reviewed_at = _parse_iso_date(last_match.group(1))
            break

    body_parts: list[str] = []
    for value in lines[date_index + 1 :]:
        last_match = LAST_REVIEWED_RE.match(value)
        if last_match:
            break
        if value in {"###", "Resources", "Return to top"}:
            if value == "###":
                break
            continue
        if value.casefold() in {
            "about the sec", "transparency", "websites", "site information",
            "stay connected. sign up for email updates.",
        }:
            break
        if value and value not in body_parts[-1:]:
            body_parts.append(value)

    full_text = clean_text(" ".join(body_parts))
    if not full_text:
        raise ValueError("SEC article body was empty after parsing.")

    searchable = clean_text(" ".join([title, description, full_text]))
    matched_keywords, bitcoin_keyword_count = _keyword_hits(
        searchable,
        keywords,
        case_sensitive=case_sensitive,
    )
    return ParsedArticle(
        title=title,
        description=description,
        full_text=full_text,
        release_number=release_number,
        published_at=published_at,
        last_reviewed_at=last_reviewed_at,
        url=url,
        canonical_url=canonical_url,
        matched_keywords=matched_keywords,
        bitcoin_keyword_count=bitcoin_keyword_count,
        is_bitcoin_related=len(matched_keywords) >= int(required_matches),
    )
