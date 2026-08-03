"""Discover SEC press release article URLs from listing pages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


RELEASE_NUMBER_RE = re.compile(r"\b(?:19|20)\d{2}-\d{1,4}\b")
ARTICLE_PATH_RE = re.compile(
    r"/(?:newsroom/press-releases|news/press-release)/(?:19|20)\d{2}-\d+",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ListingCandidate:
    url: str
    title: str
    release_number: str | None = None
    listing_date_text: str | None = None
    listing_page: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_listing_params(*, query: str, page: int) -> dict[str, object]:
    """Build the SEC Drupal listing query parameters."""
    return {
        "combine": query,
        "page": int(page),
        "year": "All",
        "month": "All",
        "order": "field_publish_date",
        "sort": "desc",
    }


def _candidate_from_anchor(anchor, *, base_url: str, page: int) -> ListingCandidate | None:  # type: ignore[no-untyped-def]
    href = (anchor.get("href") or "").strip()
    if not href:
        return None
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    if not ARTICLE_PATH_RE.search(parsed.path):
        return None

    title = " ".join(anchor.get_text(" ", strip=True).split())
    if not title:
        return None

    container = anchor.find_parent(["tr", "li", "article", "div"])
    context = " ".join(container.get_text(" ", strip=True).split()) if container else title
    release_match = RELEASE_NUMBER_RE.search(context) or RELEASE_NUMBER_RE.search(parsed.path)
    release_number = release_match.group(0) if release_match else None

    date_text = None
    if container is not None:
        time_tag = container.find("time")
        if time_tag is not None:
            date_text = time_tag.get("datetime") or time_tag.get_text(" ", strip=True)
        elif container.name == "tr":
            cells = container.find_all(["td", "th"])
            if cells:
                date_text = " ".join(cells[0].get_text(" ", strip=True).split())

    return ListingCandidate(
        url=absolute,
        title=title,
        release_number=release_number,
        listing_date_text=date_text or None,
        listing_page=page,
    )


def parse_listing_page(html: str, *, base_url: str, page: int) -> list[ListingCandidate]:
    """Extract unique press-release candidates from a listing page."""
    soup = BeautifulSoup(html, "lxml")
    deduplicated: dict[str, ListingCandidate] = {}
    for anchor in soup.find_all("a", href=True):
        candidate = _candidate_from_anchor(anchor, base_url=base_url, page=page)
        if candidate is not None:
            deduplicated.setdefault(candidate.url, candidate)
    return list(deduplicated.values())


def deduplicate_candidates(candidates: Iterable[ListingCandidate]) -> list[ListingCandidate]:
    unique: dict[str, ListingCandidate] = {}
    for candidate in candidates:
        key = candidate.release_number or candidate.url
        unique.setdefault(key, candidate)
    return list(unique.values())
