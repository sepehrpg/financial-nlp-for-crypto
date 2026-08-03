from pathlib import Path

from src.discovery import build_listing_params, deduplicate_candidates, parse_listing_page


def test_listing_parser_extracts_press_release_links() -> None:
    html = (Path(__file__).parent / "fixtures" / "listing.html").read_text(encoding="utf-8")
    candidates = parse_listing_page(html, base_url="https://www.sec.gov", page=0)
    assert len(candidates) == 2
    assert candidates[0].release_number == "2025-101"
    assert candidates[0].url.startswith("https://www.sec.gov/newsroom/press-releases/")
    assert candidates[0].listing_date_text == "2025-07-29"


def test_listing_params_and_deduplication() -> None:
    params = build_listing_params(query="crypto", page=2)
    assert params["combine"] == "crypto"
    assert params["page"] == 2

    html = (Path(__file__).parent / "fixtures" / "listing.html").read_text(encoding="utf-8")
    candidates = parse_listing_page(html, base_url="https://www.sec.gov", page=0)
    assert len(deduplicate_candidates([*candidates, *candidates])) == 2
