from pathlib import Path

from src.article_parser import parse_press_release


KEYWORDS = ["bitcoin", "bitcoins", "btc", "spot bitcoin", "bitcoin etf"]


def test_article_parser_extracts_fields_and_bitcoin_relevance() -> None:
    html = (Path(__file__).parent / "fixtures" / "article_bitcoin.html").read_text(encoding="utf-8")
    article = parse_press_release(
        html,
        url="https://www.sec.gov/newsroom/press-releases/2024-13",
        keywords=KEYWORDS,
    )
    assert article.release_number == "2024-13"
    assert article.published_at == "2024-02-02"
    assert article.last_reviewed_at == "2024-02-02"
    assert "American Bitcoin Academy" in article.title
    assert article.is_bitcoin_related is True
    assert article.bitcoin_keyword_count >= 2
    assert len(article.full_text.split()) > 50


def test_article_without_bitcoin_is_rejected_by_filter() -> None:
    html = """
    <html><body><main><h1>SEC Announces Accounting Conference</h1>
    <p>For Immediate Release</p><p>2026-99</p>
    <p>Washington D.C., July 1, 2026 —</p>
    <p>The Commission announced a conference about accounting rules.</p>
    <p>###</p><p>Last Reviewed or Updated: July 1, 2026</p>
    </main></body></html>
    """
    article = parse_press_release(
        html,
        url="https://www.sec.gov/newsroom/press-releases/2026-99",
        keywords=KEYWORDS,
    )
    assert article.is_bitcoin_related is False
    assert article.bitcoin_keyword_count == 0
