import pytest

from src.http_client import DisallowedHostError, PoliteHttpClient


def test_http_client_rejects_non_sec_hosts_without_requesting() -> None:
    client = PoliteHttpClient(
        user_agent="FinancialNLPForCrypto/1.0 test@example.com",
        allowed_hosts=["www.sec.gov"],
        delay_seconds=0,
    )
    with pytest.raises(DisallowedHostError):
        client.get("https://example.com/article")
    client.close()
