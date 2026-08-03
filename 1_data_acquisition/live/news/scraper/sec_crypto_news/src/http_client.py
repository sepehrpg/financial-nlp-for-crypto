"""Polite HTTP client for SEC.gov automated access."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DisallowedHostError(ValueError):
    """Raised when a URL does not belong to the configured SEC host allow-list."""


class SecAccessError(RuntimeError):
    """Raised when SEC blocks or rate-limits the scraper."""


@dataclass(frozen=True)
class HttpResult:
    url: str
    status_code: int
    text: str
    content_type: str
    elapsed_seconds: float


class PoliteHttpClient:
    """Requests session with host validation, retries, and a request delay."""

    def __init__(
        self,
        *,
        user_agent: str,
        allowed_hosts: Iterable[str],
        accept: str = "text/html,application/xhtml+xml",
        accept_encoding: str = "gzip, deflate",
        timeout_seconds: float = 30,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        verify_ssl: bool = True,
    ) -> None:
        self.allowed_hosts = {host.casefold() for host in allowed_hosts}
        self.timeout_seconds = float(timeout_seconds)
        self.delay_seconds = float(delay_seconds)
        self.verify_ssl = bool(verify_ssl)
        self._last_request_started = 0.0

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": accept,
                "Accept-Encoding": accept_encoding,
            }
        )
        retry = Retry(
            total=int(max_retries),
            connect=int(max_retries),
            read=int(max_retries),
            status=int(max_retries),
            backoff_factor=float(backoff_factor),
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _validate_host(self, url: str) -> None:
        hostname = (urlparse(url).hostname or "").casefold()
        if hostname not in self.allowed_hosts:
            raise DisallowedHostError(f"URL host is not allowed: {hostname!r}")

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_started
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def get(self, url: str, *, params: dict[str, object] | None = None) -> HttpResult:
        """GET one SEC page and fail safely on blocking or non-HTML responses."""
        self._validate_host(url)
        self._wait()
        started = time.perf_counter()
        self._last_request_started = time.monotonic()
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
            verify=self.verify_ssl,
            allow_redirects=True,
        )
        elapsed = time.perf_counter() - started

        if response.status_code in {403, 429}:
            raise SecAccessError(
                f"SEC returned HTTP {response.status_code}. Stop the run, verify the "
                "declared User-Agent, and reduce request frequency."
            )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type and content_type not in {"text/html", "application/xhtml+xml"}:
            raise SecAccessError(
                f"Unexpected content type for {response.url}: {content_type}"
            )
        return HttpResult(
            url=response.url,
            status_code=response.status_code,
            text=response.text,
            content_type=content_type,
            elapsed_seconds=elapsed,
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "PoliteHttpClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()
