"""Validation helpers."""

from __future__ import annotations

from urllib.parse import urlparse


def is_valid_http_url(value: str) -> bool:
    """Return True if value is a valid absolute HTTP/HTTPS URL."""
    if not value:
        return False

    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

