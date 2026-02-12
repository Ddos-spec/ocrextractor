"""PDF download helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


class DownloadError(Exception):
    """Raised when a remote file cannot be downloaded."""


class InvalidPDFError(DownloadError):
    """Raised when downloaded content is not recognized as a PDF."""


@dataclass(frozen=True)
class DownloadedFile:
    """Downloaded file payload."""

    content: bytes
    content_type: Optional[str]


def _looks_like_pdf(content_type: str, content: bytes) -> bool:
    """Check whether the response metadata/content indicate a PDF file."""
    normalized = (content_type or "").lower()
    if "application/pdf" in normalized:
        return True

    if normalized and "application/octet-stream" not in normalized and "pdf" not in normalized:
        return False

    return content.startswith(b"%PDF-")


async def download_pdf(file_url: str) -> DownloadedFile:
    """Download a PDF from URL with timeout and validation."""
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(file_url)
    except httpx.HTTPError as exc:
        raise DownloadError(f"Gagal download PDF: {exc}") from exc

    if response.status_code >= 400:
        raise DownloadError(f"Gagal download PDF: HTTP {response.status_code}.")

    content = response.content
    if not content:
        raise DownloadError("File PDF kosong.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise DownloadError("Ukuran file PDF terlalu besar (maksimal 20MB).")

    content_type = response.headers.get("content-type", "")
    if not _looks_like_pdf(content_type, content):
        raise InvalidPDFError("File bukan PDF atau content-type bukan PDF.")

    return DownloadedFile(content=content, content_type=content_type or None)

