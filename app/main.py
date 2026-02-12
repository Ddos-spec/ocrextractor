"""FastAPI entrypoint for billing PDF parser service."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Optional

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models import HealthResponse, ParseBillingRequest, ParseBillingResponse
from app.services.downloader import DownloadError, InvalidPDFError, download_pdf
from app.services.pdf_parser import (
    PDFTextExtractionError,
    ParsedBillingFields,
    is_text_too_short,
    parse_billing_text,
    extract_text_from_pdf,
)
from app.services.validation import is_valid_http_url

logger = logging.getLogger("billing_parser")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read integer env var with safe fallback and lower bound."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


OCR_CONCURRENCY = _env_int("OCR_CONCURRENCY", 1, minimum=1)
RESULT_CACHE_TTL_SECONDS = _env_int("RESULT_CACHE_TTL_SECONDS", 900, minimum=60)
RESULT_CACHE_MAX_ITEMS = _env_int("RESULT_CACHE_MAX_ITEMS", 256, minimum=16)

ocr_semaphore = asyncio.Semaphore(OCR_CONCURRENCY)
cache_lock = asyncio.Lock()
result_cache: dict[str, tuple[float, ParsedBillingFields]] = {}

app = FastAPI(
    title="Hospital Billing Parser API",
    version="1.0.0",
    description="Extract nama and total_tagihan from Indonesian hospital billing PDFs.",
)


def _build_response(
    *,
    success: bool,
    message: str,
    chat_id: Optional[str],
    file_name: Optional[str],
    nama: Optional[str] = None,
    total_tagihan_raw: Optional[str] = None,
    total_tagihan_int: Optional[int] = None,
    komponen_billing: Optional[dict[str, dict[str, object]]] = None,
) -> ParseBillingResponse:
    """Build a normalized response object."""
    return ParseBillingResponse(
        success=success,
        message=message,
        nama=nama,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
        komponen_billing=komponen_billing or {},
        chat_id=chat_id,
        file_name=file_name,
    )


def _bad_request(
    message: str,
    *,
    chat_id: Optional[str],
    file_name: Optional[str],
) -> JSONResponse:
    """Build HTTP 400 response while keeping stable JSON schema."""
    payload = _build_response(
        success=False,
        message=message,
        chat_id=chat_id,
        file_name=file_name,
    )
    return JSONResponse(status_code=400, content=payload.model_dump())


async def _cache_get(key: str) -> Optional[ParsedBillingFields]:
    """Fetch parsed result from in-memory cache if still valid."""
    now = time.monotonic()
    async with cache_lock:
        item = result_cache.get(key)
        if item is None:
            return None

        expires_at, value = item
        if expires_at <= now:
            result_cache.pop(key, None)
            return None
        return value


async def _cache_set(key: str, value: ParsedBillingFields) -> None:
    """Store parsed result in bounded in-memory cache."""
    now = time.monotonic()
    expires_at = now + RESULT_CACHE_TTL_SECONDS
    async with cache_lock:
        expired_keys = [cache_key for cache_key, (exp, _) in result_cache.items() if exp <= now]
        for expired_key in expired_keys:
            result_cache.pop(expired_key, None)

        if len(result_cache) >= RESULT_CACHE_MAX_ITEMS:
            oldest_key = min(result_cache, key=lambda cache_key: result_cache[cache_key][0])
            result_cache.pop(oldest_key, None)

        result_cache[key] = (expires_at, value)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Convert FastAPI default 422 validation responses into 400."""
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    payload = _build_response(
        success=False,
        message="Payload request tidak valid.",
        chat_id=None,
        file_name=None,
    )
    return JSONResponse(status_code=400, content=payload.model_dump())


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness/readiness probe endpoint."""
    return HealthResponse(status="ok")


@app.post("/parse-billing", response_model=ParseBillingResponse)
async def parse_billing(
    payload: Optional[ParseBillingRequest] = Body(default=None),
) -> ParseBillingResponse | JSONResponse:
    """
    Download a billing PDF and extract patient name + final total billing amount.
    """
    if payload is None:
        return _bad_request(
            "Body request wajib diisi.",
            chat_id=None,
            file_name=None,
        )

    chat_id = payload.chat_id
    file_name = payload.file_name
    file_url = (payload.file_url or "").strip()

    if not file_url:
        return _bad_request(
            "file_url wajib diisi.",
            chat_id=chat_id,
            file_name=file_name,
        )

    if not is_valid_http_url(file_url):
        return _bad_request(
            "file_url tidak valid. Gunakan URL http/https.",
            chat_id=chat_id,
            file_name=file_name,
        )

    try:
        downloaded = await download_pdf(file_url)
    except InvalidPDFError as exc:
        return _bad_request(str(exc), chat_id=chat_id, file_name=file_name)
    except DownloadError as exc:
        return _bad_request(str(exc), chat_id=chat_id, file_name=file_name)

    cache_key = hashlib.sha1(downloaded.content).hexdigest()
    parsed = await _cache_get(cache_key)
    if parsed is None:
        try:
            async with ocr_semaphore:
                text = await asyncio.to_thread(extract_text_from_pdf, downloaded.content)
        except PDFTextExtractionError as exc:
            logger.exception("PDF extraction failed: %s", exc)
            return _build_response(
                success=False,
                message=f"Gagal membaca isi PDF: {exc}",
                nama=None,
                total_tagihan_raw=None,
                total_tagihan_int=None,
                chat_id=chat_id,
                file_name=file_name,
            )
        except Exception:
            logger.exception("Unexpected exception while extracting PDF text")
            return _build_response(
                success=False,
                message="Terjadi kesalahan internal saat membaca PDF.",
                nama=None,
                total_tagihan_raw=None,
                total_tagihan_int=None,
                chat_id=chat_id,
                file_name=file_name,
            )

        if is_text_too_short(text):
            return _build_response(
                success=False,
                message="Teks PDF terlalu pendek atau tidak terbaca.",
                nama=None,
                total_tagihan_raw=None,
                total_tagihan_int=None,
                chat_id=chat_id,
                file_name=file_name,
            )

        try:
            parsed = await asyncio.to_thread(parse_billing_text, text)
        except Exception:
            logger.exception("Unexpected exception while parsing billing fields")
            return _build_response(
                success=False,
                message="Terjadi kesalahan internal saat parsing dokumen.",
                nama=None,
                total_tagihan_raw=None,
                total_tagihan_int=None,
                chat_id=chat_id,
                file_name=file_name,
            )

        await _cache_set(cache_key, parsed)

    has_component_data = any(
        bool(component.get("ditemukan"))
        for component in parsed.komponen_billing.values()
    )

    if parsed.nama is None and parsed.total_tagihan_int is None and not has_component_data:
        return _build_response(
            success=False,
            message="Gagal menemukan nama dan total tagihan di dokumen.",
            nama=None,
            total_tagihan_raw=None,
            total_tagihan_int=None,
            chat_id=chat_id,
            file_name=file_name,
        )

    if parsed.nama is not None and parsed.total_tagihan_int is not None:
        message = "Berhasil ekstrak billing."
    elif has_component_data:
        message = "Berhasil ekstrak sebagian data billing dan komponen."
    else:
        message = "Berhasil ekstrak sebagian data billing."

    return _build_response(
        success=True,
        message=message,
        nama=parsed.nama,
        total_tagihan_raw=parsed.total_tagihan_raw,
        total_tagihan_int=parsed.total_tagihan_int,
        komponen_billing=parsed.komponen_billing,
        chat_id=chat_id,
        file_name=file_name,
    )
