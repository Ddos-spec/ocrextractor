"""FastAPI entrypoint for billing PDF parser service."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models import HealthResponse, ParseBillingRequest, ParseBillingResponse
from app.services.downloader import DownloadError, InvalidPDFError, download_pdf
from app.services.pdf_parser import (
    PDFTextExtractionError,
    is_text_too_short,
    parse_billing_text,
    extract_text_from_pdf,
)
from app.services.validation import is_valid_http_url

logger = logging.getLogger("billing_parser")

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
) -> ParseBillingResponse:
    """Build a normalized response object."""
    return ParseBillingResponse(
        success=success,
        message=message,
        nama=nama,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
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

    try:
        text = extract_text_from_pdf(downloaded.content)
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
        parsed = parse_billing_text(text)
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

    if parsed.nama is None and parsed.total_tagihan_int is None:
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
    else:
        message = "Berhasil ekstrak sebagian data billing."

    return _build_response(
        success=True,
        message=message,
        nama=parsed.nama,
        total_tagihan_raw=parsed.total_tagihan_raw,
        total_tagihan_int=parsed.total_tagihan_int,
        chat_id=chat_id,
        file_name=file_name,
    )
