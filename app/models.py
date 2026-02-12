"""Pydantic request and response models for the billing parser API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Simple health-check response."""

    status: str = "ok"


class ParseBillingRequest(BaseModel):
    """Request payload for parsing a hospital billing PDF."""

    file_url: Optional[str] = Field(
        default=None,
        description="Direct URL to a PDF file.",
        examples=["https://example.com/billing.pdf"],
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Original file name, forwarded from upstream automation.",
    )
    chat_id: Optional[str] = Field(
        default=None,
        description="Routing identifier from n8n/Telegram workflow.",
    )


class ParseBillingResponse(BaseModel):
    """Normalized API response for billing parsing results."""

    success: bool
    message: str
    nama: Optional[str] = None
    total_tagihan_raw: Optional[str] = None
    total_tagihan_int: Optional[int] = None
    chat_id: Optional[str] = None
    file_name: Optional[str] = None

