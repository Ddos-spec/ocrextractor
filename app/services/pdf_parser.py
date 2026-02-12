"""PDF text extraction and billing-field parsing helpers."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

import pdfplumber


class PDFTextExtractionError(Exception):
    """Raised when PDF text extraction fails."""


@dataclass(frozen=True)
class ParsedBillingFields:
    """Normalized billing fields extracted from PDF text."""

    nama: Optional[str]
    total_tagihan_raw: Optional[str]
    total_tagihan_int: Optional[int]


_NAME_STOP_KEYWORDS = (
    "TGL",
    "TAGIHAN",
    "JENIS",
    "KELAMIN",
    "NO",
    "REKAM",
    "MEDIS",
    "ALAMAT",
    "UMUR",
    "DOKTER",
    "PENJAMIN",
    "RUANG",
    "KELAS",
    "NIK",
    "DIAGNOSA",
    "RAWAT",
    "POLI",
    "RM",
    "TOTAL",
    "BIAYA",
    "RINCIAN",
)

_NAME_PATTERNS = (
    re.compile(
        r"(?is)\bNO\.?\s*REKAM\s*MEDIS\b.*?\bNAMA(?:\s+PASIEN)?\b\s*[:\-]?\s*(?!RS(?:UD)?\b|RUMAH\s+SAKIT\b)(.+?)(?=\b(?:TGL\.?\s*TAGIHAN|JENIS\s*KELAMIN|NO\.?\s*TAGIHAN|NO\.?\s*REKAM\s*MEDIS|ALAMAT|UMUR|DOKTER|PENJAMIN|RUANG|KELAS|NIK|DIAGNOSA|RAWAT|POLI)\b|$)"
    ),
    re.compile(
        r"(?is)\bNAMA\s+PASIEN\b\s*[:\-]?\s*(?!RS(?:UD)?\b|RUMAH\s+SAKIT\b)(.+?)(?=\b(?:TGL\.?\s*TAGIHAN|JENIS\s*KELAMIN|NO\.?\s*TAGIHAN|NO\.?\s*REKAM\s*MEDIS|ALAMAT|UMUR|DOKTER|PENJAMIN|RUANG|KELAS|NIK|DIAGNOSA|RAWAT|POLI)\b|$)"
    ),
    re.compile(
        r"(?is)\bNAMA\b\s*[:\-]?\s*(?!RS(?:UD)?\b|RUMAH\s+SAKIT\b)(.+?)(?=\b(?:TGL\.?\s*TAGIHAN|JENIS\s*KELAMIN|NO\.?\s*TAGIHAN|NO\.?\s*REKAM\s*MEDIS|ALAMAT|UMUR|DOKTER|PENJAMIN|RUANG|KELAS|NIK|DIAGNOSA|RAWAT|POLI)\b|$)"
    ),
)

_TOTAL_PATTERN = re.compile(
    r"(?is)\bTOTAL\s*TAGIHAN\b[\s:.\-]*(?:(?:R\s*P)|RUPIAH)?[\s:.\-]*([0-9][0-9.,\s]{0,40})"
)
_AMOUNT_TOKEN_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:[.,\s]\d{3})+(?:,\d{1,2})?|\d{5,})(?!\d)")

_NAME_BLOCKLIST_PHRASES = (
    "RUMAH SAKIT",
    "RSUD",
    "RS ",
    " INSTALASI ",
    " POLIKLINIK ",
    " PELAYANAN ",
)

OCR_ZOOM = 2.0


def _squash_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _parse_rupiah_amount(amount_token: str) -> Optional[int]:
    """Parse rupiah text into integer while tolerating separators and optional decimals."""
    compact = re.sub(r"\s+", "", amount_token)
    if not compact:
        return None

    if "," in compact:
        parts = compact.split(",")
        if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) <= 2:
            compact = "".join(parts[:-1])

    digits = re.sub(r"[^\d]", "", compact)
    if not digits:
        return None

    return int(digits)


def _clean_name_candidate(candidate: str) -> Optional[str]:
    """Normalize and sanitize a raw candidate name chunk."""
    compact = _squash_whitespace(candidate.strip(" \t\r\n:;,.|-"))
    if not compact:
        return None

    tokens = []
    for token in compact.split(" "):
        if not token:
            continue

        cleaned = re.sub(r"[^A-Za-z'.-]", "", token)
        if not cleaned:
            if tokens:
                break
            continue

        upper = cleaned.upper()
        if upper in _NAME_STOP_KEYWORDS:
            break

        tokens.append(upper)
        if len(tokens) >= 8:
            break

    if not tokens:
        return None

    return " ".join(tokens).strip()


def _is_probable_patient_name(name: str) -> bool:
    """Return True when extracted name is likely a patient name, not hospital metadata."""
    normalized = f" {_squash_whitespace(name).upper()} "
    if not normalized.strip():
        return False

    if re.search(r"\d", normalized):
        return False

    for phrase in _NAME_BLOCKLIST_PHRASES:
        if phrase in normalized:
            return False

    tokens = [token for token in normalized.split(" ") if token]
    if not tokens or len(tokens) > 6:
        return False

    alpha_chars = sum(1 for ch in normalized if "A" <= ch <= "Z")
    if alpha_chars < 2:
        return False

    meaningful_tokens = [token for token in tokens if len(token) >= 2]
    return bool(meaningful_tokens)


def extract_nama(text: str) -> Optional[str]:
    """Extract patient name from free-form billing text."""
    for pattern in _NAME_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(1)
            normalized = _clean_name_candidate(candidate)
            if normalized and _is_probable_patient_name(normalized):
                return normalized

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if not re.search(r"(?i)\bNAMA(?:\s+PASIEN)?\b", line):
            continue

        if re.search(r"(?i)\bNAMA\s+RS\b", line):
            continue

        after_label = re.split(r"(?i)\bNAMA(?:\s+PASIEN)?\b", line, maxsplit=1)[-1]
        candidates = [after_label]
        if not after_label.strip() and index + 1 < len(lines):
            candidates.append(lines[index + 1])

        for candidate in candidates:
            normalized = _clean_name_candidate(candidate)
            if normalized and _is_probable_patient_name(normalized):
                return normalized

    return None


def extract_total_tagihan(text: str) -> tuple[Optional[str], Optional[int]]:
    """Extract total billing phrase and numeric value in rupiah."""
    selected_raw: Optional[str] = None
    selected_value: Optional[int] = None

    for match in _TOTAL_PATTERN.finditer(text):
        amount_token = match.group(1)
        parsed_amount = _parse_rupiah_amount(amount_token)
        if parsed_amount is None:
            continue

        selected_raw = _squash_whitespace(match.group(0))
        selected_value = parsed_amount

    if selected_value is not None:
        return selected_raw, selected_value

    lines = [_squash_whitespace(line) for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if not re.search(r"(?i)\bTOTAL\b", line) or not re.search(r"(?i)\bTAGIHAN\b", line):
            continue

        amount_tokens = _AMOUNT_TOKEN_PATTERN.findall(line)
        raw_phrase = line
        if not amount_tokens and index + 1 < len(lines):
            next_line = lines[index + 1]
            amount_tokens = _AMOUNT_TOKEN_PATTERN.findall(next_line)
            raw_phrase = f"{line} {next_line}"

        for amount_token in amount_tokens:
            parsed_amount = _parse_rupiah_amount(amount_token)
            if parsed_amount is None:
                continue
            selected_raw = raw_phrase
            selected_value = parsed_amount

    return selected_raw, selected_value


def is_text_too_short(text: str, min_non_space_chars: int = 40) -> bool:
    """Return True when extracted text is likely empty/truncated."""
    cleaned = re.sub(r"\s+", "", text)
    return len(cleaned) < min_non_space_chars


def _extract_text_via_pdfplumber(pdf_bytes: bytes) -> str:
    """Extract machine-readable text from PDF pages using pdfplumber."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_texts = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_texts.append(page_text)
    except Exception as exc:
        raise PDFTextExtractionError(f"Tidak bisa membaca isi PDF: {exc}") from exc

    return "\n".join(page_texts).strip()


def _extract_text_via_ocr(pdf_bytes: bytes) -> str:
    """OCR fallback for image-based PDFs."""
    try:
        import fitz  # type: ignore[import-not-found]
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image
    except Exception:
        return ""

    page_texts = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page in pdf:
                pix = page.get_pixmap(matrix=fitz.Matrix(OCR_ZOOM, OCR_ZOOM), alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                ocr_text: Optional[str] = None
                for lang in ("ind+eng", "eng"):
                    try:
                        candidate = pytesseract.image_to_string(image, lang=lang)
                    except pytesseract.TesseractNotFoundError:
                        return ""
                    except pytesseract.TesseractError:
                        continue

                    if candidate and candidate.strip():
                        ocr_text = candidate
                        break

                if ocr_text:
                    page_texts.append(ocr_text)
    except Exception:
        return ""

    return "\n".join(page_texts).strip()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Read PDF text, then fallback to OCR for image-based files."""
    extraction_error: Optional[PDFTextExtractionError] = None
    text = ""
    try:
        text = _extract_text_via_pdfplumber(pdf_bytes)
    except PDFTextExtractionError as exc:
        extraction_error = exc

    if not is_text_too_short(text):
        return text

    ocr_text = _extract_text_via_ocr(pdf_bytes)
    if not ocr_text:
        if extraction_error is not None and not text:
            raise extraction_error
        return text

    if text:
        return f"{text}\n{ocr_text}".strip()
    return ocr_text


def parse_billing_text(text: str) -> ParsedBillingFields:
    """Parse billing text into normalized name and total fields."""
    nama = extract_nama(text)
    total_tagihan_raw, total_tagihan_int = extract_total_tagihan(text)

    return ParsedBillingFields(
        nama=nama,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
    )
