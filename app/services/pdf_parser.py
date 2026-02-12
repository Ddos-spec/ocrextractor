"""PDF text extraction and billing-field parsing helpers."""

from __future__ import annotations

from difflib import SequenceMatcher
import io
import os
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
    komponen_billing: dict[str, dict[str, object]]


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
_RUPIAH_INLINE_PATTERN = re.compile(r"(?i)\bRP\.?\s*([0-9][0-9.,\s]{0,30})")

_NAME_BLOCKLIST_PHRASES = (
    "RUMAH SAKIT",
    "RSUD",
    "RS ",
    " INSTALASI ",
    " POLIKLINIK ",
    " PELAYANAN ",
)

_NAME_TAIL_NOISE_EXACT = {
    "TOL",
    "TOI",
    "TGI",
    "T6L",
    "7GL",
    "N0",
}
_NAME_TAIL_FUZZY_TARGETS = (
    "TGL",
    "TAGIHAN",
    "NO",
    "NOMOR",
    "RM",
)

_COMPONENT_ALIASES: dict[str, tuple[str, ...]] = {
    "ruangan": ("RUANGAN", "KAMAR", "BED", "RAWAT INAP", "RAWAT JALAN", "IGD"),
    "pemeriksaan_dokter": (
        "PEMERIKSAAN DOKTER",
        "KONSULTASI DOKTER",
        "VISITE DOKTER",
        "TINDAKAN DOKTER",
        "DOKTER SPESIALIS",
        "KONSUL DOKTER",
    ),
    "asuhan_keperawatan": ("ASUHAN KEPERAWATAN", "TINDAKAN KEPERAWATAN", "JASA PERAWAT", "KEPERAWATAN"),
    "laboratorium": ("LABORATORIUM", "LAB ", "LABORAT"),
    "penunjang": ("PENUNJANG", "PENUNJANG MEDIK", "USG", "ECG", "EKG", "ECHO"),
    "sewa_alat": ("SEWA ALAT", "SEWAALAT", "SEWA ALKES"),
    "radiologi": ("RADIOLOGI", "RONTGEN", "X-RAY", "CT SCAN", "MRI"),
    "obat": ("OBAT", "FARMASI", "MEDIKASI", "APOTIK"),
    "bmhp": ("BMHP", "BHP", "BAHAN MEDIS HABIS PAKAI", "ALKES", "BAHAN HABIS PAKAI"),
}

_COMPONENT_LABELS: dict[str, str] = {
    "ruangan": "Ruangan",
    "pemeriksaan_dokter": "Pemeriksaan Dokter",
    "asuhan_keperawatan": "Asuhan Keperawatan",
    "laboratorium": "Laboratorium",
    "penunjang": "Penunjang",
    "sewa_alat": "Sewa Alat",
    "radiologi": "Radiologi",
    "obat": "Obat",
    "bmhp": "BMHP",
}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read integer env var with safe fallback and lower bound."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.5) -> float:
    """Read float env var with safe fallback and lower bound."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


OCR_ZOOM = _env_float("OCR_ZOOM", 1.35)
OCR_MAX_PAGES = _env_int("OCR_MAX_PAGES", 4, minimum=1)
OCR_PSM = _env_int("OCR_PSM", 6, minimum=3)
OCR_LANG_PRIMARY = os.getenv("OCR_LANG_PRIMARY", "ind+eng")
OCR_LANG_FALLBACK = os.getenv("OCR_LANG_FALLBACK", "eng")
OCR_EARLY_STOP_COMPONENT_HITS = _env_int("OCR_EARLY_STOP_COMPONENT_HITS", 3, minimum=1)


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

    # OCR often appends a broken label token (for example "TOL." from "TGL.").
    while tokens and _is_tail_noise_token(tokens[-1]):
        tokens.pop()
    if not tokens:
        return None

    return " ".join(tokens).strip()


def _is_tail_noise_token(token: str) -> bool:
    """Return True when a trailing name token likely comes from OCR label noise."""
    normalized = re.sub(r"[^A-Za-z]", "", token).upper()
    if not normalized:
        return True

    if normalized in _NAME_STOP_KEYWORDS or normalized in _NAME_TAIL_NOISE_EXACT:
        return True

    if len(normalized) <= 6:
        for target in _NAME_TAIL_FUZZY_TARGETS:
            if SequenceMatcher(None, normalized, target).ratio() >= 0.72:
                return True

    return False


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


def _extract_amount_from_line(line: str) -> Optional[int]:
    """Pick best rupiah amount candidate from a single OCR/text line."""
    rupiah_tokens = _RUPIAH_INLINE_PATTERN.findall(line)
    if rupiah_tokens:
        rupiah_values = []
        for token in rupiah_tokens:
            value = _parse_rupiah_amount(token)
            if value is not None:
                rupiah_values.append(value)
        if rupiah_values:
            return rupiah_values[-1]

    amount_tokens = _AMOUNT_TOKEN_PATTERN.findall(line)
    if not amount_tokens:
        return None

    parsed_values = []
    for token in amount_tokens:
        value = _parse_rupiah_amount(token)
        if value is not None:
            parsed_values.append(value)

    if not parsed_values:
        return None

    # For billing rows, the right-most amount is usually line total.
    return parsed_values[-1]


def extract_billing_components(text: str) -> dict[str, dict[str, object]]:
    """Extract requested billing components and optional nominal values."""
    results: dict[str, dict[str, object]] = {
        key: {
            "label": _COMPONENT_LABELS[key],
            "ditemukan": False,
            "nilai_raw": None,
            "nilai_int": None,
        }
        for key in _COMPONENT_ALIASES
    }

    lines = [_squash_whitespace(line) for line in text.splitlines() if line.strip()]
    upper_lines = [line.upper() for line in lines]

    current_section_key: Optional[str] = None

    for index, upper_line in enumerate(upper_lines):
        line = lines[index]

        matched_header_key: Optional[str] = None
        for key, aliases in _COMPONENT_ALIASES.items():
            if any(alias in upper_line for alias in aliases):
                matched_header_key = key
                break
        if matched_header_key is not None:
            current_section_key = matched_header_key

        if "JUMLAH" in upper_line and current_section_key is not None:
            amount_on_summary = _extract_amount_from_line(line)
            if amount_on_summary is not None:
                section_result = results[current_section_key]
                section_result["ditemukan"] = True
                section_result["nilai_raw"] = line
                section_result["nilai_int"] = amount_on_summary

        for key, aliases in _COMPONENT_ALIASES.items():
            if not any(alias in upper_line for alias in aliases):
                continue

            current = results[key]
            current["ditemukan"] = True

            raw_line = line
            amount_value = _extract_amount_from_line(line)
            if amount_value is None and index + 1 < len(lines):
                next_line = lines[index + 1]
                next_upper = upper_lines[index + 1]
                next_is_component_header = any(
                    alias in next_upper for aliases in _COMPONENT_ALIASES.values() for alias in aliases
                )
                next_amount = _extract_amount_from_line(next_line)
                if next_amount is not None and not next_is_component_header:
                    raw_line = f"{line} {next_line}"
                    amount_value = next_amount

            if amount_value is not None:
                previous_value = current["nilai_int"]
                if not isinstance(previous_value, int) or amount_value > previous_value:
                    current["nilai_int"] = amount_value
                    current["nilai_raw"] = raw_line
            elif current["nilai_raw"] is None:
                current["nilai_raw"] = raw_line

    return results


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

    def target_page_indices(page_count: int) -> list[int]:
        """Prioritize likely pages containing identity, totals, and components."""
        middle = page_count // 2
        candidates = [
            0,
            page_count - 1,
            middle,
            page_count - 2,
            1,
            middle - 1,
            middle + 1,
            page_count - 3,
            2,
        ]
        selected: list[int] = []
        for index in candidates:
            if index < 0 or index >= page_count:
                continue
            if index in selected:
                continue
            selected.append(index)
            if len(selected) >= min(OCR_MAX_PAGES, page_count):
                break
        return selected

    page_texts = []
    tesseract_config = f"--oem 1 --psm {OCR_PSM}"
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page_index in target_page_indices(len(pdf)):
                page = pdf.load_page(page_index)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(OCR_ZOOM, OCR_ZOOM),
                    colorspace=fitz.csGRAY,
                    alpha=False,
                )
                image = Image.frombytes("L", (pix.width, pix.height), pix.samples)

                ocr_text: Optional[str] = None
                for lang in (OCR_LANG_PRIMARY, OCR_LANG_FALLBACK):
                    try:
                        candidate = pytesseract.image_to_string(
                            image,
                            lang=lang,
                            config=tesseract_config,
                        )
                    except pytesseract.TesseractNotFoundError:
                        image.close()
                        return ""
                    except pytesseract.TesseractError:
                        continue

                    if candidate and candidate.strip():
                        ocr_text = candidate
                        break

                image.close()
                del pix

                if ocr_text:
                    page_texts.append(ocr_text)

                    # Stop early only when core fields and enough components are found.
                    parsed = parse_billing_text("\n".join(page_texts))
                    found_components = sum(
                        1
                        for component in parsed.komponen_billing.values()
                        if bool(component.get("ditemukan"))
                    )
                    if (
                        parsed.nama is not None
                        and parsed.total_tagihan_int is not None
                        and found_components >= OCR_EARLY_STOP_COMPONENT_HITS
                    ):
                        break
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
    komponen_billing = extract_billing_components(text)

    return ParsedBillingFields(
        nama=nama,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
        komponen_billing=komponen_billing,
    )
