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
    ocr_payload: dict[str, str]
    ai_field_analysis: dict[str, dict[str, object]]
    ai_bundle: dict[str, object]


_NAME_STOP_KEYWORDS = (
    "TGL",
    "TAGIHAN",
    "CARA",
    "BAYAR",
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
        r"(?is)\bNO\.?\s*REKAM\s*MEDIS\b.*?\bNAMA(?:\s+PASIEN)?\b\s*[:\-]?\s*(?!RS(?:UD)?\b|RUMAH\s+SAKIT\b)(.+?)(?=\b(?:TGL\.?\s*TAGIHAN|CARA\s*BAYAR|JENIS\s*KELAMIN|NO\.?\s*TAGIHAN|NO\.?\s*REKAM\s*MEDIS|ALAMAT|UMUR|DOKTER|PENJAMIN|RUANG|KELAS|NIK|DIAGNOSA|RAWAT|POLI)\b|$)"
    ),
    re.compile(
        r"(?is)\bNAMA\s+PASIEN\b\s*[:\-]?\s*(?!RS(?:UD)?\b|RUMAH\s+SAKIT\b)(.+?)(?=\b(?:TGL\.?\s*TAGIHAN|CARA\s*BAYAR|JENIS\s*KELAMIN|NO\.?\s*TAGIHAN|NO\.?\s*REKAM\s*MEDIS|ALAMAT|UMUR|DOKTER|PENJAMIN|RUANG|KELAS|NIK|DIAGNOSA|RAWAT|POLI)\b|$)"
    ),
    re.compile(
        r"(?is)\bNAMA\b\s*[:\-]?\s*(?!RS(?:UD)?\b|RUMAH\s+SAKIT\b)(.+?)(?=\b(?:TGL\.?\s*TAGIHAN|CARA\s*BAYAR|JENIS\s*KELAMIN|NO\.?\s*TAGIHAN|NO\.?\s*REKAM\s*MEDIS|ALAMAT|UMUR|DOKTER|PENJAMIN|RUANG|KELAS|NIK|DIAGNOSA|RAWAT|POLI)\b|$)"
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
_NAME_EXACT_BLOCKLIST = {
    "PASIEN",
    "KELUARGA PASIEN",
    "PASIEN KELUARGA PASIEN",
    "PASIEN KELUARGA",
    "PESERTA",
    "PESERTA MANDIRI",
    "PEKERJA MANDIRI",
    "PBI JAMINAN KESEHATAN",
}

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
    "prosedur_non_bedah": (
        "PROSEDUR NON BEDAH",
        "TINDAKAN NON BEDAH",
        "TINDAKAN NON-BEDAH",
        "KATETERISASI",
        "ENDOSKOPI",
        "BRONKOSKOPI",
    ),
    "prosedur_bedah": (
        "PROSEDUR BEDAH",
        "TINDAKAN BEDAH",
        "OPERASI",
        "KAMAR OPERASI",
        "PEMBEDAHAN",
    ),
    "konsultasi": (
        "KONSULTASI",
        "KONSUL",
        "VISITE",
        "PEMERIKSAAN DOKTER",
        "JASA DOKTER",
        "DOKTER SPESIALIS",
    ),
    "tenaga_ahli": (
        "TENAGA AHLI",
        "NUTRISIONIS",
        "FISIOTERAPIS",
        "DIETISIAN",
        "PSIKOLOG",
        "TERAPIS",
    ),
    "keperawatan": (
        "KEPERAWATAN",
        "ASUHAN KEPERAWATAN",
        "TINDAKAN KEPERAWATAN",
        "JASA PERAWAT",
        "PERAWAT",
    ),
    "penunjang": ("PENUNJANG", "PENUNJANG MEDIK", "EKG", "ECG", "ECHO", "HOLTER"),
    "radiologi": ("RADIOLOGI", "RONTGEN", "X-RAY", "USG", "MRI", "CT SCAN", "ANGIOGRAM"),
    "laboratorium": ("LABORATORIUM", "LABORAT", "HEMATOLOGI", "MIKROBIOLOGI", "PATOLOGI"),
    "pelayanan_darah": ("PELAYANAN DARAH", "TRANSFUSI", "PRC", "PACKED RED CELL"),
    "rehabilitasi": ("REHABILITASI", "FISIOTERAPI", "TERAPI OKUPASI", "REHAB"),
    "kamar_akomodasi": ("KAMAR", "AKOMODASI", "RAWAT INAP", "RUANG RANAP", "BED", "ADMISI"),
    "rawat_intensif": ("RAWAT INTENSIF", "ICU", "ICCU", "NICU", "PICU", "HCU"),
    "obat": ("OBAT", "FARMASI", "APOTIK", "MEDIKASI"),
    "obat_kronis": ("OBAT KRONIS", "KRONIS"),
    "obat_kemoterapi": ("OBAT KEMOTERAPI", "KEMOTERAPI", "SITOSTATIKA"),
    "alkes": ("ALKES", "ALAT KESEHATAN", "STENT", "IMPLAN"),
    "bmhp": ("BMHP", "BHP", "BAHAN MEDIS HABIS PAKAI", "BAHAN HABIS PAKAI", "OKSIGEN", "ALKOHOL", "JELLY"),
    "sewa_alat": ("SEWA ALAT", "SEWA ALKES", "VENTILATOR", "NEBULIZER", "POMPA SUNTIK", "INFUS PUMP"),
}

_COMPONENT_LABELS: dict[str, str] = {
    "prosedur_non_bedah": "Prosedur Non Bedah",
    "prosedur_bedah": "Prosedur Bedah",
    "konsultasi": "Konsultasi",
    "tenaga_ahli": "Tenaga Ahli",
    "keperawatan": "Keperawatan",
    "penunjang": "Penunjang",
    "radiologi": "Radiologi",
    "laboratorium": "Laboratorium",
    "pelayanan_darah": "Pelayanan Darah",
    "rehabilitasi": "Rehabilitasi",
    "kamar_akomodasi": "Kamar/Akomodasi",
    "rawat_intensif": "Rawat Intensif",
    "obat": "Obat",
    "obat_kronis": "Obat Kronis",
    "obat_kemoterapi": "Obat Kemoterapi",
    "alkes": "Alkes",
    "bmhp": "BMHP",
    "sewa_alat": "Sewa Alat",
}

COMPONENT_FIELD_KEYS = tuple(_COMPONENT_ALIASES.keys())

OCR_PAYLOAD_KEYS = (
    *COMPONENT_FIELD_KEYS,
    "billingan",
    "waktu_mulai",
    "waktu_selesai",
    "total",
    "koding",
    "waktu_mulai_koding",
    "waktu_selesai_koding",
    "total_koding",
    "rekap_billingan",
    "excel",
    "kasir",
    "balance",
    "link_e_klaim",
)

_EVIDENCE_MIN_SCORE: dict[str, int] = {
    "prosedur_non_bedah": 1,
    "prosedur_bedah": 1,
    "konsultasi": 1,
    "tenaga_ahli": 1,
    "keperawatan": 1,
    "penunjang": 1,
    "radiologi": 1,
    "laboratorium": 1,
    "pelayanan_darah": 1,
    "rehabilitasi": 1,
    "kamar_akomodasi": 1,
    "rawat_intensif": 1,
    "obat": 1,
    "obat_kronis": 1,
    "obat_kemoterapi": 1,
    "alkes": 1,
    "bmhp": 1,
    "sewa_alat": 1,
    "billingan": 2,
    "waktu_mulai": 2,
    "waktu_selesai": 2,
    "total": 3,
    "koding": 2,
    "waktu_mulai_koding": 2,
    "waktu_selesai_koding": 2,
    "total_koding": 2,
    "rekap_billingan": 2,
    "excel": 1,
    "kasir": 2,
    "balance": 2,
    "link_e_klaim": 2,
}

_EVIDENCE_MAX_ITEMS: dict[str, int] = {
    "billingan": 4,
    "total": 3,
    "koding": 4,
    "rekap_billingan": 4,
    "kasir": 3,
    "balance": 2,
    "waktu_mulai": 3,
    "waktu_selesai": 3,
    "waktu_mulai_koding": 2,
    "waktu_selesai_koding": 2,
    "total_koding": 3,
    "link_e_klaim": 2,
}

_GENERIC_NOISE_PATTERNS = (
    r"\bNO\.?\s*REKAM\b",
    r"\bNO\.?\s*TAGIHAN\b",
    r"\bNO\.?\s*JAMINAN\b",
    r"\bNO\.?\s*SEP\b",
    r"\bUMUR\s*HARI\b",
    r"\bUNIT\s*LAYANAN\b",
)


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


def _env_bool(name: str, default: bool) -> bool:
    """Read boolean env var with safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


OCR_ZOOM = _env_float("OCR_ZOOM", 1.35)
OCR_MAX_PAGES = _env_int("OCR_MAX_PAGES", 0, minimum=0)
OCR_PSM = _env_int("OCR_PSM", 6, minimum=3)
OCR_LANG_PRIMARY = os.getenv("OCR_LANG_PRIMARY", "ind+eng")
OCR_LANG_FALLBACK = os.getenv("OCR_LANG_FALLBACK", "eng")
AI_BUNDLE_TEXT_MAX_CHARS = _env_int("AI_BUNDLE_TEXT_MAX_CHARS", 80000, minimum=2000)
PAYLOAD_SNIPPET_MAX_CHARS = _env_int("PAYLOAD_SNIPPET_MAX_CHARS", 320, minimum=120)
PAYLOAD_MAX_PARTS_PER_KEY = _env_int("PAYLOAD_MAX_PARTS_PER_KEY", 30, minimum=5)
OCR_ENRICH_ALWAYS = _env_bool("OCR_ENRICH_ALWAYS", False)


def create_empty_ocr_payload() -> dict[str, str]:
    """Create normalized empty payload expected by downstream AI parser."""
    return {key: "" for key in OCR_PAYLOAD_KEYS}


def _truncate_for_bundle(text: str) -> tuple[str, bool]:
    """Truncate large OCR text for response payload safety."""
    if len(text) <= AI_BUNDLE_TEXT_MAX_CHARS:
        return text, False
    trimmed = text[:AI_BUNDLE_TEXT_MAX_CHARS].rstrip()
    return f"{trimmed}\n...[TRUNCATED]...", True


def _format_rupiah(value: int) -> str:
    """Format integer rupiah value using Indonesian thousands separators."""
    return f"Rp {value:,}".replace(",", ".")


def _split_evidence(payload_value: str, max_items: int = 10) -> list[str]:
    """Split `a | b | c` payload into unique evidence snippets."""
    if not payload_value:
        return []

    seen: set[str] = set()
    evidence: list[str] = []
    for chunk in payload_value.split(" | "):
        normalized = _squash_whitespace(chunk)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        evidence.append(normalized)
        if len(evidence) >= max_items:
            break
    return evidence


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


def _is_valid_total_candidate(amount_token: str, parsed_amount: int, context: str) -> bool:
    """Validate total billing candidate and reject common false positives."""
    if parsed_amount <= 0 or parsed_amount > 999_999_999:
        return False

    compact_token = re.sub(r"\s+", "", amount_token)
    digits_only = re.sub(r"[^\d]", "", compact_token)
    has_separator = bool(re.search(r"[.,]", compact_token))

    normalized_context = _squash_whitespace(context).upper()
    has_rupiah_hint = bool(
        re.search(r"\bR\s*P\b", normalized_context) or re.search(r"\bRUPIAH\b", normalized_context)
    )

    if "NO TAGIHAN" in normalized_context or "NO. TAGIHAN" in normalized_context:
        return False
    if "RAWAT JALAN [" in normalized_context or "RAWAT INAP [" in normalized_context:
        return False

    if not has_rupiah_hint and not has_separator:
        return False

    if not has_separator and len(digits_only) >= 9 and not has_rupiah_hint:
        return False

    return True


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

    if normalized.strip() in _NAME_EXACT_BLOCKLIST:
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

        candidate_raw = _squash_whitespace(match.group(0))
        if not _is_valid_total_candidate(amount_token, parsed_amount, candidate_raw):
            continue

        selected_raw = candidate_raw
        selected_value = parsed_amount

    if selected_value is not None:
        return selected_raw, selected_value

    lines = [_squash_whitespace(line) for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if not re.search(r"(?i)\bTOTAL\b", line) or not re.search(r"(?i)\bTAGIHAN\b", line):
            continue

        amount_tokens = _RUPIAH_INLINE_PATTERN.findall(line)
        if not amount_tokens:
            amount_tokens = _AMOUNT_TOKEN_PATTERN.findall(line)
        raw_phrase = line
        if not amount_tokens and index + 1 < len(lines):
            next_line = lines[index + 1]
            amount_tokens = _RUPIAH_INLINE_PATTERN.findall(next_line)
            if not amount_tokens:
                amount_tokens = _AMOUNT_TOKEN_PATTERN.findall(next_line)
            raw_phrase = f"{line} {next_line}"

        for amount_token in amount_tokens:
            parsed_amount = _parse_rupiah_amount(amount_token)
            if parsed_amount is None:
                continue
            if not _is_valid_total_candidate(amount_token, parsed_amount, raw_phrase):
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
            compact = re.sub(r"\s+", "", token)
            digits_only = re.sub(r"[^\d]", "", compact)
            has_separator = bool(re.search(r"[.,]", compact))
            upper_line = line.upper()
            has_rupiah_hint = bool(re.search(r"\bR\s*P\b", upper_line) or re.search(r"\bRUPIAH\b", upper_line))

            if value <= 0 or value > 500_000_000:
                continue
            if (
                not has_rupiah_hint
                and not has_separator
                and not re.search(r"\b(JUMLAH|TOTAL|SUBTOTAL)\b", upper_line)
            ):
                continue
            if not has_rupiah_hint and not has_separator and len(digits_only) >= 8:
                continue
            if (
                not has_rupiah_hint
                and re.search(r"\bNO\.?\s*(TAGIHAN|REKAM|SEP|RM)\b", upper_line)
            ):
                continue
            if (
                not has_rupiah_hint
                and re.search(r"\b(UMUR|TAHUN|HARI|TGL|TANGGAL|TELEPON|TELP|JAM MASUK|JAM KELUAR)\b", upper_line)
            ):
                continue

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


def _append_payload_text(payload: dict[str, str], key: str, value: str) -> None:
    """Append unique related text into payload field using ` | ` separator."""
    normalized = _squash_whitespace(value)
    if not normalized:
        return
    if len(normalized) > PAYLOAD_SNIPPET_MAX_CHARS:
        normalized = f"{normalized[:PAYLOAD_SNIPPET_MAX_CHARS].rstrip()}...[TRUNCATED]"

    current = payload.get(key, "")
    if not current:
        payload[key] = normalized
        return

    existing_parts = {part.strip() for part in current.split(" | ") if part.strip()}
    if len(existing_parts) >= PAYLOAD_MAX_PARTS_PER_KEY:
        return
    if normalized in existing_parts:
        return
    payload[key] = f"{current} | {normalized}"


def _payload_keyword_map() -> dict[str, tuple[str, ...]]:
    """Return broad keyword aliases for each OCR payload key."""
    component_map: dict[str, tuple[str, ...]] = {
        key: aliases for key, aliases in _COMPONENT_ALIASES.items()
    }
    component_map["kamar_akomodasi"] = component_map["kamar_akomodasi"] + ("KELAS", "RUANG PERAWATAN")
    component_map["laboratorium"] = component_map["laboratorium"] + ("DARAH", "ELEKTROLIT", "UREUM", "KREATININ")
    component_map["radiologi"] = component_map["radiologi"] + ("THORAX",)
    component_map["obat"] = component_map["obat"] + ("RESEP",)

    component_map.update(
        {
            "billingan": ("RINCIAN BIAYA", "TOTAL TAGIHAN", "TOTAL BAYAR", "SISA PEMBAYARAN", "TOTAL JAMINAN"),
            "waktu_mulai": ("WAKTU MULAI", "JAM MASUK", "TGL MASUK", "TANGGAL MASUK", "TGL. TAGIHAN"),
            "waktu_selesai": ("WAKTU SELESAI", "JAM KELUAR", "TGL KELUAR", "TANGGAL KELUAR"),
            "total": ("TOTAL TAGIHAN", "TOTAL TARIF", "TOTAL BIAYA", "TOTAL BAYAR"),
            "koding": ("KODING", "ICD", "INA-CBG", "GROUPING"),
            "waktu_mulai_koding": ("WAKTU MULAI KODING", "MULAI KODING"),
            "waktu_selesai_koding": ("WAKTU SELESAI KODING", "SELESAI KODING"),
            "total_koding": ("TOTAL KODING", "TOTAL INA-CBG", "HASIL GROUPING"),
            "rekap_billingan": ("REKAP", "PENJAMIN", "TOTAL JAMINAN", "SISA PEMBAYARAN"),
            "excel": ("EXCEL", "SPREADSHEET"),
            "kasir": ("KASIR", "PETUGAS KASIR", "TOTAL BAYAR", "SISA PEMBAYARAN", "TUNAI"),
            "balance": ("BALANCE", "SELISIH", "LUNAS", "SISA PEMBAYARAN"),
            "link_e_klaim": ("E-KLAIM", "EKLAIM", "KLAIM INDIVIDUAL"),
        }
    )
    return component_map


def _score_snippet_for_key(key: str, snippet: str) -> int:
    """Score how relevant a snippet is for a payload key."""
    normalized = _squash_whitespace(snippet)
    if not normalized:
        return 0

    upper = normalized.upper()
    score = 0
    keyword_map = _payload_keyword_map()
    keywords = keyword_map.get(key, ())

    for keyword in keywords:
        if keyword in upper:
            score += 2 if len(keyword) >= 6 else 1

    if key in COMPONENT_FIELD_KEYS:
        if re.search(r"\bRP\.?\s*\d", upper):
            score += 1

    if key in {"total", "billingan", "rekap_billingan", "kasir", "balance"}:
        if re.search(r"\bTOTAL\s*TAGIHAN\b", upper):
            score += 3
        if re.search(r"\bRP\.?\s*\d", upper):
            score += 2
        if "SISA PEMBAYARAN" in upper or "TOTAL BAYAR" in upper:
            score += 2
        if "PENJAMIN" in upper:
            score += 1

    if key in {"waktu_mulai", "waktu_selesai", "waktu_mulai_koding", "waktu_selesai_koding"}:
        if re.search(r"\b(?:\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})\b", normalized):
            score += 2
        if re.search(r"\b\d{1,2}[:\.]\d{2}(?::\d{2})?\b", normalized):
            score += 1

    if key in {"koding", "total_koding"}:
        if re.search(r"\bINA-?CBG\b|\bICD\b|\bGROUPING\b", upper):
            score += 3

    if key == "link_e_klaim":
        if re.search(r"https?://", normalized, flags=re.IGNORECASE):
            score += 4
        if "E-KLAIM" in upper or "EKLAIM" in upper:
            score += 2

    for noise_pattern in _GENERIC_NOISE_PATTERNS:
        if re.search(noise_pattern, upper):
            score -= 1

    if len(normalized) > PAYLOAD_SNIPPET_MAX_CHARS:
        score -= 1

    return score


def _rank_evidence_for_key(key: str, snippets: list[str], max_items: int = 8) -> list[str]:
    """Filter and rank evidence snippets so downstream AI gets cleaner context."""
    seen: set[str] = set()
    scored: list[tuple[int, int, str]] = []
    min_score = _EVIDENCE_MIN_SCORE.get(key, 1)

    for snippet in snippets:
        normalized = _squash_whitespace(snippet)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        score = _score_snippet_for_key(key, normalized)
        if score < min_score:
            continue

        scored.append((score, len(normalized), normalized))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [snippet for _, _, snippet in scored[:max_items]]


def extract_keyword_context_payload(
    text: str,
    *,
    window: int = 0,
    max_hits_per_key: int = 8,
) -> dict[str, list[str]]:
    """Collect contextual snippets around keyword hits for each payload key."""
    lines = [_squash_whitespace(line) for line in text.splitlines() if line.strip()]
    upper_lines = [line.upper() for line in lines]

    contexts: dict[str, list[str]] = {key: [] for key in OCR_PAYLOAD_KEYS}
    if not lines:
        return contexts

    keyword_map = _payload_keyword_map()
    for key, keywords in keyword_map.items():
        seen: set[str] = set()
        snippets: list[str] = []
        for index, upper_line in enumerate(upper_lines):
            if not any(keyword in upper_line for keyword in keywords):
                continue

            start = max(0, index - window)
            end = min(len(lines), index + window + 1)
            snippet = _squash_whitespace(" ".join(lines[start:end]))
            if not snippet or snippet in seen:
                continue
            if _score_snippet_for_key(key, snippet) < _EVIDENCE_MIN_SCORE.get(key, 1):
                continue

            seen.add(snippet)
            snippets.append(snippet)
            if len(snippets) >= max_hits_per_key:
                break

        contexts[key] = snippets

    urls = re.findall(r"https?://[^\s]+", text, flags=re.IGNORECASE)
    for url in urls:
        cleaned_url = url.rstrip(".,);]")
        if cleaned_url and cleaned_url not in contexts["link_e_klaim"]:
            contexts["link_e_klaim"].append(cleaned_url)

    return contexts


def extract_ocr_payload(
    text: str,
    *,
    total_tagihan_raw: Optional[str],
    komponen_billing: dict[str, dict[str, object]],
    keyword_context: Optional[dict[str, list[str]]] = None,
) -> dict[str, str]:
    """Extract broad related text snippets for downstream AI post-processing."""
    payload = create_empty_ocr_payload()
    lines = [_squash_whitespace(line) for line in text.splitlines() if line.strip()]
    upper_lines = [line.upper() for line in lines]

    for component_key in COMPONENT_FIELD_KEYS:
        component = komponen_billing.get(component_key, {})
        if bool(component.get("ditemukan")) and isinstance(component.get("nilai_raw"), str):
            _append_payload_text(payload, component_key, component["nilai_raw"])

    if total_tagihan_raw:
        _append_payload_text(payload, "total", total_tagihan_raw)
        _append_payload_text(payload, "billingan", total_tagihan_raw)

    field_patterns = _payload_keyword_map()

    for index, upper_line in enumerate(upper_lines):
        line = lines[index]
        for key, patterns in field_patterns.items():
            if any(pattern in upper_line for pattern in patterns):
                if _score_snippet_for_key(key, line) >= _EVIDENCE_MIN_SCORE.get(key, 1):
                    _append_payload_text(payload, key, line)
                if key in {"billingan", "rekap_billingan", "koding"} and index + 1 < len(lines):
                    next_line = lines[index + 1]
                    if _score_snippet_for_key(key, next_line) >= _EVIDENCE_MIN_SCORE.get(key, 1):
                        _append_payload_text(payload, key, next_line)

    contexts = keyword_context if keyword_context is not None else extract_keyword_context_payload(text)
    for key in OCR_PAYLOAD_KEYS:
        ranked_contexts = _rank_evidence_for_key(key, contexts.get(key, []), max_items=8)
        for snippet in ranked_contexts:
            _append_payload_text(payload, key, snippet)

    urls = re.findall(r"https?://[^\s]+", text, flags=re.IGNORECASE)
    for url in urls:
        _append_payload_text(payload, "link_e_klaim", url.rstrip(".,);]"))

    if not payload["link_e_klaim"]:
        for index, upper_line in enumerate(upper_lines):
            if "E-KLAIM" in upper_line or "EKLAIM" in upper_line:
                _append_payload_text(payload, "link_e_klaim", lines[index])

    return payload


def _infer_balance(text: str) -> tuple[Optional[str], list[str]]:
    """Infer billing balance status from free-form text when explicit field is missing."""
    lunas_match = re.search(r"(?is).{0,40}\bLUNAS\b.{0,60}", text)
    if lunas_match:
        return "lunas", [_squash_whitespace(lunas_match.group(0))]

    match = re.search(
        r"(?is)SISA\s*PEMBAYARAN.{0,40}(?:RP\.?\s*)?0(?:[.,]0+)?\b",
        text,
    )
    if match:
        return "lunas", [_squash_whitespace(match.group(0))]

    match = re.search(
        r"(?is)TOTAL\s*BAYAR(?:/|\s+)?\s*TUNAI.{0,30}(?:RP\.?\s*)?0(?:[.,]0+)?\b",
        text,
    )
    if match:
        return "lunas", [_squash_whitespace(match.group(0))]

    return None, []


def build_ai_field_analysis(
    text: str,
    *,
    total_tagihan_raw: Optional[str],
    total_tagihan_int: Optional[int],
    komponen_billing: dict[str, dict[str, object]],
    ocr_payload: dict[str, str],
) -> dict[str, dict[str, object]]:
    """Build per-field status map for downstream AI with value/status/evidence."""
    component_keys = set(COMPONENT_FIELD_KEYS)

    analysis: dict[str, dict[str, object]] = {}
    for key in OCR_PAYLOAD_KEYS:
        payload_value = _squash_whitespace(ocr_payload.get(key, ""))
        payload_parts = _split_evidence(payload_value, max_items=40)
        max_items = _EVIDENCE_MAX_ITEMS.get(key, 5)
        evidence = _rank_evidence_for_key(key, payload_parts, max_items=max_items)

        value = evidence[0] if evidence else None
        status = "found" if value else "not_found"

        if key == "total" and total_tagihan_raw:
            canonical_total = _squash_whitespace(total_tagihan_raw)
            value = canonical_total
            evidence = [canonical_total]
            status = "found"

        if key == "billingan" and total_tagihan_raw:
            canonical_total = _squash_whitespace(total_tagihan_raw)
            remaining = [item for item in evidence if item != canonical_total]
            evidence = [canonical_total] + remaining[: max(0, max_items - 1)]
            value = canonical_total
            status = "found"

        if not value and key in component_keys:
            component = komponen_billing.get(key, {})
            if bool(component.get("ditemukan")):
                raw = component.get("nilai_raw")
                numeric = component.get("nilai_int")
                if isinstance(raw, str) and raw.strip():
                    ranked_raw = _rank_evidence_for_key(key, [_squash_whitespace(raw)], max_items=1)
                    if ranked_raw:
                        value = ranked_raw[0]
                        evidence = ranked_raw
                        status = "found"
                elif isinstance(numeric, int):
                    value = _format_rupiah(numeric)
                    evidence = []
                    status = "inferred"

        if not value and key == "total":
            if total_tagihan_raw:
                ranked_total = _rank_evidence_for_key(key, [_squash_whitespace(total_tagihan_raw)], max_items=1)
                if ranked_total:
                    value = ranked_total[0]
                    evidence = ranked_total
                    status = "found"
            elif isinstance(total_tagihan_int, int):
                value = _format_rupiah(total_tagihan_int)
                evidence = []
                status = "inferred"

        if not value and key == "billingan":
            if total_tagihan_raw:
                ranked_billing = _rank_evidence_for_key(key, [_squash_whitespace(total_tagihan_raw)], max_items=1)
                if ranked_billing:
                    value = ranked_billing[0]
                    evidence = ranked_billing
                    status = "inferred"

        if not value and key == "balance":
            inferred_value, inferred_evidence = _infer_balance(text)
            if inferred_value:
                value = inferred_value
                evidence = inferred_evidence
                status = "inferred"

        if not value and key == "link_e_klaim":
            mention = re.search(r"(?is)\bE-?KLAIM\b.{0,80}", text)
            if mention:
                value = "referensi e-klaim tanpa URL"
                evidence = [_squash_whitespace(mention.group(0))]
                status = "inferred"

        analysis[key] = {
            "value": value,
            "status": status,
            "evidence": evidence,
        }

    return analysis


def build_ai_bundle(
    text: str,
    *,
    nama: Optional[str],
    total_tagihan_raw: Optional[str],
    total_tagihan_int: Optional[int],
    komponen_billing: dict[str, dict[str, object]],
    ocr_payload: dict[str, str],
    ai_field_analysis: dict[str, dict[str, object]],
    keyword_context: dict[str, list[str]],
) -> dict[str, object]:
    """Build a single AI-ready package containing all extracted context."""
    raw_text, truncated = _truncate_for_bundle(text)
    return {
        "schema_version": "v2",
        "source": "hospital_billing_ocr",
        "raw_text": raw_text,
        "raw_text_truncated": truncated,
        "raw_text_chars": len(text),
        "ringkasan_terstruktur": {
            "nama": nama,
            "total_tagihan_raw": total_tagihan_raw,
            "total_tagihan_int": total_tagihan_int,
            "komponen_billing": komponen_billing,
        },
        "field_mentah": ocr_payload,
        "field_status": ai_field_analysis,
        "keyword_context": keyword_context,
    }


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


def _extract_text_via_pymupdf(pdf_bytes: bytes) -> str:
    """Extract machine-readable text from PDF pages using PyMuPDF."""
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return ""

    try:
        page_texts: list[str] = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page in pdf:
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    page_texts.append(page_text)
    except Exception:
        return ""

    return "\n".join(page_texts).strip()


def _merge_text_sources(*sources: str) -> str:
    """Merge multiple OCR/text sources while deduplicating repeated lines."""
    seen: set[str] = set()
    merged_lines: list[str] = []
    for source in sources:
        if not source:
            continue
        for line in source.splitlines():
            normalized = _squash_whitespace(line)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged_lines.append(normalized)
    return "\n".join(merged_lines).strip()


def _needs_ocr_enrichment(text: str) -> bool:
    """Decide whether OCR should be used to enrich extracted text."""
    if OCR_ENRICH_ALWAYS:
        return True
    if is_text_too_short(text):
        return True

    upper = text.upper()
    critical_markers = (
        "NAMA",
        "TOTAL TAGIHAN",
        "RINCIAN BIAYA",
        "NO TAGIHAN",
        "NO. TAGIHAN",
        "NO REKAM MEDIS",
        "NO. REKAM MEDIS",
    )
    marker_hits = sum(1 for marker in critical_markers if marker in upper)
    if marker_hits >= 2:
        return False

    return True


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
        page_limit = page_count if OCR_MAX_PAGES <= 0 else min(OCR_MAX_PAGES, page_count)
        for index in candidates:
            if index < 0 or index >= page_count:
                continue
            if index in selected:
                continue
            selected.append(index)
            if len(selected) >= page_limit:
                break

        if len(selected) < page_limit:
            for index in range(page_count):
                if index in selected:
                    continue
                selected.append(index)
                if len(selected) >= page_limit:
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
    except Exception:
        return ""

    return "\n".join(page_texts).strip()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Read PDF text, then fallback to OCR for image-based files."""
    extraction_error: Optional[PDFTextExtractionError] = None
    primary_text = ""
    secondary_text = ""
    try:
        primary_text = _extract_text_via_pdfplumber(pdf_bytes)
    except PDFTextExtractionError as exc:
        extraction_error = exc

    secondary_text = _extract_text_via_pymupdf(pdf_bytes)
    merged_text = _merge_text_sources(primary_text, secondary_text)

    if not _needs_ocr_enrichment(merged_text):
        return merged_text

    ocr_text = _extract_text_via_ocr(pdf_bytes)
    if not ocr_text:
        if extraction_error is not None and not merged_text:
            raise extraction_error
        return merged_text

    return _merge_text_sources(merged_text, ocr_text)


def parse_billing_text(text: str) -> ParsedBillingFields:
    """Parse billing text into normalized name and total fields."""
    nama = extract_nama(text)
    total_tagihan_raw, total_tagihan_int = extract_total_tagihan(text)
    komponen_billing = extract_billing_components(text)
    keyword_context = extract_keyword_context_payload(text)
    ocr_payload = extract_ocr_payload(
        text,
        total_tagihan_raw=total_tagihan_raw,
        komponen_billing=komponen_billing,
        keyword_context=keyword_context,
    )
    ai_field_analysis = build_ai_field_analysis(
        text,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
        komponen_billing=komponen_billing,
        ocr_payload=ocr_payload,
    )
    ai_bundle = build_ai_bundle(
        text,
        nama=nama,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
        komponen_billing=komponen_billing,
        ocr_payload=ocr_payload,
        ai_field_analysis=ai_field_analysis,
        keyword_context=keyword_context,
    )

    return ParsedBillingFields(
        nama=nama,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
        komponen_billing=komponen_billing,
        ocr_payload=ocr_payload,
        ai_field_analysis=ai_field_analysis,
        ai_bundle=ai_bundle,
    )
