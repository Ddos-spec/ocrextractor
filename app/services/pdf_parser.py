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


@dataclass(frozen=True)
class TextExtractionResult:
    """Extracted text plus diagnostics about OCR/machine-text pipeline."""

    text: str
    diagnostics: dict[str, object]


_NAME_STOP_KEYWORDS = (
    "TGL",
    "TAGIHAN",
    "LAHIR",
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
    "TRANSAKSI",
    "TRANSAKS",
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
    "TA",
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
    ),
    "penunjang": ("PENUNJANG", "PENUNJANG MEDIK", "EKG", "ECG", "ECHO", "HOLTER"),
    "radiologi": ("RADIOLOGI", "RONTGEN", "X-RAY", "USG", "MRI", "CT SCAN", "ANGIOGRAM"),
    "laboratorium": ("LABORATORIUM", "LABORAT", "HEMATOLOGI", "MIKROBIOLOGI", "PATOLOGI"),
    "pelayanan_darah": ("PELAYANAN DARAH", "TRANSFUSI", "PRC", "PACKED RED CELL"),
    "rehabilitasi": ("REHABILITASI", "FISIOTERAPI", "TERAPI OKUPASI", "REHAB"),
    "kamar_akomodasi": ("KAMAR", "AKOMODASI", "RUANG RANAP", "BED", "ADMISI"),
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

_DOCUMENT_PROFILE_MARKERS: dict[str, tuple[str, ...]] = {
    "rajal": (
        "RAWAT JALAN",
        "RAJAL",
        "POLI",
        "POLIKLINIK",
        "KONSULTASI DOKTER",
        "GAWAT DARURAT",
        "IGD",
    ),
    "ranap": (
        "RAWAT INAP",
        "RUANG",
        "RUANG RANAP",
        "KAMAR",
        "AKOMODASI",
        "VISITE",
        "BED",
        "ADMISI",
    ),
}

_BILLING_PAGE_MARKERS: tuple[str, ...] = (
    "RINCIAN BIAYA",
    "TOTAL TAGIHAN",
    "NO. TAGIHAN",
    "NO TAGIHAN",
    "KASIR",
    "CARA BAYAR",
)

_RAJAL_EXCLUDED_COMPONENTS = frozenset({"kamar_akomodasi", "rawat_intensif"})

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

_SUMMARY_STRATEGY: dict[str, str] = {
    "radiologi": "sum_summary",
    "penunjang": "sum_summary",
    "obat": "sum_summary",
    "kamar_akomodasi": "max_summary",
    "rawat_intensif": "max_summary",
    "laboratorium": "hybrid",
    "keperawatan": "hybrid",
    "konsultasi": "hybrid",
    "prosedur_non_bedah": "hybrid",
    "bmhp": "hybrid",
}

_NON_BEDAH_FALLBACK_KEYWORDS: tuple[str, ...] = (
    "PASANG INFUS",
    "PEMASANGAN INFUS",
    "MEMASUKKAN OBAT",
    "PENGAMBILAN DARAH VENA",
    "INJEKSI INTRA CUTAN",
    "SUBCUTAN",
    "INTRAMUSKULAR",
    "INTRAVENA PER HARI",
)

_NON_BEDAH_FALLBACK_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "OMEPRAZ",
    "ONDANSETRON",
    "METAMIZOL",
    "PARACETAMOL",
    "RANITID",
    "ANTASIDA",
    "TABLET",
    "KAPSUL",
    "SYRUP",
    "SIRUP",
)

_BMHP_FALLBACK_KEYWORDS: tuple[str, ...] = (
    "SYRINGE",
    "SPUIT",
    "INFUSET",
    "INFUS SET",
    "IV CATHETER",
    "CANNULA",
    "NEEDLE",
    "DISPOSABLE",
    "KASSA",
    "JELLY",
    "ALKOHOL",
    "OKSIGEN",
)

_BMHP_FALLBACK_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "JUMLAH LABORATORIUM",
    "JUMLAH RADIOLOGI",
)

_KONSULTASI_FALLBACK_KEYWORDS: tuple[str, ...] = (
    "PEMERIKSAAN/KONSULTASI DOKTER",
    "DOKTER UMUM",
    "DOKDER UMUM",
    "DOKTER SPESIALIS",
    "DOKDER SPESIALIS",
    "KONSULTASI DOKTER",
)

_KONSULTASI_FALLBACK_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "TOTAL TAGIHAN",
    "NO. TAGIHAN",
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
OCR_MAX_PAGES = _env_int("OCR_MAX_PAGES", 5, minimum=0)
OCR_PSM = _env_int("OCR_PSM", 6, minimum=3)
OCR_LANG_PRIMARY = os.getenv("OCR_LANG_PRIMARY", "ind+eng")
OCR_LANG_FALLBACK = os.getenv("OCR_LANG_FALLBACK", "eng")
OCR_MULTI_PASS = _env_bool("OCR_MULTI_PASS", True)
OCR_BINARIZE_THRESHOLD = _env_int("OCR_BINARIZE_THRESHOLD", 170, minimum=80)
OCR_PAGE_ACCEPT_SCORE = _env_int("OCR_PAGE_ACCEPT_SCORE", 28, minimum=5)
AI_BUNDLE_TEXT_MAX_CHARS = _env_int("AI_BUNDLE_TEXT_MAX_CHARS", 80000, minimum=2000)
PAYLOAD_SNIPPET_MAX_CHARS = _env_int("PAYLOAD_SNIPPET_MAX_CHARS", 320, minimum=120)
PAYLOAD_MAX_PARTS_PER_KEY = _env_int("PAYLOAD_MAX_PARTS_PER_KEY", 15, minimum=5)
OCR_ENRICH_ALWAYS = _env_bool("OCR_ENRICH_ALWAYS", False)
DOC_VALIDATION_MIN_SCORE = _env_int("DOC_VALIDATION_MIN_SCORE", 45, minimum=1)


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


def _count_non_space_chars(text: str) -> int:
    """Count non-space characters for OCR quality/coverage diagnostics."""
    return len(re.sub(r"\s+", "", text))


def _score_ocr_candidate(text: str) -> int:
    """Heuristic score to choose better OCR candidate text."""
    normalized = _squash_whitespace(text)
    if not normalized:
        return 0

    upper = normalized.upper()
    score = min(24, len(normalized) // 160)
    score += min(24, len(re.findall(r"(?i)\bRP\.?\s*\d", normalized)) * 4)
    score += min(18, len(re.findall(r"\b\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}\b", normalized)) * 3)

    critical_markers = (
        "RINCIAN BIAYA",
        "TOTAL TAGIHAN",
        "NO TAGIHAN",
        "NO REKAM MEDIS",
        "NAMA",
        "LABORATORIUM",
        "RADIOLOGI",
        "FARMASI",
        "KASIR",
    )
    score += sum(3 for marker in critical_markers if marker in upper)

    noise_chars = sum(1 for char in normalized if char in "{}[]|`~^_")
    if noise_chars > 0:
        noise_ratio = noise_chars / max(1, len(normalized))
        score -= min(20, int(noise_ratio * 120))

    return max(score, 0)


def _build_extraction_diagnostics(
    *,
    primary_text: str,
    secondary_text: str,
    ocr_text: str,
    merged_text: str,
    ocr_attempted: bool,
    name_hints: Optional[list[str]] = None,
) -> dict[str, object]:
    """Build stable diagnostics describing how text was extracted."""
    primary_chars = _count_non_space_chars(primary_text)
    secondary_chars = _count_non_space_chars(secondary_text)
    ocr_chars = _count_non_space_chars(ocr_text)
    merged_chars = _count_non_space_chars(merged_text)

    machine_chars = max(primary_chars, secondary_chars)
    machine_readable_detected = machine_chars >= 40
    ocr_used = ocr_chars >= 40

    if merged_chars <= 0:
        source_mode = "empty"
    elif machine_readable_detected and ocr_used:
        source_mode = "machine_plus_ocr"
    elif machine_readable_detected:
        source_mode = "machine_only"
    elif ocr_used:
        source_mode = "ocr_only"
    else:
        source_mode = "unknown"

    diagnostics = {
        "pipeline_version": "v2.1",
        "source_mode": source_mode,
        "ocr_attempted": ocr_attempted,
        "ocr_used": ocr_used,
        "machine_readable_detected": machine_readable_detected,
        "text_chars": {
            "pdfplumber": primary_chars,
            "pymupdf": secondary_chars,
            "ocr": ocr_chars,
            "merged": merged_chars,
        },
    }
    if name_hints:
        unique_hints: list[str] = []
        seen_hints: set[str] = set()
        for hint in name_hints:
            normalized = _squash_whitespace(hint)
            if not normalized or normalized in seen_hints:
                continue
            seen_hints.add(normalized)
            unique_hints.append(normalized)
        if unique_hints:
            diagnostics["name_hints"] = unique_hints

    return diagnostics


def _canonical_line_key(text: str) -> str:
    """Canonicalize OCR line for deduplication across minor OCR variants."""
    compact = _squash_whitespace(text).upper()
    compact = re.sub(r"[0-9]", " ", compact)
    compact = re.sub(r"[^A-Z\s]", " ", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact


def _sum_amount_by_keywords(
    text: str,
    *,
    include_keywords: tuple[str, ...],
    exclude_keywords: tuple[str, ...] = (),
    cap: int = 10_000_000,
) -> tuple[int, list[str]]:
    """Sum deduplicated line amounts matched by keyword include/exclude filters."""
    lines = [_squash_whitespace(line) for line in text.splitlines() if line.strip()]
    seen: set[str] = set()
    hits: list[tuple[int, str]] = []

    for line in lines:
        upper = line.upper()
        if not any(keyword in upper for keyword in include_keywords):
            continue
        if exclude_keywords and any(keyword in upper for keyword in exclude_keywords):
            continue
        if not re.search(r"(?i)\bRP\.?\s*\d", line):
            continue

        amount = _extract_amount_from_line(line)
        if amount is None or amount <= 0 or amount > cap:
            continue

        dedup_key = f"{amount}|{_canonical_line_key(line)}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        hits.append((amount, line))

    total = sum(amount for amount, _ in hits)
    raw_lines = [line for _, line in hits]
    return total, raw_lines


def _sum_pharmacy_line_items(
    text: str,
    *,
    total_tagihan_int: Optional[int],
) -> tuple[Optional[int], list[str], Optional[str]]:
    """Recompute pharmacy subtotal by summing item rows inside the farmasi section."""
    lines = [_squash_whitespace(line) for line in text.splitlines() if line.strip()]
    if not lines:
        return None, [], None

    in_section = False
    summary_line: Optional[str] = None
    item_hits: list[tuple[int, str]] = []
    seen: set[str] = set()
    cap = max(2_000_000, int(total_tagihan_int * 1.2)) if isinstance(total_tagihan_int, int) else 10_000_000

    def is_pharmacy_section_start(line: str) -> bool:
        if re.search(r"(?i)^\s*(FARMASI|APOTIK)\b", line):
            return True
        if re.search(r"(?i)^\s*JUMLAH\b.*\b(FARMASI|APOTIK)\b", line):
            return True
        if re.search(r"(?i)^\s*OBAT(?:\s+(KRONIS|KEMOTERAPI))?\b\s*:?\s*$", line):
            return True
        return False

    def is_other_section_header(line: str) -> bool:
        upper = line.upper()
        if re.search(r"(?i)\bTOTAL\s+TAGIHAN\b", line):
            return True
        if re.search(r"(?i)\b(PENJAMIN|KASIR|TOTAL BAYAR|SISA PEMBAYARAN)\b", line):
            return True
        if _extract_amount_from_line(line) is not None:
            return False
        if re.search(r"(?i)\bJUMLAH\b", line):
            return False
        for key, aliases in _COMPONENT_ALIASES.items():
            if key == "obat":
                continue
            if any(alias in upper for alias in aliases):
                return True
        return False

    for line in lines:
        upper = line.upper()
        if not in_section:
            if is_pharmacy_section_start(line):
                in_section = True
            continue

        if is_other_section_header(line):
            break

        if re.search(r"(?i)\bJUMLAH\b", line):
            amount_on_summary = _extract_amount_from_line(line)
            if amount_on_summary is not None and amount_on_summary > 0:
                summary_line = line
                break

        if not re.search(r"(?i)\bRP\.?\s*\d", line):
            continue

        amount = _extract_amount_from_line(line)
        if amount is None or amount <= 0 or amount > cap:
            continue

        dedup_key = f"{amount}|{_canonical_line_key(line)}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        item_hits.append((amount, line))

    if len(item_hits) < 2:
        summary_amount = _extract_amount_from_line(summary_line) if summary_line else None
        if isinstance(summary_amount, int) and summary_amount > 0:
            return summary_amount, [line for _, line in item_hits], summary_line
        return None, [line for _, line in item_hits], summary_line

    total = sum(amount for amount, _ in item_hits)
    if total <= 0 or total > cap:
        summary_amount = _extract_amount_from_line(summary_line) if summary_line else None
        if isinstance(summary_amount, int) and summary_amount > 0:
            return summary_amount, [line for _, line in item_hits], summary_line
        return None, [line for _, line in item_hits], summary_line

    summary_amount = _extract_amount_from_line(summary_line) if summary_line else None
    if isinstance(summary_amount, int) and summary_amount > 0:
        summary_looks_too_small = summary_amount < max(5_000, int(total * 0.75))
        if summary_looks_too_small and total <= cap:
            return total, [line for _, line in item_hits], summary_line
        if total >= summary_amount and total <= int(summary_amount * 1.15):
            return summary_amount, [line for _, line in item_hits], summary_line
        return summary_amount, [line for _, line in item_hits], summary_line

    return total, [line for _, line in item_hits], summary_line


def _pick_fallback_raw_line(lines: list[str]) -> Optional[str]:
    """Pick a representative line from fallback keyword hits."""
    if not lines:
        return None

    scored_lines = sorted(
        lines,
        key=lambda line: (
            0 if "JUMLAH" in line.upper() else 1,
            len(line),
        ),
        reverse=True,
    )
    return scored_lines[0]


def _apply_component_fallbacks(
    text: str,
    *,
    components: dict[str, dict[str, object]],
    total_tagihan_int: Optional[int],
) -> None:
    """Apply conservative fallback extraction for categories that often miss in OCR noise."""
    cap = max(2_000_000, int(total_tagihan_int * 1.5)) if isinstance(total_tagihan_int, int) else 10_000_000

    non_bedah = components.get("prosedur_non_bedah")
    if isinstance(non_bedah, dict):
        non_bedah_amount, non_bedah_lines = _sum_amount_by_keywords(
            text,
            include_keywords=_NON_BEDAH_FALLBACK_KEYWORDS,
            exclude_keywords=_NON_BEDAH_FALLBACK_EXCLUDE_KEYWORDS,
            cap=cap,
        )
        existing_non_bedah_amount = non_bedah.get("nilai_int")
        should_apply_non_bedah_fallback = not isinstance(existing_non_bedah_amount, int)
        if isinstance(existing_non_bedah_amount, int):
            should_apply_non_bedah_fallback = (
                non_bedah_amount > existing_non_bedah_amount
                and existing_non_bedah_amount < max(5_000, int(non_bedah_amount * 0.75))
            )

        if non_bedah_amount > 0 and should_apply_non_bedah_fallback:
            non_bedah["ditemukan"] = True
            non_bedah["nilai_int"] = non_bedah_amount
            non_bedah["nilai_raw"] = _pick_fallback_raw_line(non_bedah_lines)

    konsultasi = components.get("konsultasi")
    if isinstance(konsultasi, dict) and not isinstance(konsultasi.get("nilai_int"), int):
        konsultasi_amount, konsultasi_lines = _sum_amount_by_keywords(
            text,
            include_keywords=_KONSULTASI_FALLBACK_KEYWORDS,
            exclude_keywords=_KONSULTASI_FALLBACK_EXCLUDE_KEYWORDS,
            cap=cap,
        )
        if konsultasi_amount > 0:
            konsultasi["ditemukan"] = True
            konsultasi["nilai_int"] = konsultasi_amount
            konsultasi["nilai_raw"] = _pick_fallback_raw_line(konsultasi_lines)

    bmhp = components.get("bmhp")
    bmhp_fallback_applied = False
    bmhp_amount: Optional[int] = None
    if isinstance(bmhp, dict):
        if isinstance(bmhp.get("nilai_int"), int):
            bmhp_amount = bmhp["nilai_int"]  # type: ignore[assignment]
        else:
            fallback_bmhp_amount, fallback_bmhp_lines = _sum_amount_by_keywords(
                text,
                include_keywords=_BMHP_FALLBACK_KEYWORDS,
                exclude_keywords=_BMHP_FALLBACK_EXCLUDE_KEYWORDS,
                cap=cap,
            )
            if fallback_bmhp_amount > 0:
                bmhp["ditemukan"] = True
                bmhp["nilai_int"] = fallback_bmhp_amount
                bmhp["nilai_raw"] = _pick_fallback_raw_line(fallback_bmhp_lines)
                bmhp_amount = fallback_bmhp_amount
                bmhp_fallback_applied = True

    obat = components.get("obat")
    if not isinstance(obat, dict):
        return

    pharmacy_item_total, pharmacy_lines, pharmacy_summary = _sum_pharmacy_line_items(
        text,
        total_tagihan_int=total_tagihan_int,
    )
    if isinstance(pharmacy_item_total, int) and pharmacy_item_total > 0:
        existing_obat_amount = obat.get("nilai_int")
        should_replace_with_item_sum = not isinstance(existing_obat_amount, int)
        if isinstance(existing_obat_amount, int):
            should_replace_with_item_sum = (
                pharmacy_item_total > existing_obat_amount
                and existing_obat_amount < max(5_000, int(pharmacy_item_total * 0.75))
            )

        if should_replace_with_item_sum:
            obat["ditemukan"] = True
            obat["nilai_int"] = pharmacy_item_total
            if pharmacy_summary:
                obat["nilai_raw"] = (
                    f"{pharmacy_summary} (hitung item {_format_rupiah(pharmacy_item_total)})"
                )
            else:
                obat["nilai_raw"] = _pick_fallback_raw_line(pharmacy_lines) or _format_rupiah(pharmacy_item_total)

    obat_raw = obat.get("nilai_raw")
    if (
        isinstance(obat.get("nilai_int"), int)
        and isinstance(obat_raw, str)
        and re.search(r"(?i)\bJUMLAH\b", obat_raw)
        and isinstance(bmhp, dict)
        and isinstance(bmhp.get("nilai_int"), int)
    ):
        bmhp_raw = bmhp.get("nilai_raw")
        bmhp_explicit_section = bool(
            isinstance(bmhp_raw, str)
            and re.search(r"(?i)\b(BMHP|BHP|BAHAN MEDIS HABIS PAKAI|BAHAN HABIS PAKAI)\b", bmhp_raw)
        )
        if not bmhp_explicit_section and bmhp["nilai_int"] < obat["nilai_int"]:  # type: ignore[index]
            components["bmhp"] = _blank_component_result("bmhp")
            bmhp = components["bmhp"]
            bmhp_amount = None

    if not bmhp_fallback_applied:
        return
    if not isinstance(obat.get("nilai_int"), int) or not isinstance(bmhp_amount, int):
        return

    obat_amount = obat["nilai_int"]  # type: ignore[assignment]
    if obat_amount <= bmhp_amount:
        return

    obat_raw = obat.get("nilai_raw")
    if not isinstance(obat_raw, str):
        return
    if not re.search(r"(?i)\bJUMLAH\b.*\b(FARMASI|APOTIK|OBAT)\b", obat_raw):
        return

    adjusted_amount = obat_amount - bmhp_amount
    if adjusted_amount <= 0:
        return

    obat["nilai_int"] = adjusted_amount
    obat["nilai_raw"] = f"{obat_raw} (dikurangi BMHP {_format_rupiah(bmhp_amount)})"


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


def _looks_like_grouped_amount_parts(parts: list[str]) -> bool:
    """Return True for tokens shaped like `1 234 567`."""
    if not parts or not all(part.isdigit() for part in parts):
        return False
    if not (1 <= len(parts[0]) <= 3):
        return False
    return all(len(part) == 3 for part in parts[1:])


def _parse_total_amount(amount_token: str) -> Optional[int]:
    """Parse total-tagihan amount while trimming common OCR tail noise."""
    normalized = _squash_whitespace(amount_token)
    if not normalized:
        return None

    variants: list[str] = []
    parts = normalized.split(" ")
    if len(parts) > 1 and len(parts[-1]) <= 2:
        prefix = parts[:-1]
        prefix_text = " ".join(prefix)
        if prefix and (any(ch in prefix_text for ch in ".,") or _looks_like_grouped_amount_parts(prefix)):
            variants.append(prefix_text)
    variants.append(normalized)

    seen: set[str] = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        parsed = _parse_rupiah_amount(variant)
        if parsed is not None:
            return parsed

    return None


def _is_total_table_header_line(line: str) -> bool:
    """Return True when OCR line is a billing table header, not a summary total."""
    normalized = _squash_whitespace(line).upper()
    if "TOTAL TAGIHAN" not in normalized:
        return False

    header_markers = (
        "UNIT LAYANAN",
        "UNITLAYANAN",
        "ITEM",
        "TEM/",
        "TINDAKAN",
        "PEMERIKSAAN",
        "TANGGAL",
        "TARIF",
        "HARGA",
        "JML",
    )
    marker_hits = sum(1 for marker in header_markers if marker in normalized)
    return marker_hits >= 3


def _is_total_followup_line(line: str) -> bool:
    """Return True when the next line plausibly only contains the final total amount."""
    normalized = _squash_whitespace(line).upper()
    if not normalized:
        return False
    if re.search(r"\bRP\.?\s*\d", normalized):
        return True
    return bool(re.fullmatch(r"(?:RP\.?\s*)?\d[\d.,\s]{2,20}", normalized))


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

    if _is_total_table_header_line(normalized_context):
        return False
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

        token_for_cleanup = token
        if token_for_cleanup.endswith("!") and re.search(r"[A-Za-z]{3,}", token_for_cleanup):
            token_for_cleanup = f"{token_for_cleanup[:-1]}I"

        cleaned = re.sub(r"[^A-Za-z'.-]", "", token_for_cleanup)
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

    if len(tokens) == 1 and len(tokens[0]) >= 6 and tokens[0].endswith("DM"):
        single = tokens[0]
        tokens = [single[:-2], single[-2:]]

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


def _extract_name_hint_from_diagnostics(
    extraction_diagnostics: Optional[dict[str, object]],
) -> Optional[str]:
    """Prefer explicit OCR crop name hints when available."""
    if not isinstance(extraction_diagnostics, dict):
        return None

    raw_hints = extraction_diagnostics.get("name_hints")
    if not isinstance(raw_hints, list):
        return None

    candidates: list[str] = []
    for item in raw_hints:
        if not isinstance(item, str):
            continue
        normalized = _clean_name_candidate(item)
        if normalized and _is_probable_patient_name(normalized):
            candidates.append(normalized)

    if not candidates:
        return None

    best_name: Optional[str] = None
    best_score = (-1, -1)
    for candidate in candidates:
        token_count = len(candidate.split())
        alpha_count = sum(1 for char in candidate if "A" <= char <= "Z")
        score = (token_count, alpha_count)
        if score > best_score:
            best_score = score
            best_name = candidate

    return best_name


def extract_total_tagihan(text: str) -> tuple[Optional[str], Optional[int]]:
    """Extract total billing phrase and numeric value in rupiah."""
    selected_raw: Optional[str] = None
    selected_value: Optional[int] = None

    for match in _TOTAL_PATTERN.finditer(text):
        amount_token = match.group(1)
        parsed_amount = _parse_total_amount(amount_token)
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
        if _is_total_table_header_line(line):
            continue

        amount_tokens = _RUPIAH_INLINE_PATTERN.findall(line)
        if not amount_tokens:
            amount_tokens = _AMOUNT_TOKEN_PATTERN.findall(line)
        raw_phrase = line
        if not amount_tokens and index + 1 < len(lines):
            next_line = lines[index + 1]
            if _is_total_followup_line(next_line):
                amount_tokens = _RUPIAH_INLINE_PATTERN.findall(next_line)
                if not amount_tokens:
                    amount_tokens = _AMOUNT_TOKEN_PATTERN.findall(next_line)
                raw_phrase = f"{line} {next_line}"

        for amount_token in amount_tokens:
            parsed_amount = _parse_total_amount(amount_token)
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
            token_candidates = re.findall(r"\d{1,3}(?:[.,]\d{3})+(?:,\d{1,2})?|\d{4,}", token)
            value = None
            if token_candidates:
                parsed_candidates = [
                    parsed
                    for parsed in (_parse_rupiah_amount(candidate) for candidate in token_candidates)
                    if parsed is not None
                ]
                if parsed_candidates:
                    value = parsed_candidates[-1]
            if value is None:
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


def _blank_component_result(component_key: str) -> dict[str, object]:
    """Build an empty component payload for a known billing key."""
    return {
        "label": _COMPONENT_LABELS[component_key],
        "ditemukan": False,
        "nilai_raw": None,
        "nilai_int": None,
    }


def extract_billing_components(text: str, *, total_tagihan_int: Optional[int] = None) -> dict[str, dict[str, object]]:
    """Extract requested billing components and optional nominal values."""
    results: dict[str, dict[str, object]] = {
        key: _blank_component_result(key)
        for key in _COMPONENT_ALIASES
    }

    lines = [_squash_whitespace(line) for line in text.splitlines() if line.strip()]
    upper_lines = [line.upper() for line in lines]

    current_section_key: Optional[str] = None
    amount_tracker: dict[str, list[tuple[int, str, bool]]] = {key: [] for key in _COMPONENT_ALIASES}
    amount_cap = max(2_000_000, int(total_tagihan_int * 1.5)) if isinstance(total_tagihan_int, int) else 10_000_000

    def has_recent_section_header(section_key: str, current_index: int, window: int = 25) -> bool:
        """Return True if a plain section header for `section_key` appears shortly before current line."""
        aliases = _COMPONENT_ALIASES.get(section_key, ())
        if not aliases:
            return False

        start = max(0, current_index - window)
        for prev_index in range(start, current_index):
            prev_line = lines[prev_index]
            prev_upper = upper_lines[prev_index]
            if not any(alias in prev_upper for alias in aliases):
                continue
            if re.search(r"(?i)\bRP\.?\s*\d", prev_line):
                continue
            if _extract_amount_from_line(prev_line) is not None:
                continue
            return True
        return False

    for index, upper_line in enumerate(upper_lines):
        line = lines[index]

        matched_header_keys: list[str] = []
        for key, aliases in _COMPONENT_ALIASES.items():
            if any(alias in upper_line for alias in aliases):
                matched_header_keys.append(key)
        if matched_header_keys:
            current_section_key = matched_header_keys[0]

        summary_key: Optional[str] = None
        if "JUMLAH" in upper_line:
            for key, aliases in _COMPONENT_ALIASES.items():
                if any(alias in upper_line for alias in aliases):
                    summary_key = key
                    break
            if summary_key is None and current_section_key is not None:
                # Avoid assigning ambiguous lines like "Jumlah Rp. 198.000" into the latest section.
                generic_jumlah_only = bool(re.search(r"(?i)\bJUMLAH\b\s*(?:RP\.?|RUPIAH)\b", line))
                if not generic_jumlah_only:
                    summary_key = current_section_key
                elif has_recent_section_header(current_section_key, index):
                    summary_key = current_section_key

        if summary_key is not None:
            amount_on_summary = _extract_amount_from_line(line)
            if (
                amount_on_summary is not None
                and amount_on_summary <= amount_cap
                and re.search(r"(?i)\bRP\.?\s*\d", line)
            ):
                section_result = results[summary_key]
                section_result["ditemukan"] = True
                section_result["nilai_raw"] = line
                amount_tracker[summary_key].append((amount_on_summary, line, True))
                current_section_key = summary_key

        for key in matched_header_keys:
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
                if amount_value > amount_cap:
                    continue
                if re.search(r"(?i)\bRP\.?\s*\d", raw_line):
                    amount_tracker[key].append((amount_value, raw_line, False))
                if current["nilai_raw"] is None:
                    current["nilai_raw"] = raw_line
            elif current["nilai_raw"] is None:
                current["nilai_raw"] = raw_line

    for key, current in results.items():
        records = amount_tracker.get(key, [])
        if not records:
            continue

        item_records = [item for item in records if not item[2]]
        dedup_item_lines: dict[str, int] = {}
        for amount_value, raw_line, _ in item_records:
            dedup_key = f"{amount_value}|{_canonical_line_key(raw_line)}"
            existing = dedup_item_lines.get(dedup_key)
            if existing is None or amount_value > existing:
                dedup_item_lines[dedup_key] = amount_value
        dedup_item_sum = sum(dedup_item_lines.values()) if dedup_item_lines else 0

        summary_records = [item for item in records if item[2]]
        if summary_records:
            unique_summary: dict[str, int] = {}
            for amount_value, raw_line, _ in summary_records:
                dedup_key = f"{amount_value}|{_canonical_line_key(raw_line)}"
                existing = unique_summary.get(dedup_key)
                if existing is None or amount_value > existing:
                    unique_summary[dedup_key] = amount_value

            summary_sum = sum(unique_summary.values())
            summary_max_amount, summary_max_raw, _ = max(summary_records, key=lambda item: item[0])
            strategy = _SUMMARY_STRATEGY.get(key, "hybrid")

            if strategy == "sum_summary":
                current["nilai_int"] = summary_sum
                current["nilai_raw"] = summary_max_raw
            elif strategy == "max_summary":
                current["nilai_int"] = summary_max_amount
                current["nilai_raw"] = summary_max_raw
            else:
                # Hybrid: trust item sum when summary looks too small/incomplete, but keep it bounded.
                chosen_sum = summary_sum
                if dedup_item_sum > summary_sum and dedup_item_sum <= amount_cap:
                    if summary_sum == 0 or dedup_item_sum <= summary_sum * 3:
                        chosen_sum = dedup_item_sum
                current["nilai_int"] = chosen_sum
                current["nilai_raw"] = summary_max_raw
            continue

        if dedup_item_lines:
            if key in {"kamar_akomodasi", "rawat_intensif"}:
                current["nilai_int"] = max(dedup_item_lines.values())
            else:
                current["nilai_int"] = dedup_item_sum

    return results


def _component_hit_count(components: dict[str, dict[str, object]]) -> int:
    """Count how many component categories are found."""
    return sum(1 for item in components.values() if bool(item.get("ditemukan")))


def _copy_component_result(component: dict[str, object]) -> dict[str, object]:
    """Return a safe component copy only when a numeric nominal is available."""
    value = component.get("nilai_int")
    if not isinstance(value, int) or value <= 0:
        return _redact_component_result(component)

    return {
        "label": component.get("label"),
        "ditemukan": True,
        "nilai_raw": component.get("nilai_raw"),
        "nilai_int": value,
    }


def _redact_component_result(component: dict[str, object]) -> dict[str, object]:
    """Blank component values when document is not approved for automation."""
    return {
        "label": component.get("label"),
        "ditemukan": False,
        "nilai_raw": None,
        "nilai_int": None,
    }


def _sum_component_amounts(components: dict[str, dict[str, object]]) -> int:
    """Sum extracted component nominals while ignoring missing values."""
    total = 0
    for component in components.values():
        value = component.get("nilai_int")
        if isinstance(value, int) and value > 0:
            total += value
    return total


def _collect_total_candidates(text: str) -> list[int]:
    """Collect distinct total-tagihan candidates for conflict detection."""
    lines = [_squash_whitespace(line) for line in text.splitlines() if line.strip()]
    seen: set[int] = set()
    candidates: list[int] = []

    for index, line in enumerate(lines):
        if not re.search(r"(?i)\bTOTAL\b", line) or not re.search(r"(?i)\bTAGIHAN\b", line):
            continue
        if _is_total_table_header_line(line):
            continue

        tokens = _RUPIAH_INLINE_PATTERN.findall(line)
        if not tokens:
            tokens = _AMOUNT_TOKEN_PATTERN.findall(line)

        if not tokens and index + 1 < len(lines):
            next_line = lines[index + 1]
            if _is_total_followup_line(next_line):
                tokens = _RUPIAH_INLINE_PATTERN.findall(next_line)
                if not tokens:
                    tokens = _AMOUNT_TOKEN_PATTERN.findall(next_line)
            else:
                tokens = []

        for token in tokens:
            value = _parse_total_amount(token)
            if value is None or value <= 0 or value > 500_000_000:
                continue
            if not _is_valid_total_candidate(token, value, line):
                continue
            if value in seen:
                continue
            seen.add(value)
            candidates.append(value)

    return candidates


def _split_billing_segments(text: str) -> list[str]:
    """Split document text by recurring billing header blocks."""
    lines = text.splitlines()
    if not lines:
        return []

    marker_indices = [
        index
        for index, line in enumerate(lines)
        if re.search(r"(?i)\bRINCIAN\s+BIAYA\s+PELAYANAN\s+PASIEN\b", line)
    ]
    if not marker_indices:
        return [_squash_whitespace(text)]

    segments: list[str] = []
    for marker_pos, start in enumerate(marker_indices):
        end = marker_indices[marker_pos + 1] if marker_pos + 1 < len(marker_indices) else len(lines)
        segment_lines = lines[start:end]
        segment_text = "\n".join(segment_lines).strip()
        if segment_text:
            segments.append(segment_text)
    return segments or [_squash_whitespace(text)]


def _split_page_chunks(text: str) -> list[tuple[Optional[int], str]]:
    """Split OCR text into page-tagged chunks while preserving order."""
    page_header = re.compile(r"^\s*===\s*PAGE\s+(\d+)\s*===\s*$", flags=re.IGNORECASE)
    chunks: list[tuple[Optional[int], str]] = []
    current_page: Optional[int] = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        match = page_header.match(raw_line)
        if match:
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append((current_page, chunk_text))
            current_page = int(match.group(1))
            current_lines = []
            continue
        current_lines.append(raw_line)

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append((current_page, chunk_text))

    if chunks:
        return chunks

    normalized = text.strip()
    return [(None, normalized)] if normalized else []


def _select_tail_billing_window(text: str, max_pages: int = 3) -> str:
    """Prefer the last few OCR pages because billing totals usually appear at the end."""
    chunks = _split_page_chunks(text)
    if len(chunks) <= 1:
        return text

    numbered_chunks = [chunk for chunk in chunks if chunk[0] is not None]
    if numbered_chunks and len(numbered_chunks) == len(chunks):
        chunks = sorted(numbered_chunks, key=lambda item: item[0] or 0)

    tail_chunks = chunks[-max_pages:]
    tail_text = "\n\n".join(chunk_text for _, chunk_text in tail_chunks if chunk_text.strip()).strip()
    if not tail_text:
        return text

    if re.search(r"(?i)\b(RINCIAN\s+BIAYA|TOTAL\s+TAGIHAN|KASIR|CARA\s+BAYAR)\b", tail_text):
        return tail_text
    return text


def _extract_no_tagihan(text: str) -> Optional[str]:
    """Extract normalized No. Tagihan identifier from segment text."""
    match = re.search(r"(?is)\bNO\.?\s*TAGIHAN\b\D{0,12}([0-9][0-9\-\s]{5,})", text)
    if not match:
        return None
    digits = re.sub(r"[^\d]", "", match.group(1))
    if len(digits) < 6:
        return None
    return digits


def select_primary_billing_text(text: str) -> str:
    """Pick the most relevant billing segment when multiple episodes exist in one PDF."""
    narrowed_text = _select_tail_billing_window(text)
    candidates = _split_billing_segments(narrowed_text)
    if not candidates:
        return narrowed_text
    if len(candidates) == 1:
        return candidates[0]

    grouped: dict[str, list[str]] = {}
    for index, segment in enumerate(candidates):
        tagihan = _extract_no_tagihan(segment)
        group_key = tagihan if tagihan else f"_segment_{index}"
        grouped.setdefault(group_key, []).append(segment)

    grouped_candidates = ["\n".join(parts).strip() for parts in grouped.values() if any(part.strip() for part in parts)]
    if not grouped_candidates:
        grouped_candidates = candidates

    best_segment = grouped_candidates[0]
    best_sort_key = (-1, -1, -1)

    for segment in grouped_candidates:
        total_raw, total_int = extract_total_tagihan(segment)
        components = extract_billing_components(segment, total_tagihan_int=total_int)
        component_hits = _component_hit_count(components)
        summary_hits = len(re.findall(r"(?i)\bJUMLAH\b", segment))
        has_total_phrase = 1 if total_raw else 0
        total_score = total_int if isinstance(total_int, int) else 0

        sort_key = (total_score, component_hits, summary_hits + has_total_phrase)
        if sort_key > best_sort_key:
            best_sort_key = sort_key
            best_segment = segment

    return best_segment


def _count_phrase_hits(text: str, phrases: tuple[str, ...]) -> int:
    """Count how many distinct uppercase phrases are present in text."""
    upper = text.upper()
    return sum(1 for phrase in phrases if phrase in upper)


def _count_profile_markers(text: str, document_profile: str) -> int:
    """Count profile-specific marker hits for rajal/ranap templates."""
    markers = _DOCUMENT_PROFILE_MARKERS.get(document_profile)
    if not markers:
        return 0
    return _count_phrase_hits(text, markers)


def _select_profile_billing_text(text: str, document_profile: str) -> str:
    """Narrow text to pages that look like the chosen RS billing profile."""
    if document_profile not in _DOCUMENT_PROFILE_MARKERS:
        return text

    chunks = _split_page_chunks(text)
    if len(chunks) <= 1:
        return text

    selected_chunks: list[str] = []
    for _, chunk_text in chunks:
        billing_hits = _count_phrase_hits(chunk_text, _BILLING_PAGE_MARKERS)
        profile_hits = _count_profile_markers(chunk_text, document_profile)
        if billing_hits <= 0:
            continue
        if profile_hits <= 0 and billing_hits < 2:
            continue
        selected_chunks.append(chunk_text)

    if not selected_chunks:
        return text

    narrowed = "\n\n".join(chunk for chunk in selected_chunks if chunk.strip()).strip()
    return narrowed or text


def _apply_document_profile_to_components(
    components: dict[str, dict[str, object]],
    *,
    document_profile: str,
) -> None:
    """Trim components that do not belong to the active template profile."""
    if document_profile != "rajal":
        return

    for component_key in _RAJAL_EXCLUDED_COMPONENTS:
        if component_key not in components:
            continue
        components[component_key] = _blank_component_result(component_key)


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
    component_map["kamar_akomodasi"] = component_map["kamar_akomodasi"] + ("RUANG PERAWATAN",)
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


def _is_plausible_snippet_amount(key: str, snippet: str, total_tagihan_int: Optional[int]) -> bool:
    """Reject clearly implausible amount snippets for component fields."""
    if key not in COMPONENT_FIELD_KEYS:
        return True

    upper = snippet.upper()
    amount = _extract_amount_from_line(snippet)
    if "JUMLAH" in upper and amount is None:
        return False
    if amount is None:
        return True

    cap = max(2_000_000, int(total_tagihan_int * 1.5)) if isinstance(total_tagihan_int, int) else 10_000_000
    return amount <= cap


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
    total_tagihan_int: Optional[int],
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
            raw = component["nilai_raw"]
            if _is_plausible_snippet_amount(component_key, raw, total_tagihan_int):
                _append_payload_text(payload, component_key, raw)

    if total_tagihan_raw:
        _append_payload_text(payload, "total", total_tagihan_raw)
        _append_payload_text(payload, "billingan", total_tagihan_raw)

    field_patterns = _payload_keyword_map()

    for index, upper_line in enumerate(upper_lines):
        line = lines[index]
        for key, patterns in field_patterns.items():
            if any(pattern in upper_line for pattern in patterns):
                if (
                    _score_snippet_for_key(key, line) >= _EVIDENCE_MIN_SCORE.get(key, 1)
                    and _is_plausible_snippet_amount(key, line, total_tagihan_int)
                ):
                    _append_payload_text(payload, key, line)
                if key in {"billingan", "rekap_billingan", "koding"} and index + 1 < len(lines):
                    next_line = lines[index + 1]
                    if (
                        _score_snippet_for_key(key, next_line) >= _EVIDENCE_MIN_SCORE.get(key, 1)
                        and _is_plausible_snippet_amount(key, next_line, total_tagihan_int)
                    ):
                        _append_payload_text(payload, key, next_line)

    contexts = keyword_context if keyword_context is not None else extract_keyword_context_payload(text)
    for key in OCR_PAYLOAD_KEYS:
        ranked_contexts = _rank_evidence_for_key(key, contexts.get(key, []), max_items=8)
        for snippet in ranked_contexts:
            if _is_plausible_snippet_amount(key, snippet, total_tagihan_int):
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


def detect_episode_type(text: str) -> str:
    """Classify episode type as ranap/rajal when signals are clear."""
    upper = text.upper()
    explicit_ranap = bool(re.search(r"\bRAWAT\s+INAP\b", upper))
    explicit_rajal = bool(re.search(r"\bRAWAT\s+JALAN\b", upper))
    if explicit_ranap and not explicit_rajal:
        return "ranap"
    if explicit_rajal and not explicit_ranap:
        return "rajal"

    ranap_score = _count_profile_markers(text, "ranap")
    rajal_score = _count_profile_markers(text, "rajal")

    if re.search(r"\bICU\b|\bNICU\b|\bHCU\b|\bVISITE\b", upper):
        ranap_score += 1
    if re.search(r"\bPOLI\b|\bPOLIKLINIK\b|\bKONSULTASI\b", upper):
        rajal_score += 1

    if ranap_score >= rajal_score + 2:
        return "ranap"
    if rajal_score >= ranap_score + 2:
        return "rajal"
    return "unknown"


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
        payload_parts = [
            part
            for part in _split_evidence(payload_value, max_items=40)
            if _is_plausible_snippet_amount(key, part, total_tagihan_int)
        ]
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

        if key in component_keys:
            component = komponen_billing.get(key, {})
            if bool(component.get("ditemukan")):
                raw = component.get("nilai_raw")
                numeric = component.get("nilai_int")
                if isinstance(raw, str) and raw.strip():
                    ranked_raw = _rank_evidence_for_key(key, [_squash_whitespace(raw)], max_items=1)
                    if ranked_raw:
                        value = ranked_raw[0]
                        evidence = ranked_raw + [item for item in evidence if item != ranked_raw[0]]
                        evidence = evidence[:max_items]
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


def _quality_label(score: int) -> str:
    """Map quality score into label buckets for easier downstream handling."""
    if score >= 75:
        return "high"
    if score >= DOC_VALIDATION_MIN_SCORE:
        return "medium"
    return "low"


def _build_document_validation(
    *,
    text: str,
    nama: Optional[str],
    document_profile: str,
    total_tagihan_int: Optional[int],
    komponen_billing: dict[str, dict[str, object]],
    ai_field_analysis: dict[str, dict[str, object]],
    extraction_diagnostics: Optional[dict[str, object]],
) -> dict[str, object]:
    """Summarize whether OCR output is usable and document looks like billing."""
    component_hits = _component_hit_count(komponen_billing)
    marker_patterns = (
        r"\bRINCIAN\s+BIAYA\b",
        r"\bTOTAL\s+TAGIHAN\b",
        r"\bNO\.?\s*TAGIHAN\b",
        r"\bNO\.?\s*REKAM\s*MEDIS\b",
        r"\bUNIT\s+LAYANAN\b",
        r"\bLABORATORIUM\b",
        r"\bRADIOLOGI\b",
        r"\bFARMASI\b",
        r"\bKASIR\b",
    )
    marker_hits = sum(1 for pattern in marker_patterns if re.search(pattern, text, flags=re.IGNORECASE))
    rupiah_hits = len(re.findall(r"(?i)\bRP\.?\s*\d", text))
    date_hits = len(re.findall(r"\b\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}\b", text))

    compact = re.sub(r"\s+", "", text)
    noisy_chars = sum(1 for char in compact if char in "{}[]|`~^_")
    noise_ratio = noisy_chars / max(1, len(compact))

    has_name = bool(nama)
    has_total = isinstance(total_tagihan_int, int)
    has_billing_field = ai_field_analysis.get("billingan", {}).get("status") in {"found", "inferred"}
    component_amount_total = _sum_component_amounts(komponen_billing)
    total_candidates = _collect_total_candidates(text)
    conflicting_total_candidates = [
        value
        for value in total_candidates
        if not isinstance(total_tagihan_int, int) or value != total_tagihan_int
    ]

    machine_readable_detected = bool(
        extraction_diagnostics.get("machine_readable_detected")
        if isinstance(extraction_diagnostics, dict)
        else False
    )
    ocr_used = bool(extraction_diagnostics.get("ocr_used") if isinstance(extraction_diagnostics, dict) else False)
    profile_marker_hits = _count_profile_markers(text, document_profile)
    profile_is_known = document_profile in _DOCUMENT_PROFILE_MARKERS
    profile_markers_clear = (not profile_is_known) or profile_marker_hits >= 1

    score = 0
    score += min(24, marker_hits * 4)
    score += min(18, rupiah_hits * 2)
    score += min(14, date_hits * 2)
    score += min(18, component_hits * 3)
    if has_name:
        score += 10
    if has_total:
        score += 14
    if has_billing_field:
        score += 8
    if profile_is_known and profile_markers_clear:
        score += 4
    if machine_readable_detected:
        score += 6
    if ocr_used:
        score += 2
    score -= min(18, int(noise_ratio * 120))
    score = max(0, min(100, score))

    missing_critical_fields: list[str] = []
    if not has_name:
        missing_critical_fields.append("nama")
    if not has_total:
        missing_critical_fields.append("total_tagihan")
    if component_hits == 0:
        missing_critical_fields.append("komponen_billing")

    is_billing_document = (has_total and marker_hits >= 2) or (marker_hits >= 3 and component_hits >= 2)
    ocr_readable = score >= DOC_VALIDATION_MIN_SCORE and (has_total or component_hits >= 2)
    total_component_gap: Optional[int] = None
    total_component_gap_ratio: Optional[float] = None
    component_coverage_ratio: Optional[float] = None
    component_sum_exceeds_total = False
    component_sum_not_reconciled = False
    partial_component_coverage_allowed = False

    if has_total and component_amount_total > 0 and isinstance(total_tagihan_int, int):
        total_component_gap = abs(total_tagihan_int - component_amount_total)
        total_component_gap_ratio = round(total_component_gap / max(1, total_tagihan_int), 4)
        component_coverage_ratio = round(component_amount_total / max(1, total_tagihan_int), 4)
        tolerance = max(5_000, int(total_tagihan_int * 0.05))
        component_sum_exceeds_total = component_amount_total > total_tagihan_int + tolerance
        coverage_threshold = max(50_000, int(total_tagihan_int * 0.5))
        partial_component_coverage_allowed = (
            component_hits >= 2
            and not component_sum_exceeds_total
            and component_coverage_ratio >= 0.75
            and total_component_gap_ratio <= 0.25
        )
        component_sum_not_reconciled = (
            component_hits >= 2
            and component_amount_total >= coverage_threshold
            and not partial_component_coverage_allowed
            and total_component_gap_ratio > 0.1
        )

    blocked_fields: list[str] = list(missing_critical_fields)
    blocker_reasons: list[str] = []

    if conflicting_total_candidates:
        blocked_fields.append("total_tagihan")
        blocker_reasons.append("conflicting_total_candidates")
    if component_sum_exceeds_total:
        blocked_fields.extend(["total_tagihan", "komponen_billing"])
        blocker_reasons.append("component_sum_exceeds_total")
    if component_sum_not_reconciled:
        blocked_fields.extend(["total_tagihan", "komponen_billing"])
        blocker_reasons.append("component_sum_not_reconciled")
    if not is_billing_document:
        blocked_fields.append("document_type")
        blocker_reasons.append("billing_markers_incomplete")
    if not profile_markers_clear:
        blocked_fields.append("document_profile")
        blocker_reasons.append("profile_markers_incomplete")
    if not ocr_readable:
        blocked_fields.append("ocr_quality")
        blocker_reasons.append("ocr_not_reliable_enough")

    blocked_fields = list(dict.fromkeys(blocked_fields))
    blocker_reasons = list(dict.fromkeys(blocker_reasons))
    recommendation = "manual_review" if blocked_fields else "ready_for_automation"

    return {
        "is_billing_document": is_billing_document,
        "ocr_readable": ocr_readable,
        "quality_score": score,
        "quality_level": _quality_label(score),
        "quality_threshold": DOC_VALIDATION_MIN_SCORE,
        "document_profile": document_profile,
        "component_categories_found": component_hits,
        "component_nominal_sum": component_amount_total or None,
        "missing_critical_fields": missing_critical_fields,
        "blocked_fields": blocked_fields,
        "blocker_reasons": blocker_reasons,
        "signals": {
            "billing_marker_hits": marker_hits,
            "rupiah_hits": rupiah_hits,
            "date_hits": date_hits,
            "noise_ratio": round(noise_ratio, 4),
            "profile_marker_hits": profile_marker_hits,
            "total_candidates": total_candidates,
            "conflicting_total_candidates": conflicting_total_candidates,
            "total_component_gap": total_component_gap,
            "total_component_gap_ratio": total_component_gap_ratio,
            "component_coverage_ratio": component_coverage_ratio,
            "component_sum_exceeds_total": component_sum_exceeds_total,
            "component_sum_not_reconciled": component_sum_not_reconciled,
            "partial_component_coverage_allowed": partial_component_coverage_allowed,
        },
        "recommendation": recommendation,
    }


def _build_safe_summary(
    *,
    nama: Optional[str],
    document_profile: str,
    total_tagihan_raw: Optional[str],
    total_tagihan_int: Optional[int],
    jenis_layanan: str,
    komponen_billing: dict[str, dict[str, object]],
    document_validation: dict[str, object],
) -> dict[str, object]:
    """Return the automation-safe summary that downstream systems should trust."""
    approved = document_validation.get("recommendation") == "ready_for_automation"
    if approved:
        safe_components = {
            key: _copy_component_result(component)
            for key, component in komponen_billing.items()
        }
    else:
        safe_components = {
            key: _redact_component_result(component)
            for key, component in komponen_billing.items()
        }

    return {
        "approved": approved,
        "status": "approved" if approved else "needs_manual_review",
        "nama": nama if approved else None,
        "profil_dokumen": document_profile if approved else "unknown",
        "jenis_layanan": jenis_layanan if approved else "unknown",
        "total_tagihan_raw": total_tagihan_raw if approved else None,
        "total_tagihan_int": total_tagihan_int if approved else None,
        "komponen_billing": safe_components,
        "blocked_fields": list(document_validation.get("blocked_fields", [])),
        "blocker_reasons": list(document_validation.get("blocker_reasons", [])),
    }


def build_ai_bundle(
    text: str,
    *,
    nama: Optional[str],
    document_profile: str,
    total_tagihan_raw: Optional[str],
    total_tagihan_int: Optional[int],
    jenis_layanan: str,
    komponen_billing: dict[str, dict[str, object]],
    ocr_payload: dict[str, str],
    ai_field_analysis: dict[str, dict[str, object]],
    keyword_context: dict[str, list[str]],
    raw_source_text: Optional[str] = None,
    extraction_diagnostics: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """Build a single AI-ready package containing all extracted context."""
    bundle_text = raw_source_text if isinstance(raw_source_text, str) and raw_source_text else text
    raw_text, truncated = _truncate_for_bundle(bundle_text)
    document_validation = _build_document_validation(
        text=text,
        nama=nama,
        document_profile=document_profile,
        total_tagihan_int=total_tagihan_int,
        komponen_billing=komponen_billing,
        ai_field_analysis=ai_field_analysis,
        extraction_diagnostics=extraction_diagnostics,
    )
    bundle = {
        "schema_version": "v2",
        "source": "hospital_billing_ocr",
        "raw_text": raw_text,
        "raw_text_full": bundle_text,
        "raw_text_truncated": truncated,
        "raw_text_chars": len(bundle_text),
        "document_scope": "full_document" if bundle_text != text else "selected_billing_segment",
        "ringkasan_terstruktur": {
            "nama": nama,
            "profil_dokumen": document_profile,
            "jenis_layanan": jenis_layanan,
            "total_tagihan_raw": total_tagihan_raw,
            "total_tagihan_int": total_tagihan_int,
            "komponen_billing": komponen_billing,
        },
        "ringkasan_final": _build_safe_summary(
            nama=nama,
            document_profile=document_profile,
            total_tagihan_raw=total_tagihan_raw,
            total_tagihan_int=total_tagihan_int,
            jenis_layanan=jenis_layanan,
            komponen_billing=komponen_billing,
            document_validation=document_validation,
        ),
        "field_mentah": ocr_payload,
        "field_status": ai_field_analysis,
        "keyword_context": keyword_context,
        "document_profile": document_profile,
        "document_validation": document_validation,
    }
    if extraction_diagnostics:
        bundle["extraction_diagnostics"] = extraction_diagnostics
    return bundle


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


def _extract_text_via_ocr(pdf_bytes: bytes) -> tuple[str, list[str]]:
    """OCR fallback for image-based PDFs."""
    try:
        import fitz  # type: ignore[import-not-found]
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image, ImageFilter, ImageOps
    except Exception:
        return "", []

    def close_images(images: list[object]) -> None:
        """Close PIL images safely while avoiding duplicate closes."""
        seen: set[int] = set()
        for image_obj in images:
            image_id = id(image_obj)
            if image_id in seen:
                continue
            seen.add(image_id)
            try:
                image_obj.close()  # type: ignore[attr-defined]
            except Exception:
                continue

    def target_page_indices(page_count: int) -> list[int]:
        """Prioritize likely pages containing identity, totals, and components."""
        if OCR_ENRICH_ALWAYS:
            return list(range(page_count))

        middle = page_count // 2
        candidates = [
            page_count - 1,
            page_count - 2,
            page_count - 3,
            0,
            middle,
            1,
            middle + 1,
            middle - 1,
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

    def extract_name_hint(images: list[object]) -> Optional[str]:
        """Read focused billing bands to improve patient-name OCR."""
        crop_regions = [
            ("left_focus", (0.0, 0.18, 0.58, 0.32), (11, 6)),
            ("full_band", (0.0, 0.18, 1.0, 0.38), (11, 6)),
        ]
        best_name: Optional[str] = None
        best_score = (-1, -1)

        for variant in images:
            try:
                width, height = variant.size  # type: ignore[attr-defined]
            except Exception:
                continue

            for crop_label, region, psm_modes in crop_regions:
                left_ratio, top_ratio, right_ratio, bottom_ratio = region
                left = int(width * left_ratio)
                top = int(height * top_ratio)
                right = int(width * right_ratio)
                bottom = int(height * bottom_ratio)
                if right <= left or bottom <= top:
                    continue

                crop = None
                try:
                    crop = variant.crop((left, top, right, bottom))  # type: ignore[attr-defined]
                    for psm in psm_modes:
                        crop_text = ""
                        crop_score = -1
                        for lang in (OCR_LANG_PRIMARY, OCR_LANG_FALLBACK):
                            try:
                                candidate = pytesseract.image_to_string(
                                    crop,
                                    lang=lang,
                                    config=f"--oem 1 --psm {psm}",
                                )
                            except pytesseract.TesseractError:
                                continue

                            if not candidate or not candidate.strip():
                                continue

                            score = 5 if "NAMA" in candidate.upper() else 0
                            score += len(candidate.strip())
                            if score > crop_score:
                                crop_score = score
                                crop_text = candidate

                        if not crop_text:
                            continue

                        extracted = extract_nama(crop_text)
                        if not extracted:
                            continue

                        token_count = len(extracted.split())
                        alpha_count = sum(1 for char in extracted if "A" <= char <= "Z")
                        score = (token_count, alpha_count)
                        if score > best_score:
                            best_score = score
                            best_name = extracted

                        if crop_label == "left_focus" and token_count >= 2:
                            return extracted
                finally:
                    if crop is not None:
                        try:
                            crop.close()  # type: ignore[attr-defined]
                        except Exception:
                            pass

        return best_name

    page_texts: list[tuple[int, str]] = []
    name_hints: list[str] = []
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
                variant_images: list[object] = [image]

                name_hint_images: list[object] = [image]
                if OCR_MULTI_PASS:
                    enhanced = ImageOps.autocontrast(image, cutoff=2)
                    sharpened = enhanced.filter(ImageFilter.SHARPEN)
                    binary = enhanced.point(lambda px: 255 if px >= OCR_BINARIZE_THRESHOLD else 0)
                    variant_images.extend([enhanced, sharpened, binary])
                    name_hint_images.append(enhanced)

                ocr_text: Optional[str] = None
                ocr_score = -1
                accepted = False
                for variant in variant_images:
                    for lang in (OCR_LANG_PRIMARY, OCR_LANG_FALLBACK):
                        try:
                            candidate = pytesseract.image_to_string(
                                variant,
                                lang=lang,
                                config=tesseract_config,
                            )
                        except pytesseract.TesseractNotFoundError:
                            close_images(variant_images)
                            return "", []
                        except pytesseract.TesseractError:
                            continue

                        if not candidate or not candidate.strip():
                            continue

                        candidate_score = _score_ocr_candidate(candidate)
                        if candidate_score > ocr_score:
                            ocr_score = candidate_score
                            ocr_text = candidate

                        if candidate_score >= OCR_PAGE_ACCEPT_SCORE:
                            accepted = True
                            break
                    if accepted:
                        break

                should_probe_name = bool(
                    ocr_text
                    and any(
                        marker in ocr_text.upper()
                        for marker in ("NAMA", "RINCIAN BIAYA", "NO. TAGIHAN")
                    )
                )
                name_hint = extract_name_hint(name_hint_images) if should_probe_name else None
                if name_hint:
                    name_hints.append(name_hint)

                close_images(variant_images)
                del pix

                if ocr_text:
                    page_texts.append((page_index, ocr_text.strip()))
    except Exception:
        return "", []

    ordered_pages = [
        f"=== PAGE {page_index + 1} ===\n{page_text}"
        for page_index, page_text in sorted(page_texts, key=lambda item: item[0])
        if page_text
    ]
    return "\n".join(ordered_pages).strip(), name_hints


def extract_text_from_pdf(pdf_bytes: bytes) -> TextExtractionResult:
    """Read PDF text, then fallback to OCR for image-based files."""
    extraction_error: Optional[PDFTextExtractionError] = None
    primary_text = ""
    secondary_text = ""
    ocr_text = ""
    ocr_name_hints: list[str] = []
    ocr_attempted = False
    try:
        primary_text = _extract_text_via_pdfplumber(pdf_bytes)
    except PDFTextExtractionError as exc:
        extraction_error = exc

    secondary_text = _extract_text_via_pymupdf(pdf_bytes)
    merged_text = _merge_text_sources(primary_text, secondary_text)

    if _needs_ocr_enrichment(merged_text):
        ocr_attempted = True
        ocr_text, ocr_name_hints = _extract_text_via_ocr(pdf_bytes)
        if ocr_text:
            merged_text = _merge_text_sources(merged_text, ocr_text)
        elif extraction_error is not None and not merged_text:
            raise extraction_error

    diagnostics = _build_extraction_diagnostics(
        primary_text=primary_text,
        secondary_text=secondary_text,
        ocr_text=ocr_text,
        merged_text=merged_text,
        ocr_attempted=ocr_attempted,
        name_hints=ocr_name_hints,
    )
    return TextExtractionResult(text=merged_text, diagnostics=diagnostics)


def parse_billing_text(
    text: str,
    *,
    extraction_diagnostics: Optional[dict[str, object]] = None,
) -> ParsedBillingFields:
    """Parse billing text into normalized name and total fields."""
    base_focused_text = select_primary_billing_text(text)
    document_profile = detect_episode_type(base_focused_text)
    if document_profile == "unknown":
        document_profile = detect_episode_type(text)
    focused_text = _select_profile_billing_text(base_focused_text, document_profile)
    if not focused_text.strip():
        focused_text = base_focused_text

    nama = (
        _extract_name_hint_from_diagnostics(extraction_diagnostics)
        or extract_nama(focused_text)
        or extract_nama(base_focused_text)
        or extract_nama(text)
    )
    total_tagihan_raw, total_tagihan_int = extract_total_tagihan(focused_text)
    if total_tagihan_raw is None or total_tagihan_int is None:
        fallback_raw, fallback_int = extract_total_tagihan(base_focused_text)
        total_tagihan_raw = total_tagihan_raw or fallback_raw
        total_tagihan_int = total_tagihan_int if total_tagihan_int is not None else fallback_int
    if total_tagihan_raw is None or total_tagihan_int is None:
        fallback_raw, fallback_int = extract_total_tagihan(text)
        total_tagihan_raw = total_tagihan_raw or fallback_raw
        total_tagihan_int = total_tagihan_int if total_tagihan_int is not None else fallback_int

    komponen_billing = extract_billing_components(focused_text, total_tagihan_int=total_tagihan_int)
    _apply_component_fallbacks(
        focused_text,
        components=komponen_billing,
        total_tagihan_int=total_tagihan_int,
    )
    _apply_document_profile_to_components(
        komponen_billing,
        document_profile=document_profile,
    )
    jenis_layanan = document_profile or detect_episode_type(focused_text)
    if jenis_layanan == "unknown":
        jenis_layanan = detect_episode_type(text)
    keyword_context = extract_keyword_context_payload(focused_text)
    ocr_payload = extract_ocr_payload(
        focused_text,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
        komponen_billing=komponen_billing,
        keyword_context=keyword_context,
    )
    ai_field_analysis = build_ai_field_analysis(
        focused_text,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
        komponen_billing=komponen_billing,
        ocr_payload=ocr_payload,
    )
    ai_bundle = build_ai_bundle(
        focused_text,
        nama=nama,
        document_profile=document_profile,
        jenis_layanan=jenis_layanan,
        total_tagihan_raw=total_tagihan_raw,
        total_tagihan_int=total_tagihan_int,
        komponen_billing=komponen_billing,
        ocr_payload=ocr_payload,
        ai_field_analysis=ai_field_analysis,
        keyword_context=keyword_context,
        raw_source_text=text,
        extraction_diagnostics=extraction_diagnostics,
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
