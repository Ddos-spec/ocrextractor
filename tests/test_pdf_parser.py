"""Regression tests for OCR safety gates in the billing parser."""

from __future__ import annotations

import unittest

from app.services.pdf_parser import parse_billing_text


class ParseBillingSafetyTests(unittest.TestCase):
    """Validate that uncertain OCR stays blocked while clean text can pass."""

    def test_noisy_raw_ocr_requires_manual_review(self) -> None:
        raw_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: SITI AMINAH",
                "Farmasi",
                "Jumlah Rp. 25.000",
                "Laboratorium",
                "Jumlah Rp. 30.000",
                "Total Tagihan Rp. 100.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            raw_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        validation = parsed.ai_bundle["document_validation"]
        safe_summary = parsed.ai_bundle["ringkasan_final"]

        self.assertEqual("manual_review", validation["recommendation"])
        self.assertFalse(safe_summary["approved"])
        self.assertIn("total_tagihan", safe_summary["blocked_fields"])
        self.assertIsNone(safe_summary["total_tagihan_int"])

    def test_clean_billing_text_is_approved(self) -> None:
        clean_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: BUDI SANTOSO",
                "Farmasi",
                "Jumlah Rp. 25.000",
                "Laboratorium",
                "Jumlah Rp. 75.000",
                "Total Tagihan Rp. 100.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            clean_text,
            extraction_diagnostics={
                "machine_readable_detected": True,
                "ocr_used": False,
            },
        )

        validation = parsed.ai_bundle["document_validation"]
        safe_summary = parsed.ai_bundle["ringkasan_final"]

        self.assertEqual("ready_for_automation", validation["recommendation"])
        self.assertTrue(safe_summary["approved"])
        self.assertEqual("BUDI SANTOSO", safe_summary["nama"])
        self.assertEqual(100000, safe_summary["total_tagihan_int"])

    def test_rajal_profile_is_reported_for_outpatient_template(self) -> None:
        clean_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: BUDI SANTOSO",
                "Rawat Jalan",
                "Poli Penyakit Dalam",
                "Jumlah Poli Penyakit Dalam Rp. 75.000",
                "Farmasi",
                "Jumlah Rp. 25.000",
                "Total Tagihan Rp. 100.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            clean_text,
            extraction_diagnostics={
                "machine_readable_detected": True,
                "ocr_used": False,
            },
        )

        safe_summary = parsed.ai_bundle["ringkasan_final"]

        self.assertTrue(safe_summary["approved"])
        self.assertEqual("rajal", safe_summary["profil_dokumen"])
        self.assertFalse(safe_summary["komponen_billing"]["kamar_akomodasi"]["ditemukan"])

    def test_igd_template_is_treated_as_rajal(self) -> None:
        clean_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: BUDI SANTOSO",
                "Gawat Darurat",
                "IGD Triase",
                "Elektro Kardiografi (EKG) 28/06/2025 1,00 Rp. 68.000 Rp. 68.000",
                "Jumlah Rp. 198.000",
                "Farmasi",
                "Jumlah Rp. 27.372",
                "Total Tagihan Rp. 225.372",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            clean_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        structured = parsed.ai_bundle["ringkasan_terstruktur"]
        validation = parsed.ai_bundle["document_validation"]

        self.assertEqual("rajal", structured["profil_dokumen"])
        self.assertGreaterEqual(validation["signals"]["profile_marker_hits"], 1)

    def test_ranap_profile_is_reported_for_inpatient_template(self) -> None:
        clean_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: BUDI SANTOSO",
                "Rawat Inap",
                "Kamar Akomodasi",
                "Jumlah Kamar Rp. 75.000",
                "Farmasi",
                "Jumlah Rp. 25.000",
                "Total Tagihan Rp. 100.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            clean_text,
            extraction_diagnostics={
                "machine_readable_detected": True,
                "ocr_used": False,
            },
        )

        safe_summary = parsed.ai_bundle["ringkasan_final"]

        self.assertTrue(safe_summary["approved"])
        self.assertEqual("ranap", safe_summary["profil_dokumen"])
        self.assertEqual(75000, safe_summary["komponen_billing"]["kamar_akomodasi"]["nilai_int"])

    def test_tail_pages_are_prioritized_over_earlier_noise(self) -> None:
        mixed_text = "\n".join(
            [
                "=== PAGE 1 ===",
                "KEMENTERIAN KESEHATAN REPUBLIK INDONESIA",
                "Total Tagihan Rp. 9.999.999",
                "=== PAGE 2 ===",
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "Nama Pasien: SITI AMINAH",
                "Laboratorium",
                "Jumlah Rp. 20.000",
                "Farmasi",
                "Jumlah Rp. 30.000",
                "Total Tagihan Rp. 50.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        safe_summary = parsed.ai_bundle["ringkasan_final"]

        self.assertTrue(safe_summary["approved"])
        self.assertEqual("SITI AMINAH", safe_summary["nama"])
        self.assertEqual(50000, safe_summary["total_tagihan_int"])

    def test_numbered_pages_are_sorted_before_tail_selection(self) -> None:
        mixed_text = "\n".join(
            [
                "=== PAGE 7 ===",
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "Nama Pasien: SITI AMINAH",
                "Farmasi",
                "Jumlah Rp. 20.000",
                "Laboratorium",
                "Jumlah Rp. 30.000",
                "Total Tagihan Rp. 50.000",
                "Kasir",
                "=== PAGE 1 ===",
                "KEMENTERIAN KESEHATAN REPUBLIK INDONESIA",
                "Total Tagihan Rp. 9.999.999",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        safe_summary = parsed.ai_bundle["ringkasan_final"]

        self.assertTrue(safe_summary["approved"])
        self.assertEqual("SITI AMINAH", safe_summary["nama"])
        self.assertEqual(50000, safe_summary["total_tagihan_int"])

    def test_total_header_does_not_steal_row_amount(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "Nama Pasien: SITI AMINAH",
                "Unit Layanan Item/Tindakan Pemeriksaan Tanggal Jml Tarif/Harga Total Tagihan",
                "Tablet Tambah Darah 08/06/2025 2.00 Rp. 250 Rp. 500",
                "Farmasi",
                "Jumlah Rp. 500",
                "Total Tagihan Rp. 247.010",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        safe_summary = parsed.ai_bundle["ringkasan_final"]

        self.assertTrue(safe_summary["approved"])
        self.assertEqual("SITI AMINAH", safe_summary["nama"])
        self.assertEqual(247010, safe_summary["total_tagihan_int"])

    def test_total_trims_trailing_ocr_digit(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "Nama Pasien: SITI AMINAH",
                "Farmasi",
                "Jumlah Rp. 5.030.087",
                "Total Tagihan Rp. 5.030.087 1",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        safe_summary = parsed.ai_bundle["ringkasan_final"]

        self.assertTrue(safe_summary["approved"])
        self.assertEqual(5030087, safe_summary["total_tagihan_int"])

    def test_clear_total_can_pass_with_partial_component_coverage(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: SITI AMINAH",
                "Farmasi",
                "Jumlah Rp. 40.000",
                "Laboratorium",
                "Jumlah Rp. 40.000",
                "Total Tagihan Rp. 100.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        validation = parsed.ai_bundle["document_validation"]
        safe_summary = parsed.ai_bundle["ringkasan_final"]

        self.assertEqual("ready_for_automation", validation["recommendation"])
        self.assertTrue(validation["signals"]["partial_component_coverage_allowed"])
        self.assertTrue(safe_summary["approved"])
        self.assertEqual(100000, safe_summary["total_tagihan_int"])

    def test_name_hint_can_override_weaker_inline_ocr_name(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama SETIKOM Tgl. Tagihan 02/06/2025 10.49.18",
                "Jumlah Poli Penyakit Dalam Rp. 125.000",
                "Total Tagihan Rp. 125.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
                "name_hints": ["SETIKDM"],
            },
        )

        self.assertEqual("SETIK DM", parsed.nama)

    def test_name_exclamation_tail_can_be_normalized_to_i(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama NAHROW! Cara Bayar BPJS / JKN",
                "Farmasi",
                "Jumlah Rp. 27.372",
                "Total Tagihan Rp. 225.372",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        self.assertEqual("NAHROWI", parsed.nama)

    def test_name_hint_prefers_more_complete_candidate(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama SETIKOM Tgl. Tagihan 02/06/2025 10.49.18",
                "Jumlah Poli Penyakit Dalam Rp. 125.000",
                "Total Tagihan Rp. 125.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
                "name_hints": ["SETIKOM", "SETIKDM"],
            },
        )

        self.assertEqual("SETIK DM", parsed.nama)

    def test_name_hint_drops_tgl_tail_noise(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama SETIKOM Tgl. Tagihan 02/06/2025 10.49.18",
                "Jumlah Poli Penyakit Dalam Rp. 125.000",
                "Total Tagihan Rp. 125.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
                "name_hints": ["SETIKDM TA"],
            },
        )

        self.assertEqual("SETIK DM", parsed.nama)

    def test_pharmacy_item_sum_can_replace_broken_subtotal(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: SITI AMINAH",
                "Jumlah Poli Penyakit Dalam Rp. 125.000",
                "Farmasi",
                "Lansoprazole Kapsul 02/06/2025 20.00 Rp. 461 Rp. 9.220",
                "Alprazolam Tablet 02/06/2025 5.00 Rp. 732 Rp. 3.660",
                "Sucralfate Sirup 02/06/2025 1.00 Rp. 8.580 Rp. 8.580",
                "Jumlah Farmasi Rp. 1140",
                "Total Tagihan Rp. 146.460",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        self.assertEqual(21460, parsed.komponen_billing["obat"]["nilai_int"])
        self.assertTrue(parsed.ai_bundle["ringkasan_final"]["approved"])

    def test_pharmacy_summary_ignores_preheader_obat_rows_and_nested_bmhp(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: NAHROWI",
                "Gawat Darurat",
                "Memasukkan Obat 28/06/2025 15:33:25 1,00 Rp. 35.000 Rp. 35.000",
                "Jumlah Rp. 198.000",
                "Farmasi",
                "Paracetamol 28/06/2025 10,00 Rp. 182 Rp. 1.820",
                "Disposable syringe with needle 28/06/2025 1,00 Rp. 1.332 Rp. 1.332",
                "IV Catheter 28/06/2025 2,00 Rp. 7.992 Rp. 15.984",
                "Jumlah Rp. 27.372",
                "Total Tagihan Rp. 225.372",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        self.assertEqual(27372, parsed.komponen_billing["obat"]["nilai_int"])
        self.assertIsNone(parsed.komponen_billing["bmhp"]["nilai_int"])

    def test_igd_fallbacks_can_reconcile_total_with_consultation_and_actions(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: NAHROWI",
                "Gawat Darurat",
                "IGD Triase",
                "DOKDER UMUM 28/06/2025 15:32:54 1,00 Rp. 85.000 Rp. 85.000",
                "Elektro Kardiografi (EKG) 28/06/2025 15:33:02 1,00 Rp. 68.000 Rp. 68.000",
                "Pemasangan Infus Standar 28/06/2025 18:33:15 1,00 Rp. 10.000 Rp. 10.000",
                "Memasukkan Obat 28/06/2025 15:33:25 1,00 Rp. 35.000 Rp. 35.000",
                "Jumlah Rp. 198.000",
                "Farmasi",
                "Paracetamol 28/06/2025 10,00 Rp. 182 Rp. 1.820",
                "Disposable syringe with needle 28/06/2025 1,00 Rp. 1.332 Rp. 1.332",
                "IV Catheter 28/06/2025 2,00 Rp. 7.992 Rp. 15.984",
                "Ketorolac 28/06/2025 1,00 Rp. 1.620 Rp. 1.620",
                "Omeprazole 28/06/2025 6,00 Rp. 258 Rp. 1.548",
                "Vitamin B Kompleks 29/06/2025 10,00 Rp. 63 Rp. 630",
                "Ranitidin 28/06/2025 1,00 Rp. 2.040 Rp. 2.040",
                "Jumlah Rp. 27.372",
                "Total Tagihan Rp. 225.372",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        self.assertEqual(85000, parsed.komponen_billing["konsultasi"]["nilai_int"])
        self.assertEqual(45000, parsed.komponen_billing["prosedur_non_bedah"]["nilai_int"])
        self.assertEqual(68000, parsed.komponen_billing["penunjang"]["nilai_int"])
        self.assertEqual(27372, parsed.komponen_billing["obat"]["nilai_int"])
        self.assertTrue(parsed.ai_bundle["ringkasan_final"]["approved"])

    def test_single_rp_tail_with_two_amounts_uses_line_total(self) -> None:
        mixed_text = "\n".join(
            [
                "RINCIAN BIAYA PELAYANAN PASIEN",
                "No. Rekam Medis 123456",
                "Nama Pasien: NAHROWI",
                "Gawat Darurat",
                "Memasukkan Obat 28/06/2025 15:33:25 1,00 Rp. 35000 35.000",
                "Farmasi",
                "Jumlah Rp. 0",
                "Total Tagihan Rp. 35.000",
                "Kasir",
            ]
        )

        parsed = parse_billing_text(
            mixed_text,
            extraction_diagnostics={
                "machine_readable_detected": False,
                "ocr_used": True,
            },
        )

        self.assertEqual(35000, parsed.komponen_billing["prosedur_non_bedah"]["nilai_int"])


if __name__ == "__main__":
    unittest.main()
