"""Regression tests for API response payload shape."""

from __future__ import annotations

import unittest

from app.main import _build_response


class ParseBillingResponseShapeTests(unittest.TestCase):
    """Validate the HTTP response exposes easy-to-consume AI payloads."""

    def test_response_exposes_bahan_mentah_ekstrak(self) -> None:
        payload = _build_response(
            success=True,
            message="ok",
            chat_id="123",
            file_name="sample.pdf",
            ai_bundle={
                "raw_text": "TRUNCATED OCR",
                "raw_text_full": "RAW OCR FULL",
            },
        )

        data = payload.model_dump()

        self.assertNotIn("ai_bundle", data)
        self.assertEqual({"output": "RAW OCR FULL"}, data["bahanmentahekstrak"])


if __name__ == "__main__":
    unittest.main()
