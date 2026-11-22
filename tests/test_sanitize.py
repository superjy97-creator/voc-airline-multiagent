import hashlib
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from voc import sanitize


class SanitizeTest(unittest.TestCase):
    def test_each_pattern_returns_codes_only(self):
        samples = {
            "resident_id": "991332-1123456",
            "phone": "010-0000-0000",
            "email": "test@example.invalid",
            "card": "0000 0000 0000 0000",
            "account": "계좌번호 123-456-7890",
            "pnr": "Z8RT9X",
            "flight": "KE123",
            "membership": "회원번호: DEMO-0000",
            "address": "가상시 가상구 가상로 12-3",
            "url": "https://example.invalid/demo",
        }
        for code, value in samples.items():
            with self.subTest(code=code):
                result = sanitize.scan_text(value)
                self.assertIsInstance(result, tuple)
                self.assertIn(code, result)
                self.assertNotIn(value, repr(result))
        self.assertIn("phone", sanitize.scan_text("02-000-0000"))

    def test_aggregate_numbers_and_dates_are_not_flagged(self):
        harmless = [
            "전체 654321건 중 항공 12345건(비율 0.018868), 분모 543210행",
            "기간 2016-01-01 ~ 2021-01-12, 월별 2020-02=1234, 2020-03=1345",
            "2020년 2월 1,234건, 3월 1,345건으로 약 9.0% 증가 (654,321 / 12,345)",
        ]
        for text in harmless:
            with self.subTest(text=text):
                self.assertEqual(sanitize.scan_text(text), ())
                self.assertTrue(sanitize.make_checked_text(text).pii_scan_passed)

    def test_narrowed_patterns_still_catch_real_shapes(self):
        cases = {
            "pnr": ["예약 AB12CD 확인", "XKJLMN", "12345A", "예약번호 654321", "PNR: 98765"],
            "account": ["입금 110-123-456789", "계좌번호 1002345678901", "123-45-678901",
                        "입금처 1234-56-78", "1999-13-45", "2020-02-30-1234"],
            "phone": ["연락처 010-1234-5678"],
        }
        for code, texts in cases.items():
            for text in texts:
                with self.subTest(code=code, text=text):
                    self.assertIn(code, sanitize.scan_text(text))

    def test_name_limitation_and_fail_closed_flag(self):
        doc = sanitize.__doc__
        assert doc is not None
        self.assertIn("이름", doc)
        self.assertIn("미지원", doc)
        self.assertTrue(sanitize.make_checked_text("합성 상담 예시").pii_scan_passed)
        self.assertFalse(sanitize.make_checked_text("test@example.invalid").pii_scan_passed)

    def test_quarantine_contains_hash_and_codes(self):
        identifier = "합성-사건번호"
        text = "test@example.invalid"
        entry = sanitize.quarantine_entry(identifier, sanitize.scan_text(text))
        self.assertEqual(entry, {
            "source_id_hash": hashlib.sha256(identifier.encode()).hexdigest()[:16],
            "reasons": ("email",),
        })
        self.assertNotIn(identifier, repr(entry))
        self.assertNotIn(text, repr(entry))
        with self.assertRaises(ValueError):
            sanitize.quarantine_entry(identifier, (text,))
