"""실제 자료를 사용하지 않는 데이터 적합성 집계 테스트."""

import csv
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import profile_data


class ProfileDataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temp.cleanup)
        self.raw = Path(temp.name)
        self.code_names = defaultdict(set)

    def make_csv(self, name, header, rows):
        path = self.raw / f"{name}.csv"
        with path.open("w", encoding="cp949", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_exact_match_and_dates_across_chunks(self):
        path = self.make_csv("상담", [
            "번호(ACCIDENT_NO)", "일자(RCPT_YMD)", "코드(ITEM_CODE)", "이름(ITEM_NAME)",
            "제목(ACCIDENT_TITLE)",
        ], [
            ["001", "2020-02-29", "01", "항공 여객\t운송서비스", "가짜 제목\n두 번째 줄"],
            ["001", "20200301", "02", "여객운송서비스", "가짜 제목"],
            ["002", "20190101", "03", "항공화물운송서비스", "항공여객운송서비스"],
            ["003", "20200101", "04", "", "가짜 제목"],
        ])
        report = profile_data.profile_consultations(path, self.code_names, 1)
        self.assertEqual(report["air_exact_rows"], 1)
        self.assertEqual(report["non_air_rows"], 2)
        self.assertEqual(report["undetermined_reasons"]["missing_name"], 1)
        self.assertEqual(report["total_period"], ["2019-01-01", "2020-03-01"])
        self.assertEqual(report["air_exact_period"], ["2020-02-29", "2020-02-29"])
        self.assertEqual(report["unique_accident_ids"], 3)
        self.assertEqual(report["duplicate_accident_rows"], 1)

    def test_separate_denominators(self):
        monthly = self.make_csv("월별", ["연도", "월", "품목명", "상담건수"], [
            ["2020", "2", "항공여객운송서비스", "1,200"], ["2020", "12", "항공화물운송서비스", "50"],
        ])
        answers = self.make_csv("표준답변", ["품목명", "질문", "답변"], [
            ["가전", "항공", "가짜 답변"], ["항공여객운송서비스", "가짜 질문", "가짜 답변"],
            ["여객운송", "가짜 질문", "가짜 답변"],
        ])
        first = profile_data.profile_monthly(monthly, self.code_names, 1)
        second = profile_data.profile_standard_answers(answers, self.code_names, 1)
        self.assertEqual((first["total_rows"], first["air_exact_rows"]), (2, 1))
        self.assertEqual((second["total_rows"], second["air_exact_rows"]), (3, 1))
        self.assertEqual(first["consultation_count_sums"], {
            "total": 1250, "air_exact": 1200, "invalid": 0,
        })
        self.assertEqual(first["total_period"], ["2020-02", "2020-12"])
        self.assertEqual(second["total_period"], [None, None])

    def test_blocked_columns_rejected_before_pandas(self):
        with patch.object(profile_data.pd, "read_csv") as reader:
            for column in [*profile_data.BLOCKED_COLUMNS, "사건내용(ACCIDENT_CONTENT)",
                           "질문(ITEM_NAME)", "기타서술"]:
                with self.subTest(column=column), self.assertRaises(ValueError):
                    profile_data.read_chunks(self.raw / "미존재.csv", [column], 1)
            reader.assert_not_called()

    def test_cp949_and_projection(self):
        path = self.make_csv("표준답변", ["품목명", "질문"], [["힣항공", "가짜 질문"]])
        self.assertIn("힣항공".encode("cp949"), path.read_bytes())
        with patch.object(profile_data.pd, "read_csv", wraps=profile_data.pd.read_csv) as reader:
            report = profile_data.profile_standard_answers(path, self.code_names, 1)
        self.assertEqual(report["air_exact_rows"], 0)
        self.assertEqual(reader.call_args.kwargs["usecols"], ["품목명"])
        self.assertEqual(reader.call_args.kwargs["encoding"], "cp949")

    def test_missing_and_ambiguous_codes(self):
        header = ["ACCIDENT_NO", "RCPT_YMD", "ITEM_CODE", "ITEM_NAME"]
        receipt = self.make_csv("접수내역", header, [
            ["1", "20200101", "01", "항공여객운송서비스"], ["2", "20200101", "02", "가전"],
        ])
        online = self.make_csv("온라인", header, [
            ["3", "20200101", "01", "항공여객운송서비스"], ["4", "20200101", "02", "항공기"],
        ])
        profile_data.profile_consultations(receipt, self.code_names, 1)
        profile_data.profile_consultations(online, self.code_names, 1)
        path = self.make_csv("코드만", ["(ACCIDENT_NO)", "(RCPT_YMD)", "(ITEM_CODE)"], [
            ["1", "20200230", "01"], ["1", "20200101", "02"],
            [" ", " ", "99"], ["", "20200102", ""],
        ])
        report = profile_data.profile_consultations(path, self.code_names, 2, code_only=True)
        self.assertEqual(report["total_rows"], 4)
        self.assertEqual(report["air_exact_rows"], 1)
        self.assertEqual(report["undetermined_rows"], 3)
        self.assertEqual(report["undetermined_reasons"], {
            "missing_code": 1, "unregistered_code": 1, "conflicting_code": 1,
            "missing_name": 0, "no_subcategory": 0,
        })
        self.assertEqual(report["missing"], {"ACCIDENT_NO": 2, "RCPT_YMD": 1, "ITEM_CODE": 1})
        self.assertEqual(report["invalid_dates"], 1)
        self.assertEqual(report["air_exact_period"], [None, None])

    def test_relief_uses_case_id_but_has_no_receipt_period(self):
        path = self.make_csv("피해구제", ["사건ID", "품목코드", "품목명", "사건제목", "갱신일시"], [
            ["A", "01", "여객운송서비스", "가짜 제목", "20260101"],
            ["A", "01", "철도여객운송서비스", "가짜 제목", "20260102"],
            ["B", "02", "", "항공", "20260103"],
        ])
        report = profile_data.profile_relief(path, self.code_names, 2)
        self.assertIsNone(report["air_exact_rows"])
        self.assertEqual(report["ambiguous_rows"], 1)
        self.assertEqual(report["undetermined_rows"], 2)
        self.assertEqual(report["undetermined_reasons"]["no_subcategory"], 1)
        self.assertEqual(report["undetermined_reasons"]["missing_name"], 1)
        self.assertEqual(report["non_air_rows"], 1)
        self.assertEqual(report["unique_accident_ids"], 2)
        self.assertEqual(report["duplicate_accident_rows"], 1)
        self.assertIsNone(report["period_basis"])
        self.assertEqual(self.code_names, {})
