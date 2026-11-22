import io
import json
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_live_smoke import FakeSdk

from scripts import live_real as lr
from scripts import live_smoke as ls
from voc.contracts import (
    INTERPRETATION_RULE,
    REQUEST_SECTIONS,
    SCHEMAS,
    SanitizedInput,
    SourceKind,
    StageName,
    next_step,
)
from voc.stats import DatasetStats, save_summary, to_sanitized_monthly

NOTES_CAPPED = ("행 수가 내보내기 상한으로 잘렸을 가능성이 있습니다.",)
PRICES = ["--price-in", "0.10", "--price-out", "0.50", "--max-cost-usd", "0.05"]


def make_stats(dataset_id="receipt", end="2020-03-31", notes=()):
    return DatasetStats(dataset_id, "시험 자료", "개별 상담 기록", 1000, 12, 3, "2020-01-01", end,
                        (("2020-01", 5), ("2020-03", 7)), 12, 0, notes)


def no_network(*_args, **_kwargs):
    raise AssertionError("네트워크 호출이 시도되었습니다.")


class MonthlyPayloadTest(unittest.TestCase):
    def fields(self, **kw):
        return dict(to_sanitized_monthly(make_stats(**kw), SourceKind.REAL).fields)

    def test_monthly_counts_are_explicit_and_missing_months_are_zero(self):
        fields = self.fields()
        self.assertEqual((fields["건수_2020-01"], fields["건수_2020-02"], fields["건수_2020-03"]), (5, 0, 7))
        self.assertEqual((fields["건수"], fields["전체건수"], fields["판정불가건수"]), (12, 1000, 3))
        self.assertEqual(fields["비율"], 0.012)

    def test_data_limits_are_passed_as_enumerated_values(self):
        self.assertEqual(self.fields()["마지막월_부분집계"], "아니오")
        self.assertEqual(self.fields(end="2020-03-17")["마지막월_부분집계"], "예")
        self.assertEqual(self.fields()["행수_상한_절단_가능성"], "미확인")        # '아니오'라고 단정하지 않는다
        self.assertEqual(self.fields(notes=NOTES_CAPPED)["행수_상한_절단_가능성"], "예")
        fields = self.fields()
        self.assertEqual((fields["요청_절구성"], fields["해석_원칙"]), (REQUEST_SECTIONS, INTERPRETATION_RULE))

    def test_every_field_belongs_to_the_schema_and_nothing_is_free_text(self):
        schema = SCHEMAS["voc_monthly_v1"]
        for name, value in to_sanitized_monthly(make_stats(), SourceKind.REAL).fields:
            self.assertIn(name, schema)
            self.assertTrue(schema[name].accepts(value), name)
        self.assertFalse(any(isinstance(v, str) and len(v) > 80
                             for n, v in to_sanitized_monthly(make_stats(), SourceKind.REAL).fields
                             if n not in ("요청_절구성", "해석_원칙")))

    def test_inconsistent_totals_are_rejected(self):
        wrong_sum = DatasetStats("receipt", "x", "u", 1000, 99, 0, "2020-01-01", "2020-03-31",
                                 (("2020-01", 5), ("2020-03", 7)), 99, 0, ())
        too_many = DatasetStats("receipt", "x", "u", 10, 12, 0, "2020-01-01", "2020-03-31",
                                (("2020-01", 5), ("2020-03", 7)), 12, 0, ())
        for stats in (wrong_sum, too_many):
            with self.assertRaises(ValueError):
                to_sanitized_monthly(stats, SourceKind.REAL)

    def test_unknown_dataset_and_missing_period_are_rejected(self):
        with self.assertRaises(ValueError):
            to_sanitized_monthly(make_stats(dataset_id="other"), SourceKind.REAL)
        empty = DatasetStats("receipt", "x", "u", 1, 0, 0, None, None, (), None, None, ())
        with self.assertRaises(ValueError):
            to_sanitized_monthly(empty, SourceKind.REAL)

    def test_schema_still_rejects_free_text_and_bad_values(self):
        base = list(to_sanitized_monthly(make_stats(), SourceKind.REAL).fields)
        bad_fields = [("비고", "상담 내용 요약"), ("memo", "x"), ("건수_1999-01", 1), ("건수_2020-01", -1),
                      ("자료범위", "다른 자료"), ("원문", "x")]
        for extra in bad_fields:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                SanitizedInput(SourceKind.REAL, "receipt", "voc_monthly_v1", (*base, extra))

    def test_fingerprint_is_stable_and_changes_with_any_value(self):
        a = to_sanitized_monthly(make_stats(), SourceKind.REAL)
        b = to_sanitized_monthly(make_stats(), SourceKind.REAL)
        changed = to_sanitized_monthly(DatasetStats("receipt", "x", "u", 1000, 13, 3, "2020-01-01", "2020-03-31",
                                                    (("2020-01", 5), ("2020-03", 8)), 13, 0, ()), SourceKind.REAL)
        self.assertEqual(lr.payload_fingerprint(a), lr.payload_fingerprint(b))
        self.assertNotEqual(lr.payload_fingerprint(a), lr.payload_fingerprint(changed))
        self.assertEqual(lr.pii_findings(a), ())

    def test_aggregate_payload_does_not_trip_the_pii_scan_for_every_real_shape(self):
        for dataset in ("receipt", "online", "hazard"):
            stats = DatasetStats(dataset, "x", "u", 654321, 1234, 89, "2016-01-01", "2021-01-12",
                                 (("2016-01", 34), ("2020-03", 1200)), 1234, 0, NOTES_CAPPED)
            self.assertEqual(lr.pii_findings(to_sanitized_monthly(stats, SourceKind.REAL)), (), dataset)


@patch.object(socket, "socket", no_network)
class RealRunTest(unittest.TestCase):
    ARGS = ("--dataset", "receipt", "--approve-paid-calls", *PRICES, "--max-output-tokens", "2500",
            "--max-input-chars", "10000")

    def setUp(self):
        FakeSdk.instances, FakeSdk.fail_writer = [], False
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.state, self.report = self.folder / "real_state.json", self.folder / "real_report.md"
        self.synthetic_state = self.folder / "synthetic_state.json"
        self.summary = self.folder / "summary.json"
        save_summary([make_stats("receipt", notes=NOTES_CAPPED)], self.summary)
        self.code = lr.payload_fingerprint(to_sanitized_monthly(make_stats("receipt", notes=NOTES_CAPPED),
                                                                SourceKind.REAL))
        for patcher in (patch.object(lr, "SUMMARY_PATH", self.summary), patch.object(lr, "REAL_STATE_PATH", self.state),
                        patch.object(lr, "REAL_REPORT_PATH", self.report),
                        patch.object(ls, "STATE_PATH", self.synthetic_state), patch.object(lr, "SdkRunner", FakeSdk),
                        patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": "sk-test"})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            try:
                lr.main(list(argv))
            except SystemExit as error:
                return out.getvalue(), str(error)
        return out.getvalue(), None

    def start(self, code=None):
        return self.run_main("--phase", "start", *self.ARGS, "--approve-real-data-send", code or self.code)

    def calls(self):
        return [stage for sdk in FakeSdk.instances for stage in sdk.calls]

    def test_plan_shows_everything_that_will_be_sent_and_calls_nothing(self):
        out, message = self.run_main("--phase", "plan", "--dataset", "receipt", *PRICES)
        self.assertIsNone(message)
        for expected in ("전송할 자료", "건수 = 12", "월별 항공 건수(3개월)", self.code, "[확인 필요] 자료 이용 조건",
                         "[통과] 개인정보 패턴 검사", "상담 원문, 상담 제목, 개인정보"):
            self.assertIn(expected, out)
        self.assertFalse(self.state.exists())
        self.assertEqual(self.calls(), [])

    def test_start_is_refused_without_the_exact_approval_code_or_flags(self):
        for argv in (("--phase", "start", *self.ARGS),                                             # 승인 코드 없음
                     ("--phase", "start", *self.ARGS, "--approve-real-data-send", "0" * 16),        # 다른 내용
                     ("--phase", "start", "--dataset", "receipt", *PRICES,
                      "--approve-real-data-send", self.code),                                       # 유료 승인 없음
                     ("--phase", "start", *self.ARGS, "--approve-real-data-send", self.code, "--notes", "x")):
            with self.subTest(argv=argv):
                _, message = self.run_main(*argv)
                self.assertIsNotNone(message)
        self.assertEqual(FakeSdk.instances, [])
        self.assertFalse(self.state.exists())

    def test_start_runs_until_human_review_with_separate_files(self):
        out, message = self.start()
        self.assertIsNone(message)
        self.assertNotIn("[실패]", out)
        self.assertEqual(self.calls(), [StageName.COLLECTOR, StageName.ANALYST, StageName.WRITER_DRAFT,
                                        StageName.REVIEWER])
        self.assertIn("Human Review 결정이 필요합니다", out)
        saved = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(saved["dataset"], "receipt")
        self.assertEqual(saved["conditions"]["max_output_tokens"], 2500)
        self.assertFalse(self.synthetic_state.exists())          # 기존 합성 실행 상태는 건드리지 않는다
        self.assertFalse(self.report.exists())
        self.assertEqual(FakeSdk.instances[0].max_output_tokens, 2500)

    def test_second_start_is_refused_and_makes_no_calls(self):
        self.start()
        before = len(self.calls())
        _, message = self.start()
        self.assertIn("이미 실제 집계 실행 기록", message or "")
        self.assertEqual(len(self.calls()), before)

    def test_human_review_cannot_carry_free_text_and_needs_same_conditions(self):
        self.start()
        before = len(self.calls())
        base = ("--phase", "human", "--decision", "as_is", "--approve-paid-calls",
                "--approve-real-data-send", self.code, "--max-output-tokens", "2500", "--max-input-chars", "10000")
        _, notes_message = self.run_main(*base, *PRICES, "--notes", "근거를 보강해 주세요")
        self.assertIn("자유 메모", notes_message or "")
        _, other = self.run_main(*base, "--price-in", "0.10", "--price-out", "0.50", "--max-cost-usd", "0.04")
        self.assertIn("다른 비용·상한 조건", other or "")
        self.assertEqual(len(self.calls()), before)

    def test_full_flow_needs_the_users_decisions_and_saves_report_only_on_approval(self):
        self.start()
        out, message = self.run_main("--phase", "human", "--decision", "as_is", "--approve-paid-calls",
                                     "--approve-real-data-send", self.code, *PRICES,
                                     "--max-output-tokens", "2500", "--max-input-chars", "10000")
        self.assertIsNone(message)
        self.assertEqual(self.calls()[-1], StageName.MANAGER)
        self.assertIn("최종 승인/반려가 필요합니다", out)
        self.assertFalse(self.report.exists())                       # 승인 전에는 보고서를 저장하지 않는다
        _, missing = self.run_main("--phase", "final")
        self.assertIn("--decision", missing or "")
        self.assertFalse(self.report.exists())
        _, approved = self.run_main("--phase", "final", "--decision", "approve")
        self.assertIsNone(approved)
        text = self.report.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("<!-- LIVE 실제 집계 통계(receipt)"))
        self.assertIn("가설", text.splitlines()[0])
        self.assertEqual(self.calls().count(StageName.MANAGER), 1)
        saved = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(saved["dataset"], "receipt")                # 이후 저장에서도 자료·조건 유지
        self.assertIn("conditions", saved)

    def test_limits_cannot_be_widened_from_the_command_line(self):
        wide = (["--max-cost-usd", "0.50"], ["--max-input-chars", "20000"], ["--max-output-tokens", "5000"],
                ["--price-in", "0.01"], ["--price-out", "0.01"])
        base = ["--dataset", "receipt", "--approve-paid-calls", "--approve-real-data-send", self.code]
        for override in wide:
            for phase in ("plan", "start"):
                with self.subTest(override=override, phase=phase):
                    argv = ["--phase", phase, *base, *PRICES, *override]   # 뒤에 오는 값이 앞의 값을 덮어쓴다
                    _, message = self.run_main(*argv)
                    self.assertIn("승인 범위를 벗어난 조건", message or "")
        self.assertEqual(FakeSdk.instances, [])
        self.assertFalse(self.state.exists())

    def test_wrong_code_message_does_not_reveal_the_right_code(self):
        _, message = self.start("0" * 16)
        self.assertNotIn(self.code, message or "")
        self.assertIn("--phase plan", message or "")

    def test_lock_file_blocks_a_second_run_and_is_released_afterwards(self):
        lock = self.state.with_name(self.state.name + ".lock")
        lock.write_text("", encoding="utf-8")                           # 다른 실행이 진행 중이거나 비정상 종료
        _, message = self.start()
        self.assertIn("잠금 파일", message or "")
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.state.exists())
        lock.unlink()
        self.start()
        self.assertFalse(lock.exists())                                 # 정상 종료 후에는 잠금이 풀린다

    def test_crash_keeps_state_and_releases_the_lock_but_blocks_a_new_start(self):
        lock = self.state.with_name(self.state.name + ".lock")
        with patch.object(lr, "advance", side_effect=RuntimeError("crash")), self.assertRaises(RuntimeError):
            self.start()
        self.assertFalse(lock.exists())
        self.assertIn("stage_failed", self.state.read_text(encoding="utf-8"))
        _, message = self.start()
        self.assertIn("이미 실제 집계 실행 기록", message or "")

    def test_human_phase_refuses_when_the_aggregate_changed_after_approval(self):
        self.start()
        changed = DatasetStats("receipt", "시험 자료", "개별 상담 기록", 1000, 13, 3, "2020-01-01", "2020-03-31",
                               (("2020-01", 5), ("2020-03", 8)), 13, 0, NOTES_CAPPED)
        save_summary([changed], self.summary)
        new_code = lr.payload_fingerprint(to_sanitized_monthly(changed, SourceKind.REAL))
        before = len(self.calls())
        _, message = self.run_main("--phase", "human", "--decision", "as_is", "--approve-paid-calls",
                                   "--approve-real-data-send", new_code, *PRICES,
                                   "--max-output-tokens", "2500", "--max-input-chars", "10000")
        self.assertIn("승인받은 전송 내용과 현재 집계가 다릅니다", message or "")
        self.assertEqual(len(self.calls()), before)

    def test_serialized_input_limit_is_passed_to_the_runner(self):
        self.start()
        self.assertEqual(FakeSdk.instances[0].max_input_chars, 10_000)

    def test_state_restores_with_real_data_approval_only_for_real_runs(self):
        self.start()
        stats = make_stats("receipt", notes=NOTES_CAPPED)
        result, _, _, _ = ls.load_state(self.state, stats, lr.limits_for(self_args()), real_data=True)
        self.assertTrue(result.real_data_send_approved)
        self.assertIs(next_step(result).stage, StageName.HUMAN_REVIEW)
        plain, _, _, _ = ls.load_state(self.state, stats, lr.limits_for(self_args()))
        self.assertFalse(plain.real_data_send_approved)


def self_args():
    return lr.argparse.Namespace(max_input_chars=10_000, max_output_tokens=2500, max_cost_usd=0.05)


if __name__ == "__main__":
    unittest.main()
