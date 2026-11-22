"""6단계 통합 테스트: 통계 → 비식별 → DRY_RUN 5개 에이전트 → Human Review → 최종 승인, 화면 시연까지.

모든 테스트는 가짜 모델만 쓰며 소켓을 막아 외부 호출 0회를 확인한다.
"""

import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

import app as dashboard
from voc.contracts import (
    ApprovalStatus,
    CallBudget,
    HumanDecision,
    ReviewVerdict,
    SourceKind,
    StageName,
    StageStatus,
    next_step,
)
from voc.pipeline import DRY_RUN_LIMITS, HumanInput, RunContext, advance, new_run
from voc.runners import FakeRunner
from voc.sanitize import make_checked_text
from voc.stats import load_summary, synthetic_stats, to_provenance, to_sanitized

SUMMARY = ROOT / "reports" / "stats_summary.json"
V, D = ReviewVerdict, HumanDecision


def no_network(*_args, **_kwargs):
    raise AssertionError("네트워크 호출이 시도되었습니다.")


_real_connect = socket.socket.connect


def loopback_only_connect(sock, address):
    """화면 테스트용: Streamlit 내부의 로컬(루프백) 연결만 허용하고 외부 주소는 차단한다."""
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise AssertionError("외부 네트워크 연결이 시도되었습니다.")
    return _real_connect(sock, address)


def start(stats, verdicts=(), kind=SourceKind.SYNTHETIC):
    result = new_run("it", (to_provenance(stats),))
    ctx = RunContext(to_sanitized(stats, kind), FakeRunner(verdicts), CallBudget(DRY_RUN_LIMITS),
                     make_checked_text)
    return advance(result, ctx), ctx


@patch.object(socket, "socket", no_network)
class PipelineIntegrationTest(unittest.TestCase):
    def test_approved_review_still_waits_for_human_then_final_approval(self):
        result, ctx = start(synthetic_stats()[0])
        self.assertIs(next_step(result).stage, StageName.HUMAN_REVIEW)
        result = advance(result, ctx, HumanInput(D.APPROVE_AS_IS))
        self.assertIs(next_step(result).stage, StageName.FINAL_APPROVAL)
        self.assertIs(result.approval, ApprovalStatus.DRAFT)
        result = result.approve_final()
        self.assertIs(result.approval, ApprovalStatus.APPROVED)
        self.assertEqual(result.calls_used, 5)         # Collector, Analyst, Writer, Reviewer, Manager
        self.assertEqual(ctx.budget.total, 5)

    def test_revision_path_uses_all_five_agents_and_nine_calls_at_most(self):
        result, ctx = start(synthetic_stats()[1], (V.NEEDS_REVISION, V.NEEDS_REVISION, V.APPROVABLE))
        for _ in range(2):
            result = advance(result, ctx, HumanInput(D.REQUEST_REVISION, "근거를 보강해 주세요."))
        self.assertIs(next_step(result).stage, StageName.FINAL_APPROVAL)
        self.assertEqual(result.calls_used, 9)
        used = {s.stage for s in result.stages if s.status is StageStatus.SUCCEEDED}
        self.assertTrue({StageName.COLLECTOR, StageName.ANALYST, StageName.WRITER_DRAFT,
                         StageName.REVIEWER, StageName.MANAGER} <= used)
        self.assertIs(result.approve_final().approval, ApprovalStatus.APPROVED)

    def test_pii_in_human_notes_is_rejected_without_recording(self):
        result, ctx = start(synthetic_stats()[0])
        before = result.stages
        with self.assertRaises(ValueError):
            advance(result, ctx, HumanInput(D.REQUEST_REVISION, "연락처 010-1234-5678"))
        self.assertEqual(result.stages, before)

    @unittest.skipUnless(SUMMARY.exists(), "로컬 집계 파일이 없는 환경(저장소·배포)에서는 건너뜀")
    def test_local_summary_converts_to_valid_dry_run_input(self):
        for stats in load_summary(SUMMARY):
            result, _ = start(stats, kind=SourceKind.REAL)       # DRY_RUN은 실제 집계값도 로컬에서만 처리
            self.assertIs(next_step(result).stage, StageName.HUMAN_REVIEW)
            self.assertEqual(result.calls_used, 4)

    def test_report_uses_only_given_numbers(self):
        stats = synthetic_stats()[0]
        result, _ = start(stats)
        report = next(s.output_text for s in result.stages if s.stage is StageName.WRITER_DRAFT)
        self.assertIn("DRY_RUN", report)
        self.assertIn(str(stats.air_rows), report.replace(",", ""))


class DashboardPartsTest(unittest.TestCase):
    """화면의 순수 부분(에이전트 흐름 상태·KPI·출처 목록)만 검증한다."""

    @staticmethod
    def states(result):
        return {name: (state, text) for name, _, state, text in dashboard.node_states(result)}

    def test_before_start_every_node_waits(self):
        states = self.states(None)
        self.assertEqual(len(states), 8)
        self.assertEqual({state for state, _ in states.values()}, {"pending"})
        self.assertIn("최종 승인", states)

    def test_flow_follows_the_pipeline_until_final_approval(self):
        result, ctx = start(synthetic_stats()[0])
        states = self.states(result)
        for name in ("Collector", "Analyst", "Writer", "Reviewer"):
            self.assertEqual(states[name], ("done", "완료"))
        self.assertEqual(states["Human Review"], ("active", "사람 검토 필요"))
        self.assertEqual(states["Manager"][0], "pending")
        result = advance(result, ctx, HumanInput(D.APPROVE_AS_IS))
        states = self.states(result)
        self.assertEqual(states["Human Review"], ("done", "완료"))
        self.assertEqual(states["수정·재검수"], ("skip", "생략"))
        self.assertEqual(states["Manager"], ("done", "완료"))
        self.assertEqual(states["최종 승인"], ("active", "승인 대기"))
        self.assertEqual(self.states(result.approve_final())["최종 승인"], ("done", "승인됨"))
        self.assertEqual(self.states(result.reject())["최종 승인"], ("fail", "반려"))

    def test_revision_path_marks_revision_node_done(self):
        result, ctx = start(synthetic_stats()[0], (V.NEEDS_REVISION, V.APPROVABLE))
        result = advance(result, ctx, HumanInput(D.REQUEST_REVISION, "근거를 보강해 주세요."))
        self.assertEqual(self.states(result)["수정·재검수"], ("done", "완료"))
        self.assertEqual(self.states(result)["Human Review"][0], "active")

    def test_html_helpers_escape_values(self):
        self.assertNotIn("<script>", dashboard.kpi_html("<script>x</script>", "<b>1</b>", "blue"))
        self.assertIn("&lt;script&gt;", dashboard.kpi_html("<script>x</script>", "1", "blue"))
        self.assertIn("class=\"node pending\"", dashboard.flow_html(None))

    def test_local_source_is_offered_only_when_the_summary_file_exists(self):
        self.assertEqual(dashboard.available_sources(ROOT / "nope" / "stats.json"),
                         [dashboard.SYNTHETIC_SOURCE])
        self.assertEqual(dashboard.available_sources(Path(__file__)),
                         [dashboard.SYNTHETIC_SOURCE, dashboard.LOCAL_SOURCE])


@patch.object(socket.socket, "connect", loopback_only_connect)
class ScreenTest(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

    @staticmethod
    def click(at, label):
        next(b for b in at.button if b.label == label).click()
        return at.run()

    def test_full_flow_to_final_approval(self):
        at = self.app()
        self.assertFalse(at.exception)
        at = self.click(at, "DRY_RUN 시작/다시 시작")
        self.assertFalse(at.exception)
        self.assertTrue(any("Human Review" in s.value for s in at.subheader))
        self.assertTrue(any("사람 검토 필요" in m.value for m in at.markdown))      # 에이전트 흐름 표시
        at = self.click(at, "제출")
        self.assertTrue(any("최종 승인" in s.value for s in at.subheader))
        self.assertTrue(any("승인 대기" in m.value for m in at.markdown))
        self.assertTrue(any(b.label == "최종 승인" for b in at.button))
        at = self.click(at, "최종 승인")
        self.assertTrue(any("최종 승인됨" in s.value for s in at.success))
        self.assertTrue(any("승인됨" in m.value and "node done" in m.value for m in at.markdown))
        self.assertTrue(at.get("download_button"))                                  # 보고서 다운로드 버튼

    def test_header_shows_dry_run_mode_and_data_kind(self):
        at = self.app()
        html = " ".join(m.value for m in at.markdown)
        self.assertIn("DRY_RUN 모드", html)
        self.assertNotIn("DRY_RUN 승인", html)
        self.assertIn("합성 데이터(실제 수치 아님)", html)
        self.assertIn('class="t synthetic"', html)                    # KPI 카드의 데이터 구분 표시
        self.assertNotIn("로컬 실제 집계", html)

    def test_selecting_another_dataset_has_no_duplicate_widgets_and_flags_stale_results(self):
        labels = [s.label for s in synthetic_stats()]
        at = self.app()
        next(s for s in at.selectbox if s.label == "자료").set_value(labels[1])
        at = at.run()
        self.assertFalse(at.exception)                                 # StreamlitDuplicateElementId 회귀
        at = self.click(at, "DRY_RUN 시작/다시 시작")
        self.assertFalse(at.exception)
        self.assertFalse(at.warning)
        next(s for s in at.selectbox if s.label == "자료").set_value(labels[0])
        at = at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("다른 자료" in w.value for w in at.warning))

    def test_screen_rejects_personal_information_in_notes(self):
        at = self.click(self.app(), "DRY_RUN 시작/다시 시작")
        at.radio[0].set_value("Writer에게 수정 요청")
        at.text_area[0].input("전화 010-1234-5678")
        at = self.click(at, "제출")
        self.assertTrue(at.error)
        self.assertTrue(any("Human Review" in s.value for s in at.subheader))


if __name__ == "__main__":
    unittest.main()
