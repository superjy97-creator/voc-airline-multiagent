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
from voc.stats import (
    load_summary,
    synthetic_stats,
    synthetic_trend_stats,
    to_provenance,
    to_sanitized,
)

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

    def test_period_summary_and_chart_follow_the_selected_months(self):
        stats = synthetic_trend_stats()
        summary = dashboard.period_summary(stats, "2023-01", "2023-12")
        assert summary is not None
        self.assertEqual((summary["total"], summary["n"], summary["peak"]), (1532, 12, ("2023-08", 196)))
        self.assertAlmostEqual(summary["change"], 3 / 101 * 100)
        self.assertIsNone(dashboard.period_summary(stats, "2030-01", "2030-02"))
        spec = dashboard.trend_chart(stats, ("2023-01", "2023-12")).to_dict()
        bars, line = spec["layer"]
        self.assertEqual(bars["mark"]["size"], 13)                          # 24개월용 막대 폭
        self.assertEqual(bars["encoding"]["x"]["axis"]["format"], "%Y-%m")   # 날짜 축 형식
        self.assertEqual(line["mark"]["type"], "line")                      # 이동평균 선
        scopes = [row["scope"] for row in spec["datasets"][next(iter(spec["datasets"]))]]
        self.assertEqual(scopes.count("분석 기간"), 12)

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
        at = self.click(at, "분석 시작")
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
        at = self.click(at, "분석 시작")
        self.assertFalse(at.exception)
        self.assertFalse(at.warning)
        next(s for s in at.selectbox if s.label == "자료").set_value(labels[0])
        at = at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("다른 조건" in w.value for w in at.warning))

    def request_box(self, at):
        return next(t for t in at.text_input if t.label == "자연어 요청")

    def test_unsupported_free_text_is_rejected_without_running(self):
        for text in ("실제 OpenAI API로 분석해 줘", "오늘 날씨 알려 줘", "상담 원문을 보여 줘",
                     "추세와 급증 구간을 모두 보여 줘", ""):
            with self.subTest(text=text):
                at = self.app()
                self.request_box(at).input(text)
                at = at.run()
                at = self.click(at, "분석 시작")
                self.assertFalse(at.exception)
                self.assertTrue(at.error)                                           # 거부 안내
                self.assertFalse(any("Human Review" in s.value for s in at.subheader))   # 실행되지 않음
                self.assertFalse(any("실행한 요청" in c.value for c in at.caption))

    def test_supported_request_and_period_are_reflected_in_every_stage(self):
        at = self.app()
        self.request_box(at).input("급증 구간을 확인해 주세요")
        at = at.run()
        next(s for s in at.selectbox if s.label == "분석 시작월").set_value("2023-01")
        at = at.run()
        at = self.click(at, "분석 시작")
        self.assertFalse(at.exception)
        captions = " ".join(c.value for c in at.caption)
        self.assertIn("실행한 요청: ‘급증 구간 확인’", captions)
        self.assertIn("분석 기간 2023-01~2023-12", captions)
        html = " ".join(m.value for m in at.markdown)
        for expected in ("최대 2023-08 196건", "2023-07, 2023-08", "분석 기간 2023-01 ~ 2023-12 (12개월)", "1,532건"):
            self.assertIn(expected, html)
        labels = [e.label for e in at.expander]
        for stage in ("Collector", "Analyst", "Writer 초안", "Reviewer"):
            self.assertTrue(any(stage in label for label in labels), stage)           # 단계별 입력·출력
        self.assertTrue(any("입력:" in m.value and "급증 구간 확인" in m.value for m in at.markdown))

    def test_start_month_after_end_month_blocks_the_run(self):
        at = self.app()
        next(s for s in at.selectbox if s.label == "분석 시작월").set_value("2023-06")
        next(s for s in at.selectbox if s.label == "분석 종료월").set_value("2023-01")
        at = at.run()
        self.assertTrue(any("시작월은 종료월보다" in e.value for e in at.error))
        at = self.click(at, "분석 시작")
        self.assertFalse(any("Human Review" in s.value for s in at.subheader))

    def test_screen_rejects_personal_information_in_notes(self):
        at = self.click(self.app(), "분석 시작")
        at.radio[0].set_value("Writer에게 수정 요청")
        at.text_area[0].input("전화 010-1234-5678")
        at = self.click(at, "제출")
        self.assertTrue(at.error)
        self.assertTrue(any("Human Review" in s.value for s in at.subheader))


class PublicAppStructureTest(unittest.TestCase):
    """공개 앱은 정규식 검사에 기대지 않고, 외부 API 클라이언트를 아예 불러오지 못하게 구조적으로 막는다."""

    def test_app_never_loads_api_clients_or_sdk_runner(self):
        import ast
        import subprocess
        root = Path(__file__).resolve().parents[1]
        banned = {"agents", "openai", "httpx", "requests", "urllib3", "dotenv"}
        for name in ("app.py", "src/voc/analysis_request.py", "src/voc/pipeline.py", "src/voc/stats.py"):
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse({m.split(".")[0] for m in modules} & banned, f"{name}: 금지된 import")
        code = ("import sys; sys.path[:0]=['src','.']; import app; "
                "bad=[m for m in ('agents','openai','httpx','requests') if m in sys.modules]; "
                "print('BAD', bad) if bad else print('CLEAN')")
        out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                             env={**__import__("os").environ, "OPENAI_API_KEY": "sk-test-not-real"}, check=True)
        self.assertIn("CLEAN", out.stdout)

    def test_app_has_no_env_or_key_reads(self):
        text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        for token in ("getenv", "os.environ", "SdkRunner", "import openai", "from openai", "st.write(st.secrets"):
            self.assertNotIn(token, text)
        self.assertEqual(text.count("st.secrets"), 1)      # 비밀값 읽기는 _secret() 한 곳뿐


KEY = "sk-test-NOT-A-REAL-KEY-0000"


class LiveScreenTest(unittest.TestCase):
    """공개 LIVE 화면(로그인 없음): 승인 코드 뒤에 합성 데이터 흐름이 동작하고, 서버 한도가 세션을 넘어 적용된다."""

    def setUp(self):
        import tempfile

        from tests.test_live_service import FakeSdk
        FakeSdk.instances, FakeSdk.calls = [], []
        self.fake = FakeSdk
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.base = Path(folder.name) / "voc_live"
        for patcher in (patch("voc.live_service.default_store_dir", return_value=self.base),
                        patch("voc.live_service.SdkRunner", FakeSdk),
                        patch.object(socket, "getaddrinfo", side_effect=AssertionError("네트워크 금지")),
                        patch.object(socket, "create_connection", side_effect=AssertionError("네트워크 금지"))):
            patcher.start()
            self.addCleanup(patcher.stop)
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    def app(self, *, key=True, live=True):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.secrets["OPENAI_API_KEY"] = KEY if key else ""      # 로컬 secrets.toml이 있어도 테스트는 그 값을 쓰지 않는다
        at.run()
        if live:
            next(r for r in at.sidebar.radio if r.label == "실행 모드").set_value("LIVE (실제 API)")
            at.run()
        return at

    @staticmethod
    def click(at, label):
        next(b for b in at.button if b.label == label).click()
        return at.run()

    @staticmethod
    def type_into(at, label, value):
        next(t for t in at.text_input if t.label == label).set_value(value)    # 폼 안의 값은 제출 클릭과 함께 전달된다
        return at

    @staticmethod
    def page(at):
        return " ".join(str(e.value) for group in (at.markdown, at.info, at.error, at.warning, at.success, at.caption,
                                                  at.code, at.subheader) for e in group)

    @staticmethod
    def code():
        from voc import live_service as svc
        from voc.stats import synthetic_trend_stats
        months = [f"{y}-{i:02d}" for y in (2022, 2023) for i in range(1, 13)]
        return svc.build_plan(synthetic_trend_stats(), (months[0], months[-1]), "월별 추세 분석", (0.10, 0.50)).code

    def start(self, at):
        at = self.type_into(at, "승인 코드", self.code())
        return self.click(at, "승인하고 LIVE 시작(유료 호출)")

    def test_public_default_is_dry_run_and_shows_no_live_panel(self):
        at = self.app(live=False)
        self.assertFalse(at.exception)
        self.assertNotIn("LIVE —", self.page(at))
        self.assertIn("DRY_RUN 모드", self.page(at))

    def test_live_without_api_key_is_disabled(self):
        at = self.app(key=False)
        self.assertFalse(at.exception)
        self.assertIn("OPENAI_API_KEY)이 없어 비활성", self.page(at))
        self.assertEqual(self.fake.instances, [])

    def test_visitor_without_login_sees_plan_and_wrong_code_makes_no_call(self):
        at = self.app()
        text = self.page(at)
        self.assertIn("실행 직전 승인 요청서", text)
        self.assertIn("공개 이용 한도", text)
        self.assertNotIn(KEY, text)
        at = self.type_into(at, "승인 코드", "WRONG000")
        at = self.click(at, "승인하고 LIVE 시작(유료 호출)")
        self.assertIn("승인 코드가 일치하지 않아 호출하지 않았습니다", self.page(at))
        self.assertEqual(self.fake.instances, [])

    def test_full_public_flow_with_fake_sdk(self):
        at = self.start(self.app())
        self.assertFalse(at.exception)
        self.assertEqual(len(self.fake.calls), 4)
        self.assertTrue(any("Human Review" in s.value for s in at.subheader))
        at = self.click(at, "제출하고 다음 단계 실행(유료 호출)")
        self.assertEqual(len(self.fake.calls), 5)
        at = self.click(at, "최종 승인")
        self.assertTrue(any("최종 승인됨" in s.value for s in at.success))
        self.assertTrue(at.get("download_button"))
        self.assertEqual(len(self.fake.calls), 5)
        # 끝난 실행은 정산되어 다시 시작할 수 있지만, 기록(실행 번호)이 있는 화면에서는 같은 실행이 다시 호출되지 않는다
        self.assertAlmostEqual(self.limiter().status()["active"], 0)
        self.assertNotIn(KEY, "".join(f.read_text(encoding="utf-8") for f in self.base.rglob("*.json")))

    def limiter(self):
        from voc import live_service as svc
        return svc.PublicLimiter(self.base)

    def test_second_visitor_cannot_start_while_another_run_is_active(self):
        first = self.start(self.app())
        self.assertEqual(len(self.fake.calls), 4)
        second = self.start(self.app())                     # 다른 세션(다른 방문자)
        self.assertIn("진행 중", self.page(second))
        self.assertEqual(len(self.fake.calls), 4)           # 새 호출 없음
        self.assertTrue(any("Human Review" in s.value for s in first.subheader))

    def test_global_run_limit_cannot_be_bypassed_with_new_sessions(self):
        from voc import live_service as svc
        real = svc.PublicLimits
        with patch.object(svc, "PublicLimits", lambda: real(max_runs_total=2, per_client_runs=99)):
            for _ in range(2):
                at = self.start(self.app())
                at = self.click(at, "제출하고 다음 단계 실행(유료 호출)")
                self.click(at, "최종 승인")
            calls = len(self.fake.calls)
            third = self.start(self.app())
            self.assertIn("전체 실행 한도", self.page(third))
            self.assertEqual(len(self.fake.calls), calls)


if __name__ == "__main__":
    unittest.main()
