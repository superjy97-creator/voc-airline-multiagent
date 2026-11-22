import socket
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voc import live_service as svc
from voc.contracts import ApprovalStatus, HumanDecision, StageName, next_step
from voc.pipeline import HumanInput
from voc.runners import FakeRunner
from voc.stats import synthetic_trend_stats

PERIOD = ("2022-01", "2023-12")
REQUEST = "월별 추세 분석"
KEY = "sk-test-NOT-A-REAL-KEY-0000"


def no_network(*args, **kwargs):
    raise AssertionError("테스트 중 네트워크 사용 금지")


class FakeSdk:
    """SdkRunner 대역: 네트워크 없이 호출 횟수·토큰을 기록한다."""
    instances: ClassVar[list] = []
    calls: ClassVar[list] = []

    def __init__(self, model, approved_live=False, *, max_output_tokens=None, client=None, max_input_chars=None):
        self.usage_log: list = []
        self.max_output_tokens, self.approved_live, self.model = max_output_tokens, approved_live, model
        FakeSdk.instances.append(self)

    def run_stage(self, step):
        FakeSdk.calls.append(step.stage)
        self.usage_log.append({"stage": step.stage.value, "requests": 1, "input_tokens": 1000, "output_tokens": 400})
        return FakeRunner().run_stage(step)


class LongOutputSdk(FakeSdk):
    """실제 실행과 비슷한 길이(수집 1천·분석 1.8천·작성 3천 자)의 출력을 돌려주는 대역."""
    sizes: ClassVar[dict] = {StageName.COLLECTOR: 1100, StageName.ANALYST: 1800, StageName.WRITER_DRAFT: 3100,
                             StageName.WRITER_REVISION: 3100, StageName.MANAGER: 1200}

    def run_stage(self, step):
        result = super().run_stage(step)
        size = self.sizes.get(step.stage)
        if size is None:
            return result
        text = (result.output_text or "") + " " + "집계 기반 서술입니다. " * (size // 12)
        return replace(result, output_text=text[:size])


def client_factory(api_key):
    assert api_key == KEY
    return Mock(max_retries=0)


@patch.object(socket, "socket", no_network)
class LiveServiceTest(unittest.TestCase):
    def setUp(self):
        FakeSdk.instances, FakeSdk.calls = [], []
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = patch.object(svc, "SdkRunner", FakeSdk)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = svc.LiveStore(Path(folder.name))
        self.settings = svc.LiveSettings(KEY)
        self.gate = svc.SessionGate()
        self.token = self.gate.issue()
        self.stats = synthetic_trend_stats()
        self.plan = svc.build_plan(self.stats, PERIOD, REQUEST, (0.10, 0.50))

    def start(self, **override):
        args = {"store": self.store, "gate": self.gate, "token": self.token, "settings": self.settings,
                "stats": self.stats, "period": PERIOD, "request": REQUEST, "typed_code": self.plan.code,
                "client_factory": client_factory}
        args.update(override)
        svc.start_run(**args)

    def view(self) -> svc.LiveView:
        view = svc.load_view(self.store, self.gate, self.token)
        assert view is not None
        return view

    def go(self, decision=HumanDecision.APPROVE_AS_IS, notes=None, count=None):
        svc.continue_run(self.store, self.gate, self.token, self.settings,
                         self.view().stage_count if count is None else count, HumanInput(decision, notes),
                         client_factory)

    # ---- 설정·세션 토큰 ----
    def test_settings_need_only_the_key_and_reject_cheap_prices(self):
        values = {"OPENAI_API_KEY": KEY}
        self.assertIsNone(svc.load_settings({}.get))
        self.assertIsNone(svc.load_settings({"OPENAI_API_KEY": "  "}.get))
        loaded = svc.load_settings(values.get)
        assert loaded is not None
        self.assertNotIn(KEY, repr(loaded))
        self.assertFalse(hasattr(loaded, "admin_password"))
        with self.assertRaises(svc.LiveError):
            svc.load_settings({**values, "PRICE_OUT_PER_M": "0.01"}.get)
        priced = svc.load_settings({**values, "PRICE_IN_PER_M": "0.2"}.get)
        assert priced is not None
        self.assertEqual(priced.price_in, 0.2)

    def test_gate_tokens_are_server_issued_and_expire(self):
        now = [0.0]
        gate = svc.SessionGate(session_seconds=100, clock=lambda: now[0])
        token = gate.issue()
        self.assertTrue(gate.valid(token))
        self.assertFalse(gate.valid("forged-token"))
        self.assertFalse(gate.valid(None))
        now[0] = 200
        self.assertFalse(gate.valid(token))
        with self.assertRaises(svc.LiveError):
            gate.require(token)

    # ---- 접근 토큰 없이는 호출·기록 불가 ----
    def test_no_valid_token_means_no_client_no_call_no_state(self):
        for bad in (None, "", "forged", "x" * 43):
            with self.subTest(token=bad), self.assertRaises(svc.LiveError):
                self.start(token=bad)
        self.assertEqual((FakeSdk.instances, FakeSdk.calls, self.store.exists()), ([], [], False))
        self.gate.logout(self.token)
        with self.assertRaises(svc.LiveError):
            self.start()
        with self.assertRaises(svc.LiveError):
            svc.load_view(self.store, self.gate, self.token)
        with self.assertRaises(svc.LiveError):
            svc.finalize(self.store, self.gate, self.token, True)
        self.assertEqual(FakeSdk.instances, [])

    def test_wrong_approval_code_and_real_dataset_are_blocked(self):
        with self.assertRaises(svc.LiveError):
            self.start(typed_code="WRONG000")
        with self.assertRaises(svc.LiveError):
            self.start(typed_code="")
        real_like = replace(self.stats, dataset_id="online")
        with self.assertRaises(svc.LiveError):
            svc.build_plan(real_like, PERIOD, REQUEST, (0.10, 0.50))
        self.assertEqual((FakeSdk.instances, self.store.exists()), ([], False))

    def test_cost_over_cap_blocks_before_any_call(self):
        expensive = svc.build_plan(self.stats, PERIOD, REQUEST, (30.0, 60.0))
        self.assertFalse(expensive.ok)
        self.settings = svc.LiveSettings(KEY, 30.0, 60.0)
        with self.assertRaises(svc.LiveError):
            self.start(typed_code=expensive.code)
        self.assertEqual((FakeSdk.instances, self.store.exists()), ([], False))

    def test_plan_is_within_budget_and_states_scope(self):
        self.assertTrue(self.plan.ok)
        self.assertLessEqual(self.plan.worst, 0.05)
        text = self.plan.text()
        for fragment in ("gpt-6-luna", "9회", "상담 원문", "추정치"):
            self.assertIn(fragment, text)

    # ---- 정상 경로 ----
    def test_full_flow_five_agents_human_review_manager_final_approval(self):
        self.start()
        first = self.view()
        self.assertEqual(FakeSdk.calls, [StageName.COLLECTOR, StageName.ANALYST, StageName.WRITER_DRAFT,
                                         StageName.REVIEWER])
        self.assertIs(next_step(first.result).stage, StageName.HUMAN_REVIEW)
        self.assertAlmostEqual(first.spent, (4 * 1000 * 0.10 + 4 * 400 * 0.50) / 1_000_000, places=9)
        self.go()
        second = self.view()
        self.assertEqual(FakeSdk.calls[-1], StageName.MANAGER)
        self.assertIs(next_step(second.result).stage, StageName.FINAL_APPROVAL)
        report = svc.latest_text(second.result, StageName.WRITER_REVISION, StageName.WRITER_DRAFT)
        self.assertTrue(report)
        self.assertEqual(len(FakeSdk.calls), 5)
        self.assertFalse(svc.is_approved(second.result))
        svc.finalize(self.store, self.gate, self.token, True)
        self.assertIs(self.view().result.approval, ApprovalStatus.APPROVED)
        self.assertEqual(len(FakeSdk.calls), 5)       # 최종 승인은 호출이 없다

    def test_realistic_output_lengths_are_not_blocked_by_input_cap(self):
        """첫 실제 실행의 회귀: 이전 단계 출력이 길어져도 Reviewer·Manager 입력이 상한에 막히지 않아야 한다."""
        with patch.object(svc, "SdkRunner", LongOutputSdk):
            self.start()
            view = self.view()
            self.assertEqual(len(FakeSdk.calls), 4)
            self.assertIs(next_step(view.result).stage, StageName.HUMAN_REVIEW)
            self.assertIsNone(next_step(view.result).stop_reason)
            self.go()
        self.assertIs(next_step(self.view().result).stage, StageName.FINAL_APPROVAL)
        self.assertEqual(svc.INPUT_CAP_CHARS, 10_000)
        self.assertLessEqual(self.plan.worst, svc.DEFAULT_COST_CAP)

    def test_long_revision_path_reaches_manager_within_input_cap(self):
        """두 번째 실제 실행의 회귀: 수정 요청 뒤 재검수·Manager 입력도 상한 안에 들어야 한다(최신 산출물만 전송)."""
        with patch.object(svc, "SdkRunner", LongOutputSdk):
            self.start()
            self.go(HumanDecision.REQUEST_REVISION)
            view = self.view()
            self.assertIsNone(next_step(view.result).stop_reason)
            self.go(HumanDecision.APPROVE_AS_IS)
        final = self.view()
        self.assertIs(next_step(final.result).stage, StageName.FINAL_APPROVAL)
        self.assertIn(StageName.REVIEWER_RECHECK, FakeSdk.calls)
        self.assertEqual(FakeSdk.calls[-1], StageName.MANAGER)
        self.assertLessEqual(len(FakeSdk.calls), 9)

    def test_revision_path_stays_within_nine_calls(self):
        self.start()
        self.go(HumanDecision.REQUEST_REVISION, "근거를 보강해 주세요.")
        calls = len(FakeSdk.calls)
        self.assertLessEqual(calls, 9)
        self.assertIn(StageName.WRITER_REVISION, FakeSdk.calls)

    def test_secrets_never_written_to_state_file(self):
        self.start()
        text = self.store.state_path.read_text(encoding="utf-8")
        self.assertNotIn(KEY, text)

    # ---- 중복·새로고침·동시 실행 ----
    def test_second_start_is_refused_and_existing_record_preserved(self):
        self.start()
        before = self.store.state_path.read_text(encoding="utf-8")
        calls = len(FakeSdk.calls)
        with self.assertRaises(svc.LiveError):
            self.start()
        self.assertEqual(len(FakeSdk.calls), calls)
        self.assertEqual(self.store.state_path.read_text(encoding="utf-8"), before)

    def test_stale_or_repeated_continue_makes_no_call(self):
        self.start()
        stale = self.view().stage_count
        self.go(count=stale)
        calls = len(FakeSdk.calls)
        with self.assertRaises(svc.LiveError):       # 같은 화면에서 다시 제출(새로고침·더블클릭)
            self.go(count=stale)
        with self.assertRaises(svc.LiveError):       # 오래된 단계 수
            self.go(count=1)
        self.assertEqual(len(FakeSdk.calls), calls)

    def test_continue_after_final_decision_is_refused(self):
        self.start()
        self.go()
        svc.finalize(self.store, self.gate, self.token, False)
        calls = len(FakeSdk.calls)
        with self.assertRaises(svc.LiveError):
            self.go()
        with self.assertRaises(svc.LiveError):
            svc.finalize(self.store, self.gate, self.token, True)
        self.assertEqual(len(FakeSdk.calls), calls)

    def test_concurrent_lock_blocks_every_call(self):
        self.store.directory.mkdir(parents=True, exist_ok=True)
        self.store.lock_path.write_text("held", encoding="utf-8")
        with self.assertRaises(svc.LiveError):
            self.start()
        self.assertEqual((FakeSdk.instances, self.store.exists()), ([], False))

    def test_interrupted_call_marker_blocks_further_calls(self):
        self.start()
        data = self.store.read()
        assert data is not None
        data["inflight"] = "writer_draft"
        self.store.state_path.write_text(__import__("json").dumps(data, ensure_ascii=False), encoding="utf-8")
        self.assertTrue(self.view().interrupted)
        calls = len(FakeSdk.calls)
        with self.assertRaises(svc.LiveError):
            self.go()
        self.assertEqual(len(FakeSdk.calls), calls)

    def test_pii_note_is_rejected_without_calls(self):
        self.start()
        calls = len(FakeSdk.calls)
        with self.assertRaises(ValueError):
            self.go(HumanDecision.REQUEST_REVISION, "전화 010-1234-5678")
        self.assertEqual(len(FakeSdk.calls), calls)
        self.assertFalse(self.store.lock_path.exists())       # 잠금도 풀린다


class LimiterTest(unittest.TestCase):
    """공개 이용 한도: 세션·사용자를 바꿔도 전체 한도를 넘을 수 없고, 남용 시 자동 차단된다."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dir = Path(folder.name)
        self.now = [1000.0]

    def make(self, **limits):
        return svc.PublicLimiter(self.dir, svc.PublicLimits(**limits), clock=lambda: self.now[0])

    def test_one_active_run_at_a_time_then_free_after_finish(self):
        limiter = self.make()
        first = limiter.admit("a", 0.05)
        with self.assertRaises(svc.LiveError) as raised:
            limiter.admit("b", 0.05)
        self.assertIn("진행 중", str(raised.exception))
        limiter.finish(first, 0.004)
        self.assertTrue(limiter.admit("b", 0.05))

    def test_finish_settles_actual_cost_once_and_refunds_the_reservation(self):
        limiter = self.make(max_total_cost_usd=0.10)
        run = limiter.admit("a", 0.05)
        self.assertAlmostEqual(limiter.status()["budget_left"], 0.05)
        limiter.finish(run, 0.004)
        limiter.finish(run, 0.05)                       # 같은 실행의 두 번째 정산은 무시된다
        self.assertAlmostEqual(limiter.status()["budget_left"], 0.096)

    def test_expired_run_frees_the_slot_but_keeps_the_reserved_cost(self):
        limiter = self.make(active_ttl_s=60, max_total_cost_usd=0.10)
        limiter.admit("a", 0.05)
        self.now[0] += 61
        self.assertTrue(limiter.admit("b", 0.04))
        with self.assertRaises(svc.LiveError):
            self.now[0] += 61
            limiter.admit("c", 0.05)                    # 예약 합계 0.09 + 0.05 > 0.10

    def test_total_run_and_cost_caps_apply_across_clients(self):
        limiter = self.make(max_runs_total=2, max_active_runs=5)
        for client in ("a", "b"):
            limiter.admit(client, 0.01)
        with self.assertRaises(svc.LiveError) as raised:
            limiter.admit("c", 0.01)
        self.assertIn("전체 실행 한도", str(raised.exception))
        cost_dir = self.dir / "cost"
        cost = svc.PublicLimiter(cost_dir, svc.PublicLimits(max_runs_total=99, max_active_runs=99,
                                                            max_total_cost_usd=0.03), clock=lambda: self.now[0])
        cost.admit("a", 0.02)
        with self.assertRaises(svc.LiveError) as raised:
            cost.admit("b", 0.02)
        self.assertIn("전체 비용 한도", str(raised.exception))

    def test_per_client_cap_within_window(self):
        limiter = self.make(max_active_runs=9, per_client_runs=2, per_client_window_s=100)
        for _ in range(2):
            limiter.finish(limiter.admit("same", 0.01), 0.0)
        with self.assertRaises(svc.LiveError):
            limiter.admit("same", 0.01)
        self.assertTrue(limiter.admit("other", 0.01))
        self.now[0] += 101
        self.assertTrue(limiter.admit("same", 0.01))

    def test_repeated_failures_trip_an_automatic_block_that_expires(self):
        limiter = self.make(failure_trip=2, block_seconds=100, max_active_runs=9, per_client_runs=99)
        for _ in range(2):
            limiter.fail(limiter.admit("a", 0.01), None)
        self.assertTrue(limiter.status()["blocked"])
        with self.assertRaises(svc.LiveError) as raised:
            limiter.admit("b", 0.01)
        self.assertIn("일시 차단", str(raised.exception))
        self.now[0] += 101
        self.assertFalse(limiter.status()["blocked"])
        self.assertTrue(limiter.admit("b", 0.01))

    def test_many_rejections_trip_an_automatic_block(self):
        limiter = self.make(reject_trip=5, block_seconds=100)
        for _ in range(5):
            limiter.reject()
        self.assertTrue(limiter.status()["blocked"])
        with self.assertRaises(svc.LiveError):
            limiter.admit("a", 0.01)

    def test_concurrent_sessions_cannot_exceed_the_global_limits(self):
        import threading
        limiter = self.make(max_active_runs=1)
        wins, errors = [], []

        def attempt(i):
            try:
                wins.append(limiter.admit(f"c{i}", 0.01))
            except svc.LiveError as error:
                errors.append(error)

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(12)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual((len(wins), len(errors)), (1, 11))
        wide = svc.PublicLimiter(self.dir / "wide", svc.PublicLimits(max_active_runs=99, per_client_runs=99,
                                                                    max_runs_total=3, max_total_cost_usd=0.5),
                                 clock=lambda: self.now[0])
        results = []

        def safe(i):
            try:
                results.append(wide.admit(f"x{i}", 0.01))
            except svc.LiveError:
                pass

        threads = [threading.Thread(target=safe, args=(i,)) for i in range(12)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(results), 3)               # 전체 실행 한도 3을 넘지 못한다

    def test_run_ids_are_validated_and_must_be_known(self):
        limiter = self.make()
        run = limiter.admit("a", 0.01)
        self.assertTrue(limiter.known(run))
        for bad in (None, "", "../etc/passwd", "short", "x" * 40, 123):
            with self.subTest(run_id=bad):
                self.assertFalse(limiter.known(bad))
        self.assertTrue(svc.run_dir(run, self.dir).is_relative_to(self.dir))
        with self.assertRaises(svc.LiveError):
            svc.run_dir("../escape", self.dir)


if __name__ == "__main__":
    unittest.main()
