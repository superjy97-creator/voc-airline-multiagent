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
PASSWORD = "correct-horse-battery"
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
        self.settings = svc.LiveSettings(KEY, PASSWORD)
        self.gate = svc.AdminGate(PASSWORD)
        self.token = self.gate.login(PASSWORD)
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

    # ---- 설정·관리자 인증 ----
    def test_settings_require_key_password_and_safe_prices(self):
        values = {"OPENAI_API_KEY": KEY, "ADMIN_PASSWORD": PASSWORD}
        self.assertIsNone(svc.load_settings({}.get))
        self.assertIsNone(svc.load_settings({"OPENAI_API_KEY": KEY}.get))
        self.assertIsNone(svc.load_settings({"ADMIN_PASSWORD": PASSWORD}.get))
        loaded = svc.load_settings(values.get)
        assert loaded is not None
        self.assertNotIn(KEY, repr(loaded))
        self.assertNotIn(PASSWORD, repr(loaded))
        with self.assertRaises(svc.LiveError):
            svc.load_settings({**values, "ADMIN_PASSWORD": "short"}.get)
        with self.assertRaises(svc.LiveError):
            svc.load_settings({**values, "PRICE_OUT_PER_M": "0.01"}.get)
        priced = svc.load_settings({**values, "PRICE_IN_PER_M": "0.2"}.get)
        assert priced is not None
        self.assertEqual(priced.price_in, 0.2)

    def test_gate_rejects_wrong_password_locks_out_and_expires(self):
        now = [0.0]
        gate = svc.AdminGate(PASSWORD, max_failures=3, lockout_seconds=60, session_seconds=100, clock=lambda: now[0])
        self.assertIsNone(gate.login("wrong"))
        self.assertFalse(gate.valid("forged-token"))
        self.assertFalse(gate.valid(None))
        for _ in range(2):
            gate.login("wrong")
        with self.assertRaises(svc.LiveError):
            gate.login(PASSWORD)            # 잠금 중에는 맞는 암호도 거부
        now[0] = 61
        token = gate.login(PASSWORD)
        self.assertTrue(gate.valid(token))
        now[0] = 200
        self.assertFalse(gate.valid(token))
        with self.assertRaises(svc.LiveError):
            gate.require(token)

    # ---- 인증 없이는 호출·기록 불가 ----
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
        self.settings = svc.LiveSettings(KEY, PASSWORD, 30.0, 60.0)
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
        self.assertNotIn(PASSWORD, text)

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


if __name__ == "__main__":
    unittest.main()
