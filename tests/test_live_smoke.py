import io
import json
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import live_smoke as ls
from voc.contracts import CallBudget, RunLimits, StageName, StageStatus, next_step
from voc.pipeline import HumanInput, RunContext, advance, new_run
from voc.runners import FakeRunner
from voc.sanitize import make_checked_text

FULL = ["--approve-paid-calls", "--price-in", "0.10", "--price-out", "0.50", "--max-cost-usd", "0.05"]


def no_network(*_args, **_kwargs):
    raise AssertionError("네트워크 호출이 시도되었습니다.")


def waiting_for_human():
    """가짜 모델로 Human Review 직전까지 진행한 실행 상태(네트워크 없음)."""
    stats = ls.synthetic_stats()[0]
    limits = RunLimits(max_input_chars_per_call=4000, max_output_tokens_per_call=1000, max_cost_usd=0.05)
    budget = CallBudget(limits)
    ctx = RunContext(ls.to_sanitized(stats, ls.SourceKind.SYNTHETIC), FakeRunner(), budget,
                     make_checked_text, est_cost_usd=0.001)
    result = new_run("t", (ls.to_provenance(stats),), ls.RunMode.LIVE, limits, external_send_approved=True)
    return stats, limits, budget, ctx, advance(result, ctx)


@patch.object(socket, "socket", no_network)
class LiveSmokeTest(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                ls.main(list(argv))
            except SystemExit as error:
                return out.getvalue(), str(error)
        return out.getvalue(), None

    def test_plan_reports_model_calls_tokens_and_cost_only_with_prices(self):
        text, message = self.run_main()
        self.assertIsNone(message)
        for expected in ("gpt-6-luna", "9회", "출력 토큰 상한: 호출당 1,000", "실제 호출은 하지 않았습니다"):
            self.assertIn(expected, text)
        self.assertNotIn("$", text)
        priced, _ = self.run_main("--price-in", "0.10", "--price-out", "0.50")
        self.assertIn("$", priced)

    def test_estimate_and_cost_cap_for_approved_prices(self):
        est = ls.estimate(ls.synthetic_stats()[0], 1000)
        self.assertEqual(est["calls"], 9)
        self.assertGreater(est["input_tokens_ceiling"], est["input_tokens_realistic"])
        worst = 9 * ls.stage_cost(est, (0.10, 0.50))
        self.assertLessEqual(worst, 0.05)
        self.assertGreater(worst, 0)

    def test_paid_phases_require_every_approval_and_cost_flag(self):
        blocked = [[], ["--approve-paid-calls"], ["--price-in", "1", "--price-out", "2", "--max-cost-usd", "1"],
                   ["--approve-paid-calls", "--price-in", "1", "--max-cost-usd", "1"],
                   ["--approve-paid-calls", "--price-out", "2", "--max-cost-usd", "1"],
                   ["--approve-paid-calls", "--price-in", "1", "--price-out", "2"]]
        for phase in ("start", "resume", "human"):
            for flags in blocked:
                with self.subTest(phase=phase, flags=flags), patch("openai.AsyncOpenAI") as client:
                    _, message = self.run_main("--phase", phase, *flags)
                    self.assertIsNotNone(message)
                    client.assert_not_called()

    def test_start_refuses_when_a_previous_run_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "live_state.json"
            state.write_text("{}", encoding="utf-8")
            with (patch.object(ls, "STATE_PATH", state),
                  patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": "sk-test"}),
                  patch.object(ls, "advance") as run):
                _, message = self.run_main("--phase", "start", *FULL)
            self.assertIn("이미 실행 기록", message or "")
            run.assert_not_called()

    def test_final_and_status_need_saved_state(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ls, "STATE_PATH", Path(folder) / "none.json"):
            for phase in ("final", "status"):
                _, message = self.run_main("--phase", phase, "--decision", "approve")
                self.assertIn("저장된 실행 상태가 없습니다", message or "")

    def test_state_roundtrip_restores_stages_and_budget(self):
        stats, limits, budget, _, result = waiting_for_human()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            usage = [{"stage": "collector", "requests": 1, "input_tokens": 10, "output_tokens": 5}]
            ls.save_state(path, result, budget, usage, ["e"])
            loaded, loaded_budget, loaded_usage, errors = ls.load_state(path, stats, limits)
        self.assertEqual(loaded.stages, result.stages)
        self.assertIs(next_step(loaded).stage, StageName.HUMAN_REVIEW)
        self.assertEqual((loaded_budget.total, loaded_budget.committed_cost),
                         (budget.total, budget.committed_cost))
        self.assertEqual(loaded_budget.by_role, budget.by_role)
        self.assertEqual((loaded_usage, errors), (usage, ["e"]))

    def test_tampered_state_is_rejected_on_load(self):
        stats, limits, budget, _, result = waiting_for_human()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            ls.save_state(path, result, budget, [], [])
            text = path.read_text(encoding="utf-8").replace('"collector"', '"manager"', 1)
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                ls.load_state(path, stats, limits)

    def test_error_description_redacts_key_like_text(self):
        error = RuntimeError("Incorrect API key provided: sk-FAKE-EXAMPLE****1234")
        text = ls.describe_error(error)
        self.assertNotIn("EXAMPLE", text)
        self.assertIn("sk-***", text)

    def test_logging_runner_blocks_before_call_when_cost_would_exceed_cap(self):
        inner = Mock(usage_log=[])
        usage = [{"stage": "x", "requests": 1, "input_tokens": 400_000, "output_tokens": 0}]
        runner = ls.LoggingRunner(inner, (0.10, 0.50), 0.05, 0.02, usage, [])   # 이미 $0.04 사용
        with self.assertRaises(RuntimeError):
            runner.run_stage(Mock())
        inner.run_stage.assert_not_called()

    def test_logging_runner_records_failure_cause_and_reraises(self):
        inner = Mock(usage_log=[])
        inner.run_stage.side_effect = ValueError("bad parameter")
        errors: list = []
        runner = ls.LoggingRunner(inner, (0.10, 0.50), 0.05, 0.001, [], errors, fail_tokens=(20_000, 1_000))
        with redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
            runner.run_stage(SimpleNamespace(stage=StageName.COLLECTOR))
        self.assertIn("collector: ValueError", errors[0])
        # 실패한 요청도 과금됐을 수 있으므로 최악 비용(입력 20,000 + 출력 1,000 토큰)으로 누적한다.
        self.assertTrue(runner.usage_log[-1]["estimated"])
        self.assertAlmostEqual(runner.spent(), 20_000 * 0.10 / 1e6 + 1_000 * 0.50 / 1e6)

    def test_failed_request_cost_counts_toward_blocking_the_next_call(self):
        inner = Mock(usage_log=[])
        inner.run_stage.side_effect = RuntimeError("boom")
        runner = ls.LoggingRunner(inner, (0.10, 0.50), 0.003, 0.002, [], [], fail_tokens=(20_000, 1_000))
        with redirect_stderr(io.StringIO()), self.assertRaises(RuntimeError):
            runner.run_stage(SimpleNamespace(stage=StageName.COLLECTOR))   # 실패 비용 $0.0025 누적
        inner.run_stage.reset_mock()
        with self.assertRaises(RuntimeError):
            runner.run_stage(SimpleNamespace(stage=StageName.ANALYST))     # 0.0025 + 0.002 > 0.003 → 호출 전 차단
        inner.run_stage.assert_not_called()

    def test_state_is_locked_before_the_first_paid_call_and_blocks_a_second_start(self):
        stats = ls.synthetic_stats()[0]
        est = ls.estimate(stats, 1000)
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "live_state.json"
            with (patch.object(ls, "STATE_PATH", state), patch.object(ls, "estimate", return_value=est),
                  patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": "sk-test"}),
                  patch.object(ls, "advance", side_effect=RuntimeError("crash"))):
                with redirect_stderr(io.StringIO()), self.assertRaises(RuntimeError):
                    ls.main(["--phase", "start", *FULL])
                self.assertTrue(state.exists())                              # 중단되어도 잠금 파일이 남는다
                self.assertIn("stage_failed", state.read_text(encoding="utf-8"))
                _, message = self.run_main("--phase", "start", *FULL)
            self.assertIn("이미 실행 기록", message or "")

    def test_restored_state_continues_after_the_users_human_review_decision(self):
        stats, limits, budget, _, result = waiting_for_human()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            ls.save_state(path, result, budget, [], [])
            loaded, loaded_budget, _, _ = ls.load_state(path, stats, limits)
        ctx = RunContext(ls.to_sanitized(stats, ls.SourceKind.SYNTHETIC), FakeRunner(), loaded_budget,
                         make_checked_text, est_cost_usd=0.001)
        done = advance(loaded, ctx, HumanInput(ls.HumanDecision.APPROVE_AS_IS))
        self.assertIs(next_step(done).stage, StageName.FINAL_APPROVAL)
        self.assertEqual(loaded_budget.total, 5)                              # 복원된 4회 + Manager 1회

    def test_preflight_flags_unsafe_client_settings(self):
        stats, limits, _, _, result = waiting_for_human()
        est = ls.estimate(stats, 1000)
        args = SimpleNamespace(price_in=0.10, price_out=0.50, max_cost_usd=0.05, max_output_tokens=1000)
        sanitized = ls.to_sanitized(stats, ls.SourceKind.SYNTHETIC)
        runner = SimpleNamespace(inner=SimpleNamespace(max_output_tokens=1000))
        good = ls.preflight(args, est, limits, SimpleNamespace(max_retries=0), runner, result, sanitized)
        self.assertTrue(all(ok for _, ok in good))
        retrying = ls.preflight(args, est, limits, SimpleNamespace(max_retries=2), runner, result, sanitized)
        self.assertFalse(retrying[0][1])
        tiny = SimpleNamespace(**{**vars(args), "max_cost_usd": 0.001})
        over = ls.preflight(tiny, est, limits, SimpleNamespace(max_retries=0), runner, result, sanitized)
        self.assertFalse(all(ok for _, ok in over))

    def test_human_phase_refuses_when_not_waiting_for_human_without_calling(self):
        _, _, budget, ctx, result = waiting_for_human()
        result = advance(result, ctx, HumanInput(ls.HumanDecision.APPROVE_AS_IS))   # Manager까지 진행
        self.assertTrue(any(s.stage is StageName.MANAGER and s.status is StageStatus.SUCCEEDED
                            for s in result.stages))
        est = ls.estimate(ls.synthetic_stats()[0], 1000)       # advance를 가짜로 바꾸기 전에 계산
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "state.json"
            ls.save_state(state, result, budget, [], [])
            with (patch.object(ls, "STATE_PATH", state), patch.object(ls, "advance") as run,
                  patch.object(ls, "estimate", return_value=est),
                  patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": "sk-test"})):
                _, message = self.run_main("--phase", "human", "--decision", "as_is", *FULL)
        self.assertIn("Human Review 단계가 아닙니다", message or "")
        run.assert_not_called()

    def test_final_phase_requires_a_user_decision_and_writes_report_only_on_approval(self):
        _, _, budget, ctx, result = waiting_for_human()
        result = advance(result, ctx, HumanInput(ls.HumanDecision.APPROVE_AS_IS))
        with tempfile.TemporaryDirectory() as folder:
            state, report = Path(folder) / "state.json", Path(folder) / "report.md"
            ls.save_state(state, result, budget, [], [])
            with patch.object(ls, "STATE_PATH", state), patch.object(ls, "REPORT_PATH", report):
                _, message = self.run_main("--phase", "final")                   # 결정 없음 → 거부
                self.assertIn("--decision", message or "")
                self.assertFalse(report.exists())
                _, message = self.run_main("--phase", "final", "--decision", "approve")
                self.assertIsNone(message)
            self.assertTrue(report.read_text(encoding="utf-8").startswith("<!-- LIVE 합성 테스트"))
            self.assertIn("APPROVED", state.read_text(encoding="utf-8") + "APPROVED")


class WriterFailsRunner:
    """Collector·Analyst는 성공하고 Writer 초안에서 API 오류가 나는 가짜 러너(1회차 실패 재현)."""

    def run_stage(self, step):
        if step.stage is StageName.WRITER_DRAFT:
            raise RuntimeError("incomplete")
        return FakeRunner().run_stage(step)


class FakeSdk:
    """SdkRunner 대체: 네트워크 없이 호출 순서·출력 상한을 기록한다. fail_writer=True면 Writer에서 실패."""

    instances: ClassVar[list] = []
    fail_writer: ClassVar[bool] = False

    def __init__(self, model, approved_live=False, *, max_output_tokens=None, client=None, max_input_chars=None):
        self.model, self.approved_live, self.max_output_tokens, self.client = (
            model, approved_live, max_output_tokens, client)
        self.max_input_chars = max_input_chars
        self.usage_log: list = []
        self.calls: list = []
        FakeSdk.instances.append(self)

    def run_stage(self, step):
        self.calls.append(step.stage)
        if FakeSdk.fail_writer and step.stage is StageName.WRITER_DRAFT:
            raise RuntimeError("incomplete")
        self.usage_log.append({"stage": step.stage.value, "requests": 1, "input_tokens": 1000,
                               "output_tokens": 500})
        return FakeRunner().run_stage(step)


def failed_writer_state(path: Path):
    """실제 1회차 실패와 같은 모양의 상태 파일을 만든다(Collector·Analyst 성공, Writer API 실패, 장부 3회)."""
    stats = ls.synthetic_stats()[0]
    limits = RunLimits(max_input_chars_per_call=4000, max_output_tokens_per_call=1000, max_cost_usd=0.05)
    budget = CallBudget(limits)
    ctx = RunContext(ls.to_sanitized(stats, ls.SourceKind.SYNTHETIC), WriterFailsRunner(), budget,
                     make_checked_text, est_cost_usd=0.002295)
    result = new_run("live-smoke", (ls.to_provenance(stats),), ls.RunMode.LIVE, limits,
                     external_send_approved=True)
    result = advance(result, ctx)
    usage = [{"stage": "collector", "requests": 1, "input_tokens": 877, "output_tokens": 381},
             {"stage": "analyst", "requests": 1, "input_tokens": 1127, "output_tokens": 646},
             {"stage": "writer_draft", "requests": 1, "estimated": True,
              "input_tokens": 17946, "output_tokens": 1000}]
    ls.save_state(path, result, budget, usage, ["writer_draft: incomplete (과금 가능성: 최악 비용으로 계산)"])
    return stats, limits, budget, result


@patch.object(socket, "socket", no_network)
class ResumeTest(unittest.TestCase):
    RESUME = ("--phase", "resume", *FULL, "--max-output-tokens", "2500")

    def setUp(self):
        FakeSdk.instances, FakeSdk.fail_writer = [], False
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.state = self.folder / "live_state.json"
        self.stats, self.limits, self.budget, self.failed = failed_writer_state(self.state)
        self.before = self.state.read_text(encoding="utf-8")
        for patcher in (patch.object(ls, "STATE_PATH", self.state), patch.object(ls, "SdkRunner", FakeSdk),
                        patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": "sk-test"})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            try:
                ls.main(list(argv))
            except SystemExit as error:
                return out.getvalue(), str(error)
        return out.getvalue(), None

    def saved(self) -> dict:
        return json.loads(self.state.read_text(encoding="utf-8"))

    def test_resume_state_keeps_outputs_ledger_and_restarts_at_writer(self):
        limits = RunLimits(max_input_chars_per_call=4000, max_output_tokens_per_call=2500, max_cost_usd=0.05)
        result, budget, usage, errors, failed = ls.resume_state(self.state, self.stats, limits)
        self.assertEqual([s.stage for s in result.stages], [StageName.COLLECTOR, StageName.ANALYST])
        self.assertEqual(result.stages, self.failed.stages[:2])               # 성공한 출력 그대로 재사용
        self.assertIs(next_step(result).stage, StageName.WRITER_DRAFT)       # Writer부터 시작
        self.assertEqual((budget.total, budget.by_role["writer"]), (3, 1))   # 실패한 요청도 호출 횟수에 포함
        self.assertEqual(budget.committed_cost, self.budget.committed_cost)
        self.assertEqual(len(usage), 3)
        self.assertTrue(usage[-1]["estimated"])                              # 실패 비용 기록 유지
        self.assertEqual(len(errors), 1)
        self.assertEqual(failed[-1]["stage"], "writer_draft")

    def test_resume_runs_writer_first_with_new_output_cap_and_keeps_every_record(self):
        out, message = self.run_main(*self.RESUME)
        self.assertIsNone(message)
        for expected in ("[OK] OpenAI 클라이언트 max_retries=0", "[OK] max_turns=1", "[OK] 출력 토큰 상한 2500",
                         "[OK] 기존 장부", "호출 상한 총 9회"):
            self.assertIn(expected, out)
        self.assertNotIn("[실패]", out)
        sdk = FakeSdk.instances[0]
        self.assertEqual(sdk.max_output_tokens, 2500)
        self.assertEqual(sdk.calls, [StageName.WRITER_DRAFT, StageName.REVIEWER])   # Writer부터, 새 호출 2회
        data = self.saved()
        self.assertEqual((data["resumes"], len(data["failed_attempts"])), (1, 1))
        self.assertEqual(data["budget"]["total"], 5)                          # 기존 3 + 새 2 ≤ 9
        self.assertLessEqual(data["budget"]["total"], 9)
        self.assertEqual(sum(1 for u in data["usage"] if u.get("estimated")), 1)  # 실패 비용 항목 보존
        original = json.loads(self.before)
        self.assertEqual(data["stages"][:2], original["stages"][:2])          # Collector·Analyst 출력 보존
        self.assertTrue(data["budget"]["committed_cost"] <= 0.05)
        self.assertIs(next_step(ls.load_state(self.state, self.stats, self.limits)[0]).stage,
                      StageName.HUMAN_REVIEW)                                 # Human Review에서 멈춤
        self.assertIn("Human Review 결정이 필요합니다", out)

    def test_original_state_is_backed_up_before_the_first_paid_call(self):
        est = ls.estimate(self.stats, 2500)                 # advance를 가짜로 바꾸기 전에 계산
        with (patch.object(ls, "estimate", return_value=est),
              patch.object(ls, "advance", side_effect=RuntimeError("crash")),
              self.assertRaises(RuntimeError)):
            self.run_main(*self.RESUME)
        backups = list(self.folder.glob("live_state.before_resume_*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), self.before)   # 원본 그대로
        self.assertEqual(self.saved()["resumes"], 1)                            # 중단돼도 두 번째 resume 불가
        _, message = self.run_main(*self.RESUME)
        self.assertIn("이미 한 번 이어서 실행", message or "")

    def test_resume_is_limited_to_once_and_start_stays_blocked(self):
        self.run_main(*self.RESUME)
        total_calls = sum(len(i.calls) for i in FakeSdk.instances)
        self.assertEqual(total_calls, 2)
        _, message = self.run_main(*self.RESUME)
        self.assertIn("이미 한 번 이어서 실행", message or "")
        _, start_message = self.run_main("--phase", "start", *FULL)
        self.assertIn("이미 실행 기록", start_message or "")
        self.assertEqual(sum(len(i.calls) for i in FakeSdk.instances), total_calls)   # 새 모델 호출 없음

    def test_same_failure_again_stops_without_automatic_retry(self):
        FakeSdk.fail_writer = True
        self.run_main(*self.RESUME)
        self.assertEqual(FakeSdk.instances[0].calls, [StageName.WRITER_DRAFT])   # 재시도 없이 1회만 시도
        data = self.saved()
        self.assertEqual(sum(1 for u in data["usage"] if u.get("estimated")), 2)  # 두 번째 실패 비용도 장부에 기록
        self.assertEqual(len(data["failed_attempts"]), 1)
        _, message = self.run_main(*self.RESUME)
        self.assertIn("이미 한 번 이어서 실행", message or "")
        self.assertEqual(sum(len(i.calls) for i in FakeSdk.instances), 1)         # 두 번째 resume도 호출 없음

    def test_resume_refuses_when_the_last_stage_is_not_an_api_failure(self):
        _, limits, budget, _, waiting = waiting_for_human()
        other = self.folder / "other.json"
        ls.save_state(other, waiting, budget, [], [])
        with self.assertRaises(SystemExit):
            ls.resume_state(other, self.stats, limits)

    def test_resume_budget_check_blocks_when_cost_cap_is_too_small(self):
        out, message = self.run_main("--phase", "resume", "--approve-paid-calls", "--price-in", "0.10",
                                     "--price-out", "0.50", "--max-cost-usd", "0.002", "--max-output-tokens", "2500")
        self.assertIn("사전 점검 실패", message or "")
        self.assertIn("[실패]", out)
        self.assertEqual(FakeSdk.instances[0].calls, [])                        # 호출 없음
        self.assertEqual(self.saved()["resumes"], 0)                            # 잠금도 쓰지 않음


if __name__ == "__main__":
    unittest.main()
