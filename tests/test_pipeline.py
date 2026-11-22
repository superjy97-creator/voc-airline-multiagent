import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from test_contracts import D, S, V, sanitized

from voc.contracts import (
    ApprovalStatus,
    CallBudget,
    CheckedText,
    ErrorCode,
    RunMode,
    SkipReason,
    StageStatus,
    next_step,
)
from voc.pipeline import DRY_RUN_LIMITS, HumanInput, RunContext, advance, new_run
from voc.runners import FakeRunner, SdkRunner


class PipelineTest(unittest.TestCase):
    def setUp(self):
        socket = self.enterContext(patch("socket.socket", side_effect=AssertionError("네트워크 금지")))
        self.addCleanup(socket.assert_not_called)
        self.result = new_run("test-run", ())
        self.ctx = RunContext(
            sanitized(), FakeRunner(), CallBudget(self.result.limits),
            lambda text: CheckedText(text, "010-1234-5678" not in text),
        )

    def test_approve_as_is_uses_five_calls(self):
        waiting = advance(self.result, self.ctx)
        self.assertIs(next_step(waiting).stage, S.HUMAN_REVIEW)
        self.assertEqual(waiting.calls_used, 4)
        final = advance(waiting, self.ctx, HumanInput(D.APPROVE_AS_IS))
        self.assertIs(next_step(final).stage, S.FINAL_APPROVAL)
        self.assertIs(final.approval, ApprovalStatus.DRAFT)
        self.assertIs(final.stages[-1].stage, S.MANAGER)
        skips = [entry for entry in final.stages if entry.status is StageStatus.SKIPPED]
        self.assertEqual([(s.stage, s.round_no, s.skip_reason) for s in skips], [
            (stage, round_no, SkipReason.NO_REVISION_NEEDED)
            for round_no in (1, 2) for stage in (S.WRITER_REVISION, S.REVIEWER_RECHECK)
        ])
        self.assertEqual(final.calls_used, self.ctx.budget.total)
        self.assertEqual(final.calls_used, 5)
        self.assertIs(final.approve_final().approval, ApprovalStatus.APPROVED)

    def run_two_revisions(self, last_verdict):
        fake = FakeRunner((V.NEEDS_REVISION, V.NEEDS_REVISION, last_verdict))
        self.ctx.runner = Mock(wraps=fake)
        first = advance(self.result, self.ctx)
        second = advance(first, self.ctx, HumanInput(D.REQUEST_REVISION, "첫 메모"))
        self.assertEqual((next_step(second).stage, next_step(second).round_no), (S.HUMAN_REVIEW, 2))
        self.assertEqual(second.calls_used, 6)
        final = advance(second, self.ctx, HumanInput(D.APPROVE_AS_IS, "다음 메모"))
        steps = [call.args[0] for call in self.ctx.runner.run_stage.call_args_list]
        revisions = [step for step in steps if step.stage is S.WRITER_REVISION]
        self.assertEqual([(step.artifact_version, step.human_notes.text) for step in revisions],
                         [(2, "첫 메모"), (3, "다음 메모")])
        rechecks = [step for step in steps if step.stage is S.REVIEWER_RECHECK]
        self.assertEqual([step.artifact_version for step in rechecks], [2, 3])
        latest = [entry for entry in final.stages if entry.stage is S.WRITER_REVISION][-1]
        prior = {stage: checked for stage, checked in rechecks[-1].prior_outputs}
        self.assertEqual(prior[S.WRITER_REVISION].text, latest.output_text)
        return final

    def test_two_revisions_use_nine_calls(self):
        final = self.run_two_revisions(V.APPROVABLE)
        self.assertIs(next_step(final).stage, S.FINAL_APPROVAL)
        self.assertEqual(final.calls_used, 9)
        self.assertEqual(self.ctx.budget.total, 9)
        self.assertIs(final.approve_final().approval, ApprovalStatus.APPROVED)

    def test_revision_limit_stops_without_manager(self):
        final = self.run_two_revisions(V.NEEDS_REVISION)
        self.assertIs(final.stopped_reason, ErrorCode.REVISION_LIMIT_EXCEEDED)
        self.assertEqual(final.calls_used, 8)
        self.assertIs(final.stages[-1].stage, S.REVIEWER_RECHECK)

    def test_pii_note_leaves_history_and_budget_unchanged(self):
        waiting = advance(self.result, self.ctx)
        history = waiting.stages
        with self.assertRaisesRegex(ValueError, r"^개인정보 의심 패턴이 있어 입력할 수 없습니다\.$"):
            advance(waiting, self.ctx, HumanInput(D.REQUEST_REVISION, "010-1234-5678"))
        self.assertIs(waiting.stages, history)
        self.assertIs(advance(waiting, self.ctx), waiting)
        self.assertEqual(self.ctx.budget.total, 4)

    def test_preflight_and_budget_blocks_are_recorded(self):
        limited = replace(DRY_RUN_LIMITS, max_cost_usd=0.1)
        cases = (
            (new_run("budget", (), limits=limited), ErrorCode.BUDGET_EXCEEDED),
            (new_run("live", (), RunMode.LIVE), ErrorCode.EXTERNAL_SEND_NOT_APPROVED),
        )
        for result, code in cases:
            with self.subTest(code=code):
                runner = Mock()
                ctx = replace(self.ctx, budget=CallBudget(result.limits), est_cost_usd=0.2, runner=runner)
                blocked = advance(result, ctx)
                self.assertIs(blocked.stopped_reason, code)
                self.assertEqual(len(blocked.stages), 1)
                self.assertIs(blocked.stages[0].status, StageStatus.BLOCKED)
                self.assertIs(blocked.stages[0].error_code, code)
                self.assertEqual(ctx.budget.total, 0)
                runner.run_stage.assert_not_called()

    def test_sdk_cannot_run_in_dry_run(self):
        self.ctx.runner = SdkRunner("test-model", approved_live=True)
        with patch.object(self.ctx.runner, "run_stage") as run_stage:
            stopped = advance(self.result, self.ctx)
        run_stage.assert_not_called()
        self.assertIs(stopped.stopped_reason, ErrorCode.PREFLIGHT_BLOCKED)

    def test_runner_exception_records_only_fixed_error(self):
        self.ctx.runner = Mock()
        self.ctx.runner.run_stage.side_effect = RuntimeError("비공개 오류 내용")
        stopped = advance(self.result, self.ctx)
        self.assertIs(stopped.stopped_reason, ErrorCode.STAGE_FAILED)
        self.assertEqual(len(stopped.stages), 1)
        entry = stopped.stages[0]
        self.assertIs(entry.status, StageStatus.FAILED)
        self.assertIs(entry.error_code, ErrorCode.API_ERROR)
        self.assertIsNone(entry.output_text)
        self.assertNotIn("비공개 오류 내용", repr(stopped))
