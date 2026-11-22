import ast
import dataclasses
import sys
import unittest
from dataclasses import replace
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from voc.contracts import (
    ERROR_MESSAGES,
    ApprovalStatus,
    CallBudget,
    CheckedText,
    DataProvenance,
    ErrorCode,
    HumanDecision,
    NextStep,
    PipelineResult,
    ReviewVerdict,
    RunLimits,
    RunMode,
    SanitizedInput,
    SkipReason,
    SourceKind,
    StageInput,
    StageResult,
    StageStatus,
    check_live_allowed,
    next_step,
)
from voc.contracts import (
    StageName as S,
)

OK = StageStatus.SUCCEEDED
V = ReviewVerdict
D = HumanDecision
DATASET = "합성_v1"


def full_limits():
    return RunLimits(max_input_chars_per_call=1000, max_output_tokens_per_call=500,
                     max_cost_usd=1.0)


def sanitized(kind=SourceKind.SYNTHETIC, dataset=DATASET):
    return SanitizedInput(kind, dataset, "voc_aggregate_v1",
                          (("품목", "항공여객운송서비스"), ("기간", "2020-01"), ("건수", 12)))


def new(mode=RunMode.DRY_RUN, **kw):
    return PipelineResult(run_id="t", mode=mode, limits=kw.pop("limits", RunLimits()), **kw)


def ok(stage, round_no=0, version=0, **kw):
    calls = 0 if stage in (S.HUMAN_REVIEW, S.FINAL_APPROVAL) else 1
    return StageResult(stage, OK, round_no=round_no, artifact_version=version,
                       calls_used=calls, **kw)


def skipped(stage, round_no):
    return StageResult(stage, StageStatus.SKIPPED, round_no=round_no,
                       skip_reason=SkipReason.NO_REVISION_NEEDED)


def until_first_review(verdict):
    r = new()
    for st in (S.COLLECTOR, S.ANALYST):
        r = r.with_stage(ok(st))
    r = r.with_stage(ok(S.WRITER_DRAFT, version=1))
    return r.with_stage(ok(S.REVIEWER, version=1, verdict=verdict))


def drive(r, decisions, verdicts):
    """next_step을 따라 끝까지 실행하는 시뮬레이터. (결과, 실행 단계 목록, 마지막 NextStep)."""
    executed, version = [], 1
    decisions, verdicts = list(decisions), list(verdicts)
    for _ in range(40):
        step = next_step(r)
        for st, rn in step.skip:
            r = r.with_stage(skipped(st, rn))
        if step.stage is None or step.stage is S.FINAL_APPROVAL:
            if step.stage is S.FINAL_APPROVAL:
                executed.append((step.stage, step.round_no))
            return r, executed, step
        executed.append((step.stage, step.round_no))
        if step.stage is S.HUMAN_REVIEW:
            res = ok(step.stage, step.round_no, human_decision=decisions.pop(0))
        elif step.stage is S.WRITER_REVISION:
            version += 1
            res = ok(step.stage, step.round_no, version=version)
        elif step.stage is S.REVIEWER_RECHECK:
            res = ok(step.stage, step.round_no, version=version, verdict=verdicts.pop(0))
        else:
            res = ok(step.stage, step.round_no, version=version)
        r = r.with_stage(res)
    raise AssertionError("종료되지 않음")


def skipped_keys(r):
    return {(s.stage, s.round_no) for s in r.stages if s.status is StageStatus.SKIPPED}


class FlowTest(unittest.TestCase):
    def test_human_review_is_always_next_after_first_review(self):
        for verdict in (V.APPROVABLE, V.NEEDS_REVISION):
            self.assertEqual(next_step(until_first_review(verdict)), NextStep(S.HUMAN_REVIEW, 1))

    def test_as_is_skips_only_revision_and_recheck(self):
        r, executed, _ = drive(until_first_review(V.APPROVABLE), [D.APPROVE_AS_IS], [])
        self.assertEqual(executed, [(S.HUMAN_REVIEW, 1), (S.MANAGER, 0), (S.FINAL_APPROVAL, 0)])
        self.assertEqual(skipped_keys(r), {
            (S.WRITER_REVISION, 1), (S.REVIEWER_RECHECK, 1), (S.WRITER_REVISION, 2),
            (S.REVIEWER_RECHECK, 2)})
        self.assertFalse(any(stage is S.HUMAN_REVIEW for stage, _ in skipped_keys(r)))
        self.assertEqual(r.calls_used, 5)

    def test_human_request_revision_then_human_reviews_revised_draft(self):
        r, executed, _ = drive(until_first_review(V.APPROVABLE),
                               [D.REQUEST_REVISION, D.APPROVE_AS_IS], [V.APPROVABLE])
        self.assertEqual(executed, [
            (S.HUMAN_REVIEW, 1), (S.WRITER_REVISION, 1), (S.REVIEWER_RECHECK, 1),
            (S.HUMAN_REVIEW, 2), (S.MANAGER, 0), (S.FINAL_APPROVAL, 0)])
        self.assertEqual(skipped_keys(r), {(S.WRITER_REVISION, 2), (S.REVIEWER_RECHECK, 2)})
        self.assertEqual(r.calls_used, 7)

    def test_reviewer_needs_revision_cannot_be_overridden_by_human(self):
        _, executed, _ = drive(until_first_review(V.NEEDS_REVISION),
                               [D.APPROVE_AS_IS, D.APPROVE_AS_IS], [V.APPROVABLE])
        self.assertIn((S.WRITER_REVISION, 1), executed)
        self.assertIn((S.REVIEWER_RECHECK, 1), executed)

    def test_two_revisions_then_approved_uses_exactly_nine_calls(self):
        r, executed, step = drive(until_first_review(V.NEEDS_REVISION),
                                  [D.REQUEST_REVISION, D.REQUEST_REVISION],
                                  [V.NEEDS_REVISION, V.APPROVABLE])
        self.assertEqual(r.calls_used, 9)
        self.assertEqual(step.stage, S.FINAL_APPROVAL)
        self.assertEqual(executed[-2:], [(S.MANAGER, 0), (S.FINAL_APPROVAL, 0)])

    def test_still_needs_revision_after_limit_stops(self):
        _, _, last = drive(until_first_review(V.NEEDS_REVISION),
                           [D.REQUEST_REVISION, D.REQUEST_REVISION],
                           [V.NEEDS_REVISION, V.NEEDS_REVISION])
        self.assertEqual(last.stop_reason, ErrorCode.REVISION_LIMIT_EXCEEDED)

    def test_undetermined_first_review_stops(self):
        self.assertEqual(next_step(until_first_review(V.UNDETERMINED)).stop_reason,
                         ErrorCode.REVIEW_UNDETERMINED)

    def test_undetermined_recheck_stops(self):
        _, _, last = drive(until_first_review(V.NEEDS_REVISION), [D.REQUEST_REVISION],
                           [V.UNDETERMINED])
        self.assertEqual(last.stop_reason, ErrorCode.REVIEW_UNDETERMINED)

    def test_failed_stage_stops_flow(self):
        r = new().with_stage(StageResult(S.COLLECTOR, StageStatus.FAILED,
                                         error_code=ErrorCode.API_ERROR))
        self.assertEqual(next_step(r).stop_reason, ErrorCode.STAGE_FAILED)

    def test_inconsistent_history_stops_instead_of_crashing(self):
        pending = replace(new(), stages=(StageResult(S.COLLECTOR, StageStatus.PENDING),))
        self.assertEqual(next_step(pending).stop_reason, ErrorCode.STAGE_FAILED)
        forged = replace(new(), stages=(ok(S.COLLECTOR), ok(S.ANALYST), ok(S.WRITER_DRAFT, version=1),
                                        skipped(S.REVIEWER, 0)))
        self.assertEqual(next_step(forged).stop_reason, ErrorCode.STAGE_FAILED)

    def test_flow_ends_after_final_decision(self):
        r, _, _ = drive(until_first_review(V.APPROVABLE), [D.APPROVE_AS_IS], [])
        self.assertEqual(next_step(r.approve_final()), NextStep(None))


class HistoryGuardTest(unittest.TestCase):
    """with_stage()가 순서·회차·버전을 어긴 기록을 거부한다(수정 요청 우회 승인 방지)."""

    def test_cannot_skip_human_review(self):
        with self.assertRaises(ValueError):
            until_first_review(V.APPROVABLE).with_stage(ok(S.MANAGER))

    def test_request_revision_cannot_go_straight_to_manager(self):
        r = until_first_review(V.APPROVABLE).with_stage(
            ok(S.HUMAN_REVIEW, 1, human_decision=D.REQUEST_REVISION))
        with self.assertRaises(ValueError):
            r.with_stage(ok(S.MANAGER))

    def test_wrong_order_round_and_versions_rejected(self):
        with self.assertRaises(ValueError):
            new().with_stage(ok(S.ANALYST))                        # 순서 위반
        r = new().with_stage(ok(S.COLLECTOR)).with_stage(ok(S.ANALYST))
        with self.assertRaises(ValueError):
            r.with_stage(ok(S.WRITER_DRAFT, version=2))            # 초안은 버전 1
        r = r.with_stage(ok(S.WRITER_DRAFT, version=1))
        with self.assertRaises(ValueError):
            r.with_stage(ok(S.REVIEWER, version=5))                # 검수 대상 버전 불일치

    def test_stale_recheck_version_rejected(self):
        r = until_first_review(V.NEEDS_REVISION).with_stage(
            ok(S.HUMAN_REVIEW, 1, human_decision=D.REQUEST_REVISION))
        r = r.with_stage(ok(S.WRITER_REVISION, 1, version=2))
        with self.assertRaises(ValueError):
            r.with_stage(ok(S.REVIEWER_RECHECK, 1, version=1, verdict=V.APPROVABLE))

    def test_pending_final_approval_and_illegal_skip_rejected(self):
        r = new()
        with self.assertRaises(ValueError):
            r.with_stage(StageResult(S.COLLECTOR, StageStatus.PENDING))
        with self.assertRaises(ValueError):
            r.with_stage(ok(S.FINAL_APPROVAL))
        with self.assertRaises(ValueError):
            r.with_stage(skipped(S.COLLECTOR, 0))

    def test_failure_of_expected_stage_can_be_recorded_then_nothing_more(self):
        r = new().with_stage(StageResult(S.COLLECTOR, StageStatus.BLOCKED,
                                         error_code=ErrorCode.PREFLIGHT_BLOCKED))
        with self.assertRaises(ValueError):
            r.with_stage(ok(S.ANALYST))


class ApprovalTest(unittest.TestCase):
    def finished(self):
        r, _, _ = drive(until_first_review(V.APPROVABLE), [D.APPROVE_AS_IS], [])
        return r

    def test_new_result_is_draft(self):
        self.assertIs(new().approval, ApprovalStatus.DRAFT)

    def test_approve_after_full_flow(self):
        self.assertIs(self.finished().approve_final().approval, ApprovalStatus.APPROVED)

    def test_forged_histories_cannot_be_approved(self):
        good = self.finished().stages
        base = new()
        no_human = tuple(s for s in good if s.stage is not S.HUMAN_REVIEW)
        manager_early = (ok(S.COLLECTOR), ok(S.ANALYST), ok(S.WRITER_DRAFT, version=1),
                         ok(S.REVIEWER, version=1, verdict=V.APPROVABLE), ok(S.MANAGER))
        revision_ignored = manager_early[:4] + (
            ok(S.HUMAN_REVIEW, 1, human_decision=D.REQUEST_REVISION), ok(S.MANAGER))
        failed_first = (StageResult(S.COLLECTOR, StageStatus.FAILED, error_code=ErrorCode.API_ERROR),
                        ok(S.WRITER_DRAFT, version=1),
                        ok(S.REVIEWER, version=1, verdict=V.APPROVABLE),
                        ok(S.HUMAN_REVIEW, 1, human_decision=D.APPROVE_AS_IS), ok(S.MANAGER))
        for stages in (no_human, manager_early, revision_ignored, failed_first, good[:-1]):
            with self.assertRaises(ValueError):
                replace(base, stages=stages).approve_final()

    def test_failed_recheck_after_earlier_approvable_blocks_approval(self):
        r = until_first_review(V.APPROVABLE).with_stage(
            ok(S.HUMAN_REVIEW, 1, human_decision=D.REQUEST_REVISION))
        r = r.with_stage(ok(S.WRITER_REVISION, 1, version=2))
        r = r.with_stage(StageResult(S.REVIEWER_RECHECK, StageStatus.FAILED, round_no=1,
                                     artifact_version=2, error_code=ErrorCode.API_ERROR))
        with self.assertRaises(ValueError):
            r.approve_final()

    def test_stopped_run_cannot_be_approved_or_extended(self):
        r = self.finished().stop(ErrorCode.BUDGET_EXCEEDED)
        with self.assertRaises(ValueError):
            r.approve_final()
        with self.assertRaises(ValueError):
            r.with_stage(ok(S.MANAGER))

    def test_reject_is_terminal(self):
        r = self.finished().reject()
        self.assertIs(r.approval, ApprovalStatus.REJECTED)
        with self.assertRaises(ValueError):
            r.approve_final()

    def test_records_are_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            new().approval = ApprovalStatus.APPROVED  # pyright: ignore[reportAttributeAccessIssue]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ok(S.COLLECTOR).cost_usd = 5.0  # pyright: ignore[reportAttributeAccessIssue]


class ValidationTest(unittest.TestCase):
    def test_limits_defaults_hard_maximums_and_bad_values(self):
        lim = RunLimits()
        self.assertEqual((lim.max_total_calls, lim.max_writer_calls, lim.max_reviewer_calls,
                          lim.max_turns_per_stage), (9, 3, 3, 3))
        self.assertFalse(lim.is_complete())
        RunLimits(max_total_calls=5, max_turns_per_stage=1)       # 더 낮게는 허용
        for bad in ({"max_total_calls": 10}, {"max_writer_calls": 4}, {"max_reviewer_calls": 4},
                    {"max_turns_per_stage": 4}, {"max_total_calls": 0}, {"max_total_calls": True},
                    {"max_cost_usd": float("nan")}, {"max_cost_usd": -1.0},
                    {"max_cost_usd": float("inf")}, {"max_input_chars_per_call": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                RunLimits(**bad)  # pyright: ignore[reportArgumentType]

    def test_strict_types_block_approval_bypass(self):
        with self.assertRaises(ValueError):
            PipelineResult("t", "live", RunLimits())  # pyright: ignore[reportArgumentType]
        for name in ("external_send_approved", "real_data_send_approved", "tracing_disabled"):
            with self.assertRaises(ValueError):
                PipelineResult("t", RunMode.LIVE, RunLimits(), **{name: "false"})  # pyright: ignore[reportArgumentType]
        with self.assertRaises(ValueError):
            SanitizedInput("real", DATASET, "voc_aggregate_v1", ())  # pyright: ignore[reportArgumentType]
        with self.assertRaises(ValueError):
            CheckedText("x", pii_scan_passed="yes")  # pyright: ignore[reportArgumentType]

    def test_sanitized_input_accepts_only_schema_values(self):
        self.assertEqual(len(sanitized().fields), 3)
        bad_fields = [
            (("memo", "짧은 상담 원문"),), (("비고", "짧은 상담 원문"),),      # 필드 없음/금지 이름
            (("품목", "택시여객운송서비스"),), (("품목", "가" * 101),),        # 열거되지 않은 문자열
            (("기간", "어제"),), (("건수", -1),), (("건수", True),),
            (("건수", float("nan")),), (("비율", 1.5),), (("건수", [1]),),
            (("건수", 1), ("건수", 2)),                                         # 중복
            (("ACCIDENT_CONTENT", "x"),),
        ]
        for fields in bad_fields:
            with self.assertRaises(ValueError, msg=str(fields)):
                SanitizedInput(SourceKind.SYNTHETIC, DATASET, "voc_aggregate_v1", fields)
        with self.assertRaises(ValueError):
            SanitizedInput(SourceKind.SYNTHETIC, DATASET, "unknown_v9", ())
        with self.assertRaises(ValueError):
            SanitizedInput(SourceKind.SYNTHETIC, DATASET, "voc_aggregate_v1", [("건수", 1)])  # pyright: ignore[reportArgumentType]

    def test_validation_messages_do_not_echo_values(self):
        secret = "010-1234-5678"
        try:
            SanitizedInput(SourceKind.REAL, DATASET, "voc_aggregate_v1", (("품목", secret),))
        except ValueError as error:
            self.assertNotIn("010", str(error))

    def test_stage_result_rules(self):
        cases = [
            lambda: StageResult(S.COLLECTOR, verdict=V.APPROVABLE),
            lambda: StageResult(S.COLLECTOR, StageStatus.FAILED),
            lambda: StageResult(S.HUMAN_REVIEW, OK),                            # 사람 결정 없음
            lambda: StageResult(S.REVIEWER, OK),                                # 판정 없음
            lambda: StageResult(S.HUMAN_REVIEW, calls_used=1),
            lambda: StageResult(S.WRITER_REVISION, StageStatus.SKIPPED, calls_used=1,
                                skip_reason=SkipReason.NO_REVISION_NEEDED),
            lambda: StageResult(S.COLLECTOR, cost_usd=float("nan")),
            lambda: StageResult(S.COLLECTOR, round_no=3),
            lambda: StageResult("collector"),  # pyright: ignore[reportArgumentType]
        ]
        for make in cases:
            with self.assertRaises(ValueError):
                make()

    def test_every_error_code_has_fixed_message(self):
        self.assertEqual(set(ERROR_MESSAGES), set(ErrorCode))


class GateTest(unittest.TestCase):
    prov = (DataProvenance(DATASET, None, "합성 사례", 3, "합성 행 수", None),)

    def at_revision(self, mode=RunMode.LIVE, **kw):
        """다음 단계가 WRITER_REVISION(1회차)인 실행 상태를 만든다."""
        r = new(mode, **kw)
        for res in (ok(S.COLLECTOR), ok(S.ANALYST), ok(S.WRITER_DRAFT, version=1, output_text="초안"),
                    ok(S.REVIEWER, version=1, verdict=V.NEEDS_REVISION),
                    ok(S.HUMAN_REVIEW, 1, human_decision=D.REQUEST_REVISION)):
            r = r.with_stage(res)
        return r

    def live(self, **kw):
        base = {"limits": full_limits(), "external_send_approved": True, "provenance": self.prov}
        base.update(kw)
        return self.at_revision(RunMode.LIVE, **base)

    def step(self, data=None, notes=None, prior=(), stage=S.WRITER_REVISION, round_no=1):
        return StageInput(stage, round_no, 2, data or sanitized(), prior, notes)

    def test_dry_run_allowed_without_approval(self):
        r = self.at_revision(RunMode.DRY_RUN)
        self.assertTrue(check_live_allowed(r, self.step(sanitized(SourceKind.REAL))).allowed)

    def test_wrong_stage_or_finished_run_is_blocked_in_both_modes(self):
        for mode_result in (self.at_revision(RunMode.DRY_RUN), self.live(real_data_send_approved=True)):
            wrong = self.step(stage=S.MANAGER, round_no=0)
            self.assertIs(check_live_allowed(mode_result, wrong).error_code, ErrorCode.PREFLIGHT_BLOCKED)
            wrong_round = self.step(round_no=2)
            self.assertIs(check_live_allowed(mode_result, wrong_round).error_code,
                          ErrorCode.PREFLIGHT_BLOCKED)
            for ended in (mode_result.stop(ErrorCode.BUDGET_EXCEEDED), mode_result.reject()):
                self.assertIs(check_live_allowed(ended, self.step()).error_code,
                              ErrorCode.PREFLIGHT_BLOCKED)

    def test_real_data_never_sends_free_text_notes(self):
        r = self.live(real_data_send_approved=True)
        step = self.step(sanitized(SourceKind.REAL), notes=CheckedText("메모", True))
        self.assertIs(check_live_allowed(r, step).error_code, ErrorCode.FREE_TEXT_REJECTED)
        self.assertTrue(check_live_allowed(r, self.step(sanitized(SourceKind.REAL))).allowed)

    def test_live_blocked_until_each_gate_passes(self):
        unscanned = CheckedText("메모", False)
        cases = [
            (self.at_revision(RunMode.LIVE, limits=full_limits(), provenance=self.prov), self.step(),
             ErrorCode.EXTERNAL_SEND_NOT_APPROVED),
            (self.live(tracing_disabled=False), self.step(), ErrorCode.TRACING_ENABLED),
            (self.live(), self.step(sanitized(SourceKind.REAL)), ErrorCode.REAL_DATA_NOT_APPROVED),
            (self.live(limits=RunLimits()), self.step(), ErrorCode.LIMITS_NOT_SET),
            (self.live(provenance=()), self.step(), ErrorCode.DATA_NOT_APPROVED),
            (self.live(), self.step(sanitized(dataset="다른자료")), ErrorCode.DATA_NOT_APPROVED),
            (self.live(), self.step(notes=unscanned), ErrorCode.TEXT_NOT_SCANNED),
            (self.live(), self.step(prior=((S.WRITER_DRAFT, CheckedText("초안", False)),)),
             ErrorCode.TEXT_NOT_SCANNED),
            # 사람 메모를 모델 산출물처럼 위장하면 차단된다(내용 불일치).
            (self.live(), self.step(prior=((S.WRITER_DRAFT, CheckedText("메모", True)),)),
             ErrorCode.FREE_TEXT_REJECTED),
            (self.live(real_data_send_approved=True),
             self.step(sanitized(SourceKind.REAL), prior=((S.WRITER_DRAFT, CheckedText("메모", True)),)),
             ErrorCode.FREE_TEXT_REJECTED),
            (self.live(), self.step(prior=((S.ANALYST, CheckedText("초안", True)),)),
             ErrorCode.FREE_TEXT_REJECTED),                                # 다른 단계 이름으로 위장
            (self.live(), self.step(stage=S.HUMAN_REVIEW), ErrorCode.PREFLIGHT_BLOCKED),
        ]
        # 위 항목 중 마지막은 사람 단계 요청이라 모델 호출 게이트에서 차단된다.
        for result, step, code in cases:
            decision = check_live_allowed(result, step)
            self.assertFalse(decision.allowed, code)
            self.assertIs(decision.error_code, code)
            self.assertEqual(decision.message, ERROR_MESSAGES[code])

    def test_live_allowed_only_when_everything_is_approved_and_scanned(self):
        step = self.step(notes=CheckedText("메모", True),
                         prior=((S.WRITER_DRAFT, CheckedText("초안", True)),))
        self.assertTrue(check_live_allowed(self.live(), step).allowed)
        for label in (S.HUMAN_REVIEW, S.FINAL_APPROVAL):                  # 사람 단계 이름은 구조적으로 불가
            with self.assertRaises(ValueError):
                self.step(prior=((label, CheckedText("메모", True)),))
        real = self.step(sanitized(SourceKind.REAL))
        self.assertTrue(check_live_allowed(self.live(real_data_send_approved=True), real).allowed)

    def test_stage_input_counts_every_outgoing_character(self):
        step = self.step(notes=CheckedText("가나다", True), prior=((S.WRITER_DRAFT, CheckedText("라마", True)),))
        self.assertEqual(step.total_chars(), sanitized().total_chars() + 5)


class BudgetTest(unittest.TestCase):
    def test_full_flow_is_exactly_nine_calls_then_blocked(self):
        b = CallBudget(full_limits())
        plan = [S.COLLECTOR, S.ANALYST, S.WRITER_DRAFT, S.REVIEWER, S.WRITER_REVISION,
                S.REVIEWER_RECHECK, S.WRITER_REVISION, S.REVIEWER_RECHECK, S.MANAGER]
        for stage in plan:
            self.assertTrue(b.reserve(stage, 10, 0.01).allowed, stage)
        self.assertEqual(b.total, 9)
        for stage in (S.WRITER_REVISION, S.REVIEWER_RECHECK, S.MANAGER, S.COLLECTOR):
            self.assertIs(b.reserve(stage, 10, 0.01).error_code, ErrorCode.CALL_LIMIT_EXCEEDED)

    def test_retry_counts_as_another_call(self):
        b = CallBudget(full_limits())
        self.assertTrue(b.reserve(S.COLLECTOR, 10, 0.01).allowed)
        self.assertIs(b.reserve(S.COLLECTOR, 10, 0.01).error_code, ErrorCode.CALL_LIMIT_EXCEEDED)

    def test_input_size_cost_and_unset_limits(self):
        b = CallBudget(full_limits())
        self.assertIs(b.reserve(S.COLLECTOR, 1001, 0.01).error_code, ErrorCode.INPUT_TOO_LARGE)
        self.assertIs(b.reserve(S.COLLECTOR, 10, 2.0).error_code, ErrorCode.BUDGET_EXCEEDED)
        self.assertIs(b.reserve(S.COLLECTOR, 10, float("nan")).error_code, ErrorCode.BUDGET_EXCEEDED)
        self.assertEqual(b.total, 0)
        self.assertIs(CallBudget(RunLimits()).reserve(S.COLLECTOR, 1, 0.0).error_code,
                      ErrorCode.LIMITS_NOT_SET)

    def test_negative_or_non_integer_input_length_is_blocked(self):
        b = CallBudget(full_limits())
        for bad in (-1, 1.5, True):
            decision = b.reserve(S.COLLECTOR, bad, 0.01)  # pyright: ignore[reportArgumentType]
            self.assertIs(decision.error_code, ErrorCode.PREFLIGHT_BLOCKED)
        self.assertEqual(b.total, 0)

    def test_stage_input_and_dataset_id_validation(self):
        for kw in ({"round_no": 3}, {"round_no": True}, {"artifact_version": 9}):
            args = {"stage": S.WRITER_REVISION, "round_no": 1, "artifact_version": 2,
                    "sanitized": sanitized()}
            args.update(kw)
            with self.assertRaises(ValueError, msg=str(kw)):
                StageInput(**args)
        for bad in ("", "a b", "x" * 61, "010-1234 5678", "id\n"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                SanitizedInput(SourceKind.SYNTHETIC, bad, "voc_aggregate_v1", ())

    def test_human_stages_cannot_reserve_model_calls(self):
        self.assertIs(CallBudget(full_limits()).reserve(S.HUMAN_REVIEW, 1, 0.0).error_code,
                      ErrorCode.PREFLIGHT_BLOCKED)


class NoNetworkTest(unittest.TestCase):
    def test_contracts_imports_only_standard_pure_modules(self):
        tree = ast.parse((SRC / "voc" / "contracts.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertEqual(names, {"math", "re", "dataclasses", "enum", "typing"})


if __name__ == "__main__":
    unittest.main()
