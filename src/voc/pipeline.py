from collections.abc import Callable
from dataclasses import dataclass

from .contracts import (
    MODEL_STAGES,
    WRITER_STAGES,
    CallBudget,
    CheckedText,
    DataProvenance,
    ErrorCode,
    HumanDecision,
    ModelRunner,
    PipelineResult,
    PreflightDecision,
    RunLimits,
    RunMode,
    SanitizedInput,
    SkipReason,
    StageInput,
    StageName,
    StageResult,
    StageStatus,
    check_live_allowed,
    next_step,
)
from .runners import SdkRunner

DRY_RUN_LIMITS = RunLimits(
    max_input_chars_per_call=4000, max_output_tokens_per_call=1000, max_cost_usd=1.0,
)


@dataclass
class RunContext:
    sanitized: SanitizedInput
    runner: ModelRunner
    budget: CallBudget
    to_checked: Callable[[str], CheckedText]
    est_cost_usd: float = 0.0


@dataclass(frozen=True)
class HumanInput:
    decision: HumanDecision
    notes: str | None = None


def new_run(
    run_id: str, provenance: tuple[DataProvenance, ...], mode: RunMode = RunMode.DRY_RUN,
    limits: RunLimits = DRY_RUN_LIMITS, *, external_send_approved: bool = False,
    real_data_send_approved: bool = False, tracing_disabled: bool = True,
) -> PipelineResult:
    return PipelineResult(
        run_id, mode, limits, provenance=provenance, external_send_approved=external_send_approved,
        real_data_send_approved=real_data_send_approved, tracing_disabled=tracing_disabled,
    )


def _select_priors(stage: StageName, previous: dict[StageName, str]) -> dict[StageName, str]:
    """단계가 실제로 필요한 이전 산출물만 고른다(입력 글자 상한·비용 절약, 불필요한 전송 최소화).

    재검수는 수정본과 직전 검수 지시만, Manager는 최신 보고서와 최신 검수 결과만 받는다.
    첫 실제 실행에서 모든 산출물을 누적해 보내면 재검수·Manager 입력이 상한을 넘어 차단됐다.
    """
    if stage is StageName.REVIEWER_RECHECK:
        wanted = (StageName.WRITER_REVISION, StageName.REVIEWER)
    elif stage is StageName.MANAGER:
        wanted = tuple(name for name in (
            StageName.WRITER_REVISION if StageName.WRITER_REVISION in previous else StageName.WRITER_DRAFT,
            StageName.REVIEWER_RECHECK if StageName.REVIEWER_RECHECK in previous else StageName.REVIEWER,
        ))
    else:
        return previous
    return {name: previous[name] for name in wanted if name in previous}


def _model_input(
    result: PipelineResult, ctx: RunContext, stage: StageName, round_no: int,
) -> StageInput:
    previous: dict[StageName, str] = {}
    notes = None
    reviews: list[tuple[int, HumanDecision]] = []
    latest_version = 0
    for entry in result.stages:
        if entry.status is not StageStatus.SUCCEEDED:
            continue
        if entry.stage in WRITER_STAGES:
            latest_version = entry.artifact_version
        if entry.stage is StageName.HUMAN_REVIEW and entry.human_decision is not None:
            reviews.append((entry.round_no, entry.human_decision))
        if entry.output_text is None:
            continue
        if entry.stage in MODEL_STAGES:
            previous[entry.stage] = entry.output_text
        elif (entry.stage is StageName.HUMAN_REVIEW and entry.round_no == round_no
              and stage is StageName.WRITER_REVISION):
            notes = ctx.to_checked(entry.output_text)   # 자유 메모는 수정 지시를 받는 Writer(수정)에게만 전달

    version = round_no + 1 if stage in WRITER_STAGES else latest_version
    previous = _select_priors(stage, previous)
    return StageInput(
        stage, round_no, version, ctx.sanitized,
        tuple((name, ctx.to_checked(output)) for name, output in previous.items()), notes,
        tuple(reviews) if stage is StageName.MANAGER else (),
    )


def _run_model(result: PipelineResult, ctx: RunContext, step: StageInput) -> PipelineResult:
    gate = check_live_allowed(result, step)
    if gate.allowed and result.mode is RunMode.DRY_RUN and isinstance(ctx.runner, SdkRunner):
        gate = PreflightDecision(False, ErrorCode.PREFLIGHT_BLOCKED)
    if gate.allowed:
        gate = ctx.budget.reserve(step.stage, step.total_chars(), ctx.est_cost_usd)
    if not gate.allowed:
        code = gate.error_code or ErrorCode.PREFLIGHT_BLOCKED
        blocked = StageResult(
            step.stage, StageStatus.BLOCKED, round_no=step.round_no,
            artifact_version=step.artifact_version, error_code=code,
        )
        return result.with_stage(blocked).stop(code)

    try:
        entry = ctx.runner.run_stage(step)
    except Exception:  # noqa: BLE001 - 러너 예외는 계약의 고정 오류 코드로 기록한다.
        entry = StageResult(
            step.stage, StageStatus.FAILED, round_no=step.round_no,
            artifact_version=step.artifact_version, calls_used=1, error_code=ErrorCode.API_ERROR,
        )
    return result.with_stage(entry)


def advance(
    result: PipelineResult, ctx: RunContext, human: HumanInput | None = None,
) -> PipelineResult:
    if human is not None and human.notes is not None:
        stage = next_step(result).stage
        if stage not in (None, StageName.FINAL_APPROVAL):
            # 먼저 전달된 메모도 호출 예산을 쓰기 전에 검사한다.
            checked = ctx.to_checked(human.notes)
            if not checked.pii_scan_passed:
                raise ValueError("개인정보 의심 패턴이 있어 입력할 수 없습니다.")
            human = HumanInput(human.decision, checked.text)

    for _ in range(20):
        step = next_step(result)
        if step.stop_reason is not None:
            return result.stop(step.stop_reason)
        for skipped_stage, round_no in step.skip:
            result = result.with_stage(StageResult(
                skipped_stage, StageStatus.SKIPPED, round_no=round_no,
                skip_reason=SkipReason.NO_REVISION_NEEDED,
            ))
        if step.stage in (None, StageName.FINAL_APPROVAL):
            return result

        if step.stage is StageName.HUMAN_REVIEW:
            if human is None:
                return result
            result = result.with_stage(StageResult(
                step.stage, StageStatus.SUCCEEDED, round_no=step.round_no,
                human_decision=human.decision, output_text=human.notes,
            ))
            human = None
        else:
            result = _run_model(result, ctx, _model_input(result, ctx, step.stage, step.round_no))
    return result.stop(ErrorCode.STAGE_FAILED)
