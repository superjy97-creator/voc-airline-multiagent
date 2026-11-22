import json
from collections.abc import Iterable
from dataclasses import dataclass

from .config import VOC_AGENT_MODEL
from .contracts import (
    MODEL_STAGES,
    REVIEWER_STAGES,
    ReviewVerdict,
    StageInput,
    StageName,
    StageResult,
    StageStatus,
)
from .prompts import build_instructions

_DRY_RUN_LABEL = "DRY_RUN 합성 출력 — 실제 모델 결과 아님"
_REVIEW_LABELS = {
    ReviewVerdict.APPROVABLE: "승인 가능",
    ReviewVerdict.NEEDS_REVISION: "수정 필요",
    ReviewVerdict.UNDETERMINED: "확인 불가",
}


def _report(*sections: tuple[str, str]) -> str:
    return _DRY_RUN_LABEL + "\n\n" + "\n\n".join(
        f"## {title}\n{body}" for title, body in sections
    )


@dataclass(frozen=True)
class _ReviewOutput:
    verdict: ReviewVerdict
    output_text: str


class FakeRunner:
    def __init__(self, review_verdicts: Iterable[ReviewVerdict] = ()) -> None:
        self._verdicts = iter(review_verdicts)

    def run_stage(self, step: StageInput) -> StageResult:
        stage = step.stage
        version = 0
        verdict = None
        if stage in REVIEWER_STAGES:
            verdict = next(self._verdicts, ReviewVerdict.APPROVABLE)
            version = step.artifact_version
            instruction = (
                "‘위험·한계’ 절에 집계만으로 원인을 확인할 수 없음을 명시하세요."
                if verdict is ReviewVerdict.NEEDS_REVISION else "추가 수정 지시 없음."
            )
            output = _report(
                ("판정", _REVIEW_LABELS[verdict]),
                ("근거·위험", "합성 검수 판정입니다."),
                ("수정 지시", instruction),
                ("재검수 조건", "수정 시 해당 절의 근거와 한계 표현을 확인합니다."),
            )
        elif stage in (StageName.WRITER_DRAFT, StageName.WRITER_REVISION):
            version = 1 if stage is StageName.WRITER_DRAFT else step.round_no + 1
            output = self._write(step)
        else:
            output = self._summarize(step)
        return StageResult(
            stage, StageStatus.SUCCEEDED, round_no=step.round_no, artifact_version=version,
            output_text=output, verdict=verdict, calls_used=1, input_chars=step.total_chars(),
        )

    def _write(self, step: StageInput) -> str:
        stats = ", ".join(
            f"{name} {value}" for name, value in step.sanitized.fields
            if name in {"건수", "비율", "기간"}
        )
        output = _report(
            ("개요", "제공된 VOC 집계의 개선 검토 예시입니다."),
            ("VOC 현황", stats or "제공된 집계 수치가 없습니다."),
            ("핵심 원인", "가설(DRY_RUN 예시) — 안내 과정에 개선 여지가 있을 수 있습니다."),
            ("개선방안", "안내 과정과 절차를 점검합니다."),
            ("실행 우선순위", "자료의 범위와 집계 기준을 먼저 확인합니다."),
            ("위험·한계", "집계의 범위를 확인해야 합니다."),
            ("결론", "추가 근거를 확인한 뒤 개선 여부를 판단합니다."),
        )
        if step.stage is StageName.WRITER_REVISION:
            prior = {stage: checked for stage, checked in step.prior_outputs}
            previous = prior.get(StageName.WRITER_REVISION) or prior.get(StageName.WRITER_DRAFT)
            if previous is not None:
                output = previous.text
            output = output.partition("\n\n## 변경점\n")[0]
            output += "\n\n## 변경점\n집계만으로 원인을 확인할 수 없음을 명시했습니다."
        return output

    def _summarize(self, step: StageInput) -> str:
        if step.stage is StageName.COLLECTOR:
            summary = ", ".join(f"{name} {value}" for name, value in step.sanitized.fields)
            return _report(
                ("핵심 요약", summary or "제공된 집계 필드가 없습니다."),
                ("유형·키워드", "추가 분류 근거가 없습니다."),
                ("감정 신호", "집계에서 확인할 수 없습니다."),
                ("자료 품질·누락", "제공된 집계 필드만 확인했습니다."),
                ("추가 확인사항", "집계 기준을 확인합니다."),
            )
        if step.stage is StageName.ANALYST:
            return _report(
                ("원인·근거", "집계만으로 원인을 확인할 수 없습니다."),
                ("문제 우선순위", "자료 범위를 먼저 확인합니다."),
                ("고객 경험 영향", "추가 근거가 필요합니다."),
                ("개선 가설", "안내 절차를 점검할 수 있습니다(DRY_RUN 예시)."),
                ("실무 조치 제안", "집계 기준을 검토합니다."),
                ("불확실성", "원인과 영향은 미확인입니다."),
            )
        if step.stage is StageName.MANAGER:
            return _report(
                ("검수된 보고서 요약", "VOC 집계 검토 예시입니다."),
                ("개선 우선순위", "자료 범위와 집계 기준을 먼저 확인합니다."),
                ("재검수 결과", "단계 기록에서 검수 판정을 확인합니다."),
                ("미해결 위험", "원인과 영향은 미확인입니다."),
                ("사용자 최종 확인 필요사항", "보고서 및 검수 이력을 확인합니다."),
            )
        raise ValueError("모델 단계만 실행할 수 있습니다.")


MAX_TURNS = 1  # 도구가 없으므로 한 단계에 모델 요청이 정확히 1회 나가도록 고정


class SdkRunner:
    """OpenAI Agents SDK 어댑터. 승인(approved_live)과 출력 토큰 상한이 없으면 호출하지 않는다.

    한 단계에 모델 요청이 정확히 1회 나가도록 max_turns=1로 고정하며(도구 없음),
    HTTP 재시도까지 막으려면 max_retries=0인 client를 넘긴다.
    usage_log에는 호출마다 실제 요청 수·입력/출력 토큰을 남긴다(비용 확인용).
    """

    def __init__(
        self, model: str = VOC_AGENT_MODEL, approved_live: bool = False, *,
        max_output_tokens: int | None = None, client=None, max_input_chars: int | None = None,
    ) -> None:
        self.model = model
        self.approved_live = approved_live
        self.max_output_tokens = max_output_tokens
        self.client = client
        self.max_input_chars = max_input_chars   # 실제로 직렬화한 입력 글자 상한(호출 전 확인, None이면 확인 안 함)
        self.usage_log: list[dict] = []

    def run_stage(self, step: StageInput) -> StageResult:
        if self.approved_live is not True:
            raise RuntimeError("실제 SDK 호출에는 사전 승인이 필요합니다.")
        limit = self.max_output_tokens
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise RuntimeError("출력 토큰 상한이 설정되지 않아 호출할 수 없습니다.")
        if step.stage not in MODEL_STAGES:
            raise ValueError("모델 단계만 실행할 수 있습니다.")

        from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner

        is_review = step.stage in REVIEWER_STAGES
        model = (OpenAIResponsesModel(model=self.model, openai_client=self.client)
                 if self.client is not None else self.model)
        agent = Agent(
            name=step.stage.value, instructions=build_instructions(step.stage), model=model,
            model_settings=ModelSettings(max_tokens=self.max_output_tokens),
            output_type=_ReviewOutput if is_review else None,
        )
        payload = {
            "round_no": step.round_no,
            "artifact_version": step.artifact_version,
            "sanitized": {
                "source_kind": step.sanitized.source_kind.value,
                "dataset_id": step.sanitized.dataset_id,
                "schema_version": step.sanitized.schema_version,
                "fields": {name: value for name, value in step.sanitized.fields},
            },
            "prior_outputs": {stage.value: checked.text for stage, checked in step.prior_outputs},
            "human_notes": step.human_notes.text if step.human_notes is not None else None,
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        if self.max_input_chars is not None and len(serialized) > self.max_input_chars:
            raise RuntimeError("직렬화한 입력이 글자 상한을 넘어 호출하지 않습니다.")   # 요청 전 차단: 비용 없음
        result = Runner.run_sync(
            agent, serialized, max_turns=MAX_TURNS,
            run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False),
        )
        usage = result.context_wrapper.usage
        self.usage_log.append({"stage": step.stage.value, "requests": usage.requests,
                               "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens})
        return StageResult(
            step.stage, StageStatus.SUCCEEDED, round_no=step.round_no,
            artifact_version=step.artifact_version,
            output_text=result.final_output.output_text if is_review else result.final_output,
            verdict=result.final_output.verdict if is_review else None,
            calls_used=usage.requests, input_chars=step.total_chars(), output_tokens=usage.output_tokens,
        )
