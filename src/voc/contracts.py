"""1단계 인터페이스 계약 (v3: Codex 검토 2회 + 팀장·사용자 결정 반영).

단계 결과·상태 전이·오류 코드·승인 상태·호출 상한·모델 입력 경계 타입을 정의한다.
이 모듈은 데이터 구조와 검증 규칙만 담는다. API 호출·파일 읽기·네트워크 사용을 하지 않는다.
원칙(fail-closed): 값이 비었거나 승인되지 않았거나 판단할 수 없으면 허용하지 않는다.

실행 흐름(VOC_COMMON.md):
  Collector → Analyst → Writer(초안) → Reviewer → Human Review(항상 실행)
  → [수정 필요일 때만] Writer(수정) → Reviewer(재검수) → Human Review(수정본 확인)
  → [수정 필요일 때만] Writer(수정 2) → Reviewer(재검수 2) → Manager → 사용자 최종 승인
- Human Review는 Reviewer의 매 검수 직후(최대 2회) 실행하며, Reviewer가 '승인 가능'이어도 생략하지 않는다.
- 사람이 수정 없이 승인(APPROVE_AS_IS)하면 Writer 수정·Reviewer 재검수 슬롯만 SKIPPED로 기록한다.
  (발생하지 않은 Human Review는 기록하지 않는다.)
- Reviewer가 '수정 필요'로 판정했다면 사람이 그대로 승인해도 수정은 생략되지 않는다.
- 마지막 재검수(2회차) 뒤에는 Human Review 없이 Manager로 가며, 최종 사용자 승인이 마지막 사람 확인이다.

호출 상한의 의미:
- 총 9회는 '실제 외부 모델 요청 수'다. 재시도·SDK 내부 추가 요청도 같은 잔여량에서 차감한다.
- Writer 3 = 초안 1 + 수정 최대 2, Reviewer 3 = 최초 1 + 재검수 최대 2,
  Collector·Analyst·Manager 각 1. 사람 단계와 SKIPPED 단계는 0회다.
- 상한은 이 값보다 낮게만 설정할 수 있다(HARD_MAX_*).

한계(정직한 표기): CheckedText.pii_scan_passed 는 2단계 비식별화 모듈이 설정하는 '증명 표시'이며,
이 모듈이 개인정보를 직접 탐지하지는 않는다. 설정 주체는 2단계의 sanitize 모듈로 한정한다.
"""

import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol


class StageName(str, Enum):
    COLLECTOR = "collector"
    ANALYST = "analyst"
    WRITER_DRAFT = "writer_draft"
    REVIEWER = "reviewer"
    HUMAN_REVIEW = "human_review"
    WRITER_REVISION = "writer_revision"
    REVIEWER_RECHECK = "reviewer_recheck"
    MANAGER = "manager"
    FINAL_APPROVAL = "final_approval"


WRITER_STAGES = frozenset({StageName.WRITER_DRAFT, StageName.WRITER_REVISION})
REVIEWER_STAGES = frozenset({StageName.REVIEWER, StageName.REVIEWER_RECHECK})
MODEL_STAGES = frozenset({
    StageName.COLLECTOR, StageName.ANALYST, StageName.WRITER_DRAFT,
    StageName.REVIEWER, StageName.WRITER_REVISION,
    StageName.REVIEWER_RECHECK, StageName.MANAGER,
})
MAX_REVISION_ROUNDS = 2
HARD_MAX_TOTAL_CALLS = 9
HARD_MAX_WRITER_CALLS = 3
HARD_MAX_REVIEWER_CALLS = 3
HARD_MAX_TURNS_PER_STAGE = 3


class StageStatus(str, Enum):
    PENDING = "pending"      # 기록에는 넣을 수 없다(진행 중 표시는 기록 밖에서 한다)
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"      # 호출 전 검사(preflight)에서 차단됨
    SKIPPED = "skipped"


class SkipReason(str, Enum):
    NO_REVISION_NEEDED = "no_revision_needed"   # 수정이 필요 없어 수정 루프 생략


class HumanDecision(str, Enum):
    APPROVE_AS_IS = "approve_as_is"             # 수정 없이 승인
    REQUEST_REVISION = "request_revision"       # Writer에게 수정 요청


class ApprovalStatus(str, Enum):
    """사용자 최종 승인 상태. 승인 전 결과는 항상 DRAFT."""
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ReviewVerdict(str, Enum):
    """Reviewer 판정(VOC_REVIEWER.md: 승인 가능/수정 필요/확인 불가)."""
    APPROVABLE = "approvable"
    NEEDS_REVISION = "needs_revision"
    UNDETERMINED = "undetermined"


class RunMode(str, Enum):
    DRY_RUN = "dry_run"   # 가짜 모델. 네트워크 호출 0회.
    LIVE = "live"         # 실제 OpenAI API. 사용자 승인 후에만.


class SourceKind(str, Enum):
    SYNTHETIC = "synthetic"   # 합성 사례(향후 테스트 기본)
    REAL = "real"             # 실제 자료에서 비식별·집계한 값


class ErrorCode(str, Enum):
    PREFLIGHT_BLOCKED = "preflight_blocked"
    PII_DETECTED = "pii_detected"
    TEXT_NOT_SCANNED = "text_not_scanned"
    FREE_TEXT_REJECTED = "free_text_rejected"
    DATA_NOT_APPROVED = "data_not_approved"
    REAL_DATA_NOT_APPROVED = "real_data_not_approved"
    EXTERNAL_SEND_NOT_APPROVED = "external_send_not_approved"
    TRACING_ENABLED = "tracing_enabled"
    LIMITS_NOT_SET = "limits_not_set"
    CALL_LIMIT_EXCEEDED = "call_limit_exceeded"
    INPUT_TOO_LARGE = "input_too_large"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    INVALID_OUTPUT = "invalid_output"
    API_ERROR = "api_error"
    STAGE_FAILED = "stage_failed"
    REVIEW_UNDETERMINED = "review_undetermined"
    REVISION_LIMIT_EXCEEDED = "revision_limit_exceeded"
    USER_REJECTED = "user_rejected"


# 표시 문구는 코드에서만 만든다. 예외 메시지·원문이 기록에 섞이지 않게 한다.
ERROR_MESSAGES = {
    ErrorCode.PREFLIGHT_BLOCKED: "호출 전 검사에서 차단되었습니다.",
    ErrorCode.PII_DETECTED: "개인정보 의심 패턴이 탐지되어 중단했습니다.",
    ErrorCode.TEXT_NOT_SCANNED: "개인정보 검사를 통과하지 않은 텍스트가 있어 차단했습니다.",
    ErrorCode.FREE_TEXT_REJECTED: "자유서술(원문) 필드는 전송할 수 없습니다.",
    ErrorCode.DATA_NOT_APPROVED: "승인된 자료 출처 정보가 없거나 입력 자료와 일치하지 않습니다.",
    ErrorCode.REAL_DATA_NOT_APPROVED: "실제 자료의 외부 전송이 승인되지 않았습니다.",
    ErrorCode.EXTERNAL_SEND_NOT_APPROVED: "외부 전송이 승인되지 않았습니다.",
    ErrorCode.TRACING_ENABLED: "추적(tracing)이 켜져 있어 차단했습니다.",
    ErrorCode.LIMITS_NOT_SET: "실행 상한이 설정되지 않았습니다.",
    ErrorCode.CALL_LIMIT_EXCEEDED: "호출 횟수 상한을 초과했습니다.",
    ErrorCode.INPUT_TOO_LARGE: "입력 크기 상한을 초과했습니다.",
    ErrorCode.BUDGET_EXCEEDED: "예산 상한을 초과했습니다.",
    ErrorCode.MAX_TURNS_EXCEEDED: "단계별 최대 턴을 초과했습니다.",
    ErrorCode.INVALID_OUTPUT: "모델 출력 형식이 올바르지 않습니다.",
    ErrorCode.API_ERROR: "모델 호출 중 오류가 발생했습니다.",
    ErrorCode.STAGE_FAILED: "단계 실행에 실패했거나 기록이 올바르지 않아 중단했습니다.",
    ErrorCode.REVIEW_UNDETERMINED: "검수 판정을 확인할 수 없어 중단했습니다.",
    ErrorCode.REVISION_LIMIT_EXCEEDED: "수정 횟수 상한 이후에도 수정이 필요합니다.",
    ErrorCode.USER_REJECTED: "사용자가 반려했습니다.",
}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)     # 메시지에는 입력값을 넣지 않는다


@dataclass(frozen=True)
class RunLimits:
    """실행 상한. 입력·토큰·비용이 None이면 LIVE에서 차단된다(fail-closed)."""
    max_total_calls: int = HARD_MAX_TOTAL_CALLS
    max_writer_calls: int = HARD_MAX_WRITER_CALLS
    max_reviewer_calls: int = HARD_MAX_REVIEWER_CALLS
    max_turns_per_stage: int = HARD_MAX_TURNS_PER_STAGE
    max_input_chars_per_call: int | None = None
    max_output_tokens_per_call: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self):
        for name, hard_max in (("max_total_calls", HARD_MAX_TOTAL_CALLS),
                               ("max_writer_calls", HARD_MAX_WRITER_CALLS),
                               ("max_reviewer_calls", HARD_MAX_REVIEWER_CALLS),
                               ("max_turns_per_stage", HARD_MAX_TURNS_PER_STAGE)):
            value = getattr(self, name)
            _require(_is_int(value) and 1 <= value <= hard_max,
                     "호출 상한은 1 이상이며 승인된 최댓값을 넘을 수 없습니다.")
        for name in ("max_input_chars_per_call", "max_output_tokens_per_call"):
            value = getattr(self, name)
            _require(value is None or (_is_int(value) and value >= 1),
                     "크기·토큰 상한은 1 이상의 정수여야 합니다.")
        cost = self.max_cost_usd
        _require(cost is None or (_is_number(cost) and math.isfinite(cost) and cost > 0),
                 "비용 상한은 0보다 큰 유한한 수여야 합니다.")

    def is_complete(self) -> bool:
        return None not in (self.max_input_chars_per_call,
                            self.max_output_tokens_per_call, self.max_cost_usd)


@dataclass(frozen=True)
class DataProvenance:
    """통계·입력의 출처 정보. VOC_COMMON.md가 요구하는 항목."""
    dataset: str
    period: str | None
    filter_rule: str
    denominator: int
    denominator_basis: str
    dedup_basis: str | None
    limitations: tuple = ()


# ---- 모델 입력 경계 ---------------------------------------------------------

FORBIDDEN_FIELD_NAMES = frozenset(name.casefold() for name in (
    "ACCIDENT_TITLE", "ACCIDENT_CONTENT", "ANS_CONTENT",
    "사건제목", "사건내용", "답변내용", "질문", "답변", "질문본문", "답변본문",
    "content", "text", "body", "raw", "원문", "memo",
))
PERIOD_PATTERN = re.compile(r"\d{4}(-\d{2}){0,2}")
DATASET_ID_PATTERN = re.compile(r"[A-Za-z0-9가-힣_.-]{1,60}")


@dataclass(frozen=True)
class FieldSpec:
    """스키마의 한 필드. 문자열은 미리 열거된 범주값(category)과 기간 형식(period)만 허용한다."""
    kind: str                       # "category" | "count" | "ratio" | "period"
    allowed: frozenset = frozenset()

    def __post_init__(self):
        _require(self.kind in ("category", "count", "ratio", "period"), "알 수 없는 필드 종류입니다.")
        _require((self.kind == "category") == bool(self.allowed),
                 "category 필드만 비어 있지 않은 허용 목록을 가집니다.")

    def accepts(self, value) -> bool:
        if self.kind == "category":
            return isinstance(value, str) and value in self.allowed
        if self.kind == "count":
            return _is_int(value) and value >= 0
        if self.kind == "ratio":
            return _is_number(value) and math.isfinite(value) and 0 <= value <= 1
        return isinstance(value, str) and PERIOD_PATTERN.fullmatch(value) is not None


# 스키마 버전별 허용 필드. 2단계에서 필드를 늘리려면 이 표(계약)를 함께 바꾼다.
SCHEMAS = {
    "voc_aggregate_v1": {
        "품목": FieldSpec("category", frozenset({"항공여객운송서비스"})),
        "자료유형": FieldSpec("category", frozenset({"개별 상담 집계", "월·품목 집계"})),
        "기간": FieldSpec("period"),
        "건수": FieldSpec("count"),
        "비율": FieldSpec("ratio"),
    },
}

# 월별 변화 분석용 스키마(실제 집계 LIVE). 필드는 열거값·숫자·기간뿐이라 상담 원문·제목·자유 문자열은 담을 수 없다.
# 아래 두 문장은 모델에 전달하는 '고정 지시'이며(범주값 1개), 자료에서 온 값이 아니다.
REQUEST_SECTIONS = "월별변화·주요통계·확인가능한문제·원인가설·개선방안·기대효과를 구분해 작성"
INTERPRETATION_RULE = "통계로 확인할 수 없는 불만유형·원인·효과는 가설로만 표기하고 단정하지 않음"
SCOPES = frozenset({"전체 상담 접수", "온라인 상거래 상담", "위해위험 상담"})
MONTHLY_FIELD_MONTHS = tuple(f"{y}-{m:02d}" for y in range(2010, 2031) for m in range(1, 13))
SCHEMAS["voc_monthly_v1"] = {
    **SCHEMAS["voc_aggregate_v1"],
    "자료범위": FieldSpec("category", SCOPES),
    "기간_시작": FieldSpec("period"),
    "기간_종료": FieldSpec("period"),
    "전체건수": FieldSpec("count"),
    "판정불가건수": FieldSpec("count"),
    "마지막월_부분집계": FieldSpec("category", frozenset({"예", "아니오"})),
    "행수_상한_절단_가능성": FieldSpec("category", frozenset({"예", "미확인"})),
    "요청_절구성": FieldSpec("category", frozenset({REQUEST_SECTIONS})),
    "해석_원칙": FieldSpec("category", frozenset({INTERPRETATION_RULE})),
    **{f"건수_{month}": FieldSpec("count") for month in MONTHLY_FIELD_MONTHS},
}


@dataclass(frozen=True)
class SanitizedInput:
    """2단계(로더·비식별화)가 4단계(파이프라인)에 넘기는 유일한 모델 입력 타입.

    스키마에 열거된 필드·범주값·숫자·기간만 담을 수 있다. 자유로운 문자열은 구조적으로 불가능하다.
    """
    source_kind: SourceKind
    dataset_id: str
    schema_version: str
    fields: tuple    # ((이름, 값), ...)

    def __post_init__(self):
        _require(isinstance(self.source_kind, SourceKind), "source_kind는 SourceKind여야 합니다.")
        _require(isinstance(self.dataset_id, str)
                 and DATASET_ID_PATTERN.fullmatch(self.dataset_id) is not None,
                 "dataset_id 형식이 올바르지 않습니다.")
        schema = SCHEMAS.get(self.schema_version)
        if schema is None:
            raise ValueError("알 수 없는 스키마 버전입니다.")
        _require(isinstance(self.fields, tuple), "fields는 튜플이어야 합니다.")
        seen = set()
        for pair in self.fields:
            _require(isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str),
                     "fields 항목 형식이 올바르지 않습니다.")
            name, value = pair
            _require(name.casefold() not in FORBIDDEN_FIELD_NAMES, "자유서술 필드는 허용되지 않습니다.")
            _require(name in schema and name not in seen, "스키마에 없거나 중복된 필드입니다.")
            _require(schema[name].accepts(value), "필드 값이 스키마에 맞지 않습니다.")
            seen.add(name)

    def total_chars(self) -> int:
        """전송될 구조화 값의 글자 수(필드명·값·식별자 포함). 하한값이다."""
        return (len(self.dataset_id) + len(self.schema_version)
                + sum(len(str(n)) + len(str(v)) for n, v in self.fields))


@dataclass(frozen=True)
class CheckedText:
    """사람 메모·이전 단계 산출물처럼 자유 텍스트를 외부로 보낼 때 쓰는 포장.

    pii_scan_passed 는 2단계 비식별화 모듈이 검사 통과 후에만 True로 만든다.
    """
    text: str
    pii_scan_passed: bool = False

    def __post_init__(self):
        _require(isinstance(self.text, str), "text는 문자열이어야 합니다.")
        _require(isinstance(self.pii_scan_passed, bool), "pii_scan_passed는 bool이어야 합니다.")


@dataclass(frozen=True)
class StageInput:
    """모델 러너에 넘기는 단계 입력. 외부 전송될 내용은 이 객체 전체다."""
    stage: StageName
    round_no: int
    artifact_version: int
    sanitized: SanitizedInput
    prior_outputs: tuple = ()          # ((StageName, CheckedText), ...)
    human_notes: CheckedText | None = None

    def __post_init__(self):
        _require(isinstance(self.stage, StageName), "stage는 StageName이어야 합니다.")
        _require(_is_int(self.round_no) and 0 <= self.round_no <= MAX_REVISION_ROUNDS,
                 "회차 범위가 올바르지 않습니다.")
        _require(_is_int(self.artifact_version) and 0 <= self.artifact_version <= MAX_REVISION_ROUNDS + 1,
                 "보고서 버전 범위가 올바르지 않습니다.")
        _require(isinstance(self.sanitized, SanitizedInput), "sanitized 타입이 올바르지 않습니다.")
        _require(isinstance(self.prior_outputs, tuple) and all(
            isinstance(p, tuple) and len(p) == 2 and p[0] in MODEL_STAGES
            and isinstance(p[1], CheckedText) for p in self.prior_outputs),
            "prior_outputs는 모델 단계의 산출물만 담을 수 있습니다.")
        _require(self.human_notes is None or isinstance(self.human_notes, CheckedText),
                 "human_notes는 CheckedText여야 합니다.")

    def texts(self) -> tuple:
        notes = (self.human_notes,) if self.human_notes else ()
        return tuple(p[1] for p in self.prior_outputs) + notes

    def total_chars(self) -> int:
        """외부 전송 글자 수의 하한. 4단계는 실제로 직렬화한 요청 길이로 CallBudget.reserve를 호출해야 한다."""
        return self.sanitized.total_chars() + sum(len(t.text) for t in self.texts())


# ---- 기록 -------------------------------------------------------------------

@dataclass(frozen=True)
class StageResult:
    stage: StageName
    status: StageStatus = StageStatus.PENDING
    round_no: int = 0                 # 0=최초, 1..2=수정 회차
    artifact_version: int = 0         # Writer가 만든 / Reviewer가 검토한 보고서 버전
    output_text: str | None = None
    verdict: ReviewVerdict | None = None
    human_decision: HumanDecision | None = None
    error_code: ErrorCode | None = None
    skip_reason: SkipReason | None = None
    calls_used: int = 0
    input_chars: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self):
        _require(isinstance(self.stage, StageName) and isinstance(self.status, StageStatus),
                 "stage·status 타입이 올바르지 않습니다.")
        _require(self.verdict is None or isinstance(self.verdict, ReviewVerdict), "판정 타입 오류")
        _require(self.human_decision is None or isinstance(self.human_decision, HumanDecision),
                 "사람 결정 타입 오류")
        _require(self.error_code is None or isinstance(self.error_code, ErrorCode), "오류 코드 타입 오류")
        _require(self.skip_reason is None or isinstance(self.skip_reason, SkipReason), "생략 사유 타입 오류")
        _require(_is_int(self.round_no) and 0 <= self.round_no <= MAX_REVISION_ROUNDS,
                 "회차 범위가 올바르지 않습니다.")
        _require(self.verdict is None or self.stage in REVIEWER_STAGES,
                 "판정은 Reviewer 단계에만 둘 수 있습니다.")
        _require(self.human_decision is None or self.stage is StageName.HUMAN_REVIEW,
                 "사람 결정은 Human Review 단계에만 둘 수 있습니다.")
        _require(not (self.stage is StageName.HUMAN_REVIEW and self.status is StageStatus.SUCCEEDED
                      and self.human_decision is None),
                 "완료된 Human Review에는 사람 결정이 필요합니다.")
        _require(not (self.stage in REVIEWER_STAGES and self.status is StageStatus.SUCCEEDED
                      and self.verdict is None),
                 "완료된 Reviewer 단계에는 판정이 필요합니다.")
        _require(not (self.status in (StageStatus.FAILED, StageStatus.BLOCKED)
                      and self.error_code is None),
                 "실패·차단 단계에는 오류 코드가 필요합니다.")
        _require(not (self.status is StageStatus.SKIPPED
                      and (self.skip_reason is None or self.calls_used)),
                 "SKIPPED 단계는 사유가 필요하고 호출 수가 0이어야 합니다.")
        _require(not (self.stage not in MODEL_STAGES and self.calls_used),
                 "사람 단계는 모델 호출 수가 0이어야 합니다.")
        _require(_is_int(self.calls_used) and _is_int(self.input_chars) and _is_int(self.output_tokens)
                 and min(self.calls_used, self.input_chars, self.output_tokens) >= 0,
                 "사용량은 0 이상의 정수여야 합니다.")
        _require(_is_number(self.cost_usd) and math.isfinite(self.cost_usd) and self.cost_usd >= 0,
                 "비용은 0 이상의 유한한 수여야 합니다.")


@dataclass(frozen=True)
class NextStep:
    """상태 전이 결과. stage가 None이면 끝났거나 중단된 것이다."""
    stage: StageName | None
    round_no: int = 0
    skip: tuple = ()                   # ((단계, 회차), ...) SKIPPED로 먼저 기록할 단계
    stop_reason: ErrorCode | None = None


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    mode: RunMode
    limits: RunLimits
    provenance: tuple = ()
    stages: tuple = ()
    external_send_approved: bool = False     # 사용자 명시 승인 전에는 False
    real_data_send_approved: bool = False    # 실제 자료 전송은 별도 승인
    tracing_disabled: bool = True
    approval: ApprovalStatus = ApprovalStatus.DRAFT
    stopped_reason: ErrorCode | None = None

    def __post_init__(self):
        _require(isinstance(self.run_id, str) and bool(self.run_id), "run_id가 필요합니다.")
        _require(isinstance(self.mode, RunMode), "mode는 RunMode여야 합니다.")
        _require(isinstance(self.limits, RunLimits), "limits 타입이 올바르지 않습니다.")
        for name in ("external_send_approved", "real_data_send_approved", "tracing_disabled"):
            _require(isinstance(getattr(self, name), bool), "승인·설정 값은 bool이어야 합니다.")
        _require(isinstance(self.approval, ApprovalStatus), "approval 타입이 올바르지 않습니다.")
        _require(self.stopped_reason is None or isinstance(self.stopped_reason, ErrorCode),
                 "stopped_reason 타입이 올바르지 않습니다.")
        _require(isinstance(self.provenance, tuple)
                 and all(isinstance(p, DataProvenance) for p in self.provenance),
                 "provenance 형식이 올바르지 않습니다.")
        _require(isinstance(self.stages, tuple) and all(isinstance(s, StageResult) for s in self.stages),
                 "stages 형식이 올바르지 않습니다.")

    @property
    def calls_used(self) -> int:
        return sum(s.calls_used for s in self.stages)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.stages)

    def with_stage(self, res: StageResult) -> "PipelineResult":
        """다음에 와야 할 단계(또는 next_step이 지정한 SKIPPED)만 기록할 수 있다."""
        _require(isinstance(res, StageResult), "StageResult만 추가할 수 있습니다.")
        _require(self.approval is ApprovalStatus.DRAFT and self.stopped_reason is None,
                 "종결된 실행에는 단계를 추가할 수 없습니다.")
        _require(res.status is not StageStatus.PENDING, "진행 중 단계는 기록할 수 없습니다.")
        _require(res.stage is not StageName.FINAL_APPROVAL,
                 "최종 승인은 approve_final()/reject()로만 처리합니다.")
        step = next_step(self)
        if res.status is StageStatus.SKIPPED:
            _require((res.stage, res.round_no) in step.skip, "지금은 생략할 수 없는 단계입니다.")
            return replace(self, stages=self.stages + (res,))
        _require(step.stage is not None and (res.stage, res.round_no) == (step.stage, step.round_no),
                 "예상한 다음 단계가 아닙니다.")
        if res.status is StageStatus.SUCCEEDED:
            self._check_versions(res)
        return replace(self, stages=self.stages + (res,))

    def _check_versions(self, res: StageResult) -> None:
        latest = max((s.artifact_version for s in self.stages
                      if s.stage in WRITER_STAGES and s.status is StageStatus.SUCCEEDED), default=0)
        if res.stage is StageName.WRITER_DRAFT:
            _require(res.artifact_version == 1, "초안은 버전 1이어야 합니다.")
        elif res.stage is StageName.WRITER_REVISION:
            _require(res.artifact_version == res.round_no + 1, "수정본 버전이 올바르지 않습니다.")
        elif res.stage in REVIEWER_STAGES:
            _require(res.artifact_version == latest, "최신 보고서 버전을 검수해야 합니다.")

    def stop(self, reason: ErrorCode) -> "PipelineResult":
        if self.stopped_reason is not None:
            return self
        return replace(self, stopped_reason=reason)

    def approve_final(self) -> "PipelineResult":
        """사용자 최종 승인(DRAFT → APPROVED). 기록 전체를 처음부터 다시 검증한다."""
        _require(self.approval is ApprovalStatus.DRAFT and self.stopped_reason is None,
                 "승인할 수 없는 상태입니다.")
        replayed = replace(self, stages=())
        for s in self.stages:
            replayed = replayed.with_stage(s)           # 순서·회차·버전이 어긋나면 ValueError
        step = next_step(self)
        _require(step.stage is StageName.FINAL_APPROVAL and not step.skip,
                 "승인 단계에 도달하지 않았습니다.")
        reviews = [s for s in self.stages
                   if s.stage in REVIEWER_STAGES and s.status is StageStatus.SUCCEEDED]
        _require(bool(reviews) and reviews[-1].verdict is ReviewVerdict.APPROVABLE,
                 "마지막 검수가 '승인 가능'이 아닙니다.")
        _require(any(s.stage is StageName.MANAGER and s.status is StageStatus.SUCCEEDED
                     for s in self.stages), "Manager 종합이 완료되지 않았습니다.")
        return replace(self, approval=ApprovalStatus.APPROVED)

    def reject(self) -> "PipelineResult":
        _require(self.approval is ApprovalStatus.DRAFT, "이미 종결된 실행입니다.")
        return replace(self, approval=ApprovalStatus.REJECTED,
                       stopped_reason=ErrorCode.USER_REJECTED)


# ---- 상태 전이 --------------------------------------------------------------

def next_step(result: PipelineResult) -> NextStep:
    """기록(stages)만 보고 다음 단계를 결정한다. 순서 제어는 Python이 담당한다.

    skip 목록이 있으면 호출자가 먼저 SKIPPED로 기록한 뒤 stage를 실행한다.
    기록이 이상하면 예외 대신 STAGE_FAILED로 중단한다(fail-closed).
    """
    stages = result.stages
    if result.stopped_reason is not None:
        return NextStep(None, stop_reason=result.stopped_reason)
    if any(s.status not in (StageStatus.SUCCEEDED, StageStatus.SKIPPED) for s in stages):
        return NextStep(None, stop_reason=ErrorCode.STAGE_FAILED)
    if result.approval is not ApprovalStatus.DRAFT:
        return NextStep(None)

    broken = NextStep(None, stop_reason=ErrorCode.STAGE_FAILED)

    def recorded(stage, round_no):
        return any(s.stage is stage and s.round_no == round_no for s in stages)

    def find(stage, round_no):
        for s in stages:
            if s.stage is stage and s.round_no == round_no and s.status is StageStatus.SUCCEEDED:
                return s
        return None

    def finish(skips):
        pending = tuple(x for x in skips if not recorded(*x))
        stage = StageName.FINAL_APPROVAL if recorded(StageName.MANAGER, 0) else StageName.MANAGER
        return NextStep(stage, 0, skip=pending)

    for stage in (StageName.COLLECTOR, StageName.ANALYST, StageName.WRITER_DRAFT,
                  StageName.REVIEWER):
        if not recorded(stage, 0):
            return NextStep(stage, 0)
    first = find(StageName.REVIEWER, 0)
    if first is None:
        return broken

    verdict = first.verdict
    round_no = 1
    while True:
        if verdict not in (ReviewVerdict.APPROVABLE, ReviewVerdict.NEEDS_REVISION):
            return NextStep(None, stop_reason=ErrorCode.REVIEW_UNDETERMINED)
        if not recorded(StageName.HUMAN_REVIEW, round_no):      # Human Review는 항상 실행한다
            return NextStep(StageName.HUMAN_REVIEW, round_no)
        human = find(StageName.HUMAN_REVIEW, round_no)
        if human is None:
            return broken
        revise = (verdict is ReviewVerdict.NEEDS_REVISION
                  or human.human_decision is HumanDecision.REQUEST_REVISION)
        if not revise:
            # 사람이 수정 없이 승인하면 Writer 수정·Reviewer 재검수 슬롯만 SKIPPED로 남긴다.
            skips = tuple((stage, r) for r in range(round_no, MAX_REVISION_ROUNDS + 1)
                          for stage in (StageName.WRITER_REVISION, StageName.REVIEWER_RECHECK))
            return finish(skips)
        for stage in (StageName.WRITER_REVISION, StageName.REVIEWER_RECHECK):
            if not recorded(stage, round_no):
                return NextStep(stage, round_no)
        recheck = find(StageName.REVIEWER_RECHECK, round_no)
        if recheck is None:
            return broken
        verdict = recheck.verdict
        if round_no >= MAX_REVISION_ROUNDS:                      # 마지막 재검수 뒤에는 사람 검토 없이 종료
            if verdict is ReviewVerdict.APPROVABLE:
                return finish(())
            if verdict is ReviewVerdict.NEEDS_REVISION:
                return NextStep(None, stop_reason=ErrorCode.REVISION_LIMIT_EXCEEDED)
            return NextStep(None, stop_reason=ErrorCode.REVIEW_UNDETERMINED)
        round_no += 1


# ---- 호출 예산 · LIVE 게이트 ------------------------------------------------

@dataclass(frozen=True)
class PreflightDecision:
    allowed: bool
    error_code: ErrorCode | None = None

    @property
    def message(self) -> str | None:
        return ERROR_MESSAGES[self.error_code] if self.error_code else None


class CallBudget:
    """외부 요청 직전에 호출 횟수·입력 크기·예상 비용을 확보한다. 재시도도 별도 요청으로 센다."""

    def __init__(self, limits: RunLimits):
        self.limits = limits
        self.total = 0
        self.by_role = {"writer": 0, "reviewer": 0}
        self.by_stage = {StageName.COLLECTOR: 0, StageName.ANALYST: 0, StageName.MANAGER: 0}
        self.committed_cost = 0.0

    def reserve(self, stage: StageName, input_chars: int, est_cost_usd: float) -> PreflightDecision:
        lim = self.limits
        if stage not in MODEL_STAGES:
            return PreflightDecision(False, ErrorCode.PREFLIGHT_BLOCKED)
        max_chars, max_cost = lim.max_input_chars_per_call, lim.max_cost_usd
        if not lim.is_complete() or max_chars is None or max_cost is None:
            return PreflightDecision(False, ErrorCode.LIMITS_NOT_SET)
        if not _is_number(est_cost_usd) or not math.isfinite(est_cost_usd) or est_cost_usd < 0:
            return PreflightDecision(False, ErrorCode.BUDGET_EXCEEDED)
        if not _is_int(input_chars) or input_chars < 0:
            return PreflightDecision(False, ErrorCode.PREFLIGHT_BLOCKED)
        if input_chars > max_chars:
            return PreflightDecision(False, ErrorCode.INPUT_TOO_LARGE)
        if self.total >= lim.max_total_calls:
            return PreflightDecision(False, ErrorCode.CALL_LIMIT_EXCEEDED)
        if stage in WRITER_STAGES and self.by_role["writer"] >= lim.max_writer_calls:
            return PreflightDecision(False, ErrorCode.CALL_LIMIT_EXCEEDED)
        if stage in REVIEWER_STAGES and self.by_role["reviewer"] >= lim.max_reviewer_calls:
            return PreflightDecision(False, ErrorCode.CALL_LIMIT_EXCEEDED)
        if stage in self.by_stage and self.by_stage[stage] >= 1:
            return PreflightDecision(False, ErrorCode.CALL_LIMIT_EXCEEDED)
        if self.committed_cost + est_cost_usd > max_cost:
            return PreflightDecision(False, ErrorCode.BUDGET_EXCEEDED)
        self.total += 1
        if stage in WRITER_STAGES:
            self.by_role["writer"] += 1
        elif stage in REVIEWER_STAGES:
            self.by_role["reviewer"] += 1
        else:
            self.by_stage[stage] += 1
        self.committed_cost += est_cost_usd
        return PreflightDecision(True)


def check_live_allowed(result: PipelineResult, step: StageInput) -> PreflightDecision:
    """모델 요청 직전 게이트. 외부로 나갈 전체 입력(step)과 실행 상태를 검사한다.

    두 모드 모두 '진행 중인 실행의 다음 단계·회차'인 요청만 허용한다(중단·반려·승인 완료 실행은 차단).
    DRY_RUN은 가짜 모델만 쓰므로 그 외 승인 조건은 요구하지 않는다.
    """
    if step.stage not in MODEL_STAGES:
        return PreflightDecision(False, ErrorCode.PREFLIGHT_BLOCKED)
    expected = next_step(result)
    if (result.stopped_reason is not None or result.approval is not ApprovalStatus.DRAFT
            or expected.stage is None or (step.stage, step.round_no) != (expected.stage, expected.round_no)):
        return PreflightDecision(False, ErrorCode.PREFLIGHT_BLOCKED)
    if result.mode is RunMode.DRY_RUN:
        return PreflightDecision(True)
    if not result.tracing_disabled:
        return PreflightDecision(False, ErrorCode.TRACING_ENABLED)
    if not result.external_send_approved:
        return PreflightDecision(False, ErrorCode.EXTERNAL_SEND_NOT_APPROVED)
    if step.sanitized.source_kind is SourceKind.REAL and not result.real_data_send_approved:
        return PreflightDecision(False, ErrorCode.REAL_DATA_NOT_APPROVED)
    if not result.limits.is_complete():
        return PreflightDecision(False, ErrorCode.LIMITS_NOT_SET)
    if step.sanitized.dataset_id not in {p.dataset for p in result.provenance}:
        return PreflightDecision(False, ErrorCode.DATA_NOT_APPROVED)
    if step.sanitized.source_kind is SourceKind.REAL and step.human_notes is not None:
        # 실제 자료 기반 실행에서는 사람의 자유 메모를 외부로 보내지 않는다(구조화 수정 요청은 후속 설계).
        return PreflightDecision(False, ErrorCode.FREE_TEXT_REJECTED)
    if not all(t.pii_scan_passed for t in step.texts()):
        return PreflightDecision(False, ErrorCode.TEXT_NOT_SCANNED)
    # prior_outputs의 본문은 같은 실행에서 성공한 해당 모델 단계의 기록된 산출물과 일치해야 한다.
    recorded = {}
    for s in result.stages:
        if s.status is StageStatus.SUCCEEDED and s.output_text is not None:
            recorded.setdefault(s.stage, set()).add(s.output_text)
    for stage, checked in step.prior_outputs:
        if checked.text not in recorded.get(stage, ()):
            return PreflightDecision(False, ErrorCode.FREE_TEXT_REJECTED)
    return PreflightDecision(True)


class ModelRunner(Protocol):
    """모델 호출 추상화. 4단계에서 가짜(dry-run)와 실제(Agents SDK) 구현을 만든다."""

    def run_stage(self, step: StageInput) -> StageResult: ...
