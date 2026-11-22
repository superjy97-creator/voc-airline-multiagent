"""LIVE 실행 공통 부품(비용 추정·상태 저장·호출 래퍼). 스크립트와 웹 LIVE 서비스가 함께 쓴다.

네트워크·비밀키를 직접 다루지 않는다. 키는 호출하는 쪽이 만든 client로만 전달된다.
"""

import json
import re
import sys
from dataclasses import fields
from enum import Enum
from pathlib import Path

from voc.config import TOKENS_PER_CHAR_UPPER
from voc.contracts import (
    CallBudget,
    ErrorCode,
    HumanDecision,
    ReviewVerdict,
    RunLimits,
    RunMode,
    SkipReason,
    SourceKind,
    StageName,
    StageResult,
    StageStatus,
)
from voc.pipeline import DRY_RUN_LIMITS, HumanInput, RunContext, advance, new_run
from voc.prompts import build_instructions
from voc.runners import FakeRunner, SdkRunner
from voc.sanitize import make_checked_text
from voc.stats import to_provenance, to_sanitized

PAYLOAD_OVERHEAD_CHARS = 1000  # JSON 키·따옴표·구분자 등 payload 포장 여유분(필드 75개 기준 약 500자 + 여유)
MAX_CALLS = 9
DEFAULT_COST_CAP = 0.05


def worst_case_stages(stats, sanitize=None, input_cap: int | None = None) -> list[tuple[StageName, int]]:
    """수정 2회 경로(호출 9회)를 가짜 모델로 돌려 단계별 실제 payload 글자 수를 잰다(네트워크 없음).

    sanitize(stats)로 다른 입력 스키마를, input_cap으로 다른 입력 글자 상한을 쓸 수 있다(기본: 합성 입력·4,000자).
    """
    sanitized = sanitize(stats) if sanitize else to_sanitized(stats, SourceKind.SYNTHETIC)
    limits = DRY_RUN_LIMITS if input_cap is None else RunLimits(
        max_input_chars_per_call=input_cap, max_output_tokens_per_call=1000, max_cost_usd=1.0)
    ctx = RunContext(sanitized,
                     FakeRunner((ReviewVerdict.NEEDS_REVISION, ReviewVerdict.NEEDS_REVISION,
                                 ReviewVerdict.APPROVABLE)),
                     CallBudget(limits), make_checked_text)
    result = advance(new_run("plan", (to_provenance(stats),)), ctx)
    for _ in range(2):
        result = advance(result, ctx, HumanInput(HumanDecision.REQUEST_REVISION, "근거를 보강해 주세요."))
    return [(s.stage, s.input_chars) for s in result.stages if s.calls_used]


def estimate(stats, max_output_tokens: int, sanitize=None, input_cap: int | None = None) -> dict:
    calls = worst_case_stages(stats, sanitize, input_cap)
    cap = input_cap if input_cap is not None else (DRY_RUN_LIMITS.max_input_chars_per_call or 0)
    realistic = [c + len(build_instructions(stage)) + PAYLOAD_OVERHEAD_CHARS for stage, c in calls]
    ceiling = [cap + len(build_instructions(stage)) + PAYLOAD_OVERHEAD_CHARS for stage, _ in calls]
    return {
        "calls": len(calls),
        "input_tokens_realistic": sum(realistic) * TOKENS_PER_CHAR_UPPER,
        "input_tokens_ceiling": sum(ceiling) * TOKENS_PER_CHAR_UPPER,
        "max_stage_input_tokens": max(ceiling) * TOKENS_PER_CHAR_UPPER,
        "output_tokens_ceiling": len(calls) * max_output_tokens,
        "max_output_tokens": max_output_tokens,
    }


def usd(tokens: int, price_per_million: float) -> float:
    return tokens * price_per_million / 1_000_000


def stage_cost(est: dict, prices: tuple[float, float]) -> float:
    """한 호출의 최악 비용(입력 상한 + 출력 상한). CallBudget이 호출 전에 이 값을 예약한다."""
    return usd(est["max_stage_input_tokens"], prices[0]) + usd(est["max_output_tokens"], prices[1])


def _plain(value):
    return value.value if isinstance(value, Enum) else value


def _stage_dict(stage) -> dict:
    return {f.name: _plain(getattr(stage, f.name)) for f in fields(stage)}


def save_state(path: Path, result, budget: CallBudget, usage: list, errors: list,
               extra: dict | None = None) -> None:
    """상태를 저장한다. 이전 파일의 이어하기 횟수·실패 기록은 항상 이어받고(extra로만 갱신) 지우지 않는다."""
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = {
        "run_id": result.run_id, "stopped_reason": _plain(result.stopped_reason),
        "stages": [_stage_dict(s) for s in result.stages],
        "budget": {"total": budget.total, "by_role": budget.by_role,
                   "by_stage": {k.value: v for k, v in budget.by_stage.items()},
                   "committed_cost": budget.committed_cost},
        "usage": usage, "errors": errors,
        "resumes": old.get("resumes", 0), "failed_attempts": old.get("failed_attempts", []),
    }
    data.update(extra or {})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _stage_from_dict(raw: dict) -> StageResult:
    def enum(cls, value):
        return cls(value) if value is not None else None

    return StageResult(
        StageName(raw["stage"]), StageStatus(raw["status"]), round_no=raw["round_no"],
        artifact_version=raw["artifact_version"], output_text=raw["output_text"],
        verdict=enum(ReviewVerdict, raw["verdict"]), human_decision=enum(HumanDecision, raw["human_decision"]),
        error_code=enum(ErrorCode, raw["error_code"]), skip_reason=enum(SkipReason, raw["skip_reason"]),
        calls_used=raw["calls_used"], input_chars=raw["input_chars"],
        output_tokens=raw["output_tokens"], cost_usd=raw["cost_usd"],
    )


def load_state(path: Path, stats, limits: RunLimits, *, real_data: bool = False):
    """저장된 상태를 복원한다. 기록은 with_stage로 다시 쌓아 순서·버전을 재검증한다.

    real_data=True는 실제 집계 실행의 복원이며, 이 경우에만 실자료 전송 승인 상태로 되살린다.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    result = new_run(data["run_id"], (to_provenance(stats),), RunMode.LIVE, limits,
                     external_send_approved=True, real_data_send_approved=real_data)
    for raw in data["stages"]:
        result = result.with_stage(_stage_from_dict(raw))
    if data["stopped_reason"]:
        result = result.stop(ErrorCode(data["stopped_reason"]))
    return result, _restore_budget(data, limits), data["usage"], data["errors"]


def _restore_budget(data: dict, limits: RunLimits) -> CallBudget:
    budget = CallBudget(limits)
    saved = data["budget"]
    budget.total, budget.committed_cost = saved["total"], saved["committed_cost"]
    budget.by_role.update(saved["by_role"])
    budget.by_stage.update({StageName(k): v for k, v in saved["by_stage"].items()})
    return budget


def describe_error(error: Exception) -> str:
    """원인 진단용 요약. 키 일부가 섞일 수 있는 메시지는 가리고 길이를 제한한다."""
    body = getattr(error, "body", None)
    body = body if isinstance(body, dict) else {}
    message = re.sub(r"sk-[A-Za-z0-9_\-*.]+", "sk-***", str(error))[:300]
    return (f"{type(error).__name__} | HTTP {getattr(error, 'status_code', None)} | "
            f"code={body.get('code')} param={body.get('param')} | {message}")


class LoggingRunner:
    """SdkRunner 위임 + 호출 전 누적 비용 점검 + 실패 원인 기록. 보안·비용 검사를 우회하지 않는다."""

    def __init__(self, inner: SdkRunner, prices: tuple[float, float], max_cost: float,
                 per_call_cost: float, prior_usage: list, errors: list,
                 fail_tokens: tuple[int, int] = (0, 0)) -> None:
        self.inner, self.prices, self.max_cost = inner, prices, max_cost
        self.per_call_cost, self.prior_usage, self.errors = per_call_cost, prior_usage, errors
        self.fail_tokens = fail_tokens          # 실패한 요청은 과금됐을 수 있어 최악 토큰으로 계산한다
        self.failed_usage: list = []

    @property
    def usage_log(self) -> list:
        return self.prior_usage + self.inner.usage_log + self.failed_usage

    def spent(self) -> float:
        return sum(usd(u["input_tokens"], self.prices[0]) + usd(u["output_tokens"], self.prices[1])
                   for u in self.usage_log)

    def run_stage(self, step):
        if self.spent() + self.per_call_cost > self.max_cost:
            raise RuntimeError("실제 누적 비용 + 다음 호출 최악 비용이 상한을 넘어 중단합니다.")
        try:
            result = self.inner.run_stage(step)
        except Exception as error:
            self.failed_usage.append({"stage": step.stage.value, "requests": 1, "estimated": True,
                                      "input_tokens": self.fail_tokens[0], "output_tokens": self.fail_tokens[1]})
            self.errors.append(f"{step.stage.value}: {describe_error(error)} (과금 가능성: 최악 비용으로 계산)")
            print("호출 실패 원인:", self.errors[-1], file=sys.stderr)
            raise
        if self.inner.usage_log[-1]["requests"] != 1:
            self.errors.append(f"{step.stage.value}: 요청 수가 1이 아님({self.inner.usage_log[-1]['requests']})")
        return result
