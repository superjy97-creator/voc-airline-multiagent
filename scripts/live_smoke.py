"""LIVE(실제 API) 점검 스크립트 — 합성 데이터 전용. 실제 VOC 데이터를 보내는 경로는 이 스크립트에 없다.

단계(--phase):
  plan   (기본) 네트워크 없이 승인 요청서만 출력
  start  Collector→Analyst→Writer→Reviewer까지 유료 호출 후 Human Review 직전에 멈추고 상태를 저장
  resume API 호출 실패로 멈춘 실행을 저장 상태에서 1회만 이어서 실행(성공한 단계 재사용, 호출 수·비용 장부 유지)
  human  저장된 상태에서 사람의 Human Review 결정(--decision as_is|revise [--notes ...])을 반영해 이어서 실행
  final  사람의 최종 결정(--decision approve|reject) 반영(호출 없음)
  status 저장된 상태 요약(호출 없음)
start/resume/human은 --approve-paid-calls --price-in --price-out --max-cost-usd 가 모두 있어야 한다.
resume은 이어하기 직전에 상태 파일을 타임스탬프 백업하고, 1회만 허용한다(같은 오류 자동 반복 금지).
start는 상태 파일이 이미 있으면 거부한다(유료 테스트 자동 반복 방지).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voc.config import TOKENS_PER_CHAR_UPPER, VOC_AGENT_MODEL
from voc.contracts import (
    CallBudget,
    ErrorCode,
    HumanDecision,
    RunLimits,
    RunMode,
    SourceKind,
    StageName,
    StageStatus,
    next_step,
)
from voc.live_core import (
    DEFAULT_COST_CAP,
    MAX_CALLS,
    PAYLOAD_OVERHEAD_CHARS,  # noqa: F401 - 호환용 재공개
    LoggingRunner,
    _restore_budget,
    _stage_from_dict,
    describe_error,  # noqa: F401 - 테스트·호환용 재공개
    estimate,
    load_state,
    save_state,
    stage_cost,
    usd,
)
from voc.pipeline import DRY_RUN_LIMITS, HumanInput, RunContext, advance, new_run
from voc.runners import MAX_TURNS, SdkRunner
from voc.sanitize import make_checked_text
from voc.stats import synthetic_stats, to_provenance, to_sanitized

STATE_PATH = ROOT / "reports" / "live_state.json"
REPORT_PATH = ROOT / "reports" / "live_report.md"
OPENAI_URL = "https://api.openai.com/v1"


# ---- 추정·요청서 -------------------------------------------------------------



def format_plan(est: dict, prices: tuple[float, float] | None) -> str:
    lines = [
        "=== LIVE 실행 승인 요청서 (합성 데이터 전용) ===",
        f"모델: {VOC_AGENT_MODEL} (Collector·Analyst·Writer·Reviewer·Manager 공통)",
        f"최대 호출 횟수: {MAX_CALLS}회 (단계당 요청 1회, HTTP 재시도 0회, 추적 꺼짐)",
        "전송 데이터: 합성 집계값만(품목·자료유형·기간·건수·비율) + 이전 단계 모델 출력 + 사람 메모(개인정보 검사 통과분)",
        "전송 금지: 실제 상담 원문·제목·개인정보·실제 VOC 집계(이 스크립트에는 실제 자료 경로가 없음)",
        f"입력 토큰(현실 추정, 9회 합계): 약 {est['input_tokens_realistic']:,} (글자당 {TOKENS_PER_CHAR_UPPER}토큰 상한 가정)",
        f"입력 토큰(상한, 9회 합계): 최대 {est['input_tokens_ceiling']:,}",
        f"출력 토큰 상한: 호출당 {est['max_output_tokens']:,} / 9회 합계 최대 {est['output_tokens_ceiling']:,}",
    ]
    if prices:
        p_in, p_out = prices
        low = usd(est["input_tokens_realistic"], p_in) + usd(est["output_tokens_ceiling"], p_out)
        high = usd(est["input_tokens_ceiling"], p_in) + usd(est["output_tokens_ceiling"], p_out)
        lines.append(f"예상 비용(입력 {p_in}, 출력 {p_out} USD/100만 토큰): "
                     f"현실 추정 최대 약 ${low:.4f} / 상한 ${high:.4f} / 호출당 예약 ${stage_cost(est, prices):.5f}")
    else:
        lines.append("예상 비용: 단가가 없어 계산하지 않음. 비용 = 입력토큰×입력단가 + 출력토큰×출력단가")
    return "\n".join(lines)


# ---- 상태 저장·복원 ----------------------------------------------------------



def resume_state(path: Path, stats, limits: RunLimits):
    """API 실패로 멈춘 실행을 이어하기 위해 복원한다(1회만, 실패 요청의 호출 수·비용은 장부에 그대로 둔다).

    마지막 실패 단계 기록만 이력에서 빼고(failed_attempts에 보존), 호출 횟수·누적 비용·오류 기록은 유지한다.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("resumes", 0) >= 1:
        raise SystemExit("이미 한 번 이어서 실행했습니다. 같은 오류가 반복될 수 있어 자동 재시도하지 않습니다. "
                         "원인 보고와 남은 비용 재승인이 필요합니다.")
    stages = [_stage_from_dict(raw) for raw in data["stages"]]
    last = stages[-1] if stages else None
    if last is None or last.status is not StageStatus.FAILED or last.error_code is not ErrorCode.API_ERROR:
        raise SystemExit("이어서 실행할 수 없는 상태입니다(마지막 단계가 API 호출 실패여야 합니다).")
    result = new_run(data["run_id"], (to_provenance(stats),), RunMode.LIVE, limits, external_send_approved=True)
    for stage in stages[:-1]:
        result = result.with_stage(stage)
    failed = data.get("failed_attempts", []) + [data["stages"][-1]]
    return result, _restore_budget(data, limits), data["usage"], data["errors"], failed


# ---- 실행 보조 ---------------------------------------------------------------



def preflight(args, est, limits, client, runner, result, sanitized,
              budget: CallBudget | None = None) -> list[tuple[str, bool]]:
    prices = (args.price_in, args.price_out)
    worst = MAX_CALLS * stage_cost(est, prices)
    checks = [
        ("OpenAI 클라이언트 max_retries=0", client.max_retries == 0),
        (f"max_turns={MAX_TURNS} (단계당 요청 1회)", MAX_TURNS == 1),
        ("호출 상한 총 9회·Writer 3·Reviewer 3",
         (limits.max_total_calls, limits.max_writer_calls, limits.max_reviewer_calls) == (9, 3, 3)),
        (f"출력 토큰 상한 {args.max_output_tokens} 요청에 적용",
         runner.inner.max_output_tokens == limits.max_output_tokens_per_call == args.max_output_tokens),
        ("추적 꺼짐 · 외부 전송 승인(합성 한정) · 실자료 전송 미승인",
         result.tracing_disabled and result.external_send_approved and not result.real_data_send_approved),
        ("합성 데이터 전용", sanitized.source_kind is SourceKind.SYNTHETIC),
        (f"9회 최악 비용 ${worst:.5f} ≤ 상한 ${args.max_cost_usd}", worst <= args.max_cost_usd),
        (f"9회 입력·출력 상한 비용 ${usd(est['input_tokens_ceiling'], prices[0]) + usd(est['output_tokens_ceiling'], prices[1]):.5f} ≤ 상한",
         usd(est["input_tokens_ceiling"], prices[0]) + usd(est["output_tokens_ceiling"], prices[1])
         <= args.max_cost_usd),
    ]
    if budget is not None:   # 이어하기: 기존 장부(실패 요청 포함) + 남은 호출 최악 비용이 상한 이내인지
        remaining = limits.max_total_calls - budget.total
        ledger, future = runner.spent(), remaining * stage_cost(est, prices)
        checks.append((f"기존 장부 ${ledger:.5f} + 남은 {remaining}회 최악 ${future:.5f} ≤ 상한 ${args.max_cost_usd}",
                       remaining >= 0 and ledger + future <= args.max_cost_usd))
    return checks


def show(result, usage: list, errors: list, prices: tuple[float, float] | None) -> None:
    print("\n--- 진행 상황 ---")
    for s in result.stages:
        tag = s.verdict or s.human_decision
        print(f"  {s.stage.value:17s} 회차{s.round_no} {s.status.value:9s} "
              f"{(tag.value if tag else ''):15s} 호출 {s.calls_used} 출력토큰 {s.output_tokens}")

    def latest(*names):
        return next((s.output_text for s in reversed(result.stages)
                     if s.stage in names and s.status is StageStatus.SUCCEEDED and s.output_text), None)

    step = next_step(result)
    if step.stage is StageName.HUMAN_REVIEW:
        print("\n[Writer 보고서 초안]\n" + (latest(StageName.WRITER_REVISION, StageName.WRITER_DRAFT) or ""))
        print("\n[Reviewer 검수 결과]\n" + (latest(StageName.REVIEWER, StageName.REVIEWER_RECHECK) or ""))
        print("\n>>> 사용자의 Human Review 결정이 필요합니다(수정 없이 승인 / 수정 요청).")
    elif step.stage is StageName.FINAL_APPROVAL:
        print("\n[Manager 종합]\n" + (latest(StageName.MANAGER) or ""))
        print("\n>>> 사용자의 최종 승인/반려가 필요합니다.")
    elif step.stop_reason is not None:
        print(f"\n중단: {step.stop_reason.value}")
    in_tokens = sum(u["input_tokens"] for u in usage)
    out_tokens = sum(u["output_tokens"] for u in usage)
    cost = f"${usd(in_tokens, prices[0]) + usd(out_tokens, prices[1]):.6f}" if prices else "단가 미지정"
    print(f"\n--- 사용량 --- 모델 요청 {sum(u['requests'] for u in usage)}회 | 입력 {in_tokens:,} / 출력 {out_tokens:,} 토큰"
          f" | 실제 비용 {cost}")
    for line in errors:
        print("기록된 오류:", line)


def limits_for(args) -> RunLimits:
    return RunLimits(max_input_chars_per_call=DRY_RUN_LIMITS.max_input_chars_per_call,
                     max_output_tokens_per_call=args.max_output_tokens,
                     max_cost_usd=args.max_cost_usd or DEFAULT_COST_CAP)


def require_approval(args) -> None:
    missing = [name for name, value in (("--approve-paid-calls", args.approve_paid_calls),
                                        ("--price-in", args.price_in), ("--price-out", args.price_out),
                                        ("--max-cost-usd", args.max_cost_usd)) if not value]
    if missing:
        raise SystemExit(f"승인·비용 조건이 부족해 실행하지 않습니다: {', '.join(missing)}")


def run_phase(args, stats) -> None:
    prices = (args.price_in, args.price_out) if args.price_in and args.price_out else None
    limits = limits_for(args)
    if args.phase in ("final", "status"):
        if not STATE_PATH.exists():
            raise SystemExit("저장된 실행 상태가 없습니다.")
        result, budget, usage, errors = load_state(STATE_PATH, stats, limits)
        if args.phase == "final":
            if next_step(result).stage is not StageName.FINAL_APPROVAL:
                raise SystemExit("최종 승인 단계가 아닙니다.")
            if args.decision not in ("approve", "reject"):
                raise SystemExit("--decision approve|reject 가 필요합니다(사용자 결정).")
            result = result.approve_final() if args.decision == "approve" else result.reject()
            if args.decision == "approve":
                report = next(s.output_text for s in reversed(result.stages)
                              if s.stage in (StageName.WRITER_REVISION, StageName.WRITER_DRAFT)
                              and s.status is StageStatus.SUCCEEDED)
                REPORT_PATH.write_text(
                    f"<!-- LIVE 합성 테스트({VOC_AGENT_MODEL}) · 합성 데이터 · 실제 분석 결과 아님 -->\n\n{report}\n",
                    encoding="utf-8")
            save_state(STATE_PATH, result, budget, usage, errors)
        show(result, usage, errors, prices)
        return

    require_approval(args)
    if args.phase == "start" and STATE_PATH.exists():
        raise SystemExit("이미 실행 기록(reports/live_state.json)이 있어 시작하지 않습니다. "
                         "재실행은 기존 비용을 포함한 예산 재승인 후 사용자가 상태 파일을 정리해야 합니다.")
    from dotenv import dotenv_values
    from openai import AsyncOpenAI

    key = (dotenv_values(ROOT.parent / ".env").get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY가 없어 중단합니다.")
    assert prices is not None
    sanitized = to_sanitized(stats, SourceKind.SYNTHETIC)
    est = estimate(stats, args.max_output_tokens)
    client = AsyncOpenAI(api_key=key, base_url=OPENAI_URL, max_retries=0, timeout=90)
    inner = SdkRunner(VOC_AGENT_MODEL, approved_live=True, max_output_tokens=args.max_output_tokens,
                      client=client)
    extra: dict | None = None

    if args.phase == "start":
        result = new_run("live-smoke", (to_provenance(stats),), RunMode.LIVE, limits,
                         external_send_approved=True)
        budget, usage, errors = CallBudget(limits), [], []
    elif args.phase == "resume":
        result, budget, usage, errors, failed = resume_state(STATE_PATH, stats, limits)
        extra = {"resumes": 1, "failed_attempts": failed}
    else:
        result, budget, usage, errors = load_state(STATE_PATH, stats, limits)
        if next_step(result).stage is not StageName.HUMAN_REVIEW:
            raise SystemExit("Human Review 단계가 아닙니다.")

    runner = LoggingRunner(inner, prices, args.max_cost_usd, stage_cost(est, prices), usage, errors,
                           fail_tokens=(est["max_stage_input_tokens"], args.max_output_tokens))
    checks = preflight(args, est, limits, client, runner, result, sanitized, budget)
    print("--- 사전 점검 ---")
    for label, passed in checks:
        print(f"  [{'OK' if passed else '실패'}] {label}")
    if not all(passed for _, passed in checks):
        raise SystemExit("사전 점검 실패 — 호출하지 않았습니다.")

    ctx = RunContext(sanitized, runner, budget, make_checked_text, est_cost_usd=stage_cost(est, prices))
    human = None
    if args.phase == "human":
        if args.decision not in ("as_is", "revise"):
            raise SystemExit("--decision as_is|revise 가 필요합니다(사용자 결정).")
        human = HumanInput(HumanDecision.APPROVE_AS_IS if args.decision == "as_is"
                           else HumanDecision.REQUEST_REVISION, args.notes)
    if human is not None and human.notes and not make_checked_text(human.notes).pii_scan_passed:
        raise SystemExit("개인정보 의심 패턴이 있어 입력을 거부했습니다. 호출하지 않았습니다.")
    if args.phase == "resume":   # 기존 호출 기록은 지우지 않고 백업을 따로 남긴다
        shutil.copy2(STATE_PATH, STATE_PATH.with_name(f"live_state.before_resume_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    # 첫 유료 호출 전에 상태 파일을 먼저 써서 잠근다: 이후 비정상 종료가 나도 start·resume 재실행이 막힌다.
    save_state(STATE_PATH, result, budget, runner.usage_log, errors, extra)
    try:
        result = advance(result, ctx, human)
    except BaseException:
        errors.append("예상치 못한 중단 — 상태·비용 기록은 보존하고 실행을 종료함")
        save_state(STATE_PATH, result.stop(ErrorCode.STAGE_FAILED), budget, runner.usage_log, errors)
        raise
    save_state(STATE_PATH, result, budget, runner.usage_log, errors)
    show(result, runner.usage_log, errors, prices)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="LIVE(실제 API) 점검 — 합성 데이터 전용")
    parser.add_argument("--phase", choices=["plan", "start", "resume", "human", "final", "status"],
                        default="plan")
    parser.add_argument("--approve-paid-calls", action="store_true")
    parser.add_argument("--price-in", type=float)
    parser.add_argument("--price-out", type=float)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--max-output-tokens", type=int, default=DRY_RUN_LIMITS.max_output_tokens_per_call)
    parser.add_argument("--decision", help="human: as_is|revise, final: approve|reject (사용자 결정만 전달)")
    parser.add_argument("--notes", help="human revise 시 사용자가 직접 작성한 수정 메모")
    args = parser.parse_args(argv)
    stats = synthetic_stats()[0]
    if args.phase == "plan":
        prices = (args.price_in, args.price_out) if args.price_in and args.price_out else None
        print(format_plan(estimate(stats, args.max_output_tokens), prices))
        print("\n(요청서만 출력했습니다. 실제 호출은 하지 않았습니다.)")
        return
    run_phase(args, stats)


if __name__ == "__main__":
    main()
