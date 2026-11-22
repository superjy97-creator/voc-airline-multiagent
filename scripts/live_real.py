"""LIVE(실제 API) — 실제 VOC '집계 통계'만 전송하는 별도 실행. 상담 원문·제목·개인정보를 보내는 경로는 없다.

기존 합성 실행(scripts/live_smoke.py, reports/live_state.json)과 완전히 분리해 관리한다(상태·보고서 파일이 다르다).
  plan   (기본) 네트워크 없이 전송 내용 전체·사전 검사·예상 비용을 출력하고 승인 코드를 안내
  start  승인 코드가 전송 내용과 일치할 때만 Collector→Analyst→Writer→Reviewer 호출 후 Human Review 직전에 멈춤
  human  사람의 Human Review 결정(--decision as_is|revise)을 반영해 이어서 실행(사람 메모는 전송할 수 없다)
  final  사람의 최종 결정(--decision approve|reject) 반영(호출 없음)
  status 저장 상태 요약(호출 없음)
start/human은 --approve-paid-calls --approve-real-data-send <코드> --price-in --price-out --max-cost-usd 가 필요하다.
start는 상태 파일이 이미 있으면 거부한다(유료 실행 자동 반복 방지).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import live_smoke as ls
from voc.config import VOC_AGENT_MODEL
from voc.contracts import (
    CallBudget,
    HumanDecision,
    RunLimits,
    RunMode,
    SanitizedInput,
    SourceKind,
    StageName,
    next_step,
)
from voc.pipeline import HumanInput, RunContext, advance, new_run
from voc.runners import MAX_TURNS, SdkRunner
from voc.sanitize import make_checked_text, scan_text
from voc.stats import DatasetStats, load_summary, to_provenance, to_sanitized_monthly

SUMMARY_PATH = ROOT / "reports" / "stats_summary.json"
REAL_STATE_PATH = ROOT / "reports" / "live_real_state.json"
REAL_REPORT_PATH = ROOT / "reports" / "live_real_report.md"
DEFAULT_OUTPUT_CAP = 2500
DEFAULT_INPUT_CAP = 10_000
RUN_ID = "live-real"
# 사용자가 승인한 범위. 명령줄 값으로는 이 범위를 넓힐 수 없다(상한은 낮출 수만, 단가는 낮게 줄 수 없다).
APPROVED_MAX_COST = 0.05
APPROVED_MAX_INPUT_CHARS = 10_000
APPROVED_MAX_OUTPUT_TOKENS = 2500
APPROVED_PRICES = (0.10, 0.50)   # 입력·출력 USD/100만 토큰(gpt-6-luna, 사용자 제공)


def enforce_approved_limits(args) -> None:
    problems = []
    if args.max_cost_usd is not None and args.max_cost_usd > APPROVED_MAX_COST:
        problems.append(f"비용 상한은 ${APPROVED_MAX_COST} 이하여야 함")
    if args.max_input_chars > APPROVED_MAX_INPUT_CHARS:
        problems.append(f"입력 글자 상한은 {APPROVED_MAX_INPUT_CHARS:,}자 이하여야 함")
    if args.max_output_tokens > APPROVED_MAX_OUTPUT_TOKENS:
        problems.append(f"출력 토큰 상한은 {APPROVED_MAX_OUTPUT_TOKENS:,} 이하여야 함")
    if args.price_in is not None and args.price_in < APPROVED_PRICES[0]:
        problems.append(f"입력 단가는 {APPROVED_PRICES[0]} 이상으로만 지정할 수 있음(과소계상 방지)")
    if args.price_out is not None and args.price_out < APPROVED_PRICES[1]:
        problems.append(f"출력 단가는 {APPROVED_PRICES[1]} 이상으로만 지정할 수 있음(과소계상 방지)")
    if problems:
        raise SystemExit("승인 범위를 벗어난 조건입니다: " + "; ".join(problems))


# ---- 전송 내용 검사·요약 -----------------------------------------------------

def pick_stats(dataset: str) -> DatasetStats:
    for stats in load_summary(SUMMARY_PATH):
        if stats.dataset_id == dataset:
            return stats
    raise SystemExit(f"집계 파일에 없는 자료입니다: {dataset}")


def canonical_payload(sanitized: SanitizedInput) -> dict:
    """모델 첫 입력(Collector)에 들어가는 자료 부분과 동일한 구조."""
    return {"source_kind": sanitized.source_kind.value, "dataset_id": sanitized.dataset_id,
            "schema_version": sanitized.schema_version,
            "fields": {name: value for name, value in sanitized.fields}}


def payload_fingerprint(sanitized: SanitizedInput) -> str:
    """전송 내용의 지문. 사용자가 본 내용과 실제 전송 내용이 같을 때만 승인되도록 한다."""
    text = json.dumps(canonical_payload(sanitized), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def pii_findings(sanitized: SanitizedInput) -> tuple[str, ...]:
    return scan_text(json.dumps(canonical_payload(sanitized), ensure_ascii=False))


def data_checks(sanitized: SanitizedInput) -> list[tuple[str, bool | None, str]]:
    """(항목, 통과 여부, 설명). 통과 여부 None은 자동으로 확인할 수 없는 항목이다."""
    findings = pii_findings(sanitized)
    monthly = [v for n, v in sanitized.fields if n.startswith("건수_")]
    return [
        ("읽은 열", True, "접수일·품목명(또는 코드)·사건번호 등 구조화 열만 집계(상담 원문·제목 열은 읽지 않음; loader 허용 목록과 테스트로 검증)"),
        ("전송 필드 형식", True, f"스키마 {sanitized.schema_version}의 열거값·숫자·기간 {len(sanitized.fields)}개뿐(자유 문자열 불가)"),
        ("개인정보 패턴 검사", not findings, "탐지 0건" if not findings else f"탐지: {', '.join(findings)}"),
        ("소규모 셀 확인", True, f"월별 건수 최솟값 {min(monthly)}, 최댓값 {max(monthly)} (집계값만, 개인 식별 정보 없음)"),
        ("자료 이용 조건", None, "원본 CSV의 이용허락·외부 전송 조건은 파일에서 확인할 수 없음 → 사용자 확인 필요"),
    ]


def format_payload(sanitized: SanitizedInput) -> list[str]:
    head = [f"  {name} = {value}" for name, value in sanitized.fields if not name.startswith("건수_")]
    months = [(name[3:], value) for name, value in sanitized.fields if name.startswith("건수_")]
    return head + [f"  월별 항공 건수({len(months)}개월): " + ", ".join(f"{m}={v}" for m, v in months)]


def format_plan(stats: DatasetStats, args) -> str:
    sanitized = to_sanitized_monthly(stats, SourceKind.REAL)
    est = ls.estimate(stats, args.max_output_tokens, lambda s: to_sanitized_monthly(s, SourceKind.REAL),
                      args.max_input_chars)
    fingerprint = payload_fingerprint(sanitized)
    prices = (args.price_in, args.price_out) if args.price_in and args.price_out else None
    lines = [
        "=== LIVE 실행 승인 요청서 (실제 집계 통계 전송) ===",
        f"자료: {stats.label} ({stats.dataset_id}) · 항공여객운송서비스 정확 일치 · 분모 {stats.total_rows:,} / 항공 {stats.air_rows:,}",
        f"모델: {VOC_AGENT_MODEL} (Collector·Analyst·Writer·Reviewer·Manager 공통)",
        f"최대 호출 횟수: {ls.MAX_CALLS}회 (단계당 요청 1회, HTTP 재시도 0회, 추적 꺼짐)",
        f"입력 글자 상한: 호출당 {args.max_input_chars:,}자 / 출력 토큰 상한: 호출당 {args.max_output_tokens:,}",
        "", "[전송할 자료 — 모델 첫 입력에 들어가는 내용 전부]", *format_payload(sanitized),
        "  (이후 단계에는 앞 단계 모델 출력이 개인정보 검사를 거쳐 함께 전달됨. 사람 메모는 전송 불가)",
        "", "[전송 금지·미전송] 상담 원문, 상담 제목, 개인정보, 사건번호, 다른 자료의 집계(자료별로 분리)",
        "", "[사전 검사]",
    ]
    for label, ok, detail in data_checks(sanitized):
        mark = "확인 필요" if ok is None else ("통과" if ok else "실패")
        lines.append(f"  [{mark}] {label}: {detail}")
    lines += ["", "[토큰·비용]",
              f"  입력 토큰(현실 추정, 9회 합계): 약 {est['input_tokens_realistic']:,} / 상한 {est['input_tokens_ceiling']:,}",
              f"  출력 토큰: 호출당 최대 {est['max_output_tokens']:,} / 9회 합계 최대 {est['output_tokens_ceiling']:,}"]
    if prices:
        low = ls.usd(est["input_tokens_realistic"], prices[0]) + ls.usd(est["output_tokens_ceiling"], prices[1])
        high = ls.usd(est["input_tokens_ceiling"], prices[0]) + ls.usd(est["output_tokens_ceiling"], prices[1])
        lines.append(f"  예상 비용(입력 {prices[0]}, 출력 {prices[1]} USD/100만 토큰): 현실 추정 최대 약 ${low:.4f} / "
                     f"상한 ${high:.4f} / 호출당 예약 ${ls.stage_cost(est, prices):.5f}")
    else:
        lines.append("  예상 비용: 단가를 주면(--price-in/--price-out) 계산")
    command = ("python scripts\\live_real.py --phase start "
               f"--dataset {stats.dataset_id} --approve-paid-calls --approve-real-data-send {fingerprint} "
               f"--price-in {args.price_in or '<입력단가>'} --price-out {args.price_out or '<출력단가>'} "
               f"--max-cost-usd {args.max_cost_usd or 0.05} --max-output-tokens {args.max_output_tokens} "
               f"--max-input-chars {args.max_input_chars}")
    lines += ["", f"전송 내용 승인 코드: {fingerprint}", f"승인 명령(승인하신 경우에만 실행): {command}"]
    return "\n".join(lines)


# ---- 실행 --------------------------------------------------------------------

def real_preflight(args, est, limits, client, runner, result, sanitized, budget) -> list[tuple[str, bool]]:
    prices = (args.price_in, args.price_out)
    per_call = ls.stage_cost(est, prices)
    remaining = limits.max_total_calls - budget.total
    ledger = runner.spent()
    ceiling = ls.usd(est["input_tokens_ceiling"], prices[0]) + ls.usd(est["output_tokens_ceiling"], prices[1])
    return [
        ("OpenAI 클라이언트 max_retries=0", client.max_retries == 0),
        (f"max_turns={MAX_TURNS} (단계당 요청 1회)", MAX_TURNS == 1),
        ("호출 상한 총 9회·Writer 3·Reviewer 3",
         (limits.max_total_calls, limits.max_writer_calls, limits.max_reviewer_calls) == (9, 3, 3)),
        (f"출력 토큰 상한 {args.max_output_tokens} 요청에 적용",
         runner.inner.max_output_tokens == limits.max_output_tokens_per_call == args.max_output_tokens),
        (f"직렬화한 입력 {args.max_input_chars:,}자 상한을 호출 직전에 확인",
         runner.inner.max_input_chars == limits.max_input_chars_per_call == args.max_input_chars),
        ("추적 꺼짐 · 외부 전송 승인 · 실제 집계 전송 승인",
         result.tracing_disabled and result.external_send_approved and result.real_data_send_approved),
        ("실제 집계 전용(원문·제목·개인정보 필드 없음, 개인정보 패턴 0건)",
         sanitized.source_kind is SourceKind.REAL and not pii_findings(sanitized)),
        ("승인 코드 == 전송 내용 지문", args.approve_real_data_send == payload_fingerprint(sanitized)),
        (f"9회 최악 비용 ${ls.MAX_CALLS * per_call:.5f} ≤ 상한 ${args.max_cost_usd}",
         ls.MAX_CALLS * per_call <= args.max_cost_usd),
        (f"9회 입력·출력 상한 비용 ${ceiling:.5f} ≤ 상한", ceiling <= args.max_cost_usd),
        (f"기존 장부 ${ledger:.5f} + 남은 {remaining}회 최악 ${remaining * per_call:.5f} ≤ 상한 ${args.max_cost_usd}",
         remaining >= 0 and ledger + remaining * per_call <= args.max_cost_usd),
    ]


def limits_for(args) -> RunLimits:
    return RunLimits(max_input_chars_per_call=args.max_input_chars,
                     max_output_tokens_per_call=args.max_output_tokens,
                     max_cost_usd=args.max_cost_usd or ls.DEFAULT_COST_CAP)


def conditions(args) -> dict:
    return {"max_input_chars": args.max_input_chars, "max_output_tokens": args.max_output_tokens,
            "max_cost_usd": args.max_cost_usd, "price_in": args.price_in, "price_out": args.price_out}


def run_phase(args) -> None:
    prices = (args.price_in, args.price_out) if args.price_in and args.price_out else None
    if args.phase in ("final", "status"):
        if not REAL_STATE_PATH.exists():
            raise SystemExit("저장된 실제 집계 실행 상태가 없습니다.")
        saved = json.loads(REAL_STATE_PATH.read_text(encoding="utf-8"))
        stats = pick_stats(saved["dataset"])
        result, budget, usage, errors = ls.load_state(REAL_STATE_PATH, stats, limits_for(args), real_data=True)
        if args.phase == "final":
            if next_step(result).stage is not StageName.FINAL_APPROVAL:
                raise SystemExit("최종 승인 단계가 아닙니다.")
            if args.decision not in ("approve", "reject"):
                raise SystemExit("--decision approve|reject 가 필요합니다(사용자 결정).")
            result = result.approve_final() if args.decision == "approve" else result.reject()
            if args.decision == "approve":
                report = next(s.output_text for s in reversed(result.stages)
                              if s.stage in (StageName.WRITER_REVISION, StageName.WRITER_DRAFT)
                              and s.status.value == "succeeded")
                REAL_REPORT_PATH.write_text(
                    f"<!-- LIVE 실제 집계 통계({stats.dataset_id}) 기반 · {VOC_AGENT_MODEL} · 상담 원문·제목·개인정보 미전송 · "
                    f"통계로 확인할 수 없는 원인·효과는 가설 -->\n\n{report}\n", encoding="utf-8")
            ls.save_state(REAL_STATE_PATH, result, budget, usage, errors,
                          {"dataset": stats.dataset_id, "conditions": saved.get("conditions"),
                           "payload_fingerprint": saved.get("payload_fingerprint")})
        ls.show(result, usage, errors, prices)
        return

    ls.require_approval(args)
    saved: dict = {}
    if args.phase == "start":
        if REAL_STATE_PATH.exists():
            raise SystemExit("이미 실제 집계 실행 기록(reports/live_real_state.json)이 있어 시작하지 않습니다. "
                             "재실행은 기존 비용을 포함한 예산 재승인이 필요합니다.")
        if not args.dataset:
            raise SystemExit("--dataset receipt|online|hazard 가 필요합니다.")
        stats = pick_stats(args.dataset)
    else:
        if not REAL_STATE_PATH.exists():
            raise SystemExit("저장된 실제 집계 실행 상태가 없습니다.")
        saved = json.loads(REAL_STATE_PATH.read_text(encoding="utf-8"))
        if saved.get("conditions") != conditions(args):
            raise SystemExit("시작 때와 다른 비용·상한 조건입니다. 같은 조건으로 실행하세요.")
        stats = pick_stats(saved["dataset"])

    sanitized = to_sanitized_monthly(stats, SourceKind.REAL)
    fingerprint = payload_fingerprint(sanitized)
    if args.approve_real_data_send != fingerprint:
        raise SystemExit("실제 집계 전송 승인 코드가 없거나 현재 전송 내용과 다릅니다. --phase plan으로 전송 내용을 "
                         "다시 확인하고 그 화면의 승인 코드를 지정하세요. 호출하지 않았습니다.")
    if args.phase != "start" and saved.get("payload_fingerprint") != fingerprint:
        raise SystemExit("저장된 실행이 승인받은 전송 내용과 현재 집계가 다릅니다. 호출하지 않았습니다.")
    if args.notes:
        raise SystemExit("실제 집계 실행에서는 사람의 자유 메모를 전송할 수 없습니다. 호출하지 않았습니다.")

    from dotenv import dotenv_values
    from openai import AsyncOpenAI

    key = (dotenv_values(ROOT.parent / ".env").get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY가 없어 중단합니다.")
    assert prices is not None
    limits = limits_for(args)
    est = ls.estimate(stats, args.max_output_tokens, lambda s: to_sanitized_monthly(s, SourceKind.REAL),
                      args.max_input_chars)
    client = AsyncOpenAI(api_key=key, base_url=ls.OPENAI_URL, max_retries=0, timeout=90)
    inner = SdkRunner(VOC_AGENT_MODEL, approved_live=True, max_output_tokens=args.max_output_tokens,
                      client=client, max_input_chars=args.max_input_chars)

    if args.phase == "start":
        result = new_run(RUN_ID, (to_provenance(stats),), RunMode.LIVE, limits,
                         external_send_approved=True, real_data_send_approved=True)
        budget, usage, errors = CallBudget(limits), [], []
    else:
        result, budget, usage, errors = ls.load_state(REAL_STATE_PATH, stats, limits, real_data=True)
        if next_step(result).stage is not StageName.HUMAN_REVIEW:
            raise SystemExit("Human Review 단계가 아닙니다.")

    runner = ls.LoggingRunner(inner, prices, args.max_cost_usd, ls.stage_cost(est, prices), usage, errors,
                              fail_tokens=(est["max_stage_input_tokens"], args.max_output_tokens))
    checks = real_preflight(args, est, limits, client, runner, result, sanitized, budget)
    print("--- 사전 점검 ---")
    for label, passed in checks:
        print(f"  [{'OK' if passed else '실패'}] {label}")
    if not all(passed for _, passed in checks):
        raise SystemExit("사전 점검 실패 — 호출하지 않았습니다.")

    ctx = RunContext(sanitized, runner, budget, make_checked_text, est_cost_usd=ls.stage_cost(est, prices))
    human = None
    if args.phase == "human":
        if args.decision not in ("as_is", "revise"):
            raise SystemExit("--decision as_is|revise 가 필요합니다(사용자 결정).")
        human = HumanInput(HumanDecision.APPROVE_AS_IS if args.decision == "as_is"
                           else HumanDecision.REQUEST_REVISION, None)
    extra = {"dataset": stats.dataset_id, "conditions": conditions(args), "payload_fingerprint": fingerprint}
    lock = REAL_STATE_PATH.with_name(REAL_STATE_PATH.name + ".lock")
    try:   # 배타 잠금: 동시 실행·비정상 종료 뒤에 같은 유료 호출이 반복되지 않게 한다
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise SystemExit("다른 실행이 진행 중이거나 이전 실행이 비정상 종료되어 잠금 파일이 남아 있습니다. "
                         "호출하지 않았습니다. 호출 기록을 확인한 뒤 사용자가 직접 정리해야 합니다.") from None
    try:
        if args.phase == "start" and REAL_STATE_PATH.exists():     # 잠금 직전에 다른 실행이 시작했는지 재확인
            raise SystemExit("이미 실제 집계 실행 기록이 있어 시작하지 않습니다.")
        if args.phase != "start":
            fresh = json.loads(REAL_STATE_PATH.read_text(encoding="utf-8"))
            if fresh["budget"]["total"] != budget.total or len(fresh["stages"]) != len(result.stages):
                raise SystemExit("읽은 뒤 상태가 바뀌었습니다(다른 실행이 끝남). 호출하지 않았습니다.")
        ls.save_state(REAL_STATE_PATH, result, budget, runner.usage_log, errors, extra)   # 첫 유료 호출 전 기록
        try:
            result = advance(result, ctx, human)
        except BaseException:
            errors.append("예상치 못한 중단 — 상태·비용 기록은 보존하고 실행을 종료함")
            ls.save_state(REAL_STATE_PATH, result.stop(ls.ErrorCode.STAGE_FAILED), budget, runner.usage_log,
                          errors, extra)
            raise
        ls.save_state(REAL_STATE_PATH, result, budget, runner.usage_log, errors, extra)
        ls.show(result, runner.usage_log, errors, prices)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="LIVE(실제 API) — 실제 집계 통계 전송(별도 실행)")
    parser.add_argument("--phase", choices=["plan", "start", "human", "final", "status"], default="plan")
    parser.add_argument("--dataset", choices=["receipt", "online", "hazard"])
    parser.add_argument("--approve-paid-calls", action="store_true")
    parser.add_argument("--approve-real-data-send", help="plan이 출력한 전송 내용 승인 코드")
    parser.add_argument("--price-in", type=float)
    parser.add_argument("--price-out", type=float)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_OUTPUT_CAP)
    parser.add_argument("--max-input-chars", type=int, default=DEFAULT_INPUT_CAP)
    parser.add_argument("--decision", help="human: as_is|revise, final: approve|reject (사용자 결정만 전달)")
    parser.add_argument("--notes", help="(실제 집계 실행에서는 사용 불가)")
    args = parser.parse_args(argv)
    if args.phase in ("plan", "start", "human"):
        enforce_approved_limits(args)
    if args.phase == "plan":
        if not args.dataset:
            raise SystemExit("--dataset receipt|online|hazard 를 지정하세요.")
        print(format_plan(pick_stats(args.dataset), args))
        print("\n(요청서만 출력했습니다. 실제 호출은 하지 않았습니다.)")
        return
    run_phase(args)


if __name__ == "__main__":
    main()
