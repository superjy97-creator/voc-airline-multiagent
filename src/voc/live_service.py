"""웹 LIVE 서비스(합성 데이터 전용): 공개 이용 한도·승인 코드·실행별 장부·중복 호출 방지.

공개(로그인 없음) 서비스이므로 한도는 모두 서버 측에서 강제한다.
- 전체 한도(`PublicLimiter`): 동시 실행 수, 전체 실행 수, 전체 비용(실행 시작 시 최악 비용을 예약), 사용자(접속 주소)별 실행 수,
  남용 감지 시 자동 차단. 모든 세션이 같은 서버 프로세스·같은 장부 파일을 쓰므로 세션을 바꿔도 우회할 수 없다.
- 실행별 한도: 호출 9회·비용 $0.05·입력 10,000자·재시도 0회. 기록이 있으면 같은 실행은 다시 시작하지 않는다.
- OPENAI_API_KEY는 `LiveSettings`(repr 숨김)에만 있고 화면·상태 파일·오류 메시지에 쓰지 않는다.
- 전송 입력은 합성 집계(`to_sanitized_monthly`)뿐이며 자유 입력·상담 원문은 스키마상 담을 수 없다.
- 호출 직전에 '진행 중' 표시를 먼저 저장해, 도중에 끊겨도 같은 호출이 다시 나가지 않는다(fail-closed).
한계: 서버 장부는 재시작하면 초기화되므로(Cloud 파일시스템은 임시) 재시작을 넘는 지출 상한은 OpenAI 프로젝트의 지출 한도로 보장해야 한다.
"""

import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from voc.config import VOC_AGENT_MODEL
from voc.contracts import (
    ApprovalStatus,
    CallBudget,
    ErrorCode,
    PipelineResult,
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
    LoggingRunner,
    estimate,
    load_state,
    save_state,
    stage_cost,
    usd,
)
from voc.pipeline import HumanInput, RunContext, advance, new_run
from voc.runners import MAX_TURNS, SdkRunner
from voc.sanitize import make_checked_text
from voc.stats import (
    synthetic_stats,
    synthetic_trend_stats,
    to_provenance,
    to_sanitized_monthly,
)

OPENAI_URL = "https://api.openai.com/v1"
PRICE_FLOOR = (0.10, 0.50)      # 사용자가 승인한 100만 토큰당 단가(입력, 출력). 이보다 낮은 설정은 거부한다.
MAX_OUTPUT_TOKENS = 2500
INPUT_CAP_CHARS = 10_000   # 호출당 입력 글자 상한. 첫 실제 실행에서 4,000자는 Reviewer 입력(약 6,450자)보다 작아 차단됐다.
STATE_NAME, LOCK_NAME = "live_web_state.json", "live_web.lock"


class LiveError(Exception):
    """화면에 그대로 보여 줘도 안전한 메시지만 담는다(키·원문 없음)."""


@dataclass(frozen=True)
class LiveSettings:
    api_key: str = field(repr=False)
    price_in: float = PRICE_FLOOR[0]
    price_out: float = PRICE_FLOOR[1]


def load_settings(get: Callable[[str], object]) -> LiveSettings | None:
    """서버 비밀값에서 설정을 읽는다. 키가 없으면 None(LIVE 비활성). 단가는 승인된 값 이상만 허용한다."""
    key = str(get("OPENAI_API_KEY") or "").strip()
    if not key:
        return None
    try:
        price_in = float(str(get("PRICE_IN_PER_M") or PRICE_FLOOR[0]))
        price_out = float(str(get("PRICE_OUT_PER_M") or PRICE_FLOOR[1]))
    except ValueError as error:
        raise LiveError("단가 설정이 숫자가 아닙니다. LIVE를 비활성화합니다.") from error
    if price_in < PRICE_FLOOR[0] or price_out < PRICE_FLOOR[1]:
        raise LiveError("설정한 단가가 승인된 단가보다 낮아 비용을 과소평가할 수 있습니다. LIVE를 비활성화합니다.")
    return LiveSettings(key, price_in, price_out)


class SessionGate:
    """세션 토큰 발급기. 모든 LIVE 함수는 이 토큰이 유효해야만 동작한다(서버가 발급한 토큰만 유효)."""

    def __init__(self, *, session_seconds: float = 3600, clock: Callable[[], float] = time.monotonic) -> None:
        self._session, self._clock = session_seconds, clock
        self._tokens: dict[str, float] = {}
        self._mutex = threading.Lock()

    def issue(self) -> str:
        with self._mutex:
            token = secrets.token_urlsafe(32)
            self._tokens[token] = self._clock() + self._session
            return token

    def valid(self, token: object) -> bool:
        with self._mutex:
            now = self._clock()
            self._tokens = {t: e for t, e in self._tokens.items() if e > now}
            return isinstance(token, str) and token in self._tokens

    def logout(self, token: object) -> None:
        with self._mutex:
            if isinstance(token, str):
                self._tokens.pop(token, None)

    def require(self, token: object) -> None:
        if not self.valid(token):
            raise LiveError("유효한 세션이 아닙니다. 새로고침 후 다시 시도하세요.")


def default_store_dir() -> Path:
    reports = Path(__file__).resolve().parents[2] / "reports"      # 로컬 프로젝트에만 있고 저장소에는 올라가지 않는다
    return reports / "live_web" if reports.is_dir() else Path(tempfile.gettempdir()) / "voc_live"


class LiveStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else default_store_dir()
        self.state_path = self.directory / STATE_NAME
        self.lock_path = self.directory / LOCK_NAME

    def exists(self) -> bool:
        return self.state_path.exists()

    @contextmanager
    def locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise LiveError("다른 LIVE 작업이 진행 중이거나 비정상 종료된 잠금이 있어 호출하지 않았습니다.") from error
        try:
            yield
        finally:
            os.close(fd)
            self.lock_path.unlink(missing_ok=True)

    def read(self) -> dict | None:
        return json.loads(self.state_path.read_text(encoding="utf-8")) if self.exists() else None


# ---- 공개 이용 한도(전체·사용자별·남용 차단) ----------------------------------

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{12,32}")


@dataclass(frozen=True)
class PublicLimits:
    max_active_runs: int = 1            # 동시에 진행할 수 있는 실행 수(전체)
    max_runs_total: int = 10            # 서버 프로세스 수명 동안 시작할 수 있는 전체 실행 수
    max_total_cost_usd: float = 0.50    # 전체 비용 한도: 시작 시 실행당 최악 비용을 예약하고, 끝나면 실제 사용량으로 정산
    per_client_runs: int = 3            # 접속 주소별 실행 수(창 안)
    per_client_window_s: float = 86400
    active_ttl_s: float = 1800          # 이 시간 동안 진행이 없으면 실행을 만료(예약 비용은 유지)
    reject_trip: int = 30               # 창 안에 거부가 이만큼 쌓이면 자동 차단
    reject_window_s: float = 600
    failure_trip: int = 3               # API 실패·중단 실행이 이만큼이면 자동 차단
    block_seconds: float = 3600


class PublicLimiter:
    """모든 세션이 공유하는 서버 측 이용 한도. 장부 파일을 잠금으로 보호해 동시 접속에서도 한도를 넘지 못한다."""

    def __init__(self, directory: Path | None = None, limits: PublicLimits | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.directory = Path(directory) if directory else default_store_dir()
        self.limits = limits or PublicLimits()
        self.clock = clock
        self.path = self.directory / "public_ledger.json"
        self.lock_path = self.directory / "public_ledger.lock"

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 5
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:      # 아주 짧게만 잡는 잠금이므로, 오래 남은 잠금은 비정상 종료의 잔재로 보고 치운다
                    if time.time() - self.lock_path.stat().st_mtime > 30:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
                if time.monotonic() > deadline:
                    raise LiveError("요청이 몰려 잠시 처리할 수 없습니다. 잠시 후 다시 시도하세요.") from None
                time.sleep(0.02)
        try:
            yield
        finally:
            os.close(fd)
            self.lock_path.unlink(missing_ok=True)

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"runs": [], "rejects": [], "failures": 0, "blocked_until": 0.0, "blocked_reason": None}

    def _save(self, state: dict) -> None:
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _total_cost(state: dict) -> float:
        return sum(r["cost"] for r in state["runs"])

    def _block(self, state: dict, reason: str, now: float) -> None:
        state["blocked_until"], state["blocked_reason"] = now + self.limits.block_seconds, reason

    def _reject(self, state: dict, now: float) -> None:
        window = now - self.limits.reject_window_s
        state["rejects"] = [t for t in state["rejects"] if t > window] + [now]
        if len(state["rejects"]) >= self.limits.reject_trip:
            self._block(state, "거부된 요청이 짧은 시간에 너무 많아 남용으로 판단", now)
            state["rejects"] = []

    def reject(self) -> None:
        """승인 코드 불일치 등 거부된 요청을 기록한다(많이 쌓이면 자동 차단)."""
        with self._locked():
            state = self._load()
            self._reject(state, self.clock())
            self._save(state)

    def admit(self, client: str, worst_cost: float) -> str:
        """실행 시작을 허가하고 최악 비용을 예약한다. 한도를 넘으면 LiveError(호출 없음)."""
        lim = self.limits
        with self._locked():
            state, now = self._load(), self.clock()
            for run in state["runs"]:
                if run["state"] == "active" and now - run["started"] > lim.active_ttl_s:
                    run["state"] = "expired"          # 진행이 멈춘 실행: 자리는 비우되 예약 비용은 유지한다
            reason = None
            recent = [r for r in state["runs"] if r["client"] == client and now - r["started"] < lim.per_client_window_s]
            if now < state["blocked_until"]:
                self._save(state)
                raise LiveError("남용 방지를 위해 LIVE가 일시 차단되었습니다. 잠시 후 다시 시도하세요(DRY_RUN은 계속 사용할 수 있습니다).")
            if sum(1 for r in state["runs"] if r["state"] == "active") >= lim.max_active_runs:
                reason = "다른 사용자가 LIVE를 진행 중입니다. 끝난 뒤 다시 시도하세요."
            elif len(state["runs"]) >= lim.max_runs_total:
                reason = "LIVE 전체 실행 한도에 도달했습니다. DRY_RUN을 사용하세요."
            elif self._total_cost(state) + worst_cost > lim.max_total_cost_usd:
                reason = "LIVE 전체 비용 한도에 도달했습니다. DRY_RUN을 사용하세요."
            elif len(recent) >= lim.per_client_runs:
                reason = "이 접속 주소의 LIVE 이용 한도에 도달했습니다."
            if reason:
                self._reject(state, now)
                self._save(state)
                raise LiveError(reason)
            run_id = secrets.token_urlsafe(12)
            state["runs"].append({"id": run_id, "client": client, "started": now, "state": "active",
                                  "cost": worst_cost, "settled": False})
            self._save(state)
            return run_id

    def known(self, run_id: object) -> bool:
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            return False
        with self._locked():
            return any(r["id"] == run_id for r in self._load()["runs"])

    def _settle(self, run_id: str, actual_cost: float | None, failed: bool) -> None:
        with self._locked():
            state, now = self._load(), self.clock()
            run = next((r for r in state["runs"] if r["id"] == run_id), None)
            if run is None or run["settled"]:
                return
            run["settled"], run["state"] = True, "done"
            if actual_cost is not None:        # 알 수 없는 비용(끊김)은 예약한 최악 비용을 그대로 둔다
                run["cost"] = max(float(actual_cost), 0.0)
            if failed:
                state["failures"] += 1
                if state["failures"] >= self.limits.failure_trip:
                    self._block(state, "API 실패·중단이 반복되어 자동 차단", now)
                    state["failures"] = 0
            self._save(state)

    def finish(self, run_id: str, actual_cost: float | None) -> None:
        """정상 종료(최종 승인·반려)를 실제 사용량으로 정산한다. 같은 실행은 한 번만 정산된다."""
        self._settle(run_id, actual_cost, failed=False)

    def fail(self, run_id: str, actual_cost: float | None) -> None:
        """실패·중단된 실행을 정산하고 실패 횟수를 센다(반복되면 LIVE 자동 차단)."""
        self._settle(run_id, actual_cost, failed=True)

    def status(self) -> dict:
        with self._locked():
            state, now = self._load(), self.clock()
            lim = self.limits
            active = sum(1 for r in state["runs"] if r["state"] == "active" and now - r["started"] <= lim.active_ttl_s)
            return {"blocked": now < state["blocked_until"], "reason": state["blocked_reason"],
                    "active": active, "runs_left": max(lim.max_runs_total - len(state["runs"]), 0),
                    "budget_left": max(lim.max_total_cost_usd - self._total_cost(state), 0.0)}


def run_dir(run_id: str, base: Path | None = None) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise LiveError("실행 번호 형식이 올바르지 않습니다.")
    return (Path(base) if base else default_store_dir()) / "runs" / run_id


# ---- 승인 요청서 -------------------------------------------------------------

SYNTHETIC_IDS = frozenset(s.dataset_id for s in (synthetic_trend_stats(), *synthetic_stats()))


@dataclass(frozen=True)
class Plan:
    dataset_id: str
    label: str
    period: tuple[str, str]
    request: str
    est: dict
    prices: tuple[float, float]
    per_call: float
    worst: float
    ceiling: float
    code: str
    checks: tuple

    @property
    def ok(self) -> bool:
        return all(passed for _, passed in self.checks)

    def text(self) -> str:
        p_in, p_out = self.prices
        lines = (
            (f"모델: {VOC_AGENT_MODEL} (5개 에이전트 공통) · 자료: {self.label} (합성 데이터) · "
             f"기간 {self.period[0]}~{self.period[1]} · 목적 ‘{self.request}’"),
            f"최대 호출 {MAX_CALLS}회(단계당 요청 1회, 재시도 0회, 추적 꺼짐) · 출력 상한 호출당 {MAX_OUTPUT_TOKENS:,}토큰",
            ("전송 내용: 합성 집계 숫자·날짜·열거값과 이전 단계 모델 출력, 사람 메모(개인정보 검사 통과분). "
             "상담 원문·개인정보·자유 입력은 전송하지 않음"),
            (f"예상 입력 토큰(현실) 약 {self.est['input_tokens_realistic']:,} / 상한 {self.est['input_tokens_ceiling']:,}, "
             f"출력 상한 합계 {self.est['output_tokens_ceiling']:,}"),
            (f"단가(100만 토큰당 입력 {p_in} / 출력 {p_out} USD) 기준 추정: 호출당 예약 ${self.per_call:.5f}, "
             f"9회 최악 ${self.worst:.4f}, 입력·출력 상한 ${self.ceiling:.4f} (상한 ${DEFAULT_COST_CAP}). "
             "추정치이며 실제 사용량으로 별도 기록"),
        )
        return "\n".join(lines)


def build_plan(stats, period: tuple[str, str], request: str, prices: tuple[float, float]) -> Plan:
    if stats.dataset_id not in SYNTHETIC_IDS:
        raise LiveError("실제 집계 자료의 LIVE는 별도 승인 전까지 비활성입니다. 합성 자료만 선택할 수 있습니다.")
    sanitized = to_sanitized_monthly(stats, SourceKind.SYNTHETIC, period=period, request=request)
    est = estimate(stats, MAX_OUTPUT_TOKENS, lambda s: to_sanitized_monthly(
        s, SourceKind.SYNTHETIC, period=period, request=request), INPUT_CAP_CHARS)
    per_call = stage_cost(est, prices)
    worst = MAX_CALLS * per_call
    ceiling = usd(est["input_tokens_ceiling"], prices[0]) + usd(est["output_tokens_ceiling"], prices[1])
    payload = json.dumps({"fields": sanitized.fields, "model": VOC_AGENT_MODEL, "out": MAX_OUTPUT_TOKENS,
                          "prices": prices, "dataset": stats.dataset_id}, ensure_ascii=False, sort_keys=True)
    code = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8].upper()
    checks = (
        ("합성 데이터 전용", sanitized.source_kind is SourceKind.SYNTHETIC),
        ("단가가 승인 단가 이상", prices[0] >= PRICE_FLOOR[0] and prices[1] >= PRICE_FLOOR[1]),
        (f"9회 최악 비용 ${worst:.4f} ≤ 상한 ${DEFAULT_COST_CAP}", worst <= DEFAULT_COST_CAP),
        (f"입력·출력 상한 비용 ${ceiling:.4f} ≤ 상한 ${DEFAULT_COST_CAP}", ceiling <= DEFAULT_COST_CAP),
        (f"호출 {MAX_CALLS}회 · 호출당 요청 {MAX_TURNS}회 · 재시도 0회", MAX_TURNS == 1),
    )
    return Plan(stats.dataset_id, stats.label, period, request, est, prices, per_call, worst, ceiling, code, checks)


# ---- 실행 ---------------------------------------------------------------------

def default_client(api_key: str):
    from openai import (
        AsyncOpenAI,  # 필요할 때만 불러온다(공개 DRY_RUN 경로는 import하지 않음)
    )
    return AsyncOpenAI(api_key=api_key, base_url=OPENAI_URL, max_retries=0, timeout=90)


class _JournalRunner:
    """호출 직전에 '진행 중' 표시를 먼저 저장한다. 끊기면 그 표시가 남아 같은 호출이 다시 나가지 않는다."""

    def __init__(self, inner: LoggingRunner, store: LiveStore, budget: CallBudget) -> None:
        self.inner, self.store, self.budget = inner, store, budget
        self.usage_log = inner.usage_log

    def run_stage(self, step):
        data = self.store.read() or {}
        data["inflight"] = step.stage.value
        data["budget"] = {"total": self.budget.total, "by_role": self.budget.by_role,
                          "by_stage": {k.value: v for k, v in self.budget.by_stage.items()},
                          "committed_cost": self.budget.committed_cost}
        self.store.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.inner.run_stage(step)


def _limits() -> RunLimits:
    return RunLimits(max_input_chars_per_call=INPUT_CAP_CHARS,
                     max_output_tokens_per_call=MAX_OUTPUT_TOKENS, max_cost_usd=DEFAULT_COST_CAP)


def _stats_for(dataset_id: str):
    return next(s for s in (synthetic_trend_stats(), *synthetic_stats()) if s.dataset_id == dataset_id)


def _make_context(settings: LiveSettings, stats, meta: dict, budget: CallBudget, usage: list, errors: list,
                  store: LiveStore, client_factory):
    prices = (settings.price_in, settings.price_out)
    sanitized = to_sanitized_monthly(stats, SourceKind.SYNTHETIC, period=tuple(meta["period"]),
                                     request=meta["request"])
    est = estimate(stats, MAX_OUTPUT_TOKENS, lambda s: to_sanitized_monthly(
        s, SourceKind.SYNTHETIC, period=tuple(meta["period"]), request=meta["request"]), INPUT_CAP_CHARS)
    client = client_factory(settings.api_key)
    inner = SdkRunner(VOC_AGENT_MODEL, approved_live=True, max_output_tokens=MAX_OUTPUT_TOKENS, client=client)
    logging = LoggingRunner(inner, prices, DEFAULT_COST_CAP, stage_cost(est, prices), usage, errors,
                            fail_tokens=(est["max_stage_input_tokens"], MAX_OUTPUT_TOKENS))
    ctx = RunContext(sanitized, _JournalRunner(logging, store, budget), budget, make_checked_text,
                     est_cost_usd=stage_cost(est, prices))
    return ctx, logging, est, prices


def start_run(store: LiveStore, gate: SessionGate, token: object, settings: LiveSettings, stats,
              period: tuple[str, str], request: str, typed_code: str,
              client_factory: Callable = default_client) -> None:
    gate.require(token)
    plan = build_plan(stats, period, request, (settings.price_in, settings.price_out))
    if not plan.ok:
        raise LiveError("사전 점검을 통과하지 못해 호출하지 않았습니다.")
    if str(typed_code).strip().upper() != plan.code:
        raise LiveError("승인 코드가 일치하지 않아 호출하지 않았습니다.")
    with store.locked():
        if store.exists():
            raise LiveError("이미 LIVE 실행 기록이 있어 새 실행을 시작하지 않습니다(기록 보존).")
        limits = _limits()
        budget, usage, errors = CallBudget(limits), [], []
        meta = {"dataset_id": stats.dataset_id, "label": stats.label, "period": list(period), "request": request,
                "code": plan.code, "model": VOC_AGENT_MODEL, "prices": [settings.price_in, settings.price_out],
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        result = new_run("live-web", (to_provenance(stats),), RunMode.LIVE, limits, external_send_approved=True)
        ctx, logging, _, _ = _make_context(settings, stats, meta, budget, usage, errors, store, client_factory)
        save_state(store.state_path, result, budget, logging.usage_log, errors, {"meta": meta})
        _advance_and_save(store, result, ctx, logging, budget, errors, None, meta)


def _advance_and_save(store, result, ctx, logging, budget, errors, human, meta) -> None:
    try:
        result = advance(result, ctx, human)
    except ValueError:
        raise
    except BaseException:
        errors.append("예상치 못한 중단 — 상태·비용 기록은 보존하고 실행을 종료함")
        save_state(store.state_path, result.stop(ErrorCode.STAGE_FAILED), budget, logging.usage_log, errors,
                   {"meta": meta, "inflight": None})
        raise
    save_state(store.state_path, result, budget, logging.usage_log, errors, {"meta": meta, "inflight": None})


@dataclass(frozen=True)
class LiveView:
    result: PipelineResult
    usage: list
    errors: list
    meta: dict
    interrupted: bool
    spent: float
    stage_count: int


def load_view(store: LiveStore, gate: SessionGate, token: object) -> LiveView | None:
    gate.require(token)
    data = store.read()
    if data is None:
        return None
    stats = _stats_for(data["meta"]["dataset_id"])
    result, _, usage, errors = load_state(store.state_path, stats, _limits())
    if data.get("approval"):
        result = result.approve_final() if data["approval"] == "approved" else result.reject()
    prices = tuple(data["meta"]["prices"])
    spent = sum(usd(u["input_tokens"], prices[0]) + usd(u["output_tokens"], prices[1]) for u in usage)
    return LiveView(result, usage, errors, data["meta"], bool(data.get("inflight")), spent, len(result.stages))


def continue_run(store: LiveStore, gate: SessionGate, token: object, settings: LiveSettings,
                 expected_stage_count: int, human: HumanInput,
                 client_factory: Callable = default_client) -> None:
    """Human Review 결정을 반영해 이어서 실행한다. 화면이 오래된 요청(새로고침·중복 클릭)은 호출 없이 거부한다."""
    gate.require(token)
    with store.locked():
        data = store.read()
        if data is None:
            raise LiveError("진행 중인 LIVE 실행이 없습니다.")
        if data.get("inflight"):
            raise LiveError("이전 호출이 도중에 끊겨 과금 여부를 알 수 없습니다. 장부에 최악 비용으로 반영했고 더 호출하지 않습니다.")
        if data.get("approval"):
            raise LiveError("이미 최종 결정된 실행입니다.")
        stats = _stats_for(data["meta"]["dataset_id"])
        result, budget, usage, errors = load_state(store.state_path, stats, _limits())
        if len(result.stages) != expected_stage_count or next_step(result).stage is not StageName.HUMAN_REVIEW:
            raise LiveError("화면이 오래되었거나 이미 처리된 단계입니다. 새로고침 후 확인하세요(호출하지 않았습니다).")
        meta = data["meta"]
        ctx, logging, est, prices = _make_context(settings, stats, meta, budget, usage, errors, store, client_factory)
        remaining = _limits().max_total_calls - budget.total
        if not remaining >= 0 or logging.spent() + remaining * stage_cost(est, prices) > DEFAULT_COST_CAP:
            raise LiveError("기존 장부와 남은 호출의 최악 비용이 상한을 넘을 수 있어 호출하지 않았습니다.")
        _advance_and_save(store, result, ctx, logging, budget, errors, human, meta)


def finalize(store: LiveStore, gate: SessionGate, token: object, approve: bool) -> None:
    """최종 승인/반려(호출 없음). 사용자의 결정만 기록한다."""
    gate.require(token)
    with store.locked():
        data = store.read()
        if data is None or data.get("inflight") or data.get("approval"):
            raise LiveError("최종 결정을 할 수 없는 상태입니다.")
        stats = _stats_for(data["meta"]["dataset_id"])
        result, budget, usage, errors = load_state(store.state_path, stats, _limits())
        if next_step(result).stage is not StageName.FINAL_APPROVAL:
            raise LiveError("최종 승인 단계가 아닙니다.")
        decided = result.approve_final() if approve else result.reject()
        save_state(store.state_path, decided, budget, usage, errors,
                   {"meta": data["meta"], "approval": "approved" if approve else "rejected"})


def latest_text(result, *stages) -> str | None:
    return next((s.output_text for s in reversed(result.stages)
                 if s.stage in stages and s.status is StageStatus.SUCCEEDED and s.output_text), None)


def is_approved(result) -> bool:
    return result.approval is ApprovalStatus.APPROVED
