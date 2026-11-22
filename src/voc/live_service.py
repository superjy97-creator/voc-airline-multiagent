"""웹 LIVE 서비스(합성 데이터 전용): 관리자 인증·승인 코드·1회 실행 장부·중복 호출 방지.

원칙(모두 서버 측에서 강제):
- 접근 제한(비밀번호 없음): ① 이 PC(loopback)에서만 열린 로컬 실행(`local_access`) 또는 ② 플랫폼 로그인으로 확인된 열람자
  (`viewer_access`, 앱을 '지정한 사람만 열람'으로 설정해야 함)에게만 LIVE가 열린다. 공개 익명 방문자에게는 키가 있어도 열리지 않는다.
  토큰이 없으면 어떤 LIVE 함수도 동작하지 않는다.
- OPENAI_API_KEY는 `LiveSettings`(repr 숨김)에만 있고 화면·상태 파일·오류 메시지에 쓰지 않는다.
- 전송 입력은 합성 집계(`to_sanitized_monthly`)뿐이며 자유 입력·상담 원문은 스키마상 담을 수 없다.
- 실행 장부(상태 파일)가 있으면 새 실행을 시작하지 않는다. 배타 잠금 파일이 있으면 어떤 호출도 하지 않는다.
- 호출 직전에 '진행 중' 표시를 먼저 저장해, 도중에 끊겨도 같은 호출이 다시 나가지 않는다(fail-closed).
"""

import hashlib
import json
import os
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
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
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


def _host_only(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text.startswith("["):                      # [::1]:8501
        return text[1:text.find("]")] if "]" in text else text
    return text.rsplit(":", 1)[0] if text.count(":") == 1 else text


def local_access(server_address: str | None, host_header: str | None, app_dir: str | Path) -> bool:
    """이 PC에서만 열린 로컬 실행인지 서버 측 사실로 판정한다(공개 배포는 항상 False).

    세 조건을 모두 만족해야 한다: ① 서버 바인딩 주소가 loopback ② 요청 Host가 loopback
    ③ 앱이 Streamlit Cloud 경로(/mount/)가 아닌 곳에서 실행. 하나라도 어긋나면 LIVE는 열리지 않는다.
    """
    if _host_only(server_address) not in LOOPBACK_HOSTS or _host_only(host_header) not in LOOPBACK_HOSTS:
        return False
    return not str(app_dir).replace("\\", "/").startswith("/mount/")


def viewer_access(email: object) -> bool:
    """플랫폼(Streamlit Cloud)이 로그인으로 확인한 열람자만 통과시킨다.

    비공개(지정한 사람만 열람) 앱에서는 로그인하지 않으면 앱에 접근할 수 없고, 로그인한 열람자의 이메일이 `st.user`에 채워진다.
    공개 앱의 익명 방문자는 이메일이 없어 닫힌다. 이메일 값 자체는 저장·표시하지 않는다.
    """
    return isinstance(email, str) and 3 <= len(email) <= 254 and "@" in email and " " not in email


class LocalGate:
    """로컬 접근 토큰 발급기. 허용되지 않은 환경에서는 토큰을 발급하지 않는다."""

    def __init__(self, allowed: bool, *, session_seconds: float = 3600,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._allowed, self._session, self._clock = bool(allowed), session_seconds, clock
        self._tokens: dict[str, float] = {}
        self._mutex = threading.Lock()

    def issue(self) -> str | None:
        if not self._allowed:
            return None
        with self._mutex:
            token = secrets.token_urlsafe(32)
            self._tokens[token] = self._clock() + self._session
            return token

    def valid(self, token: object) -> bool:
        with self._mutex:
            now = self._clock()
            self._tokens = {t: e for t, e in self._tokens.items() if e > now}
            return self._allowed and isinstance(token, str) and token in self._tokens

    def logout(self, token: object) -> None:
        with self._mutex:
            if isinstance(token, str):
                self._tokens.pop(token, None)

    def require(self, token: object) -> None:
        if not self.valid(token):
            raise LiveError("이 환경에서는 LIVE를 실행할 수 없습니다(로컬 실행에서만 허용).")


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


def start_run(store: LiveStore, gate: LocalGate, token: object, settings: LiveSettings, stats,
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


def load_view(store: LiveStore, gate: LocalGate, token: object) -> LiveView | None:
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


def continue_run(store: LiveStore, gate: LocalGate, token: object, settings: LiveSettings,
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


def finalize(store: LiveStore, gate: LocalGate, token: object, approve: bool) -> None:
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
