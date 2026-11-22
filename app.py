"""VOC 개선 CX 대시보드 (Streamlit).

실행: cd voc_consumer_analysis  →  streamlit run app.py
- DRY_RUN(기본·공개): 가짜 모델(FakeRunner)만 사용한다. 외부 API 호출·상담 원문 사용 없음.
- LIVE(합성 데이터 한정): OPENAI_API_KEY가 있고, 로컬(loopback) 실행이거나 플랫폼 로그인으로 확인된 열람자일 때만 열린다.
  공개 익명 접속에서는 열리지 않는다. 모든 접근 검사와 유료 호출 통제는 서버 측 voc.live_service가 수행한다.
"""

import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import pandas as pd
import streamlit as st

from voc import live_service as live
from voc.analysis_request import MAX_REQUEST_CHARS, SUPPORTED_HELP, parse_request
from voc.contracts import (
    ERROR_MESSAGES,
    ApprovalStatus,
    CallBudget,
    HumanDecision,
    ReviewVerdict,
    SourceKind,
    StageName,
    StageStatus,
    next_step,
)
from voc.pipeline import DRY_RUN_LIMITS, HumanInput, RunContext, advance, new_run
from voc.runners import FakeRunner
from voc.sanitize import make_checked_text
from voc.stats import (
    load_summary,
    month_series,
    synthetic_stats,
    synthetic_trend_stats,
    to_provenance,
    to_sanitized_monthly,
)

SUMMARY_PATH = ROOT / "reports" / "stats_summary.json"
SYNTHETIC_SOURCE = "합성 예시(실제 수치 아님)"
LOCAL_SOURCE = "로컬 집계(reports/stats_summary.json)"
STAGE_LABELS = {
    StageName.COLLECTOR: "Collector", StageName.ANALYST: "Analyst",
    StageName.WRITER_DRAFT: "Writer 초안", StageName.REVIEWER: "Reviewer",
    StageName.HUMAN_REVIEW: "Human Review", StageName.WRITER_REVISION: "Writer 수정",
    StageName.REVIEWER_RECHECK: "Reviewer 재검수", StageName.MANAGER: "Manager",
}
STATUS_LABELS = {
    StageStatus.SUCCEEDED: "완료", StageStatus.SKIPPED: "생략",
    StageStatus.FAILED: "실패", StageStatus.BLOCKED: "차단",
}
# 에이전트 흐름 시각화 노드: (이름, 역할 설명, 해당 단계들)
FLOW = (
    ("Collector", "VOC 사실 정리", (StageName.COLLECTOR,)),
    ("Analyst", "원인 가설·우선순위", (StageName.ANALYST,)),
    ("Writer", "보고서 초안", (StageName.WRITER_DRAFT,)),
    ("Reviewer", "독립 품질 검수", (StageName.REVIEWER,)),
    ("Human Review", "사람 검토", (StageName.HUMAN_REVIEW,)),
    ("수정·재검수", "Writer 수정 → 재검수", (StageName.WRITER_REVISION, StageName.REVIEWER_RECHECK)),
    ("Manager", "결과 종합", (StageName.MANAGER,)),
)
V, D = ReviewVerdict, HumanDecision
SCENARIOS = {
    "Reviewer가 바로 승인": (),
    "1회 수정 필요 후 승인": (V.NEEDS_REVISION, V.APPROVABLE),
    "수정 필요 반복(한도 초과 중단 시연)": (V.NEEDS_REVISION, V.NEEDS_REVISION, V.NEEDS_REVISION),
}
DECISIONS = {"수정 없이 승인": D.APPROVE_AS_IS, "Writer에게 수정 요청": D.REQUEST_REVISION}

CSS = """
<style>
:root { --navy:#0B1F3A; --blue:#1D6FE8; --teal:#12B5A6; --ink:#14213D; --muted:#5B6B82; --line:#E3E9F2; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }
.topbar { display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
          background:#fff; border:1px solid var(--line); border-radius:16px; padding:16px 22px; margin-bottom:14px;
          box-shadow:0 1px 3px rgba(20,33,61,.05); }
.topbar h1 { font-size:1.55rem; margin:0; padding:0; color:var(--navy); line-height:1.25; }
.topbar p { margin:4px 0 0 0; color:var(--muted); font-size:.92rem; }
.badges { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
.badge { display:inline-block; border-radius:999px; padding:4px 13px; font-size:.78rem; font-weight:700;
         border:1px solid transparent; white-space:nowrap; }
.badge.mode { background:#E8F8F6; color:#0B7F75; border-color:#12B5A6; }
.badge.synthetic { background:#EAF2FF; color:#1D4FB8; border-color:#9CC0F5; }
.badge.real { background:#FFF4E0; color:#9A5B00; border-color:#F2C46B; }
.badge.safe { background:#F1F4F9; color:#42536B; border-color:#D5DDE9; }
.datanote { border-radius:12px; padding:10px 14px; font-size:.88rem; margin:0 0 14px 0; }
.datanote.synthetic { background:#EAF2FF; color:#1D3F85; border:1px solid #CFE0FB; }
.datanote.real { background:#FFF7E8; color:#7A4700; border:1px solid #F5DDAA; }
.kpi { border:1px solid var(--line); border-radius:14px; padding:14px 16px; background:#fff; height:100%;
       box-shadow:0 1px 3px rgba(20,33,61,.06); min-height:104px; }
.kpi .l { color:var(--muted); font-size:.82rem; margin-bottom:4px; display:flex; justify-content:space-between;
          gap:8px; }
.kpi .t { font-size:.68rem; font-weight:700; border-radius:6px; padding:1px 7px; white-space:nowrap; }
.kpi .t.synthetic { background:#EAF2FF; color:#1D4FB8; } .kpi .t.real { background:#FFF4E0; color:#9A5B00; }
.kpi .v { color:var(--ink); font-size:1.75rem; font-weight:700; line-height:1.15; word-break:keep-all; }
@media (max-width: 900px) {
  .topbar { padding:14px 16px; } .topbar h1 { font-size:1.3rem; } .badges { justify-content:flex-start; }
  .kpi .v { font-size:1.45rem; } .node { flex:1 1 45%; }
}
.kpi.blue { border-top:4px solid var(--blue); } .kpi.teal { border-top:4px solid var(--teal); }
.kpi.navy { border-top:4px solid var(--navy); } .kpi.gray { border-top:4px solid #A9B6C9; }
.sec { color:var(--ink); font-weight:700; font-size:1.05rem; margin:22px 0 10px 0;
       border-left:5px solid var(--teal); padding-left:10px; }
.flow { display:flex; gap:10px; flex-wrap:wrap; }
.node { flex:1 1 128px; min-width:128px; border:1.5px solid var(--line); border-radius:14px; padding:12px 12px 10px;
        background:#fff; position:relative; }
.node .n { font-weight:700; color:var(--ink); font-size:.95rem; }
.node .d { color:var(--muted); font-size:.74rem; margin:2px 0 8px 0; }
.node .s { display:inline-block; font-size:.74rem; font-weight:600; border-radius:999px; padding:2px 10px; }
.node.pending { color:#8A97AB; } .node.pending .s { background:#EEF2F7; color:#6B7A90; }
.node.active { border-color:var(--blue); background:#EAF2FF; box-shadow:0 0 0 4px rgba(29,111,232,.14); }
.node.active .s { background:var(--blue); color:#fff; }
.node.done { border-color:var(--teal); background:#E8F8F6; } .node.done .s { background:var(--teal); color:#fff; }
.node.skip { border-style:dashed; background:#F6F8FB; } .node.skip .s { background:#DDE4EE; color:#5B6B82; }
.node.fail { border-color:#D64545; background:#FDECEC; } .node.fail .s { background:#D64545; color:#fff; }
.brand { color:#fff; font-weight:700; font-size:1.15rem; margin:4px 0 2px 0; }
.brand-sub { color:#A9BEDD; font-size:.8rem; margin-bottom:14px; }
</style>
"""


@st.cache_data
def load_stats(source: str):
    if source == SYNTHETIC_SOURCE:
        return (synthetic_trend_stats(), *synthetic_stats())
    return load_summary(SUMMARY_PATH) if SUMMARY_PATH.exists() else ()


def available_sources(summary_path: Path = SUMMARY_PATH) -> list[str]:
    """로컬 집계 파일이 있을 때만 '로컬 집계'를 고를 수 있게 한다(배포 환경에는 없음)."""
    return [SYNTHETIC_SOURCE] + ([LOCAL_SOURCE] if summary_path.exists() else [])


def latest_output(result, *stages):
    for s in reversed(result.stages):
        if s.stage in stages and s.status is StageStatus.SUCCEEDED and s.output_text:
            return s.output_text
    return None


def node_states(result) -> list[tuple[str, str, str, str]]:
    """에이전트 흐름 노드별 (이름, 설명, css 상태, 상태 문구). result가 None이면 모두 대기."""
    step = next_step(result) if result is not None else None
    nodes = []
    for name, desc, stages in FLOW:
        related = [s for s in (result.stages if result else ()) if s.stage in stages]
        done = [s for s in related if s.status is StageStatus.SUCCEEDED]
        if any(s.status in (StageStatus.FAILED, StageStatus.BLOCKED) for s in related):
            state, text = "fail", "실패·차단"
        elif step is not None and step.stage in stages:      # 지금 해야 할 단계가 최우선(재검토 회차 포함)
            state, text = "active", "사람 검토 필요" if StageName.HUMAN_REVIEW in stages else "진행 대기"
        elif done:
            state, text = "done", "완료"
        elif related and all(s.status is StageStatus.SKIPPED for s in related):
            state, text = "skip", "생략"
        else:
            state, text = "pending", "대기"
        nodes.append((name, desc, state, text))
    final = ("최종 승인", "사용자 결정", "pending", "대기")
    if result is not None and result.approval is ApprovalStatus.APPROVED:
        final = (*final[:2], "done", "승인됨")
    elif result is not None and result.approval is ApprovalStatus.REJECTED:
        final = (*final[:2], "fail", "반려")
    elif step is not None and step.stage is StageName.FINAL_APPROVAL:
        final = (*final[:2], "active", "승인 대기")
    nodes.append(final)
    return nodes


def flow_html(result) -> str:
    cells = "".join(
        f'<div class="node {state}"><div class="n">{escape(name)}</div><div class="d">{escape(desc)}</div>'
        f'<span class="s">{escape(text)}</span></div>'
        for name, desc, state, text in node_states(result)
    )
    return f'<div class="flow">{cells}</div>'


def kpi_html(label: str, value: str, accent: str, tag: tuple[str, str] | None = None) -> str:
    """KPI 카드. tag=(css, 문구)는 데이터 구분(합성/실제 집계) 표시."""
    badge = f'<span class="t {tag[0]}">{escape(tag[1])}</span>' if tag else ""
    return (f'<div class="kpi {accent}"><div class="l"><span>{escape(label)}</span>{badge}</div>'
            f'<div class="v">{escape(value)}</div></div>')


def start_run(stats, source_kind, scenario, period, request):
    """DRY_RUN 실행. 사용자의 문장은 쓰지 않고, 열거된 요청 이름(request)과 월 범위만 입력에 넣는다."""
    st.session_state.runs = st.session_state.get("runs", 0) + 1
    result = new_run(f"dryrun-{st.session_state.runs}", (to_provenance(stats),))
    sanitized = to_sanitized_monthly(stats, source_kind, period=period, request=request)
    runner = FakeRunner(SCENARIOS[scenario])
    if type(runner) is not FakeRunner:       # 공개 앱은 가짜 러너만 허용한다(구조적 차단)
        raise RuntimeError("공개 앱은 DRY_RUN 가짜 러너만 사용할 수 있습니다.")
    ctx = RunContext(
        sanitized=sanitized, runner=runner,
        budget=CallBudget(DRY_RUN_LIMITS), to_checked=make_checked_text,
    )
    st.session_state.ctx = ctx
    st.session_state.run_info = {"label": stats.label, "start": period[0], "end": period[1], "request": request,
                                 "months": sum(1 for name, _ in sanitized.fields if name.startswith("건수_"))}
    st.session_state.result = advance(result, ctx)


def period_summary(stats, start: str, end: str) -> dict | None:
    """선택 기간의 합계·월평균·최대 월·첫 달 대비 마지막 달 증감(입력 집계값만 사용)."""
    items = [(m, v) for m, v in month_series(stats) if start <= m <= end]
    if not items:
        return None
    total = sum(v for _, v in items)
    first, last = items[0][1], items[-1][1]
    return {"total": total, "avg": total / len(items), "peak": max(items, key=lambda p: p[1]),
            "change": (last - first) / first * 100 if first else None, "n": len(items)}


def trend_chart(stats, period: tuple[str, str]) -> alt.LayerChart | alt.FacetChart:
    """월별 막대(선택 기간 강조) + 3개월 이동평균 선. 날짜 축은 YYYY-MM, 눈금·막대 폭은 개월 수에 맞춘다."""
    series = month_series(stats)
    frame = pd.DataFrame({
        "month": pd.PeriodIndex([m for m, _ in series], freq="M").to_timestamp(),
        "count": [v for _, v in series],
        "scope": ["분석 기간" if period[0] <= m <= period[1] else "기간 밖" for m, _ in series],
    })
    n = len(frame)
    step = 1 if n <= 14 else 2 if n <= 26 else 3 if n <= 40 else 6
    size = 18 if n <= 14 else 13 if n <= 26 else 9 if n <= 40 else 7
    axis = alt.Axis(format="%Y-%m", labelAngle=-45, tickCount={"interval": "month", "step": step},
                    labelOverlap=True, grid=False)
    base = alt.Chart(frame).encode(x=alt.X("month:T", title="접수월", axis=axis))
    bars = base.mark_bar(size=size, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        y=alt.Y("count:Q", title="항공 상담 건수"),
        color=alt.Color("scope:N", title=None, scale=alt.Scale(domain=["분석 기간", "기간 밖"],
                                                              range=["#12B5A6", "#C9D3E2"]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("month:T", title="접수월", format="%Y-%m"), alt.Tooltip("count:Q", title="건수", format=",")],
    )
    average = (base.mark_line(color="#0B1F3A", strokeWidth=2, point=False)
               .transform_window(avg="mean(count)", frame=[-1, 1]).encode(y="avg:Q"))
    return alt.layer(bars, average).properties(height=300).configure_axis(labelFontSize=12, titleFontSize=12)


def show_stats(stats, is_real: bool = False, period: tuple[str, str] | None = None):
    tag = ("real", "실제 집계") if is_real else ("synthetic", "합성")
    ratio = f"{stats.air_rows / stats.total_rows:.2%}" if stats.total_rows else "-"
    cards = (("분모(전체 상담 행)", f"{stats.total_rows:,}", "navy"),
             ("항공여객운송서비스", f"{stats.air_rows:,}", "blue"),
             ("항공 비율", ratio, "teal"),
             ("판정 불가", f"{stats.undetermined_rows:,}", "gray"))
    for column, (label, value, accent) in zip(st.columns(4), cards, strict=True):
        column.markdown(kpi_html(label, value, accent, tag), unsafe_allow_html=True)
    st.caption(f"{stats.label} · {stats.unit} · 전체 기간 {stats.period_start} ~ {stats.period_end}")
    series = month_series(stats)
    if series and period is not None:
        summary = period_summary(stats, *period)
        if summary:
            change = f"{summary['change']:+.1f}%" if summary["change"] is not None else "-"
            sub = (("분석 기간 합계", f"{summary['total']:,}건", "blue"),
                   ("월평균", f"{summary['avg']:,.1f}건", "teal"),
                   ("최대 월", f"{summary['peak'][0]} ({summary['peak'][1]:,})", "navy"),
                   ("첫 달 대비 마지막 달", change, "gray"))
            for column, (label, value, accent) in zip(st.columns(4), sub, strict=True):
                column.markdown(kpi_html(label, value, accent, tag), unsafe_allow_html=True)
            st.caption(f"분석 기간 {period[0]} ~ {period[1]} ({summary['n']}개월)")
        st.markdown('<div class="sec">월별 항공여객운송서비스 상담 건수 (막대: 월별, 선: 3개월 이동평균)</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            st.altair_chart(trend_chart(stats, period), width="stretch")
    for note in stats.notes:
        st.caption(f"• {note}")


def stage_io(result):
    """단계별 입력(자료·기간·요청·이전 단계 출력)과 출력을 펼쳐 볼 수 있게 한다. DRY_RUN은 외부로 아무것도 보내지 않는다."""
    info = st.session_state.get("run_info") or {}
    st.markdown('<div class="sec">단계별 입력·출력</div>', unsafe_allow_html=True)
    previous: list[str] = []
    for s in result.stages:
        if s.status is StageStatus.SKIPPED:
            continue
        label = str(STAGE_LABELS.get(s.stage, s.stage.value)) + (f" · {s.round_no}회차" if s.round_no else "")
        with st.expander(f"{label} — {STATUS_LABELS.get(s.status, s.status.value)}"):
            if s.stage is StageName.HUMAN_REVIEW:
                decision = s.human_decision.value if s.human_decision else "-"
                st.markdown(f"**입력:** 이전 단계 출력({', '.join(previous) or '없음'}) · **사람 결정:** {decision}")
            else:
                st.markdown(
                    f"**입력:** 자료 {info.get('label', '-')} · 분석 기간 {info.get('start', '-')}~{info.get('end', '-')}"
                    f" · 요청 ‘{info.get('request', '-')}’ · 월별 집계 {info.get('months', 0)}개(원문·개인정보 없음)"
                    f" · 이전 단계 출력: {', '.join(previous) or '없음'}")
                st.markdown("**출력:**")
                st.markdown(s.output_text or "출력 없음")
            if s.status is StageStatus.SUCCEEDED:
                previous.append(label)


def show_progress(result, live_mode: bool = False):
    st.markdown('<div class="sec">AI 에이전트 진행 상황</div>', unsafe_allow_html=True)
    info = st.session_state.get("run_info")
    if info:
        st.caption(f"실행한 요청: ‘{info['request']}’ · 자료 {info['label']} · 분석 기간 {info['start']}~{info['end']}")
    st.markdown(flow_html(result), unsafe_allow_html=True)
    note = "LIVE: 실제 API 호출" if live_mode else "DRY_RUN: 외부 호출 0회"
    st.progress(min(result.calls_used / 9, 1.0), text=f"모델 호출 {result.calls_used} / 9회 ({note})")
    rows = []
    for s in result.stages:
        tag = s.verdict or s.human_decision
        rows.append({"단계": STAGE_LABELS.get(s.stage, s.stage.value), "회차": s.round_no,
                     "상태": STATUS_LABELS.get(s.status, s.status.value),
                     "판정/결정": tag.value if tag else "", "모델 호출": s.calls_used})
    with st.expander("단계별 기록 보기"):
        st.dataframe(rows, width="stretch", hide_index=True)


def human_review_form(result, ctx):
    st.subheader("Human Review")
    review = latest_output(result, StageName.REVIEWER, StageName.REVIEWER_RECHECK)
    with st.expander("Reviewer 검수 결과", expanded=True):
        st.markdown(review or "검수 결과 없음")
    with st.form("human_review"):
        label = st.radio("결정", list(DECISIONS))
        notes = st.text_area("수정 요청 메모(개인정보·상담 원문 입력 금지)", "")
        if st.form_submit_button("제출"):
            try:
                st.session_state.result = advance(
                    result, ctx, HumanInput(DECISIONS[label], notes.strip() or None))
                st.rerun()
            except ValueError as error:
                st.error(str(error))


def final_approval(result):
    st.subheader("최종 승인")
    st.info("승인 전 결과는 DRAFT입니다. 사용자가 직접 승인해야 확정됩니다.")
    left, right = st.columns(2)
    try:
        if left.button("최종 승인", type="primary"):
            st.session_state.result = result.approve_final()
            st.rerun()
        if right.button("반려"):
            st.session_state.result = result.reject()
            st.rerun()
    except ValueError as error:
        st.error(str(error))


def _secret(name: str):
    """서버 비밀값을 읽는다(화면·상태 파일·로그에 쓰지 않는다). 설정이 없으면 None."""
    try:
        return st.secrets.get(name)
    except Exception:  # noqa: BLE001 - 비밀 파일이 없는 환경(공개 DRY_RUN)은 LIVE 비활성
        return None


def _host_header() -> str | None:
    try:
        return st.context.headers.get("Host")
    except Exception:  # noqa: BLE001 - 헤더를 읽을 수 없으면 로컬로 인정하지 않는다
        return None


def _viewer_email() -> str | None:
    try:
        value = st.user.get("email")
        return value if isinstance(value, str) else None
    except Exception:  # noqa: BLE001 - 로그인 정보가 없으면 열람자로 인정하지 않는다
        return None


def _live_allowed() -> bool:
    """로컬(loopback) 실행이거나 플랫폼 로그인으로 확인된 열람자일 때만 LIVE 접근을 허용한다."""
    return (live.local_access(st.get_option("server.address"), _host_header(), ROOT)
            or live.viewer_access(_viewer_email()))


def _local_gate() -> live.LocalGate:
    """허용된 환경에서만 토큰을 발급하는 게이트(공개 익명 방문자에게는 항상 닫힘)."""
    return _gate_cache(_live_allowed())


@st.cache_resource
def _gate_cache(allowed: bool) -> live.LocalGate:
    return live.LocalGate(allowed)


@st.cache_resource
def _live_store() -> live.LiveStore:
    return live.LiveStore()


def live_usage_table(view) -> None:
    rows = [{"단계": u["stage"], "요청": u["requests"], "입력 토큰": u["input_tokens"],
             "출력 토큰": u["output_tokens"], "구분": "실패·최악값 계상" if u.get("estimated") else "실제 사용량"}
            for u in view.usage]
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(f"실제 사용량 기반 계산 비용 ${view.spent:.6f} (단가 입력 {view.meta['prices'][0]} / 출력 "
               f"{view.meta['prices'][1]} USD/100만 토큰). 사전 추정치와 다르며 청구액은 OpenAI 대시보드에서 확인하세요.")


def live_panel(stats, period: tuple[str, str], request_text: str, is_real: bool) -> None:
    """LIVE 영역. 인증·승인·중복 방지는 모두 voc.live_service가 서버 측에서 강제한다."""
    st.markdown('<div class="sec">LIVE — 실제 API 실행(로그인 필요, 합성 데이터 한정)</div>', unsafe_allow_html=True)
    if not _live_allowed():
        st.warning("LIVE는 로그인으로 확인된 열람자(앱을 ‘지정한 사람만 열람’으로 설정한 경우)이거나 이 PC에서 직접 실행한 "
                   "로컬 화면에서만 열립니다. 무단 유료 호출을 막기 위해 공개 익명 접속에서는 비활성입니다. DRY_RUN을 사용하세요.")
        return
    try:
        settings = live.load_settings(_secret)
    except live.LiveError as error:
        st.error(str(error))
        return
    if settings is None:
        st.info("LIVE는 서버 설정(OPENAI_API_KEY)이 없어 비활성입니다.")
        return
    gate, store = _local_gate(), _live_store()
    token = st.session_state.get("live_token")
    if not gate.valid(token):
        token = gate.issue()
        if token is None:
            st.warning("이 환경에서는 LIVE 접근 토큰을 발급할 수 없습니다.")
            return
        st.session_state.live_token = token
    if is_real:
        st.error("실제 집계 자료의 LIVE는 이용 조건 확인과 별도 승인 전까지 비활성입니다. 합성 자료를 선택하세요.")
        return
    try:
        view = live.load_view(store, gate, token)
    except live.LiveError as error:
        st.error(str(error))
        return
    if view is None:
        live_start_form(stats, period, request_text, settings, gate, store, token)
        return
    live_record(view, settings, gate, store, token)


def live_start_form(stats, period, request_text, settings, gate, store, token) -> None:
    request, message = parse_request(request_text)
    if request is None:
        st.error(message)
        return
    try:
        plan = live.build_plan(stats, period, request, (settings.price_in, settings.price_out))
    except live.LiveError as error:
        st.error(str(error))
        return
    st.markdown("**실행 직전 승인 요청서**")
    st.code(plan.text(), language=None, wrap_lines=True)
    for label, passed in plan.checks:
        st.markdown(f"{'✅' if passed else '❌'} {label}")
    st.info(f"승인하려면 승인 코드 **{plan.code}** 를 입력하세요. (자료·기간·목적·모델·단가가 바뀌면 코드도 바뀝니다.) "
            "한 번 시작하면 기록이 보존되며 새 실행은 시작되지 않습니다.")
    with st.form("live_start"):
        typed = st.text_input("승인 코드")
        if st.form_submit_button("승인하고 LIVE 시작(유료 호출)", disabled=not plan.ok):
            try:
                with st.spinner("실제 API 호출 중입니다…(Collector → Reviewer, 최대 4회)"):
                    live.start_run(store, gate, token, settings, stats, period, request, typed)
            except live.LiveError as error:
                st.error(str(error))
            else:
                st.rerun()


def live_record(view, settings, gate, store, token) -> None:
    result = view.result
    st.session_state.run_info = {
        "label": view.meta["label"], "start": view.meta["period"][0], "end": view.meta["period"][1],
        "request": view.meta["request"], "months": 0}
    st.caption(f"LIVE 실행 기록 · 모델 {view.meta['model']} · 시작 {view.meta['started']} · "
               "합성 데이터 · 실제 VOC 분석 결과 아님")
    show_progress(result, live_mode=True)
    live_usage_table(view)
    step = next_step(result)
    if view.interrupted:
        st.error("이전 호출이 도중에 끊겨 과금 여부를 알 수 없습니다. 더 호출하지 않습니다.")
    elif result.approval is ApprovalStatus.APPROVED:
        st.success("최종 승인됨 (LIVE · 합성 데이터)")
    elif result.approval is ApprovalStatus.REJECTED:
        st.warning("사용자가 반려했습니다.")
    elif step.stop_reason is not None:
        st.error(f"중단: {ERROR_MESSAGES[step.stop_reason]}")
    elif step.stage is StageName.HUMAN_REVIEW:
        st.subheader("Human Review")
        review = latest_output(result, StageName.REVIEWER, StageName.REVIEWER_RECHECK)
        with st.expander("Reviewer 검수 결과", expanded=True):
            st.markdown(review or "검수 결과 없음")
        left = 9 - result.calls_used
        st.warning(f"제출하면 다음 단계의 유료 호출이 실행됩니다(남은 호출 최대 {left}회, 누적 계산 비용 ${view.spent:.6f}).")
        with st.form("live_human"):
            choice = st.radio("결정", list(DECISIONS))
            notes = st.text_area("수정 요청 메모(개인정보·상담 원문 입력 금지)", "")
            if st.form_submit_button("제출하고 다음 단계 실행(유료 호출)"):
                try:
                    with st.spinner("실제 API 호출 중입니다…"):
                        live.continue_run(store, gate, token, settings, view.stage_count,
                                          HumanInput(DECISIONS[choice], notes.strip() or None))
                except (live.LiveError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.rerun()
    elif step.stage is StageName.FINAL_APPROVAL:
        st.subheader("최종 승인")
        st.info("승인 전 결과는 DRAFT입니다. 사용자가 직접 결정합니다(추가 호출 없음).")
        approve_col, reject_col = st.columns(2)
        decision = (True if approve_col.button("최종 승인", type="primary") else
                    False if reject_col.button("반려") else None)
        if decision is not None:
            try:
                live.finalize(store, gate, token, decision)
            except live.LiveError as error:
                st.error(str(error))
            else:
                st.rerun()
    report = latest_output(result, StageName.WRITER_REVISION, StageName.WRITER_DRAFT)
    if report:
        st.subheader(f"보고서 ({result.approval.value})")
        with st.container(border=True):
            st.markdown(report)
        st.download_button("보고서 내려받기(.md)",
                           f"<!-- LIVE 합성 데이터 · 실제 분석 결과 아님 -->\n\n{report}\n",
                           file_name="voc_live_report.md")
    summary = latest_output(result, StageName.MANAGER)
    if summary:
        with st.expander("Manager 종합"):
            st.markdown(summary)
    stage_io(result)


def topbar(is_real: bool, live_mode: bool = False) -> None:
    """화이트 헤더: 왼쪽 네이비 제목, 오른쪽 상태 배지(실행 모드·데이터 구분·안전 표시)."""
    mode_badge = ('<span class="badge real">LIVE 모드(로그인 필요)</span>' if live_mode
                  else '<span class="badge mode">DRY_RUN 모드</span>')
    safe_badge = ('<span class="badge safe">합성 데이터만 · 승인 후 유료 호출</span>' if live_mode
                  else '<span class="badge safe">가짜 모델 · 외부 호출 0회</span>')
    data_badge = ('<span class="badge real">로컬 실제 집계(상담 원문 아님)</span>' if is_real
                  else '<span class="badge synthetic">합성 데이터(실제 수치 아님)</span>')
    st.markdown(
        '<div class="topbar"><div><h1>항공여객운송서비스 VOC 개선 대시보드</h1>'
        "<p>5개 AI 에이전트가 VOC를 정리·분석·작성·검수·종합하고, 사람이 검토·승인합니다.</p></div>"
        f'<div class="badges">{mode_badge}{data_badge}{safe_badge}</div></div>',
        unsafe_allow_html=True,
    )
    if is_real:
        st.markdown('<div class="datanote real"><b>로컬 실제 집계</b> — 이 PC에서 구조화 열만 집계한 수치입니다. '
                    "외부로 전송되지 않으며, DRY_RUN에서는 가짜 모델이 처리합니다.</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="datanote synthetic"><b>합성 데이터</b> — 시연용 가상 수치이며 실제 VOC 현황이 '
                    "아닙니다.</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="VOC 개선 CX 대시보드", page_icon="✈️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown('<div class="brand">VOC 에이전트 콘솔</div>'
                        '<div class="brand-sub">1372 항공여객운송서비스</div>', unsafe_allow_html=True)
    source = st.sidebar.radio("통계 출처", available_sources())
    mode = st.sidebar.radio("실행 모드", ["DRY_RUN (공개 시연)", "LIVE (로그인 필요)"])
    live_mode = mode.startswith("LIVE")
    scenario = st.sidebar.selectbox("Reviewer 시나리오(DRY_RUN)", list(SCENARIOS), disabled=live_mode)
    st.sidebar.caption("DRY_RUN은 가짜 모델만 쓰며 외부 호출이 없습니다. LIVE는 로그인된 열람자 또는 로컬 실행에서만 열리며 "
                       "합성 데이터로만 실행되며, 시작 전에 전송 범위·비용 승인이 필요합니다.")
    if len(available_sources()) == 1:
        st.sidebar.caption("로컬 집계 파일이 없어 합성 예시만 사용할 수 있습니다.")
    is_real = source == LOCAL_SOURCE
    topbar(is_real, live_mode)

    all_stats = load_stats(source)
    if not all_stats:
        st.error("집계 파일이 없습니다. 합성 예시를 선택하세요.")
        return
    kind = SourceKind.REAL if is_real else SourceKind.SYNTHETIC
    with st.container(border=True):
        st.markdown('<div class="sec" style="margin-top:0">분석 요청</div>', unsafe_allow_html=True)
        pick_data, pick_start, pick_end = st.columns([2, 1, 1])
        label = pick_data.selectbox("자료", [s.label for s in all_stats])   # 위젯은 한 번만 만든다(중복 ID 방지)
        stats = next(s for s in all_stats if s.label == label)
        months = [m for m, _ in month_series(stats)]
        start = pick_start.selectbox("분석 시작월", months, index=0)
        end = pick_end.selectbox("분석 종료월", months, index=len(months) - 1)
        request_text = st.text_input("자연어 요청", value="월별 추세를 분석해 주세요", max_chars=MAX_REQUEST_CHARS,
                                     placeholder="예: 급증 구간을 확인해 주세요")
        st.caption(SUPPORTED_HELP + " · 입력 문장은 AI 모델·외부 API로 보내지 않고, 목적 이름만 숫자·날짜와 함께 전달됩니다.")
        clicked = st.button("분석 시작", type="primary", disabled=live_mode)
    period_ok = start <= end
    if not period_ok:
        st.error("분석 시작월은 종료월보다 늦을 수 없습니다.")
    show_stats(stats, is_real, (start, end) if period_ok else None)
    st.caption("통계는 자료별로 분리되어 있으며 서로 합산하지 않습니다.")

    if live_mode:
        if period_ok:
            live_panel(stats, (start, end), request_text, is_real)
        return

    if clicked and period_ok:
        request, message = parse_request(request_text)
        if request is None:
            st.error(message)            # 지원하지 않는 자유 입력은 거부하고 실행하지 않는다
        else:
            start_run(stats, kind, scenario, (start, end), request)
            st.session_state.run_basis = (source, label, start, end)
            st.rerun()

    result, ctx = st.session_state.get("result"), st.session_state.get("ctx")
    if result is not None and st.session_state.get("run_basis") != (source, label, start, end):
        st.warning("현재 선택한 자료·기간과 다른 조건으로 실행한 결과입니다. 선택한 조건으로 보려면 ‘분석 시작’을 다시 누르세요.")
    if result is None:
        st.markdown('<div class="sec">AI 에이전트 진행 상황</div>', unsafe_allow_html=True)
        st.markdown(flow_html(None), unsafe_allow_html=True)
        st.caption("‘분석 시작’을 누르면 Collector부터 Reviewer까지 자동 진행하고, Human Review에서 멈춥니다.")
        return
    st.divider()
    show_progress(result)
    step = next_step(result)
    if result.approval is ApprovalStatus.APPROVED:
        st.success("최종 승인됨 (DRY_RUN 모드 · 가짜 모델이 만든 시연 보고서)")
    elif result.approval is ApprovalStatus.REJECTED:
        st.warning("사용자가 반려했습니다.")
    elif step.stop_reason is not None:
        st.error(f"중단: {ERROR_MESSAGES[step.stop_reason]}")
    elif step.stage is StageName.HUMAN_REVIEW:
        human_review_form(result, ctx)
    elif step.stage is StageName.FINAL_APPROVAL:
        final_approval(result)

    report = latest_output(result, StageName.WRITER_REVISION, StageName.WRITER_DRAFT)
    if report:
        st.subheader(f"보고서 ({result.approval.value})")
        with st.container(border=True):
            st.markdown(report)
        st.download_button("보고서 내려받기(.md)", report, file_name="voc_dryrun_report.md")
    summary = latest_output(result, StageName.MANAGER)
    if summary:
        with st.expander("Manager 종합"):
            st.markdown(summary)
    stage_io(result)


if __name__ == "__main__":
    main()
