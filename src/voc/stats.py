"""자료별 항공 상담 집계와 계약 타입 변환."""

import calendar
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .contracts import (
    DATASET_ID_PATTERN,
    INTERPRETATION_RULE,
    REQUEST_SECTIONS,
    DataProvenance,
    SanitizedInput,
    SourceKind,
)
from .loader import AIR_ITEM, DATASETS, consultation_chunks

UNIT = "개별 상담 기록"
FILTER_RULE = "ITEM_NAME == '항공여객운송서비스'(코드 자료는 검증 대응표)"
NOTES = (
    "자료별 분모는 독립적이며 자료 간 사건 중복은 미검증입니다.",
    "월·품목 집계표와 표준답변은 집계에서 제외했습니다.",
    "사건번호 중복 행도 건수와 분모에 포함합니다.",
)


@dataclass(frozen=True)
class DatasetStats:
    dataset_id: str
    label: str
    unit: str
    total_rows: int
    air_rows: int
    undetermined_rows: int
    period_start: str | None
    period_end: str | None
    monthly: tuple[tuple[str, int], ...]
    unique_air_ids: int | None
    duplicate_air_rows: int | None
    notes: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.dataset_id, str) or not DATASET_ID_PATTERN.fullmatch(self.dataset_id):
            raise ValueError("dataset_id 형식이 올바르지 않습니다.")


def compute_stats(chunks, dataset_id: str, label: str, *, unit=UNIT, notes=()) -> DatasetStats:
    """기간은 전체 유효 접수일, 월별 집계와 사건번호 지표는 항공 행 기준이다."""
    total_rows = air_rows = undetermined_rows = id_count = 0
    missing_air_dates = missing_air_ids = 0
    start = end = None
    unique_ids = set()
    monthly = Counter()
    for chunk in chunks:
        total_rows += len(chunk)
        air = chunk["is_air"].fillna(False)
        air_count = int(air.sum())
        air_rows += air_count
        undetermined_rows += int(chunk["is_air"].isna().sum())
        ids = chunk.loc[air, "ACCIDENT_NO"].dropna()
        unique_ids.update(ids)
        id_count += len(ids)
        missing_air_ids += air_count - len(ids)
        date_text = chunk["RCPT_YMD"].str.replace(r"[-/.]", "", regex=True)
        date_text = date_text.where(date_text.str.fullmatch(r"[0-9]{8}", na=False))
        dates = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            first, last = (date.strftime("%Y-%m-%d") for date in (valid_dates.min(), valid_dates.max()))
            start = first if start is None else min(start, first)
            end = last if end is None else max(end, last)
        air_dates = dates[air].dropna()
        missing_air_dates += air_count - len(air_dates)
        monthly.update(air_dates.dt.strftime("%Y-%m").value_counts().to_dict())
    notes = tuple(notes)
    if missing_air_dates:
        notes += (f"접수일이 없거나 유효하지 않은 항공 {missing_air_dates:,}행은 월별 집계에서 제외했습니다.",)
    if missing_air_ids:
        notes += (f"사건번호가 없는 항공 {missing_air_ids:,}행은 중복 지표에서 제외했습니다.",)
    return DatasetStats(
        dataset_id, label, unit, total_rows, air_rows, undetermined_rows, start, end,
        tuple(sorted(monthly.items())), len(unique_ids), id_count - len(unique_ids), notes,
    )


def compute_directory(raw, chunksize=50_000) -> tuple[DatasetStats, ...]:
    code_names = {}
    results = []
    for dataset_id, label in DATASETS:
        paths = list(Path(raw).glob(f"*{label}*.csv"))
        if len(paths) != 1:
            raise ValueError(f"자료별 CSV가 정확히 하나여야 합니다 ({label}).")
        notes = NOTES
        if dataset_id == "receipt":
            notes += ("접수내역은 내보내기 상한으로 잘렸을 가능성이 있습니다.",)
        code_only = dataset_id == "hazard"
        if code_only:
            notes += ("접수내역·온라인 자료에서 한 품목명에만 대응하는 코드를 사용했습니다.",)
        chunks = consultation_chunks(paths[0], code_names, code_only=code_only, chunksize=chunksize)
        results.append(compute_stats(chunks, dataset_id, label, notes=notes))
    return tuple(results)


def save_summary(stats_list, path):
    payload = [asdict(stats) for stats in stats_list]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_summary(path) -> tuple[DatasetStats, ...]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    for entry in entries:
        entry["monthly"] = tuple(tuple(pair) for pair in entry["monthly"])
        entry["notes"] = tuple(entry["notes"])
    return tuple(DatasetStats(**entry) for entry in entries)


def to_sanitized(stats: DatasetStats, source_kind: SourceKind) -> SanitizedInput:
    if stats.period_end is None:
        raise ValueError("유효한 접수 기간이 있어야 합니다.")
    return SanitizedInput(
        source_kind=source_kind, dataset_id=stats.dataset_id, schema_version="voc_aggregate_v1",
        fields=(
            ("품목", AIR_ITEM), ("자료유형", "개별 상담 집계"), ("기간", stats.period_end[:7]),
            ("건수", stats.air_rows),
            ("비율", stats.air_rows / stats.total_rows if stats.total_rows else 0.0),
        ),
    )


DATASET_SCOPES = {"receipt": "전체 상담 접수", "online": "온라인 상거래 상담", "hazard": "위해위험 상담"}


def _month_range(start: str, end: str) -> list[str]:
    """'YYYY-MM' 시작~끝(포함)의 모든 달."""
    year, month = int(start[:4]), int(start[5:7])
    months = []
    while (year, month) <= (int(end[:4]), int(end[5:7])):
        months.append(f"{year}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def to_sanitized_monthly(stats: DatasetStats, source_kind: SourceKind) -> SanitizedInput:
    """월별 변화 분석용 입력(voc_monthly_v1). 집계값·열거값·기간만 담으며 원문·제목·자유 문자열은 불가능하다.

    기간 안에서 항공 행이 없는 달은 0으로 명시하고(필드 누락과 구분), 자료 한계는 열거값으로 전달한다.
    """
    if stats.period_start is None or stats.period_end is None:
        raise ValueError("유효한 접수 기간이 있어야 합니다.")
    scope = DATASET_SCOPES.get(stats.dataset_id)
    if scope is None:
        raise ValueError("월별 분석 대상이 아닌 자료입니다.")
    counts = dict(stats.monthly)
    if sum(counts.values()) != stats.air_rows or stats.air_rows > stats.total_rows:
        raise ValueError("월별 합계가 항공 건수와 다르거나 항공 건수가 전체 건수보다 큽니다.")
    first = min([stats.period_start[:7], *counts])
    last = max([stats.period_end[:7], *counts])
    year, month, day = int(stats.period_end[:4]), int(stats.period_end[5:7]), int(stats.period_end[8:10])
    partial = "예" if day != calendar.monthrange(year, month)[1] else "아니오"
    truncated = "예" if any("상한" in note for note in stats.notes) else "미확인"
    ratio = round(stats.air_rows / stats.total_rows, 6) if stats.total_rows else 0.0
    fields = [
        ("품목", AIR_ITEM), ("자료유형", "개별 상담 집계"), ("자료범위", scope),
        ("기간", stats.period_end[:7]), ("기간_시작", stats.period_start), ("기간_종료", stats.period_end),
        ("전체건수", stats.total_rows), ("건수", stats.air_rows), ("판정불가건수", stats.undetermined_rows),
        ("비율", ratio), ("마지막월_부분집계", partial), ("행수_상한_절단_가능성", truncated),
        ("요청_절구성", REQUEST_SECTIONS), ("해석_원칙", INTERPRETATION_RULE),
    ]
    fields += [(f"건수_{m}", counts.get(m, 0)) for m in _month_range(first, last)]
    return SanitizedInput(source_kind=source_kind, dataset_id=stats.dataset_id,
                          schema_version="voc_monthly_v1", fields=tuple(fields))


def to_provenance(stats: DatasetStats) -> DataProvenance:
    period = f"{stats.period_start}~{stats.period_end}" if stats.period_end is not None else None
    return DataProvenance(
        dataset=stats.dataset_id, period=period, filter_rule=FILTER_RULE,
        denominator=stats.total_rows, denominator_basis="개별 상담 기록 행 수",
        dedup_basis="ACCIDENT_NO", limitations=stats.notes,
    )


def synthetic_stats() -> tuple[DatasetStats, ...]:
    notes = ("합성 예시이며 실제 수치가 아닙니다.",)
    return (
        DatasetStats("합성_A", "합성 A", UNIT, 10, 2, 1, "2020-01-01", "2020-02-28",
                     (("2020-01", 1), ("2020-02", 1)), 2, 0, notes),
        DatasetStats("합성_B", "합성 B", UNIT, 20, 3, 0, "2020-03-01", "2020-03-31",
                     (("2020-03", 3),), 2, 1, notes),
    )


def write_markdown(stats_list, path):
    lines = ["# 항공여객운송서비스 개별 상담 집계", ""]
    for stats in stats_list:
        provenance = to_provenance(stats)
        lines.extend([
            f"## {stats.label}", "",
            f"- 접수 기간 {provenance.period or '산출 불가'}",
            f"- 필터 `{provenance.filter_rule}` (품목명 공백 제거 후 정확 일치)",
            f"- 항공 {stats.air_rows:,}행 / 전체 {stats.total_rows:,}행 ({provenance.denominator_basis})",
            f"- 판정 불가 {stats.undetermined_rows:,}행",
            f"- 항공 고유 사건번호 {stats.unique_air_ids}개 / 중복 {stats.duplicate_air_rows}행",
            *(f"- {note}" for note in stats.notes),
            "", "| 접수월 | 항공 행 수 |", "| --- | ---: |",
            *(f"| {month} | {count:,} |" for month, count in stats.monthly),
            "",
        ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
