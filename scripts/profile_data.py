"""CSV 원문 열을 제외하고 자료별 데이터 적합성 지표를 집계한다."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BLOCKED_COLUMNS = frozenset({
    "ACCIDENT_TITLE", "ACCIDENT_CONTENT", "ANS_CONTENT",
    "사건제목", "사건내용", "답변내용", "질문", "답변", "질문본문", "답변본문",
})
STRUCTURED_COLUMNS = frozenset({
    "ACCIDENT_NO", "RCPT_YMD", "ITEM_CODE", "ITEM_NAME",
    "사건ID", "품목코드", "품목명", "연도", "월", "상담건수",
})
HEADER_CODE = re.compile(r"\(([A-Z][A-Z0-9_]*)\)\s*$")
AIR_ITEM = "항공여객운송서비스"
AMBIGUOUS_ITEM = "여객운송서비스"
UNDETERMINED_REASONS = {
    "missing_code": "품목코드 결측",
    "unregistered_code": "미등록 코드",
    "conflicting_code": "코드 충돌",
    "missing_name": "품목명 결측",
    "no_subcategory": "세분류 없음",
}
REPORT_NOTES = [
    "자료별 분모는 독립적이며 합산하지 않습니다.",
    "자료 간 사건 중복은 미검증입니다.",
    "공정위 상담 3종은 로컬 분석 대상이며 외부 API 전송은 미승인입니다.",
]
PROJECT = Path(__file__).resolve().parents[1]


def column_key(header):
    match = HEADER_CODE.search(header)
    return match.group(1) if match else header.strip()


def read_header(path):
    # 버퍼 없이 첫 줄의 바이트만 요청해 헤더 검사 범위를 제한한다.
    with path.open("rb", buffering=0) as stream:
        line = bytearray()
        while (byte := stream.read(1)) and byte != b"\n":
            line.extend(byte)
    return next(csv.reader([line.decode("cp949").rstrip("\r")]))


def read_chunks(path, columns, chunksize):
    for column in columns:
        key = column_key(column)
        label = column.partition("(")[0].strip()
        if key in BLOCKED_COLUMNS or label in BLOCKED_COLUMNS:
            raise ValueError("차단된 열이 로딩 목록에 포함되었습니다.")
        if key not in STRUCTURED_COLUMNS:
            raise ValueError("허용 목록에 없는 열이 로딩 목록에 포함되었습니다.")
    return pd.read_csv(
        path, encoding="cp949", usecols=list(columns), chunksize=chunksize,
        dtype="string", keep_default_na=False, index_col=False,
    )


def project_chunks(path, fields, chunksize):
    selected = {}
    for column in read_header(path):
        key = column_key(column)
        if key in fields:
            if key in selected:
                raise ValueError("구조화 헤더가 중복되었습니다.")
            selected[key] = column
    if selected.keys() != fields.keys():
        raise ValueError("필수 구조화 헤더가 없습니다.")
    rename = {column: fields[key] for key, column in selected.items()}
    for chunk in read_chunks(path, rename, chunksize):
        chunk = chunk.rename(columns=rename)
        for field in chunk:
            values = chunk[field].str.strip()
            chunk[field] = values.mask(values.eq(""))
        yield chunk


def update_period(period, dates):
    dates = dates.dropna()
    if dates.empty:
        return
    start, end = dates.min(), dates.max()
    period[0] = start if period[0] is None else min(period[0], start)
    period[1] = end if period[1] is None else max(period[1], end)


def profile(path, fields, unit, notes, code_names, chunksize, *,
            collect_codes=False, coarse_items=False):
    has_date = "RCPT_YMD" in fields.values()
    monthly = "year" in fields.values()
    has_id = "ACCIDENT_NO" in fields.values()
    row_count = air_rows = invalid_dates = all_volume = air_volume = invalid_volume = 0
    missing = Counter(dict.fromkeys(fields.values(), 0))
    unknown = dict.fromkeys(UNDETERMINED_REASONS, 0)
    ids = set()
    id_count = 0
    all_period = [None, None]
    air_period = [None, None]
    code_only = "ITEM_NAME" not in fields.values()
    lookup: dict = {}
    conflicting_codes: set = set()
    if code_only:
        lookup = {code: next(iter(names)) for code, names in code_names.items()
                  if len(names) == 1}
        conflicting_codes = {code for code, names in code_names.items() if len(names) > 1}
    for chunk in project_chunks(path, fields, chunksize):
        row_count += len(chunk)
        missing.update(chunk.isna().sum().to_dict())
        if code_only:
            codes = chunk["ITEM_CODE"]
            names = codes.map(lookup).astype("string")
            conflicts = codes.isin(conflicting_codes)
            unknown["missing_code"] += int(codes.isna().sum())
            unknown["conflicting_code"] += int(conflicts.sum())
            unknown["unregistered_code"] += int((codes.notna() & names.isna() & ~conflicts).sum())
        else:
            names = chunk["ITEM_NAME"]
            unknown["missing_name"] += int(names.isna().sum())
            if collect_codes:
                pairs = chunk[["ITEM_CODE", "ITEM_NAME"]].dropna().drop_duplicates()
                for code, name in pairs.itertuples(index=False, name=None):
                    code_names[code].add(name)
        names = names.str.replace(r"\s+", "", regex=True)
        air = names.eq(AIR_ITEM).fillna(False)
        if coarse_items:
            unknown["no_subcategory"] += int(names.eq(AMBIGUOUS_ITEM).sum())
            # 이 자료는 항공 건수를 산출하지 않습니다.
            air[:] = False
        air_rows += int(air.sum())
        if has_id:
            values = chunk["ACCIDENT_NO"].dropna()
            id_count += len(values)
            ids.update(values)
        if has_date or monthly:
            if has_date:
                source = chunk["RCPT_YMD"]
                date_text = source.str.replace(r"[-/.]", "", regex=True)
                date_text = date_text.where(date_text.str.fullmatch(r"[0-9]{8}", na=False))
                date_format = "%Y%m%d"
            else:
                year, month = chunk["year"], chunk["month"]
                source = year + month
                date_text = year + month.str.zfill(2)
                valid_month = year.str.fullmatch(r"[0-9]{4}", na=False)
                valid_month &= month.str.fullmatch(r"[0-9]{1,2}", na=False)
                date_text = date_text.where(valid_month)
                date_format = "%Y%m"
            dates = pd.to_datetime(date_text, format=date_format, errors="coerce")
            invalid_dates += int((source.notna() & dates.isna()).sum())
            dates = dates.dt.strftime("%Y-%m" if monthly else "%Y-%m-%d")
            update_period(all_period, dates)
            update_period(air_period, dates[air])
        if monthly:
            counts = chunk["consultation_count"].str.replace(",", "", regex=False)
            valid_counts = counts.str.fullmatch(r"[0-9]+", na=False)
            invalid_volume += int((counts.notna() & ~valid_counts).sum())
            all_volume += sum(map(int, counts[valid_counts]))
            air_volume += sum(map(int, counts[valid_counts & air]))

    undetermined_rows = sum(unknown.values())
    return {
        "file": path.name, "unit": unit, "total_rows": row_count,
        "air_exact_rows": None if coarse_items else air_rows,
        "ambiguous_rows": unknown["no_subcategory"],
        "undetermined_rows": undetermined_rows,
        "undetermined_reasons": unknown,
        "non_air_rows": row_count - undetermined_rows - air_rows,
        "period_basis": "집계 연월" if monthly else "접수일" if has_date else None,
        "total_period": all_period, "air_exact_period": air_period,
        "unique_accident_ids": len(ids) if has_id else None,
        "duplicate_accident_rows": id_count - len(ids) if has_id else None,
        "missing": missing, "invalid_dates": invalid_dates if has_date or monthly else None,
        "consultation_count_sums": {
            "total": all_volume, "air_exact": air_volume, "invalid": invalid_volume,
        } if monthly else None,
        "notes": notes,
    }


def profile_consultations(path, code_names, chunksize, *, code_only=False, receipt_export=False):
    fields = {key: key for key in ("ACCIDENT_NO", "RCPT_YMD", "ITEM_CODE")}
    notes = ["사건번호 중복 행도 분모에 포함합니다."]
    if code_only:
        notes.append("소비자상담 접수내역과 온라인 상거래 소비자상담 내역에서 한 품목명에만 대응하는 코드를 사용합니다.")
    else:
        fields["ITEM_NAME"] = "ITEM_NAME"
    if receipt_export:
        notes.append("행 수가 내보내기 상한으로 잘렸을 가능성이 있고, 접수 기간이 파일명의 기준일보다 일찍 끝납니다.")
    return profile(path, fields, "개별 상담 기록", notes, code_names, chunksize,
                   collect_codes=not code_only)


def profile_monthly(path, code_names, chunksize):
    fields = {"연도": "year", "월": "month", "품목명": "ITEM_NAME", "상담건수": "consultation_count"}
    notes = ["분모는 월·품목별 집계 행 수입니다. 상담건수는 유효한 값만 별도로 합산합니다."]
    notes.append("집계 기간은 2026-09 한 달뿐입니다.")
    return profile(path, fields, "월·품목별 집계표", notes, code_names, chunksize)


def profile_standard_answers(path, code_names, chunksize):
    return profile(path, {"품목명": "ITEM_NAME"}, "표준답변 항목",
                   ["접수일과 사건번호 열이 없습니다."], code_names, chunksize)


def profile_relief(path, code_names, chunksize):
    fields = {"사건ID": "ACCIDENT_NO", "품목코드": "ITEM_CODE", "품목명": "ITEM_NAME"}
    notes = ["갱신일시는 접수 기간에 사용하지 않습니다. 사건ID로 고유 수와 중복 행을 계산합니다."]
    notes.append("일반 ‘여객운송서비스’에는 항공을 가릴 세분류가 없습니다.")
    return profile(path, fields, "피해구제 사건 분류 기록", notes, code_names, chunksize,
                   coarse_items=True)


def profile_directory(raw, chunksize):
    # 품목코드 전용 자료는 코드와 이름이 함께 있는 자료를 처리한 뒤 집계한다.
    sources = [
        ("소비자상담 접수내역", profile_consultations, {"receipt_export": True}),
        ("온라인 상거래 소비자상담 내역", profile_consultations, {}),
        ("피해구제 사건 유형 분류", profile_relief, {}),
        ("소비자상담 표준답변", profile_standard_answers, {}),
        ("품목별 상담 현황", profile_monthly, {}),
        ("위해위험 소비자상담 내역", profile_consultations, {"code_only": True}),
    ]
    code_names = defaultdict(set)
    reports = []
    for label, adapter, options in sources:
        paths = list(raw.glob(f"*{label}*.csv"))
        if len(paths) != 1:
            raise ValueError(f"자료별 CSV가 정확히 하나여야 합니다 ({label}).")
        reports.append(adapter(paths[0], code_names, chunksize, **options))
    return reports


def period_text(period):
    start, end = period
    return f"{start} ~ {end}" if start is not None else "산출 불가"


def air_rows_text(report):
    rows = report["air_exact_rows"]
    return "항공 판정 불가 자료" if rows is None else f"{rows:,}행"


def write_reports(output, reports):
    lines = [
        "# 데이터 적합성 게이트", "",
        "품목명의 공백을 제거한 뒤 ‘항공여객운송서비스’와 정확히 일치하는 행을 집계합니다.", "",
        "## 공통 주의사항", "",
        *(f"- {note}" for note in REPORT_NOTES), "",
    ]
    for report in reports:
        reasons = ", ".join(
            f"{label} {report['undetermined_reasons'][key]:,}행"
            for key, label in UNDETERMINED_REASONS.items()
        )
        lines.extend([
            f"## {report['file']}", "",
            f"- 관측단위 {report['unit']}",
            (f"- 전체(분모) {report['total_rows']:,}행 / 항공 정확일치 {air_rows_text(report)}"
             f" / 기타 품목(비항공) {report['non_air_rows']:,}행"),
            f"- 판정 불가 {report['undetermined_rows']:,}행 ({reasons})",
            f"- 기간 기준 {report['period_basis'] or '없음'}",
            f"- 전체 기간 {period_text(report['total_period'])}",
            f"- 항공 정확일치 기간 {period_text(report['air_exact_period'])}",
        ])
        if report["air_exact_rows"] is None:
            lines.append(f"- 일반 ‘여객운송서비스’ {report['ambiguous_rows']:,}행")
        unique = report["unique_accident_ids"]
        if unique is not None:
            lines.append(f"- 고유 사건번호 {unique:,}개 / 중복 행 {report['duplicate_accident_rows']:,}개")
        else:
            lines.append("- 고유 사건번호·중복 행 산출 불가 (사건번호 열 없음)")
        lines.append("- 결측 " + ", ".join(f"{key} {n:,}개" for key, n in report["missing"].items()))
        if report["invalid_dates"] is not None:
            lines.append(f"- 결측 이외의 날짜 파싱 실패 {report['invalid_dates']:,}행")
        volume = report["consultation_count_sums"]
        if volume is not None:
            lines.append(f"- 상담건수 합계 전체 {volume['total']:,}건 / 항공 정확일치 {volume['air_exact']:,}건"
                         f" / 결측 이외의 파싱 실패 {volume['invalid']:,}행")
        lines.extend(["", "### 주의사항", ""])
        lines.extend(f"- {note}" for note in report["notes"])
        lines.append("")
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_fit_report.json").write_text(
        json.dumps({"notes": REPORT_NOTES, "datasets": reports}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "data_fit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=50_000)
    args = parser.parse_args()
    if args.chunksize <= 0:
        parser.error("chunksize는 양수여야 합니다.")
    try:
        reports = profile_directory(PROJECT.parent / "voc_private_data" / "raw", args.chunksize)
        write_reports(PROJECT / "reports", reports)
    except Exception as error:  # noqa: BLE001 - 의도적: 예외 종류만 출력해 원문 유출을 막는다
        # 파서 예외의 세부 메시지에는 입력값이 들어 있을 수 있다.
        parser.exit(1, f"집계 실패 ({type(error).__name__}).\n")
    for report in reports:
        print(f"{report['file']} | 전체 {report['total_rows']:,}행"
              f" | 항공 정확일치 {air_rows_text(report)}"
              f" | 판정 불가 {report['undetermined_rows']:,}행")


if __name__ == "__main__":
    main()
