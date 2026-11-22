"""공정위 개별 상담 CSV에서 허용된 구조화 열만 읽는다."""

import csv
import re
from pathlib import Path

import pandas as pd

AIR_ITEM = "항공여객운송서비스"
BLOCKED_COLUMNS = frozenset({
    "ACCIDENT_TITLE", "ACCIDENT_CONTENT", "ANS_CONTENT",
    "사건제목", "사건내용", "답변내용", "질문", "답변", "질문본문", "답변본문",
})
STRUCTURED_COLUMNS = frozenset({
    "ACCIDENT_NO", "RCPT_YMD", "ITEM_CODE", "ITEM_NAME", "연도", "월", "품목명", "상담건수",
})
HEADER_CODE = re.compile(r"\(([A-Z][A-Z0-9_]*)\)\s*$")
DATASETS = (
    ("receipt", "소비자상담 접수내역"),
    ("online", "온라인 상거래 소비자상담 내역"),
    ("hazard", "위해위험 소비자상담 내역"),
)


def column_key(header):
    match = HEADER_CODE.search(header)
    return match.group(1) if match else header.strip()


def read_chunks(path, columns, chunksize=50_000, *, encoding="cp949"):
    """요청한 열 전체를 검증한 뒤 CSV 파서에 전달한다."""
    columns = list(columns)
    for column in columns:
        key = column_key(column)
        label = column.partition("(")[0].strip()
        if key in BLOCKED_COLUMNS or label in BLOCKED_COLUMNS:
            raise ValueError("차단된 열이 로딩 목록에 포함되었습니다.")
        if key not in STRUCTURED_COLUMNS:
            raise ValueError("허용 목록에 없는 열이 로딩 목록에 포함되었습니다.")
    return pd.read_csv(
        path, encoding=encoding, usecols=columns, chunksize=chunksize,
        dtype="string", keep_default_na=False, index_col=False,
    )


def consultation_chunks(path, code_names, *, code_only=False, chunksize=50_000, encoding="cp949"):
    """분모 행 전체를 반환한다. is_air의 결측은 판정 불가를 뜻한다.

    이름 자료를 끝까지 읽으면 code_names에 코드별 품목명이 누적된다.
    코드 자료는 두 이름 자료의 순회가 끝난 뒤 읽는다.
    """
    fields = {"ACCIDENT_NO", "RCPT_YMD", "ITEM_CODE"}
    if not code_only:
        fields.add("ITEM_NAME")
    with Path(path).open("rb", buffering=0) as stream:
        header = bytearray()
        while (byte := stream.read(1)) and byte != b"\n":
            header.extend(byte)
    selected = {}
    for column in next(csv.reader([header.decode(encoding).rstrip("\r")])):
        key = column_key(column)
        if key in fields:
            if key in selected:
                raise ValueError("구조화 헤더가 중복되었습니다.")
            selected[key] = column
    if selected.keys() != fields:
        raise ValueError("필수 구조화 헤더가 없습니다.")
    names_by_code = {
        code: next(iter(names)) for code, names in code_names.items() if len(names) == 1
    } if code_only else {}
    rename = {column: key for key, column in selected.items()}
    with read_chunks(path, rename, chunksize, encoding=encoding) as reader:
        for chunk in reader:
            chunk = chunk.rename(columns=rename)
            for field in chunk:
                values = chunk[field].str.strip()
                chunk[field] = values.mask(values.eq(""))
            if code_only:
                names = chunk["ITEM_CODE"].map(names_by_code).astype("string")
            else:
                names = chunk["ITEM_NAME"]
                pairs = chunk[["ITEM_CODE", "ITEM_NAME"]].dropna().drop_duplicates()
                for code, name in pairs.itertuples(index=False, name=None):
                    code_names.setdefault(code, set()).add(name)
            chunk["is_air"] = names.str.replace(r"\s+", "", regex=True).eq(AIR_ITEM)
            yield chunk[["ACCIDENT_NO", "RCPT_YMD", "is_air"]]
