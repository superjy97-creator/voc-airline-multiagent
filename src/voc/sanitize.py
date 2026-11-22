"""정규식으로 개인정보 후보를 찾는다. 이름은 오탐이 커서 미지원이며 노출 위험이 남는다."""

import hashlib
import re

from .contracts import CheckedText

PATTERNS = {
    "resident_id": r"(?<!\d)\d{6}[ -]?[1-8]\d{6}(?!\d)",
    "phone": r"(?<!\d)(?:\+82[ .-]?(?:10|1[16789]|2|[3-6][1-5]|70)|0(?:1[016789]|2|[3-6][1-5]|70))[ .-]?\d{3,4}[ .-]?\d{4}(?!\d)",
    "email": r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}",
    "card": r"(?<!\d)(?:\d[ -]?){12,18}\d(?![ -]?\d)",
    # 계좌번호 후보: 실제로 유효한 'YYYY-MM-DD' 날짜 형태만 제외한다(집계 기간·보고서 날짜의 오탐 방지).
    "account": (r"(?:계좌(?:번호)?\s*[:：]?\s*\d(?:[ -]?\d){7,19}(?!\d)"
                r"|(?<!\d)(?!(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d))"
                r"\d{2,6}(?:-\d{2,8}){2,3}(?!\d))"),
    # 예약번호 후보: ① 6자리 대문자·숫자이되 글자가 하나 이상(숫자만 있는 6자리 건수·금액은 제외),
    # ② '예약번호'·'PNR' 라벨이 붙은 5~10자리 영숫자(숫자만이어도 탐지).
    "pnr": (r"(?<![A-Za-z0-9])(?=[0-9]*[A-Z])[A-Z0-9]{6}(?![A-Za-z0-9])"
            r"|(?:예약\s*번호|PNR)\s*[:：]?\s*[A-Za-z0-9]{5,10}(?![A-Za-z0-9])"),
    "flight": r"(?<![A-Za-z0-9])[A-Z]{2}\s?\d{2,4}(?![A-Za-z0-9])",
    "membership": r"(?:회원\s*(?:번호|ID)|멤버십\s*(?:번호|ID))\s*[:：]?\s*[A-Za-z0-9][A-Za-z0-9-]{3,}",
    "address": r"[가-힣]+(?:시|도)\s+(?:[가-힣]+(?:시|군|구)\s+)*[가-힣0-9]+(?:구|동|로|길)\s*\d+(?:-\d+)?",
    "url": r"(?i)(?:https?://|www\.)[^\s<>]+",
}
_PATTERNS = {code: re.compile(pattern) for code, pattern in PATTERNS.items()}


def scan_text(text: str) -> tuple[str, ...]:
    return tuple(code for code, pattern in _PATTERNS.items() if pattern.search(text))


def make_checked_text(text: str) -> CheckedText:
    return CheckedText(text=text, pii_scan_passed=not scan_text(text))


def quarantine_entry(source_id: str, reasons) -> dict:
    codes = tuple(reasons)
    if any(code not in _PATTERNS for code in codes):
        raise ValueError("등록된 탐지 사유 코드만 허용됩니다.")
    return {
        "source_id_hash": hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16],
        "reasons": codes,
    }
