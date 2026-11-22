"""화면의 자연어 요청을 사전 정의된 분석 요청 3종 중 하나로만 매핑한다.

공개 DRY_RUN 앱은 사용자의 문장을 모델·외부 API로 보내지 않는다. 문장은 여기서 키워드로 분류되어
열거된 요청 이름(ANALYSIS_REQUESTS) 하나로 바뀔 때만 쓰이며, 지원하지 않는 입력은 거부한다.
"""

from .contracts import ANALYSIS_REQUESTS

MAX_REQUEST_CHARS = 100
_KEYWORDS = {
    "월별 추세 분석": ("추세", "월별", "트렌드", "변화"),
    "급증 구간 확인": ("급증", "피크", "정점", "최대", "치솟"),
    "개선 방안 보고서": ("개선", "방안", "보고서", "대응", "요약"),
}
# 지시 우회·민감 정보·외부 호출을 암시하는 표현은 키워드가 맞더라도 거부한다.
_BLOCKED = ("api", "키", "원문", "상담 내용", "개인정보", "주민", "전화", "프롬프트", "지시", "무시",
            "gpt", "openai", "실제 분석", "삭제", "비밀", "password", "ignore")
SUPPORTED_HELP = "지원하는 요청: " + " / ".join(f"‘{name}’" for name in ANALYSIS_REQUESTS)

assert set(_KEYWORDS) == set(ANALYSIS_REQUESTS)


def parse_request(text: str) -> tuple[str | None, str]:
    """(요청 이름, 안내 문구). 지원하지 않으면 요청 이름이 None이다."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None, f"요청을 입력해 주세요. {SUPPORTED_HELP}"
    if len(cleaned) > MAX_REQUEST_CHARS:
        return None, f"요청은 {MAX_REQUEST_CHARS}자 이내로 입력해 주세요. {SUPPORTED_HELP}"
    lowered = cleaned.casefold()
    if any(word in lowered for word in _BLOCKED):
        return None, f"이 화면에서는 지원하지 않는 요청입니다(외부 호출·원문·지시 변경 불가). {SUPPORTED_HELP}"
    matches = [name for name, words in _KEYWORDS.items() if any(word in cleaned for word in words)]
    if len(matches) == 1:
        return matches[0], f"‘{matches[0]}’ 요청으로 실행합니다."
    if not matches:
        return None, f"이해할 수 없는 요청입니다. {SUPPORTED_HELP}"
    return None, f"요청이 여러 종류로 해석됩니다. 하나만 요청해 주세요. {SUPPORTED_HELP}"
