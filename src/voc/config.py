"""VOC 런타임 에이전트 설정. 모델 ID는 여기서만 정의하며 코드에서 임의로 다른 모델로 대체하지 않는다."""

# Collector·Analyst·Writer·Reviewer·Manager 공통 API 모델(사용자 지정). 2026-11-22 API 모델 목록 조회로 존재 확인.
VOC_AGENT_MODEL = "gpt-6-luna"

# 토큰 수의 엄밀한 상한: BPE 토큰 수는 UTF-8 바이트 수를 넘지 않고, BMP 글자는 최대 3바이트이므로 글자당 3토큰.
TOKENS_PER_CHAR_UPPER = 3
