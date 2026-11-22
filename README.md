# 항공여객운송서비스 VOC 개선 멀티에이전트

1372 소비자상담센터의 항공여객운송서비스 VOC를 분석해 개선 보고서를 만드는 **5개 AI 에이전트**(Collector · Analyst · Writer · Reviewer · Manager) 시연 프로젝트입니다. Python이 실행 순서·호출 상한·승인 게이트를 통제하고, 사람이 **Human Review**와 **최종 승인**을 직접 결정합니다.

> 이 저장소의 화면은 **DRY_RUN 전용**(가짜 모델, 외부 호출 0회)입니다. 실제 VOC 자료·집계 결과·LIVE 실행 기록은 포함하지 않으며, 화면에는 **합성 예시**만 표시됩니다.

## 실행 흐름
`Collector → Analyst → Writer(초안) → Reviewer → Human Review → [수정 필요 시 Writer 수정 → Reviewer 재검수] → Manager → 사용자 최종 승인`

- Human Review는 항상 실행합니다. 사람이 수정 없이 승인하면 Writer 수정·Reviewer 재검수만 생략합니다.
- 모델 호출 상한: 총 9회(Writer 3, Reviewer 3, 나머지 각 1). 재시도도 같은 상한에서 차감합니다.
- 개인정보 의심 패턴이 있는 사람 메모는 호출 전에 거부합니다(정규식 기반, 이름은 탐지하지 못함).

## 구조
| 경로 | 내용 |
|---|---|
| `app.py` | Streamlit CX 대시보드(KPI·차트·에이전트 진행·Human Review·보고서 다운로드) |
| `src/voc/contracts.py` | 단계 상태·승인·호출 예산·LIVE 승인 게이트 계약 |
| `src/voc/pipeline.py`, `runners.py`, `prompts.py` | 순차 실행, 가짜/SDK 러너, 공통+역할 지침 주입 |
| `src/voc/sanitize.py`, `loader.py`, `stats.py` | 개인정보 검사, 구조화 열 전용 로더, 통계 |
| `instructions/` | 에이전트 공통·역할별 지침(에이전트에는 공통 + 해당 역할 지침만 주입) |
| `scripts/live_smoke.py` | LIVE 점검(합성 데이터 전용, 사용자 승인·비용 상한 필요) |
| `scripts/live_real.py` | 로컬에서만 쓰는 실제 집계 LIVE(전송 내용 승인 코드·유료 승인·비용 상한 필요). 저장소에는 실제 집계가 없어 실행할 수 없음 |
| `tests/` | unittest 110개 |

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```
개발·테스트: `pip install -r requirements-dev.txt` 후 `python -m unittest discover -s tests`, `ruff check .`

## Streamlit Community Cloud 배포
1. 이 저장소를 GitHub에 올린 뒤 https://share.streamlit.io 에서 **New app**을 선택합니다.
2. Repository: 이 저장소, Branch: `main`, Main file path: `app.py`
3. 환경변수·비밀키는 필요하지 않습니다(DRY_RUN은 외부 API를 쓰지 않습니다).

## 데이터·보안 원칙
- API 키(`.env`), 실제 VOC 자료, 집계 결과(`reports/`), LIVE 상태·비용 기록은 `.gitignore`로 제외합니다.
- 상담 원문·제목 열은 읽지 않고, 구조화 열(날짜·품목 등)만 사용합니다.
- 외부 유료 API 호출은 합성 데이터라도 사전 승인(모델·비용·전송 데이터·최대 호출 횟수)이 필요하며, 실제 VOC 데이터의 외부 전송은 별도 승인이 필요합니다.

## 알려진 문제
[KNOWN_ISSUES.md](KNOWN_ISSUES.md)를 참고하세요.
