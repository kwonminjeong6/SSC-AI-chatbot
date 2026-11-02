# 대학 규정 안내 RAG 챗봇 (ChatSSC)

PDF 규정 문서를 Qdrant Cloud에 적재하고, 질문에 맞는 근거를 검색해 OpenRouter LLM이 답변하는 챗봇입니다.

```
PDF ─ ingest.py ─▶ 청크(조문/표/개정 대비표) ─ bge-m3 ─▶ Qdrant Cloud
질문 ─ category.py(자동 분류) ─▶ 검색(현행 문서·점수 하한) ─▶ rag.py ─▶ LLM ─▶ 답변 + 출처
```

## 폴더 구조

```
app/
  config.py      모든 설정(환경변수)
  embedding.py   bge-m3 임베딩 (지연 로딩, 배치 인코딩)
  qdrant.py      컬렉션 관리 · 적재 · 검색
  category.py    질문 카테고리 자동 분류
  rag.py         검색 + 프롬프트 + LLM 호출
  ingest.py      PDF 파싱 · 청킹 · 적재 CLI
  main.py        FastAPI 서버 (프론트엔드도 함께 제공)
frontend/index.html
data/documents/  적재할 PDF
```

## 실행 방법

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # QDRANT_URL, QDRANT_API_KEY, OPENROUTER_API_KEY 입력

# 1) 문서 적재 (처음이거나 적재 스키마가 바뀐 경우 --reset 권장)
python -m app.ingest --reset

# 2) 서버 실행 → http://127.0.0.1:8000
uvicorn app.main:app --reload

## 문서 추가 방법

`data/documents/`에 PDF를 넣고 `python -m app.ingest`를 실행합니다. 같은 파일명은 기존 청크를 지우고 교체됩니다.
카테고리는 파일명으로 정해집니다(`config.py`의 `CATEGORY_BY_FILENAME`: 파일명에 "학칙" → 학사규정, "장학" → 장학금).
새 카테고리를 추가하려면 `CATEGORY_BY_FILENAME`, `CATEGORIES`, `category.py`의 `KEYWORDS`, 프론트 탭을 함께 수정하세요.

## 설계 포인트

- **개정 버전 관리**: 신·구조문 대비표는 셀 단위로 나누고 시행일 기준으로 개정 후만 `status=current`, 개정 전은 `inactive`로 저장합니다. 검색은 항상 `current`만 대상으로 합니다.
- **표 처리**: 입학정원 표는 `[2027학년도 입학정원] 4년제 간호학과: 주간 138명, ...` 형태로 변환해 연도 혼동을 막습니다.
- **환각 방지**: 유사도 `SCORE_THRESHOLD` 미만이면 "확인할 수 없음"으로 응답하고, 이때 출처는 표시하지 않습니다.
- **카테고리 폴백**: 자동 분류가 틀려 결과가 없으면 필터 없이 재검색합니다.

## 알려진 한계

- `SCORE_THRESHOLD=0.40`은 초기값입니다. 결과를 보며 조정하세요.
- 조문 분할은 `제N조(제목)` 형식을 기준으로 합니다. 제목 괄호가 없는 조문은 길이 기준으로만 분할됩니다.
- 스캔본(이미지) PDF는 텍스트가 추출되지 않으며 OCR이 필요합니다.
