# 수원과학대 규정 안내 RAG 챗봇 (ChatSSC)

규정 PDF를 검색해 근거와 함께 답하는 챗봇입니다. 기본은 **전체 규정 검색**이고, 화면의 **세부 검색** 토글을 켜면 분야를 골라 검색합니다.

```
질문 ─▶ (선택) 분야 필터 ─▶ Qdrant 검색(현행 규정·점수 하한) ─▶ rag.py ─▶ LLM ─▶ 답변 + 출처(규정명 p.쪽)
```

## 폴더 구조
- `app/config.py` 설정과 분야(GROUPS) 정의 · `app/ingest.py` PDF → 청크 → 임베딩 → 적재
- `app/qdrant.py` 저장/검색 · `app/category.py` 분야 검증 · `app/rag.py` 프롬프트·답변 · `app/main.py` API
- `frontend/index.html` 채팅 화면 


## 파일명 규칙 → 분야
파일명은 `A-B-C 규정명.pdf` 형식입니다. **A가 분야(대분류)**, B는 세부 묶음(`subgroup_no`로 저장), C는 일련번호입니다.

| A | 분야 | A | 분야 |
|---|---|---|---|
| 1 | 학교법인 | 6 | 학생·장학 |
| 2 | 학칙 | 7 | 부속기관·센터 |
| 3 | 위원회 | 8 | 산학협력·연구 |
| 4 | 인사·행정 | 9 | 노사 |
| 5 | 교육과정·학사 | | 번호가 없거나 1~9 밖 → "기타" |

- 이름이 `폐지`로 끝나는 파일은 폐지 규정으로 보고 **기본적으로 적재하지 않습니다**(`--include-abolished`를 주면 비현행으로 저장, 검색에는 나오지 않음).
- 분야를 바꾸려면 `config.py`의 `GROUPS`와 `frontend/index.html`의 `FALLBACK_GROUPS`를 함께 고칩니다.

## 실행
```bash
pip install -r requirements.txt
cp .env.example .env            # 키 입력
python -m app.ingest --reset --limit 5     # 먼저 5개로 시험 (--reset은 기존 데이터를 모두 지움)
python -m app.ingest --skip-existing       # 전체 적재 (중단돼도 같은 명령으로 이어서)
uvicorn app.main:app --reload              # http://127.0.0.1:8000

```
적재 옵션: `--dir 폴더`, `--group 1~9`(폴더 전체를 한 분야로), `--skip-existing`, `--limit N`, `--include-abolished`, `--reset`.
같은 파일명을 다시 적재하면 기존 청크를 지우고 새로 넣습니다. 한 파일이 실패해도 나머지는 계속되며, 끝에 요약과 분야별 문서 수가 출력됩니다.

## 검색 동작
- 항상 `status=current`(현행)인 청크만 검색합니다. 개정 대비표의 '개정 전'은 비현행으로 저장됩니다.
- 세부 검색에서 분야를 고르면 그 분야 안에서만 찾고, 결과가 없어도 범위를 넓혀 재검색하지 않습니다(사용자가 고른 범위 존중).
- 분야를 모두 고르면 전체 검색과 동일하게 처리합니다.

## 배포 후 문서를 바꿀 때
Qdrant에 적재만 다시 하면 서버는 재배포 없이 새 내용을 사용합니다. 코드·화면을 바꾼 경우에만 이미지를 다시 빌드해 같은 Cloud Run 서비스에 배포하세요(URL·QR 유지).

## 알려진 한계
- 스캔(이미지) PDF는 텍스트가 없어 "내용 없음"으로 건너뜁니다.
- `SCORE_THRESHOLD`(기본 0.40)는 초기값입니다. 튜닝 스크립트로 조정하세요.
