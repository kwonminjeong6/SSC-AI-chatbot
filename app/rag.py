"""(후속 질문 고쳐 쓰기) → 검색 → 프롬프트 구성 → LLM 답변 생성."""
import logging
import re
from datetime import date
from functools import lru_cache
from typing import Optional

from fastapi.concurrency import run_in_threadpool
from openai import AsyncOpenAI

from app import config, embedding
from app.cache import answer_cache, make_key
from app.qdrant import search_docs

logger = logging.getLogger(__name__)

NO_ANSWER = "현재 등록된 대학 공식 자료에서는 해당 내용을 확인할 수 없습니다."
ERROR_ANSWER = "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
SYSTEM_PROMPT = "너는 대학 규정 및 학사 정보 안내 챗봇이다. 제공된 문서만을 근거로, 주어진 규칙을 엄격히 지켜 답변하라."
CONTENT_TYPE_LABEL = {"article": "조문", "table": "표", "revision": "개정 대비표"}

RULES = f"""다음 규칙을 반드시 지키십시오:
1. 질문과 문서의 표현이 달라도 의미가 같은 내용을 찾아 답변하십시오.
2. 표·수치 데이터는 행과 열을 정확히 대조하십시오. 표 제목의 연도(예: 2027학년도)가 질문과 일치하는 수치만 사용하고, 다른 연도의 수치를 섞지 마십시오.
3. '[개정 후 ...]'로 표시된 내용이 현재 시행 중인 규정입니다.
4. 여러 규정이 검색되면 질문에 가장 직접 해당하는 규정을 근거로 삼고, 답변에 규정명을 밝히십시오.
5. 기간·횟수·학기·학점·금액 등 한도를 묻는 질문은 학칙뿐 아니라 시행세칙·내규 등 검색된 모든 문서의 관련 조항을 끝까지 확인하십시오. 1회당 한도와 총 횟수(통산 한도)가 따로 있으면 둘 다 빠짐없이 답하고, 한도에서 제외되는 예외도 함께 알려 주십시오.
6. 문서에 없는 내용은 추측하거나 만들어내지 마십시오.
7. 문서에서 질문의 근거를 전혀 찾을 수 없을 때만 정확히 다음 문장으로 답하십시오: "{NO_ANSWER}"
8. 학생과 교직원이 이해하기 쉽게 간결하고 명확하게 답하십시오."""

REWRITE_SYSTEM = "너는 대화의 마지막 질문을 검색용 질문으로 고쳐 쓰는 도우미다."
REWRITE_PROMPT = """아래 대화를 참고해서 '마지막 질문'을 앞 대화 없이도 이해되는 하나의 완결된 질문으로 고쳐 쓰십시오.
- 마지막 질문이 이미 완결되어 있으면 그대로 출력하십시오.
- 대화에 없는 내용을 새로 만들어 넣지 마십시오.
- 설명 없이 질문 한 문장만 출력하십시오.

[대화]
{history}

[마지막 질문]
{query}

[고쳐 쓴 질문]
"""

# 앞 질문을 가리키는 말이 들어 있거나 질문이 아주 짧을 때만 고쳐 씁니다(LLM 호출 비용 절약).
FOLLOW_UP_HINTS = ("그럼", "그러면", "그렇다면", "그건", "그거", "그것", "그게", "거기", "이건", "이거", "방금", "앞서", "아까", "그 경우", "그때")
SHORT_QUERY_LEN = 15  # 공백을 뺀 글자 수


@lru_cache(maxsize=1)
def get_llm() -> AsyncOpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다.")
    return AsyncOpenAI(
        base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY, timeout=config.LLM_TIMEOUT
    )


def retrieve(query: str, groups: Optional[list[str]] = None) -> list[dict]:
    """질문을 임베딩해 현행 규정 청크를 검색합니다(동기 함수, 스레드풀에서 호출)."""
    return search_docs(embedding.encode(query), groups=groups)


def doc_title(doc: dict) -> str:
    """사람이 읽는 규정 이름. 규정명이 없으면 파일명을 씁니다."""
    return doc.get("regulation_name") or doc.get("document_name") or "알 수 없음"


def format_context(docs: list[dict]) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        title = doc_title(doc)
        if doc.get("regulation_no"):
            title += f" ({doc['regulation_no']})"
        meta = [f"규정명: {title}"]
        if doc.get("category"):
            meta.append(f"분야: {doc['category']}")
        meta += [
            f"페이지: {doc.get('page') or '-'}",
            f"유형: {CONTENT_TYPE_LABEL.get(doc.get('content_type'), '본문')}",
        ]
        if doc.get("article"):
            meta.append(f"조항: {doc['article']}")
        blocks.append(f"[검색 문서 {i}]\n" + "\n".join(meta) + f"\n내용:\n{doc.get('text', '')}")
    return "\n\n".join(blocks)


def create_prompt(question: str, docs: list[dict]) -> str:
    return (
        "아래 대학 공식 문서만을 근거로 사용자의 질문에 답변하십시오.\n"
        f"오늘 날짜: {date.today().isoformat()}\n\n"
        f"{RULES}\n\n[대학 공식 문서 검색 결과]\n\n{format_context(docs)}\n\n"
        f"[사용자 질문]\n{question}\n\n[답변]\n"
    )


def format_sources(docs: list[dict]) -> list[str]:
    """'규정명 p.페이지' 형태로, 등장 순서를 유지하며 중복을 제거합니다."""
    labels = (f"{doc_title(d)} p.{d['page']}" if d.get("page") else doc_title(d) for d in docs)
    return list(dict.fromkeys(labels))


def needs_rewrite(query: str, history: Optional[list[dict]]) -> bool:
    if not history:
        return False
    compact = re.sub(r"\s+", "", query)
    return len(compact) <= SHORT_QUERY_LEN or any(hint in query for hint in FOLLOW_UP_HINTS)


def format_history(history: list[dict]) -> str:
    lines = []
    for item in history[-config.HISTORY_MAX_MESSAGES:]:
        role = "사용자" if item.get("role") == "user" else "챗봇"
        text = " ".join(str(item.get("content", "")).split())[:300]
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


async def rewrite_query(query: str, history: Optional[list[dict]]) -> str:
    """앞 대화를 참고해 '그럼 최대 몇 학점이야?' 같은 질문을 완결된 문장으로 고칩니다. 실패하면 원래 질문을 씁니다."""
    if not needs_rewrite(query, history):
        return query
    try:
        response = await get_llm().chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": REWRITE_PROMPT.format(history=format_history(history), query=query)},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        lines = [l.strip().strip('"\'“”‘’') for l in (response.choices[0].message.content or "").splitlines() if l.strip()]
        rewritten = lines[0] if lines else ""
        if not rewritten or len(rewritten) > 200:
            return query
        logger.info("후속 질문 고쳐 쓰기: %r -> %r", query, rewritten)
        return rewritten
    except Exception:
        logger.exception("질문 고쳐 쓰기 실패 — 원래 질문으로 검색합니다")
        return query


async def generate_rag_answer(
    query: str, groups: Optional[list[str]] = None, history: Optional[list[dict]] = None
) -> dict:
    standalone = await rewrite_query(query, history)

    key = make_key(standalone, groups)
    if config.CACHE_ENABLED:  # 같은 질문이면 검색·LLM 호출 없이 바로 답변
        cached = answer_cache.get(key)
        if cached:
            logger.info("캐시 적중 (groups=%s)", groups)
            return {"answer": cached["answer"], "sources": list(cached["sources"])}

    docs = await run_in_threadpool(retrieve, standalone, groups)
    logger.info("검색 %d건 (groups=%s, scores=%s)", len(docs), groups, [round(d["score"], 3) for d in docs])
    if not docs:
        return {"answer": NO_ANSWER, "sources": []}  # LLM 비용이 없으므로 캐시하지 않음

    try:
        response = await get_llm().chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": create_prompt(standalone, docs)},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise ValueError("LLM이 빈 응답을 반환했습니다.")
    except Exception:
        logger.exception("LLM 호출 실패")
        return {"answer": ERROR_ANSWER, "sources": []}  # 오류는 캐시하지 않음

    sources = [] if NO_ANSWER in answer else format_sources(docs)
    result = {"answer": answer, "sources": sources}
    if config.CACHE_ENABLED:  # 성공한 답만 저장
        answer_cache.set(key, result)
    return result