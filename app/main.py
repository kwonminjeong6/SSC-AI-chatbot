import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, embedding
from app.category import list_groups, normalize_groups
from app.qdrant import init_collection
from app.rag import generate_rag_answer
from app.ratelimit import RateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.main")

FRONTEND_DIR = config.BASE_DIR / "frontend"

# 접속자(IP)별 한도와 서버 전체 한도. 후자는 주소를 속이는 요청까지 막는 안전장치입니다.
ip_limiter = RateLimiter(config.RATE_LIMIT_PER_MIN)
global_limiter = RateLimiter(config.RATE_LIMIT_GLOBAL_PER_MIN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(init_collection)
    except Exception:
        logger.exception("Qdrant 초기화 실패: QDRANT_URL / QDRANT_API_KEY를 확인하세요")
    await run_in_threadpool(embedding.get_model)  # 첫 요청 지연 방지
    yield


app = FastAPI(
    title="대학 규정 안내 RAG 챗봇 API",
    description="Qdrant Cloud + OpenRouter LLM 기반 대학 규정 안내 챗봇 (기본: 전체 규정 검색, 선택: 분야별 세부 검색)",
    version="1.3.0",
    lifespan=lifespan,
)

# 쿠키를 쓰지 않으므로 allow_credentials는 켜지 않습니다(와일드카드 origin과 함께 쓸 수 없음).
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="사용자 질문", examples=["계절학기는 어떻게 운영되나요?"])
    groups: Optional[list[str]] = Field(
        None, max_length=20, description="세부 검색: 분야 번호('1'~'9') 목록. 비우면 전체 규정에서 검색"
    )
    history: Optional[list[HistoryItem]] = Field(
        None, max_length=20, description="후속 질문용: 앞 대화(최근 것). 질문을 완결된 문장으로 고쳐 쓰는 데만 사용"
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM이 생성한 답변")
    sources: list[str] = Field(..., description="출처 목록 (규정명 p.페이지)")


def client_ip(http_request: Request) -> str:
    """Cloud Run 뒤에서는 접속자 주소가 X-Forwarded-For 헤더에 들어 있습니다."""
    forwarded = http_request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"


def check_rate_limit(http_request: Request) -> None:
    if not config.RATE_LIMIT_ENABLED:
        return
    ip = client_ip(http_request)
    if not ip_limiter.allow(ip) or not global_limiter.allow("*"):
        logger.warning("요청 제한 적용 (ip=%s)", ip)
        raise HTTPException(
            status_code=429,
            detail="요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
            headers={"Retry-After": "30"},
        )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, http_request: Request):
    check_rate_limit(http_request)

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="질문을 입력해 주세요.")

    groups = normalize_groups(body.groups)
    history = [{"role": h.role, "content": h.content} for h in (body.history or [])]
    logger.info("질문 수신 (groups=%s, history=%d)", groups, len(history))

    try:
        result = await generate_rag_answer(query, groups, history or None)
    except Exception:
        logger.exception("챗봇 API 처리 중 오류")
        raise HTTPException(status_code=500, detail="챗봇 서버에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
    return ChatResponse(**result)


@app.get("/api/groups")
async def groups_endpoint():
    """화면의 '세부 검색' 선택지."""
    return list_groups()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "university-rag-api"}


# API 라우트 뒤에 마운트: http://127.0.0.1:8000/ 에서 프론트엔드를 함께 제공합니다.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")