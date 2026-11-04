import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, embedding
from app.category import list_groups, normalize_groups
from app.qdrant import init_collection
from app.rag import generate_rag_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.main")

FRONTEND_DIR = config.BASE_DIR / "frontend"


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
    version="1.2.0",
    lifespan=lifespan,
)

# 쿠키를 쓰지 않으므로 allow_credentials는 켜지 않습니다(와일드카드 origin과 함께 쓸 수 없음).
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="사용자 질문", examples=["계절학기는 어떻게 운영되나요?"])
    groups: Optional[list[str]] = Field(
        None, max_length=20, description="세부 검색: 분야 번호('1'~'9') 목록. 비우면 전체 규정에서 검색"
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM이 생성한 답변")
    sources: list[str] = Field(..., description="출처 목록 (규정명 p.페이지)")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="질문을 입력해 주세요.")

    groups = normalize_groups(request.groups)
    logger.info("질문 수신 (groups=%s)", groups)

    try:
        result = await generate_rag_answer(query, groups)
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
