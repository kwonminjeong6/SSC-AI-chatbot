import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, embedding
from app.category import detect_category
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
    title="대학 지식기반 RAG 챗봇 API",
    description="Qdrant Cloud + OpenRouter LLM 기반 대학 규정 안내 챗봇 (카테고리 수동/자동 분류)",
    version="1.1.0",
    lifespan=lifespan,
)

# 쿠키를 쓰지 않으므로 allow_credentials는 켜지 않습니다(와일드카드 origin과 함께 쓸 수 없음).
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="사용자 질문", examples=["휴학은 최대 몇 학기까지 할 수 있나요?"])
    category_filter: Optional[str] = Field(None, description="'학사규정' 또는 '장학금'. 비우면 질문을 분석해 자동 분류")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM이 생성한 답변")
    sources: list[str] = Field(..., description="출처 목록 (문서명 p.페이지)")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="질문을 입력해 주세요.")

    # 프론트에서 유효한 카테고리를 보냈으면 그 값을 우선, 아니면 자동 분류
    category = request.category_filter if request.category_filter in config.CATEGORIES else detect_category(query)
    logger.info("질문 수신 (category=%s)", category)

    try:
        result = await generate_rag_answer(query, category)
    except Exception:
        logger.exception("챗봇 API 처리 중 오류")
        raise HTTPException(status_code=500, detail="챗봇 서버에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
    return ChatResponse(**result)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "university-rag-api"}


# API 라우트 뒤에 마운트: http://127.0.0.1:8000/ 에서 프론트엔드를 함께 제공합니다.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
