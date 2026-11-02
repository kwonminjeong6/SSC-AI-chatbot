"""모든 설정은 이 파일에서 환경변수(.env)로 한 번만 읽습니다."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_DIR = Path(os.getenv("DOCUMENT_DIR", BASE_DIR / "data" / "documents"))

# --- Qdrant Cloud ---
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "university_knowledge")

# --- 임베딩 ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # bge-m3 = 1024

# --- LLM (OpenRouter) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen-2.5-7b-instruct")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

# --- 검색 ---
TOP_K = int(os.getenv("TOP_K", "4"))
# 코사인 유사도 하한. 이보다 낮은 청크는 "근거 없음"으로 취급합니다.
# 기본값은 출발점일 뿐이므로 `python -m evaluation.run_eval` 결과를 보고 조정하세요.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.40"))

# --- 청킹 ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- 카테고리 ---
CATEGORIES = ("학사규정", "장학금")
# 파일명에 키워드가 포함되면 해당 카테고리로 적재됩니다. (매칭 없으면 DEFAULT_CATEGORY)
CATEGORY_BY_FILENAME = {"학칙": "학사규정", "장학": "장학금"}
DEFAULT_CATEGORY = "기타"

# --- 서버 ---
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
