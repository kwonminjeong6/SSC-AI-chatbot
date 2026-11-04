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

# --- 분야(대분류) ---
# 파일명 앞의 규정 번호(예: 4-1-8)에서 첫 숫자가 대분류입니다.
# 화면의 '세부 검색'은 이 번호로 필터하므로, 이름(오른쪽 값)은 바꿔도 검색 동작에는 영향이 없습니다.
GROUPS = {
    "1": "학교법인",
    "2": "학칙",
    "3": "위원회",
    "4": "인사·행정",
    "5": "교육과정·학사",
    "6": "학생·장학",
    "7": "부속기관·센터",
    "8": "산학협력·연구",
    "9": "노사",
}
DEFAULT_CATEGORY = "기타"  # 규정 번호가 없는 파일

# --- 서버 ---
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]


# --- 답변 캐시 ---
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "21600"))  # 초 (6시간)
CACHE_MAX = int(os.getenv("CACHE_MAX", "500"))
