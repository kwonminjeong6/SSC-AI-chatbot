"""임베딩 모델 래퍼. 모델은 처음 사용할 때 한 번만 로드합니다(import 시점 로딩 없음)."""
import logging
from functools import lru_cache

from app import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    logger.info("임베딩 모델 로딩: %s", config.EMBEDDING_MODEL)
    return SentenceTransformer(config.EMBEDDING_MODEL)


def encode(text: str) -> list[float]:
    return get_model().encode(text, normalize_embeddings=True).tolist()


def encode_batch(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 32,
    )
    return vectors.tolist()
