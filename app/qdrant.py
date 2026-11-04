"""Qdrant Cloud 접근 계층: 컬렉션 관리, 적재, 검색."""
import logging
import uuid
from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app import config

logger = logging.getLogger(__name__)

# 필터/삭제에 쓰는 필드는 인덱스를 만들어 둡니다.
INDEXED_FIELDS = ("category", "group_no", "status", "document_name", "document_id")


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if not config.QDRANT_URL:
        raise RuntimeError("QDRANT_URL 환경변수가 설정되지 않았습니다. .env를 확인하세요.")
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None, timeout=30)


def init_collection() -> None:
    """컬렉션이 없으면 만들고, 빠진 payload 인덱스만 추가합니다."""
    client = get_client()
    name = config.QDRANT_COLLECTION
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=config.EMBEDDING_DIM, distance=models.Distance.COSINE),
        )
        logger.info("컬렉션 생성: %s", name)

    existing = client.get_collection(name).payload_schema or {}
    for field in INDEXED_FIELDS:
        if field not in existing:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("payload 인덱스 생성: %s", field)


def reset_collection() -> None:
    """컬렉션을 삭제 후 다시 만듭니다(스키마가 바뀐 옛 데이터 정리용)."""
    get_client().delete_collection(config.QDRANT_COLLECTION)
    init_collection()


def delete_document(document_name: str) -> None:
    """같은 파일명의 기존 청크를 모두 지웁니다(재적재 시 중복·잔여 청크 방지)."""
    get_client().delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="document_name", match=models.MatchValue(value=document_name))]
            )
        ),
        wait=True,
    )


def document_exists(document_name: str, document_id: str) -> bool:
    """같은 이름·같은 내용(해시)의 문서가 이미 적재되어 있는지 확인합니다 (이어서 적재할 때 사용)."""
    result = get_client().count(
        collection_name=config.QDRANT_COLLECTION,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(key="document_name", match=models.MatchValue(value=document_name)),
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            ]
        ),
        exact=True,
    )
    return result.count > 0


def upsert_documents(chunks: list[dict], vectors: list[list[float]], batch_size: int = 64) -> int:
    """chunk_id(문자열)를 UUID로 변환해 저장합니다. Qdrant는 정수/UUID ID만 허용합니다."""
    points = [
        models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])), vector=v, payload=c)
        for c, v in zip(chunks, vectors)
    ]
    client = get_client()
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points[i : i + batch_size], wait=True)
    return len(points)


def _query(vector: list[float], conditions: list, top_k: int, score_threshold: float) -> list[dict]:
    result = get_client().query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vector,
        query_filter=models.Filter(must=conditions),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return [{**(p.payload or {}), "score": p.score} for p in result.points]


def search_docs(
    query_vector: list[float],
    groups: Optional[list[str]] = None,
    top_k: int = config.TOP_K,
    score_threshold: float = config.SCORE_THRESHOLD,
) -> list[dict]:
    """현행(status=current) 청크만 검색하고 payload 전체와 score를 돌려줍니다.

    groups(대분류 번호 목록)가 있으면 그 분야의 규정 안에서만 검색합니다.
    사용자가 직접 고른 범위이므로 결과가 없어도 범위를 넓혀 다시 검색하지 않습니다.
    groups가 없으면 전체 규정을 검색합니다.
    """
    conditions = [models.FieldCondition(key="status", match=models.MatchValue(value="current"))]
    if groups:
        conditions.append(models.FieldCondition(key="group_no", match=models.MatchAny(any=list(groups))))
    return _query(query_vector, conditions, top_k, score_threshold)
